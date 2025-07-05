import math
from functools import partial
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
            
        # label_B的形状是(B,)
        # 拆解一下first token map的来头
        # 其中 torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)
        # 这个 张量的shape是(2B,)
        # 首先进行由生成带有embeding信息的sos (2B,C)
        # 然后让unsqueezec向sos中插入新维度1 (2B, 1, C)
        # pos_start (1,1,C) 广播 2B份 (2B, 1, C)
        # 操作完的两个张亮相加
        # 最后复制（实际上是广播） B份 (2B,1,C)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        next_token_map = torch.zeros((2 * B, self.first_l, self.C))
        input_token_map = torch.zeros((2 * B, self.first_l, self.C))

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            # 如果是第一层，生成first_token_map,否则input_token_map = next_token_map
            # 第一层因为设计 embeding后加参数，而不是加参数后embeding,导致第一层无法实现加入token_hub
            # 也就是说无论怎么样都没有办法实现两个模型之间的统一
            if si == 0:
                input_token_map = (
                    sos.unsqueeze(1).expand(2 * B, self.first_l, -1) 
                    + self.pos_start.expand(2 * B, self.first_l, -1) 
                    + lvl_pos[:, :self.first_l]
                )
            else: 
                input_token_map = next_token_map
                input_token_map = input_token_map.view(B, self.Cvae, -1).transpose(1,2)
                input_token_map = self.word_embed(input_token_map) + lvl_pos[:, cur_L:cur_L + pn * pn]
                input_token_map = input_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
            ratio = si / self.num_stages_minus_1
            cur_L = cur_L + pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = input_token_map

            for block in self.blocks:
                x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits = (1+t) * logits[:B] - t * logits[B:]
            
            token_id = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(token_id)
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

        for block in self.blocks: block.attn.kv_caching(False)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

    # '''
    # helper1是给予任意一部分让他接下去生成
    # 首先一定需要给的参数是current step表示当前生成的步数， draft_step表示需要生成的内容，
    @torch.no_grad()
    def autoregressive_infer_cfg_sd_helper1(
        self, B: int,
        current_step: int, 
        step: int,
        next_token_map,
        f_hat,
        rng,
        sos,
        lvl_pos,

        cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
        
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]

        # if g_seed is None: rng = None
        # else: self.rng.manual_seed(g_seed); rng = self.rng
        rng = rng
        
        # if label_B is None:
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        sos = sos
        cond_BD = sos
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        lvl_pos = lvl_pos

        cur_L = 0
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat = f_hat

        if next_token_map == None:
            next_token_map = torch.zeros((2 * B, self.first_l, self.C))
        else:
            next_token_map = next_token_map
        
        logits_history = []
        token_id_history = []
        input_token_history = []
        f_hat_history = []
        
        # input_token_map = torch.zeros((2 * B, self.first_l, self.C))
        
        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            # 快进到current_step
            if si < current_step:
                continue
            
            # 实际上不会出现
            if si == 0:
                input_token_map = (
                    sos.unsqueeze(1).expand(2 * B, self.first_l, -1) 
                    + self.pos_start.expand(2 * B, self.first_l, -1) 
                    + lvl_pos[:, :self.first_l]
                )
            # 将上一层内容进行处理转换为包含初始到上一层所有内容，与本层空内容的张量
            # 加上位置编码
            else: 
                input_token_map = next_token_map
                input_token_map = input_token_map.view(B, self.Cvae, -1).transpose(1,2)
                # 我们会保存从输入开始的input_token_map到最后
                input_token_history.append(input_token_map)
                # print(f"si {si}, input_token_map.shape: {input_token_map.shape}")
                input_token_map = self.word_embed(input_token_map) + lvl_pos[:, cur_L:cur_L + pn * pn]
                input_token_map = input_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

            ratio = si / self.num_stages_minus_1
            cur_L = cur_L + pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            # print(f"{si}")           

            # 我们会保存从输入的f_hat开始到最后
            f_hat_history.append(f_hat)
            
            # input_token_map_history, f_hat_history储存的内容是从current_step-> current_step + step - 1的input
            # 他没有保存current_step + step - 1的output, 这两个是最后的返回值

            x = input_token_map

            for block in self.blocks:
                x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits = (1+t) * logits[:B] - t * logits[B:]
            # 我们会保存从输入的logits(融合后的)到最后
            logits_history.append(logits)
            
            token_id = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # 我们会保存从输入的token_id(融合后的)到最后
            token_id_history.append(token_id)

            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(token_id)
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

            # 条件满足就直接走了
            if si == current_step + step - 1:
                # f_hat是可以直接append，但是next_token_map需要先变换才能append进入input_token_map 
                f_hat_history.append(f_hat)
                
                input_token_map.view(B, self.Cvae, -1).transpose(1,2)
                input_token_history.append(next_token_map)
                break

        for block in self.blocks: block.attn.kv_caching(False)

        # return input_token_history, f_hat_history, logits_history, token_id_history
        # input_token_history: current_step -> current_step + step, len = step + 1
        # f_hat_history: current_step -> current_step + step, len = step + 1
        # logits_history: current_step -> current_step + step - 1, len = step
        # token_id_history: current_step -> current_step + step - 1, len = step
        return input_token_history, f_hat_history, logits_history, token_id_history
    # ''' 


    # '''
    # helper2是给予draft_model生成的一长串target_model进行一次性验证
    # 首先一定需要给的参数是current step表示当前生成的步数， draft_step表示需要生成的内容，
    @torch.no_grad()
    def autoregressive_infer_cfg_sd_helper2(
        self, B: int,
        current_step: int, 
        step: int,
        unified_next_token_map,
        f_hat,
        rng,
        sos,
        lvl_pos,
        attn_bias,
        t,
        start_points,
        exit_points,

        cfg=1.5, top_k=0, top_p=0.0,more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]

        rng = rng
        sos = sos
        cond_BD = sos
        lvl_pos = lvl_pos
        cur_L = 0
        f_hat = f_hat

        if next_token_map == None:
            next_token_map = torch.zeros((2 * B, self.first_l, self.C))
        else:
            next_token_map = next_token_map
        
        logits_history = []
        token_id_history = []
        input_token_history = []
        f_hat_history = []
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # input_token_map = torch.zeros((2 * B, self.first_l, self.C))
        
        # target model 会利用attn_bias一次性生成所有
        for b in self.blocks: b.attn.kv_caching(True)
        
        input_token_map = unified_next_token_map
        # 这里位置编码的编号要改，从current_step的到current_step + step(没有-1因为我们此时已经有current_step + step的预备内容了)
        input_token_map = self.word_embed(input_token_map) + lvl_pos[:, start_points[current_step]:exit_points[current_step + step]] 
        input_token_map.repeat(2,1,1)

        x = input_token_map

        for block in self.blocks:
            x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        logits = self.get_logits(x, cond_BD)
        
        # 这里需要改，要把t改成每组对应的t,已经改好了
        logits = (1+t) * logits[:B] - t * logits[B:]
        
        token_id = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

        for block in self.blocks: block.attn.kv_caching(False)

        # 我们只需要logits
        return token_id, logits
    # ''' 

class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )

class SDVAR(nn.Module):
    def __init__(
        self,
        draft_model,
        target_model,
        similarity_thresh: float = 0.8
    ):

        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.similarity_thresh = similarity_thresh

        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        block_sizes = [p ** 2 for p in patch_nums]
        total_tokens = sum(block_sizes)

        block_ids = []
        for block, size in enumerate(block_sizes):
            block_ids += [block] * size
        block_ids = torch.tensor(block_ids)

        attn_bias_for_sdmasking = torch.full((total_tokens, total_tokens), float('-inf'))

        for i in range(total_tokens):
            for j in range(total_tokens):
                if j > i:
                    continue
                if block_ids[i] == block_ids[j] and i != j:
                    continue
                attn_bias_for_sdmasking[i, j] = 0.0

        attn_bias_for_sdmasking = attn_bias_for_sdmasking.reshape(1, 1, total_tokens, total_tokens)
        
        self.attn_bias_for_sdmasking = attn_bias_for_sdmasking

        blockmasking = torch.full((total_tokens, total_tokens), float('-inf'))  # 默认禁止注意力
        for i in range(total_tokens):
            for j in range(total_tokens):
                if block_ids[i] == block_ids[j]:  # 只允许相同 block_id 之间互相注意
                    blockmasking[i, j] = 0.0

        blockmasking = blockmasking.reshape(1, 1, total_tokens, total_tokens)
        self.attn_bias_for_block = blockmasking

    def init_param(
            self,
            model: VAR,
            B: int,
            label_B,
        ):        
        sos = cond_BD = model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=model.num_classes)), dim=0)
        )   
        cond_BD_or_gss = model.shared_ada_lin(cond_BD)

        lvl_pos = model.lvl_embed(model.lvl_1L) + model.pos_1LC

        first_token_map = (
            sos.unsqueeze(1).expand(2*B, model.first_l, -1)
            + model.pos_start.expand(2*B, model.first_l, -1)
            + lvl_pos[:, :model.first_l]
        )

        first_f_hat = sos.new_zeros(B, model.Cvae, model.patch_nums[-1], model.patch_nums[-1])

        return sos, cond_BD, cond_BD_or_gss, lvl_pos, first_token_map, first_f_hat

    # 初始化分离和选择掩码方式
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test3(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10, 
        sd_mask: int = 0
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param entry_num: 转换模型的节点
        :param sd_mask: 是否使用我们自己写的block_wise的掩码
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        ###### 通用参数参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        self.vae_proxy = self.target_model.vae_proxy
        self.vae_quant_proxy = self.target_model.vae_quant_proxy

        if g_seed is not None:
            self.rng = self.target_model.rng.manual_seed(g_seed)
        else:
            self.rng = None
        

        if label_B is None:
            label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )

        draft_sos, draft_cond_BD, draft_cond_BD_or_gss, \
        draft_lvl_pos, draft_first_token_map, draft_f_hat = self.init_param(self.draft_model, B, label_B)

        draft_cur_L = 0
        draft_next_token_map = draft_first_token_map
        draft_token_hub = []
        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)
            # blk.attn.kv_caching(False)

        for si, pn in enumerate(self.patch_nums):
            
            # 生成0-entry_num-1
            if si >= entry_num:
                break

            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn*pn
            x = draft_next_token_map
            
            AdaLNSelfAttn.forward
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)            
            
            t = cfg * ratio
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            # 在所有模型中都使用同一个vae
            if not more_smooth:
                draft_h_BChw = self.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L : draft_cur_L + next_pn**2]
                )
                draft_next_token_map = draft_next_token_map.repeat(2,1,1)

            if si == self.num_stages_minus_1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                return self.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        # draft模型生成完毕  
        if len(draft_token_hub) != 0:   
            draft_token_hub = torch.cat(draft_token_hub, dim = 1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
    
        ###### target模型接受draft模型生成的内容然后生成最后一层的内容
        start_points = [0,1,5,14,30,55,91,155,255,424]
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        pindex = exit_points[entry_num]
        sindex = start_points[entry_num]
        device = torch.device("cuda:0")


        target_sos, target_cond_BD, target_cond_BD_or_gss, \
        target_lvl_pos, target_first_token_map, target_f_hat = self.init_param(self.target_model, B, label_B)

        target_f_hat = draft_f_hat

        target_cur_L = 0
        target_f_hat = draft_f_hat

        # 如果draft_token_hub不为0
        if not len(draft_token_hub) == 0:
            # 接受之前生成的做为target_model输出的prefix
            target_next_token_map = draft_token_hub    

            target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:,1:pindex]  
            
            # 正常来说前边的已经进行过调整，所以这里应该只有最后一段需要cfg的修改。
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            if len(target_next_token_map) != 0:
                target_next_token_map = torch.cat([target_first_token_map,target_next_token_map],dim=1)
            else:
                target_next_token_map = target_first_token_map
        else: 
            target_next_token_map = target_first_token_map
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)
            # blk.attn.kv_caching(False)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            # 我们实际上只需要让进入那一层找到对应的next_token_map就可以了，剩下的就是x = target_next_token_map
            if sd_mask != 0:
                if sd_mask == 1:
                    # sd_mask = 1, 全部层包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
                    attn_bias = attn_bias.to(device)
                elif sd_mask == 2:
                    # sd_mask = 2, 全部层不包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                elif sd_mask == 3:
                    # sd_mask = 3, 进行因果掩码
                    attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]
                elif sd_mask == 4: 
                    # sd_mask = 1, 全部层包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_block[:,:,0:pindex,0:pindex]
                    attn_bias = attn_bias.to(device)
                elif sd_mask == 5:
                    # sd_mask = 1, 全部层包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_block[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                x = target_next_token_map
                AdaLNSelfAttn.forward
                # 这里我们暂时不检测也不用attn_bias，因为我们当前只截取了进入层的
                if si == entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
                else:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)

                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                else:
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                    
            else:
                # sd_mask = 0, 不需要使用掩码
                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
                else:
                    x = target_next_token_map
                AdaLNSelfAttn.forward
                if si >= entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

            # 这里进行了改动，我们没有进行重新采样，因为实际上我们应该继续使用之前的f_hat,
            target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, target_h_BChw
            )
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
        # target模型生成完成
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)   
                    
        return self.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        
    # =======================================================================================
    # Week 1 实现：核心并行验证框架
    # =======================================================================================
    
    def _initialize_inference_state(self, B: int, label_B, g_seed: Optional[int], 
                                   cfg: float, gamma: int):
        """初始化SDVAR推理状态"""
        from dataclasses import dataclass
        
        # 验证模型兼容性
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        patch_nums = self.draft_model.patch_nums
        total_stages = len(patch_nums)
        
        # 处理随机数生成器
        if g_seed is not None:
            draft_rng = torch.Generator(device=self.draft_model.lvl_1L.device)
            draft_rng.manual_seed(g_seed)
            target_rng = torch.Generator(device=self.target_model.lvl_1L.device) 
            target_rng.manual_seed(g_seed)
            rng = target_rng  # 主要使用target的rng
        else:
            rng = None
            
        # 处理标签
        if label_B is None:
            uniform_prob = self.target_model.uniform_prob
            label_B = torch.multinomial(
                uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,), fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )
        
        # 初始化draft模型状态
        draft_sos, draft_cond_BD, _, draft_lvl_pos, draft_first_token_map, draft_f_hat = \
            self.init_param(self.draft_model, B, label_B)
            
        # 初始化target模型状态
        target_sos, target_cond_BD, _, target_lvl_pos, target_first_token_map, target_f_hat = \
            self.init_param(self.target_model, B, label_B)
        
        # 创建状态对象（使用简单的类）
        class SDVARInferenceState:
            def __init__(self):
                self.current_stage = 0
                self.gamma = gamma
                self.total_stages = total_stages
                self.accept_count = 0
                self.reject_count = 0
                self.target_calls = 0
                self.patch_nums = patch_nums
                
                # 运行时状态
                self.draft_f_hat = draft_f_hat
                self.target_f_hat = target_f_hat
                self.draft_next_token_map = draft_first_token_map
                self.target_next_token_map = target_first_token_map
                
                # 模型参数
                self.draft_cond_BD = draft_cond_BD
                self.target_cond_BD = target_cond_BD
                self.draft_lvl_pos = draft_lvl_pos
                self.target_lvl_pos = target_lvl_pos
                
                # 生成参数
                self.cfg = cfg
                self.rng = rng
                self.top_k = 0
                self.top_p = 0.0
                self.more_smooth = False
                
                # token buffers for concatenation
                self.accepted_tokens = []
                self.current_tokens_length = 0
        
        state = SDVARInferenceState()
        
        return state
    
    def draft_generate_batch(self, state, B: int, verbose: bool = False) -> List[torch.Tensor]:
        """draft模型批量生成gamma层"""
        gamma = min(state.gamma, state.total_stages - state.current_stage)
        if gamma <= 0:
            return []
        
        if verbose:
            print(f"[SDVAR] Draft generating batch: gamma={gamma}, current_stage={state.current_stage}")
        
        # 确保KV cache开启
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)
        
        draft_tokens = []
        current_tokens = state.draft_next_token_map
        current_f_hat = state.draft_f_hat
        
        for gamma_step in range(gamma):
            stage_idx = state.current_stage + gamma_step
            pn = state.patch_nums[stage_idx]
            
            if verbose:
                print(f"[SDVAR] Draft stage {stage_idx}, pn={pn}, tokens_shape={current_tokens.shape}")
            
            # 前向计算
            x = current_tokens
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=state.draft_cond_BD, attn_bias=None)
            
            logits = self.draft_model.get_logits(x, state.draft_cond_BD)
            
            # CFG处理
            ratio = stage_idx / self.draft_model.num_stages_minus_1
            t = state.cfg * ratio
            logits = (1 + t) * logits[:B] - t * logits[B:]
            
            # 采样
            tokens = sample_with_top_k_top_p_(
                logits, rng=state.rng, top_k=state.top_k, 
                top_p=state.top_p, num_samples=1
            )[:, :, 0]
            
            draft_tokens.append(tokens)
            
            # 准备下一层的输入（如果不是最后一层）
            if gamma_step < gamma - 1:
                h_BChw = self.draft_model.vae_quant_proxy[0].embedding(tokens)
                h_BChw = h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)
                
                # 更新f_hat和next_token_map
                current_f_hat, next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                    stage_idx, state.total_stages, current_f_hat, h_BChw
                )
                
                if stage_idx + 1 < state.total_stages:
                    next_pn = state.patch_nums[stage_idx + 1]
                    current_tokens_length = sum(p**2 for p in state.patch_nums[:stage_idx+1])
                    
                    next_token_map = next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                    next_token_map = (
                        self.draft_model.word_embed(next_token_map) + 
                        state.draft_lvl_pos[:, current_tokens_length:current_tokens_length + next_pn**2]
                    )
                    current_tokens = next_token_map.repeat(2, 1, 1)  # CFG doubling
        
        # 更新state中的f_hat
        if draft_tokens:
            # 处理最后一个stage的f_hat更新
            final_stage_idx = state.current_stage + len(draft_tokens) - 1
            final_pn = state.patch_nums[final_stage_idx]
            final_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_tokens[-1])
            final_h_BChw = final_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, final_pn, final_pn)
            
            state.draft_f_hat, _ = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                final_stage_idx, state.total_stages, current_f_hat, final_h_BChw
            )
        
        return draft_tokens

    def target_verify_batch(self, draft_tokens: List[torch.Tensor], 
                           state, B: int) -> Tuple[List[torch.Tensor], int]:
        """target模型批量验证draft生成的tokens"""
        if not draft_tokens:
            return [], 0
        
        gamma = len(draft_tokens)
        print(f"[SDVAR] Target verifying batch: gamma={gamma}, current_stage={state.current_stage}")
        
        # 构建联合查询序列
        combined_query = self._build_combined_query(draft_tokens, state, B)
        print(f"[SDVAR] Combined query shape: {combined_query.shape}")
        
        # 计算适当的注意力掩码
        mask_length = combined_query.shape[1]
        attn_bias = self._get_attention_mask(mask_length, state.current_stage, gamma)
        
        # 确保KV cache开启
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)
        
        # target前向计算
        state.target_calls += 1  # 统计调用次数
        print(f"[SDVAR] Target forward call #{state.target_calls}")
        
        x = combined_query
        for blk in self.target_model.blocks:
            x = blk(x=x, cond_BD=state.target_cond_BD, attn_bias=attn_bias)
        
        target_logits = self.target_model.get_logits(x, state.target_cond_BD)
        
        # 分割logits回对应的层
        logits_per_stage = self._split_logits_by_stage(target_logits, draft_tokens, B, state)
        
        # 应用CFG
        cfg_logits = []
        for stage_idx, stage_logits in enumerate(logits_per_stage):
            current_stage = state.current_stage + stage_idx
            ratio = current_stage / self.target_model.num_stages_minus_1
            t = state.cfg * ratio
            cfg_stage_logits = (1 + t) * stage_logits[:B] - t * stage_logits[B:]
            cfg_logits.append(cfg_stage_logits)
        
        print(f"[SDVAR] Target verification completed, generated {len(cfg_logits)} stage logits")
        return cfg_logits, gamma

    def _build_combined_query(self, draft_tokens: List[torch.Tensor], 
                             state, B: int) -> torch.Tensor:
        """将多层draft tokens组合成target的查询序列"""
        print(f"[SDVAR] Building combined query for {len(draft_tokens)} stages")
        
        # 构建完整的输入序列，模仿VAR的正常推理流程
        all_embeddings = []
        
        # 计算当前位置偏移 - 注意：这里只处理单batch，CFG在最后处理
        if state.current_stage == 0:
            # 如果是从第一阶段开始，需要包含first_token_map
            # 这里只取B个样本，不是2*B（CFG在最后处理）
            sos = state.target_cond_BD[:B]  # 只取前B个
            first_l = self.target_model.first_l
            first_token_map = (
                sos.unsqueeze(1).expand(B, first_l, -1) +
                self.target_model.pos_start.expand(B, first_l, -1) +
                state.target_lvl_pos[:1, :first_l].expand(B, -1, -1)  # 扩展到B个样本
            )
            all_embeddings.append(first_token_map)
            current_pos = first_l
        else:
            current_pos = sum(p**2 for p in state.patch_nums[:state.current_stage])
            # TODO: 添加之前已接受tokens的embedding
            # 目前简化处理，假设state中会维护这些信息
        
        # 处理每个draft token stage
        for stage_idx, tokens in enumerate(draft_tokens):
            current_stage = state.current_stage + stage_idx
            pn = state.patch_nums[current_stage]
            
            print(f"[SDVAR] Processing stage {current_stage}, tokens shape: {tokens.shape}, pn={pn}")
            
            # 处理draft tokens - 所有阶段都使用相同的处理方式
            # 获取VAE embedding
            h_BChw = self.target_model.vae_quant_proxy[0].embedding(tokens)  # (B, pn*pn, Cvae)
            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)  # (B, Cvae, pn, pn)
            
            # 转换为word_embed的输入格式
            h_embed_input = h_BChw.view(B, self.target_model.Cvae, -1).transpose(1, 2)  # (B, pn*pn, Cvae)
            
            # 添加位置编码 - 确保维度匹配
            pos_embed = state.target_lvl_pos[:1, current_pos:current_pos + pn * pn].expand(B, -1, -1)
            
            print(f"[SDVAR] h_embed_input shape: {h_embed_input.shape}, pos_embed shape: {pos_embed.shape}")
            
            # 应用word embedding和位置编码
            h_embed = self.target_model.word_embed(h_embed_input) + pos_embed
            all_embeddings.append(h_embed)
            
            print(f"[SDVAR] Stage {current_stage} embedding shape: {h_embed.shape}")
            
            current_pos += pn * pn
        
        # 拼接所有embeddings
        combined = torch.cat(all_embeddings, dim=1)  # B, total_tokens, C
        print(f"[SDVAR] Combined embeddings shape: {combined.shape}")
        
        # CFG doubling
        combined = combined.repeat(2, 1, 1)
        
        return combined

    def _split_logits_by_stage(self, target_logits: torch.Tensor, 
                              draft_tokens: List[torch.Tensor], B: int, state) -> List[torch.Tensor]:
        """将target的logits分割回对应的层"""
        print(f"[SDVAR] Splitting logits: input shape {target_logits.shape}")
        
        logits_per_stage = []
        current_pos = 0
        
        # 如果有前缀tokens，先跳过它们
        if state.current_stage > 0:
            prefix_length = sum(p**2 for p in state.patch_nums[:state.current_stage])
            current_pos = prefix_length
            print(f"[SDVAR] Skipping prefix tokens: {prefix_length}")
        
        # 分割每个stage的logits
        for stage_idx, tokens in enumerate(draft_tokens):
            current_stage = state.current_stage + stage_idx
            pn = state.patch_nums[current_stage]
            stage_length = pn * pn
            
            # 提取这个stage的logits
            stage_logits = target_logits[:, current_pos:current_pos + stage_length, :]
            logits_per_stage.append(stage_logits)
            current_pos += stage_length
            
            print(f"[SDVAR] Stage {current_stage} logits shape: {stage_logits.shape}")
        
        return logits_per_stage

    def _get_attention_mask(self, mask_length: int, current_stage: int, gamma: int) -> torch.Tensor:
        """获取适当的注意力掩码用于target批量验证"""
        # 对于Week 1的简化版本，暂时使用基础的因果掩码
        # TODO: Week 2将实现更复杂的块级掩码策略
        
        if hasattr(self.target_model, 'attn_bias_for_masking'):
            # 使用预计算的因果掩码
            full_mask = self.target_model.attn_bias_for_masking
            if mask_length <= full_mask.shape[-1]:
                return full_mask[:, :, :mask_length, :mask_length]
        
        # 如果没有预计算的掩码，创建简单的因果掩码
        mask = torch.triu(torch.full((mask_length, mask_length), float('-inf')), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度

    def basic_token_matching(self, draft_tokens: List[torch.Tensor], 
                           target_logits: List[torch.Tensor], state, B: int, 
                           similarity_threshold: float = 0.5, verbose: bool = False) -> int:
        """
        基础token匹配验证逻辑 - Week 1简化版本
        
        比较draft生成的tokens与target logits，决定接受多少层
        使用简单的top-1匹配策略：如果draft token是target logits的top-1，则接受
        
        :param draft_tokens: draft模型生成的tokens，每层一个tensor
        :param target_logits: target模型对应的logits，每层一个tensor  
        :param state: 推理状态
        :param B: batch size
        :return: 接受的层数
        """
        if not draft_tokens or not target_logits:
            return 0
        
        if len(draft_tokens) != len(target_logits):
            print(f"[SDVAR] Mismatch: {len(draft_tokens)} draft vs {len(target_logits)} target")
            return 0
        
        gamma = len(draft_tokens)
        if verbose:
            print(f"[SDVAR] Starting basic token matching for {gamma} stages")
        
        accepted_stages = 0
        total_tokens = 0
        matched_tokens = 0
        
        # 逐层进行匹配验证
        for stage_idx in range(gamma):
            current_stage = state.current_stage + stage_idx
            pn = state.patch_nums[current_stage]
            
            draft_stage_tokens = draft_tokens[stage_idx]  # B, pn*pn
            target_stage_logits = target_logits[stage_idx]  # B, pn*pn, V
            
            if verbose:
                print(f"[SDVAR] Matching stage {current_stage}: draft_tokens shape {draft_stage_tokens.shape}, target_logits shape {target_stage_logits.shape}")
            
            # 获取target的top-1预测
            target_top1 = torch.argmax(target_stage_logits, dim=-1)  # B, pn*pn
            
            # 比较draft tokens和target top-1
            stage_matches = (draft_stage_tokens == target_top1)  # B, pn*pn
            stage_match_rate = stage_matches.float().mean().item()
            
            stage_total = draft_stage_tokens.numel()
            stage_matched = stage_matches.sum().item()
            
            total_tokens += stage_total
            matched_tokens += stage_matched
            
            if verbose:
                print(f"[SDVAR] Stage {current_stage} match rate: {stage_match_rate:.3f} ({stage_matched}/{stage_total})")
            
            # 基础匹配策略：如果匹配率 >= 阈值，接受这一层
            # Week 1使用简单的层级匹配阈值
            match_threshold = similarity_threshold  # 使用传入的相似度阈值
            
            if stage_match_rate >= match_threshold:
                accepted_stages += 1
                if verbose:
                    print(f"[SDVAR] ✅ Accepting stage {current_stage} (match rate: {stage_match_rate:.3f})")
            else:
                if verbose:
                    print(f"[SDVAR] ❌ Rejecting stage {current_stage} and all subsequent stages (match rate: {stage_match_rate:.3f})")
                break  # 一旦有层被拒绝，后续层也全部拒绝（保持因果性）
        
        overall_match_rate = matched_tokens / total_tokens if total_tokens > 0 else 0.0
        if verbose:
            print(f"[SDVAR] Basic matching completed: accepted {accepted_stages}/{gamma} stages, overall match rate: {overall_match_rate:.3f}")
        
        return accepted_stages

    def advanced_token_matching(self, draft_tokens: List[torch.Tensor], 
                               target_logits: List[torch.Tensor], state, B: int) -> int:
        """
        高级token匹配验证逻辑 - 为Week 2准备
        
        将支持更复杂的匹配策略：
        - 概率分布KL散度比较
        - top-k匹配而非仅top-1
        - 动态阈值调整
        - 部分接受（token级回滚）
        
        目前返回基础匹配结果
        """
        # TODO: Week 2实现高级匹配逻辑
        return self.basic_token_matching(draft_tokens, target_logits, state, B)

    def update_state_with_accepted_tokens(self, draft_tokens: List[torch.Tensor], 
                                        accept_length: int, state, B: int):
        """
        更新推理状态，处理被接受的tokens
        
        将接受的tokens更新到f_hat中，准备下一轮推理
        """
        if accept_length <= 0:
            return
        
        print(f"[SDVAR] Updating state with {accept_length} accepted stages")
        
        # 处理接受的每一层
        current_f_hat = state.draft_f_hat
        
        for stage_idx in range(accept_length):
            current_stage = state.current_stage + stage_idx
            pn = state.patch_nums[current_stage]
            
            # 获取这一层的tokens
            stage_tokens = draft_tokens[stage_idx]
            
            # 转换为embedding并reshape
            h_BChw = self.draft_model.vae_quant_proxy[0].embedding(stage_tokens)
            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)
            
            # 更新f_hat
            current_f_hat, _ = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                current_stage, state.total_stages, current_f_hat, h_BChw
            )
            
            print(f"[SDVAR] Updated f_hat for stage {current_stage}")
        
        # 更新状态中的f_hat
        state.draft_f_hat = current_f_hat
        state.target_f_hat = current_f_hat.clone()  # target也同步
        
        print(f"[SDVAR] State update completed for {accept_length} stages")

    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_parallel_v1(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]] = None,
        g_seed: Optional[int] = None, cfg: float = 1.5,
        gamma: int = 2, top_k: int = 0, top_p: float = 0.0, more_smooth: bool = False,
        similarity_threshold: float = 0.5, max_retries: int = 3, verbose: bool = False
    ) -> torch.Tensor:
        """
        SDVAR并行推理 v1.0 - 固定gamma版本
        
        核心改进：
        1. 使用while循环代替for循环，支持γ批处理
        2. draft模型连续生成γ层，target模型批量验证
        3. 基础的匹配验证机制
        4. 为后续的回滚和动态γ奠定基础
        
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param gamma: 批量处理的层数，固定为2
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax
        :param similarity_threshold: token匹配的相似度阈值
        :param max_retries: 最大重试次数
        :param verbose: 是否输出详细信息
        :return: reconstructed image (B, 3, H, W) in [0, 1]
        """
        if verbose:
            print(f"[SDVAR] Starting parallel inference v1.0 with gamma={gamma}")
        
        # 初始化状态
        state = self._initialize_inference_state(B, label_B, g_seed, cfg, gamma)
        state.top_k = top_k
        state.top_p = top_p
        state.more_smooth = more_smooth
        
        # 主推理循环 - 这里是关键改进：while循环代替for循环
        retry_count = 0
        while state.current_stage < state.total_stages and retry_count < max_retries:
            if verbose:
                print(f"[SDVAR] === Processing stages {state.current_stage} to {min(state.current_stage + gamma - 1, state.total_stages - 1)} ===")
            
            # 1. Draft批量生成
            draft_tokens = self.draft_generate_batch(state, B, verbose)
            if not draft_tokens:
                if verbose:
                    print("[SDVAR] No more tokens to generate, breaking")
                break
            
            actual_gamma = len(draft_tokens)
            if verbose:
                print(f"[SDVAR] Draft generated {actual_gamma} stages")
            
            # 2. Target批量验证 (Week 1新增功能)
            target_logits, verified_gamma = self.target_verify_batch(draft_tokens, state, B)
            
            # 3. 基础匹配验证 - 集成真正的token匹配逻辑
            if target_logits and len(target_logits) > 0:
                accept_length = self.basic_token_matching(draft_tokens, target_logits, state, B, similarity_threshold, verbose)
                if verbose:
                    print(f"[SDVAR] Token matching result: accepting {accept_length}/{verified_gamma} stages")
            else:
                accept_length = 0
                if verbose:
                    print(f"[SDVAR] Target verification failed, rejecting all")
            
            # 4. 更新推理状态
            if accept_length > 0:
                self.update_state_with_accepted_tokens(draft_tokens, accept_length, state, B)
                if verbose:
                    print(f"[SDVAR] Successfully processed {accept_length} stages")
            else:
                if verbose:
                    print(f"[SDVAR] No stages accepted, will retry with smaller gamma")
            
            # 5. 提交接受的部分
            state.accept_count += accept_length
            state.current_stage += accept_length
            
            # 6. 动态gamma调整策略
            if accept_length == 0:
                # 如果没有接受任何stage，降低gamma重试
                if state.gamma > 1:
                    state.gamma = max(1, state.gamma - 1)
                    if verbose:
                        print(f"[SDVAR] Reducing gamma to {state.gamma} due to rejection")
                else:
                    # gamma=1还是失败，强制接受一层避免死循环
                    if verbose:
                        print(f"[SDVAR] Emergency fallback: force accepting 1 stage at {state.current_stage}")
                    if draft_tokens:
                        self.update_state_with_accepted_tokens(draft_tokens[:1], 1, state, B)
                        state.accept_count += 1
                        state.current_stage += 1
                retry_count += 1
            elif accept_length == verified_gamma:
                # 如果全部接受，可以考虑增加gamma（但Week 1保持固定）
                if verbose:
                    print(f"[SDVAR] All stages accepted, maintaining gamma={state.gamma}")
                retry_count = 0  # 重置重试计数器
            
            # 7. 检查是否有进展
            if accept_length == 0 and state.gamma == 1:
                if verbose:
                    print(f"[SDVAR] No progress possible, stopping inference")
                break

        # 清理KV cache
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)
        
        if verbose:
            print(f"[SDVAR] Inference completed. Accepted: {state.accept_count}, Target calls: {state.target_calls}")
        
        # 返回最终图像
        return self.draft_model.vae_proxy[0].fhat_to_img(state.draft_f_hat).add_(1).mul_(0.5)


