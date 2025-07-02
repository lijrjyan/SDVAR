import math
from functools import partial
from typing import Optional, Tuple, Union

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
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
                
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
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
        similarity_thresh: float = 0.8,
        draft_steps: int = 2,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.similarity_thresh = similarity_thresh
        self.draft_steps = draft_steps

    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 5,
        sd_mask: int = 5,  # 默认使用修改版block-wise掩码
    ) -> torch.Tensor:
        """
        修复版SDVAR推理函数，解决不同entry_num导致的随机数同步问题
        
        主要改进：
        1. 为draft和target模型创建分离的随机数生成器
        2. 确保target模型始终从相同的随机数状态开始生成
        3. 添加掩码支持，默认使用修改版block-wise掩码
        4. 保证相同模型和初始化下，无论entry_num如何，最终输出保持一致
        
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax
        :param entry_num: 模型切换点，draft_model生成前entry_num个阶段
        :param sd_mask: 掩码类型选择
                      0: 不使用掩码
                      1: 全部层包括未预测这层进行block-wise的掩码
                      2: 全部层不包括未预测这层进行block-wise的掩码
                      3: 因果掩码
                      4: block-wise掩码
                      5: 修改版block-wise掩码（默认）
        :return: reconstructed image (B, 3, H, W) in [0, 1]
        """
        # === 修复1: 创建分离的随机数生成器 ===
        if g_seed is not None:
            # 为draft模型创建专用的随机数生成器
            draft_rng = torch.Generator(device=self.draft_model.lvl_1L.device)
            draft_rng.manual_seed(g_seed)
            
            # 为target模型创建专用的随机数生成器，使用相同种子
            target_rng = torch.Generator(device=self.target_model.lvl_1L.device) 
            target_rng.manual_seed(g_seed)
            
            # label生成使用target的随机数生成器
            label_rng = target_rng
        else:
            draft_rng = None
            target_rng = None
            label_rng = None

        # 验证模型兼容性
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        patch_nums = self.draft_model.patch_nums
        total_stages = len(patch_nums)
        
        # 处理标签
        if label_B is None:
            label_B = torch.multinomial(
                self.target_model.uniform_prob, 
                num_samples=B, 
                replacement=True, 
                generator=label_rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )

        # === 阶段1: Draft模型生成前entry_num个阶段 ===
        # 初始化draft模型参数
        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )
        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
        draft_next_token_map = (
            draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae, patch_nums[-1], patch_nums[-1])
        draft_cur_L = 0
        draft_token_hub = []

        # 启用draft模型的KV缓存
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        # Draft模型生成
        for si in range(min(entry_num, total_stages)):
            pn = patch_nums[si]
            ratio = si / self.draft_model.num_stages_minus_1 if self.draft_model.num_stages_minus_1 > 0 else 0
            draft_cur_L += pn * pn

            # 前向传播
            draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)
            x = draft_next_token_map
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)

            # CFG处理
            t = cfg * ratio
            draft_logits_BlV = (1 + t) * draft_logits_BlV[:B] - t * draft_logits_BlV[B:]

            # === 修复2: 使用draft专用随机数生成器 ===
            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=draft_rng,  # 使用draft专用RNG
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            # 获取embedding
            if not more_smooth:
                draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), 
                    tau=draft_gum_t, 
                    hard=False, 
                    dim=-1, 
                    rng=draft_rng  # 使用draft专用RNG
                ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            draft_h_BChw = draft_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)

            # 更新状态
            draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            # 准备下一阶段
            if si != total_stages - 1:
                next_pn = patch_nums[si + 1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                draft_token_hub.append(draft_next_token_map.clone())
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L:draft_cur_L + next_pn * next_pn]
                )
                draft_next_token_map = draft_next_token_map.repeat(2, 1, 1)

        # 关闭draft模型的KV缓存
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)

        # 如果draft模型已经生成完所有阶段
        if entry_num >= total_stages:
            return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)

        # === 阶段2: Target模型接管并生成剩余阶段 ===
        # 初始化target模型参数
        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.target_model.num_classes)), dim=0)
        )
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        target_first_token_map = (
            target_sos.unsqueeze(1).expand(2*B, self.target_model.first_l, -1)
            + self.target_model.pos_start.expand(2*B, self.target_model.first_l, -1)
            + target_lvl_pos[:, :self.target_model.first_l]
        )

        # 继承draft模型的状态
        target_f_hat = draft_f_hat.clone()
        
        # 构建target模型的token map
        if len(draft_token_hub) > 0:
            # 连接draft生成的tokens
            draft_tokens = torch.cat(draft_token_hub, dim=1)
            target_next_token_map = self.target_model.word_embed(draft_tokens) + target_lvl_pos[:, self.target_model.first_l:draft_cur_L]
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)
            target_next_token_map = torch.cat([target_first_token_map, target_next_token_map], dim=1)
        else:
            target_next_token_map = target_first_token_map

        # 启用target模型的KV缓存
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        target_cur_L = draft_cur_L

        # === 修复3: 设置掩码 ===
        start_points = [0,1,5,14,30,55,91,155,255,424]
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Target模型生成剩余阶段
        for si in range(entry_num, total_stages):
            pn = patch_nums[si]
            ratio = si / self.target_model.num_stages_minus_1 if self.target_model.num_stages_minus_1 > 0 else 0
            target_cur_L += pn * pn

            # 前向传播
            target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)
            x = target_next_token_map

            # === 根据sd_mask参数选择掩码策略（便于消融实验） ===
            if sd_mask != 0:
                pindex = exit_points[si]
                sindex = start_points[si]
                
                if sd_mask == 1:
                    # SD掩码，包括未预测层的block-wise掩码
                    attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex].to(device)
                elif sd_mask == 2:
                    # SD掩码，不包括未预测层的block-wise掩码
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                elif sd_mask == 3:
                    attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]
                elif sd_mask == 4:
                    attn_bias = self.attn_bias_for_block[:,:,0:pindex,0:pindex].to(device)
                elif sd_mask == 5:
                    attn_bias = self.attn_bias_for_block[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                
                # 使用选定的掩码进行前向传播
                for blk in self.target_model.blocks:
                    x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
            else:
                # sd_mask = 0: 不使用掩码（baseline）
                for blk in self.target_model.blocks:
                    x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)

            target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

            # CFG处理
            t = cfg * ratio
            target_logits_BlV = (1 + t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]

            # === 修复4: 使用target专用随机数生成器 ===
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=target_rng,  # 使用target专用RNG
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            # 获取embedding
            if not more_smooth:
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)
            else:
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False,
                    dim=-1,
                    rng=target_rng  # 使用target专用RNG
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            # 更新状态
            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, target_f_hat, target_h_BChw
            )

            # 准备下一阶段
            if si != total_stages - 1:
                next_pn = patch_nums[si + 1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = (
                    self.target_model.word_embed(target_next_token_map)
                    + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                )
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)

        # 关闭target模型的KV缓存
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)

        # 返回最终图像
        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)

def measure_similarity_with_target_parallel(
    expansions,
    target_model,
    B,
    more_smooth=False,
):
    pass

@torch.no_grad()
def sdvar_autoregressive_infer_cfg_sd_test4(
    self,
    B: int,
    label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None,
    cfg: float = 1.5,
    top_k: int = 0,
    top_p: float = 0.0,
    more_smooth: bool = False,
    entry_num: int = 5,
    sd_mask: int = 5,  # 默认使用修改版block-wise掩码
) -> torch.Tensor:
    """
    修复版SDVAR推理函数，解决不同entry_num导致的随机数同步问题，并支持多种掩码策略
    
    主要改进：
    1. 为draft和target模型创建分离的随机数生成器
    2. 确保target模型始终从相同的随机数状态开始生成
    3. 添加完整的掩码支持，便于消融实验
    4. 保证相同模型和初始化下，无论entry_num如何，最终输出保持一致
    
    :param B: batch size
    :param label_B: imagenet label; if None, randomly sampled
    :param g_seed: random seed
    :param cfg: classifier-free guidance ratio
    :param top_k: top-k sampling
    :param top_p: top-p sampling
    :param more_smooth: smoothing the pred using gumbel softmax
    :param entry_num: 模型切换点，draft_model生成前entry_num个阶段
    :param sd_mask: 掩码类型选择，便于消融实验
                    0: 不使用掩码（baseline）
                    1: 使用SD掩码，包括未预测层的block-wise掩码
                    2: 使用SD掩码，不包括未预测层的block-wise掩码  
                    3: 使用标准因果掩码
                    4: 使用严格block-wise掩码
                    5: 使用修改版block-wise掩码（默认，推荐）
    :return: reconstructed image (B, 3, H, W) in [0, 1]
    """
    # === 修复1: 创建分离的随机数生成器 ===
    if g_seed is not None:
        # 为draft模型创建专用的随机数生成器
        draft_rng = torch.Generator(device=self.draft_model.lvl_1L.device)
        draft_rng.manual_seed(g_seed)
        
        # 为target模型创建专用的随机数生成器，使用相同种子
        target_rng = torch.Generator(device=self.target_model.lvl_1L.device) 
        target_rng.manual_seed(g_seed)
        
        # label生成使用target的随机数生成器
        label_rng = target_rng
    else:
        draft_rng = None
        target_rng = None
        label_rng = None

    # 验证模型兼容性
    assert self.draft_model.patch_nums == self.target_model.patch_nums
    assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
    patch_nums = self.draft_model.patch_nums
    total_stages = len(patch_nums)
    
    # 处理标签
    if label_B is None:
        label_B = torch.multinomial(
            self.target_model.uniform_prob, 
            num_samples=B, 
            replacement=True, 
            generator=label_rng
        ).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full(
            (B,),
            fill_value=self.target_model.num_classes if label_B < 0 else label_B,
            device=self.target_model.lvl_1L.device
        )

    # === 阶段1: Draft模型生成前entry_num个阶段 ===
    # 初始化draft模型参数
    draft_sos = draft_cond_BD = self.draft_model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=self.draft_model.num_classes)), dim=0)
    )
    draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
    draft_next_token_map = (
        draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
        + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
        + draft_lvl_pos[:, :self.draft_model.first_l]
    )
    draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae, patch_nums[-1], patch_nums[-1])
    draft_cur_L = 0
    draft_token_hub = []

    # 启用draft模型的KV缓存
    for blk in self.draft_model.blocks:
        blk.attn.kv_caching(True)

    # Draft模型生成
    for si in range(min(entry_num, total_stages)):
        pn = patch_nums[si]
        ratio = si / self.draft_model.num_stages_minus_1 if self.draft_model.num_stages_minus_1 > 0 else 0
        draft_cur_L += pn * pn

        # 前向传播
        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)
        x = draft_next_token_map
        for blk in self.draft_model.blocks:
            x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
        draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)

        # CFG处理
        t = cfg * ratio
        draft_logits_BlV = (1 + t) * draft_logits_BlV[:B] - t * draft_logits_BlV[B:]

        # === 修复2: 使用draft专用随机数生成器 ===
        draft_idx_Bl = sample_with_top_k_top_p_(
            draft_logits_BlV,
            rng=draft_rng,  # 使用draft专用RNG
            top_k=top_k,
            top_p=top_p,
            num_samples=1
        )[:, :, 0]

        # 获取embedding
        if not more_smooth:
            draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
        else:
            draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            draft_h_BChw = gumbel_softmax_with_rng(
                draft_logits_BlV.mul(1 + ratio), 
                tau=draft_gum_t, 
                hard=False, 
                dim=-1, 
                rng=draft_rng  # 使用draft专用RNG
            ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        draft_h_BChw = draft_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)

        # 更新状态
        draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
            si, total_stages, draft_f_hat, draft_h_BChw
        )

        # 准备下一阶段
        if si != total_stages - 1:
            next_pn = patch_nums[si + 1]
            draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
            draft_token_hub.append(draft_next_token_map.clone())
            draft_next_token_map = (
                self.draft_model.word_embed(draft_next_token_map)
                + draft_lvl_pos[:, draft_cur_L:draft_cur_L + next_pn * next_pn]
            )
            draft_next_token_map = draft_next_token_map.repeat(2, 1, 1)

    # 关闭draft模型的KV缓存
    for blk in self.draft_model.blocks:
        blk.attn.kv_caching(False)

    # 如果draft模型已经生成完所有阶段
    if entry_num >= total_stages:
        return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)

    # === 阶段2: Target模型接管并生成剩余阶段 ===
    # 初始化target模型参数
    target_sos = target_cond_BD = self.target_model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=self.target_model.num_classes)), dim=0)
    )
    target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
    target_first_token_map = (
        target_sos.unsqueeze(1).expand(2*B, self.target_model.first_l, -1)
        + self.target_model.pos_start.expand(2*B, self.target_model.first_l, -1)
        + target_lvl_pos[:, :self.target_model.first_l]
    )

    # 继承draft模型的状态
    target_f_hat = draft_f_hat.clone()
    
    # 构建target模型的token map
    if len(draft_token_hub) > 0:
        # 连接draft生成的tokens
        draft_tokens = torch.cat(draft_token_hub, dim=1)
        target_next_token_map = self.target_model.word_embed(draft_tokens) + target_lvl_pos[:, self.target_model.first_l:draft_cur_L]
        target_next_token_map = target_next_token_map.repeat(2, 1, 1)
        target_next_token_map = torch.cat([target_first_token_map, target_next_token_map], dim=1)
    else:
        target_next_token_map = target_first_token_map

    # 启用target模型的KV缓存
    for blk in self.target_model.blocks:
        blk.attn.kv_caching(True)

    target_cur_L = draft_cur_L

    # === 设置掩码参数（用于消融实验） ===
    start_points = [0,1,5,14,30,55,91,155,255,424]
    exit_points = [1,5,14,30,55,91,155,255,424,680]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Target模型生成剩余阶段
    for si in range(entry_num, total_stages):
        pn = patch_nums[si]
        ratio = si / self.target_model.num_stages_minus_1 if self.target_model.num_stages_minus_1 > 0 else 0
        target_cur_L += pn * pn

        # 前向传播
        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)
        x = target_next_token_map

        # === 根据sd_mask参数选择掩码策略（便于消融实验） ===
        if sd_mask != 0:
            pindex = exit_points[si]
            sindex = start_points[si]
            
            if sd_mask == 1:
                # SD掩码，包括未预测层的block-wise掩码
                attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex].to(device)
            elif sd_mask == 2:
                # SD掩码，不包括未预测层的block-wise掩码
                attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                attn_bias[:, :, sindex:pindex, :] = 0.0
                attn_bias = attn_bias.to(device)
            elif sd_mask == 3:
                # 标准因果掩码
                attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]
            elif sd_mask == 4:
                # 严格block-wise掩码
                attn_bias = self.attn_bias_for_block[:,:,0:pindex,0:pindex].to(device)
            elif sd_mask == 5:
                # 修改版block-wise掩码（默认推荐）
                attn_bias = self.attn_bias_for_block[:, :, 0:pindex, 0:pindex].clone()
                attn_bias[:, :, sindex:pindex, :] = 0.0
                attn_bias = attn_bias.to(device)
            
            # 使用选定的掩码进行前向传播
            for blk in self.target_model.blocks:
                x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
        else:
            # sd_mask = 0: 不使用掩码（baseline）
            for blk in self.target_model.blocks:
                x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)

        target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

        # CFG处理
        t = cfg * ratio
        target_logits_BlV = (1 + t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]

        # === 使用target专用随机数生成器进行采样 ===
        target_idx_Bl = sample_with_top_k_top_p_(
            target_logits_BlV,
            rng=target_rng,  # 使用target专用RNG，保证一致性
            top_k=top_k,
            top_p=top_p,
            num_samples=1
        )[:, :, 0]

        # 获取embedding
        if not more_smooth:
            target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)
        else:
            target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            target_h_BChw = gumbel_softmax_with_rng(
                target_logits_BlV.mul(1 + ratio),
                tau=target_gum_t,
                hard=False,
                dim=-1,
                rng=target_rng  # 使用target专用RNG
            ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        target_h_BChw = target_h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

        # 更新状态
        target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
            si, total_stages, target_f_hat, target_h_BChw
        )

        # 准备下一阶段
        if si != total_stages - 1:
            next_pn = patch_nums[si + 1]
            target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
            target_next_token_map = (
                self.target_model.word_embed(target_next_token_map)
                + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
            )
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)

    # 关闭target模型的KV缓存
    for blk in self.target_model.blocks:
        blk.attn.kv_caching(False)

    # 返回最终图像
    return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)
    
# ---------------------------------------------------------------------
#  Test‑5 logits comparison  -------------------------------------------
# ---------------------------------------------------------------------

# ------------------------------------------------------------------
#  Test-5 -- 对齐 logits（KV-cache ON，shape-safe）  ✅
# ------------------------------------------------------------------
@torch.no_grad()
def sdvar_autoregressive_infer_cfg_sd_test5(
    self,
    B: int,
    label_B: Optional[Union[int, torch.LongTensor]] = None,
    *,
    g_seed: Optional[int] = None,
    cfg: float = 1.5,
    top_k: int = 0,
    top_p: float = 0.0,
    more_smooth: bool = False,      # 保留以防以后用
    entry_num: int = 2,             # 要比较的 “接管点”
    verbose: bool = False,          # 打印中间 shape 与 slice 信息
):
    """
    比较 draft / target 在 **entry_num-1** 阶段 token-slice 上的 logits。

    为什么用上一阶段？  
    部分 stage（如 2× 上采样）会使 token 数翻倍，直接比较
    entry_num 阶段会出现切片长度不一致（4 ↔ 9）。  
    取 *上一阶段* 则两边长度必相同。
    """
    # ---------- 基本检查 ----------
    draft, target = self.draft_model, self.target_model
    assert draft.patch_nums == target.patch_nums, "patch_nums 不一致"
    pnums = draft.patch_nums
    num_stage_m1 = len(pnums) - 1
    assert 1 <= entry_num <= num_stage_m1, "entry_num 必须 ∈[1, len-1]"

    # ---------- RNG ----------
    rng = torch.Generator(device=target.lvl_1L.device)
    if g_seed is not None:
        rng.manual_seed(g_seed)

    # ---------- 标签 ----------
    if label_B is None:
        label_B = torch.multinomial(
            target.uniform_prob, num_samples=B, generator=rng
        ).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), label_B, device=target.lvl_1L.device)

    # ---------- slice 位置：上一 stage ----------
    prev_stage   = entry_num - 1
    slice_len    = pnums[prev_stage] ** 2
    cum_tokens   = sum(p ** 2 for p in pnums[:prev_stage])
    sidx, eidx   = cum_tokens, cum_tokens + slice_len
    if verbose:
        print(f"[TEST5] prev_stage={prev_stage}  slice={slice_len}  pos=({sidx},{eidx})")

    # ---------- 工具：初始化上下文并打开 KV-cache ----------
    def _make_ctx(model):
        sos, cond, _, lvl_pos, nxt, f_hat = self.init_param(model, B, label_B)
        for blk in model.blocks:            # <-- KV-cache ON
            blk.attn.kv_caching(True)
        return sos, cond, lvl_pos, nxt, f_hat

    # ============= 1. draft 先跑到 prev_stage 末尾 =============
    d_sos, d_cond, d_lvl, d_next, d_f = _make_ctx(draft)
    cur_L = 0
    for si, pn in enumerate(pnums):
        if si > prev_stage:
            break
        x = d_next
        for blk in draft.blocks:
            x = blk(x=x, cond_BD=d_cond, attn_bias=None)
        logits = draft.get_logits(x, d_cond)
        idx = sample_with_top_k_top_p_(logits, rng=rng, num_samples=1)[:, :, 0]
        h   = self.vae_quant_proxy[0].embedding(idx).transpose(1, 2).reshape(B, draft.Cvae, pn, pn)
        d_f, d_next = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(pnums), d_f, h)

        cur_L += pn * pn
        if si != len(pnums) - 1:            # 预处理下一 stage 的输入
            nxt_pn = pnums[si + 1]
            d_next = d_next.view(B, draft.Cvae, -1).transpose(1, 2)
            d_next = draft.word_embed(d_next) + d_lvl[:, cur_L : cur_L + nxt_pn * nxt_pn]
            d_next = d_next.repeat(2, 1, 1)

    # ============= 2. target 复用前缀，构造同样 slice ==========
    t_sos, t_cond, t_lvl, t_next0, _ = _make_ctx(target)
    if cur_L:                               # 把 draft 已生成的 token 嵌入 target
        prev_tok = d_next[:, :cur_L].view(B, draft.Cvae, -1).transpose(1, 2)
        prev_emb = target.word_embed(prev_tok) + t_lvl[:, 1 : 1 + cur_L]
        t_next   = torch.cat([t_next0, prev_emb.repeat(2, 1, 1)], dim=1)
    else:
        t_next = t_next0

    slice_tok = t_next[:, sidx:eidx]        # (2B, slice_len, C)
    if verbose:
        print("[TEST5] slice_tok shape:", slice_tok.shape)

    # ---------- 计算 logits ----------
    def _forward_slice(model, cond):
        x = slice_tok.clone()
        for blk in model.blocks:
            x = blk(x=x, cond_BD=cond, attn_bias=None)
        return model.get_logits(x, cond)[:B]   # 只拿前 B，去掉 CFG 翻倍

    lg_d = _forward_slice(draft,  d_cond)
    lg_t = _forward_slice(target, t_cond)

    # ---------- 对齐 CFG 缩放并比较 ----------
    ratio  = cfg * (entry_num / num_stage_m1)
    scale  = 1 + ratio
    maxdiff = (scale * lg_t - scale * lg_d).abs().max().item()
    print(f"[TEST5][COMPARE] entry_num={entry_num}  max|Δlogits| = {maxdiff:.6f}")

    # ---------- 关闭 KV-cache ----------
    for m in (draft, target):
        for blk in m.blocks:
            blk.attn.kv_caching(False)

    # 返回 draft 图像，方便肉眼对比
    return self.vae_proxy[0].fhat_to_img(d_f).add_(1).mul_(0.5)


def get_t_per_token(patch_nums, cfg, device=None):
    num_stages_minus_1 = len(patch_nums) - 1
    t_per_token = []
    for si, p in enumerate(patch_nums):
        t = cfg * (si / num_stages_minus_1)
        token_count = p * p
        t_per_token += [t] * token_count
    return torch.tensor(t_per_token, device=device).view(1, -1, 1)  # shape: (1, total_tokens, 1)
