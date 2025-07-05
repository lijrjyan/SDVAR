# -*- coding: utf-8 -*-
"""
SDVAR辅助函数模块

这个文件包含SDVAR的辅助函数，暂时存放非核心功能的实现。
核心的并行推理功能保留在main var.py文件中。

包含的功能：
1. SDVAR的初始化和参数设置辅助函数
2. 测试和实验用的函数 (如test3)  
3. 历史版本的推测解码实现
4. 调试和可视化工具
"""

import math
import time
from functools import partial
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from .helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_


class SDVARHelpers:
    """SDVAR辅助函数类，包含测试和实验用的方法"""
    
    def __init__(self, sdvar_instance):
        """
        初始化辅助函数类
        :param sdvar_instance: SDVAR主实例的引用
        """
        self.sdvar = sdvar_instance
        self.draft_model = sdvar_instance.draft_model
        self.target_model = sdvar_instance.target_model
    
    # =======================================================================
    # SDVAR Test3 函数 - 推测解码测试版本
    # =======================================================================
    
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test3(
        self,
        B: int,
        label_B,
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10, 
        sd_mask: int = 0
    ) -> torch.Tensor:
        """
        SDVAR推测解码test3版本 - 多种掩码策略的对比测试
        
        这个函数被移动到辅助模块中，主要用于实验和测试不同的掩码策略。
        生产环境推荐使用主模块中的 sdvar_autoregressive_infer_cfg_parallel_v1。
        
        === 核心设计思想 ===
        这个函数实现了混合模型推测解码：
        1. Draft模型(更快)：生成前entry_num层的tokens
        2. Target模型(更精确)：生成剩余层的tokens
        3. 使用不同的注意力掩码策略控制信息流
        
        === 掩码类型 ===
        - sd_mask = 0: 无特殊掩码
        - sd_mask = 1: SD掩码 - 包括当前层
        - sd_mask = 2: SD掩码 - 排除当前层  
        - sd_mask = 3: 标准因果掩码
        - sd_mask = 4: Block掩码 - 包括当前层
        - sd_mask = 5: Block掩码 - 排除当前层
        """
        print(f"[SDVAR Test3] Running with entry_num={entry_num}, sd_mask={sd_mask}")
        
        # 通用参数设置
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        patch_nums = self.draft_model.patch_nums
        total_stages = len(patch_nums)
        
        # 处理随机数生成器
        if g_seed is not None:
            rng = torch.Generator(device=self.target_model.lvl_1L.device)
            rng.manual_seed(g_seed)
        else:
            rng = None
        
        # 处理标签
        if label_B is None:
            label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,), fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )
        
        # === Draft模型阶段 ===
        draft_sos, draft_cond_BD, draft_cond_BD_or_gss, \
        draft_lvl_pos, draft_first_token_map, draft_f_hat = self.sdvar.init_param(self.draft_model, B, label_B)
        
        draft_cur_L = 0
        draft_next_token_map = draft_first_token_map
        draft_token_hub = []
        
        # 启用KV缓存
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)
        
        # Draft生成循环
        for si, pn in enumerate(patch_nums):
            if si >= entry_num:
                break
            
            ratio = si / (total_stages - 1)
            draft_cur_L += pn * pn
            x = draft_next_token_map
            
            # Draft前向传播
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)
            
            # CFG处理
            t = cfg * ratio
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]
            
            # 采样
            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]
            
            # 转换为VAE embedding
            if not more_smooth:
                draft_h_BChw = self.target_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)
            
            # 更新f_hat和token_map
            draft_f_hat, draft_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )
            
            # 准备下一层输入
            if si != total_stages - 1:
                next_pn = patch_nums[si+1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map) +
                    draft_lvl_pos[:, draft_cur_L : draft_cur_L + next_pn**2]
                )
                draft_next_token_map = draft_next_token_map.repeat(2,1,1)
            
            # 如果draft处理完所有层，直接返回
            if si == total_stages - 1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                return self.target_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)
        
        # 清理draft模型的KV缓存
        if len(draft_token_hub) != 0:   
            draft_token_hub = torch.cat(draft_token_hub, dim=1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
        # === Target模型阶段 ===
        start_points = [0,1,5,14,30,55,91,155,255,424]
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        pindex = exit_points[entry_num]
        sindex = start_points[entry_num]
        device = self.target_model.lvl_1L.device
        
        target_sos, target_cond_BD, target_cond_BD_or_gss, \
        target_lvl_pos, target_first_token_map, target_f_hat = self.sdvar.init_param(self.target_model, B, label_B)
        
        target_f_hat = draft_f_hat
        target_cur_L = 0
        
        # 处理draft tokens作为target的输入
        if len(draft_token_hub) > 0:
            target_next_token_map = draft_token_hub    
            target_next_token_map = (
                self.target_model.word_embed(target_next_token_map) + 
                target_lvl_pos[:,1:pindex]
            )
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)
            target_next_token_map = torch.cat([target_first_token_map, target_next_token_map], dim=1)
        else: 
            target_next_token_map = target_first_token_map
        
        # 启用target的KV缓存
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)
        
        # Target生成循环
        for si, pn in enumerate(patch_nums):
            if si < entry_num:
                continue
            
            ratio = si / (total_stages - 1)
            target_cur_L += pn * pn
            t = cfg * ratio 
            
            # 应用掩码策略
            if sd_mask != 0:
                if sd_mask == 1:
                    attn_bias = self.sdvar.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex].to(device)
                elif sd_mask == 2:
                    attn_bias = self.sdvar.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                elif sd_mask == 3:
                    attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]
                elif sd_mask == 4: 
                    attn_bias = self.sdvar.attn_bias_for_block[:,:,0:pindex,0:pindex].to(device)
                elif sd_mask == 5:
                    attn_bias = self.sdvar.attn_bias_for_block[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                
                x = target_next_token_map
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
                # 无掩码模式
                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
                else:
                    x = target_next_token_map
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
            
            # CFG和采样
            target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]
            
            # 转换为VAE embedding
            if not more_smooth:
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)
            else:
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio), tau=target_gum_t, hard=False, dim=-1, rng=rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)
            
            # 更新f_hat
            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(patch_nums), target_f_hat, target_h_BChw
            )
            
            # 准备下一层
            if si != total_stages - 1:
                next_pn = patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = (
                    self.target_model.word_embed(target_next_token_map) + 
                    target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                )
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)
        
        # 清理target的KV缓存
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)
        
        print(f"[SDVAR Test3] Completed with entry_num={entry_num}, sd_mask={sd_mask}")
        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)
    
    # =======================================================================
    # 其他辅助函数
    # =======================================================================
    
    def get_debug_info(self):
        """获取调试信息"""
        info = {
            "draft_model_depth": len(self.draft_model.blocks),
            "target_model_depth": len(self.target_model.blocks),
            "patch_nums": self.draft_model.patch_nums,
            "total_stages": len(self.draft_model.patch_nums),
            "vocab_size": self.draft_model.V,
            "embed_dim": self.draft_model.C,
            "vae_dim": self.draft_model.Cvae,
        }
        return info
    
    def estimate_speedup(self, entry_num: int, draft_latency: float, target_latency: float):
        """
        估算推测解码的理论加速比
        
        :param entry_num: draft模型处理的层数
        :param draft_latency: draft模型单层延迟
        :param target_latency: target模型单层延迟
        :return: 理论加速比
        """
        total_stages = len(self.draft_model.patch_nums)
        
        # 传统方式：target模型处理所有层
        traditional_time = target_latency * total_stages
        
        # SDVAR方式：draft处理前entry_num层，target处理剩余层
        sdvar_time = draft_latency * entry_num + target_latency * (total_stages - entry_num)
        
        speedup = traditional_time / sdvar_time if sdvar_time > 0 else 1.0
        
        return {
            "speedup": speedup,
            "traditional_time": traditional_time,
            "sdvar_time": sdvar_time,
            "draft_contribution": draft_latency * entry_num,
            "target_contribution": target_latency * (total_stages - entry_num)
        }
    
    def analyze_attention_patterns(self, sd_mask: int, sequence_length: int):
        """
        分析不同掩码策略的注意力模式
        
        :param sd_mask: 掩码类型 (0-5)
        :param sequence_length: 序列长度
        :return: 注意力模式分析
        """
        if sd_mask == 0:
            return "Standard causal mask - tokens can attend to all previous positions"
        elif sd_mask == 1:
            return "SD masking with current layer - block-wise attention including current generation layer"
        elif sd_mask == 2:
            return "SD masking without current layer - block-wise attention excluding current generation layer"
        elif sd_mask == 3:
            return "VAR original causal mask - strict triangular masking"
        elif sd_mask == 4:
            return "Block masking with current layer - only same-block attention"
        elif sd_mask == 5:
            return "Block masking without current layer - conservative block attention"
        else:
            return "Unknown mask type"
    
    def validate_tensor_shapes(self, stage_idx: int, tokens: torch.Tensor, expected_pn: int):
        """
        验证tensor形状是否符合预期
        
        :param stage_idx: 当前stage索引
        :param tokens: token tensor
        :param expected_pn: 预期的patch number
        """
        expected_tokens = expected_pn ** 2
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Stage {stage_idx}: Expected {expected_tokens} tokens (pn={expected_pn}), "
                f"got {tokens.shape[1]} tokens. Token shape: {tokens.shape}"
            )
        return True
    
    def compute_cumulative_positions(self):
        """计算累积位置信息，用于位置编码"""
        patch_nums = self.draft_model.patch_nums
        cumulative_tokens = []
        total = 0
        
        for pn in patch_nums:
            cumulative_tokens.append(total)
            total += pn ** 2
        
        # 添加最终位置
        cumulative_tokens.append(total)
        
        return {
            "start_points": cumulative_tokens[:-1],
            "exit_points": cumulative_tokens[1:],
            "total_tokens": total,
            "patch_nums": patch_nums
        }
    
    @torch.no_grad()
    def run_quality_comparison(self, B: int, label_B, entry_num: int, sd_mask: int,
                              g_seed: Optional[int] = None, cfg: float = 1.5):
        """
        运行质量对比测试，比较SDVAR与原始VAR的输出差异
        
        :param B: batch size
        :param label_B: 标签
        :param entry_num: 转换点
        :param sd_mask: 掩码类型
        :param g_seed: 随机种子
        :param cfg: CFG scale
        :return: 对比结果
        """
        print(f"[Quality Comparison] SDVAR vs Original VAR")
        print(f"   entry_num={entry_num}, sd_mask={sd_mask}")
        
        # 使用相同的随机种子
        if g_seed is not None:
            torch.manual_seed(g_seed)
        
        # 1. 生成SDVAR结果
        start_time = time.time()
        sdvar_result = self.sdvar.sdvar_autoregressive_infer_cfg_sd_test3(
            B=B, label_B=label_B, g_seed=g_seed, cfg=cfg,
            entry_num=entry_num, sd_mask=sd_mask
        )
        sdvar_time = time.time() - start_time
        
        # 2. 生成原始VAR结果 (target模型)
        if g_seed is not None:
            torch.manual_seed(g_seed)  # 重置种子
        
        start_time = time.time()
        var_result = self.target_model.autoregressive_infer_cfg(
            B=B, label_B=label_B, g_seed=g_seed, cfg=cfg
        )
        var_time = time.time() - start_time
        
        # 3. 计算差异
        mse_loss = torch.nn.functional.mse_loss(sdvar_result, var_result).item()
        l1_loss = torch.nn.functional.l1_loss(sdvar_result, var_result).item()
        
        # 4. 计算加速比
        speedup = var_time / sdvar_time if sdvar_time > 0 else 1.0
        
        comparison_result = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "speedup": speedup,
            "sdvar_time": sdvar_time,
            "var_time": var_time,
            "sdvar_result": sdvar_result,
            "var_result": var_result
        }
        
        print(f"[Quality Comparison] Results:")
        print(f"   MSE Loss: {mse_loss:.6f}")
        print(f"   L1 Loss: {l1_loss:.6f}")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   SDVAR Time: {sdvar_time:.2f}s")
        print(f"   VAR Time: {var_time:.2f}s")
        
        return comparison_result


def create_experimental_masks(patch_nums: Tuple, total_tokens: int):
    """
    创建实验用的注意力掩码
    
    这个函数是从主SDVAR类中分离出来的，专门用于实验不同的掩码策略
    """
    block_sizes = [p ** 2 for p in patch_nums]
    
    # 创建block_id映射
    block_ids = []
    for block, size in enumerate(block_sizes):
        block_ids += [block] * size
    block_ids = torch.tensor(block_ids)
    
    # 1. SD掩码 (推测解码掩码)
    attn_bias_for_sdmasking = torch.full((total_tokens, total_tokens), float('-inf'))
    for i in range(total_tokens):
        for j in range(total_tokens):
            if j > i:
                continue
            if block_ids[i] == block_ids[j] and i != j:
                continue
            attn_bias_for_sdmasking[i, j] = 0.0
    attn_bias_for_sdmasking = attn_bias_for_sdmasking.reshape(1, 1, total_tokens, total_tokens)
    
    # 2. Block掩码 (严格的block-wise)
    attn_bias_for_block = torch.full((total_tokens, total_tokens), float('-inf'))
    for i in range(total_tokens):
        for j in range(total_tokens):
            if block_ids[i] == block_ids[j]:
                attn_bias_for_block[i, j] = 0.0
    attn_bias_for_block = attn_bias_for_block.reshape(1, 1, total_tokens, total_tokens)
    
    return {
        "sd_mask": attn_bias_for_sdmasking,
        "block_mask": attn_bias_for_block,
        "block_ids": block_ids
    }


def analyze_mask_properties(mask: torch.Tensor, name: str):
    """
    分析掩码的性质
    
    :param mask: 注意力掩码 tensor
    :param name: 掩码名称
    :return: 掩码分析结果
    """
    mask_2d = mask.squeeze()  # 移除batch和head维度
    
    # 计算稀疏度
    total_elements = mask_2d.numel()
    allowed_connections = (mask_2d != float('-inf')).sum().item()
    sparsity = 1.0 - (allowed_connections / total_elements)
    
    # 分析对角线结构
    diagonal_allowed = torch.diag(mask_2d) != float('-inf')
    diagonal_ratio = diagonal_allowed.float().mean().item()
    
    # 分析上三角和下三角
    upper_tri = torch.triu(mask_2d, diagonal=1)
    lower_tri = torch.tril(mask_2d, diagonal=-1)
    
    upper_allowed = (upper_tri != float('-inf')).sum().item()
    lower_allowed = (lower_tri != float('-inf')).sum().item()
    
    analysis = {
        "name": name,
        "shape": mask_2d.shape,
        "sparsity": sparsity,
        "allowed_connections": allowed_connections,
        "total_elements": total_elements,
        "diagonal_ratio": diagonal_ratio,
        "upper_triangle_allowed": upper_allowed,
        "lower_triangle_allowed": lower_allowed,
        "is_causal": upper_allowed == 0,
        "is_symmetric": torch.allclose(mask_2d, mask_2d.T, equal_nan=True)
    }
    
    return analysis


# 工具函数：可视化掩码模式
def visualize_mask_pattern(mask: torch.Tensor, save_path: Optional[str] = None):
    """
    可视化注意力掩码的模式
    
    :param mask: 注意力掩码
    :param save_path: 保存路径 (可选)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        mask_2d = mask.squeeze().cpu().numpy()
        
        # 将-inf转换为0，0转换为1，便于可视化
        vis_mask = np.where(mask_2d == float('-inf'), 0, 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_mask, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Attention Allowed (1=Yes, 0=No)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Attention Mask Pattern')
        
        # 添加网格线显示block边界
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        cumulative = [0]
        for pn in patch_nums:
            cumulative.append(cumulative[-1] + pn**2)
        
        for pos in cumulative[1:-1]:  # 排除首尾
            plt.axhline(y=pos-0.5, color='red', linestyle='--', alpha=0.5)
            plt.axvline(x=pos-0.5, color='red', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mask visualization saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Warning: matplotlib not available, cannot visualize mask")


# 性能分析工具
class SDVARProfiler:
    """SDVAR性能分析工具"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        import time
        self.timings[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str):
        """结束计时"""
        import time
        if f"{name}_start" in self.timings:
            duration = time.time() - self.timings[f"{name}_start"]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            
            # 更新调用次数
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            
            del self.timings[f"{name}_start"]
            return duration
        return None
    
    def get_summary(self):
        """获取性能总结"""
        summary = {}
        for name, durations in self.timings.items():
            if isinstance(durations, list) and durations:
                summary[name] = {
                    "total_time": sum(durations),
                    "avg_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                    "call_count": len(durations)
                }
        return summary
    
    def print_summary(self):
        """打印性能总结"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("SDVAR Performance Summary")
        print("="*60)
        
        for name, stats in summary.items():
            print(f"{name}:")
            print(f"  Total Time: {stats['total_time']:.3f}s")
            print(f"  Average Time: {stats['avg_time']:.3f}s")
            print(f"  Min/Max Time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
            print(f"  Call Count: {stats['call_count']}")
            print() 