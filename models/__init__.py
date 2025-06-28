"""
Sdvar  DEBUG / TEST UTILITIES
---------------------------------
* `build_vae_var_speculative_decoding`  —  **new arg** `same_weights_debug`  ⇢  if `True`,
  copies draft → target weights so两边参数完全一致，便于纯 RNG / entry_num 调试。
* `sdvar_autoregressive_infer_cfg_sd_test5` —  同 slice logits 对比，修复此前重复 `return`。
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
#  build helpers  ------------------------------------------------------
# ---------------------------------------------------------------------

def build_vae_var_speculative_decoding(
    device,
    # shared
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    num_classes=1000,
    depth_draft=16,
    depth_target=30,
    shared_aln=False,
    attn_l2_norm=True,
    flash_if_available=True,
    fused_if_available=True,
    init_adaln=0.5,
    init_adaln_gamma=1e-5,
    init_head=0.02,
    init_std=-1,
    similarity_thresh=0.8,
    *,
    same_weights_debug: bool = False,  # <── 新增
):
    """构建 draft / target / SDVAR；若 `same_weights_debug=True` 则拷贝权重。"""
    from .vqvae import VQVAE
    from .var import VAR, SDVAR

    # disable default init for speed
    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=True,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)

    # draft
    heads = depth_draft
    width = depth_draft * 64
    dpr = 0.1 * depth_draft / 24
    var_draft = VAR(
        vae_local=vae_local,
        num_classes=num_classes,
        depth=depth_draft,
        embed_dim=width,
        num_heads=heads,
        drop_path_rate=dpr,
        shared_aln=shared_aln,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
    ).to(device)
    var_draft.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )

    # target
    heads = depth_target
    width = depth_target * 64
    dpr = 0.1 * depth_target / 24
    var_target = VAR(
        vae_local=vae_local,
        num_classes=num_classes,
        depth=depth_target,
        embed_dim=width,
        num_heads=heads,
        drop_path_rate=dpr,
        shared_aln=shared_aln,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
    ).to(device)
    var_target.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )

    # ─── DEBUG: copy weights if requested ────────────────────────────
    if same_weights_debug:
        var_target.load_state_dict(var_draft.state_dict(), strict=True)
        print("[DEBUG] target weights cloned from draft (same_weights_debug=True)")

    sd_var = SDVAR(var_draft.to(device), var_target.to(device), similarity_thresh)
    return vae_local, var_draft, var_target, sd_var
