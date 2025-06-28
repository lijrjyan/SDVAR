"""
Sdvar  DEBUG / TEST UTILITIES  – rev-4  (KV-cache **ON**)
---------------------------------------------------------
* same as rev-3 but **re-enables KV caching** everywhere so that the
  speculative-decoding path now matches the real inference path.

  ▸   helper `_ctx()` now calls `blk.attn.kv_caching(True)` so every
      forward keeps its keys/values for subsequent tokens.
  ▸   after each big section we turn it back **off** to avoid memory
      leaks during the analysis-only pass.

* `test-5` unchanged in logic – it now benefits from the speed-up of
  cached KV while continuing to compute the exact same slice logits.
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
    same_weights_debug: bool = False,
):
    """Builder that can optionally clone draft weights into target for debugging."""

    # delay heavy imports – keep top of file light
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

    # ─── shared VQVAE ────────────────────────────────────────────────
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=True,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)

    def _make_var(depth):
        heads = depth
        width = depth * 64
        dpr = 0.1 * depth / 24
        var = VAR(
            vae_local=vae_local,
            num_classes=num_classes,
            depth=depth,
            embed_dim=width,
            num_heads=heads,
            drop_path_rate=dpr,
            shared_aln=shared_aln,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        ).to(device)
        var.init_weights(
            init_adaln=init_adaln,
            init_adaln_gamma=init_adaln_gamma,
            init_head=init_head,
            init_std=init_std,
        )
        return var

    var_draft  = _make_var(depth_draft)
    var_target = _make_var(depth_target)

    # ─── DEBUG: clone draft weights into target ──────────────────────
    if same_weights_debug:
        var_target.load_state_dict(var_draft.state_dict(), strict=True)
        print("[DEBUG] target weights cloned from draft (same_weights_debug=True)")

    sd_var = SDVAR(var_draft, var_target, similarity_thresh)
    return vae_local, var_draft, var_target, sd_var
