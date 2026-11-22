"""Encoder builder shared by training and every evaluator.

The training script and the evaluators (probe, retrieval, epoch curve) all
reconstruct the same EEG encoder from a config. Centralising it here keeps the
architecture defaults in one place so the call sites cannot drift.
"""

from __future__ import annotations

from eb_jepa.architectures import EEGEncoderTokens


def build_encoder(cfg, *, n_chans: int, n_times: int, chs_info,
                  n_windows: int) -> EEGEncoderTokens:
    """Build the REVE-style EEG token encoder described by ``cfg.model``."""
    return EEGEncoderTokens(
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=cfg.model.encoder_embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=n_windows,
        patch_size=cfg.model.get("patch_size", 200),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=chs_info,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    )
