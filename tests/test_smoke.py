"""Smoke tests that need no HBN data: imports, configs, and one training step."""
from pathlib import Path

import mne
import pytest
import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.clip import SoftTargetCLIPPretrain
from eb_jepa.training.builder import build_encoder

ROOT = Path(__file__).resolve().parents[1]


def test_all_modules_import():
    import eb_jepa.datasets.hbn  # noqa: F401
    import eb_jepa.evaluation.clip_probe.epoch_curve  # noqa: F401
    import eb_jepa.evaluation.clip_probe.probe_traintest  # noqa: F401
    import eb_jepa.evaluation.clip_probe.retrieval  # noqa: F401
    import eb_jepa.training.clip_pretrain  # noqa: F401


def _chs_info(n=129):
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(montage.ch_names[:n], 200.0, "eeg")
    info.set_montage(montage)
    return info["chs"]


@pytest.mark.parametrize("cfg_name,n_params_m", [("depth22", 69.2), ("depth12", 37.9)])
def test_one_training_step(cfg_name, n_params_m):
    cfg = OmegaConf.load(ROOT / "configs" / "train" / f"{cfg_name}.yaml")
    torch.manual_seed(0)
    enc = build_encoder(cfg, n_chans=129, n_times=400, chs_info=_chs_info(), n_windows=1)
    # Encoder size quoted in the paper's compute table (App. C).
    assert abs(sum(p.numel() for p in enc.parameters()) / 1e6 - n_params_m) < 0.1
    head = MovieCLIPHead(eeg_in_dim=512, vision_in_dim=1408, proj_dim=512, temperature=0.07,
                         drop_proj=0.5, vision_passthrough=False, n_residual_blocks=1)
    model = SoftTargetCLIPPretrain(enc, head, alpha=cfg.loss.soft_alpha,
                                   tau_teacher=cfg.loss.soft_tau_teacher,
                                   temporal_buffer_s=cfg.loss.temporal_buffer_s)
    B = 8
    eeg = torch.randn(B, 1, 129, 400)
    vis = torch.randn(B, 1, 1408)
    t_starts = torch.arange(B, dtype=torch.float32) * 10
    loss, info = model(eeg, vis, torch.zeros(B, dtype=torch.long), t_starts)
    loss.backward()
    assert torch.isfinite(loss)
    assert info["clip_tau_teacher"] == 0.05
