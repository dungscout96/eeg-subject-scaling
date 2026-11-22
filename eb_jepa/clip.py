"""Soft-target CLIP objective for EEG <-> V-JEPA-2 alignment.

The EEG encoder produces one embedding per 2 s window; it is aligned to the
mean-centred, frozen V-JEPA-2 embedding of the same movie span with a
symmetric cross-entropy against a blended target -- a hard diagonal mixed with
a teacher distribution built from V-JEPA-2's own window-to-window cosine
similarity (Andonian et al., 2022). See the paper's Appendix B.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCLIPPretrain(nn.Module):
    """Soft-target CLIP via Andonian-2022-style distillation from V-JEPA-2 similarity.

    Blends a hard diagonal target with a teacher distribution built from the
    pairwise cosine similarity of the (mean-centered, shot-mean) V-JEPA-2
    targets themselves. The teacher is *fixed* — V-JEPA-2 is frozen — so the
    Andonian progressive schedule collapses to a fixed alpha.

    Objective for the e→v direction:
        p_ij  = softmax_j(logits_ij)                   # logits = scale · z_eeg z_vis^T
        S_ij  = <v_i, v_j> / (‖v_i‖ ‖v_j‖)             # centered-cosine on the target space
        q_ij  = softmax_j(S_ij / tau_teacher)          # teacher soft label
        y_ij  = 1{i = j}
        t     = (1 - alpha) · y + alpha · q            # blended target
        L_e2v = -mean_i Σ_j t_ij log p_ij
    Symmetric v→e uses logits.T and t.T (q is symmetric by construction, so t is too).

    A temporal buffer masks cross-pairs with |Δt| < buffer_s out of both the
    student softmax denominator *and* the teacher softmax denominator, so those
    shot-cut-boundary pairs contribute to neither the target nor the prediction.
    Self-pairs are never masked.

    Forward consumes ``(eeg, frame_embedding_target, scene_ids, t_starts)`` —
    scene_ids is accepted for interface compatibility with the dataset's batch
    layout but is unused (all label structure comes from the teacher).
    """

    def __init__(
        self,
        encoder,
        clip_head,
        *,
        alpha: float = 0.5,
        tau_teacher: float = 0.1,
        temporal_buffer_s: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.clip_head = clip_head
        self.alpha = float(alpha)
        self.tau_teacher = float(tau_teacher)
        self.temporal_buffer_s = float(temporal_buffer_s)

    def forward(
        self,
        eeg: torch.Tensor,
        frame_embedding_target: torch.Tensor,
        scene_ids: torch.Tensor,
        t_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        del scene_ids  # unused; the teacher carries all label structure
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        pooled = self.encoder.pool_to_windows(tokens)            # [B, D, T, 1, 1]
        z_eeg = self.clip_head.project_eeg(pooled)               # [B*T, P]
        tgt = frame_embedding_target.to(z_eeg.dtype)
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])                # [N, D_vjepa]
        z_vis = self.clip_head.project_vision(tgt_flat)          # [N, P]
        scale = self.clip_head.logit_scale.exp().clamp(max=100.0)
        logits = scale * (z_eeg @ z_vis.T)                       # [N, N]
        n = logits.shape[0]
        device = logits.device

        # Temporal-buffer mask on cross-pairs (self always kept).
        ts = t_starts.reshape(-1).to(device)
        dt = (ts.unsqueeze(0) - ts.unsqueeze(1)).abs()
        eye = torch.eye(n, dtype=torch.bool, device=device)
        buffer_mask = (dt < self.temporal_buffer_s) & ~eye        # [N, N]

        # Teacher soft targets from centered-cosine on the target space.
        v = F.normalize(tgt_flat.float(), dim=-1)
        S = v @ v.T                                              # [N, N] in [-1, 1]
        neg_inf = torch.finfo(S.dtype).min
        S_masked = S.masked_fill(buffer_mask, neg_inf)
        q = F.softmax(S_masked / self.tau_teacher, dim=-1)       # [N, N]

        # Blended target: (1 - α) · diag + α · q. q already zero on buffer cells.
        y = torch.eye(n, dtype=q.dtype, device=device)
        t = (1.0 - self.alpha) * y + self.alpha * q

        # Student log-probs with the same buffer mask.
        logits_masked = logits.masked_fill(buffer_mask, torch.finfo(logits.dtype).min)
        log_p_e2v = F.log_softmax(logits_masked, dim=-1)
        log_p_v2e = F.log_softmax(logits_masked.T, dim=-1)

        loss_e2v = -(t.to(log_p_e2v.dtype) * log_p_e2v).sum(dim=-1).mean()
        loss_v2e = -(t.T.to(log_p_v2e.dtype) * log_p_v2e).sum(dim=-1).mean()
        loss = 0.5 * (loss_e2v + loss_v2e)

        with torch.no_grad():
            labels = torch.arange(n, device=device)
            q_entropy = -(q.clamp_min(1e-12).log() * q).sum(dim=-1).mean()
            eff_pos = q_entropy.exp()
            n_excl = buffer_mask.float().sum(dim=-1).mean().item()
            loss_dict = {
                "clip_loss": loss.item(),
                "clip_loss_e2v": loss_e2v.item(),
                "clip_loss_v2e": loss_v2e.item(),
                "clip_top1_e2v": (logits.argmax(-1) == labels).float().mean().item(),
                "clip_top1_v2e": (logits.argmax(0) == labels).float().mean().item(),
                "clip_logit_scale": scale.item(),
                "clip_teacher_entropy": q_entropy.item(),
                "clip_teacher_eff_positives": eff_pos.item(),
                "clip_n_excluded_mean": n_excl,
                "clip_alpha": self.alpha,
                "clip_tau_teacher": self.tau_teacher,
                "total_loss": loss.item(),
            }
        return loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor, keep_channels: bool = False) -> torch.Tensor:
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        return self.encoder.pool_to_windows(tokens, keep_channels=keep_channels)
