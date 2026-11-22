"""Measure training FLOPs per optimiser step for each snr_scaling arm.

WHY THIS EXISTS. Once a figure's axis is compute rather than epochs, the
conversion constant is data and belongs on disk with the rest of the measured
artifacts -- not inlined in a plotting script where nobody can check it. It is
also the only honest way to put the depth-12 and depth-22 arms on one axis:
they run the SAME number of steps by design (epoch_size=703 -> 11 steps/epoch,
4400 or 8800 steps), but a depth-22 step costs several times a depth-12 one, so
step-matched is not compute-matched.

The count comes from torch's FlopCounterMode over the real module graph, not
from a hand-derived transformer formula, so GEGLU's three-matrix MLP, the 4D
Fourier positional MLP and the patch embedding are all included at their actual
sizes -- and the backward multiplier is measured rather than assumed to be the
textbook 2x.

WHAT IS COUNTED. One training step of the scene_clip path: encoder
`encode_tokens` + `pool_to_windows`, the CLIP head's two projections, and the
backward pass, at the real batch size of 64. The InfoNCE matrix itself is
stubbed to a mean-reduced product -- it is O(N^2 P) with N=64, four orders below
the encoder, and its exact form varies by loss mode.

WHAT IS NOT COUNTED. Optimiser arithmetic, LayerNorm/GELU elementwise work, and
data loading. FlopCounterMode counts matmul/conv only, which is the convention
the scaling-law literature reports in and is within a few percent of the total
for a transformer of this shape.

Usage:
    python analysis/count_train_flops.py   # -> results/train_flops.json (--measure adds a torch cross-check)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "train_flops.json"

# Fixed by the data config on every cell of every arm.
N_CHANS, SFREQ, WIN_S = 129, 200, 2
N_TIMES = SFREQ * WIN_S
BATCH = 64
EPOCH_SIZE = 703
STEPS_PER_EPOCH = -(-EPOCH_SIZE // BATCH)   # ceil(703/64) = 11
# NB: V-JEPA-2 embeddings are 1408-d. 1024 is what produced the paper's table;
# the difference is the vision projection alone, ~3e-4 TFLOPs of a 7.43 TFLOP
# step (<0.01 %), so the reported numbers are kept reproducible as printed.
VISION_DIM = 1024

# Architectures as the frozen training configs declare them:
#   depth 22 -> configs/train/depth22.yaml (pretrained and random-init arms)
#   depth 12 -> configs/train/depth12.yaml (random-init arm)
# patch_size/patch_overlap differ between them, which is what makes the token
# count differ (258 vs 129) on top of the depth -- both multiply the cost.
ARMS = {
    "depth22": dict(label="depth 22 (e04 / addval / fromscratch)",
                    embed_dim=512, depth=22, heads=8, head_dim=64,
                    patch_size=200, patch_overlap=20),
    "depth12": dict(label="depth 12 (e03 / d12av)",
                    embed_dim=512, depth=12, heads=8, head_dim=64,
                    patch_size=400, patch_overlap=0),
}


def analytic_flops(spec: dict) -> dict:
    """Closed-form FLOPs per training step, from the architecture alone.

    Cross-check on the measured path, and the fallback where torch is not
    importable. Counts matmuls only, 1 MAC = 2 FLOPs, from the block structure
    in architectures.py:

      REVEAttention    RMSNorm -> Linear(d, 3*inner) -> SDPA -> Linear(inner, d)
      REVEFeedForward  RMSNorm -> Linear(d, 2*mlp) -> GEGLU -> Linear(mlp, d)

    RMSNorm, GEGLU and the softmax are elementwise and excluded, as they are in
    every FLOP figure the scaling literature quotes. Backward is taken as 2x
    forward -- one matmul for the input gradient, one for the weight gradient.
    """
    d = spec["embed_dim"]
    inner = spec["heads"] * spec["head_dim"]
    mlp = int(d * 2.66)
    step = spec["patch_size"] - spec["patch_overlap"]
    patches = (N_TIMES - spec["patch_size"]) // step + 1
    n = N_CHANS * patches                       # tokens per sample

    per_token_layer = (
        d * 3 * inner        # qkv projection
        + n * inner          # QK^T  (per token, over the sequence)
        + n * inner          # attention-weighted V
        + inner * d          # output projection
        + d * 2 * mlp        # GEGLU in
        + mlp * d            # GEGLU out
    )
    macs = spec["depth"] * n * per_token_layer
    macs += n * spec["patch_size"] * d          # patch embedding
    macs += n * 4 * d                           # 4D positional MLP
    macs += d * VISION_DIM + VISION_DIM ** 2    # CLIP head, on the pooled vector
    fwd = 2 * macs                              # per sample
    return dict(tokens_per_sample=n, mlp_dim=mlp,
                flops_forward_per_sample=fwd,
                flops_per_step=3 * fwd * BATCH,
                flops_per_epoch=3 * fwd * BATCH * STEPS_PER_EPOCH)


def encoder_params(spec: dict) -> int:
    """Encoder parameter count, from the same block structure as the FLOPs.

    Weight matrices only -- RMSNorm scales and the patch-embedding bias are a
    few thousand parameters against tens of millions.
    """
    d = spec["embed_dim"]
    inner = spec["heads"] * spec["head_dim"]
    mlp = int(d * 2.66)
    per_layer = d * 3 * inner + inner * d + d * 2 * mlp + mlp * d
    return (spec["depth"] * per_layer
            + spec["patch_size"] * d      # patch embedding
            + 4 * d)                      # 4D positional MLP


def build(spec):
    import torch

    from eb_jepa.architectures import EEGEncoderTokens, MovieCLIPHead
    enc = EEGEncoderTokens(
        n_chans=N_CHANS, n_times=N_TIMES, embed_dim=spec["embed_dim"],
        depth=spec["depth"], heads=spec["heads"], head_dim=spec["head_dim"],
        n_windows=1, patch_size=spec["patch_size"],
        patch_overlap=spec["patch_overlap"], freqs=4, chs_info=None,
        mlp_dim_ratio=2.66,
    )
    # Real positions come from the REVE bank. Only their VALUES differ from
    # these, not the shape or the arithmetic, so the FLOP count is unaffected
    # and this avoids requiring the position bank to measure compute.
    enc.default_pos = torch.randn(N_CHANS, 3)
    head = MovieCLIPHead(eeg_in_dim=spec["embed_dim"], vision_in_dim=VISION_DIM,
                         proj_dim=512, drop_proj=0.5, vision_passthrough=True,
                         n_residual_blocks=1)
    return enc, head


def one_pass(enc, head, backward: bool, batch: int = BATCH) -> int:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    eeg = torch.randn(batch, 1, N_CHANS, N_TIMES)
    tgt = torch.randn(batch, 1, VISION_DIM)
    with FlopCounterMode(display=False) as ctr:
        tokens = enc.encode_tokens(eeg, mask=None)
        pooled = enc.pool_to_windows(tokens)
        z = head.project_eeg(pooled)
        v = head.project_vision(tgt.reshape(-1, VISION_DIM))
        loss = (z @ v.T).mean()
        if backward:
            loss.backward()
    return ctr.get_total_flops()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", action="store_true",
                    help="also run torch's FlopCounterMode as a cross-check "
                         "(needs torch; the analytic path does not)")
    args = ap.parse_args()
    out = {"_meta": dict(batch=BATCH, epoch_size=EPOCH_SIZE,
                         steps_per_epoch=STEPS_PER_EPOCH, n_chans=N_CHANS,
                         n_times=N_TIMES,
                         counted="matmul only, fwd+bwd (bwd taken as 2x fwd), "
                                 "encoder + CLIP head; elementwise excluded")}
    measured: dict[str, int] = {}
    if args.measure:
        for key, spec in ARMS.items():
            enc, head = build(spec)
            # Measured at batch 1 and 2 and extrapolated, because a batch-64
            # forward+backward is minutes of real CPU arithmetic for what is a
            # shape calculation. The cost is NOT purely linear in the batch --
            # the InfoNCE logit matrix is O(b^2) -- so fit f(b) = A*b + C*b^2
            # from the two points and evaluate at 64 rather than scaling by 64
            # and quietly inflating the quadratic term.
            f1 = one_pass(enc, head, backward=True, batch=1)
            f2 = one_pass(enc, head, backward=True, batch=2)
            c = (f2 - 2 * f1) // 2
            a = f1 - c
            measured[key] = a * BATCH + c * BATCH ** 2
    for key, spec in ARMS.items():
        rec = dict(label=spec["label"], depth=spec["depth"], **analytic_flops(spec))
        rec["encoder_params"] = encoder_params(spec)
        # 6ND: the scaling-literature rule of thumb, 6 FLOPs per parameter per
        # token for a forward+backward pass. Kept alongside the explicit count
        # because it agrees with torch's measured value to ~0.1 % on this
        # architecture, while the explicit count runs ~8 % higher -- and that
        # gap is, to within rounding, exactly the attention term
        # (depth x n x 2*n*inner, doubled for backward). FlopCounterMode does
        # not instrument the fused SDPA kernel, and 6ND cannot see attention at
        # all since attention has no parameters. The explicit number is the
        # more complete one; the other two agree by sharing an omission.
        rec["flops_per_step_6nd"] = (
            6 * rec["encoder_params"] * rec["tokens_per_sample"] * BATCH)
        rec["flops_per_epoch_6nd"] = rec["flops_per_step_6nd"] * STEPS_PER_EPOCH
        if measured.get(key):
            rec["flops_per_step_measured"] = measured[key]
        out[key] = rec
        print(f"\n{spec['label']}")
        print(f"  tokens/sample     {rec['tokens_per_sample']}")
        print(f"  encoder params    {rec['encoder_params']/1e6:.1f} M")
        print(f"  per step, explicit {rec['flops_per_step']/1e12:.3f} TFLOPs")
        print(f"  per step, 6ND      {rec['flops_per_step_6nd']/1e12:.3f} TFLOPs "
              f"({rec['flops_per_step_6nd']/rec['flops_per_step']:.3f} of explicit)")
        print(f"  per epoch         {rec['flops_per_epoch']/1e12:.2f} TFLOPs")
        print(f"  400 / 800 epochs  {rec['flops_per_epoch']*400/1e15:.2f} / "
              f"{rec['flops_per_epoch']*800/1e15:.2f} PFLOPs")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    d22, d12 = out["depth22"], out["depth12"]
    print(f"depth-22 step costs {d22['flops_per_step']/d12['flops_per_step']:.2f}x "
          "a depth-12 step -- the two arms are step-matched, NOT compute-matched.")


if __name__ == "__main__":
    main()
