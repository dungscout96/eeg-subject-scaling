"""Shared constants, palette and loaders for the paper figures and numbers.

Arm keys predate the paper's vocabulary: ``warm`` is the PRETRAINED arm,
``scratch22`` / ``scratch12`` are the RANDOM-INIT arms at depth 22 / 12. The
display labels live in ARM3, the single place to change them.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "results"
FIGS = ROOT / "figures"

# Palette shared by every paper figure.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"

WARM_SEL = RAW / "addval_selection_probe.json"
SCRATCH_SEL = RAW / "fromscratch_optimum_selection.json"
ARM = {"warm": dict(color=BLUE, mark="o", label="pretrained (REVE)"),
       "scratch": dict(color=ORANGE, mark="s", label="random-init")}
S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]

# --- is there an optimum to locate at all? -----------------------------------
# HEADROOM is the fraction of the peak still unreached at the FIRST saved
# checkpoint: (peak - v[first]) / peak. Near zero means the run was essentially
# over before epoch 25, so the curve is a plateau with noise on it and its
# argmax is placed by that noise rather than by convergence. Measured on the
# curves: <=0.11 at S<=50 in both arms, >=0.19 from S=100 up -- a clean break,
# so the 0.15 cut is reading a gap in the data, not imposing a round number.
#
# Two other criteria were tried and rejected. Prominence-over-noise
# ((peak-median)/sigma, sigma from successive differences) is contaminated by
# the trend's own curvature -- it called S=50 resolved and S=400 not, which is
# backwards. Draw spread of the selected epoch is the right idea but too noisy
# at n=3 to threshold on (warm scatters 0.64 of the grid at S=50 and 0.29 at
# S=1400, which does not separate).
RESOLVE_MIN_HEADROOM = 0.15
COMPOSITE_MIN_S = 701
WARM_CURVE = RAW / "addval_epoch_curve.json"
SCRATCH_CURVES = (RAW / "fromscratch_epoch_curve.json",
                  RAW / "fromscratch800_epoch_curve.json")


def s_of(c): return int(c.split("_")[1][1:])
def fixed_slug(c): return c.replace("_ep800", "")


def mean_r(p):
    if not p.exists(): return None
    f = json.loads(p.read_text())["features"]
    return sum(v["pearson_r"] for v in f.values()) / len(f)


def e2v1(p):
    if not p.exists(): return None
    return json.loads(p.read_text())["levels"]["time"]["e2v_top_k"]["1"]


def cells(sel):
    out = {}
    for c in json.loads(sel.read_text()):
        out.setdefault(s_of(c), []).append(c)
    return out


D12_CURVE = RAW / "d12av_epoch_curve.json"


def load_curves(arm: str) -> dict[str, dict]:
    """Per-epoch curves for one arm, composite-routed for depth-22 random-init.

    Above S=701 that arm was retrained at 800 epochs because the original 375
    budget was binding, so BOTH slugs exist there; the plain one sorts first, so
    a naive merge silently keeps the truncated cell. The depth-12 arm needs no
    routing -- its high-S cells were written straight to _ep800 slugs, so its
    curve file already holds exactly one cell per draw.

    Accepts the ARM3 keys as well as the two legacy names, so callers can pass
    whichever vocabulary they already use.
    """
    if arm in ("warm",):
        return json.loads(WARM_CURVE.read_text())["cells"]
    if arm in ("scratch12", "d12"):
        return json.loads(D12_CURVE.read_text())["cells"]
    orig, ep800 = (json.loads(p.read_text())["cells"] for p in SCRATCH_CURVES)
    out = {c: v for c, v in orig.items() if s_of(c) < COMPOSITE_MIN_S}
    out.update({c: v for c, v in ep800.items() if s_of(c) >= COMPOSITE_MIN_S})
    return out


def headroom(curve: dict) -> float:
    """Fraction of the peak still unreached at the first saved checkpoint."""
    epochs = sorted(int(e) for e in curve)
    vals = [curve[str(e)]["probe_mean_r"] for e in epochs]
    peak = max(vals)
    return (peak - vals[0]) / peak if peak > 0 else 0.0


def resolved_by_s(arm: str) -> dict[int, bool]:
    """{S: is this cell's optimum locatable at all}, averaged over its draws."""
    curves = load_curves(arm)
    by: dict[int, list[float]] = {}
    for cell, curve in curves.items():
        by.setdefault(s_of(cell), []).append(headroom(curve))
    return {s: st.mean(h) >= RESOLVE_MIN_HEADROOM for s, h in by.items()}


def split_runs(xs, ys, ok):
    """-> (solid_segments, dashed_segments) of (xs, ys).

    A segment joining a resolved point to an unresolved one is drawn dashed:
    the claim "these two epochs differ" is only as good as its weaker end.
    """
    solid, dashed = [], []
    for i in range(len(xs) - 1):
        seg = ([xs[i], xs[i + 1]], [ys[i], ys[i + 1]])
        (solid if (ok[i] and ok[i + 1]) else dashed).append(seg)
    return solid, dashed


PINK = "#a03060"
ARM3 = {
    "warm": dict(color=BLUE, mark="o", label="pretrained (REVE), depth 22",
                 probe="addval_selection_probe.json",
                 retr="addval_selection_retr.json", rand="e04"),
    "scratch22": dict(color=ORANGE, mark="s", label="random-init, depth 22",
                      probe="fromscratch_optimum_selection.json",
                      retr="fromscratch_selection_retr.json", rand="e04"),
    "scratch12": dict(color=PINK, mark="^", label="random-init, depth 12",
                      probe="d12av_selection_probe.json",
                      retr="d12av_selection_retr.json", rand="e03"),
}
# Chance depends on the architecture, so each depth has its own random-encoder
# artifact. They are close on three readouts and NOT on the cross-task probe
# (0.090 depth-22 vs 0.099 depth-12); one line there would put the depth-12
# arm's floor 0.008 in the wrong place, in a panel whose whole range is 0.005.
RAND = {
    ("e04_tt_test", "e04"): "e04_tt_test_random.json",
    ("e04_tt_test", "e03"): "e03_tt_test_random.json",
    ("e04_retr_test", "e04"): "e04_retr_test_random.json",
    ("e04_retr_test", "e03"): "e03_retr_test_random.json",
    ("xtask_tt_DMtest", "e04"): "xtask_tt_DMtest_random_e04.json",
    ("xtask_tt_DMtest", "e03"): "xtask_tt_DMtest_random.json",
    ("xtask_retr_DMtest", "e04"): "xtask_retr_DMtest_random_e04.json",
    ("xtask_retr_DMtest", "e03"): "xtask_retr_DMtest_random.json",
}
