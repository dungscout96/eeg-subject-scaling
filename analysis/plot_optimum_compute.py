"""Where each arm's optimum sits, per S, in training compute.

Figure 1 (bottom). A median epoch per S answers "does the optimum move with
S"; this figure also answers "how well-determined is that number" -- for all
three arms -- and it needs the whole training axis to do it.

The pretrained arm is in the figure because the comparison is the finding: its
optimum barely moves across the ladder and sits inside its original schedule,
while both random-init arms' optima travel the length of the ladder and pin
against the ceiling at the top. Reading that off two panels of random-init
arms alone requires the reader to remember a number from elsewhere.

WHY COMPUTE AND NOT EPOCHS. `data.epoch_size=703` on every cell of every arm,
so `len(dataset)` -- and therefore steps/epoch -- is 11 regardless of S. An
"epoch" here is a fixed 703 items, NOT a pass over the cohort: at S=10 it is
~70 passes, at S=1863 it is 0.38 of one. Labelling the axis "epoch" invites the
usual reading and is wrong above S=701. Gradient steps would be honest but are
not comparable ACROSS the two panels, because the arms are step-matched by
design and a depth-22 step costs ~4x a depth-12 one. Compute is the axis on
which both comparisons are legitimate, so it is the axis. The conversion is
measured, not assumed -- see `count_train_flops.py` and
`results/train_flops.json`.

Three things are drawn per cell, because an argmax alone is not a measurement:

  * the ARGMAX of each draw (three markers per S),
  * the NEAR-PEAK BAND: every checkpoint scoring within one noise sigma of
    that draw's own peak, shaded by how many of the three draws agree. A cell
    whose band is one checkpoint wide has a located optimum; a cell whose band
    spans a third of the run has a plateau, and its argmax is a coin flip
    inside that plateau, and
  * the MEDIAN across draws, connecting the columns.

TWO FIGURES, TWO AXES. Compute and passes-over-the-cohort are different
questions and cannot share one vertical scale: compute is a fixed multiple of
the epoch (the same multiple in every column), while passes is
703 x epoch / S, which depends on the column as well as the checkpoint. So
each gets its own figure over the identical marks:

  `optimum_compute_*`  y = cumulative training FLOPs. The axis on which the
                       panels are comparable at all -- the arms are
                       step-matched by design, and a depth-22 step costs ~4x a
                       depth-12 one, so equal steps is NOT equal compute.
  `optimum_passes_*`   y = how many times the cell has seen its own cohort.
                       Log-scaled, because the fixed 703-item epoch means this
                       spans ~300x across the S axis. This is the axis on
                       which the arms ARE matched, and the one the
                       data-constrained scaling literature reports repetition
                       effects in.

The arms do not share a budget. The random-init arms step from 375 epochs
below S=701 to 775 at and above it, because only their high-S cells were
retrained at the doubled budget; the pretrained arm stayed at 375 throughout,
since its optimum never approached that ceiling. Each panel draws its own
budget as a ceiling, and any argmax sitting ON it is a lower bound rather than
an optimum -- those markers are drawn hollow.

X is categorical, not log: the S values are compared to each other here, not
fitted against, and equal columns give each cell's band the same width.

Usage:
    python analysis/plot_optimum_compute.py   # -> figures/optimum_compute_e2v1_time.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "results"
FIGS = ROOT / "figures"

# Same tokens as the other paper figures (analysis/common.py), so an arm is one
# colour throughout.
BLUE, ORANGE, PINK = "#2a78d6", "#eb6834", "#a03060"
INK, INK_2, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
# This figure goes into the paper, where SURFACE's faint cream reads as a
# tinted panel against the page. Paint on the page's own background instead so
# no figure boundary is visible -- same choice as plot_paper_figures.PAPER_BG.
BG = "#ffffff"

S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
COMPOSITE_MIN_S = 701          # where the 800-epoch retrain takes over
STEP = 25                      # checkpoint spacing, both budgets
EPOCH_SIZE = 703               # items per epoch, fixed on every cell
BATCH = 64
STEPS_PER_EPOCH = -(-EPOCH_SIZE // BATCH)   # ceil(703/64) = 11, every cell
MAX_EPOCH = 800                # top of the plotted range (budget is 775)

# For this pool S is also the recording count -- the 2156-pool arms are one
# recording per subject, verified from the run logs.
# So passes over the cohort is 703 * epoch / S with no conversion factor.

# The metrics the curve files carry at every checkpoint. probe_mean_r is the
# mean Pearson r over 12 visual features; the e2v1_* are top-1 EEG->video
# retrieval at three temporal granularities.
METRICS = {
    "probe_mean_r": "held-out probe (mean r over 12 features)",
    "e2v1_time": "time-level retrieval (top-1 EEG→video)",
    "e2v1_scene": "scene-level retrieval (top-1 EEG→video)",
    "e2v1_shot": "shot-level retrieval (top-1 EEG→video)",
}

# "As good as the peak" has to be defined in units of the metric's OWN noise,
# not as a fixed fraction of the peak, because the metrics are not equally
# noisy: measured on these curves, sigma is 1-2 % of peak for probe_mean_r and
# 3-7 % for e2v1_time. A fixed 2 % rule would call retrieval's peaks SHARPER
# than the probe's, when what it is really reading is that a 2 % window sits
# below retrieval's noise floor and admits only the noise spike itself.
#
# sigma is estimated from SECOND differences, |v[i-1] - 2v[i] + v[i+1]|, via a
# MAD scaled by 1.4826/sqrt(6). Second differences annihilate a local linear
# trend, so a steeply rising high-S curve does not inflate its own sigma and
# buy itself a wider band -- the trend-contamination objection that sank the
# first-difference estimate as a resolution criterion (see
# analysis/common.py, RESOLVE_MIN_HEADROOM). MAD rather than the mean so one bad checkpoint
# does not set the width.
K_SIGMA = 1.0

ARMS = {
    # The pretrained arm has no doubled-budget retrain -- its optimum sat inside
    # the original schedule, which is the finding -- so it is one curve file and
    # a flat 375-epoch ceiling in every column. Drawn first so the panel order
    # matches the arm order in the scaling figure.
    "warm": dict(
        color=BLUE, mark="o", label="pretrained (REVE), depth 22", flops="depth22",
        curves=(RAW / "addval_epoch_curve.json",),
    ),
    "scratch22": dict(
        color=ORANGE, mark="s", label="random-init, depth 22", flops="depth22",
        curves=(RAW / "fromscratch_epoch_curve.json",
                RAW / "fromscratch800_epoch_curve.json"),
    ),
    "scratch12": dict(
        color=PINK, mark="^", label="random-init, depth 12", flops="depth12",
        curves=(RAW / "d12av_epoch_curve.json",),
    ),
}


def s_of(cell: str) -> int:
    return int(cell.split("_")[1][1:])


def pflops_per_epoch(spec: dict) -> float:
    """PFLOPs of one epoch for this arm, from the measured constants on disk.

    Loaded lazily and only by the compute figure, so the passes figure keeps
    working on a machine that cannot import torch to regenerate them.
    """
    global _FLOPS
    if _FLOPS is None:
        p = RAW / "train_flops.json"
        if not p.exists():
            raise SystemExit(
                f"{p.name} is missing -- run src/count_train_flops.py to "
                "measure it, or pass --axis passes, which does not need it.")
        _FLOPS = json.loads(p.read_text())
    return _FLOPS[spec["flops"]]["flops_per_epoch"] / 1e15


_FLOPS = None


def load_curves(spec: dict) -> dict[str, dict]:
    """Per-epoch curves for one arm, composite-routed where both slugs exist.

    The depth-22 arm has BOTH a 375-epoch and a 775-epoch cell at S>=701, in
    two files; the plain slug sorts first, so a naive merge silently keeps the
    truncated one. The depth-12 arm wrote its high-S cells straight to _ep800
    slugs in a single file and needs no routing.
    """
    if len(spec["curves"]) == 1:
        return json.loads(spec["curves"][0].read_text())["cells"]
    orig, ep800 = (json.loads(p.read_text())["cells"] for p in spec["curves"])
    out = {c: v for c, v in orig.items() if s_of(c) < COMPOSITE_MIN_S}
    out.update({c: v for c, v in ep800.items() if s_of(c) >= COMPOSITE_MIN_S})
    return out


def noise_sigma(vals: list[float]) -> float:
    """Checkpoint-to-checkpoint noise, trend-free (see K_SIGMA)."""
    d2 = [abs(a - 2 * b + c) for a, b, c in zip(vals, vals[1:], vals[2:])]
    return 1.4826 * st.median(d2) / 6 ** 0.5 if d2 else 0.0


def cell_stats(curve: dict, metric: str) -> dict:
    epochs = sorted(int(e) for e in curve)
    vals = [curve[str(e)][metric] for e in epochs]
    peak = max(vals)
    sigma = noise_sigma(vals)
    return dict(
        epochs=epochs,
        argmax=epochs[vals.index(peak)],
        last=epochs[-1],
        peak=peak,
        sigma=sigma,
        near={e for e, v in zip(epochs, vals) if v >= peak - K_SIGMA * sigma},
    )


def by_s(spec: dict, metric: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for cell, curve in load_curves(spec).items():
        out.setdefault(s_of(cell), []).append(cell_stats(curve, metric))
    return {s: out[s] for s in S_AXIS if s in out}


# --- the two vertical axes -----------------------------------------------
# Each is a map (epoch, S, arm) -> y. Compute ignores S; passes ignores the
# arm. Everything else about the figure is identical, which is the point:
# the same marks, read against two different questions.
AXES = {
    "compute": dict(
        y=lambda e, S, spec: e * pflops_per_epoch(spec),
        label="training compute (PFLOPs, fwd+bwd)",
        scale="linear",
        blurb="the arms are step-matched, so equal compute is NOT equal steps",
    ),
    "passes": dict(
        y=lambda e, S, spec: e * EPOCH_SIZE / S,
        label="passes over the cell's own cohort",
        scale="log",
        blurb="a fixed 703-item epoch, so one epoch is ~70 passes at S=10 "
              "and 0.38 at S=1863",
    ),
    # Steps is the axis the EXPERIMENT was designed on -- the quantity held
    # equal across every cell and both arms. It ignores S and the arm alike,
    # so it is the epoch axis times 11 and nothing more; its value is that it
    # shows the design, and that both panels being identical in range is the
    # point rather than an artifact. Compute and passes are the two ways that
    # equality stops being an equality.
    "steps": dict(
        y=lambda e, S, spec: e * STEPS_PER_EPOCH,
        label="gradient steps (11 per epoch)",
        scale="linear",
        blurb="the quantity the design holds equal: 4400 steps below S=701, "
              "8800 at and above, in both arms",
    ),
}


def style_axes(ax, title: str, ylabel: bool, ylim, axis: dict) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.set_xticks(range(len(S_AXIS)))
    ax.set_xticklabels([str(s) for s in S_AXIS], rotation=45, ha="right")
    ax.set_xlim(-0.6, len(S_AXIS) - 0.4)
    ax.set_yscale(axis["scale"])
    ax.set_ylim(*ylim)
    if axis["scale"] == "log":
        ax.minorticks_off()
        ax.set_yticks([300, 1000, 3000, 10000, 30000])
        ax.set_yticklabels([f"{v:,}×" for v in (300, 1000, 3000, 10000, 30000)])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.set_xlabel("S — training subjects (= recordings)", color=INK_2,
                  fontsize=9.5)
    if ylabel:
        ax.set_ylabel(axis["label"], color=INK_2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def draw_arm(ax, spec: dict, stats: dict[int, list[dict]], axis: dict,
             ylim) -> None:
    """The same marks under either vertical map -- see AXES.

    Band rectangles are built from the checkpoint's own +/- half-interval
    mapped through `y`, not from a constant height, so they stay attached to
    the checkpoints they describe on a log axis as well as a linear one.
    """
    y = axis["y"]
    half = 0.40
    # Everything above a cell's budget was never trained. Drawn as one stepped
    # ceiling rather than per-column hatching, so it reads as the experimental
    # design it is -- and so the low-S columns are not swamped by texture that
    # carries no per-cell information. On the passes axis this staircase FALLS
    # with S even though the budget rises, which is the whole point of the
    # second figure.
    edge_x, edge_y = [], []
    for i, S in enumerate(S_AXIS):
        if S not in stats:
            continue
        top = y(stats[S][0]["last"] + STEP / 2, S, spec)
        edge_x += [i - 0.5, i + 0.5]
        edge_y += [top, top]
    ax.fill_between(edge_x, edge_y, ylim[1], color=MUTED, alpha=0.07,
                    linewidth=0, zorder=1)
    ax.plot(edge_x, edge_y, color=MUTED, lw=1.1, ls=(0, (4, 2.6)), zorder=1)
    for i, S in enumerate(S_AXIS):
        draws = stats.get(S, [])
        if not draws:
            continue
        ceiling = draws[0]["last"]
        for e in draws[0]["epochs"]:
            n = sum(e in d["near"] for d in draws)
            if n:
                lo, hi = y(e - STEP / 2, S, spec), y(e + STEP / 2, S, spec)
                ax.add_patch(Rectangle((i - half, lo), 2 * half, hi - lo,
                                       facecolor=spec["color"], linewidth=0,
                                       alpha=0.16 + 0.17 * n, zorder=2))
        # Jitter the three argmaxes across the column so ties stay countable.
        for j, d in enumerate(sorted(draws, key=lambda d: d["argmax"])):
            pinned = d["argmax"] == ceiling
            ax.plot(i - 0.16 + 0.16 * j, y(d["argmax"], S, spec),
                    marker=spec["mark"], ms=5.5, lw=0, zorder=4,
                    color=spec["color"],
                    markerfacecolor=BG if pinned else spec["color"],
                    markeredgecolor=spec["color"], markeredgewidth=1.3)
    xs = [i for i, S in enumerate(S_AXIS) if S in stats]
    med = [y(st.median([d["argmax"] for d in stats[S]]), S, spec)
           for S in S_AXIS if S in stats]
    ax.plot(xs, med, color=spec["color"], lw=1.8, alpha=0.85, zorder=3)


def y_limits(axis_name: str, axis: dict, metric: str):
    """One shared range across both panels, so the panels stay comparable.

    Derived from what is actually DRAWN -- band edges, argmaxes and the budget
    ceiling -- rather than from the full epoch grid. On the passes axis the two
    differ enormously: the grid's low end is the largest cohort's first
    checkpoint (~9 passes), but nothing near-peak is ever drawn below ~270, so
    a grid-derived limit leaves the bottom third of the panel empty.
    """
    ys = []
    for spec in ARMS.values():
        for S, draws in by_s(spec, metric).items():
            ys.append(axis["y"](draws[0]["last"] + STEP / 2, S, spec))
            for d in draws:
                for e in d["near"] | {d["argmax"]}:
                    ys += [axis["y"](e - STEP / 2, S, spec),
                           axis["y"](e + STEP / 2, S, spec)]
    lo, hi = min(ys), max(ys)
    if axis["scale"] == "log":
        return lo * 0.75, hi * 1.35
    return 0, hi * 1.04


def fig_distribution(metric: str, axis_name: str):
    axis = AXES[axis_name]
    ylim = y_limits(axis_name, axis, metric)
    # Wide and short: at \linewidth in a two-column-width NeurIPS body this
    # costs ~1.6in of page, which is what the paper's page budget allows. The
    # suptitle and blurb the standalone version carried are dropped -- in the
    # paper the caption says both, and repeating them wastes the same height
    # the aspect ratio is trying to buy back.
    fig, axes = plt.subplots(1, len(ARMS), figsize=(12.6, 3.0), facecolor=BG,
                             sharey=True)
    for ax, spec in zip(axes, ARMS.values()):
        draw_arm(ax, spec, by_s(spec, metric), axis, ylim)
        style_axes(ax, spec["label"], ax is axes[0], ylim, axis)
    grey = dict(color=MUTED, lw=0, ms=5.5)
    handles = [
        Line2D([], [], marker="o", markerfacecolor=MUTED, markeredgecolor=MUTED,
               label="draw argmax", **grey),
        Line2D([], [], marker="o", markerfacecolor=BG, markeredgewidth=1.3,
               markeredgecolor=MUTED,
               label="argmax at the last checkpoint (a bound, not an optimum)",
               **grey),
        Patch(facecolor=MUTED, alpha=0.55, linewidth=0,
              label="within 1σ of peak — all 3 draws"),
        Patch(facecolor=MUTED, alpha=0.2, linewidth=0,
              label="within 1σ of peak — 1 draw"),
        Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 2.6)),
               label="trained budget"),
    ]
    leg = fig.legend(handles=handles, frameon=False, fontsize=8.2, ncol=5,
                     loc="lower center", bbox_to_anchor=(0.5, -0.035),
                     handlelength=1.8, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return fig


def report(spec: dict, metric: str) -> None:
    stats = by_s(spec, metric)
    fpe = pflops_per_epoch(spec)
    print(f"\n{spec['label']}  ({metric}; band = within 1σ of each draw's peak)")
    print("    S | argmax epochs (3 draws) | median | PFLOPs | passes | "
          "band (ckpts) | pinned")
    for S in S_AXIS:
        draws = stats.get(S)
        if not draws:
            continue
        am = sorted(d["argmax"] for d in draws)
        med = st.median(am)
        widths = sorted(len(d["near"]) for d in draws)
        pinned = sum(d["argmax"] == d["last"] for d in draws)
        print(f" {S:>5} | {str(am):<23} | {med:>6.0f} | {med*fpe:>6.3f} | "
              f"{EPOCH_SIZE*med/S:>6.0f} | {str(widths):<12} | {pinned}/3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="e2v1_time", choices=list(METRICS),
                    help="which per-checkpoint score the argmax is taken over")
    ap.add_argument("--axis", default="compute",
                    choices=[*AXES, "both"],
                    help="vertical axis; 'both' writes one figure per axis")
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)
    # Both the metric and the axis are in the filename. Neither is recoverable
    # from the image once it is in a draft, and a figure whose y-axis is a
    # different question under the same name is exactly the confusion these
    # two axes exist to prevent.
    for axis_name in (list(AXES) if args.axis == "both" else [args.axis]):
        stem = f"optimum_{axis_name}_{args.metric}"
        fig = fig_distribution(args.metric, axis_name)
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{stem}.{ext}", dpi=200, facecolor=BG,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{stem}.png / .pdf")
    for spec in ARMS.values():
        report(spec, args.metric)


if __name__ == "__main__":
    main()
