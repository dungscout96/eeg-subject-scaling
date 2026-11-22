"""Figure 1 (top): subject scaling on four readouts, three arms.

Rows are task (within: ThePresent, cross: DespicableMe), columns are readout
(ridge probe, fit-free retrieval). Every cell is read at its own held-out
retrieval optimum (the ``_epretr`` result files), test release only. Absolute
units; the dotted rule and band are the untrained-encoder floor per depth.

Usage:
    python analysis/plot_fig1_scaling.py      # -> figures/paper_fig1_scaling_law.{png,pdf}
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (            # noqa: E402
    ARM3, FIGS, INK, INK_2, MUTED, RAND, RAW, RULE, S_AXIS,
    e2v1, mean_r, s_of,
)

PRIOR_WORK_MAX_S = 50      # largest single EEG cohort in the meta-analysis (48)
# The shared palette's SURFACE is a faint cream, which reads as a tinted panel
# when the figure is dropped into a white page. Paper figures use the page's own
# background instead so no figure boundary is visible.
PAPER_BG = "#ffffff"

READOUTS = [
    ("within", "probe", "e04_tt_test", mean_r, "-"),
    ("within", "retrieval", "e04_retr_test", e2v1, (0, (4, 2))),
    ("cross", "probe", "xtask_tt_DMtest", mean_r, "-"),
    ("cross", "retrieval", "xtask_retr_DMtest", e2v1, (0, (4, 2))),
]
ARM_CURVE = {"warm": "warm", "scratch22": "scratch22", "scratch12": "scratch12"}


def floor_for(prefix: str, arm_key: str, reader):
    """Chance for this readout at THIS arm's depth. A single shared floor would
    misplace depth-12 by ~0.008 on the cross-task probe."""
    return reader(RAW / RAND[(prefix, ARM3[arm_key]["rand"])])


def _axes(ax, xlab, ylab, title):
    ax.set_facecolor(PAPER_BG)
    ax.grid(True, which="major", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_color(INK_2)
    ax.set_xlabel(xlab, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def _s_axis(ax):
    ax.set_xscale("log"); ax.set_xticks(S_AXIS)
    ax.set_xticklabels([str(s) for s in S_AXIS], rotation=45, ha="right")
    ax.minorticks_off()


# ---------------------------------------------------------------- figure 1
PANEL_GRID = [                     # (row=task, col=readout)
    [("within", "probe", "e04_tt_test", "mean(12) Pearson r"),
     ("within", "retrieval", "e04_retr_test", "time-pool e\u2192v@1")],
    [("cross", "probe", "xtask_tt_DMtest", "mean(12) Pearson r"),
     ("cross", "retrieval", "xtask_retr_DMtest", "time-pool e\u2192v@1")],
]


def fig1_scaling_law(arms=("warm", "scratch22", "scratch12")):
    """Subject scaling on four readouts: rows = task, columns = readout.

    2x2, all three arms in every panel, absolute units. Three lines per panel
    needs no ghosting or linestyle encoding, so colour alone carries arm
    identity and the shapes stay directly comparable.

    EACH CELL IS READ AT ITS OWN HELD-OUT OPTIMUM, selected on that draw's
    293-subject cohort complement by held-out retrieval (the `_epretr`
    artifacts). This is the paper's protocol, not a fixed epoch; reading these
    curves as fixed-epoch numbers is what made an earlier draft report a
    random-init plateau that does not exist.

    Each panel keeps its OWN y-scale. Sharing a scale down a column would be
    tempting -- it would make within- vs cross-task directly comparable -- but
    cross-task retrieval spans 0.005 against within-task's 0.07, so a shared
    axis would flatten the panel whose whole point is that it is flat, and the
    reader would lose the ability to see it hugging its floor.

    The chance band is what makes absolute units interpretable, and it is drawn
    per depth: the two architectures' floors differ by ~0.008 on the cross-task
    probe, which is larger than the entire range of the cross-task retrieval
    panel. Each arm's endpoint is labelled against ITS OWN floor for the same
    reason -- one shared secondary axis can only be exact for one depth.

    TICK LABELS ON THE TOP ROW ARE SUPPRESSED. Both rows share one S axis, and
    left on, the top row's 45-degree labels reach into the lower axes and are
    painted over by their background -- which truncates "1000" to "000" rather
    than looking broken.

    A third column used to carry held-out retrieval against epoch at S>=701, as
    evidence that the random-init arms had not converged inside the shared
    budget. It was dropped as redundant: the optimum-location figure says the
    same thing more directly and across the whole ladder rather than at one
    budget -- the random-init argmaxes sit ON their ceiling and the pretrained
    ones do not. Two panels making one argument is one panel too many.
    """
    fig = plt.figure(figsize=(11.2, 4.3), facecolor=PAPER_BG)
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.20)
    axes = [[fig.add_subplot(gs[r, c]) for c in (0, 1)] for r in (0, 1)]
    for r, row in enumerate(PANEL_GRID):
        for c, (task, rd_lbl, prefix, ylab) in enumerate(row):
            ax = axes[r][c]
            reader = mean_r if rd_lbl == "probe" else e2v1
            ax.axvspan(min(S_AXIS) * 0.82, PRIOR_WORK_MAX_S * 1.35,
                       color=MUTED, alpha=0.10, linewidth=0, zorder=0)
            tops, ends = [], []
            for arm in arms:
                spec = ARM3[arm]
                sel = json.loads((RAW / spec["retr"]).read_text())
                xs, ys = [], []
                for S in S_AXIS:
                    v = [reader(RAW / f"{prefix}_{cell}_epretr.json")
                         for cell in sel if s_of(cell) == S]
                    v = [x for x in v if x is not None]
                    if v:
                        xs.append(S); ys.append(st.mean(v))
                tops += ys
                ax.plot(xs, ys, color=spec["color"], lw=2.2, marker=spec["mark"],
                        ms=6, markerfacecolor=spec["color"],
                        markeredgecolor=spec["color"], zorder=3)
                # Each arm's terminal value against ITS OWN floor. The secondary
                # axis can only be exact for one depth; on the cross-task probe
                # the two floors differ by 9 %, enough that reading depth-12 off
                # the depth-22 axis overstates it (1.99x against a true 1.82x).
                own = floor_for(prefix, arm, reader)
                if xs:
                    ends.append((ys[-1], f"{ys[-1]/own:.2f}×", spec["color"]))
            floors = [reader(RAW / RAND[(prefix, d)]) for d in ("e04", "e03")]
            floors = [f for f in floors if f is not None]
            lo_c, hi_c = min(floors), max(floors)
            ax.axhspan(lo_c, hi_c, color=MUTED, alpha=0.22, linewidth=0, zorder=1)
            ax.axhline(hi_c, color=MUTED, lw=1.0, ls=(0, (1.5, 2)), zorder=1)
            ax.annotate("untrained encoder", xy=(0.985, hi_c),
                        xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", ha="right", color=MUTED,
                        fontsize=8)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, lo_c - 0.07 * (hi - lo)), hi)
            # No secondary xChance axis: it can be exact for only one depth, and
            # the per-arm endpoint labels below are exact for all three. Two
            # scales for one quantity, one of them approximate, is a way to be
            # misread rather than a way to be clear.
            # De-collide: two arms can finish within 4 % of each other, so
            # nudge labels apart in axes-fraction space before drawing.
            lo2, hi2 = ax.get_ylim()
            fr = sorted(((v - lo2) / (hi2 - lo2), lab, col) for v, lab, col in ends)
            placed = []
            for f_, lab, col in fr:
                if placed and f_ - placed[-1] < 0.062:
                    f_ = placed[-1] + 0.062
                placed.append(f_)
                ax.annotate(lab, xy=(1.012, f_), xycoords="axes fraction",
                            va="center", ha="left", color=col, fontsize=8.5,
                            fontweight="bold", annotation_clip=False)
            _axes(ax,
                  "S \u2014 training subjects (log scale)" if r == 1 else "",
                  ylab,
                  f"{rd_lbl.capitalize()}" if r == 0 else "")
            _s_axis(ax)
            # Both rows share one S axis, so only the bottom row is labelled.
            # Left on, the top row's 45-degree labels reach down into the lower
            # axes and are painted over by its background, which silently
            # truncates them to "000" and "400" rather than looking wrong.
            if r == 0:
                ax.set_xticklabels([])
            if c == 0:
                ax.annotate(f"{task}-task", xy=(0.035, 0.95),
                            xycoords="axes fraction", va="top", color=INK,
                            fontsize=10, fontweight="bold")
            if max(tops) - hi_c < 0.05:
                ax.annotate(f"whole panel spans {max(tops)-hi_c:.3f} above chance",
                            xy=(0.035, 0.83), xycoords="axes fraction", va="top",
                            color=MUTED, fontsize=8)
    # Sits inside the shaded band, below the "within-task" label rather than
    # beside it -- at this panel height the two collide if both hug the top.
    axes[0][0].annotate("range of\nprior work", xy=(0.035, 0.78),
                        xycoords="axes fraction", va="top", color=MUTED,
                        fontsize=8)

    handles = [Line2D([], [], color=ARM3[a]["color"], lw=2.2, marker=ARM3[a]["mark"],
                      ms=6, markerfacecolor=ARM3[a]["color"],
                      markeredgecolor=ARM3[a]["color"], label=ARM3[a]["label"])
               for a in arms]
    handles.append(Line2D([], [], color=MUTED, lw=1.0, ls=(0, (1.5, 2)),
                          label="untrained encoder (by depth)"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=8.5, bbox_to_anchor=(0.5, 0.004))
    for t_ in leg.get_texts():
        t_.set_color(INK_2)
    # The bottom strip has to hold three stacked things: 45-degree tick labels,
    # the x-axis label, and the legend. At this figure height 0.135 is not
    # enough for all three and the legend lands on top of the axis label, so
    # the margin is set from what has to fit rather than left at the default.
    fig.subplots_adjust(left=0.052, right=0.985, top=0.965, bottom=0.215)
    return fig


# ---------------------------------------------------------------- figure 2


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig = fig1_scaling_law()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"paper_fig1_scaling_law.{ext}", dpi=200,
                    facecolor=PAPER_BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/paper_fig1_scaling_law.png / .pdf")


if __name__ == "__main__":
    main()
