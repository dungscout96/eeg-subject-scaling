"""The method schematic, drawn in matplotlib rather than TikZ.

WHY NOT TIKZ. The schematic and the results figures were drifting apart: the
TikZ version carried its own palette and its own type, so a reader learned one
visual vocabulary on the method page and a different one on the results page.
Drawing it here lets it import the SAME tokens the results figures use, from
the same module, so the two cannot drift again.

WHAT COLOUR MEANS. Pipeline components are one neutral warm grey; the three
arms carry the exact hues the scaling figure gives them. So colour is arm
identity and nothing else, and this figure teaches that figure's legend. An
earlier draft gave every component type its own fill -- movie, EEG, objective
-- which spent the whole colour budget on distinctions the labels already make.

ROOM IS THE POINT. Boxes are sized from their text plus a fixed pad rather
than the other way round. A previous version fixed the box height to save page
space and let the type fill it edge to edge, which is what made it look
cramped: at print size the two lines of text were taller than the box holding
them. If this figure has to shrink again, cut a row -- do not shrink the pad.

The figure is drawn at 2x its printed size (11in wide for a 5.5in text block),
so point sizes here are twice the point sizes on the page.

Usage:
    python analysis/plot_method_figure.py     # -> figures/paper_method.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BLUE, FIGS, ORANGE  # noqa: E402

PINK = "#a03060"           # random-init depth 12, as in the scaling figure
PAPER_BG = "#ffffff"

# Tinted fill / saturated edge / dark text, one triple per role. The arm tints
# are washes of the results palette, so the hue survives at low saturation.
TINT = {BLUE: "#e4edfb", ORANGE: "#fce9e1", PINK: "#f6e4ea"}

# The supporting cast is grouped by ROLE, not left flat grey: data you bring in,
# the vision side, the EEG side, and what the model is scored against. Hues are
# chosen to sit well away from the three arm colours -- blue, orange and
# magenta are spoken for, so the roles take sand, violet, teal and gold. Every
# one is a low-saturation wash, so the arms stay the most saturated thing on
# the page and colour still reads as arm identity first.
# The first draft put inputs on sand and the objective on gold, which read as
# one family at print size and wasted the distinction. Inputs are now a muted
# parchment and the objective a clearly warmer amber.
ROLE = {
    "input":  ("#efeae0", "#bfb29b", "#6d6050"),   # the two stimulus inputs
    "vision": ("#ece7f9", "#a99bd8", "#524394"),   # V-JEPA-2 and its projection
    "eeg":    ("#e0efeb", "#8bbfb0", "#27705c"),   # pooling and the EEG projection
    "score":  ("#fdeec7", "#d8a93c", "#795608"),   # loss and the two readouts
}
GROUP_EDGE = "#c6bba6"     # the dashed encoder container
NOTE_INK = "#8b7f6c"
WIRE = "#8a8478"

SCALE = 2.0                # figure is drawn at twice its printed size
TITLE, SUB, NOTE = 6.5 * SCALE, 5.0 * SCALE, 5.0 * SCALE
PAD_T, PAD_S = 0.150, 0.135    # text offsets from a two-line box's centre

# --- layout, in centimetres of the printed figure ---------------------------
# Named rows, because the wiring has to land on the same lines as the boxes and
# this figure has been retuned for height more than once.
X0, X1 = -0.05, 14.00
Y0, Y1 = -0.10, 5.00
H2, H1 = 0.62, 0.50        # two-line and one-line box heights

VIS_Y = 4.24               # movie -> V-JEPA-2 -> vision projection
ARM_YS = (2.80, 2.15, 1.50)
TAIL_Y = ARM_YS[1]         # EEG in, pooling / projection / loss out

# The two projections are the same operation on opposite legs of one loss, so
# they are drawn in one column -- vision above EEG -- with the objective to
# their right. Widths below are the measured label widths plus the pad; the
# column x is shared by both, which is what makes the symmetry visible.
PROJ_X, PROJ_W = 10.90, 2.50
POOL_X, POOL_W = 8.625, 1.65
LOSS_X, LOSS_W = 13.125, 1.55
BOX_L, BOX_R = 2.50, 7.30  # the dashed encoder group
BOX_T, BOX_B = 3.74, 1.05
TAP_Y = 0.86               # elbow for the feature-encoding tap, below the group
READ_Y = 0.34

# The two inputs are shown rather than named: one frame of the stimulus and the
# EEG recorded against it. Both come from the SAME 2 s window of the test split
# (t = 60 s, The Present), which is the point -- they are what one training
# example actually is. Both are shipped under figures/assets/: frame_t60.png is
# the movie frame at t = 61 s, and eeg_t60.npz holds six channels of one test
# recording's preprocessed EEG for that window.
ASSETS = Path(__file__).resolve().parent.parent / "figures" / "assets"
T_ASSET = 60.0             # the window both assets are cut from, seconds
FRAME_W, EEG_W = 1.68, 1.94


def box(ax, x, y, w, title, sub=None, *, role="input", edge=None, face=None,
        ink=None, lw=1.0, h=None):
    """One rounded block. Returns (left, right, bottom, top) for wiring.

    Pass `role` for a pipeline component, or face/edge/ink explicitly for the
    arms, whose colours come from the results figure rather than from here."""
    r_face, r_edge, r_ink = ROLE[role]
    face, edge, ink = face or r_face, edge or r_edge, ink or r_ink
    h = h or (H2 if sub else H1)
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 + 0.10, y - h / 2 + 0.10), w - 0.20, h - 0.20,
        boxstyle="round,pad=0.10,rounding_size=0.13",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))
    if sub:
        ax.text(x, y + PAD_T, title, ha="center", va="center", zorder=4,
                fontsize=TITLE, color=ink, fontweight="bold")
        ax.text(x, y - PAD_S, sub, ha="center", va="center", zorder=4,
                fontsize=SUB, color=ink)
    else:
        ax.text(x, y, title, ha="center", va="center", zorder=4,
                fontsize=TITLE, color=ink, fontweight="bold")
    return x - w / 2, x + w / 2, y - h / 2, y + h / 2


def arrow(ax, x1, y1, x2, y2, color=WIRE, dashed=False, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, zorder=2, shrinkA=0, shrinkB=0,
        linestyle=(0, (2.6, 2.0)) if dashed else "solid"))


def line(ax, xs, ys, color=WIRE, lw=1.1, dashed=False):
    ax.add_line(Line2D(xs, ys, color=color, linewidth=lw, zorder=2,
                       linestyle=(0, (2.6, 2.0)) if dashed else "solid"))


def _card(ax, x, y, w, h, label):
    """Rounded frame plus a label above it, shared by the two input panels."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.0,
        edgecolor=ROLE["input"][1], facecolor="white", zorder=5))
    ax.text(x, y + h / 2 + 0.17, label, ha="center", va="center",
            fontsize=SUB, color=ROLE["input"][2], fontweight="bold")
    return x - w / 2, x + w / 2, y - h / 2, y + h / 2


def thumbnail(ax, x, y, w, label):
    """One frame of the stimulus, at its native 16:9."""
    img = mpimg.imread(ASSETS / "frame_t60.png")
    h = w * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
              zorder=6, interpolation="bilinear", aspect="auto")
    return _card(ax, x, y, w, h, label)


def eeg_panel(ax, x, y, w, label):
    """Six channels of the matching window, spread anterior to posterior.

    Drawn from the recording rather than simulated: this box is labelled EEG,
    so it should be EEG. Channels are picked by scalp position, not by index,
    so the six are actually spread over the montage."""
    d = np.load(ASSETS / "eeg_t60.npz")
    tr = d["traces"].astype(float)
    h = 0.86
    sub = ax.inset_axes((x - w / 2, y - h / 2, w, h), transform=ax.transData,
                        zorder=6)
    step = 1.35 * np.median(np.ptp(tr, axis=1))
    for i, ch in enumerate(tr):
        sub.plot(ch - i * step, lw=0.7, color=ROLE["eeg"][2], alpha=0.9)
    sub.set_xlim(0, tr.shape[1]); sub.axis("off")
    sub.patch.set_alpha(0)
    return _card(ax, x, y, w, h, label)


def method_figure():
    w_units, h_units = X1 - X0, Y1 - Y0
    fig, ax = plt.subplots(figsize=(11.0, h_units * 11.0 / w_units),
                           facecolor=PAPER_BG)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor(PAPER_BG)

    # --- vision band: frozen end to end ----------------------------------
    mov = thumbnail(ax, 1.05, VIS_Y, FRAME_W, "Movie frame")
    vje = box(ax, 4.90, VIS_Y, 4.8, "V-JEPA-2 (frozen)", "1408-d targets",
              role="vision")
    vto = box(ax, PROJ_X, VIS_Y, PROJ_W, "Vision projection", "linear, 512-d",
              role="vision")

    # --- the encoder group -------------------------------------------------
    # A dashed container rather than three parallel pipelines: the arms are
    # identical in every respect but initialisation and depth, so they are one
    # slot with three occupants.
    ax.add_patch(FancyBboxPatch(
        (BOX_L + 0.10, BOX_B + 0.10), BOX_R - BOX_L - 0.20, BOX_T - BOX_B - 0.20,
        boxstyle="round,pad=0.10,rounding_size=0.15", linewidth=1.0,
        edgecolor=GROUP_EDGE, facecolor="none", linestyle=(0, (3.5, 2.6)),
        zorder=1))
    ax.text(4.90, 3.50, "EEG encoder — one arm per run", ha="center",
            va="center", fontsize=SUB, color=NOTE_INK)

    arms = list(zip(ARM_YS, (BLUE, ORANGE, PINK),
                    ("Pretrained, depth 22", "Random-init, depth 22",
                     "Random-init, depth 12"),
                    ("REVE weights", "random weights", "random weights")))
    for y, col, lab, sub in arms:
        box(ax, 4.90, y, 4.2, lab, sub, edge=col, face=TINT[col], ink=col,
            lw=1.2)

    # --- EEG input: the same window, as recorded --------------------------
    eeg = eeg_panel(ax, 1.05, TAIL_Y, EEG_W, "EEG, 129 ch")
    ax.text(1.05, eeg[2] - 0.16, "$S$ subjects, 10–1863", ha="center",
            va="center", fontsize=NOTE, color=NOTE_INK)

    # --- shared tail ------------------------------------------------------
    pool = box(ax, POOL_X, TAIL_Y, POOL_W, "Mean pool", "512-d vector",
               role="eeg")
    head = box(ax, PROJ_X, TAIL_Y, PROJ_W, "EEG projection", "MLP, 512-d",
               role="eeg")
    loss = box(ax, LOSS_X, TAIL_Y, LOSS_W, "CLIP loss", "soft-target",
               role="score")

    # --- readouts ---------------------------------------------------------
    feat = box(ax, 4.60, READ_Y, 5.2, "Feature encoding",
               "ridge", role="score")
    retr = box(ax, PROJ_X, READ_Y, 4.4, "Retrieval",
               "zero-shot", role="score")

    # --- wiring -----------------------------------------------------------
    arrow(ax, mov[1], VIS_Y, vje[0], VIS_Y)
    arrow(ax, vje[1], VIS_Y, vto[0], VIS_Y)
    # EEG fans out to all three arms inside the group; they converge on one
    # pooling step outside it.
    line(ax, [eeg[1], 2.20], [TAIL_Y, TAIL_Y])
    line(ax, [2.20, 2.20], [ARM_YS[-1], ARM_YS[0]])
    line(ax, [7.60, 7.60], [ARM_YS[-1], ARM_YS[0]])
    for y, _c, _l, _s in arms:
        arrow(ax, 2.20, y, 2.80, y)
        line(ax, [7.00, 7.60], [y, y])
    arrow(ax, 7.60, TAIL_Y, pool[0], TAIL_Y)
    arrow(ax, pool[1], TAIL_Y, head[0], TAIL_Y)
    arrow(ax, head[1], TAIL_Y, loss[0], TAIL_Y)
    # The vision projection joins the objective from above.
    line(ax, [vto[1], LOSS_X], [VIS_Y, VIS_Y])
    arrow(ax, LOSS_X, VIS_Y, LOSS_X, loss[3])
    # Dashed taps: where each readout reads the model. Feature encoding taps the
    # pooled vector, so its elbow runs BELOW the group rather than through it.
    line(ax, [POOL_X, POOL_X], [pool[2], TAP_Y], dashed=True)
    line(ax, [POOL_X, 4.60], [TAP_Y, TAP_Y], dashed=True)
    arrow(ax, 4.60, TAP_Y, 4.60, feat[3], dashed=True)
    arrow(ax, PROJ_X, head[2], PROJ_X, retr[3], dashed=True)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig = method_figure()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"paper_method.{ext}", dpi=200, facecolor=PAPER_BG,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote figures/paper_method.pdf / .png")


if __name__ == "__main__":
    main()
