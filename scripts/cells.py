"""The experimental design in one place: arms, cohort ladder, draws, budgets, seeds.

Every pipeline script reads this module, so the cell list cannot drift between
training, selection and evaluation.

Naming. Cell and result-file names keep the identifiers the runs were made
under, so the shipped results/ files and the analysis code line up with a
fresh run one-for-one:

    paper arm                       cell prefix   result-file key
    ------------------------------  ------------  ----------------------------
    pretrained (REVE), depth 22     e05           addval
    random-init, depth 22           e05fs         fromscratch / fromscratch800
    random-init, depth 12           e03           d12av

A cell is ``<prefix>_s<S>_a101_av[_ep800]_d<draw>``: S subjects, all 101 movie
anchors, the 2156-subject ("av") pool, an optional 800-epoch budget marker,
and the draw seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

S_LADDER = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
DRAWS = [11, 22, 33]
ANCHORS = 101
POOL = 2156               # recordings (= subjects) in the training pool
COMPLEMENT_OF = 1863      # select on the 293 subjects outside the S=1863 cohort
SAVE_EVERY = 25
DEFAULT_EPOCHS = 400      # 703-window epochs -> 11 steps -> 4400 steps
LONG_EPOCHS = 800         # 8800 steps
LONG_MIN_S = 701          # random-init arms are (re)trained at 800 epochs from here
SEED = 2026

# Cells whose seed-2026 run collapsed at initialisation (training loss flat at
# ln(64) = 4.16 from epoch 1) and were re-run with the next seed in the fixed
# sequence 2026 -> 7 -> 2025 (max three attempts). Detected by
# scripts/screen_collapse.py. Only the pretrained arm had collapses.
RERUN_SEED = {
    ("pretrained_d22", 1400, 11): 7,
    ("pretrained_d22", 701, 33): 7,
    ("pretrained_d22", 1000, 11): 7,
    ("pretrained_d22", 400, 11): 2025,   # collapsed at both 2026 and 7
}


@dataclass(frozen=True)
class Arm:
    name: str
    prefix: str
    train_config: str
    select_config: str
    test_configs: dict = field(default_factory=dict)
    pretrained: bool = False
    # Which budgets each S is trained at.
    budgets_low: tuple = (DEFAULT_EPOCHS,)    # S < LONG_MIN_S
    budgets_high: tuple = (DEFAULT_EPOCHS,)   # S >= LONG_MIN_S
    # Budget whose cells are REPORTED at S >= LONG_MIN_S.
    reported_high: int = DEFAULT_EPOCHS
    depth: int = 22


ARMS = {
    "pretrained_d22": Arm(
        name="pretrained_d22", prefix="e05",
        train_config="configs/train/depth22.yaml",
        select_config="configs/eval/select_TP_depth22.yaml",
        test_configs={"within": "configs/eval/test_TP_depth22.yaml",
                      "cross": "configs/eval/test_DM_depth22.yaml"},
        pretrained=True,
    ),
    # The 400-epoch cells at S >= 701 are trained too: comparing them with the
    # 800-epoch cells measures what the shorter budget cost (paper App. C.3).
    "random_d22": Arm(
        name="random_d22", prefix="e05fs",
        train_config="configs/train/depth22.yaml",
        select_config="configs/eval/select_TP_depth22.yaml",
        test_configs={"within": "configs/eval/test_TP_depth22.yaml",
                      "cross": "configs/eval/test_DM_depth22.yaml"},
        budgets_high=(DEFAULT_EPOCHS, LONG_EPOCHS), reported_high=LONG_EPOCHS,
    ),
    "random_d12": Arm(
        name="random_d12", prefix="e03",
        train_config="configs/train/depth12.yaml",
        select_config="configs/eval/select_TP_depth12.yaml",
        test_configs={"within": "configs/eval/test_TP_depth12.yaml",
                      "cross": "configs/eval/test_DM_depth12.yaml"},
        budgets_high=(LONG_EPOCHS,), reported_high=LONG_EPOCHS, depth=12,
    ),
}


def cell_name(arm: str, S: int, draw: int, epochs: int) -> str:
    marker = "" if epochs == DEFAULT_EPOCHS else f"_ep{epochs}"
    return f"{ARMS[arm].prefix}_s{S}_a{ANCHORS}_av{marker}_d{draw}"


def seed_for(arm: str, S: int, draw: int) -> int:
    return RERUN_SEED.get((arm, S, draw), SEED)


def training_cells(arm: str) -> list[tuple[str, int, int, int]]:
    """Every cell to train: [(cell, S, draw, epochs)]."""
    a = ARMS[arm]
    out = []
    for S in S_LADDER:
        for ep in (a.budgets_high if S >= LONG_MIN_S else a.budgets_low):
            for d in DRAWS:
                out.append((cell_name(arm, S, d, ep), S, d, ep))
    return out


def reported_cells(arm: str) -> list[tuple[str, int, int]]:
    """The 30 cells the paper reports for this arm: [(cell, S, draw)]."""
    a = ARMS[arm]
    return [(cell_name(arm, S, d, a.reported_high if S >= LONG_MIN_S else DEFAULT_EPOCHS), S, d)
            for S in S_LADDER for d in DRAWS]


def s_of(cell: str) -> int:
    return int(cell.split("_s", 1)[1].split("_", 1)[0])


def draw_of(cell: str) -> int:
    return int(cell.rsplit("_d", 1)[1])
