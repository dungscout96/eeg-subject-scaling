"""Train one cell of the subject-scaling ladder, or list them all.

A cell is (arm, S, draw, budget). The per-cell knobs are written into a copy of
the arm's training config inside the run directory, and the rest is passed to
the trainer exactly as for the reported runs:

    data.max_subjects   = S
    data.subsample_seed = draw         (cohorts nest within a draw seed)
    optim.epochs        = 400 | 800
    meta.seed           = 2026, or the collapse re-run seed (scripts/cells.py)
    meta.encoder_init_from = REVE checkpoint   (pretrained arm only)
    logging.save_every  = 25

Each cell needs one ~44 GB GPU; a 400-epoch depth-22 cell takes ~80 min on an
A40. Run cells in parallel on a cluster by looping over ``--list``.

Usage:
    python scripts/train_cell.py --list [--arm pretrained_d22]
    python scripts/train_cell.py --arm pretrained_d22 --S 100 --draw 11 \\
        --runs-dir runs --reve-ckpt checkpoints/reve_base_eet_init.pth.tar
    python scripts/train_cell.py --arm random_d22 --S 1000 --draw 22 --epochs 800 --runs-dir runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import ARMS, ROOT, SAVE_EVERY, cell_name, seed_for, training_cells  # noqa: E402


def run_dir(runs_dir: Path, arm: str, cell: str) -> Path:
    return runs_dir / arm / cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="Print every cell to train and exit.")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--S", type=int)
    ap.add_argument("--draw", type=int)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=None,
                    help="Override meta.seed (default: 2026 or the recorded re-run seed).")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--reve-ckpt", type=Path, default=None,
                    help="Converted REVE weights (scripts/prepare_reve_checkpoint.py). "
                         "Required for the pretrained arm.")
    ap.add_argument("--preprocessed-dir", default=None,
                    help="Root of scripts/preprocess_hbn.py output (else $HBN_PREPROCESS_DIR).")
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    args = ap.parse_args()

    if args.list:
        for arm in ([args.arm] if args.arm else ARMS):
            for cell, S, d, ep in training_cells(arm):
                print(f"{arm}\t{cell}\tS={S}\tdraw={d}\tepochs={ep}\tseed={seed_for(arm, S, d)}")
        return

    if None in (args.arm, args.S, args.draw):
        ap.error("--arm, --S and --draw are required (or use --list)")
    arm = ARMS[args.arm]
    if arm.pretrained and args.reve_ckpt is None:
        ap.error("the pretrained arm needs --reve-ckpt")
    if not arm.pretrained and args.reve_ckpt is not None:
        ap.error(f"{args.arm} is a random-init arm; do not pass --reve-ckpt")

    cell = cell_name(args.arm, args.S, args.draw, args.epochs)
    out = run_dir(args.runs_dir, args.arm, cell).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(ROOT / arm.train_config)
    cfg.data.max_subjects = args.S
    cfg.data.subsample_seed = args.draw
    if args.preprocessed_dir:
        cfg.data.preprocessed_dir = str(Path(args.preprocessed_dir).resolve())
    cfg_path = out / "config.yaml"
    OmegaConf.save(cfg, cfg_path)

    overrides = {
        "optim.epochs": args.epochs,
        "meta.seed": args.seed if args.seed is not None else seed_for(args.arm, args.S, args.draw),
        "logging.save_every": SAVE_EVERY,
        "logging.log_wandb": bool(args.wandb),
        "logging.wandb_group": args.arm,
    }
    if arm.pretrained:
        overrides["meta.encoder_init_from"] = str(args.reve_ckpt.resolve())
    print(f"[train_cell] {cell} -> {out}\n  overrides: {overrides}", flush=True)

    from eb_jepa.training.clip_pretrain import run
    run(fname=str(cfg_path), folder=str(out), **overrides)


if __name__ == "__main__":
    main()
