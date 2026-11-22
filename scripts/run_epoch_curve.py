"""Score every saved checkpoint on held-out subjects (the checkpoint-selection curve).

Runs eb_jepa/evaluation/clip_probe/epoch_curve.py once per (arm, draw): the
held-out set is the draw's cohort complement -- the 293 subjects outside its
S=1863 cohort -- which no cell of that draw ever trained on. 80 of them are
read (60 fit the fixed-alpha ridge probe, 20 score it; retrieval uses all 80).

`merge` then combines the per-draw files into the per-arm curve files that
scripts/select_checkpoints.py and analysis/ read:

    pretrained_d22 -> results/addval_epoch_curve.json
    random_d22     -> results/fromscratch_epoch_curve.json     (400-epoch cells)
                      results/fromscratch800_epoch_curve.json  (800-epoch cells)
    random_d12     -> results/d12av_epoch_curve.json

Usage:
    python scripts/run_epoch_curve.py run   --arm pretrained_d22 --draw 11 --runs-dir runs
    python scripts/run_epoch_curve.py merge --arm pretrained_d22
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import (ARMS, COMPLEMENT_OF, DEFAULT_EPOCHS, ROOT, draw_of,  # noqa: E402
                   training_cells)

RESULTS = ROOT / "results"
N_RECORDINGS = 80
N_FIT = 60
SEED = 42
# Fixed ridge alpha for the selection probe, held constant across every cell and
# checkpoint so the curve cannot move because regularisation moved. Calibrated
# once with epoch_curve.py --calibrate-alpha (RidgeCV median, 10^3.5).
ALPHA = 3162.2776601683795

# Output groups: (merged file stem, which training budgets go in it).
GROUPS = {
    "pretrained_d22": [("addval", None)],
    "random_d22": [("fromscratch", DEFAULT_EPOCHS), ("fromscratch800", 800)],
    "random_d12": [("d12av", None)],
}


def cells_for(arm: str, draw: int, budget: int | None) -> list[str]:
    return [c for c, S, d, ep in training_cells(arm)
            if d == draw and S < 2156 and (budget is None or ep == budget)]


def run(arm: str, draw: int, runs_dir: Path, device: str) -> None:
    spec = ARMS[arm]
    RESULTS.mkdir(exist_ok=True)
    for stem, budget in GROUPS[arm]:
        cells = cells_for(arm, draw, budget)
        out = RESULTS / f"{stem}_epoch_curve_d{draw}.json"
        cmd = [sys.executable, str(ROOT / "eb_jepa/evaluation/clip_probe/epoch_curve.py"),
               "--ckpt-root", str((runs_dir / arm).resolve()), "--cells", *cells,
               "--config", str(ROOT / spec.select_config),
               "--split", "train", "--holdout-complement-of", str(COMPLEMENT_OF),
               "--subsample-seed", str(draw), "--device", device,
               "--n-recordings", str(N_RECORDINGS), "--n-fit", str(N_FIT),
               "--seed", str(SEED), "--alpha", str(ALPHA), "--output", str(out)]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def merge(arm: str) -> None:
    for stem, _ in GROUPS[arm]:
        parts = sorted(RESULTS.glob(f"{stem}_epoch_curve_d*.json"))
        if not parts:
            raise SystemExit(f"no {stem}_epoch_curve_d*.json in {RESULTS}")
        merged, meta, seed_by_cell = {}, None, {}
        for p in parts:
            doc = json.loads(p.read_text())
            if meta is None:
                meta = dict(doc["_meta"])
            for k in ("split", "seed", "n_recordings", "n_fit_recordings", "alpha",
                      "holdout_complement_of"):
                if doc["_meta"].get(k) != meta.get(k):
                    raise SystemExit(f"{p.name}: incompatible _meta[{k!r}]")
            draw = doc["_meta"]["subsample_seed"]
            for cell in doc["cells"]:
                if cell in merged:
                    raise SystemExit(f"duplicate cell {cell} in {p.name}")
                # Each draw must be scored on ITS OWN complement, or some cells
                # would be selected on subjects they trained on.
                if draw_of(cell) != draw:
                    raise SystemExit(f"{cell} scored on draw {draw}'s complement")
                seed_by_cell[cell] = draw
            merged.update(doc["cells"])
        meta["subsample_seed"] = "per-draw"
        meta["subsample_seed_by_cell"] = seed_by_cell
        dest = RESULTS / f"{stem}_epoch_curve.json"
        dest.write_text(json.dumps({"_meta": meta, "cells": merged}, indent=2))
        print(f"wrote {dest} ({len(merged)} cells from {len(parts)} files)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "merge"])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--draw", type=int, help="run: which draw (11, 22 or 33)")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.action == "run":
        if args.draw is None:
            ap.error("run needs --draw")
        run(args.arm, args.draw, args.runs_dir, args.device)
    else:
        merge(args.arm)


if __name__ == "__main__":
    main()
