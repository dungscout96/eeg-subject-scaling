"""Evaluate selected checkpoints on the untouched test release (HBN R6).

Four readouts per cell, each at the cell's selected checkpoint:

    within-task probe      e04_tt_test_<cell><sfx>.json       (ThePresent, ridge refit)
    within-task retrieval  e04_retr_test_<cell><sfx>.json     (ThePresent, 101-way, fit-free)
    cross-task probe       xtask_tt_DMtest_<cell><sfx>.json   (DespicableMe, ridge refit)
    cross-task retrieval   xtask_retr_DMtest_<cell><sfx>.json (DespicableMe, zero-shot)

<sfx> is _epretr for the retrieval-selected checkpoints (the paper) and _epprb
for the probe-selected ones (App. C.2). The probe's ridge head is fit on every
train-pool recording outside R5 (1863 ThePresent / 1832 DespicableMe), the same
set for every cell, and scored with a 2000-iteration cluster bootstrap.

`random` writes the untrained-encoder floors, one per depth:
    depth 22: e04_tt_test_random.json, e04_retr_test_random.json,
              xtask_tt_DMtest_random_e04.json, xtask_retr_DMtest_random_e04.json
    depth 12: e03_tt_test_random.json, e03_retr_test_random.json,
              xtask_tt_DMtest_random.json, xtask_retr_DMtest_random.json

Usage:
    python scripts/evaluate_test.py cells  --arm pretrained_d22 --runs-dir runs
    python scripts/evaluate_test.py cells  --arm pretrained_d22 --selector probe
    python scripts/evaluate_test.py random --depth 22
Existing outputs are skipped, so a run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import ARMS, ROOT  # noqa: E402

RESULTS = ROOT / "results"
BOOTSTRAP = 2000
TOPKS = ["1", "5", "10"]
PROBE = ROOT / "eb_jepa/evaluation/clip_probe/probe_traintest.py"
RETR = ROOT / "eb_jepa/evaluation/clip_probe/retrieval.py"
SELECTION = {
    ("pretrained_d22", "retr"): "addval_selection_retr.json",
    ("random_d22", "retr"): "fromscratch_selection_retr.json",
    ("random_d12", "retr"): "d12av_selection_retr.json",
    ("pretrained_d22", "probe"): "addval_selection_probe.json",
    ("random_d22", "probe"): "fromscratch_optimum_selection.json",
    ("random_d12", "probe"): "d12av_selection_probe.json",
}
SUFFIX = {"retr": "_epretr", "probe": "_epprb"}


def _run(cmd: list[str], out: Path) -> None:
    if out.exists():
        print(f"SKIP {out.name}")
        return
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def probe_cmd(cfg, out, ckpt=None):
    src = ["--checkpoint", ckpt] if ckpt else ["--random-baseline"]
    return [sys.executable, PROBE, *src, "--config", cfg, "--eval-split", "test",
            "--device", DEVICE, "--bootstrap", str(BOOTSTRAP), "--output", out]


def retr_cmd(cfg, out, ckpt=None):
    src = ["--checkpoint", ckpt] if ckpt else ["--random-baseline"]
    return [sys.executable, RETR, *src, "--config", cfg, "--split", "test",
            "--device", DEVICE, "--topks", *TOPKS, "--output", out]


def readouts(arm: str):
    """(file prefix, config, command builder) for the four readouts."""
    c = ARMS[arm].test_configs
    return [("e04_tt_test", ROOT / c["within"], probe_cmd),
            ("e04_retr_test", ROOT / c["within"], retr_cmd),
            ("xtask_tt_DMtest", ROOT / c["cross"], probe_cmd),
            ("xtask_retr_DMtest", ROOT / c["cross"], retr_cmd)]


def eval_cells(arm: str, selector: str, runs_dir: Path) -> None:
    sel = json.loads((RESULTS / SELECTION[(arm, selector)]).read_text())
    for cell, info in sel.items():
        ckpt = runs_dir / arm / cell / f"epoch_{info['selected_epoch']}.pth.tar"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        for prefix, cfg, build in readouts(arm):
            out = RESULTS / f"{prefix}_{cell}{SUFFIX[selector]}.json"
            _run(build(cfg, out, ckpt), out)


def eval_random(depth: int) -> None:
    arm = "pretrained_d22" if depth == 22 else "random_d12"
    names = ({"e04_tt_test": "e04_tt_test_random", "e04_retr_test": "e04_retr_test_random",
              "xtask_tt_DMtest": "xtask_tt_DMtest_random_e04",
              "xtask_retr_DMtest": "xtask_retr_DMtest_random_e04"} if depth == 22 else
             {"e04_tt_test": "e03_tt_test_random", "e04_retr_test": "e03_retr_test_random",
              "xtask_tt_DMtest": "xtask_tt_DMtest_random",
              "xtask_retr_DMtest": "xtask_retr_DMtest_random"})
    for prefix, cfg, build in readouts(arm):
        out = RESULTS / f"{names[prefix]}.json"
        _run(build(cfg, out), out)


DEVICE = "cuda"


def main() -> None:
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["cells", "random"])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--selector", choices=["retr", "probe"], default="retr")
    ap.add_argument("--depth", type=int, choices=[12, 22])
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    DEVICE = args.device
    RESULTS.mkdir(exist_ok=True)
    if args.action == "cells":
        if not args.arm:
            ap.error("cells needs --arm")
        eval_cells(args.arm, args.selector, args.runs_dir)
    else:
        if not args.depth:
            ap.error("random needs --depth")
        eval_random(args.depth)


if __name__ == "__main__":
    main()
