"""Pick each cell's checkpoint from its held-out epoch curve.

Paper protocol: the argmax of held-out time-pool e->v@1 (``e2v1_time``) over
every saved checkpoint of the cell -- no smoothing, no snapping. The appendix's
robustness check re-selects on the held-out 12-feature probe (``probe_mean_r``).

    --metric e2v1_time     -> results/<key>_selection_retr.json   (paper)
    --metric probe_mean_r  -> results/<key>_selection_probe.json  (App. C.2)

The random-init depth-22 arm is a COMPOSITE: 400-epoch cells below S=701,
800-epoch cells from S=701 up (the reported budget). For the probe selector
that composite is written to fromscratch_optimum_selection.json, the name the
analysis reads.

Usage:
    python scripts/select_checkpoints.py --write
    python scripts/select_checkpoints.py --metric probe_mean_r --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import LONG_MIN_S, ROOT, s_of  # noqa: E402

RESULTS = ROOT / "results"
ARMS = {
    "addval": dict(curves=["addval_epoch_curve.json"]),
    "d12av": dict(curves=["d12av_epoch_curve.json"]),
    "fromscratch": dict(curves=["fromscratch_epoch_curve.json",
                                "fromscratch800_epoch_curve.json"], composite=True),
}
OUT_NAME = {
    ("addval", "e2v1_time"): "addval_selection_retr.json",
    ("d12av", "e2v1_time"): "d12av_selection_retr.json",
    ("fromscratch", "e2v1_time"): "fromscratch_selection_retr.json",
    ("addval", "probe_mean_r"): "addval_selection_probe.json",
    ("d12av", "probe_mean_r"): "d12av_selection_probe.json",
    ("fromscratch", "probe_mean_r"): "fromscratch_optimum_selection.json",
}


def gather(arm: str) -> tuple[dict[str, dict], str]:
    """-> ({cell: {epoch: record}}, ckpt_root), composite routing applied."""
    spec = ARMS[arm]
    cells, ckpt_root = {}, ""
    for name in spec["curves"]:
        doc = json.loads((RESULTS / name).read_text())
        ckpt_root = doc["_meta"]["ckpt_root"]
        is800 = "800" in name
        for cell, curve in doc["cells"].items():
            if spec.get("composite") and is800 != (s_of(cell) >= LONG_MIN_S):
                continue
            cells[cell] = curve
    return cells, ckpt_root


def argmax_epoch(curve: dict, metric: str) -> tuple[int, float]:
    epochs = sorted(int(e) for e in curve)
    vals = [curve[str(e)][metric] for e in epochs]
    best = max(range(len(vals)), key=lambda i: vals[i])
    return epochs[best], vals[best]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="e2v1_time", choices=["e2v1_time", "probe_mean_r"])
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS), choices=sorted(ARMS))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for arm in args.arms:
        cells, ckpt_root = gather(arm)
        out = {}
        for cell in sorted(cells, key=lambda c: (s_of(c), c)):
            ep, val = argmax_epoch(cells[cell], args.metric)
            out[cell] = {"best_epoch": ep, "best_value": val, "selected_epoch": ep,
                         "snap_distance": 0,
                         "checkpoint": f"{ckpt_root}/{cell}/epoch_{ep}.pth.tar",
                         "selector": args.metric, "smooth_window": 1}
            print(f"{arm:<12} {cell:<34} epoch {ep:>4}  {args.metric}={val:.4f}")
        if args.write:
            dest = RESULTS / OUT_NAME[(arm, args.metric)]
            dest.write_text(json.dumps(out, indent=2))
            print(f"-> {dest} ({len(out)} cells)")


if __name__ == "__main__":
    main()
