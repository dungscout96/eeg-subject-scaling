"""Flag cells that collapsed at initialisation (paper App. C.1).

A collapsed cell never leaves the chance value of the contrastive loss,
ln(64) = 4.16 at batch 64, and still writes a full, well-formed checkpoint
grid -- so it has to be screened for. Training loss grows with S (about 0.70
at S=10, 2.8 at S=1863), so the rule is relative: flag a cell whose FINAL
training loss exceeds 1.30x the median of its S-group in the same arm and
budget. On the reported runs failures sat at 1.67-3.09x and healthy cells at
<= 1.19x. Flagged cells are re-run with the next seed in the fixed sequence
2026 -> 7 -> 2025 (max three attempts; recorded in scripts/cells.py).

Reads the last ``loss=<x>`` printed by the trainer, so save each cell's
stdout+stderr to ``<runs-dir>/<arm>/<cell>/train.log``.

Usage:
    python scripts/screen_collapse.py --runs-dir runs
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import ARMS, ROOT, training_cells  # noqa: E402

THRESHOLD = 1.30
LOSS_RE = re.compile(r"loss=([0-9.]+)")


def final_loss(log: Path) -> float | None:
    if not log.exists():
        return None
    hits = LOSS_RE.findall(log.read_text(errors="replace"))
    return float(hits[-1]) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    n_flagged = 0
    for arm in ARMS:
        groups: dict[tuple[int, int], list[tuple[str, float]]] = defaultdict(list)
        for cell, S, _d, ep in training_cells(arm):
            loss = final_loss(args.runs_dir / arm / cell / "train.log")
            if loss is None:
                print(f"{arm:<15} {cell:<34} no log")
                continue
            groups[(S, ep)].append((cell, loss))
        for (S, ep), members in sorted(groups.items()):
            med = st.median(v for _, v in members)
            for cell, loss in members:
                ratio = loss / med
                flag = ratio > THRESHOLD
                n_flagged += flag
                print(f"{arm:<15} {cell:<34} loss={loss:.3f}  {ratio:.2f}x S-median"
                      + ("   <-- COLLAPSED: re-run with next seed" if flag else ""))
    print(f"\n{n_flagged} cell(s) flagged")


if __name__ == "__main__":
    main()
