"""Derive the two non-pixel probe features from annotate.py's per-frame table.

    position_in_movie      t / (N - 1)
    narrative_event_score  mean over f in {motion_energy, luminance_mean,
                           n_faces, depth_mean} of z(|Delta f[t]|), with
                           Delta f[0] = 0 and z taken over the whole movie
                           (sample standard deviation)

(paper, Appendix A.2). The other ten probe features come straight from
annotate.py. tests/test_movie_features.py checks this reproduces the shipped
movie_artifacts/*/features_enriched.parquet exactly.

Usage:
    python movie_annotation/enrich_features.py \\
        --features output/The_Present/features.parquet \\
        --output   output/The_Present/features_enriched.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NARRATIVE_COMPONENTS = ["motion_energy", "luminance_mean", "n_faces", "depth_mean"]


def position_in_movie(n_frames: int) -> np.ndarray:
    return np.arange(n_frames) / (n_frames - 1)


def narrative_event_score(df: pd.DataFrame) -> np.ndarray:
    zs = []
    for f in NARRATIVE_COMPONENTS:
        x = df[f].to_numpy(dtype=float)
        d = np.abs(np.diff(x, prepend=x[0]))         # |Delta f|, Delta f[0] = 0
        zs.append((d - d.mean()) / d.std(ddof=1))
    return np.mean(zs, axis=0)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["position_in_movie"] = position_in_movie(len(out))
    out["narrative_event_score"] = narrative_event_score(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.features).sort_values("frame_idx").reset_index(drop=True)
    enrich(df).to_parquet(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} frames)")


if __name__ == "__main__":
    main()
