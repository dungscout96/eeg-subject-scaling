"""The feature derivation in movie_annotation/ reproduces the shipped artifacts."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "enrich", ROOT / "movie_annotation" / "enrich_features.py")
enrich = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enrich)

PROBE_FEATURES = [
    "luminance_mean", "contrast_rms", "saturation_mean", "edge_density", "entropy",
    "motion_energy", "depth_mean", "n_faces", "face_area_frac",
    "scene_natural_score", "position_in_movie", "narrative_event_score",
]


@pytest.mark.parametrize("task,n_frames", [("ThePresent", 4877), ("DespicableMe", 4266)])
def test_derived_features_match_artifacts(task, n_frames):
    df = pd.read_parquet(ROOT / "movie_artifacts" / task / "features_enriched.parquet")
    assert len(df) == n_frames
    assert set(PROBE_FEATURES) <= set(df.columns)
    np.testing.assert_allclose(enrich.position_in_movie(len(df)), df["position_in_movie"], atol=1e-12)
    np.testing.assert_allclose(enrich.narrative_event_score(df), df["narrative_event_score"], atol=1e-9)
