"""The shipped results reproduce the paper: selection, then every reported number."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sel = _load("select_checkpoints", ROOT / "scripts" / "select_checkpoints.py")


@pytest.mark.parametrize("arm,metric", sorted(sel.OUT_NAME))
def test_selection_reproduces_shipped_selection(arm, metric):
    cells, _ = sel.gather(arm)
    shipped = json.loads((sel.RESULTS / sel.OUT_NAME[(arm, metric)]).read_text())
    assert set(cells) == set(shipped)
    for cell, curve in cells.items():
        assert sel.argmax_epoch(curve, metric)[0] == shipped[cell]["selected_epoch"], cell


def test_every_paper_number_recomputes(capsys):
    numbers = _load("paper_numbers", ROOT / "analysis" / "paper_numbers.py")
    assert numbers.main() == 0
    assert "280/280 checks agree" in capsys.readouterr().out
