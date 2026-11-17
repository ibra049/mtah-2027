"""Every committed model must reproduce its own stored predictions bit for bit.

models/<name>/regress.npz holds a small fixed sequence and the artefact's predictions on it,
written with the code of the commit that produced the artefact. A change to mtah.features or
mtah.model that alters what an existing artefact computes fails here (CONTRACT: a feature name
never changes its computation; new behaviour gets a new name or a new feature-set version).
"""
from pathlib import Path

import pytest

from mtah import model

MODELS = Path(__file__).resolve().parents[1] / "models"
# Frame models only (they carry dead.txt). Stacker artefacts (models/stack_*) get their own
# bit-exact case from the decoder lane (tests/test_stack_artefacts.py).
ARTEFACTS = sorted(p.parent for p in MODELS.glob("*/meta.json") if (p.parent / "dead.txt").exists())


def test_there_are_artefacts_and_each_has_a_regression_case():
    assert ARTEFACTS
    missing = [p.name for p in ARTEFACTS if not (p / model.REGRESS_FILE).exists()]
    assert not missing, f"no {model.REGRESS_FILE} for {missing}"


@pytest.mark.parametrize("path", ARTEFACTS, ids=lambda p: p.name)
def test_artefact_reproduces_its_stored_predictions(path):
    r = model.check_regression(path)
    assert r["n"] >= 500
    assert r["equal"], f"{path.name}: predictions changed (max abs diff {r['max_abs_diff']:.3g})"
