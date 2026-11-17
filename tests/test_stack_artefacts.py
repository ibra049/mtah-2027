"""Every committed stacker artefact must reproduce its own stored predictions bit for bit.

models/stack_*/regress.npz holds the refined probabilities of frame model + stacker on the base
model's small fixed sequence (models/<base>/regress.npz). No external data is needed, so this
protects a public clone: a change to mtah.features, mtah.model, mtah.stack or mtah.model_stacked
that alters what an existing artefact computes fails here.
"""
from pathlib import Path

import pytest

from mtah import model_stacked

MODELS = Path(__file__).resolve().parents[1] / "models"
STACKS = sorted(p.parent for p in MODELS.glob("stack_*/meta.json"))


def test_there_are_stack_artefacts_and_each_has_a_regression_case():
    assert STACKS
    missing = [p.name for p in STACKS if not (p / model_stacked.REGRESS_FILE).exists()]
    assert not missing, f"no {model_stacked.REGRESS_FILE} for {missing}"


@pytest.mark.parametrize("path", STACKS, ids=lambda p: p.name)
def test_stack_artefact_reproduces_its_stored_predictions(path):
    r = model_stacked.check_regression(path)
    assert r["n"] >= 500
    assert r["base_case_unchanged"], f"{path.name}: the base model's regression input changed"
    assert r["equal"], f"{path.name}: predictions changed (max abs diff {r['max_abs_diff']:.3g})"
