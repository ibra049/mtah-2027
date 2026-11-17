"""mtah.model_hybrid: the combination rule, and bit-exact reproduction of its stored predictions."""
import json
from pathlib import Path

import numpy as np
import pytest

from mtah import model, model_hybrid, predict
from mtah.canonical import A, D, H

ROOT = Path(__file__).resolve().parents[1]
REGRESS = ROOT / "tests" / "data" / "hybrid_regress.npz"


def test_combine_rule():
    p1 = np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 1.0], [0.6, 0.4, 0.0]], np.float32)
    p2 = np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4], [0.0, 0.0, 1.0]], np.float32)  # last: no alive mass
    out = model_hybrid.combine(p1, p2)
    assert out.dtype == np.float32 and out.shape == (3, 3)
    np.testing.assert_allclose(out[:, D], p1[:, D])                       # dead from the first
    np.testing.assert_allclose(out[0, H] / (out[0, H] + out[0, A]), 0.75, rtol=1e-6)  # split from the second
    assert out[1, H] == 0 and out[1, A] == 0                               # P(D)=1 -> nothing alive
    np.testing.assert_allclose(out[2, [H, A]], [0.5, 0.5])                 # guarded division
    assert np.abs(out.sum(1) - 1).max() < 1e-6


def test_combine_identity_split():
    """With the same proba twice the split is the model's own (up to float rounding)."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], 1000).astype(np.float32)
    np.testing.assert_allclose(model_hybrid.combine(p, p), p, atol=2e-7)


def test_combine_rejects_bad_shapes():
    with pytest.raises(ValueError):
        model_hybrid.combine(np.zeros((3, 3)), np.zeros((4, 3)))


@pytest.mark.parametrize("team", ["v2", "v3"])
def test_hybrid_reproduces_stored_predictions(team):
    seq = model.regression_input(ROOT / "models" / "lgb_v1")
    art = model_hybrid.load_artefact(None, {"dead": "models/lgb_v1", "team": f"models/lgb_{team}"})
    proba, aux = model_hybrid.predict_both(art, seq)
    with np.load(REGRESS) as z:
        assert np.array_equal(proba, z[f"proba_{team}"])
        assert np.array_equal(aux["turn_onset"], z[f"turn_onset_{team}"])
    # and it is exactly combine() of the two base artefacts
    p1 = model.predict_proba(model.load(ROOT / "models" / "lgb_v1"), seq)
    p2 = model.predict_proba(model.load(ROOT / "models" / f"lgb_{team}"), seq)
    assert np.array_equal(proba, model_hybrid.combine(p1, p2))


@pytest.mark.parametrize("name", ["hybrid_v1d_v2t", "hybrid_v1d_v3t"])
def test_configs_load_through_the_predict_path(name):
    with open(ROOT / "configs" / f"{name}.json") as fh:
        cfg = json.load(fh)
    comp = predict.Components(cfg)
    assert comp.model_mod is model_hybrid and set(comp.artefact) >= {"dead", "team"}
    assert comp.use_aux and cfg["kickoff_rule"]
