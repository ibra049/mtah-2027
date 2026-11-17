import numpy as np
import pytest

from _synth import make_seq
from mtah import features, model
from mtah.canonical import A, D, H

CFG = {"seed": 7, "stride": 2, "swap_prob": 0.5,
       "lgb": {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20,
               "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
               "lambda_l2": 1.0, "max_bin": 63, "num_threads": 1},
       "rounds": {"dead": 40, "team": 40}}


@pytest.fixture(scope="module")
def train():
    return [make_seq(T=1500, seed=s, home_attacks_plus=bool(s % 2), match_id=f"S{s}") for s in range(4)]


@pytest.fixture(scope="module")
def art(train):
    return model.fit(train, CFG)


@pytest.fixture(scope="module")
def test_seq():
    return make_seq(T=1500, seed=99, match_id="TEST")


def test_predict_shape_and_simplex(art, test_seq):
    p = model.predict_proba(art, test_seq)
    assert p.dtype == np.float32 and p.shape == (len(test_seq), 3)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)


def test_learns_the_synthetic_task(art, test_seq):
    p = model.predict_proba(art, test_seq)
    y = test_seq.y
    ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)))
    prior = np.bincount(y, minlength=3) / len(y)
    ll_prior = -np.sum(prior * np.log(prior))
    assert ll < 0.6 * ll_prior
    assert ((p[:, D] > 0.5) == (y == D)).mean() > 0.9
    alive = y != D
    assert ((p[alive, H] > p[alive, A]) == (y[alive] == H)).mean() > 0.85


def test_prediction_is_team_symmetric(art, test_seq):
    p = model.predict_proba(art, test_seq)
    ps = model.predict_proba(art, features.swap_seq(test_seq))
    np.testing.assert_allclose(ps, p[:, [A, H, D]], atol=1e-6)


def test_fit_is_deterministic(train, art):
    again = model.fit(train, CFG)
    for head in model.HEADS:
        assert again[head].model_to_string() == art[head].model_to_string()


def test_sample_swaps_and_masks_unknown(train):
    s = train[0].copy()
    s.y[::3] = -1
    X, names = features.compute(s)
    Xs, y, _ = model.sample(s, CFG, draw=0, X=X, names=names)
    assert (y >= 0).all() and len(y) == len(Xs) and 0 < len(y) <= len(s) // CFG["stride"] + 1
    nofl = dict(CFG, swap_prob=0.0)
    X0, y0, _ = model.sample(s, nofl, draw=0, X=X, names=names)
    flipped = (y != y0)
    assert 0.2 < flipped[y0 != D].mean() < 0.8          # about half of the alive rows are swapped
    np.testing.assert_array_equal(Xs[flipped], features.swap_features(X0[flipped], names))


def test_save_load_roundtrip(art, test_seq, tmp_path):
    model.save(art, tmp_path / "m")
    back = model.load(tmp_path / "m")
    np.testing.assert_array_equal(model.predict_proba(back, test_seq), model.predict_proba(art, test_seq))
    assert back["names"] == art["names"]


def test_feature_order_is_checked(art, test_seq):
    X, names = features.compute(test_seq)
    with pytest.raises(ValueError):
        model.predict_proba_X(art, X, names[::-1])
