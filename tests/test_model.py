import numpy as np
import pytest

from _synth import make_seq
from mtah import features, model
from mtah.canonical import A, D, H

CFG = {"seed": 7, "stride": 2, "swap_prob": 0.5,
       "lgb": {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20,
               "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
               "lambda_l2": 1.0, "max_bin": 63, "num_threads": 1},
       "rounds": {"dead": 40, "team": 40, "change": 20}}


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
    Xs, Y, _ = model.sample(s, CFG, draw=0, X=X, names=names)
    y = Y[:, 0]
    assert Y.shape[1] == 3 and (Y[:, 2] == 0).all()        # no dense sampling configured
    assert (y >= 0).all() and len(y) == len(Xs) and 0 < len(y) <= len(s) // CFG["stride"] + 1
    nofl = dict(CFG, swap_prob=0.0)
    X0, Y0, _ = model.sample(s, nofl, draw=0, X=X, names=names)
    y0 = Y0[:, 0]
    np.testing.assert_array_equal(Y[:, 1], Y0[:, 1])     # a team change is one for both teams
    flipped = (y != y0)
    assert 0.2 < flipped[y0 != D].mean() < 0.8          # about half of the alive rows are swapped
    np.testing.assert_array_equal(Xs[flipped], features.swap_features(X0[flipped], names))


def test_save_load_roundtrip(art, test_seq, tmp_path):
    model.save(art, tmp_path / "m")
    back = model.load(tmp_path / "m")
    np.testing.assert_array_equal(model.predict_proba(back, test_seq), model.predict_proba(art, test_seq))
    assert back["names"] == art["names"]


def test_columns_are_matched_by_name(art, test_seq):
    X, names = features.compute(test_seq)
    p = model.predict_proba_X(art, X, names)
    np.testing.assert_array_equal(model.predict_proba_X(art, X[:, ::-1], names[::-1]), p)
    with pytest.raises(ValueError):
        model.predict_proba_X(art, X[:, 1:], names[1:])       # a trained feature is missing


def test_drop_features(train, test_seq):
    cfg = dict(CFG, rounds={"dead": 5, "team": 5}, drop_features=["ctrl", "^n_gap"])
    small = model.fit(train[:1], cfg)
    assert small["names"] and not any("ctrl" in n or n.startswith("n_gap") for n in small["names"])
    assert len(small["names"]) < len(features.compute(test_seq)[1])
    p = model.predict_proba(small, test_seq)
    np.testing.assert_allclose(model.predict_proba(small, features.swap_seq(test_seq)), p[:, [A, H, D]], atol=1e-6)


def test_load_artefact_convention(art, test_seq, tmp_path):
    model.save(art, tmp_path / "m")
    back = model.load_artefact(tmp_path / "m", {"anything": 1})
    p = model.predict_proba(back, test_seq)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_change_labels():
    y = np.array([2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0, 0, -1, 1, 1], np.int8)
    c = model.change_labels(y, radius=2)
    assert np.flatnonzero(c).tolist() == [5, 6, 7, 8]      # only the direct H->A change at 7
    assert not model.change_labels(np.array([0, 0, 2, 1, 1], np.int8)).any()   # via dead ball: none
    assert model.change_labels(np.array([0, 1], np.int8), radius=3).tolist() == [True, True]


def test_predict_aux(art, test_seq):
    aux = model.predict_aux(art, test_seq)
    p = aux["turn_onset"]
    assert p.dtype == np.float32 and p.shape == (len(test_seq),) and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(model.predict_aux(art, features.swap_seq(test_seq))["turn_onset"], p, atol=1e-6)
    c = model.change_labels(test_seq.y)
    assert p[c].mean() > 2 * p[~c].mean()                  # it does point at changes
    both = model.predict_both(art, test_seq)
    np.testing.assert_array_equal(both[0], model.predict_proba(art, test_seq))
    np.testing.assert_array_equal(both[1]["turn_onset"], p)


def test_no_change_head_means_no_aux(train, test_seq):
    cfg = dict(CFG, rounds={"dead": 5, "team": 5})
    small = model.fit(train[:1], cfg)
    assert model.predict_aux(small, test_seq) == {}
    assert model.predict_proba(small, test_seq).shape == (len(test_seq), 3)


def test_fit_arrays_list_equals_concatenated(train):
    cfg = dict(CFG, rounds={"dead": 8, "team": 8, "change": 8})
    parts = [model.sample(s, cfg, draw=i) for i, s in enumerate(train[:3])]
    names = parts[0][2]
    a = model.fit_arrays([p[0] for p in parts], [p[1] for p in parts], names, cfg)
    b = model.fit_arrays(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), names, cfg)
    for head in model.HEADS:
        assert a[head].model_to_string() == b[head].model_to_string()


def test_validation_curve_is_reported_not_used(train):
    cfg = dict(CFG, rounds={"dead": 6, "team": 6, "change": 6})
    X, Y, names = model.sample(train[0], cfg)
    Xv, Yv, _ = model.sample(train[1], cfg)
    log = {}
    a = model.fit_arrays(X, Y, names, cfg, valid=(Xv, Yv), log=log)
    b = model.fit_arrays(X, Y, names, cfg)
    assert set(log) == set(model.HEADS) and all(len(v) == 6 for v in log.values())
    for head in model.HEADS:
        np.testing.assert_array_equal(a[head].predict(Xv), b[head].predict(Xv))


def test_near_change_mask():
    y = np.array([2] * 5 + [0] * 20 + [1] * 4 + [0] * 20 + [2] * 5 + [1] * 3 + [2] * 3, np.int8)
    m = model.near_change_mask(y, radius=3, short=6)
    want = np.zeros(len(y), bool)
    want[22:28] = True          # +-3 around the change at 25
    want[25:29] = True          # the 4-frame away possession
    want[26:32] = True          # +-3 around the change at 29
    want[54:57] = True          # 3-frame possession between stoppages (no direct change)
    np.testing.assert_array_equal(m, want)
    assert not model.near_change_mask(np.array([0] * 50 + [2] * 10 + [1] * 50, np.int8), 5, 10).any()


def test_dense_sampling_and_weights(train):
    cfg = dict(CFG, stride=8, stride_near=2, near_radius=10, short_poss=20)
    s = train[0]
    X, names = features.compute(s)
    Xs, Y, _ = model.sample(s, dict(cfg, swap_prob=0.0), draw=0, X=X, names=names)
    base, _, _ = model.sample(s, dict(CFG, stride=8, swap_prob=0.0), draw=0, X=X, names=names)
    near = model.near_change_mask(s.y, 10, 20)
    assert len(Xs) > len(base) and near.any()
    # the dense rows are exactly the near frames on the fine grid; the rest sit on the coarse one
    n_near = int(Y[:, 2].sum())
    assert abs(n_near - near.sum() / 2) <= 1 and abs((len(Y) - n_near) - (~near).sum() / 8) <= 1
    w = model._head_weight("dead", Y, cfg)
    assert set(np.unique(w)) == {0.25, 1.0} and (w[Y[:, 2] == 1] == 0.25).all()
    assert model._head_weight("team", Y, cfg) is None
    # weighted share of near rows ~= their share of all frames (unbiased for the dead head)
    assert w[Y[:, 2] == 1].sum() / w.sum() == pytest.approx(near.mean(), abs=0.02)
    art = model.fit([s, train[1]], dict(cfg, rounds={"dead": 5, "team": 5, "change": 5}))
    assert set(model.HEADS) <= set(art)


def test_change_radius_from_config(train):
    y = train[0].y
    assert model.change_labels(y, 5).sum() > model.change_labels(y, 3).sum()
    cfg = dict(CFG, change_radius=5)
    _, Y, _ = model.sample(train[0], cfg, draw=0)
    _, Y3, _ = model.sample(train[0], CFG, draw=0)
    assert Y[:, 1].sum() >= Y3[:, 1].sum()


def test_regression_case_roundtrip_and_detection(art, test_seq, tmp_path):
    model.save(art, tmp_path / "m", regress_seq=test_seq)
    r = model.check_regression(tmp_path / "m")
    assert r["equal"] and r["max_abs_diff"] == 0.0 and r["n"] == len(test_seq)
    # a model that computes something else is caught
    other = model.fit([test_seq], dict(CFG, seed=8, rounds={"dead": 5, "team": 5, "change": 5}))
    model.save(other, tmp_path / "m")                      # overwrite the heads, keep the stored case
    assert not model.check_regression(tmp_path / "m")["equal"]


def test_artefact_remembers_its_feature_version(train, test_seq, tmp_path):
    old = model.fit(train[:1], dict(CFG, feature_version=2, rounds={"dead": 5, "team": 5}))
    new = model.fit(train[:1], dict(CFG, rounds={"dead": 5, "team": 5}))
    assert old["feature_version"] == 2 and new["feature_version"] == features.FEATURE_VERSION
    model.save(old, tmp_path / "old")
    back = model.load(tmp_path / "old")
    assert back["feature_version"] == 2
    np.testing.assert_array_equal(model.predict_proba(back, test_seq), model.predict_proba(old, test_seq))
    # the two versions really differ on this sequence (objects far outside the pitch exist in it)
    X2, n2 = features.compute(test_seq, 2)
    X3, n3 = features.compute(test_seq, 3)
    assert n2 == n3 and not np.array_equal(np.nan_to_num(X2), np.nan_to_num(X3))
