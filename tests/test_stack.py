"""Temporal stacker features: finite, edge-safe and exactly team-symmetric."""
import numpy as np

from mtah import stack
from mtah.canonical import A, H


class Seq:
    def __init__(self, ball, home, away):
        self.ball, self.home, self.away = ball, home, away


def make(T=400, seed=0):
    rng = np.random.default_rng(seed)
    ball = np.c_[rng.normal(0, 20, (T, 2)), np.full(T, np.nan)].astype(np.float32)
    ball[50:120] = np.nan
    ball[300:] = np.nan
    home = rng.normal(0, 20, (T, 5, 2)).astype(np.float32)
    away = rng.normal(0, 20, (T, 5, 2)).astype(np.float32)
    home[10:30, 2] = np.nan
    proba = rng.dirichlet([1, 1, 1], T).astype(np.float32)
    onset = rng.random(T).astype(np.float32)
    return proba, onset, Seq(ball, home, away)


def test_features_are_finite_and_named():
    proba, onset, seq = make()
    X, names, sign = stack.features(proba, onset, seq)
    assert X.shape == (400, len(names)) == (400, len(sign)) and X.dtype == np.float32
    assert np.isfinite(X).all() and len(set(names)) == len(names)
    assert set(sign.tolist()) == {-1, 1}
    X0, _, _ = stack.features(proba, None, seq)            # works without an onset output
    assert np.isfinite(X0).all()


def test_probability_only_features_need_no_sequence():
    proba, onset, seq = make()
    X, names, sign = stack.features(proba, onset, None, use_cues=False)
    full, full_names, _ = stack.features(proba, onset, seq)
    assert X.shape[1] < full.shape[1] and set(names) < set(full_names)
    assert not any(n[0] in "mb" or "ball" in n for n in names)
    keep = [full_names.index(n) for n in names]
    np.testing.assert_array_equal(X, full[:, keep])               # same columns, same values
    Xs, _, _ = stack.features(proba[:, [A, H, 2]], onset, None, use_cues=False)
    np.testing.assert_allclose(Xs, stack.swap(X, sign), atol=1e-6)


def test_cues():
    proba, onset, seq = make()
    m, b, ts, tu = stack.cues(seq)
    assert b[49] == 1 and b[50] == 0 and b[119] == 0 and b[120] == 1
    assert ts[49] == 0 and ts[50] == 1 and ts[119] == 70 and ts[120] == 0
    assert tu[50] == 70 and tu[119] == 1 and tu[399] == stack.GAP_CAP
    assert (m[50:120] == 0).all() and np.abs(m).max() <= stack.MARGIN_CLIP
    d_h = np.nanmin(np.linalg.norm(seq.home[200] - seq.ball[200, :2], axis=1))
    d_a = np.nanmin(np.linalg.norm(seq.away[200] - seq.ball[200, :2], axis=1))
    assert np.isclose(m[200], np.clip(d_a - d_h, -10, 10), atol=1e-4)


def test_home_away_swap_is_a_sign_flip_of_the_antisymmetric_columns():
    proba, onset, seq = make(seed=3)
    X, _, sign = stack.features(proba, onset, seq)
    swapped = Seq(seq.ball, seq.away, seq.home)
    Xs, _, _ = stack.features(proba[:, [A, H, 2]], onset, swapped)
    np.testing.assert_allclose(Xs, stack.swap(X, sign), atol=1e-6)
    assert stack.swap_labels(np.array([0, 1, 2, -1])).tolist() == [1, 0, 2, -1]


def test_window_helpers_match_a_direct_loop():
    x = np.random.default_rng(1).random(60).astype(np.float32)
    for lo, hi in ((-5, -1), (1, 4), (-3, 3)):
        ref_mean = [x[max(0, t + lo):max(0, min(60, t + hi + 1))].mean()
                    if min(60, t + hi + 1) > max(0, t + lo) else 0.0 for t in range(60)]
        np.testing.assert_allclose(stack._wmean(x, lo, hi), ref_mean, atol=1e-6)
        ref_max = [x[np.clip(np.arange(t + lo, t + hi + 1), 0, 59)].max() for t in range(60)]
        np.testing.assert_allclose(stack._wmax(x, lo, hi), ref_max, atol=1e-6)


# ------------------------------------------------------------------ pipeline wrapper
def _fake_base(proba, onset):
    import sys
    import types
    mod = types.ModuleType("fake_base_model")
    mod.load_artefact = lambda path, cfg=None: {"path": path}
    mod.predict_both = lambda art, seq: (proba, {"turn_onset": onset})
    sys.modules["fake_base_model"] = mod


def test_model_stacked_wraps_a_base_model(tmp_path):
    import pytest
    from mtah import model_stacked
    proba, onset, seq = make(T=600, seed=5)
    X, names, sign = stack.features(proba, onset, seq)
    y = np.argmax(proba, axis=1)
    booster = stack.fit(X, y, sign, {"num_boost_round": 5, "min_data_in_leaf": 5, "num_threads": 1})
    art_dir = tmp_path / "models" / "stack_test"
    art_dir.mkdir(parents=True)
    booster.save_model(str(art_dir / "stack.txt"))
    model_stacked.write_meta(art_dir, names, base_module="fake_base_model",
                             base_artefact="models/base", onset_key="turn_onset", use_cues=True)
    _fake_base(proba, onset)
    art = model_stacked.load_artefact(art_dir)
    assert art["base"]["path"].endswith("models/base")           # resolved against the repo root
    out, aux = model_stacked.predict_both(art, seq)
    assert out.shape == (600, 3) and out.dtype == np.float32
    assert np.allclose(out.sum(1), 1.0, atol=1e-5) and (out >= 0).all()
    assert aux["turn_onset"] is onset                            # aux passes through untouched
    np.testing.assert_array_equal(out, model_stacked.refine(art, proba, aux, seq))
    np.testing.assert_array_equal(out, model_stacked.predict_proba(art, seq))
    np.testing.assert_array_equal(out, stack.predict_proba(booster, X))
    # home/away swap of the whole input swaps the H and A outputs (up to tree asymmetry: the
    # symmetric predictor is exact)
    sym = stack.predict_proba(booster, X, sign, symmetric=True)
    sym_sw = stack.predict_proba(booster, stack.swap(X, sign), sign, symmetric=True)
    np.testing.assert_allclose(sym_sw, sym[:, [A, H, 2]], atol=1e-6)
    # a changed booster file or feature list is refused
    art["names"] = names[:-1]
    with pytest.raises(ValueError):
        model_stacked.refine(art, proba, aux, seq)
    (art_dir / "stack.txt").write_text((art_dir / "stack.txt").read_text() + "\n")
    with pytest.raises(ValueError):
        model_stacked.load_artefact(art_dir)
