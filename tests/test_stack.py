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
