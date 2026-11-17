import numpy as np
import pytest

from _synth import make_seq
from mtah import features
from mtah.canonical import A, H


@pytest.fixture(scope="module")
def seq():
    return make_seq(T=1200, seed=1)


@pytest.fixture(scope="module")
def XN(seq):
    return features.compute(seq)


def test_shape_dtype_names(seq, XN):
    X, names = XN
    assert X.dtype == np.float32 and X.shape == (len(seq), len(names))
    assert len(set(names)) == len(names)
    assert all(n[:2] in ("n_", "h_", "a_", "d_") for n in names)
    assert not np.isinf(X).any()
    assert np.isfinite(X).mean() > 0.9          # NaN only where something is truly unknown


def test_deterministic_and_fixed_order(seq, XN):
    X, names = XN
    X2, names2 = features.compute(seq)
    assert names2 == names
    np.testing.assert_array_equal(X, X2)
    other = features.compute(make_seq(T=300, seed=5, slots=23))[1]
    assert other == names                        # order does not depend on the data


def test_swap_is_exact(seq, XN):
    X, names = XN
    Xs, names_s = features.compute(features.swap_seq(seq))
    assert names_s == names
    np.testing.assert_array_equal(Xs, features.swap_features(X, names))
    # and it is an involution
    np.testing.assert_array_equal(features.swap_features(features.swap_features(X, names), names), X)


def test_swap_seq_labels(seq):
    s = features.swap_seq(seq)
    assert ((seq.y == H) == (s.y == A)).all() and ((seq.y == A) == (s.y == H)).all()


@pytest.mark.parametrize("flip", [(-1, 1), (1, -1), (-1, -1)])
def test_pitch_mirror_invariance(seq, XN, flip):
    X, names = XN
    m = seq.copy()
    f = np.array(flip, np.float32)
    m.ball[:, :2] *= f
    m.home *= f
    m.away *= f
    Xm, _ = features.compute(m)
    np.testing.assert_allclose(Xm, X, rtol=1e-4, atol=1e-4)


def test_attack_dir_estimate():
    assert features.attack_dir(make_seq(T=400, seed=2, home_attacks_plus=True)) == 1.0
    assert features.attack_dir(make_seq(T=400, seed=2, home_attacks_plus=False)) == -1.0


def test_slot_order_does_not_matter(seq, XN):
    X, names = XN
    rng = np.random.default_rng(0)
    p = seq.copy()
    p.home = p.home[:, rng.permutation(p.home.shape[1])]
    p.away = p.away[:, rng.permutation(p.away.shape[1])]
    Xp, _ = features.compute(p)
    np.testing.assert_allclose(Xp, X, rtol=1e-4, atol=1e-4)


def test_ignores_ball_height_and_labels(seq, XN):
    X, _ = XN
    s = seq.copy()
    s.ball[:, 2] = np.nan
    s.y[:] = -1
    np.testing.assert_array_equal(features.compute(s)[0], X)
    s.ball[:, 2] = 7.0
    np.testing.assert_array_equal(features.compute(s)[0], X)


def test_gap_features(seq, XN):
    X, names = XN
    col = {n: X[:, i] for i, n in enumerate(names)}
    obs = ~np.isnan(seq.ball[:, 0])
    np.testing.assert_array_equal(col["n_obs"], obs.astype(np.float32))
    assert (col["n_t_since"][obs] == 0).all() and (col["n_gap"][obs] == 0).all()
    t = int(np.flatnonzero(~obs & np.r_[False, obs[:-1]])[0])      # first frame of a gap
    assert col["n_t_since"][t] == pytest.approx(1 / 25)
    n = int(np.argmax(obs[t:]))                                    # gap length in frames
    assert col["n_gap"][t] == pytest.approx(n / 25)
    assert col["n_t_until"][t] == pytest.approx(n / 25)
    # the last-seen snapshot is the value at the last observed frame
    np.testing.assert_array_equal(col["n_ls_btouch"][t], col["n_btouch"][t - 1])
    assert np.isnan(col["d_closer"][~obs]).all()


def test_nan_heavy_inputs_do_not_break():
    s = make_seq(T=400, seed=3)
    names = features.compute(s)[1]
    cases = []
    a = s.copy(); a.ball[:] = np.nan; cases.append(a)                      # ball never seen
    b = s.copy(); b.home[:] = np.nan; cases.append(b)                      # a team never seen
    c = s.copy(); c.home[100:200] = np.nan; c.away[150:250] = np.nan; c.ball[:300] = np.nan
    cases.append(c)
    d = s.copy(); d.home = d.home[:, :2]; d.away = d.away[:, :1]; cases.append(d)   # K < 3
    e = make_seq(T=7, seed=4); cases.append(e)                             # shorter than the lags
    for q in cases:
        X, n = features.compute(q)
        assert n == names and X.shape == (len(q), len(names))
        assert not np.isinf(X).any()
    Xa = features.compute(a)[0]
    assert np.isnan(Xa[:, names.index("n_t_since")]).all()
    assert np.isfinite(Xa[:, names.index("n_sp_mean")]).mean() > 0.9       # player cues survive


def test_windows_are_past_and_future(seq, XN):
    X, names = XN
    col = {n: X[:, i] for i, n in enumerate(names)}
    t = 600
    obs = col["n_obs"]
    assert col["n_obs_p12"][t] == pytest.approx(obs[t - 12:t].mean())
    assert col["n_obs_f50"][t] == pytest.approx(obs[t + 1:t + 51].mean())
    bt = col["n_btouch"]
    assert col["n_btouch_pmin50"][t] == pytest.approx(np.nanmin(bt[t - 50:t]))
    assert col["n_btouch_fmin50"][t] == pytest.approx(np.nanmin(bt[t + 1:t + 51]))


def test_rounded_to_1e4(XN):
    X, _ = XN
    np.testing.assert_array_equal(np.round(X.astype(np.float64), 4).astype(np.float32), X)
