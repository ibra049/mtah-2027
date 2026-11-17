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


def test_control_events(seq, XN):
    X, names = XN
    col = {n: X[:, i] for i, n in enumerate(names)}
    obs = col["n_obs"] == 1
    ctrl = col["d_ctrl15"]
    assert np.isnan(ctrl[~obs]).all() and set(np.unique(ctrl[obs])) <= {-1.0, 0.0, 1.0}
    home_ev = np.flatnonzero(obs & (ctrl == 1))
    t = int(home_ev[len(home_ev) // 2])
    assert col["h_ctrl15_tlast"][t] == 0 and col["h_ctrl15_tnext"][t] == 0
    assert col["d_ctrl15_last"][t] == 1 and col["d_ctrl15_next"][t] == 1
    # time since the last home control grows by one frame per frame until the next one
    nxt = home_ev[home_ev > t]
    if len(nxt) and nxt[0] > t + 1:
        assert col["h_ctrl15_tlast"][t + 1] == pytest.approx(1 / 25)
        assert col["h_ctrl15_tnext"][t + 1] == pytest.approx((nxt[0] - t - 1) / 25)
    # the synthetic possessor carries the ball, so the last controller is the label
    alive = seq.y != 2
    agree = (col["d_ctrl15_last"][alive] > 0) == (seq.y[alive] == 0)
    assert agree.mean() > 0.85


def test_kick_events_need_a_seen_velocity_jump(seq, XN):
    X, names = XN
    col = {n: X[:, i] for i, n in enumerate(names)}
    obs = col["n_obs"] == 1
    kick = col["d_kick"]
    assert np.isnan(kick[~obs]).all()
    ev = np.flatnonzero(obs & (kick != 0))
    assert len(ev) > 0                                    # carriers change -> the ball jumps
    assert (col["n_kink"][ev] > features.KICK_DV).all()
    assert (np.fmin(col["h_d1"][ev], col["a_d1"][ev]) < features.KICK_RADIUS).all()


def test_regime_descriptors(seq, XN):
    X, names = XN
    col = {n: X[:, i] for i, n in enumerate(names)}
    miss = np.isnan(seq.ball[:, 0])
    reg = [n for n in names if n.startswith("n_reg_")]
    assert len(reg) == 6
    for n in reg:
        assert (col[n] == col[n][0]).all()                 # one value per half
    assert col["n_reg_miss"][0] == pytest.approx(miss.mean(), abs=1e-4)
    assert 0 <= col["n_reg_gap10"][0] <= col["n_reg_gap4"][0] <= col["n_reg_gap2"][0] <= 1
    vis = (~np.isnan(seq.home[..., 0])).sum(1) + (~np.isnan(seq.away[..., 0])).sum(1)
    assert col["n_reg_players"][0] == pytest.approx(vis.mean(), abs=1e-4)
    s = seq.copy()
    s.ball[:] = np.nan
    s.ball[0, :2] = 0.0
    c2 = dict(zip(names, features.compute(s)[0].T))
    assert c2["n_reg_gap10"][0] == 1.0 and c2["n_reg_gapmed"][0] == pytest.approx((len(s) - 1) / 25, abs=1e-3)


def test_gap_clock_runs_past_30_s():
    s = make_seq(T=1200, seed=6)
    s.ball[100:1000] = np.nan                               # a gap of at least 36 s
    seen = np.flatnonzero(~np.isnan(s.ball[:, 0]))
    last, nxt = seen[seen < 100].max(), seen[seen >= 1000].min()
    X, names = features.compute(s)
    col = dict(zip(names, X.T))
    assert col["n_t_since"][990] == pytest.approx((990 - last) / 25, abs=1e-3) and col["n_t_since"][990] > 30
    assert col["n_t_until"][101] == pytest.approx((nxt - 101) / 25, abs=1e-3) and col["n_t_until"][101] > 30
    assert col["n_gap"][500] == pytest.approx((nxt - last - 1) / 25, abs=1e-3)
