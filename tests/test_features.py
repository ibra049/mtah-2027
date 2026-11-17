import numpy as np
import pytest

from _synth import make_seq
from mtah import features
from mtah.canonical import A, H


def compute(seq, version=3):
    """The tests written for feature versions 2/3 keep running on version 3; version 4 has its
    own tests at the end of this file, and the invariance tests run on both."""
    return features.compute(seq, version)


@pytest.fixture(scope="module")
def seq():
    return make_seq(T=1200, seed=1)


@pytest.fixture(scope="module")
def XN(seq):
    return compute(seq)


def test_shape_dtype_names(seq, XN):
    X, names = XN
    assert X.dtype == np.float32 and X.shape == (len(seq), len(names))
    assert len(set(names)) == len(names)
    assert all(n[:2] in ("n_", "h_", "a_", "d_") for n in names)
    assert not np.isinf(X).any()
    assert np.isfinite(X).mean() > 0.9          # NaN only where something is truly unknown


def test_deterministic_and_fixed_order(seq, XN):
    X, names = XN
    X2, names2 = compute(seq)
    assert names2 == names
    np.testing.assert_array_equal(X, X2)
    other = compute(make_seq(T=300, seed=5, slots=23))[1]
    assert other == names                        # order does not depend on the data


def test_swap_is_exact(seq, XN):
    X, names = XN
    Xs, names_s = compute(features.swap_seq(seq))
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
    Xm, _ = compute(m)
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
    Xp, _ = compute(p)
    np.testing.assert_allclose(Xp, X, rtol=1e-4, atol=1e-4)


def test_ignores_ball_height_and_labels(seq, XN):
    X, _ = XN
    s = seq.copy()
    s.ball[:, 2] = np.nan
    s.y[:] = -1
    np.testing.assert_array_equal(compute(s)[0], X)
    s.ball[:, 2] = 7.0
    np.testing.assert_array_equal(compute(s)[0], X)


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
    names = compute(s)[1]
    cases = []
    a = s.copy(); a.ball[:] = np.nan; cases.append(a)                      # ball never seen
    b = s.copy(); b.home[:] = np.nan; cases.append(b)                      # a team never seen
    c = s.copy(); c.home[100:200] = np.nan; c.away[150:250] = np.nan; c.ball[:300] = np.nan
    cases.append(c)
    d = s.copy(); d.home = d.home[:, :2]; d.away = d.away[:, :1]; cases.append(d)   # K < 3
    e = make_seq(T=7, seed=4); cases.append(e)                             # shorter than the lags
    for q in cases:
        X, n = compute(q)
        assert n == names and X.shape == (len(q), len(names))
        assert not np.isinf(X).any()
    Xa = compute(a)[0]
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


def _kick_seq(hidden):
    """Ball at rest next to a home player, kicked at frame 100, received by an away player at
    frame 150 (8 m/s along x). hidden=True blanks the ball around the kick."""
    T = 300
    ball = np.zeros((T, 3), np.float32)
    ball[:, 0] = np.clip((np.arange(T) - 100) * (8.0 / 25), 0, 16.0)
    home = np.full((T, 3, 2), np.nan, np.float32)
    away = np.full((T, 3, 2), np.nan, np.float32)
    home[:, 0] = (-0.5, 0.3)
    home[:, 1] = (-20.0, 10.0)
    home[:, 2] = (-40.0, 0.0)
    away[:, 0] = (16.5, -0.3)
    away[:, 1] = (25.0, 12.0)
    away[:, 2] = (45.0, 0.0)
    if hidden:
        ball[90:125] = np.nan
    from mtah.canonical import Sequence
    return Sequence("K", 1, np.arange(T, dtype=np.int64), ball, home, away,
                    np.array([105.0, 68.0], np.float32), np.zeros(T, np.int8))


def test_seen_touches():
    s = _kick_seq(hidden=False)
    X, names = compute(s)
    col = dict(zip(names, X.T))
    ev = features.touch_events(s.ball[:, :2], ~np.isnan(s.ball[:, 0]), s.home, s.away,
                               np.linalg.norm(s.home - s.ball[:, None, :2], axis=2).min(1),
                               np.linalg.norm(s.away - s.ball[:, None, :2], axis=2).min(1), 5.0)
    assert ev["s"].tolist() == [1.0, -1.0] and (ev["hid"] == 0).all()
    assert abs(ev["t"][0] - 100) <= 2 and abs(ev["t"][1] - 150) <= 2
    assert ev["mag"][0] == pytest.approx(8.0, abs=0.5)
    k, r = int(ev["t"][0]), int(ev["t"][1])
    # in flight: the last touch is home's, the next one away's; before the kick nothing behind
    assert col["d_tch_last"][125] == 1 and col["d_tch_next"][125] == -1
    assert col["n_tch_tlast"][125] == pytest.approx((125 - k) / 25, abs=1e-3)
    assert col["n_tch_tnext"][125] == pytest.approx((r - 125) / 25, abs=1e-3)
    assert col["d_tch_last"][50] == 0 and col["n_tch_tlast"][50] == features.T_CAP_S
    assert col["d_tch_last2"][200] == 1 and col["d_tch_last"][200] == -1
    assert col["n_tch_last_d"][125] == pytest.approx(np.hypot(0.5, 0.3), abs=0.3)
    assert col["d_kick_last"][125] == 1 and col["d_kick_next"][125] == -1      # the strict stream sees it too
    assert col["h_tch_p50"][k + 10] == 1 and col["a_tch_p50"][k + 10] == 0
    assert col["a_tch_f50"][r - 10] == 1 and col["d_tch_f50"][r - 10] == -1
    assert col["h_tch_p25"][k + 40] == 0 and col["h_tch_p125"][k + 40] == 1


def test_hidden_touch_is_reconstructed_inside_the_gap():
    s = _kick_seq(hidden=True)
    obs = ~np.isnan(s.ball[:, 0])
    dh = np.linalg.norm(s.home - s.ball[:, None, :2], axis=2).min(1)
    da = np.linalg.norm(s.away - s.ball[:, None, :2], axis=2).min(1)
    ev = features.touch_events(s.ball[:, :2], obs, s.home, s.away, dh, da, 5.0)
    hid = ev["hid"] == 1
    assert hid.sum() == 1 and ev["s"][hid][0] == 1               # the kicker's team
    assert abs(int(ev["t"][hid][0]) - 100) <= 2                  # at the true kick time
    assert ev["d"][hid][0] == pytest.approx(np.hypot(0.5, 0.3), abs=0.3)
    X, names = compute(s)
    col = dict(zip(names, X.T))
    assert col["d_tch_last"][130] == 1 and col["n_tch_last_hid"][130] == 1
    # a ball that crosses a gap on a straight line was not touched
    straight = _kick_seq(hidden=False)
    straight.ball[110:140] = np.nan
    obs2 = ~np.isnan(straight.ball[:, 0])
    ev2 = features.touch_events(straight.ball[:, :2], obs2, straight.home, straight.away, dh, da, 5.0)
    assert (ev2["hid"] == 0).all()


def test_regime_descriptors(seq, XN):
    X, names = XN
    col = {n: X[:, i] for i, n in enumerate(names)}
    miss = np.isnan(seq.ball[:, 0])
    reg = [n for n in names if n.startswith("n_reg_")]
    assert len(reg) == 9
    for n in reg:
        assert (col[n] == col[n][0]).all()                 # one value per half
    assert col["n_reg_miss"][0] == pytest.approx(miss.mean(), abs=1e-4)
    assert 0 <= col["n_reg_gap10"][0] <= col["n_reg_gap4"][0] <= col["n_reg_gap2"][0] <= 1
    inp = features._in_play(seq)                           # objects far off the pitch do not count
    vis = (~np.isnan(inp.home[..., 0])).sum(1) + (~np.isnan(inp.away[..., 0])).sum(1)
    assert col["n_reg_players"][0] == pytest.approx(vis.mean(), abs=1e-4)
    s = seq.copy()
    s.ball[:] = np.nan
    s.ball[0, :2] = 0.0
    c2 = dict(zip(names, compute(s)[0].T))
    assert c2["n_reg_gap10"][0] == 1.0 and c2["n_reg_gapmed"][0] == pytest.approx((len(s) - 1) / 25, abs=1e-3)


def test_gap_clock_runs_past_30_s():
    s = make_seq(T=1200, seed=6)
    s.ball[100:1000] = np.nan                               # a gap of at least 36 s
    seen = np.flatnonzero(~np.isnan(s.ball[:, 0]))
    last, nxt = seen[seen < 100].max(), seen[seen >= 1000].min()
    X, names = compute(s)
    col = dict(zip(names, X.T))
    assert col["n_t_since"][990] == pytest.approx((990 - last) / 25, abs=1e-3) and col["n_t_since"][990] > 30
    assert col["n_t_until"][101] == pytest.approx((nxt - 101) / 25, abs=1e-3) and col["n_t_until"][101] > 30
    assert col["n_gap"][500] == pytest.approx((nxt - last - 1) / 25, abs=1e-3)


def test_touch_threshold_follows_the_noise():
    """A ball passed back and forth (a kick every 2 s). On a jittery track the absolute jump
    threshold of a clean track would fire on noise; the relative one keeps the strict stream
    close to the true number of kicks on both."""
    from mtah.canonical import Sequence
    T = 3000
    tri = np.abs((np.arange(T) % 100) - 50) * (8.0 / 25)          # 0..16 m and back, 8 m/s
    rng = np.random.default_rng(0)
    n_true = T // 50
    counts = {}
    for name, sigma in (("clean", 0.0), ("jitter", 0.4)):
        ball = np.zeros((T, 3), np.float32)
        ball[:, 0] = tri
        e = np.cumsum(rng.normal(0, 1, (T + 20, 2)), 0)
        ball[:, :2] += ((e[20:] - e[:-20]) / np.sqrt(20) * sigma).astype(np.float32)
        home = np.full((T, 3, 2), np.nan, np.float32)
        away = np.full((T, 3, 2), np.nan, np.float32)
        home[:, 0], home[:, 1], home[:, 2] = (-0.7, 0.2), (-20.0, 10.0), (-40.0, 0.0)
        away[:, 0], away[:, 1], away[:, 2] = (16.7, -0.2), (25.0, 12.0), (45.0, 0.0)
        s = Sequence("N", 1, np.arange(T, dtype=np.int64), ball, home, away,
                     np.array([105.0, 68.0], np.float32), np.zeros(T, np.int8))
        X, names = compute(s)
        col = dict(zip(names, X.T))
        counts[name] = (float(col["n_reg_jscale"][0]), int((col["n_kick_tlast"] == 0).sum()),
                        int((col["n_tch_tlast"] == 0).sum()))
    assert counts["jitter"][0] > counts["clean"][0] + 1.0                  # the noise scale sees the jitter
    assert 0.8 * n_true <= counts["clean"][1] <= 1.2 * n_true              # every kick, nothing else
    assert 0.6 * n_true <= counts["jitter"][1] <= 1.6 * n_true             # still selective under jitter
    assert counts["jitter"][2] >= counts["jitter"][1]                      # the sensitive stream fires more


def test_objects_far_outside_the_pitch_are_ignored(seq, XN):
    X, names = XN
    s = seq.copy()
    K = s.home.shape[1]
    s.home = np.concatenate([s.home, np.full((len(s), 1, 2), np.nan, np.float32)], 1)
    s.home[:, K] = (-60.0, 5.0) if features.attack_dir(seq) > 0 else (60.0, 5.0)   # 7.5 m behind the home goal
    X2, _ = compute(s)
    skip = [i for i, n in enumerate(names)]
    np.testing.assert_allclose(X2[:, skip], X[:, skip], rtol=1e-4, atol=1e-4)
    # a taker one metre outside the touchline still counts
    s2 = seq.copy()
    s2.home = np.concatenate([s2.home, np.full((len(s2), 1, 2), np.nan, np.float32)], 1)
    s2.home[:, K] = (0.0, 35.0)
    X3, _ = compute(s2)
    assert X3[0, names.index("h_touch")] == pytest.approx(-1.0, abs=1e-3)


def test_relative_kinematics_have_a_twin_for_every_raw_column(seq, XN):
    """Every absolute-unit kinematic column listed in RAW_KINEMATICS has a "_rel" twin (so a
    model for another tracking provider can drop the raw ones)."""
    import re
    X, names = XN
    col = dict(zip(names, X.T))
    assert col["d_blockv_rel"][300] == pytest.approx(col["d_blockv"][300] / col["n_reg_spscale"][0], abs=2e-4)
    raw = [n for n in names if any(re.search(p, n) for p in features.RAW_KINEMATICS)]
    assert len(raw) > 30
    for n in raw:
        base = re.sub(r"(_[pf]\d+)?$", "", n)
        tail = n[len(base):]
        assert base + "_rel" + tail in names, n


def test_speed_floor_makes_standing_and_jittering_players_equal():
    """Players that stand still (smooth tracker) and players that jitter on the spot by 0.2 m
    (noisy tracker) give the same "_rel" speed statistics; the raw ones differ."""
    from mtah.canonical import Sequence
    T, K = 600, 11
    rng = np.random.default_rng(3)
    spots = np.stack([rng.uniform(-45, 45, (2, K)), rng.uniform(-30, 30, (2, K))], -1).astype(np.float32)
    spots[0, :, 0] -= 5                       # home a little behind away: a defined attack direction
    out = {}
    for name, sigma in (("still", 0.0), ("jitter", 0.2)):
        e = np.cumsum(rng.normal(0, 1, (T + 20, 2, K, 2)), 0)
        wander = ((e[20:] - e[:-20]) / np.sqrt(20) * sigma).astype(np.float32)
        ball = np.zeros((T, 3), np.float32)
        s = Sequence("F", 1, np.arange(T, dtype=np.int64), ball, spots[0] + wander[:, 0],
                     spots[1] + wander[:, 1], np.array([105.0, 68.0], np.float32), np.zeros(T, np.int8))
        X, names = compute(s)
        out[name] = dict(zip(names, X.T))
    mid = slice(50, 550)
    assert np.nanmean(out["jitter"]["n_sp_mean"][mid]) > np.nanmean(out["still"]["n_sp_mean"][mid]) + 0.2
    for n in ("n_sp_mean_rel", "n_sp_q10_rel", "n_sp_med_rel", "h_sp_mean_rel", "a_sp_q25_rel"):
        np.testing.assert_allclose(np.nanmean(out["jitter"][n][mid]), np.nanmean(out["still"][n][mid]),
                                   rtol=0.03, atol=0.03, err_msg=n)
    # the slow share only loses the tail of the jitter above the floor
    assert np.nanmean(out["still"]["n_slow_rel"][mid]) == pytest.approx(1.0)
    assert np.nanmean(out["jitter"]["n_slow_rel"][mid]) > 0.8


def test_movement_coherence():
    """Everyone moving the same way: coherence near 1; everyone drifting at random: near 0."""
    from mtah.canonical import Sequence
    T, K = 400, 11
    rng = np.random.default_rng(4)
    base = np.stack([rng.uniform(-30, 0, (2, K)), rng.uniform(-25, 25, (2, K))], -1).astype(np.float32)
    base[1, :, 0] += 8
    t = np.arange(T, dtype=np.float32)[:, None, None]
    shift = np.concatenate([t * (3.0 / 25), 0 * t], 2)                    # 3 m/s along x for all
    ball = np.zeros((T, 3), np.float32)
    mk = lambda h, a: Sequence("C", 1, np.arange(T, dtype=np.int64), ball, h.astype(np.float32),   # noqa: E731
                               a.astype(np.float32), np.array([105.0, 68.0], np.float32), np.zeros(T, np.int8))
    X, names = compute(mk(base[0] + shift, base[1] + shift))
    assert np.nanmean(X[50:350, names.index("n_coh")]) > 0.9
    ang = rng.uniform(0, 2 * np.pi, (2, K))
    drift = np.stack([np.cos(ang), np.sin(ang)], -1) * (2.0 / 25)         # 2 m/s, random headings
    X, names = compute(mk(base[0] + t * drift[0], base[1] + t * drift[1]))
    assert np.nanmean(X[50:350, names.index("n_coh")]) < 0.4


def test_feature_versions(seq):
    X2, n2 = compute(seq, version=2)
    X3, n3 = compute(seq, version=3)
    assert n2 == n3
    with pytest.raises(ValueError):
        compute(seq, version=1)
    col2, col3 = dict(zip(n2, X2.T)), dict(zip(n3, X3.T))
    # version 2: every tracked object counts; version 3 ignores the ones far outside the pitch
    raw = (~np.isnan(seq.home[..., 0])).sum(1) + (~np.isnan(seq.away[..., 0])).sum(1)
    assert col2["n_reg_players"][0] == pytest.approx(raw.mean(), abs=1e-4)
    assert col3["n_reg_players"][0] < col2["n_reg_players"][0]
    # both stay exactly team-symmetric
    for v, X in ((2, X2), (3, X3)):
        np.testing.assert_array_equal(compute(features.swap_seq(seq), v)[0],
                                      features.swap_features(X, n2))


# ------------------------------------------------------------------ feature version 4
@pytest.fixture(scope="module")
def XN4(seq):
    return features.compute(seq, 4)


def test_v4_is_the_default_and_keeps_every_invariance(seq, XN4):
    X, names = XN4
    Xd, nd = features.compute(seq)
    assert features.FEATURE_VERSION == 4 and nd == names
    np.testing.assert_array_equal(Xd, X)
    assert X.dtype == np.float32 and len(set(names)) == len(names) and not np.isinf(X).any()
    assert all(n[:2] in ("n_", "h_", "a_", "d_") for n in names)
    np.testing.assert_array_equal(np.round(X.astype(np.float64), 4).astype(np.float32), X)
    # exact team symmetry
    Xs, ns = features.compute(features.swap_seq(seq), 4)
    assert ns == names
    np.testing.assert_array_equal(Xs, features.swap_features(X, names))
    # pitch mirroring
    for flip in ((-1, 1), (1, -1), (-1, -1)):
        m = seq.copy()
        f = np.array(flip, np.float32)
        m.ball[:, :2] *= f
        m.home *= f
        m.away *= f
        np.testing.assert_allclose(features.compute(m, 4)[0], X, rtol=1e-4, atol=1e-4)
    # slot order, ball height, labels
    rng = np.random.default_rng(0)
    q = seq.copy()
    q.home = q.home[:, rng.permutation(q.home.shape[1])]
    q.away = q.away[:, rng.permutation(q.away.shape[1])]
    q.ball[:, 2] = 9.0
    q.y[:] = -1
    np.testing.assert_allclose(features.compute(q, 4)[0], X, rtol=1e-4, atol=1e-4)


def test_v4_nan_heavy_inputs_do_not_break():
    s = make_seq(T=400, seed=3)
    names = features.compute(s, 4)[1]
    cases = []
    a = s.copy(); a.ball[:] = np.nan; cases.append(a)
    b = s.copy(); b.home[:] = np.nan; cases.append(b)
    c = s.copy(); c.home[100:200] = np.nan; c.away[150:250] = np.nan; c.ball[:300] = np.nan; cases.append(c)
    d = s.copy(); d.home = d.home[:, :2]; d.away = d.away[:, :1]; cases.append(d)
    cases.append(make_seq(T=7, seed=4))
    for q in cases:
        X, n = features.compute(q, 4)
        assert n == names and X.shape == (len(q), len(names)) and not np.isinf(X).any()


def test_canonicalize_fills_short_gaps_and_keeps_long_ones():
    s = make_seq(T=1200, seed=6, ball_missing=0.0)
    s.ball[100:120] = np.nan                    # 0.8 s: filled
    s.ball[400:500] = np.nan                    # 4 s: stays a gap
    before = s.ball.copy()
    c = features.canonicalize(s)
    np.testing.assert_array_equal(s.ball, before)                     # input untouched
    assert c.ball.shape == s.ball.shape and c.home.shape == s.home.shape and (c.frame == s.frame).all()
    seen = ~np.isnan(c.ball[:, 0])
    assert seen[100:120].all() and not seen[400:500].any() and seen[:100].all() and seen[500:].all()
    # players: observed mask unchanged (never interpolated), apart from the off-pitch filter
    inp = features._in_play(s)
    np.testing.assert_array_equal(np.isnan(c.home), np.isnan(inp.home))
    X, names = features.compute(s, 4)
    col = dict(zip(names, X.T))
    assert (col["n_obs"][100:120] == 1).all() and (col["n_gap"][100:120] == 0).all()
    assert col["n_gap"][450] == pytest.approx(100 / 25, abs=1e-3)
    assert not any(n in names for n in ("n_reg_players", "n_reg_pjit", "n_reg_jscale", "n_reg_gapmed", "n_reg_gap2"))
    assert all(n in names for n in ("n_reg_miss", "n_reg_gap4", "n_reg_gap10", "n_reg_gap20", "n_reg_gap30"))


def test_canonicalize_makes_jittered_and_smooth_tracks_alike():
    """A straight 8 m/s ball and standing players, once clean and once with 0.3 m of slow
    jitter: after canonicalisation frame-to-frame velocities are smooth on both (lag-1
    autocorrelation near 1, as on a low-pass-filtered tracker), and the filter does not move
    a clean straight track."""
    from mtah.canonical import Sequence
    T, K = 1500, 6
    rng = np.random.default_rng(5)
    x = np.abs((np.arange(T) % 500) - 250) * (8.0 / 25) - 40.0
    spots = np.stack([rng.uniform(-40, 40, (2, K)), rng.uniform(-30, 30, (2, K))], -1).astype(np.float32)
    def autocorr(b):
        v = np.diff(b, axis=0)
        a, c = v[:-1] - v[:-1].mean(0), v[1:] - v[1:].mean(0)
        return float((a * c).sum() / np.sqrt((a * a).sum() * (c * c).sum()))
    out = {}
    for name, sigma in (("clean", 0.0), ("jitter", 0.3)):
        ball = np.zeros((T, 3), np.float32)
        ball[:, 0] = x
        e = np.cumsum(rng.normal(0, 1, (T + 10, 2)), 0)
        ball[:, :2] += ((e[10:] - e[:-10]) / np.sqrt(10) * sigma).astype(np.float32)
        s = Sequence("J", 1, np.arange(T, dtype=np.int64), ball,
                     np.broadcast_to(spots[0], (T, K, 2)).copy(), np.broadcast_to(spots[1], (T, K, 2)).copy(),
                     np.array([105.0, 68.0], np.float32), np.zeros(T, np.int8))
        c = features.canonicalize(s)
        out[name] = (autocorr(s.ball[:, :2]), autocorr(c.ball[:, :2]), c.ball[:, :2], s.ball[:, :2])
    assert out["jitter"][0] < 0.8 and out["jitter"][1] > 0.93          # raw jittery, canonical smooth
    assert out["clean"][1] > 0.93
    mid = slice(20, 230)                                                # inside a straight leg
    np.testing.assert_allclose(out["clean"][2][mid], out["clean"][3][mid], atol=1e-3)


def test_v4_touches_on_a_filtered_track():
    """A kick and a reception survive the low-pass filter: found by both v4 streams with the
    right teams, near the true frames, also when the ball is hidden around the kick for 2 s."""
    s = _kick_seq(hidden=False)
    c = features.canonicalize(s)
    ev = features.touch_events_v4(c)
    assert set(ev) >= {"t", "s", "hid", "mag", "stream"}
    m = ev["stream"] == 0                                               # the sensitive stream
    assert ev["s"][m].tolist() == [1.0, -1.0]
    assert abs(int(ev["t"][m][0]) - 100) <= 4 and abs(int(ev["t"][m][1]) - 150) <= 4
    strict = ev["t"][ev["stream"] == 1]                                  # the strict one is a subset
    assert set(strict.tolist()) <= set(ev["t"][m].tolist())
    X, names = features.compute(s, 4)
    col = dict(zip(names, X.T))
    assert col["d_tch_last"][125] == 1 and col["d_tch_next"][125] == -1
    assert col["d_turn"][100] > 0 and col["d_turn"][150] < 0           # home turns it, then away
    h = _kick_seq(hidden=False)
    h.ball[80:130] = np.nan                                             # 2 s gap around the kick
    ev = features.touch_events_v4(features.canonicalize(h))
    hid = ev["hid"] == 1
    assert hid.any() and (ev["s"][hid] == 1).all() and abs(int(ev["t"][hid][0]) - 100) <= 6


def test_v4_hold_features():
    """Home is nearest (ball at its feet) until the pass arrives at the away player."""
    s = _kick_seq(hidden=False)
    X, names = features.compute(s, 4)
    col = dict(zip(names, X.T))
    assert (col["d_hold"][10:120] == 1).all() and (col["d_hold"][140:290] == -1).all()
    flip = int(np.flatnonzero(col["d_hold"] == -1)[0])                  # the frame away becomes nearest
    assert 115 <= flip <= 135
    assert col["n_hold_since"][flip + 25] == pytest.approx(1.0, abs=1e-3)
    assert col["n_hold_until"][flip - 25] == pytest.approx(1.0, abs=1e-3)
    assert col["d_hold_since"][flip + 25] == pytest.approx(-1.0, abs=1e-3)
    assert col["n_hold_prev"][flip + 25] == pytest.approx(flip / 25, abs=0.05)   # home held it from frame 0
    assert col["n_hold_next"][flip - 25] == features.T_CAP_S                      # nobody after away


def test_v4_ball_free_stoppage_cues(seq, XN4):
    X, names = XN4
    col = dict(zip(names, X.T))
    assert {"n_box_max", "n_box_max_p375", "n_sp_std_rel", "n_sp_std_rel_f375", "n_coh_p375"} <= set(names)
    # the fuller box: home's own box holds home defenders and away attackers, and vice versa
    np.testing.assert_array_equal(
        col["n_box_max"], np.maximum(col["h_own_box"] + col["a_opp_box"], col["h_opp_box"] + col["a_own_box"]))
    assert (col["n_sp_std_rel"][~np.isnan(col["n_sp_std_rel"])] >= 0).all()
