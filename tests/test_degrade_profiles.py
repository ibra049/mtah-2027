import numpy as np
from mtah import degrade
from mtah.canonical import Sequence


def _seq(T=3000, seed=0):
    r = np.random.default_rng(seed)
    home = np.cumsum(r.normal(0, 0.1, (T, 11, 2)), 0).astype(np.float32) + r.uniform(-30, 30, (1, 11, 2)).astype(np.float32)
    away = np.cumsum(r.normal(0, 0.1, (T, 11, 2)), 0).astype(np.float32) + r.uniform(-30, 30, (1, 11, 2)).astype(np.float32)
    ball = np.c_[np.cumsum(r.normal(0, 0.3, (T, 2)), 0), np.zeros(T)].astype(np.float32)
    y = np.repeat(np.array([0, 2, 1, 2, 0], np.int8), T // 5)
    return Sequence("SYN001", 1, np.arange(T), ball, home, away, np.array([105, 68], np.float32), y)


def test_base_cfg_ignores_profiles_and_new_features_off():
    cfg = degrade.load_cfg()
    assert "profiles" not in cfg and cfg["ball_dead_mult"] == 1.0
    assert cfg["false_ball_per_min"] == 0 and cfg["wrong_team_prob"] == 0


def test_profile_deterministic_and_drawn_in_range():
    s = _seq()
    for name in degrade.PROFILES:
        cfg = degrade.load_profile(name)
        a, da = degrade.degraded_profile(s, cfg, seed=0)
        b, db = degrade.degraded_profile(s, cfg, seed=0)
        c, dc = degrade.degraded_profile(s, cfg, seed=1)
        assert da == db and da != dc
        assert np.array_equal(a.ball, b.ball, equal_nan=True) and np.array_equal(a.home, b.home, equal_nan=True)
        for k, (lo, hi) in cfg["randomize"].items():
            assert lo <= da[k] <= hi
        assert np.isnan(a.ball[:, 2]).all()
        assert a.home.shape[1] == cfg["slots_per_team"]
        assert np.array_equal(a.y, s.y)


def test_false_ball_and_wrong_team_fire():
    s = _seq()
    cfg = degrade.load_cfg() | {"false_ball_per_min": 50.0}
    out = degrade.apply(s, np.random.default_rng(0), cfg)
    jumps = np.nansum(np.linalg.norm(np.diff(out.ball[:, :2], axis=0), axis=1) > 5)
    assert jumps > 5
    cfg = degrade.load_cfg() | {"wrong_team_prob": 1.0}
    h = np.full((100, 3, 2), np.nan, np.float32); a = h.copy()
    h[10:40, 0] = 1.0
    h2, a2 = degrade.wrong_team(h.copy(), a.copy(), np.random.default_rng(0), cfg)
    assert np.isnan(h2).all() and (~np.isnan(a2[10:40, :, 0])).sum() == 30
