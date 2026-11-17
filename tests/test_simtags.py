import numpy as np
from mtah import simtags
from mtah.canonical import Sequence


def _seq():
    T = 1000
    y = np.array([2] * 50 + [0] * 300 + [2] * 100 + [1] * 550, np.int8)
    ball = np.zeros((T, 3), np.float32)
    ball[450] = [10, 33.5, 0]            # touchline -> throw-in
    z = np.zeros((T, 1, 2), np.float32)
    return Sequence("SYN", 1, np.arange(T) + 5000, ball, z, z, np.array([105, 68], np.float32), y)


def test_restarts_types_and_window_contains_restart():
    cfg = simtags.load_cfg()
    s = _seq()
    rec = simtags.restart_records(s, cfg)
    assert [r[0] for r in rec] == [5050, 5450]
    assert rec[0][1] == "Kick Off (Start)" and rec[0][2] == "H"
    assert rec[1][1] == "Throw-in" and rec[1][2] == "N"
    tags = simtags.make_tags(s, cfg, cfg["noise"]["clean"], np.random.default_rng(0))
    assert len(tags) == 2
    for t, (f, *_r) in zip(tags, rec):
        assert t["start"] <= f < t["end"] and 75 <= t["end"] - t["start"] <= 175
