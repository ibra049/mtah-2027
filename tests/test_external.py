import numpy as np
from mtah.relabel import min_control
from mtah.evaluate import join_halves, spell_f1, report, X
from mtah.io_idsse import labels_from_raw

def test_labels_from_raw():
    st = np.array([0, 1, 1, 1, 0], np.int8); po = np.array([1, 1, 2, 0, 2], np.int8)
    assert labels_from_raw(st, po).tolist() == [2, 0, 1, -1, 2]

def test_min_control_absorbs_short_flip_but_keeps_restart():
    y = np.array([2]*3 + [1]*2 + [0]*10 + [1]*3 + [0]*10 + [1]*20 + [2]*3, np.int8)
    out = min_control(y, 5)
    assert out[3:5].tolist() == [1, 1]          # first run after restart kept even if short
    assert (out[15:18] == 0).all()              # 3-frame A touch inside H absorbed
    assert (out[28:48] == 1).all()              # 20-frame A run is a real change
    assert (out == 2).sum() == 6

def test_join_halves_and_metric():
    y = [np.array([0]*50 + [1]*50), np.array([2]*10 + [0]*90)]
    j = join_halves(y)
    assert j[100] == X and len(j) == 201
    assert report(y, y)["spell_f1_mean"] == 1.0
    assert spell_f1(np.full(100, 0), y[0]) < 1.0
