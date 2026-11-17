import numpy as np
import pandas as pd
from mtah.io_metrica import event_labels


def _ev(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


def test_event_chain_rules():
    ev = _ev([
        ("H", "SET PIECE", "KICK OFF", 1, 10, 0),
        ("H", "PASS", "", 1, 10, 20),
        ("A", "CHALLENGE", "GROUND-LOST", 1, 25, 0),     # duel: no change
        ("H", "PASS", "", 1, 30, 40),
        ("H", "BALL LOST", "INTERCEPTION", 1, 50, 55),   # H had it at 50
        ("A", "RECOVERY", "INTERCEPTION", 1, 55, 0),     # A from 55
        ("A", "BALL OUT", "", 1, 60, 70),                # dead from 70
        ("H", "SET PIECE", "THROW IN", 1, 90, 0),        # restart H at 90
        ("A", "FAULT RECEIVED", "", 1, 100, 0),          # dead from 100
        ("A", "SET PIECE", "FREE KICK", 1, 120, 0),
    ])
    f = np.arange(0, 130)
    y = event_labels(ev, f, 1)
    assert (y[:10] == -1).all()
    assert (y[10:55] == 0).all()        # challenge at 25 does not flip
    assert (y[55:70] == 1).all()
    assert (y[70:90] == 2).all()
    assert (y[90:100] == 0).all()
    assert (y[100:120] == 2).all()
    assert (y[120:] == -1).all()        # after the last marker: unknown
