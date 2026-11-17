import numpy as np
from mtah.touch_eval import score


def test_score_one_to_one_and_team():
    tf, tt = np.array([10, 50, 100]), np.array([0, 1, 0])
    pf, pt = np.array([12, 13, 49, 200]), np.array([0, 0, 0, 1])
    r = score(pf, pt, tf, tt, 3, team=False)
    assert r["tp"] == 2 and r["n_pred"] == 4          # 12->10, 49->50; 13 is a duplicate
    r = score(pf, pt, tf, tt, 3, team=True)
    assert r["tp"] == 1                               # 49 has the wrong team
