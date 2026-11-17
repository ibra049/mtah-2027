"""Label-semantics bridge: DFL BallPossession flips on any touch; the challenge's GT does not.

CHALLENGE.md: a change of possession needs *controlled* possession; "a simple touch of the ball
without control" and "a deflection of a pass that does not intercept it" do not count, and "in
phases where the ball is contested or loose, possession stays with the last team that had it".

Rule (min_control): inside each alive stretch (between D runs), walk the H/A runs in order.
The first run after a restart keeps its team (CHALLENGE.md: "whenever play restarts after a
stoppage, there is a clear possessor"). Each later run of the *other* team becomes a change of
possession only if it lasts >= tau frames; a shorter run is re-assigned to the team in
possession (a touch/deflection without control). D frames and -1 are never changed.
"""
import numpy as np

from .canonical import A, D, H


def min_control(y, tau):
    y = np.asarray(y).copy()
    if tau <= 1:
        return y
    n = len(y)
    b = np.flatnonzero(np.diff(y) != 0) + 1
    starts, ends = np.r_[0, b], np.r_[b, n]
    cur = None
    for s, e in zip(starts, ends):
        lab = y[s]
        if lab not in (H, A):
            cur = None          # D or unknown: next alive run is a restart
            continue
        if cur is None or lab == cur:
            cur = lab
        elif e - s >= tau:
            cur = lab
        else:
            y[s:e] = cur
    return y
