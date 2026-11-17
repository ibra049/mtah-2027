"""Synthetic CONTRACT-conformant Sequences for unit tests (no real data involved).

A scripted match: alive spells alternate between the teams, the ball sits next to a carrier of
the team in possession and travels between carriers; in dead stretches it lies still beyond a
touchline. Then target-like damage: ball dropout in bursts, players scattered over anonymous
tracklet slots with gaps.
"""
import numpy as np

from mtah.canonical import A, D, FPS, H, Sequence

L, W = 105.0, 68.0


def _smooth_noise(rng, T, n, scale, k=25):
    e = rng.normal(0, 1, (T + k, n, 2))
    c = np.cumsum(e, 0)
    return ((c[k:] - c[:-k]) / np.sqrt(k) * scale).astype(np.float32)


def make_seq(T=1500, seed=0, home_attacks_plus=True, ball_missing=0.4, slots=14,
             player_missing=0.07, match_id="SYN", period=1):
    rng = np.random.default_rng(seed)
    sgn = 1.0 if home_attacks_plus else -1.0
    # labels: D, then alternating possession spells with occasional stoppages
    y = np.empty(T, np.int8)
    t, team = 0, int(rng.integers(0, 2))
    while t < T:
        if t == 0 or rng.random() < 0.3:
            n = int(rng.integers(2 * FPS, 6 * FPS))
            y[t:t + n] = D
            t += n
            team = int(rng.integers(0, 2))
        else:
            team = 1 - team
        n = int(rng.integers(3 * FPS, 10 * FPS))
        y[t:t + n] = team
        t += n
    # players: 11 per team around a formation in their own half, drifting
    ax = np.array([-48, -32, -32, -32, -32, -18, -18, -18, -6, -6, -2], np.float32)
    ay = np.array([0, -24, -8, 8, 24, -16, 0, 16, -10, 10, 0], np.float32)
    anchor = np.stack([ax, ay], 1)
    home = anchor * [sgn, 1] + _smooth_noise(rng, T, 11, 2.5)
    away = anchor * [-sgn, 1] + _smooth_noise(rng, T, 11, 2.5)
    home, away = home.astype(np.float32), away.astype(np.float32)
    # the block follows the team in possession toward the opponent's goal
    push = np.where(y == H, 12.0, np.where(y == A, -12.0, 0.0)) * sgn
    k = min(50, T)
    push = np.convolve(push, np.ones(k) / k, mode="same").astype(np.float32)
    home[..., 0] += push[:, None]
    away[..., 0] += push[:, None]
    # ball: next to a carrier of the possessing team; carriers change every 1-3 s
    ball = np.full((T, 3), np.nan, np.float32)
    t = 0
    while t < T:
        n = min(int(rng.integers(FPS, 3 * FPS)), T - t)
        if y[t] == D:
            ball[t:t + n, 0] = rng.uniform(-40, 40)
            ball[t:t + n, 1] = (W / 2 + 1.0) * rng.choice([-1, 1])
        else:
            carrier = int(rng.integers(1, 11))
            src = home if y[t] == H else away
            ball[t:t + n, :2] = src[t:t + n, carrier] + rng.normal(0, 0.3, 2)
        t += n
    ball[:, 2] = 0.0
    # damage: ball bursts
    miss = np.zeros(T, bool)
    t = 0
    while t < T and ball_missing > 0:
        on = int(rng.geometric(ball_missing / (1 - ball_missing) / 30))
        off = int(rng.geometric(1 / 30))
        miss[t + on:t + on + off] = True
        t += on + off
    ball[miss] = np.nan
    seq = Sequence(match_id, period, np.arange(T, dtype=np.int64) + 1000, ball,
                   _to_slots(home, rng, slots, player_missing),
                   _to_slots(away, rng, slots, player_missing),
                   np.array([L, W], np.float32), y)
    return seq


def _to_slots(pos, rng, K, missing):
    """[T,P,2] persons -> [T,K,2] anonymous tracklet slots with gaps."""
    T, P, _ = pos.shape
    out = np.full((T, K, 2), np.nan, np.float32)
    free_at = np.zeros(K, np.int64)
    starts = [(0, p) for p in range(P)]
    while starts:
        starts.sort()
        s, p = starts.pop(0)
        if s >= T:
            continue
        n = int(rng.geometric(1 / 400)) if missing > 0 else T
        e = min(T, s + n)
        free = np.flatnonzero(free_at <= s)
        if len(free):
            k = int(rng.choice(free))
            out[s:e, k] = pos[s:e, p]
            free_at[k] = e + 5
        gap = int(rng.geometric(1 / max(400 * missing / (1 - missing), 1.0))) if missing > 0 else 0
        starts.append((e + gap, p))
    return out
