"""DFL on-ball events -> ~/mtah-external/cache_events/<MATCH>_<P>.npy + timing report.
Alignment: EventTime -> clock frame; then the named player's slot; touch frame = strongest ball
velocity change while that player is within 1.5 m of the ball, in a +-2 s window around the half's
median clock offset. Report: offsets, share aligned, kick score at touch vs random control."""
import collections

import numpy as np

from mtah import io_idsse, io_idsse_events as E
from mtah.canonical import A, H

rng = np.random.default_rng(0)
by_kind = collections.defaultdict(lambda: {"ok": [], "dist": [], "dv": [], "resid": []})
ctrl = []
for mid in sum(io_idsse.SPLITS.values(), []):
    halves, offs = E.build(mid)
    seqs = {s.period: s for s in io_idsse.load(mid)}
    line = []
    for p, ev in halves.items():
        E.CACHE.mkdir(parents=True, exist_ok=True)
        np.save(E.path(mid, p), ev)
        s = seqs[p]
        k = E.kick_score(s.ball[:, :2])
        f0 = int(s.frame[0])
        for r in ev:
            b = by_kind[str(r["kind"])]
            b["ok"].append(r["aligned"]); b["dist"].append(r["player_dist"])
            b["dv"].append(k[int(r["frame"]) - f0])
            b["resid"].append(int(r["frame"] - r["clock_frame"]) - offs[p])
        alive = np.flatnonzero(np.isin(s.y, (H, A)))
        ctrl += list(k[rng.choice(alive, 500, replace=False)])
        line.append(f"P{p} clock offset {offs[p]/25:+.2f}s aligned {np.mean(ev['aligned']):.2f} n={len(ev)}")
    print(mid, " | ".join(line), flush=True)
print()
print(f"{'kind':20s} {'n':>5s} {'aligned':>8s} {'dist med':>9s} {'|resid|<=1s':>12s} {'dv med m/s':>11s}")
for kind, b in sorted(by_kind.items(), key=lambda kv: -len(kv[1]["ok"])):
    ok = np.array(b["ok"], bool)
    print(f"{kind:20s} {len(ok):5d} {ok.mean():8.2f} {np.nanmedian(b['dist']):9.2f} "
          f"{np.mean(np.abs(b['resid']) <= 25):12.2f} {np.median(np.array(b['dv'])[ok]) if ok.any() else float('nan'):11.1f}")
print(f"{'CONTROL random alive':20s} {len(ctrl):5d} {'':8s} {'':9s} {'':12s} {np.median(ctrl):11.1f}")
