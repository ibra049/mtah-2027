"""Where does the possession label change sit relative to the physical touch?
DFL: min_control(0.5 s) labels vs aligned DFL events (cache_events). Metrica: event-chain labels vs
Metrica events aligned to the ball track the same way (named player within 1.5 m, max velocity
change, +-1 s around the annotated frame). Offsets are event_frame - change_frame (frames, 25 fps):
negative = the touch happens before the label flips."""
import collections

import numpy as np

from mtah import io_idsse, io_idsse_events as E, io_metrica
from mtah.canonical import A, D, H
from mtah.relabel import min_control

W = 75
POSS_KINDS = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST", "CHALLENGE"}


def changes(y):
    """Indices c where y[c-1], y[c] are different teams (direct change, no D between)."""
    return [int(c) for c in np.flatnonzero((y[1:] != y[:-1]) & np.isin(y[1:], (H, A)) & np.isin(y[:-1], (H, A))) + 1]


def offsets(y, ef, et, ek):
    """Per change: (new_off, old_off, kind_of_new_event, gap old->new)."""
    out = []
    for c in changes(y):
        new, old = y[c], y[c - 1]
        win = np.abs(ef - c) <= W
        mn = win & (et == new)
        if not mn.any():
            continue
        jn = np.flatnonzero(mn)[np.argmin(np.abs(ef[mn] - c))]
        mo = win & (et == old) & (ef <= ef[jn])
        jo = np.flatnonzero(mo)[np.argmax(ef[mo])] if mo.any() else None
        out.append((int(ef[jn] - c), None if jo is None else int(ef[jo] - c), str(ek[jn]),
                    None if jo is None else int(ef[jn] - ef[jo])))
    return out


def report(name, rows):
    if not rows:
        print(f"{name:28s} n=0"); return
    nw = np.array([r[0] for r in rows]); od = np.array([r[1] for r in rows if r[1] is not None])
    gp = np.array([r[3] for r in rows if r[3] is not None])
    q = lambda a: "%+d [%+d, %+d]" % tuple(np.percentile(a, [50, 25, 75]))
    print(f"{name:28s} n={len(nw):4d} | NEW touch: median[IQR] {q(nw):16s} |off|<=3 {np.mean(np.abs(nw)<=3):.2f} <=8 {np.mean(np.abs(nw)<=8):.2f}"
          f" | OLD last touch: {q(od) if len(od) else '-':16s} | old->new gap median {np.median(gp) if len(gp) else float('nan'):.0f} fr")


def group(kind, src):
    if src == "dfl":
        return {"duel": "duel (tackle)", "claim": "claim/interception"}.get(kind, "other (loose/recovery)")
    if kind == "RECOVERY:INTERCEPTION":
        return "claim/interception"
    if kind == "RECOVERY:THEFT" or kind.startswith("CHALLENGE"):
        return "duel (tackle)"
    return "other (loose/recovery)"


# ---------------------------------------------------------------- DFL
dfl = collections.defaultdict(list)
for mid in ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]:
    for s in io_idsse.load(mid):
        ev = E.load(mid, s.period)
        ev = ev[(ev["aligned"] == 1) & (ev["team"] >= 0)]
        y = min_control(s.y, 12)
        for r in offsets(y, ev["frame"] - s.frame[0], ev["team"], ev["kind"]):
            dfl["ALL"].append(r); dfl[group(r[2], "dfl")].append(r)
print("== DFL, labels min_control(0.5 s), touches = aligned DFL events")
for k in ("ALL", "duel (tackle)", "claim/interception", "other (loose/recovery)"):
    report(k, dfl[k])

# ---------------------------------------------------------------- Metrica
met = collections.defaultdict(list)
align_stats = []
for g in io_metrica.GAMES:
    seqs = {s.period: s for s in io_metrica.load(g)}
    ev = io_metrica.load_events_with_player(g)
    slots = io_metrica.player_slots(g)
    for p, s in seqs.items():
        k = E.kick_score(s.ball[:, :2])
        e = ev[(ev["period"] == p) & ev["type"].isin(POSS_KINDS)]
        ef, et, ek = [], [], []
        f0 = int(s.frame[0])
        for _, r in e.iterrows():
            i = int(r["start"]) - f0
            if r["player"] not in slots or not 0 <= i < len(s):
                continue
            side, j = slots[r["player"]]
            pl = (s.home if side == "home" else s.away)[:, j]
            lo, hi = max(0, i - 25), min(len(s), i + 26)
            d = np.linalg.norm(pl[lo:hi] - s.ball[lo:hi, :2], axis=1)
            near = d < 1.5
            ok = bool(np.any(near))
            align_stats.append(ok)
            if not ok:
                continue
            a = lo + int(np.argmax(np.where(near, k[lo:hi], -1)))
            ef.append(a); et.append(H if r["team"] == "H" else A)
            ek.append(r["type"] + (":" + r["subtype"] if r["subtype"] else ""))
        for r in offsets(s.y, np.array(ef), np.array(et), np.array(ek)):
            met["ALL"].append(r); met[group(r[2], "met")].append(r)
print(f"\n== Metrica, event-chain labels, touches = Metrica events aligned to the ball track (aligned share {np.mean(align_stats):.2f})")
for k in ("ALL", "duel (tackle)", "claim/interception", "other (loose/recovery)"):
    report(k, met[k])
