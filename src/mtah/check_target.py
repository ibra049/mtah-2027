"""Parse assertions on the target, against documented aggregates only (DATA.md).

    scripts/run_target.sh .venv/bin/python -m mtah.check_target

Prints only SAFE| lines: counts, fractions, pass/fail. Never per-frame values. Nothing here is
used to choose or tune anything; it checks that the parse means what the documentation says.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from mtah import io_target
from mtah.submit import graded_mask


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def f4(x):
    return f"{float(x):.4f}"


def main(data_dir=None):
    t0 = time.time()
    data_dir = data_dir or os.environ["MTAH_DATA"]
    cfg = io_target.load_config()
    doc = cfg["documented"]
    from mtah.baseline import provenance
    provenance()
    seqs, info = io_target.load(data_dir, cfg, dtype=np.float32, return_meta=True,
                                coverage="full")
    meta, seen, st, tags = info["meta"], info["seen"], info["stats"], info["tags"]
    n = cfg["n_frames"]
    checks = {}

    # --- layout / metadata
    lay = meta.layouts
    say(n_layouts=len(lay), n_slots=len(meta.slot_player),
        n_home_slots=len(meta.home_slots), n_away_slots=len(meta.away_slots),
        group_fields=",".join(str(v) for v in sorted(set(np.concatenate([l.group_nf for l in lay]).tolist()))),
        ball_fields=lay[0].ball_nf, ball_has_z=int(lay[0].ball_z >= 0))
    spec_order = int(all(np.array_equal(l.group_slot, np.arange(len(meta.slot_player))) for l in lay))
    say(spec_order_equals_player_order=spec_order, spec_frame_min=lay[0].start,
        spec_frame_max=lay[-1].end, xml_field_size=f"{meta.field_size[0]:g}x{meta.field_size[1]:g}")
    home_ok = int(cfg["home_name_contains"] in meta.team_names[0].lower())
    say(home_is_barcelona=home_ok, n_teams=len(meta.team_names))
    checks["home"] = home_ok
    checks["slots"] = int(len(meta.slot_player) == doc["n_slots"])

    # --- frames
    frames = np.concatenate([s.frame for s in seqs])
    contiguous = int(np.array_equal(frames, np.arange(n)))
    say(lines=st["lines"], frames_seen=int(seen.sum()), n_frames=n, contiguous=contiguous,
        all_seen=int(seen.all()), malformed=st["malformed"], out_of_range=st["out_of_range"],
        duplicate=st["duplicate"], no_layout=st["no_layout"], odd_groups=st["odd_groups"],
        odd_tokens=st["odd_tokens"], odd_ball=st["odd_ball"])
    checks["frames"] = int(contiguous and seen.all() and st["lines"] == n and st["malformed"] == 0)

    # --- the default (graded) load: exactly the two published blocks as periods 1 and 2
    graded = io_target.load(data_dir, cfg, dtype=np.float32)
    blocks_ok = int(len(graded) == 2 and [s.period for s in graded] == [1, 2] and all(
        s.frame[0] == a and s.frame[-1] == b and np.array_equal(s.frame, np.arange(a, b + 1))
        for s, (a, b) in zip(graded, cfg["graded_blocks"])))
    same = int(all(np.array_equal(np.isnan(s.ball), np.isnan(seqs[0].ball[s.frame])) and
                   np.array_equal(np.isnan(s.home), np.isnan(seqs[0].home[s.frame]))
                   for s in graded))
    slots_extra = int(all("home_slot" in s.extra and "away_slot" in s.extra for s in graded))
    say(graded_periods=len(graded), graded_lengths=",".join(str(len(s)) for s in graded),
        graded_blocks_ok=blocks_ok, graded_matches_full=same, slot_extra=slots_extra,
        tags_per_period=",".join(str(0 if s.tags is None else len(s.tags)) for s in graded))
    checks["graded_load"] = int(blocks_ok and same and slots_extra)
    del graded

    # --- coordinates within pitch bounds (metres about the centre)
    L, W = cfg["pitch"]
    for name in ("home", "away", "ball"):
        arr = np.concatenate([getattr(s, name)[..., :2] for s in seqs]).reshape(-1, 2)
        fin = np.isfinite(arr).all(1)
        v = arr[fin]
        inside = (np.abs(v[:, 0]) <= L / 2) & (np.abs(v[:, 1]) <= W / 2)
        inside5 = (np.abs(v[:, 0]) <= L / 2 + 5) & (np.abs(v[:, 1]) <= W / 2 + 5)
        say(obj=name, n_finite=int(fin.sum()), frac_in_pitch=f4(inside.mean()),
            frac_in_pitch_5m=f4(inside5.mean()))
        checks[f"bounds_{name}"] = int(inside5.mean() > 0.98)

    # --- ball presence (documented 113,488 all / 94,214 graded)
    ball_ok = np.concatenate([np.isfinite(s.ball[:, :2]).all(1) for s in seqs])
    g = graded_mask(n, cfg["graded_blocks"])
    n_all, n_gr = int(ball_ok.sum()), int(ball_ok[g].sum())
    say(ball_present_all=n_all, ball_present_graded=n_gr, n_graded=int(g.sum()),
        ball_missing_frac_graded=f4(1 - n_gr / g.sum()), ball_missing_frac_all=f4(1 - n_all / n))
    checks["ball"] = int(n_all == doc["n_ball_present_all"] and n_gr == doc["n_ball_present_graded"])

    # --- tracked players during graded play (documented mean 20.5)
    cnt = np.concatenate([np.isfinite(s.home[..., 0]).sum(1) + np.isfinite(s.away[..., 0]).sum(1)
                          for s in seqs])
    say(mean_tracked_graded=f"{cnt[g].mean():.2f}", mean_tracked_ungraded=f"{cnt[~g].mean():.2f}")

    # --- tags (documented 638: 314 H / 288 A / 36 N)
    if tags is None:
        say(tags_loaded=0)
        checks["tags"] = 0
    else:
        nt = {k: int((tags["team"] == k).sum()) for k in ("H", "A", "N", "?")}
        say(n_tags=len(tags), tags_H=nt["H"], tags_A=nt["A"], tags_N=nt["N"], tags_unknown=nt["?"],
            n_codes=len(set(tags["code"].tolist())), n_typed=int((tags["type"] != "").sum()),
            tags_in_range=int(((tags["start"] >= 0) & (tags["end"] <= n) &
                               (tags["end"] >= tags["start"])).all()))
        checks["tags"] = int(len(tags) == doc["n_tags"] and nt["H"] == doc["n_tags_H"]
                             and nt["A"] == doc["n_tags_A"] and nt["N"] == doc["n_tags_N"])

    # --- kickoff tags: found per half, and contains the published kickoff frame (check only)
    if tags is not None:
        win = io_target.kickoff_windows(tags, cfg)
        pub = cfg["documented"]["published_kickoff_frames"]
        found = {p: int(win[p] is not None) for p in (1, 2)}
        contains = {p: int(win[p] is not None and win[p][0] <= pub[str(p)] <= win[p][1])
                    for p in (1, 2)}
        say(kickoff_tag_found_h1=found[1], kickoff_tag_found_h2=found[2],
            kickoff_tag_contains_published_h1=contains[1],
            kickoff_tag_contains_published_h2=contains[2])
        checks["kickoff"] = int(all(found.values()) and all(contains.values()))

    # --- teams on opposite sides per half, swapping at halftime (sign only)
    signs = []
    for (a, b) in cfg["graded_blocks"]:
        mh = np.nanmean(np.concatenate([s.home[np.isin(s.frame, np.arange(a, b + 1)), :, 0].ravel()
                                        for s in seqs]))
        ma = np.nanmean(np.concatenate([s.away[np.isin(s.frame, np.arange(a, b + 1)), :, 0].ravel()
                                        for s in seqs]))
        signs.append((np.sign(mh), np.sign(ma)))
    opp = [int(h * a < 0) for h, a in signs]
    swap = int(signs[0][0] == -signs[1][0] and signs[0][1] == -signs[1][1])
    say(opposite_sides_h1=opp[0], opposite_sides_h2=opp[1], sides_swap_at_halftime=swap)
    checks["sides"] = int(all(opp) and swap)

    ok = all(checks.values())
    say(**{f"pass_{k}": v for k, v in checks.items()})
    say(all_pass=int(ok), seconds=f"{time.time() - t0:.1f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
