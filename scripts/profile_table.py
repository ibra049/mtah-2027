"""Measured statistics per degradation profile for the technical document (external DFL data).
Seed 0, all 14 DFL halves per profile; clean DFL as reference. Ball statistics from
mtah.ball_audit (the code the target audit uses), player statistics from degrade.player_motion,
stillness from degrade.still_share, missing shares split by the clean alive/dead labels."""
import json
from pathlib import Path

import numpy as np

from mtah import ball_audit, degrade, io_idsse
from mtah.canonical import D, Sequence

EXT = Path.home() / "mtah-external" / "cache_deg"
MATCHES = sum(io_idsse.SPLITS.values(), [])
PROFILES = ["clean", "nominal", "stoppage", "harsh", "nominal_lj", "stoppage_lj", "nominal_sm", "stoppage_sm", "gapstress"]
COLS = ["ball_obs", "alive_miss", "dead_miss", "ball_ac1", "dv3_99", "player_ac1", "player_speed", "still", "tracklet_med_s"]


def stats(seq, y):
    a = ball_audit.audit(seq)
    pm = degrade.player_motion(seq)
    miss = np.isnan(seq.ball[:, 0])
    dead, alive = y == D, np.isin(y, (0, 1))
    return dict(ball_obs=1 - miss.mean(), alive_miss=miss[alive].mean(), dead_miss=miss[dead].mean(),
                ball_ac1=a["g4"]["ac1"], dv3_99=a["g3"]["dv3_99"], player_ac1=pm["player_ac1"],
                player_speed=pm["player_speed_med"], still=degrade.still_share(seq),
                tracklet_med_s=pm["tracklet_med_s"])


rows = {}
for prof in PROFILES:
    vals = []
    for m in MATCHES:
        for s in io_idsse.load(m):
            seq = s if prof == "clean" else Sequence.load(EXT / prof / "0" / f"{m}_{s.period}.npz")
            vals.append(stats(seq, s.y))
    rows[prof] = {k: (float(np.nanmedian([v[k] for v in vals])), float(np.nanmin([v[k] for v in vals])),
                      float(np.nanmax([v[k] for v in vals]))) for k in COLS}
    print(prof, " ".join(f"{k}={rows[prof][k][0]:.3g}" for k in COLS), flush=True)
Path("/private/tmp/claude-501/-Users-ibra-Desktop-mtah-2027-wt-external/fb78e130-6172-4496-b6e7-0d3d5f407bb4/scratchpad/profile_table.json").write_text(json.dumps(rows, indent=1))
