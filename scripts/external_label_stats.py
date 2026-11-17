"""Per-match label stats for the IDSSE matches (clean DFL labels). Compare with the published
target facts: public window = 17 possessions in 5 min, mean 14.2 s (DATA.md)."""
import numpy as np
from mtah import io_idsse
from mtah.evaluate import label_stats, print_table

rows = []
for split, mids in io_idsse.SPLITS.items():
    for mid in mids:
        seqs = io_idsse.load(mid)
        rows.append((mid, label_stats([s.y for s in seqs])))
rows.append(("MEAN", {k: float(np.mean([r[k] for _, r in rows])) for k in rows[0][1]}))
print_table(rows, list(rows[0][1]))
