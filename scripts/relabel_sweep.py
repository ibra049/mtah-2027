"""Sensitivity of DFL label structure to the min_control threshold (information only)."""
import numpy as np
from mtah import io_idsse
from mtah.relabel import min_control
from mtah.evaluate import label_stats, report

mids = sum(io_idsse.SPLITS.values(), [])
data = {m: [s.y for s in io_idsse.load(m)] for m in mids}
print(f"{'tau_s':>6}{'spells':>9}{'mean_s':>9}{'med_s':>8}{'first5min':>10}{'turn/min':>9}{'frames_changed':>15}{'spellF1_vs_raw':>15}")
for tau_s in (0, 0.5, 1.0, 1.5, 2.0, 3.0):
    tau = int(round(tau_s * 25))
    st, ch, f1 = [], 0, []
    tot = 0
    for m, ys in data.items():
        new = [min_control(y, tau) for y in ys]
        st.append(label_stats(new)); ch += sum(int((a != b).sum()) for a, b in zip(ys, new)); tot += sum(len(a) for a in ys)
        f1.append(report(ys, new)["spell_f1_mean"])
    g = lambda k: np.mean([s[k] for s in st])
    print(f"{tau_s:>6}{g('n_spells'):>9.1f}{g('mean_spell_s'):>9.2f}{g('median_spell_s'):>8.2f}{g('spells_first_5min'):>10.1f}{g('turnovers_per_alive_min'):>9.2f}{ch/tot:>15.4f}{np.mean(f1):>15.4f}")
