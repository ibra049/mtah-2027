import time
from mtah import io_idsse, degrade
cfg = degrade.load_cfg()
for mid in ["J03WPY"]:
    for s in io_idsse.load(mid):
        t = time.time(); d = degrade.degraded(s, cfg)
        print(mid, s.period, "clean", degrade.summary(s)); print(mid, s.period, "degr ", degrade.summary(d), f"{time.time()-t:.1f}s")
