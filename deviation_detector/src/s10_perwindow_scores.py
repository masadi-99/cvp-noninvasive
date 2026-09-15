"""Step 10 (side quest). The RAW per-window score, without the running maximum.

The deployed score is a running max: causal, but it never comes back down. This writes, for
every cohort, the raw per-window regressor output and the signed model's output next to the
running-max version and the truth, so a score that can fall again when the patient does can be
studied later. Nothing else changes: same features, same fits on VitalDB.
Writes results/perwindow_<cohort>.npz  (rec, stretch, t, since, score, score_signed, score_max, dev, sdev)
"""
import os, sys, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES
from lib.build import VitalDB, load_vitaldb_generic, load_mimic, fit, score

t0 = time.time()
VD = load_vitaldb_generic(VitalDB(tracker=False))
MS = fit(VD, list(VD), drop_nan_rows=True)
SG = fit(VD, list(VD), drop_nan_rows=True, target_fn=lambda s: s["sdev"])


def dump(segs, tag):
    if not segs: return
    raw = score(MS, segs, running_max=False); mx = score(MS, segs); sg = score(SG, segs, running_max=False)
    np.savez_compressed(f"{RES}/perwindow_{tag}.npz",
                        rec=np.concatenate([[s["key"]] * len(s["t"]) for s in segs]),
                        stretch=np.concatenate([[s["k"]] * len(s["t"]) for s in segs]),
                        t=np.concatenate([s["t"] for s in segs]), since=np.concatenate([s["since"] for s in segs]),
                        score=np.concatenate(raw), score_signed=np.concatenate(sg), score_max=np.concatenate(mx),
                        dev=np.concatenate([s["dev"] for s in segs]), sdev=np.concatenate([s["sdev"] for s in segs]))
    print(f"  {tag}: {sum(len(s['t']) for s in segs):,} windows, {len(segs)} stretches -> results/perwindow_{tag}.npz", flush=True)


dump([s for v in VD.values() for s in v], "vitaldb")
for src in ("mimic3wdb", "mimic4wdb"):
    MI = load_mimic(src); dump([s for k in sorted(MI) for s in MI[k]], src)
print(f"[{time.time()-t0:.0f}s] done")
