"""Step 13. The other ~800 usable MIMIC-III records, never looked at before this step.

Every MIMIC number in the deck comes from the first 400 usable MIMIC-III records in id order (a
compute-driven cut made before any result existed). The extraction in step 6 covers all ~1,200
usable records, so the remaining ones are a second, larger, untouched external set. Same model
(fitted on all of VitalDB), same scoring; pooled and per-record AUC.
Writes results/mimic_extra.json
"""
import os, sys, time, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib.build import VitalDB, load_vitaldb_generic, load_mimic, fit, score, target, evaluate

t0 = time.time()
VD = load_vitaldb_generic(VitalDB(tracker=False)); MS = fit(VD, list(VD), drop_nan_rows=True)
out = {}
for subset in ("paper", "extra"):
    MI = load_mimic("mimic3wdb", subset=subset); keys = sorted(MI)
    segs = [s for k in keys for s in MI[k]]; P = np.concatenate(score(MS, segs)); TRUE = np.concatenate([target(s) for s in segs])
    REC = np.concatenate([[s["key"]] * len(s["dev"]) for s in segs])
    d = dict(records=len(keys), stretches=len(segs), windows=int(len(P)))
    for thr in THRS:
        a, ci, pv, n = evaluate(MS, MI, keys, thr)
        aus = [roc_auc_score((TRUE[REC == r] > thr).astype(int), P[REC == r]) for r in keys if len(np.unique((TRUE[REC == r] > thr).astype(int))) > 1]
        d[f"n>{thr:g}"] = dict(auc=a, ci=list(ci), prevalence=pv, per_record_mean=float(np.mean(aus)), per_record_median=float(np.median(aus)),
                              evaluable=len(aus), frac_gt_0_5=float(np.mean(np.array(aus) > .5)), frac_gt_0_7=float(np.mean(np.array(aus) > .7)))
    out[subset] = d
    print(f"{subset:6s} {len(keys)} records, {len(P):,} windows: " + "  ".join(f"n>{t:g} {d[f'n>{t:g}']['auc']:.3f}[{d[f'n>{t:g}']['ci'][0]:.3f},{d[f'n>{t:g}']['ci'][1]:.3f}] per-record {d[f'n>{t:g}']['per_record_mean']:.3f}" for t in THRS), flush=True)
json.dump(out, open(f"{RES}/mimic_extra.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] -> results/mimic_extra.json")
