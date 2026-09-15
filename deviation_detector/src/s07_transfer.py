"""Step 7. Does the VitalDB-trained detector work on MIMIC ICU records? Three arms.

  A  ZERO-SHOT   train on VitalDB only, test on MIMIC-III and MIMIC-IV. The external-validity number.
  B  MIMIC-ONLY  train and test within MIMIC, five-fold by record. The ceiling for this cohort:
                 it says whether a shortfall in A is a transfer failure or a harder cohort.
  C  POOLED      train on VitalDB + half of the MIMIC records, test on the other half; compared
                 with that same half trained alone, on the identical split.

Both cohorts go through the record builder (baseline and block shape from the gated block, no
tracker columns), so the feature semantics are identical on both sides. Rows with a NaN feature
(a patient's first stretch has no block-to-block history) are dropped from training.
Writes results/transfer.json
"""
import os, sys, time, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib.build import VitalDB, load_vitaldb_generic, load_mimic, fit, score, target, evaluate, boot_ci

t0 = time.time()
MI = load_mimic()
print(f"MIMIC: {len(MI)} records, {sum(len(v) for v in MI.values())} stretches, "
      f"{sum(len(s['dev']) for v in MI.values() for s in v):,} windows [{time.time()-t0:.0f}s]", flush=True)
VD = load_vitaldb_generic(VitalDB(tracker=False))
print(f"VitalDB: {len(VD)} patients, {sum(len(v) for v in VD.values())} stretches, "
      f"{sum(len(s['dev']) for v in VD.values() for s in v):,} windows [{time.time()-t0:.0f}s]", flush=True)
m3 = [k for k in MI if k.startswith("mimic3")]; m4 = [k for k in MI if k.startswith("mimic4")]
print(f"  MIMIC-III records {len(m3)}   MIMIC-IV records {len(m4)}\n", flush=True)
out = dict(counts=dict(mimic_records=len(MI), mimic_stretches=sum(len(v) for v in MI.values()),
                       mimic_windows=int(sum(len(s['dev']) for v in MI.values() for s in v)),
                       vitaldb_patients=len(VD), vitaldb_stretches=sum(len(v) for v in VD.values()),
                       vitaldb_windows=int(sum(len(s['dev']) for v in VD.values() for s in v)),
                       mimic3=len(m3), mimic4=len(m4)), arms={})


SAVE = {}
def keep_scores(nm, MS, R, keys):
    segs = [s for k in keys for s in R[k]]
    SAVE[nm] = dict(P=np.concatenate(score(MS, segs)), T=np.concatenate([target(s) for s in segs]),
                    G=np.concatenate([[s["key"]] * len(s["dev"]) for s in segs]))


def row(nm, res):
    out["arms"][nm] = {f"n>{th:g}": dict(auc=a, ci=list(ci), prevalence=pv, n=n)
                       for th, (a, ci, pv, n) in zip(THRS, res)}
    print(f"{nm:34s} " + "  ".join(f"{a:.3f}[{lo:.3f},{hi:.3f}]" for a, (lo, hi), pv, n in res), flush=True)


MS = fit(VD, list(VD), drop_nan_rows=True)
row("A zero-shot VitalDB->MIMIC-III", [evaluate(MS, MI, m3, th) for th in THRS]); keep_scores("A3", MS, MI, m3)
if m4: row("A zero-shot VitalDB->MIMIC-IV", [evaluate(MS, MI, m4, th) for th in THRS]); keep_scores("A4", MS, MI, m4)
row("  (same model on VitalDB, in-sample)", [evaluate(MS, VD, list(VD), th) for th in THRS]); keep_scores("Ain", MS, VD, list(VD))

keys = sorted(MI); rng = np.random.default_rng(11); ids = np.array(keys); rng.shuffle(ids)
FOLD = np.array_split(ids, 5)
PF = {}
for te in FOLD:
    tr = [k for k in keys if k not in set(te.tolist())]
    ms = fit(MI, tr, nseed=2, drop_nan_rows=True)
    segs = [s for k in te.tolist() for s in MI[k]]
    for s, p in zip(segs, score(ms, segs)): PF[id(s)] = p
res = []
segs = [s for k in keys for s in MI[k]]
P = np.concatenate([PF[id(s)] for s in segs]); G = np.concatenate([[s["key"]] * len(s["dev"]) for s in segs])
SAVE["B"] = dict(P=P, T=np.concatenate([target(s) for s in segs]), G=G)
for th in THRS:
    Y = np.concatenate([(target(s) > th).astype(int) for s in segs])
    res.append((roc_auc_score(Y, P), boot_ci(P, Y, G), float(Y.mean()), len(Y)))
row("B MIMIC-only ceiling (5-fold)", res)

half = len(keys) // 2; trK, teK = keys[:half], keys[half:]
POOL = dict(VD); POOL.update({k: MI[k] for k in trK})
msC = fit(POOL, list(POOL), drop_nan_rows=True)
row("C pooled VitalDB+MIMIC -> MIMIC", [evaluate(msC, MI, teK, th) for th in THRS]); keep_scores("C", msC, MI, teK)
msH = fit(MI, trK, drop_nan_rows=True)
row("  (MIMIC-half only, same split)", [evaluate(msH, MI, teK, th) for th in THRS]); keep_scores("Chalf", msH, MI, teK)
json.dump(out, open(f"{RES}/transfer.json", "w"), indent=1)
np.savez_compressed(f"{RES}/transfer_scores.npz", **{f"{k}__{q}": v for k, d in SAVE.items() for q, v in d.items()})
print(f"\n[{time.time()-t0:.0f}s]  columns are n>2, n>3, n>4  -> results/transfer.json")
