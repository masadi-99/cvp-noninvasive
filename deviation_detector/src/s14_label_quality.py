"""Step 14. Label quality on MIMIC: flat plateaus, and what every arm scores without them.

The best-scoring MIMIC records contain long spans where the CVP waveform median is constant to
the decimal for many minutes at 24-37 mmHg: a transducer that is clamped, closed or being
flushed, not a patient. Those windows are 'stable' by the label and the model scores them low,
so they inflate the AUC for free. Windows inside a run of >= k identical consecutive label values
are removed and every MIMIC number is recomputed on the same stored scores: the zero-shot model,
the no-model baselines, the no-ECG/PPG model, the ECG/PPG-only model, and the MIMIC-only ceiling.
The same test is run on the VitalDB out-of-fold scores.
Writes results/label_quality.json
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib.build import load_mimic, VitalDB, build_vitaldb, target


def flat_mask(devs, kmin):
    """Windows inside a run of >= kmin consecutive identical |dev| values, per stretch."""
    out = []
    for d in devs:
        d = np.round(d, 3); f = np.zeros(len(d), bool); run = 1; start = 0
        for j in range(1, len(d) + 1):
            if j < len(d) and d[j] == d[j - 1]: run += 1; continue
            if run >= kmin: f[start:j] = True
            start = j; run = 1
        out.append(f)
    return np.concatenate(out)


def per_record(P, Y, G, keep):
    a = []
    for r in np.unique(G):
        m = (G == r) & keep
        if m.sum() > 20 and Y[m].min() != Y[m].max(): a.append(roc_auc_score(Y[m], P[m]))
    return float(np.mean(a)), len(a)


MI = load_mimic("mimic3wdb"); m3 = sorted(MI); segs = [s for k in m3 for s in MI[k]]
Z = np.load(f"{RES}/mimic_scores.npz", allow_pickle=True); assert len(Z["P"]) == sum(len(s["dev"]) for s in segs)
S = np.load(f"{RES}/investigation_scores.npz", allow_pickle=True); T = np.load(f"{RES}/transfer_scores.npz", allow_pickle=True)
TRUE, G = Z["TRUE"], Z["REC"]
ARMS = {"zero-shot full model": Z["P"], "clock (minutes since reading)": S["mi__base:clock: minutes since last CVP reading"],
        "block spread x since (no model)": S["mi__base:block spread: std x since"],
        "learned, no ECG/PPG (32 cols)": S["mi__abl:- all ECG/PPG (91)"], "ECG/PPG + time only (92 cols)": S["mi__abl:ECG/PPG + time only (92)"]}
# the MIMIC-only ceiling covers MIMIC-III + IV in sorted-key order; restrict it to the MIMIC-III part
MIall = load_mimic(); keysB = sorted(MIall); segB = [s for k in keysB for s in MIall[k]]
isB3 = np.concatenate([[k.startswith("mimic3")] * len(s["dev"]) for k in keysB for s in MIall[k]])
assert (T["B__G"][isB3] == G).all()
ARMS["MIMIC-only ceiling (5-fold)"] = T["B__P"][isB3]
out = {}
for kmin in (5, 10):
    flat = flat_mask([s["dev"] for s in segs], kmin); keep = ~flat
    d = dict(flat_fraction=float(flat.mean()), records_with_flat=int(len(np.unique(G[flat]))), arms={})
    print(f"\nflat runs >= {kmin} identical windows: {100*flat.mean():.1f}% of MIMIC-III windows, in {d['records_with_flat']} of {len(m3)} records")
    for nm, P in ARMS.items():
        d["arms"][nm] = {}
        for thr in THRS:
            Y = (TRUE > thr).astype(int)
            pa, na = per_record(P, Y, G, np.ones(len(P), bool)); pk, nk = per_record(P, Y, G, keep)
            d["arms"][nm][f"n>{thr:g}"] = dict(pooled_all=float(roc_auc_score(Y, P)), pooled_without_flat=float(roc_auc_score(Y[keep], P[keep])),
                                              per_record_all=pa, per_record_without_flat=pk, n_records=nk,
                                              prevalence_all=float(Y.mean()), prevalence_without_flat=float(Y[keep].mean()))
        r = d["arms"][nm]
        print(f"  {nm:34s} " + "  ".join(f"{k} {r[k]['pooled_all']:.3f}->{r[k]['pooled_without_flat']:.3f} (rec {r[k]['per_record_all']:.3f}->{r[k]['per_record_without_flat']:.3f})" for k in r))
    out[f"mimic_run>={kmin}"] = d
# VitalDB: the same test on the out-of-fold scores
V = VitalDB(); R = build_vitaldb(V); segV = [s for c in sorted(R) for s in R[c]]
Pv = S["vd__full"]; Gv = S["G"]; TV = np.concatenate([target(s) for s in segV])
flat = flat_mask([s["dev"] for s in segV], 5)
d = dict(flat_fraction=float(flat.mean()), arms={"out-of-fold full model": {}})
for thr in THRS:
    Y = (TV > thr).astype(int); keep = ~flat
    d["arms"]["out-of-fold full model"][f"n>{thr:g}"] = dict(pooled_all=float(roc_auc_score(Y, Pv)), pooled_without_flat=float(roc_auc_score(Y[keep], Pv[keep])) if keep.sum() < len(keep) else float(roc_auc_score(Y, Pv)))
out["vitaldb_run>=5"] = d
print(f"\nVitalDB: {100*flat.mean():.2f}% of windows in flat runs >= 5")
json.dump(out, open(f"{RES}/label_quality.json", "w"), indent=1); print("-> results/label_quality.json")
