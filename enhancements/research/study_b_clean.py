"""Clean controlled diagnostic for the per-window result. Two questions:
  (1) Does using MORE windows (aggregated properly) help?  -> convergence curve.
  (2) Is the per-window-training collapse a BUG or the within-window feature noise?
      -> replicated-median control: per-window rows whose features are the patient's MEDIAN.
         If the pipeline is sound this MUST recover ~0.756; if it gives ~0.53 there is a bug.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import enh

THR = 12.0; FE = enh.FEATS5
Xc, yc, cases = enh.per_case(FE); ybc = (yc > THR).astype(int)
d = enh.load_merged(); cid = d["cid"]; cset = set(cases.tolist())
wm = np.array([c in cset for c in cid]); wcid = cid[wm]
Xw = np.column_stack([d["F"][f][wm] for f in FE]); wnum = d["numeric"][wm]
cidx = {c: i for i, c in enumerate(cases)}
rows_by_case = {c: np.where(wcid == c)[0] for c in cases}
medfeat = {c: np.nanmedian(Xw[rows_by_case[c]], axis=0) for c in cases}   # each patient's median feature vector

def ens(): return [Pipeline([("im",SimpleImputer()),("sc",StandardScaler()),("m",Ridge(alpha=enh.RIDGE_ALPHA))]),
                   Pipeline([("im",SimpleImputer()),("m",HistGradientBoostingRegressor(**enh.HGB))])]
def ridge(): return [Pipeline([("im",SimpleImputer()),("sc",StandardScaler()),("m",Ridge(alpha=enh.RIDGE_ALPHA))])]

print("baseline per-case median:", round(enh.evaluate(Xc, yc, cases, threshold=THR, reps=20)["auc"], 3), flush=True)

# ---- (1) CONVERGENCE: median of K random windows per patient ----------------
print("\n(1) aggregate K random windows per patient (median) -> does more help?", flush=True)
for K in [1, 2, 3, 5, 8, 12, 9999]:
    a = []
    for s in range(6):
        rng = np.random.default_rng(300 + s)
        X = np.full((len(cases), len(FE)), np.nan)
        for i, c in enumerate(cases):
            idx = rows_by_case[c]
            pick = idx if (K >= len(idx)) else rng.choice(idx, K, replace=False)
            X[i] = np.nanmedian(Xw[pick], axis=0)
        a.append(enh.evaluate(X, yc, cases, threshold=THR, reps=8)["auc"])
    lab = "ALL" if K == 9999 else str(K)
    print(f"   K={lab:>4} windows/patient: AUC = {np.mean(a):.3f}", flush=True)

# ---- per-window trainer ------------------------------------------------------
def perwindow_caseauc(Xrows, label="casemed", models=ens, reps=10):
    y_w = np.array([yc[cidx[c]] for c in wcid]) if label == "casemed" else wnum
    aucs = []
    for r in range(reps):
        rng = np.random.default_rng(400 + r); u = cases.copy(); rng.shuffle(u)
        fo = {c: i % 5 for i, c in enumerate(u)}
        wfold = np.array([fo[c] for c in wcid]); cfold = np.array([fo[c] for c in cases])
        coof = np.full(len(cases), np.nan)
        for f in range(5):
            m = (wfold != f) & np.isfinite(y_w)
            mods = [mm.fit(Xrows[m], y_w[m]) for mm in models()]
            for c in np.where(cfold == f)[0]:
                wp = np.mean([mm.predict(Xrows[wcid == cases[c]]) for mm in mods], 0)
                coof[c] = np.nanmean(wp)
        ok = np.isfinite(coof); aucs.append(roc_auc_score(ybc[ok], coof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs))

# ---- (2) replicated-median CONTROL vs actual per-window ----------------------
Xw_rep = np.array([medfeat[c] for c in wcid])     # each window carries its patient's MEDIAN features
print("\n(2) per-window TRAINING — control (replicated median) vs actual (real window features):", flush=True)
a, s = perwindow_caseauc(Xw_rep, "casemed", ens);   print(f"   CONTROL replicated-median, ensemble: CASE AUC = {a:.3f} ±{s:.3f}  (must be ~0.756 if pipeline is sound)", flush=True)
a, s = perwindow_caseauc(Xw_rep, "casemed", ridge); print(f"   CONTROL replicated-median, RIDGE   : CASE AUC = {a:.3f} ±{s:.3f}", flush=True)
a, s = perwindow_caseauc(Xw,     "casemed", ens);   print(f"   ACTUAL  real-window-features, ens  : CASE AUC = {a:.3f} ±{s:.3f}", flush=True)
a, s = perwindow_caseauc(Xw,     "casemed", ridge); print(f"   ACTUAL  real-window-features, RIDGE: CASE AUC = {a:.3f} ±{s:.3f}", flush=True)

# how noisy is a single window vs the median? per-feature within/between
print("\n(3) per-window feature reliability (how noisy is ONE window vs the patient median):", flush=True)
for j, f in enumerate(FE):
    wn = np.nanmean([np.nanstd(Xw[rows_by_case[c], j]) for c in cases])
    bt = np.nanstd([medfeat[c][j] for c in cases])
    print(f"   {f:14} within-patient std={wn:.3g}  between-patient std={bt:.3g}  reliability={bt**2/(bt**2+wn**2):.2f}", flush=True)
print("\nDONE", flush=True)
