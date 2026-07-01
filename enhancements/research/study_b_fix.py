"""Confirm the decomposition: weight each patient equally (sample_weight = 1/n_windows) in per-window
training. If window-count weighting is the cause of the replicated-median control's drop, balancing
must recover ~0.756 there; with real features it recovers only the weighting part (feature noise stays)."""
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
Xw = np.column_stack([d["F"][f][wm] for f in FE]); cidx = {c: i for i, c in enumerate(cases)}
rows_by_case = {c: np.where(wcid == c)[0] for c in cases}
medfeat = {c: np.nanmedian(Xw[rows_by_case[c]], axis=0) for c in cases}
ncnt = {c: len(rows_by_case[c]) for c in cases}
Xw_rep = np.array([medfeat[c] for c in wcid])
y_w = np.array([yc[cidx[c]] for c in wcid])
w_bal = np.array([1.0 / ncnt[c] for c in wcid])      # equal weight per patient

def ens(): return [("rdg", Pipeline([("im",SimpleImputer()),("sc",StandardScaler()),("m",Ridge(alpha=enh.RIDGE_ALPHA))])),
                   ("hgb", Pipeline([("im",SimpleImputer()),("m",HistGradientBoostingRegressor(**enh.HGB))]))]

def run(Xrows, balanced, reps=12):
    aucs = []
    for r in range(reps):
        rng = np.random.default_rng(500 + r); u = cases.copy(); rng.shuffle(u)
        fo = {c: i % 5 for i, c in enumerate(u)}
        wfold = np.array([fo[c] for c in wcid]); cfold = np.array([fo[c] for c in cases])
        coof = np.full(len(cases), np.nan)
        for f in range(5):
            tr = wfold != f
            preds = []
            for _, pipe in ens():
                if balanced:
                    pipe.fit(Xrows[tr], y_w[tr], m__sample_weight=w_bal[tr])
                else:
                    pipe.fit(Xrows[tr], y_w[tr])
                preds.append(pipe)
            for c in np.where(cfold == f)[0]:
                wp = np.mean([p.predict(Xrows[wcid == cases[c]]) for p in preds], 0)
                coof[c] = np.nanmean(wp)
        ok = np.isfinite(coof); aucs.append(roc_auc_score(ybc[ok], coof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs))

print("decomposition of the per-window 'collapse':", flush=True)
print(f"  B0 per-case median (1 row/pt, equal weight)         : 0.756", flush=True)
a, s = run(Xw_rep, False); print(f"  replicated-median, window-count weighted (control)  : {a:.3f} ±{s:.3f}", flush=True)
a, s = run(Xw_rep, True);  print(f"  replicated-median, BALANCED weight (FIX)            : {a:.3f} ±{s:.3f}  <- should recover ~0.756", flush=True)
a, s = run(Xw,     False); print(f"  real per-window features, window-count weighted     : {a:.3f} ±{s:.3f}", flush=True)
a, s = run(Xw,     True);  print(f"  real per-window features, BALANCED weight           : {a:.3f} ±{s:.3f}  <- weighting fixed; feature noise remains", flush=True)
print("DONE", flush=True)
