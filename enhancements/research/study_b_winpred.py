"""The decisive test: train the GOOD per-case model, apply it to individual WINDOWS.
If window-level AUC is high (~0.73), window-level prediction is fine — the failure was per-window
TRAINING, not per-window scoring. Grouped 5-fold by case, repeated."""
import json, numpy as np
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
wyb = (wnum > THR).astype(float); cidx = {c: i for i, c in enumerate(cases)}

def ens(): return [Pipeline([("im",SimpleImputer()),("sc",StandardScaler()),("m",Ridge(alpha=enh.RIDGE_ALPHA))]),
                   Pipeline([("im",SimpleImputer()),("m",HistGradientBoostingRegressor(**enh.HGB))])]

win_aucs, case_aucs = [], []
for r in range(20):
    rng = np.random.default_rng(200+r); u = cases.copy(); rng.shuffle(u)
    fo = {c: i % 5 for i, c in enumerate(u)}; cfold = np.array([fo[c] for c in cases]); wfold = np.array([fo[c] for c in wcid])
    woof = np.full(len(wcid), np.nan); coof = np.full(len(cases), np.nan)
    for f in range(5):
        tr = cfold != f
        mods = [m.fit(Xc[tr], yc[tr]) for m in ens()]          # TRAIN on per-case median features (the good model)
        teW = wfold == f
        woof[teW] = np.mean([m.predict(Xw[teW]) for m in mods], 0)   # APPLY to individual windows
        for c in np.where(cfold == f)[0]:
            coof[c] = np.nanmean(woof[wcid == cases[c]])
    okW = np.isfinite(wnum) & np.isfinite(woof)
    win_aucs.append(roc_auc_score(wyb[okW], woof[okW]))
    case_aucs.append(roc_auc_score(ybc, coof))
res = dict(percase_model_window_auc=float(np.mean(win_aucs)), percase_model_window_sd=float(np.std(win_aucs)),
           percase_model_case_auc=float(np.mean(case_aucs)))
print(f"per-CASE model applied to WINDOWS -> WINDOW AUC = {np.mean(win_aucs):.3f} ±{np.std(win_aucs):.3f}", flush=True)
print(f"per-CASE model, windows pooled    -> CASE   AUC = {np.mean(case_aucs):.3f}", flush=True)
print("(compare: per-WINDOW-trained model -> window AUC 0.43, case AUC 0.53)", flush=True)
json.dump(res, open("results_b_winpred.json","w"), indent=2)
print("saved results_b_winpred.json", flush=True)
