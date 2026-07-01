"""Study B AUDIT — verify the per-window result is correct and understand WHY.

Answers three questions the result raises:
  (1) Is the per-window CVP LABEL correct (the actual contemporaneous CVP, not a mislabel)?
  (2) WHY is per-window prediction (~0.43) so much worse than per-patient (0.756)?
  (3) Is the per-window CASE-AUC collapse (0.53) fundamental, or a fixable modelling choice?
"""
import json, numpy as np
import enh
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

THR = 12.0; FE = enh.FEATS5
d = enh.load_merged(); cid = d["cid"]
Xc, yc, cases = enh.per_case(FE); ybc = (yc > THR).astype(int)
cset = set(cases.tolist())
wm = np.array([c in cset for c in cid]); wcid = cid[wm]
Xw = np.column_stack([d["F"][f][wm] for f in FE])
wnum = d["numeric"][wm]                                    # per-window CVP label
cidx = {c: i for i, c in enumerate(cases)}
out = {}

print("="*70); print("PART 1 — LABEL AUDIT (is the per-window CVP correct?)"); print("="*70)
fin = np.isfinite(wnum)
print(f"windows total={len(wcid)}  with finite per-window CVP={fin.sum()} ({100*fin.mean():.0f}%)", flush=True)
# does per-window CVP vary WITHIN a case? show 6 example cases
ex = [int(cases[i]) for i in [0, 5, 20, 60, 120, np.argmax(yc)]]
print("\nexample cases — per-window CVP values (numeric), case median, case label:")
for c in ex:
    v = wnum[wcid == c]; vf = v[np.isfinite(v)]
    print(f"  case {c:5d}: n={len(v):3d}  CVP per-window={np.round(np.sort(vf)[:8],1).tolist()}... "
          f"median={np.nanmedian(v):.1f} min={np.nanmin(v):.1f} max={np.nanmax(v):.1f}  label(med>12)={int(np.nanmedian(v)>12)}", flush=True)
# CRITICAL: does aggregating per-window CVP to the case median reproduce the case label yc?
agg = np.array([np.nanmedian(wnum[wcid == c]) for c in cases])
match = np.allclose(agg, yc, equal_nan=True)
print(f"\nper-window CVP aggregated to case-median == enh per-case y ?  {match}  (max abs diff {np.nanmax(np.abs(agg-yc)):.3f})", flush=True)
# within-case CVP movement + variance decomposition (ICC)
wstd = np.array([np.nanstd(wnum[wcid == c]) for c in cases])
cmean = np.array([np.nanmean(wnum[wcid == c]) for c in cases])
within_var = np.nanmean(wstd**2); between_var = np.nanvar(cmean)
icc = between_var / (between_var + within_var)
print(f"\nwithin-case CVP std: median={np.nanmedian(wstd):.2f} mmHg (IQR {np.nanpercentile(wstd,25):.2f}-{np.nanpercentile(wstd,75):.2f})", flush=True)
print(f"CVP variance: between-patient={between_var:.2f}  within-patient={within_var:.2f}  ICC={icc:.2f} "
      f"({100*icc:.0f}% of CVP variance is between-patient)", flush=True)
# how often does a window's own label disagree with its case label?
wyb = (wnum > THR).astype(float)
disagree = np.nanmean([np.nanmean((wnum[wcid==c]>THR) != (np.nanmedian(wnum[wcid==c])>THR)) for c in cases])
print(f"per-window label disagrees with case label in {100*disagree:.0f}% of windows (CVP fluctuates across the 12 cut)", flush=True)
out["label_audit"] = dict(frac_finite=float(fin.mean()), agg_matches_case_label=bool(match),
                          within_std_median=float(np.nanmedian(wstd)), icc=float(icc),
                          between_var=float(between_var), within_var=float(within_var),
                          window_label_disagree=float(disagree))

print("\n"+"="*70); print("PART 2 — BETWEEN- vs WITHIN-patient predictability"); print("="*70)
# between: corr of case-mean feature with case-mean CVP
cm_feat = np.array([[np.nanmean(Xw[wcid==c, j]) for j in range(len(FE))] for c in cases])
print("between-patient corr(case-mean feature, case-mean CVP):")
betw = {}
for j, f in enumerate(FE):
    g = np.isfinite(cm_feat[:, j]) & np.isfinite(cmean)
    r = np.corrcoef(cm_feat[g, j], cmean[g])[0, 1]; betw[f] = float(r)
    print(f"  {f:14} {r:+.3f}", flush=True)
# within: demean per case, pooled corr of feature-deviation with CVP-deviation
withn = {}
print("within-patient corr(feature deviation, CVP deviation) [pooled over windows]:")
for j, f in enumerate(FE):
    dv, dc = [], []
    for c in cases:
        m = (wcid == c) & np.isfinite(wnum) & np.isfinite(Xw[:, j])
        if m.sum() < 5: continue
        dv.append(Xw[m, j] - Xw[m, j].mean()); dc.append(wnum[m] - wnum[m].mean())
    dv = np.concatenate(dv); dc = np.concatenate(dc)
    r = np.corrcoef(dv, dc)[0, 1]; withn[f] = float(r)
    print(f"  {f:14} {r:+.3f}", flush=True)
out["between_corr"] = betw; out["within_corr"] = withn

print("\n"+"="*70); print("PART 3 — window-level CEILING (oracle between-patient)"); print("="*70)
# best possible window predictor that uses ONLY between-patient info = each case's TRUE median CVP
g = np.isfinite(wnum)
oracle_between = np.array([yc[cidx[c]] for c in wcid])      # true case-median CVP as the window score
auc_oracle = roc_auc_score(wyb[g], oracle_between[g])
print(f"oracle-between window AUC (score=true case-median CVP, label=window CVP>12): {auc_oracle:.3f}", flush=True)
print("  -> even PERFECT between-patient knowledge caps window-level AUC here; within-patient CVP is unscoreable from stable features", flush=True)
out["oracle_between_window_auc"] = float(auc_oracle)

print("\n"+"="*70); print("PART 4 — MODELLING AUDIT (is the 0.53 case-AUC fixable?)"); print("="*70)
def ridge(): return Pipeline([("im",SimpleImputer()),("sc",StandardScaler()),("m",Ridge(alpha=enh.RIDGE_ALPHA))])
def hgb():   return Pipeline([("im",SimpleImputer()),("m",HistGradientBoostingRegressor(**enh.HGB))])
def ens():   return [ridge(), hgb()]

def _folds(rng,k=5):
    u=cases.copy(); rng.shuffle(u); fo={c:i%k for i,c in enumerate(u)}; return fo

def perwindow_caseauc(models, label="casemed", agg="mean", reps=15):
    y_w = np.array([yc[cidx[c]] for c in wcid]) if label == "casemed" else wnum
    aucs = []
    for r in range(reps):
        rng = np.random.default_rng(100 + r); fo = _folds(rng)
        wfold = np.array([fo[c] for c in wcid]); cfold = np.array([fo[c] for c in cases])
        coof = np.full(len(cases), np.nan)
        for f in range(5):
            m = (wfold != f) & np.isfinite(y_w)
            mods = [mm.fit(Xw[m], y_w[m]) for mm in models()]
            for c in np.where(cfold == f)[0]:
                wp = np.mean([mm.predict(Xw[wcid == cases[c]]) for mm in mods], 0)
                coof[c] = np.nanmean(wp) if agg == "mean" else np.nanmedian(wp)
        ok = np.isfinite(coof); aucs.append(roc_auc_score(ybc[ok], coof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs))

# baseline
rb=enh.evaluate(Xc,yc,cases,threshold=THR,reps=20); print(f"B0 per-case median:                 {rb['auc']:.3f}", flush=True)
for lbl,(ml,lk,ag) in {
    "ensemble, case-label, mean-agg":   (ens,"casemed","mean"),
    "RIDGE-only, case-label, mean-agg": (lambda:[ridge()],"casemed","mean"),
    "HGB-only, case-label, mean-agg":   (lambda:[hgb()],"casemed","mean"),
    "ensemble, case-label, median-agg": (ens,"casemed","median"),
    "ensemble, WINDOW-label, mean-agg": (ens,"perwin","mean"),
}.items():
    a,s=perwindow_caseauc(ml,lk,ag); print(f"  per-window [{lbl:34}] CASE AUC = {a:.3f} ±{s:.3f}", flush=True)
    out.setdefault("modeling",{})[lbl]=dict(auc=a,sd=s)
out["B0"]=float(rb["auc"])

# Ridge coefficient attenuation: per-case vs per-window fit
rc=ridge().fit(Xc,yc); cw=ridge().fit(Xw, np.array([yc[cidx[c]] for c in wcid]))
out["ridge_coef_percase"]={f:float(v) for f,v in zip(FE, rc.named_steps['m'].coef_)}
out["ridge_coef_perwindow"]={f:float(v) for f,v in zip(FE, cw.named_steps['m'].coef_)}
print("\nRidge standardized coefs — per-case fit vs per-window fit (attenuation):")
for f in FE:
    print(f"  {f:14} per-case {out['ridge_coef_percase'][f]:+.3f}   per-window {out['ridge_coef_perwindow'][f]:+.3f}", flush=True)

json.dump(out, open("results_b_audit.json","w"), indent=2)
print("\nsaved results_b_audit.json", flush=True)
