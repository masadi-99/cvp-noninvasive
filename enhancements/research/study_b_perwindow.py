"""Study B — windows as individual samples, with case-level cross-validation.

Proposal: instead of collapsing each case to its per-window MEDIAN, treat every 30-s window as
its own training sample (≈9000 samples instead of 304), keeping all windows of a case in the same
CV fold (GroupKFold by case) to avoid leakage.

We test this honestly and expose the evaluation subtlety. All configs use the same Ridge+HGB
ensemble regressing continuous CVP, binarised at 12. Five configurations:

  B0  per-case MEDIAN, grouped-by-case CV            -> CASE-level AUC   (the current baseline)
  B1  per-window rows, label = case-median CVP,      -> CASE-level AUC   (augmentation, case label)
        grouped-by-case CV, predictions averaged per case
  B2  per-window rows, label = per-window CVP,        -> CASE-level AUC   (literal 'each window a sample')
        grouped-by-case CV, predictions averaged per case
  B3  same model as B2                                 -> WINDOW-level AUC (NOT comparable to case AUC)
  B4  per-window rows, per-window label, RANDOM        -> WINDOW-level AUC (LEAK: windows of one case
        window-level CV (NObody-grouping)                  split across folds -> inflated)

The clinical prediction is per-PATIENT, so the only fair comparison is CASE-level AUC (B0/B1/B2).
B3 vs B0 shows window-AUC is a different (lower) quantity; B4 vs B3 shows what the case-grouping
prevents (the leak the proposal correctly guards against).
"""
import json, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import enh

REPS = 20
THR = 12.0
FE = enh.FEATS5


def _ens():
    return [Pipeline([("im", SimpleImputer()), ("sc", StandardScaler()), ("m", Ridge(alpha=enh.RIDGE_ALPHA))]),
            Pipeline([("im", SimpleImputer()), ("m", HistGradientBoostingRegressor(**enh.HGB))])]


def _fit_predict(Xtr, ytr, Xte):
    return np.mean([m.fit(Xtr, ytr).predict(Xte) for m in _ens()], 0)


# ---- per-case baseline (B0) -------------------------------------------------
Xc, yc, cases = enh.per_case(FE)
ybc = (yc > THR).astype(int)
r0 = enh.evaluate(Xc, yc, cases, threshold=THR, reps=40)
print(f"B0 per-case median (CASE AUC):        {r0['auc']:.3f}  [{r0['ci'][0]:.3f},{r0['ci'][1]:.3f}]", flush=True)

# ---- per-window matrices (restricted to the 304-case cohort) ----------------
d = enh.load_merged(); cid = d["cid"]
case_set = set(cases.tolist())
wm = np.array([c in case_set for c in cid])
wcid = cid[wm]
Xw = np.column_stack([d["F"][f][wm] for f in FE])
wnum = d["numeric"][wm]                      # per-window CVP (may be nan)
ycase_of = dict(zip(cases.tolist(), yc.tolist()))
wy_casemed = np.array([ycase_of[c] for c in wcid])   # case-median CVP broadcast to each window
cidx = {c: i for i, c in enumerate(cases)}


def _case_folds(rng, k=5):
    u = cases.copy(); rng.shuffle(u); fo = {c: i % k for i, c in enumerate(u)}
    return np.array([fo[c] for c in wcid]), fo


def run_window(label_kind, reps=REPS, grouped=True, level="case"):
    """label_kind: 'casemed' or 'perwin'. grouped: case-level folds vs random window folds.
    level: aggregate to 'case' or score at 'window'. Returns mean AUC over reps."""
    aucs = []
    for r in range(reps):
        rng = np.random.default_rng(123 + r * 7)
        if grouped:
            wfold, _ = _case_folds(rng)
        else:
            wfold = rng.integers(0, 5, size=len(wcid))          # random window-level folds (LEAK)
        case_oof = np.full(len(cases), np.nan); case_cnt = np.zeros(len(cases))
        win_oof = np.full(len(wcid), np.nan)
        for f in range(5):
            tr, te = wfold != f, wfold == f
            if label_kind == "casemed":
                ytr = wy_casemed[tr]; mtr = np.isfinite(ytr)
            else:
                ytr = wnum[tr]; mtr = np.isfinite(ytr)
            pred = _fit_predict(Xw[tr][mtr], ytr[mtr], Xw[te])
            win_oof[te] = pred
            for c in np.unique(wcid[te]):
                sel = wcid[te] == c
                case_oof[cidx[c]] = np.nanmean(pred[sel]); case_cnt[cidx[c]] = sel.sum()
        if level == "case":
            ok = np.isfinite(case_oof)
            aucs.append(roc_auc_score(ybc[ok], case_oof[ok]))
        else:
            wyb = (wnum > THR).astype(float); ok = np.isfinite(wnum) & np.isfinite(win_oof)
            aucs.append(roc_auc_score(wyb[ok], win_oof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs))


res = {"B0_case_median": dict(auc=r0["auc"], ci=r0["ci"], level="case")}

a, s = run_window("casemed", grouped=True, level="case")
res["B1_perwindow_caseLabel_caseAUC"] = dict(auc=a, sd=s, level="case"); print(f"B1 per-window, case label  (CASE AUC):  {a:.3f} ±{s:.3f}", flush=True)

a, s = run_window("perwin", grouped=True, level="case")
res["B2_perwindow_winLabel_caseAUC"] = dict(auc=a, sd=s, level="case"); print(f"B2 per-window, win  label  (CASE AUC):  {a:.3f} ±{s:.3f}", flush=True)

a, s = run_window("perwin", grouped=True, level="window")
res["B3_perwindow_winLabel_windowAUC"] = dict(auc=a, sd=s, level="window"); print(f"B3 per-window, win  label  (WINDOW AUC, not comparable): {a:.3f} ±{s:.3f}", flush=True)

a, s = run_window("perwin", grouped=False, level="window")
res["B4_perwindow_randomCV_windowAUC_LEAK"] = dict(auc=a, sd=s, level="window"); print(f"B4 per-window, RANDOM window CV (WINDOW AUC, LEAK): {a:.3f} ±{s:.3f}", flush=True)

json.dump(res, open("results_b_perwindow.json", "w"), indent=2)
print("\nsaved results_b_perwindow.json", flush=True)
