"""Reproduce the windowing study from data/windows.csv (per-window rows), using the repo's model
params. Shows why the per-patient MEDIAN (the deployed design) beats treating each window as its own
training row, and decomposes the difference into its two causes.

    python enhancements/reproduce_windowing.py     # needs numpy + scikit-learn (a few minutes)
"""
import os, csv, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from cvp.model import RIDGE_ALPHA, HGB, THRESHOLD

FE5 = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height"]
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "windows.csv"))))
wcid = np.array([int(r["caseid"]) for r in rows])
Xw = np.array([[float(r[f]) if r[f] not in ("", "nan") else np.nan for f in FE5] for r in rows])
wcvp = np.array([float(r["cvp_window"]) if r["cvp_window"] not in ("", "nan") else np.nan for r in rows])
cases = np.unique(wcid); cidx = {c: i for i, c in enumerate(cases)}
rows_by_case = {c: np.where(wcid == c)[0] for c in cases}
yc = np.array([np.nanmedian(wcvp[rows_by_case[c]]) for c in cases]); ybc = (yc > THRESHOLD).astype(int)
medfeat = {c: np.nanmedian(Xw[rows_by_case[c]], axis=0) for c in cases}


def ens():
    return [Pipeline([("im", SimpleImputer()), ("sc", StandardScaler()), ("m", Ridge(alpha=RIDGE_ALPHA))]),
            Pipeline([("im", SimpleImputer()), ("m", HistGradientBoostingRegressor(**HGB))])]

def folds(rng, k=5):
    u = cases.copy(); rng.shuffle(u); return {c: i % k for i, c in enumerate(u)}

def eval_percase(Xc, reps=20):
    aucs = []
    for r in range(reps):
        fo = folds(np.random.default_rng(r)); cf = np.array([fo[c] for c in cases]); oof = np.zeros(len(cases))
        for f in range(5):
            tr, te = cf != f, cf == f
            oof[te] = np.mean([m.fit(Xc[tr], yc[tr]).predict(Xc[te]) for m in ens()], 0)
        aucs.append(roc_auc_score(ybc, oof))
    return float(np.mean(aucs))

def eval_perwindow(Xrows, label="casemed", balanced=False, reps=12):
    yw = np.array([yc[cidx[c]] for c in wcid]) if label == "casemed" else wcvp
    sw = np.array([1.0 / len(rows_by_case[c]) for c in wcid])
    aucs = []
    for r in range(reps):
        fo = folds(np.random.default_rng(100 + r)); wf = np.array([fo[c] for c in wcid]); cf = np.array([fo[c] for c in cases])
        coof = np.full(len(cases), np.nan)
        for f in range(5):
            m = (wf != f) & np.isfinite(yw)
            mods = []
            for p in ens():
                p.fit(Xrows[m], yw[m], **({"m__sample_weight": sw[m]} if balanced else {})); mods.append(p)
            for c in np.where(cf == f)[0]:
                coof[c] = np.nanmean([np.mean([p.predict(Xrows[wcid == cases[c]]) for p in mods], 0)])
        aucs.append(roc_auc_score(ybc, coof))
    return float(np.mean(aucs))

def leak_windowAUC(reps=8):     # random window CV (no grouping) -> inflated window-level AUC
    yw = wcvp; wyb = (yw > THRESHOLD).astype(float); aucs = []
    for r in range(reps):
        rng = np.random.default_rng(7 + r); wf = rng.integers(0, 5, len(wcid)); oof = np.full(len(wcid), np.nan)
        for f in range(5):
            m = (wf != f) & np.isfinite(yw)
            oof[wf == f] = np.mean([p.fit(Xw[m], yw[m]).predict(Xw[wf == f]) for p in ens()], 0)
        ok = np.isfinite(yw) & np.isfinite(oof); aucs.append(roc_auc_score(wyb[ok], oof[ok]))
    return float(np.mean(aucs))


if __name__ == "__main__":
    print(f"BASELINE per-patient MEDIAN (the deployed design): CASE AUC {eval_percase(np.array([medfeat[c] for c in cases])):.3f}\n")

    print("Treating each window as a training row (grouped-by-patient CV, predictions pooled per patient):")
    print(f"  per-window rows, case label -> CASE AUC {eval_perwindow(Xw, 'casemed'):.3f}   (collapses; see why below)\n")

    print("More windows DO help when AGGREGATED (median of K random windows per patient):")
    for K in [1, 3, 8, 9999]:
        Xk = np.full((len(cases), 5), np.nan)
        rng = np.random.default_rng(0)
        for i, c in enumerate(cases):
            idx = rows_by_case[c]; pick = idx if K >= len(idx) else rng.choice(idx, K, replace=False)
            Xk[i] = np.nanmedian(Xw[pick], axis=0)
        print(f"  K={('all' if K > 999 else K):>3} windows: CASE AUC {eval_percase(Xk):.3f}")
    print()

    print("Decomposition of the per-window drop into two ordinary, fixable artifacts:")
    Xrep = np.array([medfeat[c] for c in wcid])   # replicated-median features = ZERO feature noise
    print(f"  windows-as-rows, replicated-median features (no feature noise) -> {eval_perwindow(Xrep, 'casemed'):.3f}   [artifact 1: window-count weighting]")
    print(f"  ... + balance patient weights (1/n_windows)                     -> {eval_perwindow(Xrep, 'casemed', balanced=True):.3f}   [fixes artifact 1]")
    print(f"  windows-as-rows, real window features                           -> {eval_perwindow(Xw, 'casemed'):.3f}   [+ artifact 2: single-window feature noise]")
    print(f"\n  random-window CV (no grouping) -> WINDOW AUC {leak_windowAUC():.3f}   [the leak that grouped CV prevents]")
    print("\nConclusion: aggregate to the per-patient median (denoises features AND weights patients equally).")
