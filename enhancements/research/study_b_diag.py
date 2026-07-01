"""Study B diagnostic — is the per-window collapse real (regression dilution) or a bug?
B1b: train on per-window rows (case-median label, grouped CV) but TEST on each case's MEDIAN
features (one prediction per case). If this recovers ~0.75, the collapse is caused by NOISY
per-window TEST inputs; if it stays low, training on noisy per-window inputs itself diluted the model.
Also reports the within-case variability of each per-window feature (confirms windows really vary).
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
d = enh.load_merged(); cid = d["cid"]; case_set = set(cases.tolist())
wm = np.array([c in case_set for c in cid]); wcid = cid[wm]
Xw = np.column_stack([d["F"][f][wm] for f in FE])
ycase_of = dict(zip(cases.tolist(), yc.tolist()))
wy = np.array([ycase_of[c] for c in wcid])
cidx = {c: i for i, c in enumerate(cases)}

# within-case variability of each feature (relative to its overall std)
print("within-case feature variability (median over cases of within-case std / overall std):")
for j, f in enumerate(FE):
    overall = np.nanstd(Xw[:, j]) + 1e-9
    wcs = [np.nanstd(Xw[wcid == c, j]) for c in cases]
    print(f"  {f:14} {np.nanmedian(wcs)/overall:.2f}", flush=True)


def _ens():
    return [Pipeline([("im", SimpleImputer()), ("sc", StandardScaler()), ("m", Ridge(alpha=enh.RIDGE_ALPHA))]),
            Pipeline([("im", SimpleImputer()), ("m", HistGradientBoostingRegressor(**enh.HGB))])]


def _folds(rng, k=5):
    u = cases.copy(); rng.shuffle(u); fo = {c: i % k for i, c in enumerate(u)}
    return fo

aucs = []
for r in range(20):
    rng = np.random.default_rng(7 + r)
    fo = _folds(rng); wfold = np.array([fo[c] for c in wcid]); cfold = np.array([fo[c] for c in cases])
    oof = np.full(len(cases), np.nan)
    for f in range(5):
        tr = wfold != f
        mods = [m.fit(Xw[tr], wy[tr]) for m in _ens()]
        te_cases = np.where(cfold == f)[0]
        pred = np.mean([m.predict(Xc[te_cases]) for m in mods], 0)   # TEST on per-case MEDIAN features
        oof[te_cases] = pred
    aucs.append(roc_auc_score(ybc, oof))
print(f"\nB1b train-per-window / test-on-case-median-features (CASE AUC): {np.mean(aucs):.3f} ±{np.std(aucs):.3f}", flush=True)
print("(compare B0 per-case median = 0.756, B1 per-window test = 0.526)", flush=True)
