"""TASK 3 — predict Cardiac Output (CO) and Cardiac Index (CI) from the SAME non-invasive PPG
features, for a broader physiological view. CO/CI ground truth = per-case median of the monitor
CO/CI already present in medium.npz (EV1000/Vigileo), 142/333 cases have it.

Reports, under the SAME nested grouped-by-case CV as the CVP model:
  * continuous: out-of-fold Spearman r and R^2 (does PPG track CO/CI at all?)
  * binary: AUC for LOW output (clinically the concern) and for a balanced median split.
"""
import json
import numpy as np
import common as K
import enh
from cvpkit import config as C
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# ── per-case CO / CI from medium.npz ──────────────────────────────────────────
z = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = z["data"]; cols = [str(c) for c in z["cols"]]
cid = M[:, cols.index("cid")].astype(int)


def percase(name):
    v = M[:, cols.index(name)].astype(float)
    return {int(c): float(np.nanmedian(v[cid == c])) for c in np.unique(cid)}


CO, CI = percase("co"), percase("ci")

# ── non-invasive features (same 5 as the CVP model), 304 cohort ───────────────
Xb, yb_cvp, casesb = enh.per_case(enh.FEATS5)


def regression_cv(X, y, cases, reps=40, seed=0):
    """Averaged out-of-fold ensemble prediction under repeated grouped 5-fold CV."""
    X = np.asarray(X, float); y = np.asarray(y, float); cases = np.asarray(cases)
    oof = np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(seed + r * 11); fo = enh._folds(cases, 5, rng); pr = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            pr[te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in enh._ensemble()], 0)
        oof += pr
    oof /= reps
    sr = float(spearmanr(oof, y).correlation)
    r2 = float(1 - np.sum((y - oof) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return oof, sr, r2


out = {}
LOWCUT = {"CO": 4.0, "CI": 2.5}   # clinical low-output cuts (CO<4 L/min, CI<2.5 L/min/m^2)
for name, D in [("CO", CO), ("CI", CI)]:
    tgt = np.array([D.get(int(c), np.nan) for c in casesb])
    m = np.isfinite(tgt)
    Xs, ys, cs = Xb[m], tgt[m], casesb[m]
    q = np.percentile(ys, [5, 25, 50, 75, 95])
    print(f"\n===== {name}  (N={m.sum()} cases with monitor {name}) =====")
    print(f"  per-case {name}: median {np.median(ys):.2f} | IQR {q[1]:.2f}-{q[3]:.2f} | 5-95% {q[0]:.2f}-{q[4]:.2f}")
    oof, sr, r2 = regression_cv(Xs, ys, cs)
    print(f"  continuous: Spearman r={sr:+.3f}  R^2={r2:+.3f}")
    # binary — LOW output (positive = below the clinical cut); score = -predicted
    lo = (ys < LOWCUT[name]).astype(int)
    auc_lo = roc_auc_score(lo, -oof) if 0 < lo.sum() < len(lo) else np.nan
    # binary — balanced median split (positive = above median); score = +predicted
    hi = (ys > np.median(ys)).astype(int)
    auc_md = roc_auc_score(hi, oof) if 0 < hi.sum() < len(hi) else np.nan
    print(f"  binary AUC: low-{name}(<{LOWCUT[name]}) [{lo.sum()}/{len(lo)} pos] = {auc_lo:.3f}"
          f"   |  median-split = {auc_md:.3f}")
    out[name] = dict(n=int(m.sum()), median=float(np.median(ys)), spearman=sr, r2=r2,
                     low_cut=LOWCUT[name], n_low=int(lo.sum()), auc_low=float(auc_lo), auc_median=float(auc_md))

json.dump(out, open("results_task3_co_ci.json", "w"), indent=2)
print("\nsaved results_task3_co_ci.json")
