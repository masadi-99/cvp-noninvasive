"""TASK 3 (expanded) — what ELSE can the non-invasive PPG model predict, beyond CVP?
Every physiologically-plausible hemodynamic target present in the data (monitor CO/CI/SV/SVV/SVRI +
arterial MAP), same 5 PPG features, same nested grouped CV. Continuous Spearman + clinical-threshold
AUC. For flow targets (CO/SV) we also decompose PPG-only vs height to expose any body-size artifact.

Physiological rationale for each:
  SVV  -> the PPG PVI feature IS the non-invasive analog of stroke-volume variation (fluid responsiveness)
  SVRI -> vascular tone; PPG upstroke/dicrotic morphology tracks it
  SV/CO-> pulse contour ~ stroke volume (body-size confounded)
  MAP  -> arterial pressure / hypotension (cuffless-BP flavour)
"""
import json
import numpy as np
import common as K
import enh
from cvpkit import config as C
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

z = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = z["data"]; cols = [str(c) for c in z["cols"]]
cid = M[:, cols.index("cid")].astype(int)


def percase(name):
    if name not in cols:
        return None
    v = M[:, cols.index(name)].astype(float)
    return {int(c): float(np.nanmedian(v[cid == c])) for c in np.unique(cid)}


Xb, _, casesb = enh.per_case(enh.FEATS5)     # 4 PPG feats + height (col 4)


def reg_oof(X, y, cases, reps=40, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float); cases = np.asarray(cases)
    oof = np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(seed + r * 11); fo = enh._folds(cases, 5, rng); pr = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            pr[te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in enh._ensemble()], 0)
        oof += pr
    return oof / reps


def binary_auc(y_true_bin, oof, positive_is_low):
    if not (0 < y_true_bin.sum() < len(y_true_bin)):
        return np.nan
    return float(roc_auc_score(y_true_bin, -oof if positive_is_low else oof))


# target: (column, unit, clinical-cut, positive_is_low, decompose_bodysize)
TARGETS = [
    ("co",   "L/min",        4.0,  True,  True),
    ("ci",   "L/min/m^2",    2.5,  True,  False),
    ("sv",   "mL",          60.0,  True,  True),
    ("svv",  "%",           13.0,  False, False),   # SVV>13 = fluid responsive
    ("svri", "dyn.s.cm-5.m2", 1970.0, True, False), # low SVRI = vasodilation
    ("art_mbp_n", "mmHg (MAP)", 65.0, True, False), # MAP<65 = hypotension
]

out = {}
for name, unit, cut, low, decomp in TARGETS:
    D = percase(name)
    if D is None:
        print(f"{name}: column absent"); continue
    tgt = np.array([D.get(int(c), np.nan) for c in casesb]); m = np.isfinite(tgt)
    if m.sum() < 40:
        print(f"{name}: only {m.sum()} cases — skip"); continue
    X, y, cs = Xb[m], tgt[m], casesb[m]
    q = np.percentile(y, [10, 50, 90])
    oof = reg_oof(X, y, cs)
    sr = float(spearmanr(oof, y).correlation)
    yb = (y < cut).astype(int) if low else (y > cut).astype(int)
    auc_clin = binary_auc(yb, oof, low)
    ybm = (y > np.median(y)).astype(int)
    auc_med = binary_auc(ybm, oof, False)
    print(f"\n{name} ({unit})  N={m.sum()}  median={q[1]:.1f} [10-90%: {q[0]:.1f}-{q[2]:.1f}]")
    print(f"   Spearman r={sr:+.3f} | {'low' if low else 'high'}-cut {'<' if low else '>'}{cut} "
          f"[{yb.sum()}/{len(yb)} pos] AUC={auc_clin:.3f} | median-split AUC={auc_med:.3f}")
    rec = dict(n=int(m.sum()), median=q[1], spearman=sr, cut=cut,
               n_pos=int(yb.sum()), auc_clin=float(auc_clin), auc_median=float(auc_med))
    if decomp:
        oofp = reg_oof(X[:, :4], y, cs); oofh = reg_oof(X[:, [4]], y, cs)
        ap = binary_auc(yb, oofp, low); ah = binary_auc(yb, oofh, low)
        print(f"   decompose: PPG-only AUC={ap:.3f} | height-only AUC={ah:.3f} | full AUC={auc_clin:.3f}")
        rec["auc_ppg_only"] = float(ap); rec["auc_height_only"] = float(ah)
    out[name] = rec

json.dump(out, open("results_task3c_outcomes.json", "w"), indent=2)
print("\nsaved results_task3c_outcomes.json")
