"""TASK 3 (follow-up) — is the low-CO AUC 0.80 real PPG signal or just body size?
CO scales with body size and `height` is a model feature. Decompose: PPG-features-only (no height),
height-only, and full. If PPG-only stays high, the pulse waveform carries genuine CO information."""
import json
import numpy as np
import common as K
import enh
from cvpkit import config as C
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

z = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = z["data"]; cols = [str(c) for c in z["cols"]]
cid = M[:, cols.index("cid")].astype(int)
CO = {int(c): float(np.nanmedian(M[cid == c, cols.index("co")].astype(float))) for c in np.unique(cid)}
Xb, _, casesb = enh.per_case(enh.FEATS5)          # cols: 4 PPG + height (idx 4)


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


tgt = np.array([CO.get(int(c), np.nan) for c in casesb]); m = np.isfinite(tgt)
X, y, cs = Xb[m], tgt[m], casesb[m]
lo = (y < 4.0).astype(int)
variants = {"PPG-only (no height)": [0, 1, 2, 3], "height-only": [4], "full (PPG+height)": [0, 1, 2, 3, 4]}
out = {}
print(f"CO decomposition  (N={m.sum()}, low-CO<4 = {lo.sum()} positives)\n")
for tag, idx in variants.items():
    oof = reg_oof(X[:, idx], y, cs)
    sr = float(spearmanr(oof, y).correlation)
    auc = float(roc_auc_score(lo, -oof))
    print(f"  {tag:22s}: Spearman r={sr:+.3f}  low-CO AUC={auc:.3f}")
    out[tag] = dict(spearman=sr, auc_low=auc)
json.dump(out, open("results_task3b_co_decompose.json", "w"), indent=2)
print("\nsaved results_task3b_co_decompose.json")
