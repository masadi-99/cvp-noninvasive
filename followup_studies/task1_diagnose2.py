"""TASK 1 (diagnosis 2) — pin down WHY 90 s collapses AUC even though 3/4 features are ~unchanged.
Compute per-case matrices at 30 s and 90 s on the shared anchors; report single-feature AUCs and
feature-subset AUCs at BOTH scales, plus alternans correlations. Save matrices for reuse."""
import json, time
import numpy as np
import common as K
from cvpkit import config as C
from collections import defaultdict
from sklearn.metrics import roc_auc_score

t0 = time.time(); FS = K.FS
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
cid = M[:, cols.index("cid")].astype(int); start = M[:, cols.index("start")].astype(int)
anchors = defaultdict(list)
for c, s in zip(cid, start):
    anchors[int(c)].append(int(s))
W30, W90 = 30 * FS, 90 * FS
r30, r90 = [], []
for n, c in enumerate(sorted(anchors)):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, cvpn = lc; L = len(ppg)
    for s in anchors[c]:
        if s + W90 > L or np.isnan(ppg[s:s + W90]).any():
            continue
        lab30 = K.cvp_label(cvpn, s // FS, 30); lab90 = K.cvp_label(cvpn, s // FS, 90)
        if not (np.isfinite(lab30) and np.isfinite(lab90)):
            continue
        r30.append((c, K.ppg_features(ppg[s:s + W30]), lab30))
        r90.append((c, K.ppg_features(ppg[s:s + W90]), lab90))
print(f"extraction {time.time()-t0:.0f}s", flush=True)

X30, y30, cs30, _ = K.per_case_matrix(r30)
X90, y90, cs90, _ = K.per_case_matrix(r90)
assert list(cs30) == list(cs90)
yb = (y30 > 12).astype(int); names = K.FEATS + ["height"]
np.savez("task1_matrices.npz", X30=X30, X90=X90, y=y30, cases=cs30, names=names)


def auc_cols(X, idxs):
    """quick per-case AUC of the ensemble on a feature subset (single split, reps-avg OOF)."""
    r = K.evaluate(X[:, idxs], y30, cs30, threshold=12.0, reps=30)
    return r["auc"]


def single_auc(x):
    m = np.isfinite(x)
    if m.sum() < 20:
        return np.nan
    a = roc_auc_score(yb[m], x[m])
    return max(a, 1 - a)   # direction-agnostic univariate separability


print("\n--- univariate separability (direction-agnostic AUC) per feature ---")
print(f"{'feature':16s} {'30s':>6s} {'90s':>6s}")
for j, f in enumerate(names):
    print(f"{f:16s} {single_auc(X30[:,j]):6.3f} {single_auc(X90[:,j]):6.3f}")

print("\n--- multivariate ensemble AUC, feature subsets, 30s vs 90s ---")
subsets = {"all4+h": [0, 1, 2, 3, 4], "no-alternans(+h)": [1, 2, 3, 4],
           "alternans-only": [0], "upstroke-only": [2], "upstroke+ac+pvi+h": [1, 2, 3, 4]}
res = {}
for tag, idx in subsets.items():
    a30, a90 = auc_cols(X30, idx), auc_cols(X90, idx)
    print(f"{tag:20s} 30s={a30:.3f}  90s={a90:.3f}  Δ={a90-a30:+.3f}")
    res[tag] = dict(auc30=a30, auc90=a90)

print("\n--- alternans correlation 30s vs 90s (per case) ---")
m = np.isfinite(X30[:, 0]) & np.isfinite(X90[:, 0])
print(f"corr(alt30, alt90) = {np.corrcoef(X30[m,0], X90[m,0])[0,1]:+.3f}  (n={m.sum()})")
print(f"alt30 finite {np.isfinite(X30[:,0]).sum()}/{len(X30)} | alt90 finite {np.isfinite(X90[:,0]).sum()}/{len(X90)}")
json.dump(res, open("results_task1_diagnose2.json", "w"), indent=2)
print(f"\ndone {time.time()-t0:.0f}s")
