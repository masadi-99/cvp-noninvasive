"""Fair, SAME-COHORT window comparison: for each W>30, restrict to the cases that HAVE a clean
W-window, and compare 30-s features vs W-s features on THAT identical cohort (fixed deployed label).
Answers: does a longer window beat 30 s on the cases where the longer window is available?"""
import time
import numpy as np
import common as K
import enh
from cvpkit import config as C
from collections import defaultdict

FS = K.FS; TILE = 30 * FS
WSET = [30, 45, 60, 90]

Xb, yb, cases = enh.per_case(enh.FEATS5); cases = [int(c) for c in cases]; cidx = {c: i for i, c in enumerate(cases)}
height = {c: Xb[cidx[c], 4] for c in cases}
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
mcid = M[:, cols.index("cid")].astype(int); mstart = M[:, cols.index("start")].astype(int)
pass_starts = defaultdict(list)
for c, s in zip(mcid, mstart):
    if int(c) in cidx:
        pass_starts[int(c)].append(int(s))
pass_set = {c: set(v) for c, v in pass_starts.items()}

t0 = time.time()
feats = {W: defaultdict(list) for W in WSET}
for n, c in enumerate(cases):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, _ = lc; L = len(ppg); ps = pass_set.get(c, set())
    for W in WSET:
        WN = W * FS; ntile = int(np.ceil(WN / TILE))
        for s0 in pass_starts.get(c, []):
            if s0 + WN > L or not all((s0 + k * TILE) in ps for k in range(ntile)):
                continue
            w = ppg[s0:s0 + WN]
            if np.isnan(w).any():
                continue
            feats[W][c].append(K.ppg_features(w))
    if (n + 1) % 80 == 0:
        print(f"  [{n+1}/{len(cases)}] {time.time()-t0:.0f}s", flush=True)


def matrix(W, subset):
    X = np.full((len(subset), 5), np.nan)
    for i, c in enumerate(subset):
        A = np.array([[fd[f] for f in K.FEATS] for fd in feats[W][c]], float)
        if len(A):
            for j in range(4):
                v = A[:, j][np.isfinite(A[:, j])]; X[i, j] = np.median(v) if len(v) else np.nan
        X[i, 4] = height[c]
    return X


print(f"\nextraction {time.time()-t0:.0f}s\nSAME-COHORT paired comparison (30 s vs W), fixed deployed label:")
for W in [45, 60, 90]:
    subset = [c for c in cases if feats[W].get(c)]          # cases with a clean W-window
    yy = np.array([yb[cidx[c]] for c in subset])
    r30 = enh.evaluate(matrix(30, subset), yy, np.array(subset), threshold=12.0, reps=40)
    rW = enh.evaluate(matrix(W, subset), yy, np.array(subset), threshold=12.0, reps=40)
    print(f"  cohort=cases-with-{W}s-window (N={len(subset)}, pos={int((yy>12).sum())}): "
          f"30s AUC={r30['auc']:.3f}  vs  {W}s AUC={rW['auc']:.3f}   Δ({W}-30)={rW['auc']-r30['auc']:+.3f}",
          flush=True)
