"""TASK 2 (clean, anchored to deployed 0.756) — does ADDING the ECG-gate-failed windows to the
DEPLOYED pipeline destroy performance? Holds the 304 cohort + deployed label fixed; only the
window pool changes. No config-drift confound (deployed windows come from cvpkit_matrix)."""
import numpy as np
import common as K
import enh
from cvpkit import config as C, aggregate
from collections import defaultdict

# deployed cohort + label
Xb, yb, cases = enh.per_case(enh.FEATS5); cases = [int(c) for c in cases]; d304 = set(cases)
cidx = {c: i for i, c in enumerate(cases)}
from cvpkit import demographics as DEMO
height = {c: DEMO.demo_for(c).get("height", np.nan) for c in cases}

# per-window DEPLOYED features from cvpkit_matrix (only 304-cohort windows)
Mk, ck, _ = aggregate.load_matrix(C.MATRIX_NPZ)
mcid = Mk[:, ck.index("cid")].astype(int)
dep = defaultdict(list)
for i in range(len(Mk)):
    c = int(mcid[i])
    if c in d304:
        dep[c].append((Mk[i, ck.index("ppg_alternans")], Mk[i, ck.index("ppg_ac_amp")],
                        Mk[i, ck.index("ppg_upstroke")], Mk[i, ck.index("ppg_pvi")]))

# ECG-failed windows = ppgcvp windows for 304 cases NOT in medium.npz
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; mc = [str(x) for x in d["cols"]]
mkey = set(zip(M[:, mc.index("cid")].astype(int).tolist(), M[:, mc.index("start")].astype(int).tolist()))
S = np.load("sweep_ppgcvp_W30.npy")
ecgfail = defaultdict(list)
for r in S:
    c = int(r[0])
    if c in d304 and (c, int(r[1])) not in mkey:
        ecgfail[c].append((r[2], r[3], r[4], r[5]))   # (alt, ac, up, pvi) — cols 2..5 of the npy


def build_eval(pools, tag):
    X = np.full((len(cases), 5), np.nan)
    for i, c in enumerate(cases):
        A = np.array(pools.get(c, []), float)
        if len(A):
            for j in range(4):
                v = A[:, j][np.isfinite(A[:, j])]; X[i, j] = np.median(v) if len(v) else np.nan
        X[i, 4] = height[c]
    r = K.evaluate(X, yb, np.array(cases), threshold=12.0, reps=40)
    print(f"{tag:48s} AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  N={r['n']} pos={r['npos']}")
    return r

# combined pools = deployed + ECG-failed
comb = {c: dep.get(c, []) + ecgfail.get(c, []) for c in cases}
n_ecgfail = sum(len(v) for v in ecgfail.values()); n_dep = sum(len(v) for v in dep.values())
print(f"deployed windows: {n_dep} | ECG-failed windows added: {n_ecgfail} "
      f"({100*n_ecgfail/(n_dep+n_ecgfail):.0f}% of the combined pool)\n")
build_eval(dep, "A: deployed windows only (must be ~0.756)")
build_eval(comb, "B: deployed + ECG-gate-failed windows")
build_eval(ecgfail, "C: ECG-gate-failed windows only")
