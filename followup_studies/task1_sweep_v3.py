"""TASK 1 (definitive) — window length with the DEPLOYED cohort+label held FIXED.

Cohort = the deployed 304 cases; label = the deployed per-case CVP (cvpkit yb). ONLY the feature
window length varies. Features per case = median over that case's medium.npz gate-clean W-windows
(a W-window is valid iff every 30-s tile it spans passed the deployed gate). Cases with no clean
W-window at a given W get NaN features (imputed) -> this shows the availability cost of longer
windows honestly, on a constant cohort/label. W=30 == deployed features -> MUST give 0.756.

Also reports, per W, the AUC restricted to cases that DO have a clean W-window (feature-quality,
availability aside)."""
import json, time
import numpy as np
import common as K
import enh
from cvpkit import config as C
from collections import defaultdict

FS = K.FS; TILE = 30 * FS
WLIST = [8, 10, 15, 20, 25, 30, 40, 45, 60, 90, 120]

# deployed cohort + label + height
Xb, yb, cases = enh.per_case(enh.FEATS5)
cases = [int(c) for c in cases]; cidx = {c: i for i, c in enumerate(cases)}
height = {c: Xb[cidx[c], 4] for c in cases}

d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
mcid = M[:, cols.index("cid")].astype(int); mstart = M[:, cols.index("start")].astype(int)
pass_starts = defaultdict(list);
for c, s in zip(mcid, mstart):
    if int(c) in cidx:
        pass_starts[int(c)].append(int(s))
pass_set = {c: set(v) for c, v in pass_starts.items()}

t0 = time.time()
# feats_by_w[W][case] = list of feature dicts over that case's valid-W windows
feats_by_w = {W: defaultdict(list) for W in WLIST}
for n, c in enumerate(cases):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, _ = lc; L = len(ppg); ps = pass_set.get(c, set())
    for W in WLIST:
        WN = W * FS; ntile = int(np.ceil(WN / TILE))
        for s0 in pass_starts.get(c, []):
            if s0 + WN > L or not all((s0 + k * TILE) in ps for k in range(ntile)):
                continue
            w = ppg[s0:s0 + WN]
            if np.isnan(w).any():
                continue
            feats_by_w[W][c].append(K.ppg_features(w))
    if (n + 1) % 60 == 0:
        print(f"  [{n+1}/{len(cases)}] {time.time()-t0:.0f}s", flush=True)

print(f"\nextraction {time.time()-t0:.0f}s  (cohort={len(cases)}, fixed label, pos={(yb>12).sum()})")
out = {}
for W in WLIST:
    X = np.full((len(cases), 5), np.nan)
    havew = np.zeros(len(cases), bool)
    for i, c in enumerate(cases):
        fl = feats_by_w[W].get(c, [])
        if fl:
            havew[i] = True
            for j, f in enumerate(K.FEATS):
                v = np.array([fd[f] for fd in fl], float); v = v[np.isfinite(v)]
                X[i, j] = np.median(v) if len(v) else np.nan
        X[i, 4] = height[c]
    r_full = enh.evaluate(X, yb, np.array(cases), threshold=12.0, reps=40)     # fixed 304, NaN imputed
    m = havew
    r_avail = enh.evaluate(X[m], yb[m], np.array(cases)[m], threshold=12.0, reps=40) if m.sum() > 40 else None
    print(f"W={W:3d}s  cases_with_clean_window={m.sum():3d}/{len(cases)}  "
          f"AUC(fixed304)={r_full['auc']:.3f}  " +
          (f"AUC(avail-only,N={r_avail['n']})={r_avail['auc']:.3f}" if r_avail else "avail: n/a"), flush=True)
    out[str(W)] = dict(win_sec=W, n_with_window=int(m.sum()), auc_fixed304=r_full["auc"],
                       ci_fixed304=r_full["ci"], auc_avail=(r_avail["auc"] if r_avail else None),
                       n_avail=(r_avail["n"] if r_avail else 0))
json.dump(out, open("results_task1_sweep_v3.json", "w"), indent=2)
print(f"\nW=30 AUC(fixed304) = {out['30']['auc_fixed304']:.3f}  (MUST be ~0.756)")
best = max((v for v in out.values()), key=lambda v: v["auc_fixed304"])
print(f"optimal (fixed-cohort) W = {best['win_sec']}s at AUC {best['auc_fixed304']:.3f}")
