"""TASK 1 (redo, correct) — optimal analysis-window length.

Uses medium.npz (the DEPLOYED tri-modal-gated 30-s tiles, which give 0.756) as the gate ORACLE,
so we never re-run the drifted current-config gate. For window length W:
  * a W-window starting at a passing tile s0 is VALID iff EVERY 30-s tile it spans is a passing
    tile (so the whole W-window lies in gated-clean signal — no un-gated tails).
  * features are recomputed over the full W seconds; the CVP label is the median monitor CVP over
    the SAME W seconds. Per-case median, same nested grouped CV, thr 12.
W=30 reproduces medium.npz exactly -> must give ~0.756 (sanity).
"""
import json, time
import numpy as np
import common as K
from cvpkit import config as C
from collections import defaultdict

FS = K.FS; TILE = 30 * FS
WLIST = [8, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120]

d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
cid = M[:, cols.index("cid")].astype(int); start = M[:, cols.index("start")].astype(int)
pass_starts = defaultdict(list)
for c, s in zip(cid, start):
    pass_starts[int(c)].append(int(s))
pass_set = {c: set(v) for c, v in pass_starts.items()}

t0 = time.time()
rows_by_w = {W: [] for W in WLIST}
for n, c in enumerate(sorted(pass_starts)):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, cvpn = lc; L = len(ppg); ps = pass_set[c]
    for W in WLIST:
        WN = W * FS; ntile = int(np.ceil(WN / TILE))
        for s0 in pass_starts[c]:
            if s0 + WN > L:
                continue
            if not all((s0 + k * TILE) in ps for k in range(ntile)):
                continue                      # some spanned tile did not pass the deployed gate
            w = ppg[s0:s0 + WN]
            if np.isnan(w).any():
                continue
            lab = K.cvp_label(cvpn, s0 // FS, W)
            if not np.isfinite(lab):
                continue
            rows_by_w[W].append((c, K.ppg_features(w), lab))
    if (n + 1) % 60 == 0:
        print(f"  [{n+1}/{len(pass_starts)}] {time.time()-t0:.0f}s", flush=True)

print(f"\nextraction {time.time()-t0:.0f}s")
out = {}
for W in WLIST:
    X, y, cases, fn = K.per_case_matrix(rows_by_w[W])
    r = K.evaluate(X, y, cases, threshold=12.0, reps=40)
    nwin = len(rows_by_w[W]); wpc = nwin / max(len(np.unique([r[0] for r in rows_by_w[W]])), 1)
    print(f"W={W:3d}s  windows={nwin:6d} (~{wpc:.0f}/case)  N={r['n']:3d} pos={r['npos']:3d}  "
          f"AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  Sens {r['sens']:.2f} Spec {r['spec']:.2f}",
          flush=True)
    out[str(W)] = dict(win_sec=W, nwin=nwin, **{k: r[k] for k in ["auc", "ci", "sens", "spec", "n", "npos"]})
json.dump(out, open("results_task1_sweep_v2.json", "w"), indent=2)
best = max(out.values(), key=lambda v: v["auc"])
print(f"\nsaved. optimal W = {best['win_sec']}s (AUC {best['auc']:.3f}); W=30 = {out['30']['auc']:.3f} (must be ~0.756)")
