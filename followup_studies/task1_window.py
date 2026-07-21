"""TASK 1 — does a 90-s analysis window beat the 30-s window?

Clean controlled comparison: for every existing (clean) window anchor whose 90-s span is valid,
compute the 4 PPG features over BOTH a 30-s and a 90-s window starting at that same anchor, and
recompute the CVP label over the MATCHING span (30 s label for 30 s features, 90 s label for 90 s
features — honouring 'CVP must correspond to the same window'). Identical anchor set for both arms,
so the ONLY difference is window length. Per-case median aggregation, same nested grouped CV, thr 12.
"""
import json, time
import numpy as np
import common as K
from cvpkit import config as C
from collections import defaultdict

t0 = time.time()
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
cid = M[:, cols.index("cid")].astype(int); start = M[:, cols.index("start")].astype(int)
anchors = defaultdict(list)
for c, s in zip(cid, start):
    anchors[int(c)].append(int(s))

W30, W90 = 30 * K.FS, 90 * K.FS
rows30, rows90 = [], []          # (cid, featdict, cvp_label) on the SHARED valid-90 anchor set
n_anchor_valid90 = 0

for n, c in enumerate(sorted(anchors)):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, cvpn = lc; L = len(ppg)
    for s in anchors[c]:
        if s + W90 > L:
            continue
        w90 = ppg[s:s + W90]
        if np.isnan(w90).any():
            continue
        sec0 = s // K.FS
        lab30 = K.cvp_label(cvpn, sec0, 30); lab90 = K.cvp_label(cvpn, sec0, 90)
        if not (np.isfinite(lab30) and np.isfinite(lab90)):
            continue
        n_anchor_valid90 += 1
        rows30.append((c, K.ppg_features(ppg[s:s + W30]), lab30))
        rows90.append((c, K.ppg_features(w90), lab90))
    if (n + 1) % 60 == 0:
        print(f"  [{n+1}/{len(anchors)}] anchors={n_anchor_valid90}  {time.time()-t0:.0f}s", flush=True)

print(f"\nshared valid-90 anchors: {n_anchor_valid90}   ({time.time()-t0:.0f}s)")
out = {}
for tag, rows in [("30s", rows30), ("90s", rows90)]:
    X, y, cases, fn = K.per_case_matrix(rows)
    r = K.evaluate(X, y, cases, threshold=12.0, reps=40)
    print(f"WINDOW={tag}: N={r['n']} pos={r['npos']}  AUC={r['auc']:.3f} "
          f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  Sens {r['sens']:.2f} Spec {r['spec']:.2f}")
    out[tag] = {k: r[k] for k in ["auc", "ci", "sens", "spec", "n", "npos"]}
out["n_anchors"] = n_anchor_valid90
json.dump(out, open("results_task1_window.json", "w"), indent=2)
print("saved results_task1_window.json  |  delta(90-30) AUC = %+.3f" % (out["90s"]["auc"] - out["30s"]["auc"]))
