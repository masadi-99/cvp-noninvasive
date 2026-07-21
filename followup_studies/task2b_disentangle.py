"""TASK 2 (disentangle) — the 2b collapse to chance conflates two changes:
  (i) FEATURE windows: curated tri-modal anchors  ->  PPG-quality>0.7 windows across the record
  (ii) LABEL: per-case CVP over the anchors        ->  per-case CVP over the whole record
Hold the 304 cohort fixed and vary ONE at a time.

  A  baseline features + baseline label                 (known 0.755)
  B  PPG-quality feature windows + BASELINE label       -> isolates feature-window selection
  C  baseline features + WHOLE-RECORD label             -> isolates the label redefinition
  D  PPG-quality windows + whole-record label           (= 2b overlap, ~0.50)
"""
import json, time
import numpy as np
import common as K
import enh
from cvpkit import config as C
from collections import defaultdict

QT = 0.7; WIN = 30; W = WIN * K.FS; FS = K.FS; CMAX, KMAX = 80, 20
t0 = time.time()

# baseline: features (X) + label (y) on the 304 cohort
Xbase, ybase, casesb = enh.per_case(enh.FEATS5)
base = [int(c) for c in casesb]
yb_bin = (ybase > 12).astype(int)

# per-case: PPG-quality feature windows across the record + whole-record CVP label
Xq = np.full((len(base), 5), np.nan)          # 2b-style features (4 PPG + height)
y_whole = np.full(len(base), np.nan)
from cvpkit import demographics as DEMO
for i, c in enumerate(base):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, cvpn = lc; L = len(ppg); nwin = L // W
    cf = cvpn[np.isfinite(cvpn)]
    y_whole[i] = np.median(cf) if len(cf) else np.nan
    idx = np.linspace(0, nwin - 1, min(nwin, CMAX)).astype(int)
    fds = []
    for wi in idx:
        if len(fds) >= KMAX:
            break
        s = int(wi) * W; w = ppg[s:s + W]
        if len(w) < W or np.isnan(w).any():
            continue
        q = K.ppg_quality_score(w)
        if not (np.isfinite(q) and q > QT):
            continue
        fds.append(K.ppg_features(w))
    if fds:
        for j, f in enumerate(K.FEATS):
            v = np.array([fd[f] for fd in fds], float); v = v[np.isfinite(v)]
            Xq[i, j] = np.median(v) if len(v) else np.nan
    Xq[i, 4] = DEMO.demo_for(int(c)).get("height", np.nan)
    if (i + 1) % 60 == 0:
        print(f"  [{i+1}/{len(base)}] {time.time()-t0:.0f}s", flush=True)

print(f"\nfeature build done {time.time()-t0:.0f}s")
print(f"label positives: baseline(anchor CVP)={yb_bin.sum()}/{len(base)}  whole-record CVP>12={int(np.nansum(y_whole>12))}/{np.isfinite(y_whole).sum()}")

out = {}
def run(tag, X, y):
    m = np.isfinite(y)
    r = K.evaluate(X[m], y[m], np.array(base)[m], threshold=12.0, reps=40)
    print(f"{tag:38s} N={r['n']} pos={r['npos']} AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")
    out[tag] = {k: r[k] for k in ["auc", "ci", "n", "npos"]}

run("A baseline feats + baseline label", Xbase, ybase)
run("B PPGqual feats + baseline label", Xq, ybase)
run("C baseline feats + whole-record label", Xbase, y_whole)
run("D PPGqual feats + whole-record label", Xq, y_whole)
json.dump(out, open("results_task2b_disentangle.json", "w"), indent=2)
print("saved results_task2b_disentangle.json")
