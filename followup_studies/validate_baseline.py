"""CONTROL: prove the lean extractor matches cvpkit by holding the baseline cohort+labels fixed.
(1) baseline cvpkit features on the 304 cohort -> must be ~0.756 (harness sanity).
(2) OUR features on the SAME 304 cases + SAME labels -> must also be ~0.756 (extractor sanity).
(3) per-case correlation of our features vs cvpkit's."""
import json, time
import numpy as np
import common as K
from cvpkit import config as C
import enh

t0 = time.time()

# (1) baseline: cvpkit features, 304 cohort, per-case CVP from the cvpkit matrix
Xb, yb, casesb = enh.per_case(enh.FEATS5)
rb = enh.evaluate(Xb, yb, casesb, threshold=12.0, reps=40)
print(f"(1) baseline cvpkit feats: N={rb['n']} pos={rb['npos']} AUC={rb['auc']:.3f}", flush=True)
base_h = {int(c): Xb[i, 4] for i, c in enumerate(casesb)}   # height per case (col 4)

# our per-window features over the SAME 30-s anchors
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
cid = M[:, cols.index("cid")].astype(int); start = M[:, cols.index("start")].astype(int)
from collections import defaultdict
anchors = defaultdict(list)
for c, s in zip(cid, start):
    anchors[int(c)].append(int(s))

WIN = 30; percase_feats = {}
for c in casesb:                       # only the 304 baseline cases
    c = int(c); lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, _ = lc; fds = []
    for s in anchors[c]:
        w = ppg[s:s + WIN * K.FS]
        if len(w) < WIN * K.FS or np.isnan(w).any():
            continue
        fds.append(K.ppg_features(w))
    if fds:
        percase_feats[c] = {f: np.nanmedian([fd[f] for fd in fds]) for f in K.FEATS}
print(f"    our features computed for {len(percase_feats)}/{len(casesb)} baseline cases  {time.time()-t0:.0f}s", flush=True)

# (2) our features, SAME cases + SAME labels yb
Xm = np.full((len(casesb), 5), np.nan)
for i, c in enumerate(casesb):
    c = int(c); fd = percase_feats.get(c)
    if fd:
        Xm[i, :4] = [fd[f] for f in K.FEATS]
    Xm[i, 4] = base_h.get(c, np.nan)
rm = enh.evaluate(Xm, yb, casesb, threshold=12.0, reps=40)
print(f"(2) OUR feats, same cohort+labels: N={rm['n']} pos={rm['npos']} AUC={rm['auc']:.3f}", flush=True)

# (3) feature agreement (per-case correlation, our vs cvpkit)
print("(3) per-case correlation our-vs-cvpkit:")
for j, f in enumerate(K.FEATS):
    a = np.array([percase_feats.get(int(c), {}).get(f, np.nan) for c in casesb])
    b = Xb[:, j]; m = np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 5 else np.nan
    print(f"    {f:16s} r={r:.3f}  (n={m.sum()})")

verdict = "EXTRACTOR VALIDATED" if abs(rm['auc'] - rb['auc']) < 0.015 else "MISMATCH — investigate"
print("\n" + verdict)
json.dump(dict(baseline_auc=rb['auc'], our_auc=rm['auc'], n=rb['n'], npos=rb['npos']),
          open("results_validate.json", "w"), indent=2)
print(f"done {time.time()-t0:.0f}s")
