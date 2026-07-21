"""Diagnose the ppgcvp collapse (0.525). Split by cohort and by window provenance."""
import numpy as np
import common as K
import enh
from cvpkit import config as C
from collections import defaultdict

S = np.load("sweep_ppgcvp_W30.npy")   # cid,start,alt,ac,up,pvi,numeric
_, _, c304 = enh.per_case(enh.FEATS5); d304 = set(int(c) for c in c304)
# medium deployed windows (keys)
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; mc = [str(x) for x in d["cols"]]
mkey = set(zip(M[:, mc.index("cid")].astype(int).tolist(), M[:, mc.index("start")].astype(int).tolist()))


def aggregate_eval(rows, tag):
    byf = defaultdict(list); byy = defaultdict(list)
    for r in rows:
        byf[int(r[0])].append((r[2], r[3], r[4], r[5])); byy[int(r[0])].append(r[6])  # (alt,ac,up,pvi)
    from cvpkit import demographics as DEMO
    cases = sorted(byf); X = np.full((len(cases), 5), np.nan); y = np.full(len(cases), np.nan)
    for i, c in enumerate(cases):
        A = np.array(byf[c], float)
        for j in range(4):
            v = A[:, j][np.isfinite(A[:, j])]; X[i, j] = np.median(v) if len(v) else np.nan
        X[i, 4] = DEMO.demo_for(int(c)).get("height", np.nan)
        yy = np.array(byy[c], float); yy = yy[np.isfinite(yy)]; y[i] = np.median(yy) if len(yy) else np.nan
    ok = np.isfinite(y)
    r = K.evaluate(X[ok], y[ok], np.array(cases)[ok], threshold=12.0, reps=40)
    print(f"{tag:42s} N={r['n']:4d} pos={r['npos']:3d} AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")
    return r

rows = [tuple(r) for r in S]
overlap = [r for r in rows if int(r[0]) in d304]
new = [r for r in rows if int(r[0]) not in d304]
# for overlap cases: only the windows that ALSO passed the deployed gate (in medium) vs ppgcvp-only windows
ov_inmed = [r for r in overlap if (int(r[0]), int(r[1])) in mkey]
ov_new_win = [r for r in overlap if (int(r[0]), int(r[1])) not in mkey]

print("=== ppgcvp cohort splits ===")
aggregate_eval(rows, "all ppgcvp (357)")
aggregate_eval(overlap, "overlap-with-304 cases, ALL ppgcvp windows")
aggregate_eval(new, "NEW cases (not in 304)")
print("\n=== overlap cases: window provenance ===")
aggregate_eval(ov_inmed, "overlap cases, only deployed-gate windows")
aggregate_eval(ov_new_win, "overlap cases, only ppgcvp-ONLY (ECG-failed) windows")
frac_new = len(ov_new_win) / max(len(overlap), 1)
print(f"\noverlap cases: {len(overlap)} windows, {100*frac_new:.0f}% are ppgcvp-only (not deployed-gate)")
