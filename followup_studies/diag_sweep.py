"""Diagnose why sweep2 trimodal@30s (0.613/203) != deployed (0.756/304).
Compare sweep2's windows/features/labels against medium.npz + cvpkit on SHARED keys."""
import numpy as np
import common as K
import enh
from cvpkit import config as C, aggregate

# sweep2 30s output
S = np.load("sweep_valid_W30.npy")   # cid,start,alt,ac,up,pvi,numeric
skey = set(zip(S[:, 0].astype(int).tolist(), S[:, 1].astype(int).tolist()))
scases = set(S[:, 0].astype(int).tolist())
print(f"sweep2@30s: {len(S)} windows, {len(scases)} cases")

# medium.npz deployed windows
d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; mc = [str(c) for c in d["cols"]]
mcid = M[:, mc.index("cid")].astype(int); mstart = M[:, mc.index("start")].astype(int)
mkey = set(zip(mcid.tolist(), mstart.tolist())); mcases = set(mcid.tolist())
print(f"medium.npz : {len(M)} windows, {len(mcases)} cases")
print(f"window overlap sweep2∩medium: {len(skey & mkey)}  | sweep2-only: {len(skey-mkey)}  | medium-only: {len(mkey-skey)}")
print(f"case overlap: {len(scases & mcases)} | sweep2-only cases: {len(scases-mcases)}")

# cvpkit features (deployed 304 cohort)
Xb, yb, casesb = enh.per_case(enh.FEATS5)
base = {int(c): i for i, c in enumerate(casesb)}

# (A) is sweep2's 203 cohort just a HARDER subset? evaluate cvpkit feats on sweep2's cases
common_cases = [c for c in scases if c in base]
idx = [base[c] for c in common_cases]
r = enh.evaluate(Xb[idx], yb[idx], np.array(common_cases), threshold=12.0, reps=40)
print(f"\n(A) cvpkit feats on sweep2's {len(idx)} cases: AUC={r['auc']:.3f} pos={r['npos']}  "
      f"(if ~0.61 -> cohort-selection; if ~0.75 -> sweep2 feature/label bug)")

# (B) per-case CVP label agreement (sweep2 median-numeric vs cvpkit yb)
sy = {}
for row in S:
    sy.setdefault(int(row[0]), []).append(row[6])
sy = {c: np.nanmedian(v) for c, v in sy.items()}
a = np.array([sy[c] for c in common_cases]); b = np.array([yb[base[c]] for c in common_cases])
mfin = np.isfinite(a) & np.isfinite(b)
print(f"(B) per-case CVP label corr sweep2-vs-cvpkit r={np.corrcoef(a[mfin],b[mfin])[0,1]:.3f} "
      f"| mean|Δ|={np.nanmean(np.abs(a[mfin]-b[mfin])):.2f} mmHg | sweep2 pos={int((a>12).sum())} cvpkit pos={int((b>12).sum())}")

# (C) feature agreement on SHARED windows: sweep2 vs cvpkit matrix
Mk, ck, _ = aggregate.load_matrix(C.MATRIX_NPZ)
kkey = {(int(Mk[i, ck.index('cid')]), int(Mk[i, ck.index('start')])): i for i in range(len(Mk))}
shared = list(skey & mkey)[:4000]
srow = {(int(r[0]), int(r[1])): r for r in S}
print("\n(C) per-window feature agreement on shared windows (sweep2 vs cvpkit):")
for j, f in zip([2, 3, 4, 5], K.FEATS):
    aa, bb = [], []
    for kk in shared:
        if kk in kkey:
            aa.append(srow[kk][j]); bb.append(Mk[kkey[kk], ck.index(f)])
    aa, bb = np.array(aa), np.array(bb); mm = np.isfinite(aa) & np.isfinite(bb)
    rr = np.corrcoef(aa[mm], bb[mm])[0, 1] if mm.sum() > 5 else np.nan
    print(f"    {f:16s} r={rr:.3f} (n={mm.sum()})")
