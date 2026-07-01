"""Extract the rich arterial-waveform features for every case (per-case median over its clean windows).
Saves artwave_features.npz."""
import sys, warnings, os; sys.path.insert(0, "/home/masadi/cvp-ecg-ppg"); sys.path.insert(0, "/home/masadi/run")
warnings.filterwarnings("ignore")
import numpy as np
from cvpkit import config as C
import enh, artwave

AUX = "/home/masadi/cvp_data/aux"
WIN = C.WIN
d = enh.load_merged(); cid = d["cid"]; start = d["start"]
Xc, yc, cases = enh.per_case(enh.FEATS5)
starts_by_case = {}
for c, s in zip(cid.tolist(), start.tolist()):
    starts_by_case.setdefault(c, []).append(s)

rows = []
for n, c in enumerate(cases):
    c = int(c); p = os.path.join(AUX, f"aux_{c}.npz")
    feats = {k: [] for k in artwave.FEATURES}; nwin = 0
    if os.path.exists(p):
        art = np.load(p)["art"]
        for s in sorted(starts_by_case[c]):
            if s + WIN > len(art):
                continue
            f = artwave.window_features(art[s:s + WIN])
            if np.isfinite(f["artw_upstroke"]):
                nwin += 1
                for k in artwave.FEATURES:
                    feats[k].append(f[k])
    row = {"cid": c, "nwin": nwin}
    for k in artwave.FEATURES:
        row[k] = float(np.nanmedian(feats[k])) if feats[k] else np.nan
    rows.append(row)
    if (n + 1) % 50 == 0:
        print(f"  {n+1}/{len(cases)} cases", flush=True)

cids = np.array([r["cid"] for r in rows])
np.savez_compressed("artwave_features.npz", cid=cids, nwin=np.array([r["nwin"] for r in rows]),
                    **{k: np.array([r[k] for r in rows]) for k in artwave.FEATURES})
print("cases with >=1 arterial window:", int((np.array([r['nwin'] for r in rows]) > 0).sum()), "/", len(cases))
print("saved artwave_features.npz")
