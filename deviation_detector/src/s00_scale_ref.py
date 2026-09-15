"""Step 0 (check). Recompute the VitalDB amplitude reference used to normalise MIMIC signals.

Median over the first 40 VitalDB cases (sorted by file name) of each case's raw-signal median and
inter-quartile range, on every 37th sample. PPG samples outside (-150, 150) are dropped first.
The values are frozen in config.SCALE_REF; this script recomputes them and confirms they match.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import VITALDB_CASES, SCALE_REF

mp, ip, me, ie = [], [], [], []
for p in sorted(glob.glob(f"{VITALDB_CASES}/case_*.npz"))[:40]:
    z = np.load(p)
    g = z["ppg"][::37].astype(float); g = g[np.isfinite(g) & (g > -150) & (g < 150)]
    e = z["ecg"][::37].astype(float); e = e[np.isfinite(e)]
    if len(g) > 1000:
        mp.append(np.median(g)); ip.append(np.percentile(g, 75) - np.percentile(g, 25))
    if len(e) > 1000:
        me.append(np.median(e)); ie.append(np.percentile(e, 75) - np.percentile(e, 25))
ref = dict(ppg_med=float(np.median(mp)), ppg_iqr=float(np.median(ip)),
           ecg_med=float(np.median(me)), ecg_iqr=float(np.median(ie)))
for k, v in ref.items():
    print(f"  {k:8s} recomputed {v:.6f}   frozen {SCALE_REF[k]:.6f}   "
          f"{'OK' if abs(v - SCALE_REF[k]) < 1e-9 else 'MISMATCH'}")
