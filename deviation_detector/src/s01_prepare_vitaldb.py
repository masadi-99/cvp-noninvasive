"""Step 1. Pull the VitalDB inputs out of the source archives into three small files under data/.

  data/vitaldb_windows.npz   the 39,466 gated 30-s windows of 815 patients: X (n x 91 ECG/PPG
                             features), y (the CVP a-wave label, mmHg), cid, t (minutes), names
  data/vitaldb_dense.npz     every 30-s tile of every case, gated or not: cid, t (minutes), the
                             end-expiratory CVP waveform summary (cvp_wf_ee, the label's own
                             quantity on the dense grid), the CVP waveform mean, and the monitor
                             numerics HR / MAP / SpO2 (never features; used to study confounding)
  data/vitaldb_clinical.csv  the clinical_data rows of those 815 cases (demographics)

The gated table is what the model is trained on. The dense grid supplies the calibration
baseline (many more tiles per block than the gated windows) and the artefact filter.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, VITALDB_TABLE, VITALDB_TILES, VITALDB_CLIN

D = np.load(VITALDB_TABLE, allow_pickle=True)
names = np.array([str(s) for s in D["names"]])
np.savez_compressed(f"{DATA}/vitaldb_windows.npz", X=D["X"].astype(np.float32),
                    y=D["y"].astype(np.float64), cid=D["cid"].astype(np.int64),
                    t=D["t"].astype(np.float64), names=names)
cids = np.unique(D["cid"])
print(f"windows: {D['X'].shape[0]:,} x {D['X'].shape[1]} features, {len(cids)} patients")

T = np.load(VITALDB_TILES, allow_pickle=True)
K = {str(c): i for i, c in enumerate(T["cols"])}; M = T["data"]
tc = M[:, K["cid"]].astype(np.int32)
keep = np.isin(tc, cids)
np.savez_compressed(f"{DATA}/vitaldb_dense.npz", cid=tc[keep],
                    t=(M[keep, K["t_sec"]] / 60.0).astype(np.float64),
                    cvp_wf_ee=M[keep, K["cvp_wf_ee"]].astype(np.float64),
                    cvp_wf_mean=M[keep, K["cvp_wf_mean"]].astype(np.float64),
                    hr=M[keep, K["hr_n"]].astype(np.float64),
                    map=M[keep, K["art_mbp_n"]].astype(np.float64),
                    spo2=M[keep, K["spo2_n"]].astype(np.float64))
print(f"dense tiles: {keep.sum():,} of {len(tc):,} belong to the 815 patients")

rows = [r for r in csv.DictReader(open(VITALDB_CLIN, encoding="utf-8-sig"))
        if int(r["caseid"]) in set(cids.tolist())]
cols = ["caseid", "age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype",
        "approach", "position", "ane_type", "preop_htn", "preop_dm", "preop_ecg", "preop_hb",
        "preop_cr", "preop_alb", "icu_days", "death_inhosp", "intraop_ebl", "intraop_crystalloid",
        "intraop_colloid", "intraop_rbc", "casestart", "caseend", "opstart", "opend", "opname", "dx"]
with open(f"{DATA}/vitaldb_clinical.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r[c] for c in cols})
print(f"clinical rows: {len(rows)}")
