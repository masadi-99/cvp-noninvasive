"""Generate self-contained CSVs for the public repo's enhancements/ directory:
  features_ext.csv  — per-case: the 5 model features + arterial numerics + Dawber class fractions + CVP
  windows.csv       — per-window: the 5 features + arterial numerics + per-window CVP (for the windowing study)
Derived once from the full private pipeline; the repo scripts then reproduce all results from these files."""
import csv, numpy as np, enh
THR = 12.0; FE = enh.FEATS5
d = enh.load_merged(); cid = d["cid"]
Xc, yc, cases = enh.per_case(FE); ybc = (yc > THR).astype(int); cidx = {c: i for i, c in enumerate(cases)}
rows_by_case = {c: np.where(cid == c)[0] for c in cases}
ART = ["art_sbp_n", "art_dbp_n", "art_mbp_n"]
pcmed = lambda f: np.array([np.nanmedian(d["F"][f][rows_by_case[c]]) for c in cases])
artpc = {f: pcmed(f) for f in ART}
m = np.load("morph_features.npz"); mp = {int(c): i for i, c in enumerate(m["cid"])}
mo = np.array([mp[int(c)] for c in cases]); f1 = m["morph_f1"][mo]; f4 = m["morph_f4"][mo]
r = lambda v: "" if not np.isfinite(v) else round(float(v), 5)

with open("/home/masadi/cvp-noninvasive/enhancements/data/features_ext.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["caseid", "ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height",
                                    "art_sbp_n", "art_dbp_n", "art_mbp_n", "morph_f1", "morph_f4", "cvp_numeric", "elevated"])
    for i, c in enumerate(cases):
        w.writerow([int(c)] + [r(Xc[i, j]) for j in range(5)] + [r(artpc[a][i]) for a in ART] +
                   [r(f1[i]), r(f4[i]), round(float(yc[i]), 2), int(ybc[i])])

with open("/home/masadi/cvp-noninvasive/enhancements/data/windows.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["caseid", "ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height",
                                    "art_sbp_n", "art_dbp_n", "art_mbp_n", "cvp_window"])
    F = d["F"]
    for c in cases:
        for idx in rows_by_case[c]:
            w.writerow([int(c)] + [r(F[f][idx]) for f in FE] + [r(F[a][idx]) for a in ART] + [r(d["numeric"][idx])])
print("features_ext.csv rows:", len(cases))
import subprocess; print(subprocess.run(["wc","-l","/home/masadi/cvp-noninvasive/enhancements/data/windows.csv"],capture_output=True,text=True).stdout.strip())
