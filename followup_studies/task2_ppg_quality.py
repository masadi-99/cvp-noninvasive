"""TASK 2 — rely on PPG signal quality (neurokit ppg_quality > 0.7), dropping the ECG gate.

The deployed cohort (304) required a clean ECG AND a cardiac-locked CVP waveform. Since the model
uses PPG features only, ECG quality just shrinks the cohort. Two experiments:

  2a  QUALITY FILTER, fixed 304 cohort: keep only windows with PPG quality > 0.7 (vs no filter).
      -> does selecting high-quality PPG windows change AUC?

  2b  PPG-ONLY EXPANSION: for EVERY case with a usable CVP (no ECG requirement), select windows by
      PPG quality > 0.7 across the record, extract PPG features, per-case median, evaluate.
      -> how many more cases do we recover, and what is the AUC on the expanded cohort?

Same 5 features (height from the clinical table), same nested grouped CV, thr 12.
"""
import json, time, glob, os
import numpy as np
import common as K
from cvpkit import config as C, demographics as DEMO
from collections import defaultdict
from multiprocessing import Pool

QT = 0.7; WIN = 30; W = WIN * K.FS; FS = K.FS
CMAX, KMAX = 80, 20        # 2b: scan <=80 candidate windows spread across the record, keep <=20 with q>0.7


def _assemble(rows):
    """rows -> (X[+height], y, cases). Height from the clinical demo table (covers all cases)."""
    by = defaultdict(list); yby = defaultdict(list)
    for cid, fd, lab in rows:
        by[cid].append(fd); yby[cid].append(lab)
    cases = np.array(sorted(by)); feats = K.FEATS
    X = np.full((len(cases), len(feats) + 1), np.nan); y = np.full(len(cases), np.nan)
    for i, c in enumerate(cases):
        for j, f in enumerate(feats):
            v = np.array([fd.get(f, np.nan) for fd in by[c]], float); v = v[np.isfinite(v)]
            X[i, j] = np.median(v) if len(v) else np.nan
        X[i, -1] = DEMO.demo_for(int(c)).get("height", np.nan)
        yy = np.array(yby[c], float); yy = yy[np.isfinite(yy)]
        y[i] = np.median(yy) if len(yy) else np.nan
    ok = np.isfinite(y)
    return X[ok], y[ok], cases[ok]


# ── 2a: fixed 304 cohort, with / without the PPG-quality filter ───────────────
def run_2a():
    import enh
    _, _, casesb = enh.per_case(enh.FEATS5); base304 = [int(c) for c in casesb]
    d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
    cid = M[:, cols.index("cid")].astype(int); start = M[:, cols.index("start")].astype(int)
    anchors = defaultdict(list)
    for c, s in zip(cid, start):
        anchors[int(c)].append(int(s))
    rows_nf, rows_q = [], []; kept_q = 0; tot = 0
    for c in base304:
        lc = K.load_ppg_cvp(c)
        if lc is None:
            continue
        ppg, cvpn = lc; L = len(ppg)
        for s in anchors[c]:
            if s + W > L:
                continue
            w = ppg[s:s + W]
            if np.isnan(w).any():
                continue
            lab = K.cvp_label(cvpn, s // FS, WIN)
            if not np.isfinite(lab):
                continue
            tot += 1; fd = K.ppg_features(w); rows_nf.append((c, fd, lab))
            q = K.ppg_quality_score(w)
            if np.isfinite(q) and q > QT:
                rows_q.append((c, fd, lab)); kept_q += 1
    out = {}
    for tag, rows in [("no-filter", rows_nf), ("quality>0.7", rows_q)]:
        X, y, cs = _assemble(rows)
        r = K.evaluate(X, y, cs, threshold=12.0, reps=40)
        print(f"2a {tag:14s}: N={r['n']} pos={r['npos']} windows={len(rows)} AUC={r['auc']:.3f} "
              f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]", flush=True)
        out[tag] = {k: r[k] for k in ["auc", "ci", "n", "npos"]}; out[tag]["windows"] = len(rows)
    out["windows_kept_frac"] = kept_q / max(tot, 1)
    return out


# ── 2b: PPG-only expansion over every case with usable CVP ────────────────────
def process_case(cid):
    lc = K.load_ppg_cvp(cid)
    if lc is None:
        return None
    ppg, cvpn = lc; L = len(ppg); nwin = L // W
    if nwin < 1:
        return None
    cvpf = cvpn[np.isfinite(cvpn)]
    if len(cvpf) == 0 or not (0 < np.median(cvpf) < 60):
        return None
    idx = np.linspace(0, nwin - 1, min(nwin, CMAX)).astype(int)
    feats, labs = [], []
    for wi in idx:
        if len(feats) >= KMAX:
            break
        s = int(wi) * W; w = ppg[s:s + W]
        if len(w) < W or np.isnan(w).any():
            continue
        q = K.ppg_quality_score(w)
        if not (np.isfinite(q) and q > QT):
            continue
        lab = K.cvp_label(cvpn, s // FS, WIN)
        if not np.isfinite(lab):
            continue
        feats.append(K.ppg_features(w)); labs.append(lab)
    if not feats:
        return None
    return (int(cid), feats, labs)


def run_2b(base304):
    cids = sorted(int(os.path.basename(f).split("_")[1].split(".")[0])
                  for f in glob.glob(os.path.join(C.CASES_DIR, "case_*.npz")))
    print(f"2b scanning {len(cids)} cases (PPG q>0.7, no ECG gate)...", flush=True)
    rows = []; kept_cases = 0; t0 = time.time()
    with Pool(6) as pool:
        for i, res in enumerate(pool.imap_unordered(process_case, cids, chunksize=4)):
            if res is not None:
                cid, feats, labs = res; kept_cases += 1
                for fd, lab in zip(feats, labs):
                    rows.append((cid, fd, lab))
            if (i + 1) % 200 == 0:
                print(f"    [{i+1}/{len(cids)}] kept_cases={kept_cases} rows={len(rows)} {time.time()-t0:.0f}s", flush=True)
    X, y, cs = _assemble(rows)
    out = {}
    r = K.evaluate(X, y, cs, threshold=12.0, reps=40)
    print(f"2b EXPANDED PPG-only: N={r['n']} pos={r['npos']} AUC={r['auc']:.3f} "
          f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] Sens {r['sens']:.2f} Spec {r['spec']:.2f}", flush=True)
    out["expanded"] = {k: r[k] for k in ["auc", "ci", "sens", "spec", "n", "npos"]}
    # split: cases already in the 304 vs newly added
    csi = np.array([int(c) for c in cs]); isnew = np.array([c not in base304 for c in csi])
    out["n_total"] = int(len(cs)); out["n_new"] = int(isnew.sum()); out["n_overlap"] = int((~isnew).sum())
    for tag, mask in [("added-cases-only", isnew), ("overlap-with-304", ~isnew)]:
        if mask.sum() > 30:
            rr = K.evaluate(X[mask], y[mask], csi[mask], threshold=12.0, reps=40)
            print(f"2b {tag:18s}: N={rr['n']} pos={rr['npos']} AUC={rr['auc']:.3f}", flush=True)
            out[tag] = {k: rr[k] for k in ["auc", "ci", "n", "npos"]}
    return out


if __name__ == "__main__":
    import sys
    res = {}
    import enh
    _, _, cb = enh.per_case(enh.FEATS5); base304 = set(int(c) for c in cb)
    if len(sys.argv) < 2 or sys.argv[1] == "2a":
        res["2a"] = run_2a()
    if len(sys.argv) < 2 or sys.argv[1] == "2b":
        res["2b"] = run_2b(base304)
    prev = json.load(open("results_task2_ppg_quality.json")) if os.path.exists("results_task2_ppg_quality.json") else {}
    prev.update(res); json.dump(prev, open("results_task2_ppg_quality.json", "w"), indent=2)
    print("saved results_task2_ppg_quality.json")
