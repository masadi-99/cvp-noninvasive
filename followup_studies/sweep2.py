"""Configurable re-tiler = the deployed cvp/sweep.py pipeline, but with (a) a tunable window
length and (b) a selectable gate, and it extracts the 4 validated PPG features per window.

Every window goes through the FULL deployed path at the chosen length:
  tile (non-overlapping W-sample) -> reject NaN -> ECG R-peaks -> contemporaneous numeric CVP over
  the SAME W seconds -> GATE -> per-window PPG features (validated extractor) + numeric label.

GATE modes:
  trimodal : the deployed gate (ECG+PPG SQI + burst + CVP morphology). Reproduces medium.npz.
  ppgcvp   : drop the ECG *quality* gate (SQI/burst) but keep the PPG channel gate AND the CVP
             morphology gate. R-peaks are still used for CVP beat timing (the a/c/x/v/y fit is
             R-peak-referenced and cannot be done without them).

env: CVP_CASES_DIR, CVP_GATE, CVP_WINS (comma secs), CVP_CASELIST (npy of cids), CVP_WORKERS, CVP_TAG
Run:  CVP_WINS=30 CVP_GATE=trimodal python sweep2.py
"""
import os, sys, glob, json, time
os.environ.setdefault("CVP_CASES_DIR", "/home/masadi/cvp_data/cases")
sys.path.insert(0, "/home/masadi/cvp-ecg-ppg")
sys.path.insert(0, "/home/masadi/run")
sys.path.insert(0, "/home/masadi/run/analyses/collab_followups")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from scipy.signal import find_peaks

import cvp.config as K
from cvp.load import load_case
from cvp.sqi import detect_rpeaks, score_window, score_cvp_window
import cvp.eppg as EP
from cvp.labels import extract_labels
import common as CF                       # validated PPG feature extractor (ppg_features)

FS = K.FS
WINS = [int(x) for x in os.environ.get("CVP_WINS", "30").split(",")]
GATE = os.environ.get("CVP_GATE", "trimodal")
TAG = os.environ.get("CVP_TAG", GATE)
WORKERS = int(os.environ.get("CVP_WORKERS", "8"))
CVP_GATE_LO, CVP_GATE_HI = K.CVP_NUMERIC_GATE


# ── PPG-only channel gate (the PPG half of the deployed ECG+PPG gate; no ECG) ──
def _pp(seg):
    return float(np.percentile(seg, 95) - np.percentile(seg, 5)) if len(seg) > 10 else 0.0


def ppg_channel_ok(p_w, win_sec):
    if EP.longest_const_run(p_w) > K.FLAT_MAX_SEC * FS:
        return False
    ppg = EP.clean_ppg(p_w)
    if ppg.std() < 1e-6:
        return False
    pn = (ppg - ppg.mean()) / ppg.std()
    ppk, _ = find_peaks(pn, height=K.PPG_PEAK_HEIGHT_Z, distance=int(K.PPG_PEAK_MIN_DIST_SEC * FS))
    if len(ppk) < K.N_BEATS_MIN:
        return False
    if EP._template_corr(pn, ppk, int(K.PPG_TEMPL_HALF_SEC * FS)) < K.EPPG_SQI_MIN:
        return False
    # PPG within-window burst (the PPG part of score_eppg_burst)
    Wb = int(K.PPG_BURST_WIN_SEC * FS)
    pps = [_pp(p_w[s:s + Wb]) for s in range(0, len(p_w) - Wb + 1, FS)]
    ref = (np.median(pps) + 1e-6) if pps else 1e-6
    if any(v > K.PPG_BURST_FACTOR * ref for v in pps):
        return False
    return True


def contemp_numeric(num, s, win_sec):
    sec = s // FS
    cvn = num[sec:sec + win_sec]
    cvn = cvn[(~np.isnan(cvn)) & (cvn >= CVP_GATE_LO) & (cvn <= CVP_GATE_HI)]
    return float(np.median(cvn)) if len(cvn) >= K.MIN_VALID_CVP_SAMPLES else np.nan


def process_case(cid):
    try:
        d = load_case(cid)
    except Exception:
        return cid, {}
    ecg, ppg, cvp, cvp_raw, num = d["ecg"], d["ppg"], d["cvp"], d["cvp_raw"], d["numeric"]
    if np.isnan(ppg).all() or np.isnan(cvp).all():
        return cid, {}
    n = min(len(ecg), len(ppg), len(cvp))
    out = {}
    for W in WINS:
        K.WIN_SEC = W; K.WIN = W * FS                 # patch upstream window length
        WN = W * FS; rows = []
        for s in range(0, n - WN + 1, WN):
            e_w = ecg[s:s + WN]; p_w = ppg[s:s + WN]; c_w = cvp[s:s + WN]; cr_w = cvp_raw[s:s + WN]
            if np.isnan(e_w).any() or np.isnan(p_w).any() or np.isnan(c_w).any():
                continue
            rpk = detect_rpeaks(e_w)
            numv = contemp_numeric(num, s, W)
            if GATE == "trimodal":
                if not score_window(e_w, p_w, c_w, rpk, numeric=numv, cvp_raw=cr_w)["passed"]:
                    continue
            else:  # ppgcvp: PPG channel + CVP morphology, no ECG quality/burst
                if not ppg_channel_ok(p_w, W):
                    continue
                if not score_cvp_window(c_w, e_w, rpk, numeric=numv, cvp_raw=cr_w)["passed"]:
                    continue
            lab = extract_labels(c_w, rpk, numv)
            if lab is None:
                continue
            f = CF.ppg_features(p_w)
            rows.append((s, f["ppg_alternans"], f["ppg_ac_amp"], f["ppg_upstroke"], f["ppg_pvi"],
                         lab["numeric"]))
        out[W] = rows
    return cid, out


def main():
    if os.environ.get("CVP_CASELIST"):
        cids = [int(c) for c in np.load(os.environ["CVP_CASELIST"])]
    else:
        cids = sorted(int(os.path.basename(f).split("_")[1].split(".")[0])
                      for f in glob.glob(os.path.join(K.CASES_DIR, "case_*.npz")))
    print(f"sweep2 GATE={GATE} WINS={WINS} over {len(cids)} cases | workers={WORKERS}", flush=True)
    by_w = {W: [] for W in WINS}; t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (cid, out) in enumerate(pool.imap_unordered(process_case, cids, chunksize=4)):
            for W, rows in out.items():
                for r in rows:
                    by_w[W].append((cid,) + r)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(cids)}] " +
                      " ".join(f"W{W}={len(by_w[W])}" for W in WINS) + f"  {time.time()-t0:.0f}s", flush=True)

    # aggregate per case + evaluate, per W
    from cvpkit import demographics as DEMO
    from collections import defaultdict
    results = {}
    for W in WINS:
        rows = by_w[W]
        np.save(f"sweep_{TAG}_W{W}.npy", np.array(rows, float))   # (cid,start,alt,ac,up,pvi,numeric)
        byc_f = defaultdict(list); byc_y = defaultdict(list)
        for cid, s, alt, ac, up, pvi, numv in rows:
            byc_f[int(cid)].append((alt, ac, up, pvi)); byc_y[int(cid)].append(numv)
        cases = sorted(byc_f)
        X = np.full((len(cases), 5), np.nan); y = np.full(len(cases), np.nan)
        for i, c in enumerate(cases):
            A = np.array(byc_f[c], float)
            for j in range(4):
                col = A[:, j][np.isfinite(A[:, j])]
                X[i, j] = np.median(col) if len(col) else np.nan
            X[i, 4] = DEMO.demo_for(int(c)).get("height", np.nan)
            yy = np.array(byc_y[c], float); yy = yy[np.isfinite(yy)]
            y[i] = np.median(yy) if len(yy) else np.nan
        ok = np.isfinite(y); X, y, cs = X[ok], y[ok], np.array(cases)[ok]
        r = CF.evaluate(X, y, cs, threshold=12.0, reps=40)
        results[str(W)] = dict(win_sec=W, nwin=len(rows), auc=r["auc"], ci=r["ci"],
                               sens=r["sens"], spec=r["spec"], n=r["n"], npos=r["npos"])
        print(f"[{GATE}] W={W:3d}s  windows={len(rows):6d}  N={r['n']:4d} pos={r['npos']:3d}  "
              f"AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]", flush=True)
    json.dump(results, open(f"results_sweep_{TAG}.json", "w"), indent=2)
    print(f"saved results_sweep_{TAG}.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
