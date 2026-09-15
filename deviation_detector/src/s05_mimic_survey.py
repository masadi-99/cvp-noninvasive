"""Step 5. Survey the MIMIC-III / MIMIC-IV waveform records: which carry enough joint CVP+ECG+PPG?

A record is usable if it has at least 25 minutes of consecutive 30-s blocks in which CVP, ECG
and PPG are all >= 90 % finite, the CVP median is plausible (-5..40 mmHg) and neither ECG nor
PPG is flat. That is one 10-min calibration block plus one 15-min tracking stretch.

Reads DATA ONLY from the record archive; nothing from the project that built it is used.
    python src/s05_mimic_survey.py mimic3wdb      (SHARD=i NSHARD=n to split the work)
    python src/s05_mimic_survey.py mimic4wdb
Writes results/mimic/survey_<dataset>_<shard>.json
"""
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES_BASE, MIMIC_RECORDS
OUT = f"{RES_BASE}/mimic"; os.makedirs(OUT, exist_ok=True)


def survey(p):
    try:
        z = np.load(p, allow_pickle=True, mmap_mode=None)
    except Exception as e:
        return dict(rec=os.path.basename(p), err=str(e)[:60])
    have = set(z.files)
    meta = {}
    if "__meta__" in have:
        try: meta = json.loads(str(z["__meta__"]))
        except Exception: meta = {}
    fs = float(meta.get("fs", 125.0))
    need = ["sig_cvp", "sig_ecg", "sig_ppg"]
    if not all(k in have for k in need):
        return dict(rec=os.path.basename(p), fs=fs, missing=[k for k in need if k not in have])
    cvp, ecg, ppg = (z[k] for k in need)
    n = min(len(cvp), len(ecg), len(ppg))
    W = int(30 * fs)
    nb = n // W
    ok = np.zeros(nb, bool)
    for i in range(0, nb, 1):
        a, b = i * W, (i + 1) * W
        c = cvp[a:b]; e = ecg[a:b]; g = ppg[a:b]
        fc = np.isfinite(c).mean(); fe = np.isfinite(e).mean(); fg = np.isfinite(g).mean()
        if fc < .9 or fe < .9 or fg < .9: continue
        cm = np.nanmedian(c)
        if not (-5 < cm < 40): continue
        if np.nanstd(e) < 1e-6 or np.nanstd(g) < 1e-6: continue
        ok[i] = True
    best = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return dict(rec=os.path.basename(p).replace(".npz", ""), fs=fs,
                hours=n / fs / 3600.0, blocks=int(nb), ok_blocks=int(ok.sum()),
                ok_frac=float(ok.mean()), longest_run_min=best * 0.5,
                cvp_med=float(np.nanmedian(cvp[np.isfinite(cvp)][:2000000])) if np.isfinite(cvp).any() else np.nan,
                has_art=int("sig_art" in have))


def usable(ds):
    """Record ids with >= 25 min of joint coverage, from the survey files."""
    keep = set()
    for p in glob.glob(f"{OUT}/survey_{ds}_*.json"):
        for r in json.load(open(p)):
            if r.get("longest_run_min", 0) >= 25: keep.add(r["rec"])
    return keep


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "mimic3wdb"
    sh = int(os.environ.get("SHARD", 0)); ns = int(os.environ.get("NSHARD", 1))
    F = sorted(glob.glob(f"{MIMIC_RECORDS}/{ds}/*.npz"))[sh::ns]
    t0 = time.time(); rows = []
    for i, p in enumerate(F):
        rows.append(survey(p))
        if i % 20 == 0:
            u = sum(1 for r in rows if r.get("longest_run_min", 0) >= 25)
            print(f"  [{sh}] {i+1}/{len(F)}  usable so far {u}  [{time.time()-t0:.0f}s]", flush=True)
    with open(f"{OUT}/survey_{ds}_{sh}.json", "w") as f: json.dump(rows, f)
    u = [r for r in rows if r.get("longest_run_min", 0) >= 25]
    print(f"  [{sh}] DONE {len(rows)} records, {len(u)} with >=25 min of joint CVP+ECG+PPG "
          f"[{time.time()-t0:.0f}s]", flush=True)
