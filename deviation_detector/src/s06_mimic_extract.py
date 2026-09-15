"""Step 6. Compute the SAME 91 features on every usable MIMIC record.

Per record: cut consecutive 30-s windows at 125 Hz, keep those with >= 90 % finite CVP and
>= 95 % finite ECG/PPG, a plausible CVP median and non-flat signals; resample ECG and PPG 4x to
500 Hz (polyphase; the features live at 0.5-10 Hz, far below the 62.5 Hz Nyquist of 125 Hz);
run the vendored VitalDB extractors. Label = median of the CVP waveform in the window.

AMPLITUDE NORMALISATION. The pleth is in arbitrary units in both datasets and MIMIC is
internally inconsistent (some records span 0-1, others 0-4) while VitalDB sits near 0-100; ECG
offset and gain differ too. Each MIMIC record's PPG and ECG are therefore mapped affinely onto
VitalDB's pooled median / IQR before extraction. Shape and ratio features are unaffected;
amplitude features become comparable instead of meaningless. The raw per-record median / IQR
are saved so the "device scale" can be studied as a confounder later.

Also saved per window, for the confounder analysis and never used as a feature: the mean
arterial pressure when an arterial line is present (sig_art).

    python src/s06_mimic_extract.py mimic3wdb     (SHARD=i NSHARD=n)
Writes results/mimic_feat/<dataset>_<record>.npz  with t (min), y (mmHg), X (n x 91), map, scale
"""
import os, sys, glob, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES_BASE, MIMIC_RECORDS, SCALE_REF
from lib.features import window_features
FS5, FSM, WIN5, WINM = 500, 125, 15000, 3750
OUT = f"{RES_BASE}/mimic_feat"


def rescale(x, med_ref, iqr_ref):
    """Affine map onto VitalDB's reference amplitude distribution, per record."""
    g = x[np.isfinite(x)]
    if len(g) < 1000: return x, (np.nan, np.nan)
    m = np.median(g); q = np.percentile(g, 75) - np.percentile(g, 25)
    if not np.isfinite(q) or q < 1e-9: return x, (float(m), float(q))
    return (x - m) * (iqr_ref / q) + med_ref, (float(m), float(q))


def do_record(path, maxwin=100000):
    rid = os.path.basename(path).replace(".npz", "")
    o = f"{OUT}/{os.path.basename(os.path.dirname(path))}_{rid}.npz"
    if os.path.exists(o): return "cached", 0
    z = np.load(path, allow_pickle=True)
    if not all(k in z.files for k in ("sig_cvp", "sig_ecg", "sig_ppg")): return "nosig", 0
    cvp, ecg, ppg = z["sig_cvp"], z["sig_ecg"], z["sig_ppg"]
    art = z["sig_art"] if "sig_art" in z.files else None
    ppg, ps = rescale(ppg.astype(np.float32), SCALE_REF["ppg_med"], SCALE_REF["ppg_iqr"])
    ecg, es = rescale(ecg.astype(np.float32), SCALE_REF["ecg_med"], SCALE_REF["ecg_iqr"])
    n = min(len(cvp), len(ecg), len(ppg)); nb = min(n // WINM, maxwin)
    T, Y, X, MAP = [], [], [], []
    for i in range(nb):
        a, b = i * WINM, (i + 1) * WINM
        c = cvp[a:b].astype(float); e = ecg[a:b].astype(float); g = ppg[a:b].astype(float)
        if np.isfinite(c).mean() < .9 or np.isfinite(e).mean() < .95 or np.isfinite(g).mean() < .95:
            continue
        cm = float(np.nanmedian(c))
        if not (-5 < cm < 40): continue
        if np.nanstd(e) < 1e-6 or np.nanstd(g) < 1e-6: continue
        e5 = resample_poly(np.nan_to_num(e, nan=float(np.nanmedian(e))), 4, 1)
        g5 = resample_poly(np.nan_to_num(g, nan=float(np.nanmedian(g))), 4, 1)
        if len(e5) != WIN5 or len(g5) != WIN5: continue
        row = window_features(g5, e5)
        if row is None or np.isfinite(row).sum() < 40: continue
        mp = np.nan
        if art is not None:
            aa = art[a:b].astype(float); aa = aa[np.isfinite(aa) & (aa > 20) & (aa < 250)]
            if len(aa) > 0.5 * WINM: mp = float(np.mean(aa))
        T.append(i * 0.5); Y.append(cm); X.append(row); MAP.append(mp)
    if not T: return "empty", 0
    np.savez_compressed(o, t=np.array(T, np.float32), y=np.array(Y, np.float32),
                        X=np.array(X, np.float32), map=np.array(MAP, np.float32),
                        scale=np.array([ps[0], ps[1], es[0], es[1]], np.float64),
                        rid=rid, src=os.path.basename(os.path.dirname(path)))
    return "ok", len(T)


if __name__ == "__main__":
    from s05_mimic_survey import usable
    ds = sys.argv[1] if len(sys.argv) > 1 else "mimic3wdb"
    lim = int(os.environ.get("LIMIT", "0"))
    sh = int(os.environ.get("SHARD", 0)); ns = int(os.environ.get("NSHARD", 1))
    keep = usable(ds)
    F = [p for p in sorted(glob.glob(f"{MIMIC_RECORDS}/{ds}/*.npz"))
         if os.path.basename(p).replace(".npz", "") in keep]
    if lim: F = F[:lim]
    F = F[sh::ns]
    print(f"{ds}: {len(F)} usable records for shard {sh}", flush=True)
    t0 = time.time(); tot = 0
    for i, p in enumerate(F):
        st, k = do_record(p); tot += k
        if i % 5 == 0 or st not in ("ok", "cached"):
            print(f"  [{sh}] {i+1}/{len(F)} {os.path.basename(p)} {st} {k}  "
                  f"{tot:,} windows [{time.time()-t0:.0f}s]", flush=True)
    print(f"  [{sh}] DONE {tot:,} windows [{time.time()-t0:.0f}s]", flush=True)
