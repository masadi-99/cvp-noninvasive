"""Shared helpers for the collaborator follow-up experiments (window=90s, PPG-quality>0.7, CO/CI).

Design goals:
  * PPG-ONLY, window-length-agnostic re-implementation of the 4 canonical PPG features
    (ppg_alternans, ppg_ac_amp, ppg_upstroke, ppg_pvi). It reuses cvpkit's EXACT beat logic
    (morph_ppg / find_feet / window_template / beat_fiducials) so the beat SET is identical to
    the production pipeline — validated to reproduce the 0.756 baseline (see validate_baseline.py).
  * Re-compute the CVP label over an ARBITRARY window span from the raw 1-Hz cvp_numeric track,
    so the label always matches the exact feature window (required for the 90-s experiment).
  * A neurokit PPG-quality scorer = the collaborator's `_score_ppg_segment` (mean nk.ppg_quality).

Nothing here touches ECG — the model features and the window selection are pure PPG.
"""
import sys, os, warnings
sys.path.insert(0, "/home/masadi/cvp-ecg-ppg")
sys.path.insert(0, "/home/masadi/run")
sys.path.insert(0, "/home/masadi/run/analyses/enhancements")
warnings.filterwarnings("ignore")
import numpy as np
from cvpkit import config as C
from cvpkit.signals import morph_ppg, find_feet, window_template, beat_fiducials
from enh import evaluate                    # canonical repeated nested grouped-CV harness

FS = C.FS
CASES_DIR = C.CASES_DIR
FEATS = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi"]   # PPG-only; +height added per-case


# ── lean PPG-only feature extraction (beat set identical to cvpkit) ───────────
def ppg_beats(ppg_win):
    """(foot_idx, amp, upstroke_s) per accepted beat — same acceptance as cvpkit WindowContext."""
    pc = morph_ppg(ppg_win)
    feet = find_feet(pc)
    tmpl = window_template(pc, feet)
    out = []
    for fi, fj in zip(feet[:-1], feet[1:]):
        fid = beat_fiducials(pc, fi, fj, template=tmpl)
        if fid is None:
            continue
        seg = pc[fi:fj]; sp = fid["sys"] - fi
        out.append((fi, float(seg[sp] - seg[0]), sp / FS))
    return out


def feats_from_beats(beats):
    """The 4 canonical PPG features from a beat list (verbatim formulas from cvpkit.features)."""
    d = dict(ppg_alternans=np.nan, ppg_ac_amp=np.nan, ppg_upstroke=np.nan, ppg_pvi=np.nan)
    if not beats:
        return d
    amps = np.array([b[1] for b in beats]); foots = np.array([b[0] for b in beats], float)
    ups = np.array([b[2] for b in beats])
    d["ppg_ac_amp"] = float(np.median(amps))
    d["ppg_upstroke"] = float(np.median(ups))
    # pulsus alternans: period-2 power fraction of the de-respired amplitude residual
    if len(amps) >= 15 and np.median(amps) >= 1e-9:
        mm = np.array([np.median(amps[max(0, i - 2):i + 3]) for i in range(len(amps))])
        r = amps - mm
        if np.std(r) >= 1e-9:
            S = np.abs(np.fft.rfft(r - r.mean())) ** 2
            d["ppg_alternans"] = float(S[-1] / (S.sum() + 1e-12))
    # PVI: respiratory swing of pulse amplitude over sliding 4-s windows (step 2 s)
    if len(amps) >= 6:
        bt = foots / FS; vals = []; s = bt.min()
        while s < bt.max() - 2.0:
            m = (bt >= s) & (bt < s + 4.0)
            if m.sum() >= 3:
                seg = amps[m]; mx, mn = seg.max(), seg.min()
                vals.append(100.0 * (mx - mn) / (mx + mn + 1e-9))
            s += 2.0
        if vals:
            d["ppg_pvi"] = float(np.median(vals))
    return d


def ppg_features(ppg_win):
    return feats_from_beats(ppg_beats(ppg_win))


# ── PPG signal-quality score = collaborator's _score_ppg_segment (neurokit) ───
import neurokit2 as nk  # noqa: E402


def ppg_quality_score(ppg_win, fs=FS):
    """Mean neurokit PPG quality (templatematch, 0..1) over the window — the collaborator's
    `_score_ppg_segment`. Returns NaN if the window is unscorable."""
    w = np.asarray(ppg_win, float)
    if not np.isfinite(w).all() or w.std() < 1e-9:
        return np.nan
    try:
        p = nk.ppg_clean(w, sampling_rate=fs)
        q = np.asarray(nk.ppg_quality(p, sampling_rate=fs), float)
        q = q[np.isfinite(q)]
        return float(np.mean(q)) if len(q) else np.nan
    except Exception:
        return np.nan


# ── raw case access + CVP-label recompute over an arbitrary span ──────────────
def load_ppg_cvp(cid):
    """(ppg @500Hz, cvp_numeric @1Hz) for a case, or None. cvp_waveform is NOT loaded (huge)."""
    fp = os.path.join(CASES_DIR, f"case_{cid}.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp)
    return d["ppg"], d["cvp_numeric"]


def cvp_label(cvp_num, sec0, win_sec):
    """Median monitor CVP over [sec0, sec0+win_sec) — matches the exact feature window."""
    seg = cvp_num[sec0:sec0 + win_sec]; seg = seg[np.isfinite(seg)]
    return float(np.median(seg)) if len(seg) else np.nan


# ── per-case assembly + evaluation (adds height from the demo matrix) ─────────
def height_by_case():
    """cid -> height, from the cvpkit matrix (the one demographic in the 5-feature model)."""
    from cvpkit import aggregate
    Mk, ck, _ = aggregate.load_matrix(C.MATRIX_NPZ)
    cid = Mk[:, ck.index("cid")].astype(int); h = Mk[:, ck.index("height")].astype(float)
    out = {}
    for c in np.unique(cid):
        v = h[cid == c]; v = v[np.isfinite(v)]
        out[int(c)] = float(np.median(v)) if len(v) else np.nan
    return out


def per_case_matrix(rows, add_height=True):
    """rows: list of (cid, feat_dict, label). Aggregate per case by median.
    Returns X (with height col if add_height), y (label), cases, feat_names."""
    from collections import defaultdict
    by = defaultdict(list); yby = defaultdict(list)
    for cid, fd, lab in rows:
        by[cid].append(fd); yby[cid].append(lab)
    hbc = height_by_case() if add_height else {}
    cases = np.array(sorted(by))
    feat_names = FEATS + (["height"] if add_height else [])
    X = np.full((len(cases), len(feat_names)), np.nan); y = np.full(len(cases), np.nan)
    for i, c in enumerate(cases):
        for j, f in enumerate(FEATS):
            v = np.array([fd.get(f, np.nan) for fd in by[c]], float); v = v[np.isfinite(v)]
            X[i, j] = np.median(v) if len(v) else np.nan
        if add_height:
            X[i, len(FEATS)] = hbc.get(int(c), np.nan)
        yy = np.array(yby[c], float); yy = yy[np.isfinite(yy)]
        y[i] = np.median(yy) if len(yy) else np.nan
    ok = np.isfinite(y)
    return X[ok], y[ok], cases[ok], feat_names
