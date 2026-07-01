"""Extract the four PPG waveform features from one 30-s window of a finger pleth (PPG) signal.

These four — pulse amplitude, systolic upstroke time, pleth-variability index, and pulsus-alternans
power — plus one chart value (height) are the whole model. Only the pulse foot, systolic peak
and amplitude are needed, so the per-beat processing is deliberately simple (no dicrotic-notch or
2nd-derivative analysis). The per-window values are taken as the median over a patient's clean
windows (see build.py); the height value comes from the VitalDB clinical table.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

FS = 500                                   # Hz (VitalDB PPG)
WIN_SEC = 30
_BP = butter(4, (0.5, 10.0), btype="band", fs=FS, output="sos")


def _morph(ppg):
    """Band-pass the PPG to the pulse-morphology band (0.5-10 Hz)."""
    return sosfiltfilt(_BP, ppg - np.nanmean(ppg))


def _feet(pc):
    """Pulse onsets (feet) = prominent troughs of the normalised morphology PPG."""
    pn = (pc - pc.mean()) / (pc.std() + 1e-9)
    feet, _ = find_peaks(-pn, distance=int(0.35 * FS), prominence=0.3)
    return feet


def _resample(x, n):
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def beats(ppg_window):
    """Accepted beats of a window as dicts {foot, amp, upstroke}. A template-correlation gate
    (>=0.85) and a duration gate reject malformed/noisy beats."""
    pc = _morph(ppg_window); feet = _feet(pc)
    segs = [pc[i:j] for i, j in zip(feet[:-1], feet[1:])
            if int(0.45 * FS) <= j - i <= int(1.6 * FS) and pc[i:j].std() > 1e-9]
    tmpl = np.median([_resample(s - s.mean(), 100) for s in segs], 0) if len(segs) >= 5 else None
    out = []
    for fi, fj in zip(feet[:-1], feet[1:]):
        seg = pc[fi:fj]; n = len(seg)
        if not (int(0.45 * FS) <= n <= int(1.6 * FS)) or seg.max() - seg.min() < 1e-9:
            continue
        if tmpl is not None:
            b = _resample(seg - seg.mean(), 100)
            if b.std() < 1e-9 or np.corrcoef(b, tmpl)[0, 1] < 0.85:
                continue
        sp = int(np.argmax(seg))
        if sp < int(0.04 * FS) or sp > int(0.55 * n) or sp >= n - 4:
            continue
        out.append(dict(foot=fi, amp=float(seg[sp] - seg[0]), upstroke=sp / FS))
    return out


# ── the four PPG features (per window) ────────────────────────────────────────
def ppg_ac_amp(bts):
    """Median AC pulse amplitude (systolic peak − foot)."""
    a = [b["amp"] for b in bts]
    return float(np.median(a)) if a else np.nan


def ppg_upstroke(bts):
    """Median systolic upstroke time, foot→peak (s) — the dominant carrier."""
    u = [b["upstroke"] for b in bts]
    return float(np.median(u)) if u else np.nan


def ppg_pvi(bts):
    """Pleth Variability Index: respiratory amplitude variation over sliding 4-s windows
    (step 2 s), 100·(max−min)/(max+min) — the non-invasive pulse-pressure-variation analog."""
    amps = np.array([b["amp"] for b in bts]); bt = np.array([b["foot"] for b in bts], float) / FS
    if len(amps) < 6:
        return np.nan
    vals, s = [], bt.min()
    while s < bt.max() - 2.0:
        m = (bt >= s) & (bt < s + 4.0)
        if m.sum() >= 3:
            seg = amps[m]; vals.append(100.0 * (seg.max() - seg.min()) / (seg.max() + seg.min() + 1e-9))
        s += 2.0
    return float(np.median(vals)) if vals else np.nan


def ppg_alternans(bts):
    """Power in the period-2 (beat-to-beat) bin of the de-respired pulse-amplitude series —
    mechanical pulsus alternans (alternating strong/weak beats). Gain-invariant power fraction."""
    amps = np.array([b["amp"] for b in bts])
    if len(amps) < 15 or np.median(amps) < 1e-9:
        return np.nan
    mm = np.array([np.median(amps[max(0, i - 2):i + 3]) for i in range(len(amps))])
    r = amps - mm
    if np.std(r) < 1e-9:
        return np.nan
    S = np.abs(np.fft.rfft(r - r.mean())) ** 2
    return float(S[-1] / (S.sum() + 1e-12))


def extract_window(ppg_window):
    """The four PPG features for one 30-s window (NaN where a feature can't be computed)."""
    bts = beats(ppg_window)
    return dict(ppg_alternans=ppg_alternans(bts), ppg_ac_amp=ppg_ac_amp(bts),
                ppg_upstroke=ppg_upstroke(bts), ppg_pvi=ppg_pvi(bts))
