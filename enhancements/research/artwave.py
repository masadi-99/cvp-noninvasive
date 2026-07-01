"""Rich arterial-pressure WAVEFORM morphology features (analogous to the PPG feature set), from the
raw ART trace (SNUADC/ART, 500 Hz, cached in cvp_data/aux/aux_<cid>.npz). For each 30-s window we
detect foot-to-foot beats and read a full set of shape / timing / dynamic features."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

FS = 500
_LP = butter(4, 20, "low", fs=FS, output="sos")
_area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
FEATURES = ["artw_upstroke", "artw_sys_frac", "artw_notch_frac", "artw_aug_index", "artw_dic_height",
            "artw_sysdia_area", "artw_decay_tau", "artw_width50", "artw_dpdt_max", "artw_dndt_max",
            "artw_reflect_idx", "artw_alternans", "artw_amp_cv", "artw_ppv", "artw_spv"]


def _clean(x):
    m = np.isfinite(x)
    if m.sum() < FS * 5:
        return None
    xi = x.astype(float).copy()
    if not m.all():
        idx = np.arange(len(x)); xi[~m] = np.interp(idx[~m], idx[m], x[m])
    return sosfiltfilt(_LP, xi)


def _alternans(amp):
    """Period-2 (beat-to-beat) power fraction of the systolic-amplitude series, de-respired."""
    x = np.asarray(amp, float)
    if len(x) < 8:
        return np.nan
    x = x - np.convolve(x, np.ones(5) / 5, mode="same")   # remove slow (respiratory) drift
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    if X.sum() < 1e-12:
        return np.nan
    return float(X[-1] / X.sum())                          # last bin = period-2 (alternating)


def _decay_tau(dia, floor):
    """Windkessel diastolic decay time constant (s) from the runoff (notch -> foot)."""
    y = dia - floor
    ok = y > 0.5
    if ok.sum() < 6:
        return np.nan
    t = np.arange(len(dia))[ok] / FS
    try:
        slope = np.polyfit(t, np.log(y[ok]), 1)[0]
    except Exception:
        return np.nan
    return float(np.clip(-1.0 / (slope - 1e-9), 0.05, 3.0)) if slope < 0 else np.nan


def window_features(art_win):
    out = {k: np.nan for k in FEATURES}
    a = _clean(art_win)
    if a is None:
        return out
    feet, _ = find_peaks(-a, distance=int(0.4 * FS), prominence=4)
    if len(feet) < 6:
        return out
    ups, sysf, notchf, aug, dic, sda, tau, w50, dpdt, dndt, refl = ([] for _ in range(11))
    amps, syst = [], []
    for f0, f1 in zip(feet[:-1], feet[1:]):
        seg = a[f0:f1]; n = len(seg)
        if n < int(0.4 * FS) or n > int(1.6 * FS):
            continue
        sp = int(np.argmax(seg)); amp = seg[sp] - seg[0]
        if amp < 5 or sp < int(0.03 * FS) or sp > int(0.5 * n) or not (40 <= seg[sp] <= 260):
            continue
        d1 = np.gradient(seg)
        # dicrotic notch: first upward zero-crossing of slope on the descending limb (a real incisura),
        # else the 2nd-derivative shoulder
        desc = np.arange(sp + int(0.04 * n), min(n - 2, sp + int(0.6 * n)))
        notch = None
        if len(desc) > 3:
            up = np.where((d1[desc[:-1]] < 0) & (d1[desc[1:]] >= 0))[0]
            if len(up):
                notch = desc[up[0]]
            else:
                d2 = np.gradient(d1)
                notch = desc[int(np.argmax(d2[desc]))]
        if notch is None or notch <= sp:
            notch = min(n - 2, sp + int(0.3 * n))
        # dicrotic peak (reflected wave) = local max after the notch
        post = np.arange(notch, min(n - 1, notch + int(0.35 * n)))
        dicpk = post[int(np.argmax(seg[post]))] if len(post) > 2 else notch
        ups.append(sp / FS)
        sysf.append(notch / n)
        notchf.append(notch / n)
        aug.append((seg[dicpk] - seg[0]) / amp)                 # augmentation index
        dic.append((seg[notch] - seg[0]) / amp)                 # dicrotic notch height (rel.)
        sysA = _area(seg[:notch] - seg[0]); diaA = _area(seg[notch:] - seg[0])
        sda.append(sysA / (diaA + 1e-6))                         # systolic/diastolic area ratio
        tau.append(_decay_tau(seg[notch:], seg[-1]))
        w = np.where(seg - seg[0] >= 0.5 * amp)[0]
        w50.append((w[-1] - w[0]) / FS if len(w) > 1 else np.nan)
        dpdt.append(float(np.max(d1[:sp + 1]) * FS) if sp > 0 else np.nan)
        dndt.append(float(-np.min(d1[sp:]) * FS))
        refl.append((dicpk - sp) / FS)                          # reflected-wave transit time
        amps.append(amp); syst.append(seg[sp])
    if len(amps) < 5:
        return out
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    for k, v in zip(["artw_upstroke", "artw_sys_frac", "artw_notch_frac", "artw_aug_index", "artw_dic_height",
                     "artw_sysdia_area", "artw_decay_tau", "artw_width50", "artw_dpdt_max", "artw_dndt_max",
                     "artw_reflect_idx"], [ups, sysf, notchf, aug, dic, sda, tau, w50, dpdt, dndt, refl]):
        out[k] = med(np.array(v, float))
    amps = np.array(amps, float); syst = np.array(syst, float)
    out["artw_alternans"] = _alternans(amps)
    out["artw_amp_cv"] = float(np.std(amps) / (np.mean(amps) + 1e-9))
    out["artw_ppv"] = float(100 * (np.percentile(amps, 95) - np.percentile(amps, 5)) / (np.median(amps) + 1e-9))
    out["artw_spv"] = float(np.percentile(syst, 95) - np.percentile(syst, 5))
    return out
