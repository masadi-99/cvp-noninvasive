"""The 91 ECG/PPG window features, vendored VERBATIM from the VitalDB extractors.

One 30-second window at 500 Hz in, one dict of 91 numbers out. Three families:
  39 ppg_*                 pulse geometry, amplitude, notch, respiratory modulation, ensemble shape,
                           Dawber damping class          (temporal_sweep.ppg_features)
  32 nf_*                  higher-order ensemble shape, SDPPG a-e, pulse-arrival time vs the ECG,
                           respiratory phase/coherence, beat-to-beat dynamics, two-Gaussian
                           decomposition                 (novel_features.features)
  20 p_/rr_/af_/qrs_/r_    ECG P wave, PR interval, RR statistics, QRS width, R amplitude
                           (the per-window body of pwave.one)

The beat pipeline underneath (morphology band-pass -> feet -> window template -> template-gated
fiducials) and the R-peak detector are the deployed, audited ones from cvpkit / cvp-ecg-ppg. The
code is copied, not re-derived, so a MIMIC window and a VitalDB window go through byte-identical
arithmetic. Constants are inlined from the source packages' config.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks, welch

FS = 500; WIN_SEC = 30; WIN = FS * WIN_SEC

# ---------------------------------------------------------------------------------------------
# ECG cleaning and R peaks  (cvp-ecg-ppg/cvp/eppg.py + config.py)
# ---------------------------------------------------------------------------------------------
ECG_HP_HZ, ECG_LP_HZ, POWERLINE_HZ, NOTCH_Q = 0.5, 40.0, 60.0, 30
R_PEAK_HEIGHT_Z, R_PEAK_MIN_DIST_SEC = 2.0, 0.3
_ECG_HP = butter(3, ECG_HP_HZ, btype="high", fs=FS, output="sos")
_ECG_LP = butter(3, ECG_LP_HZ, btype="low", fs=FS, output="sos")


def clean_ecg(x):
    """Band-pass + 60 Hz notch on a NaN-free ECG window."""
    y = sosfiltfilt(_ECG_LP, sosfiltfilt(_ECG_HP, x))
    bn, an = iirnotch(POWERLINE_HZ, NOTCH_Q, FS)
    return filtfilt(bn, an, y)


def detect_rpeaks(raw_ecg):
    """R-peak sample indices: peaks of the z-normalised cleaned ECG above 2 SD, >= 0.3 s apart."""
    e = clean_ecg(raw_ecg)
    if e.std() < 1e-6:
        return np.array([], int)
    z = (e - e.mean()) / e.std()
    pk, _ = find_peaks(z, height=R_PEAK_HEIGHT_Z, distance=int(R_PEAK_MIN_DIST_SEC * FS))
    return pk


# ---------------------------------------------------------------------------------------------
# PPG beat pipeline  (run/cvpkit/signals.py)
# ---------------------------------------------------------------------------------------------
_BP = butter(4, (0.5, 10.0), btype="band", fs=FS, output="sos")   # PPG morphology band


def morph_ppg(raw):
    """Band-pass the PPG to the morphology band (0.5-10 Hz)."""
    return sosfiltfilt(_BP, raw - np.nanmean(raw))


def find_feet(pc):
    """Pulse onsets (feet) = prominent troughs of the normalized morphology PPG."""
    pn = (pc - pc.mean()) / (pc.std() + 1e-9)
    feet, _ = find_peaks(-pn, distance=int(0.35 * FS), prominence=0.3)
    return feet


def _resample(x, n):
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def window_template(pc, feet):
    """Median normalized beat across the window - used as a per-beat quality gate."""
    beats = []
    for fi, fj in zip(feet[:-1], feet[1:]):
        seg = pc[fi:fj]
        if int(0.45 * FS) <= len(seg) <= int(1.6 * FS) and seg.std() > 1e-9:
            beats.append(_resample(seg - seg.mean(), 100))
    return np.median(np.asarray(beats), 0) if len(beats) >= 5 else None


def beat_fiducials(pc, fi, fj, template=None, corr_min=0.85):
    """Fiducials for one beat (foot fi to foot fj), or None if the beat is rejected.

    Systolic peak = max; dicrotic notch = the 'e' wave (max of the acceleration PPG on the
    descending limb); dicrotic peak = next PPG local max after it. A template-correlation gate
    (>= 0.85 against the window's median beat) rejects malformed or noisy beats."""
    seg = pc[fi:fj]; n = len(seg)
    if n < int(0.45 * FS) or n > int(1.6 * FS):
        return None
    amp = seg.max() - seg.min()
    if amp < 1e-9:
        return None
    if template is not None:
        b = _resample(seg - seg.mean(), len(template))
        if b.std() < 1e-9 or np.corrcoef(b, template)[0, 1] < corr_min:
            return None
    sp = int(np.argmax(seg))
    if sp < int(0.04 * FS) or sp > int(0.55 * n) or sp >= n - 4:
        return None
    d2 = np.gradient(np.gradient(seg))
    notch = dia = None
    lo, hi = sp + int(0.06 * n), min(n - 2, sp + int(0.55 * n))
    if hi - lo >= 3:
        cn = lo + int(np.argmax(d2[lo:hi]))
        if d2[cn] > 0 and 0.15 <= cn / n <= 0.60:
            notch = cn
            d1p = np.gradient(seg[cn:min(n - 1, cn + int(0.45 * n))])
            cross = np.where((np.sign(d1p[:-1]) >= 0) & (np.sign(d1p[1:]) < 0))[0]
            dia = cn + int(cross[0]) + 1 if len(cross) else cn
    return dict(foot=fi, sys=fi + sp,
                notch=(fi + notch) if notch is not None else None,
                dia=(fi + dia) if dia is not None else None,
                dur=n, sys_amp=float(seg[sp] - seg[0]),
                notch_rel=(notch / n) if notch is not None else np.nan)


# ---------------------------------------------------------------------------------------------
# Dawber four-class pulse morphology  (run/analyses/enhancements/dawber.py)
# ---------------------------------------------------------------------------------------------
REGION = (6, 82); RISE_MIN = 0.010; FLAT_EPS = 0.0020; D2_MIN = 0.0010


def _descending(seg):
    if len(seg) < 12:
        return None
    sp = int(np.argmax(seg)); A = seg[sp] - seg[0]
    if A <= 1e-9 or sp >= len(seg) - 6:
        return None
    desc = seg[sp:]
    sn = (desc - desc[0]) / A
    g = np.linspace(0, len(sn) - 1, 100)
    return np.interp(g, np.arange(len(sn)), sn)


def dawber_classify(seg):
    """1 = true dicrotic notch, 2 = plateau, 3 = shoulder only, 4 = smooth monotone decay."""
    sn = _descending(seg)
    if sn is None:
        return None
    d1 = np.gradient(sn); d2 = np.gradient(d1)
    lo, hi = REGION
    r1, r2 = d1[lo:hi], d2[lo:hi]
    up = np.where((r1[:-1] < 0) & (r1[1:] >= 0))[0]
    if len(up):
        i = up[0] + lo
        post = sn[i:hi]
        if post.size and (post.max() - sn[i]) > RISE_MIN:
            return 1
    if r1.max() > -FLAT_EPS:
        return 2
    if r2.max() > D2_MIN:
        return 3
    return 4


# ---------------------------------------------------------------------------------------------
# 39 PPG features  (run/analyses/temporal_tracking/scripts/temporal_sweep.py :: ppg_features)
# ---------------------------------------------------------------------------------------------
PPG_FEATS = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi",
             "ppg_rate", "ppg_nbeat", "ppg_crest", "ppg_w50", "ppg_area", "ppg_amp_cv",
             "ppg_dc", "ppg_riiv", "ppg_resp_rate", "ppg_lf_frac",
             "ppg_notch_frac", "ppg_notch_rel", "ppg_dic_height", "ppg_ibi_rmssd",
             "ppg_skew", "ppg_kurt", "ppg_amp_slope",
             "ppg_vci", "ppg_riiv_norm", "ppg_resp_pow_frac", "ppg_baseline_amp",
             "ppg_alt_amp", "ppg_ibi_cv", "ppg_up_norm", "ppg_dia_frac",
             "ppg_awave", "ppg_tau", "ppg_dct1", "ppg_dct2", "ppg_dct3", "ppg_dct4", "ppg_dct5",
             "ppg_morph_f1", "ppg_morph_f4", "ppg_morph_mean"]
_SLOW = butter(4, (0.04, 10.0), "band", fs=FS, output="sos")   # keeps the respiratory baseline
_RESP = butter(3, (0.1, 0.5), "band", fs=FS, output="sos")     # respiratory band


def ppg_features(raw):
    d = {k: np.nan for k in PPG_FEATS}
    if raw is None or len(raw) < WIN // 2 or not np.isfinite(raw).all() or np.std(raw) < 1e-9:
        return d
    raw = raw.astype(float)
    pc = morph_ppg(raw)
    slow = sosfiltfilt(_SLOW, raw - np.mean(raw))
    feet = find_feet(pc)
    tmpl = window_template(pc, feet)

    d["ppg_dc"] = float(np.mean(raw))
    f, P = welch(raw - np.mean(raw), fs=FS, nperseg=min(4096, len(raw)))
    tot = P.sum() + 1e-12
    d["ppg_lf_frac"] = float(P[(f >= 0.04) & (f < 0.5)].sum() / tot)
    rb = (f >= 0.1) & (f <= 0.5)
    if rb.any() and P[rb].sum() > 0:
        d["ppg_resp_rate"] = float(f[rb][np.argmax(P[rb])] * 60.0)
    d["ppg_skew"] = float(((pc - pc.mean()) ** 3).mean() / (pc.std() ** 3 + 1e-12))
    d["ppg_kurt"] = float(((pc - pc.mean()) ** 4).mean() / (pc.std() ** 4 + 1e-12))
    respb = sosfiltfilt(_RESP, slow)
    d["ppg_riiv"] = float(np.percentile(respb, 95) - np.percentile(respb, 5))

    amps, ups, crests, w50s, areas, nrel, dich, foots, diafr = [], [], [], [], [], [], [], [], []
    dawber = []
    nnotch = 0
    for fi, fj in zip(feet[:-1], feet[1:]):
        fid = beat_fiducials(pc, fi, fj, template=tmpl)
        if fid is None:
            continue
        seg = pc[fi:fj]; n = len(seg); sp = fid["sys"] - fi
        a = float(seg[sp] - seg[0])
        if a <= 0:
            continue
        kcl = dawber_classify(seg)
        if kcl is not None:
            dawber.append(kcl)
        foots.append(fi); amps.append(a); ups.append(sp / FS); crests.append((n - sp) / FS)
        areas.append(float(np.sum(seg - seg[0])) / FS)
        th = seg[0] + 0.5 * a; ab = np.where(seg >= th)[0]
        w50s.append((ab[-1] - ab[0]) / FS if len(ab) > 1 else np.nan)
        if fid["notch"] is not None:
            nnotch += 1
            ni = fid["notch"] - fi
            nrel.append(ni / n)
            diafr.append(float(np.sum(seg[ni:] - seg[0])) / (np.sum(seg - seg[0]) + 1e-9))
            if fid["dia"] is not None:
                di = fid["dia"] - fi
                dich.append(float(seg[di] - seg[ni]) / (a + 1e-9))
    nb = len(amps)
    d["ppg_nbeat"] = float(nb)
    if nb == 0:
        return d
    amps = np.asarray(amps, float); foots = np.asarray(foots, float)
    med_a = float(np.median(amps))
    d["ppg_ac_amp"] = med_a
    d["ppg_upstroke"] = float(np.median(ups))
    d["ppg_crest"] = float(np.median(crests))
    d["ppg_area"] = float(np.median(areas))
    d["ppg_w50"] = float(np.nanmedian(w50s))
    d["ppg_rate"] = 60.0 * nb / WIN_SEC
    d["ppg_notch_frac"] = nnotch / nb
    if nrel:
        d["ppg_notch_rel"] = float(np.median(nrel))
    if dich:
        d["ppg_dic_height"] = float(np.median(dich))
    if diafr:
        d["ppg_dia_frac"] = float(np.median(diafr))
    if nb > 3:
        d["ppg_amp_cv"] = float(np.std(amps) / (np.mean(amps) + 1e-9))
    if dawber:
        dk = np.asarray(dawber, float)
        d["ppg_morph_f1"] = float(np.mean(dk == 1))
        d["ppg_morph_f4"] = float(np.mean(dk == 4))
        d["ppg_morph_mean"] = float(np.mean(dk))
    if nb >= 15 and med_a > 1e-9:
        mm = np.array([np.median(amps[max(0, i - 2):i + 3]) for i in range(nb)])
        r = amps - mm
        if np.std(r) >= 1e-9:
            S = np.abs(np.fft.rfft(r - r.mean())) ** 2
            d["ppg_alternans"] = float(S[-1] / (S.sum() + 1e-12))
            d["ppg_alt_amp"] = float(np.sqrt(2.0 * S[-1] / len(r)) / (med_a + 1e-9))
    if nb >= 6:
        bt = foots / FS; vals = []; s0 = bt.min()
        while s0 < bt.max() - 2.0:
            m = (bt >= s0) & (bt < s0 + 4.0)
            if m.sum() >= 3:
                sg = amps[m]; mx, mn = sg.max(), sg.min()
                vals.append(100.0 * (mx - mn) / (mx + mn + 1e-9))
            s0 += 2.0
        if vals:
            d["ppg_pvi"] = float(np.median(vals))
    bl = slow[foots.astype(int)]
    if len(bl) >= 6:
        d["ppg_baseline_amp"] = float(np.percentile(bl, 95) - np.percentile(bl, 5))
        d["ppg_riiv_norm"] = float(d["ppg_riiv"] / (med_a + 1e-9))
        d["ppg_vci"] = float(d["ppg_baseline_amp"] / (med_a + 1e-9))
        fb, Pb = welch(bl - bl.mean(), fs=max(len(bl) / WIN_SEC, 1e-6), nperseg=min(64, len(bl)))
        if Pb.sum() > 0:
            d["ppg_resp_pow_frac"] = float(Pb[(fb >= 0.1) & (fb <= 0.6)].sum() / (Pb.sum() + 1e-12))
    ibi = np.diff(foots) / FS
    if len(ibi) > 3:
        d["ppg_ibi_rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
        d["ppg_ibi_cv"] = float(np.std(ibi) / (np.mean(ibi) + 1e-9))
        d["ppg_up_norm"] = float(np.median(ups) / (np.mean(ibi) + 1e-9))
    if nb >= 6:
        d["ppg_amp_slope"] = float(np.polyfit(foots / FS, amps, 1)[0])
    if nb >= 8:
        norm = []
        for fi, fj in zip(feet[:-1], feet[1:]):
            fid = beat_fiducials(pc, fi, fj, template=tmpl)
            if fid is None:
                continue
            seg = pc[fi:fj].astype(float)
            a = seg.max() - seg.min()
            if a <= 1e-9 or len(seg) < 20:
                continue
            g = np.interp(np.linspace(0, len(seg) - 1, 100), np.arange(len(seg)), seg)
            norm.append((g - g.min()) / a)
        if len(norm) >= 8:
            Tm = np.median(np.asarray(norm), axis=0)
            k = np.arange(100)
            for q in range(1, 6):
                d[f"ppg_dct{q}"] = float(np.sum(Tm * np.cos(np.pi * q * (2 * k + 1) / 200.0)) * 2 / 100.0)
            tail = Tm[55:]
            if tail.min() >= 0 and tail[0] > tail[-1] + 1e-6:
                yv = tail - tail.min() + 1e-6
                try:
                    sl = np.polyfit(np.arange(len(yv)) / 100.0, np.log(yv), 1)[0]
                    if sl < 0:
                        d["ppg_tau"] = float(-1.0 / sl)
                except Exception:
                    pass
            late = Tm[75:]
            if len(late) > 6:
                base = np.polyval(np.polyfit(np.arange(len(late)), late, 1), np.arange(len(late)))
                d["ppg_awave"] = float(np.max(late - base))
    return d


# ---------------------------------------------------------------------------------------------
# 32 nf_* features  (cvp-step-examples/src/novel_features.py :: features)
# ---------------------------------------------------------------------------------------------
NF_FEATS = (["nf_dct%d" % q for q in range(6, 13)] +
            ["nf_curv", "nf_asym", "nf_area1", "nf_area2", "nf_area3",
             "nf_sd_b", "nf_sd_c", "nf_sd_d", "nf_sd_e",
             "nf_pat", "nf_pat_cv", "nf_pat_rr", "nf_slope_delay",
             "nf_resp_phase", "nf_resp_coh", "nf_mod_ratio",
             "nf_sd1", "nf_sd2", "nf_sd12", "nf_ibi_sd1", "nf_amp_ibi_xcorr", "nf_dfa",
             "nf_g_ratio", "nf_g_sep", "nf_g_width"])


def _dfa(x, lo=4, hi=16):
    x = np.asarray(x, float)
    if len(x) < hi * 2:
        return np.nan
    y = np.cumsum(x - x.mean())
    F = []
    ns = range(lo, min(hi, len(x) // 3))
    for n in ns:
        k = len(y) // n
        if k < 2:
            continue
        seg = y[:k * n].reshape(k, n)
        t = np.arange(n)
        res = [s - np.polyval(np.polyfit(t, s, 1), t) for s in seg]
        F.append(np.sqrt(np.mean(np.square(res))))
    if len(F) < 3:
        return np.nan
    nn = np.array(list(ns)[:len(F)], float)
    return float(np.polyfit(np.log(nn), np.log(np.array(F) + 1e-12), 1)[0])


def nf_features(ppg, ecg):
    d = {k: np.nan for k in NF_FEATS}
    if not (np.isfinite(ppg).all() and np.isfinite(ecg).all()) or np.std(ppg) < 1e-9:
        return d
    pc = morph_ppg(ppg.astype(float))
    slow = sosfiltfilt(_SLOW, ppg.astype(float) - np.mean(ppg))
    feet = find_feet(pc); tmpl = window_template(pc, feet)
    beats = []
    for fi, fj in zip(feet[:-1], feet[1:]):
        fid = beat_fiducials(pc, fi, fj, template=tmpl)
        if fid is not None:
            beats.append((fi, fj, fid))
    if len(beats) < 8:
        return d
    amps = []; ibis = []; norm = []; feet_i = []
    for fi, fj, fid in beats:
        seg = pc[fi:fj]; a = seg.max() - seg.min()
        if a <= 1e-9 or len(seg) < 20:
            continue
        amps.append(float(seg[fid["sys"] - fi] - seg[0])); feet_i.append(fi)
        g = np.interp(np.linspace(0, len(seg) - 1, 100), np.arange(len(seg)), seg)
        norm.append((g - g.min()) / a)
    if len(norm) < 8:
        return d
    amps = np.asarray(amps, float); feet_i = np.asarray(feet_i, float)
    ibis = np.diff(feet_i) / FS
    Tm = np.median(np.asarray(norm), 0)
    k = np.arange(100)
    for q in range(6, 13):
        d["nf_dct%d" % q] = float(np.sum(Tm * np.cos(np.pi * q * (2 * k + 1) / 200.0)) * 2 / 100.0)
    d2 = np.gradient(np.gradient(Tm))
    d["nf_curv"] = float(np.sum(d2 ** 2))
    pk = int(np.argmax(Tm))
    d["nf_asym"] = float(pk / 100.0)
    d["nf_area1"] = float(np.trapezoid(Tm[:33]) / (np.trapezoid(Tm) + 1e-9))
    d["nf_area2"] = float(np.trapezoid(Tm[33:66]) / (np.trapezoid(Tm) + 1e-9))
    d["nf_area3"] = float(np.trapezoid(Tm[66:]) / (np.trapezoid(Tm) + 1e-9))
    a_i = int(np.argmax(d2[:max(pk, 5)])) if pk > 5 else 0
    av = d2[a_i] if abs(d2[a_i]) > 1e-12 else np.nan
    if np.isfinite(av) and av != 0:
        seg2 = d2[a_i:]
        tr, _ = find_peaks(-seg2); pks, _ = find_peaks(seg2)
        if len(tr):
            d["nf_sd_b"] = float(seg2[tr[0]] / av)
        if len(pks):
            d["nf_sd_c"] = float(seg2[pks[0]] / av)
        if len(tr) > 1:
            d["nf_sd_d"] = float(seg2[tr[1]] / av)
        if len(pks) > 1:
            d["nf_sd_e"] = float(seg2[pks[1]] / av)
    try:
        rp = detect_rpeaks(ecg)
    except Exception:
        rp = np.array([])
    if len(rp) >= 6:
        pats = []; sl = []
        for r in rp:
            nxt = feet_i[feet_i > r]
            if len(nxt):
                dt = (nxt[0] - r) / FS
                if 0.05 < dt < 0.60:
                    pats.append(dt)
            w = pc[r:min(len(pc), r + int(0.5 * FS))]
            if len(w) > 10:
                sl.append(float(np.argmax(np.gradient(w))) / FS)
        if len(pats) >= 5:
            d["nf_pat"] = float(np.median(pats))
            d["nf_pat_cv"] = float(np.std(pats) / (np.mean(pats) + 1e-9))
            if len(ibis):
                d["nf_pat_rr"] = float(np.median(pats) / (np.mean(ibis) + 1e-9))
        if len(sl) >= 5:
            d["nf_slope_delay"] = float(np.median(sl))
    bl = slow[feet_i.astype(int)]
    if len(bl) >= 10 and len(amps) == len(bl):
        fs_b = len(bl) / WIN_SEC
        try:
            fb, Pa = welch(amps - amps.mean(), fs=fs_b, nperseg=min(32, len(amps)))
            _, Pb = welch(bl - bl.mean(), fs=fs_b, nperseg=min(32, len(bl)))
            band = (fb >= 0.1) & (fb <= 0.6)
            if band.any() and Pa[band].sum() > 0:
                j = np.argmax(Pa[band])
                f0 = fb[band][j]
                tb = feet_i / FS
                ca = np.sum((amps - amps.mean()) * np.exp(-2j * np.pi * f0 * tb))
                cb = np.sum((bl - bl.mean()) * np.exp(-2j * np.pi * f0 * tb))
                d["nf_resp_phase"] = float(np.angle(ca) - np.angle(cb))
                d["nf_resp_coh"] = float(abs(ca * np.conj(cb)) / (abs(ca) * abs(cb) + 1e-12))
                d["nf_mod_ratio"] = float((np.percentile(bl, 95) - np.percentile(bl, 5)) /
                                          (np.percentile(amps, 95) - np.percentile(amps, 5) + 1e-9))
        except Exception:
            pass
    if len(amps) >= 12:
        da = np.diff(amps)
        sd1 = float(np.std(da) / np.sqrt(2)); sd2v = 2 * np.var(amps) - sd1 ** 2
        d["nf_sd1"] = sd1 / (np.mean(amps) + 1e-9)
        d["nf_sd2"] = float(np.sqrt(max(sd2v, 0))) / (np.mean(amps) + 1e-9)
        d["nf_sd12"] = float(sd1 / (np.sqrt(max(sd2v, 1e-12)) + 1e-12))
        d["nf_dfa"] = _dfa(amps)
    if len(ibis) >= 12:
        d["nf_ibi_sd1"] = float(np.std(np.diff(ibis)) / np.sqrt(2) / (np.mean(ibis) + 1e-9))
        n = min(len(amps), len(ibis))
        if n >= 10 and np.std(amps[:n]) > 0 and np.std(ibis[:n]) > 0:
            d["nf_amp_ibi_xcorr"] = float(np.corrcoef(amps[:n], ibis[:n])[0, 1])
    try:
        t100 = np.arange(100)
        best = None
        for s1 in (6, 9, 12):
            for s2 in (10, 15, 20):
                for m2 in range(45, 90, 5):
                    G = np.column_stack([np.exp(-0.5 * ((t100 - pk) / s1) ** 2),
                                         np.exp(-0.5 * ((t100 - m2) / s2) ** 2)])
                    co, *_ = np.linalg.lstsq(G, Tm, rcond=None)
                    if (co < 0).any():
                        continue
                    e = float(np.sum((Tm - G @ co) ** 2))
                    if best is None or e < best[0]:
                        best = (e, co[0], co[1], m2 - pk, s2 / s1)
        if best:
            d["nf_g_ratio"] = float(best[2] / (best[1] + 1e-9))
            d["nf_g_sep"] = float(best[3])
            d["nf_g_width"] = float(best[4])
    except Exception:
        pass
    return d


# ---------------------------------------------------------------------------------------------
# 20 ECG features  (cvp-step-examples/src/pwave.py helpers + the per-window body of pwave.one)
# ---------------------------------------------------------------------------------------------
PRE, POST = int(0.35 * FS), int(0.30 * FS)
P_END = int(-0.080 * FS)        # never search inside QRS onset
P_START_MAX = int(-0.300 * FS)  # never search further back than 300 ms
RR_FRAC = 0.42                  # ... nor past 42 % of the RR, where the preceding T wave lives
PW_COLS = ["p_amp", "p_absamp", "p_dur", "p_area", "p_pr", "p_snr", "p_found",
           "p_amp_cv", "p_amp_sd", "p_rise", "p_sym", "p_notch", "p_r_ratio",
           "p_present_frac", "rr_mean", "rr_cv", "rr_rmssd", "af_score",
           "qrs_width", "r_amp", "r_amp_cv", "t_amp", "n_beats_used"]
ECG_FEATS = [c for c in PW_COLS if c not in ("t_amp", "n_beats_used", "p_found")]
_bp = butter(2, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype="band")


def pw_clean(x):
    return filtfilt(_bp[0], _bp[1], x)


def detrend_span(s):
    """Remove the linear trend defined by the two ends of the span."""
    k = max(3, len(s) // 8)
    x0, x1 = np.median(s[:k]), np.median(s[-k:])
    return s - np.linspace(x0, x1, len(s))


def find_p(beat, rr):
    """Locate a genuine P hump in the RR-adaptive, detrended search span; None if it is a ramp."""
    i0 = PRE + int(max(P_START_MAX, -RR_FRAC * rr))
    i1 = PRE + P_END
    if i1 - i0 < int(0.05 * FS) or i0 < 0:
        return None
    s = detrend_span(beat[i0:i1])
    n = len(s)
    k = int(np.argmax(np.abs(s)))
    amp = s[k]
    if k <= 1 or k >= n - 2:
        return None
    thr = 0.30 * abs(amp)
    l = k
    while l > 0 and abs(s[l]) > thr:
        l -= 1
    r = k
    while r < n - 1 and abs(s[r]) > thr:
        r += 1
    if abs(s[l]) > thr or abs(s[r]) > thr:
        return None
    dur = (r - l) / FS * 1000
    if not (30 <= dur <= 200):
        return None
    seg = s[l:r + 1]
    rise = (k - l) / FS * 1000
    return dict(i0=i0, k=i0 + k, on=i0 + l, off=i0 + r, amp=amp, dur=dur,
                area=float(np.sum(seg)) / FS * 1000, pr=(PRE - (i0 + l)) / FS * 1000,
                rise=rise, sym=rise / max(dur, 1e-6),
                notch=int(np.sum((seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])) > 1)
                if amp > 0 else int(np.sum((seg[1:-1] < seg[:-2]) & (seg[1:-1] < seg[2:])) > 1))


def ecg_features(raw):
    """The per-window body of pwave.one."""
    v = dict.fromkeys(PW_COLS, np.nan)
    if not np.isfinite(raw).all() or np.std(raw) < 1e-6: return v
    try: rp = detect_rpeaks(raw)
    except Exception: return v
    if len(rp) < 8: return v
    rr = np.diff(rp) / FS * 1000.0
    rr = rr[(rr > 250) & (rr < 2500)]
    if len(rr) < 5: return v
    rr_mean, rr_sd = float(np.mean(rr)), float(np.std(rr))
    rr_cv = rr_sd / rr_mean
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) / rr_mean
    e = pw_clean(raw)
    idx = [k for k in rp if k - PRE >= 0 and k + POST < WIN]
    if len(idx) < 6: return v
    B = np.array([e[k - PRE:k + POST] for k in idx])
    med = np.median(B, 0)
    keep = np.array([np.corrcoef(b, med)[0, 1] for b in B]) > 0.90
    if keep.sum() < 6: return v
    B = B[keep]; avg = B.mean(0); rmed = float(np.median(rr))
    p = find_p(avg, rmed)
    v.update(rr_mean=rr_mean, rr_cv=rr_cv, rr_rmssd=rmssd, n_beats_used=int(keep.sum()),
             p_found=0.0)
    r_amp = float(avg[PRE]); v["r_amp"] = r_amp
    v["r_amp_cv"] = float(np.std(B[:, PRE]) / (abs(np.mean(B[:, PRE])) + 1e-9))
    q = np.abs(avg - np.median(avg[:int(0.05 * FS)])); thr = 0.15 * np.max(q)
    a_ = b_ = PRE
    while a_ > 0 and q[a_] > thr: a_ -= 1
    while b_ < len(q) - 1 and q[b_] > thr: b_ += 1
    v["qrs_width"] = (b_ - a_) / FS * 1000
    ts, te = PRE + int(0.15 * FS), PRE + int(0.30 * FS)
    if te < len(avg):
        seg = avg[ts:te] - np.median(avg[:int(0.05 * FS)])
        v["t_amp"] = float(seg[np.argmax(np.abs(seg))])
    if p is not None:
        resid = B[:, p["on"]:p["off"] + 1] - avg[p["on"]:p["off"] + 1]
        noise = float(np.median(np.std(resid, axis=1)))
        pa = []
        for b in B:
            sspan = detrend_span(b[p["i0"]:PRE + P_END])
            k = p["k"] - p["i0"]
            if 0 <= k < len(sspan): pa.append(sspan[k])
        pa = np.array(pa)
        v.update(p_amp=p["amp"], p_absamp=abs(p["amp"]), p_dur=p["dur"], p_area=p["area"],
                 p_pr=p["pr"], p_snr=abs(p["amp"]) / (noise + 1e-9), p_found=1.0,
                 p_rise=p["rise"], p_sym=p["sym"], p_notch=float(p["notch"]),
                 p_r_ratio=abs(p["amp"]) / (abs(r_amp) + 1e-9),
                 p_amp_sd=float(np.std(pa)) if len(pa) > 3 else np.nan,
                 p_amp_cv=float(np.std(pa) / (abs(np.mean(pa)) + 1e-9)) if len(pa) > 3 else np.nan,
                 p_present_frac=float(np.mean(np.abs(pa) > 2 * noise)) if len(pa) > 3 else np.nan)
        v["af_score"] = float(rmssd > 0.15) + float(v["p_snr"] < 2.0) + float(rr_cv > 0.12)
    else:
        v["af_score"] = 1.0 + float(rmssd > 0.15) + float(rr_cv > 0.12)
    return v


# ---------------------------------------------------------------------------------------------
# the 91 columns in the training table's order: 39 ppg_* (alphabetical core, then the 9 later
# additions), 20 ECG, 32 nf_*
NAMES = ['ppg_ac_amp',
         'ppg_alt_amp',
         'ppg_alternans',
         'ppg_amp_cv',
         'ppg_area',
         'ppg_awave',
         'ppg_crest',
         'ppg_dct1',
         'ppg_dct2',
         'ppg_dct3',
         'ppg_dct4',
         'ppg_dct5',
         'ppg_dic_height',
         'ppg_ibi_cv',
         'ppg_kurt',
         'ppg_morph_f1',
         'ppg_morph_f4',
         'ppg_morph_mean',
         'ppg_notch_frac',
         'ppg_notch_rel',
         'ppg_pvi',
         'ppg_rate',
         'ppg_resp_pow_frac',
         'ppg_riiv_norm',
         'ppg_skew',
         'ppg_tau',
         'ppg_up_norm',
         'ppg_upstroke',
         'ppg_vci',
         'ppg_w50',
         'ppg_nbeat',
         'ppg_dc',
         'ppg_riiv',
         'ppg_resp_rate',
         'ppg_lf_frac',
         'ppg_ibi_rmssd',
         'ppg_amp_slope',
         'ppg_baseline_amp',
         'ppg_dia_frac',
         'p_amp',
         'p_absamp',
         'p_dur',
         'p_area',
         'p_pr',
         'p_snr',
         'p_amp_cv',
         'p_amp_sd',
         'p_rise',
         'p_sym',
         'p_notch',
         'p_r_ratio',
         'p_present_frac',
         'rr_mean',
         'rr_cv',
         'rr_rmssd',
         'af_score',
         'qrs_width',
         'r_amp',
         'r_amp_cv',
         'nf_dct6',
         'nf_dct7',
         'nf_dct8',
         'nf_dct9',
         'nf_dct10',
         'nf_dct11',
         'nf_dct12',
         'nf_curv',
         'nf_asym',
         'nf_area1',
         'nf_area2',
         'nf_area3',
         'nf_sd_b',
         'nf_sd_c',
         'nf_sd_d',
         'nf_sd_e',
         'nf_pat',
         'nf_pat_cv',
         'nf_pat_rr',
         'nf_slope_delay',
         'nf_resp_phase',
         'nf_resp_coh',
         'nf_mod_ratio',
         'nf_sd1',
         'nf_sd2',
         'nf_sd12',
         'nf_ibi_sd1',
         'nf_amp_ibi_xcorr',
         'nf_dfa',
         'nf_g_ratio',
         'nf_g_sep',
         'nf_g_width']


def window_features(ppg, ecg):
    """All 91 features for one 30-s, 500 Hz window (ppg and ecg are length-15000 float arrays)."""
    d = {}
    try: d.update(ppg_features(ppg))
    except Exception: return None
    try: d.update(nf_features(ppg, ecg))
    except Exception: pass
    try: d.update(ecg_features(ecg))
    except Exception: pass
    return np.array([d.get(k, np.nan) for k in NAMES], np.float32)
