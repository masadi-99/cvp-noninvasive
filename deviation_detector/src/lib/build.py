"""The pipeline every step shares: schedule -> artefact filter -> baseline -> 123-feature vector
-> magnitude regressor -> causal running-max score -> patient-clustered evaluation.

A patient's record is cut into cycles of W_CAL minutes of CVP calibration followed by S_TRACK
minutes of tracking. For every tracked window the model sees only ECG/PPG features differenced
against the calibration block's medians, the true time since the last CVP reading, the shape of
the calibration block's own CVP, what earlier blocks did, and (VitalDB only) the auxiliary
tracker's output. It predicts the LARGEST |CVP - baseline| seen so far in the stretch; the
reported score is the running maximum of that prediction, so nothing after the current window
is ever used.
"""
import os, sys, glob
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA, RES, RES_BASE, W_CAL, S_TRACK, MIN_N, MAX_STEP, HGB, NSEED, BOOT_SEED,
                    DESC_NAMES, HIST_NAMES, EXTRA_NAMES, MIMIC3_PAPER_N, MIMIC_GATE, DROP_COLS)


# ---- descriptors --------------------------------------------------------------------------------
def worst_step(u, y, lo, hi, win=5.0):
    """Largest change between two rolling 5-min medians five minutes apart, inside [lo, hi].
    Above MAX_STEP the stretch is treated as a transducer / line artefact and excluded."""
    m = (u >= lo) & (u <= hi)
    if m.sum() < 6: return 0.0
    uu, yy = u[m], y[m]
    med = np.array([np.median(yy[(uu >= a - win / 2) & (uu <= a + win / 2)]) for a in uu])
    w = 0.0
    for i, a in enumerate(uu):
        j = np.searchsorted(uu, a + win)
        if j < len(uu): w = max(w, abs(med[j] - med[i]))
    return float(w)


def desc(t, y):
    """13 numbers describing the calibration block's own CVP trace (time is relative to its end)."""
    n = len(y); t0 = t - t[-1]
    sl = np.polyfit(t0, y, 1)[0]
    cu = np.polyfit(t0, y, 2)[0] if n >= 5 else 0.0
    h = n // 2
    s1 = np.polyfit(t0[:h], y[:h], 1)[0] if h >= 2 else sl
    s2 = np.polyfit(t0[h:], y[h:], 1)[0] if n - h >= 2 else sl
    med = float(np.median(y))
    return np.array([sl, cu, s1, s2, s2 - s1, float(y[-1]) - med, float(np.mean(y)) - med,
                     float(np.std(y)), float(np.ptp(y)),
                     float(np.mean(np.abs(np.diff(y)))) if n > 1 else 0.0,
                     med, float(y[-1]), float(n)])


def feature_names(names91):
    """All 123 column names of the feature block (the layout is fixed; DROP_COLS are masked at fit time)."""
    return ([f"d_{n}" for n in names91] + ["since_last_cvp"] + DESC_NAMES + HIST_NAMES + EXTRA_NAMES)


_ALL = ["d_%d" % i for i in range(91)] + ["since_last_cvp"] + DESC_NAMES + HIST_NAMES + EXTRA_NAMES
ACTIVE = [i for i, n in enumerate(_ALL) if n not in DROP_COLS]       # the variant's column set


def active(cols=None):
    """The columns a model may use: the variant's active set, intersected with `cols` if given."""
    return ACTIVE if cols is None else [c for c in cols if c in set(ACTIVE)]


def active_names(names91):
    return [n for i, n in enumerate(feature_names(names91)) if i in set(ACTIVE)]


def _vector(x_idx, xb, since, dd, hf, gap, nd, ng, sdg, rate, qq):
    """The 123-column feature block for one stretch (rows = tracked windows)."""
    n = len(since)
    cm = np.cumsum(qq) / np.arange(1, n + 1)
    return np.column_stack([np.nan_to_num(x_idx - xb, nan=0.0), since,
                            np.tile(dd, (n, 1)), np.tile(hf, (n, 1)),
                            np.full(n, gap), np.full(n, nd), np.full(n, ng), np.full(n, sdg),
                            dd[0] * since, np.abs(dd[0]) * since, dd[7] * since,
                            np.nan_to_num(rate) * since,
                            qq, np.abs(qq), cm, np.abs(cm), np.maximum.accumulate(np.abs(qq))])


# ---- VitalDB ------------------------------------------------------------------------------------
class VitalDB:
    """The gated window table, the dense CVP grid and the tracker output, keyed by patient."""
    def __init__(self, tracker=True):
        D = np.load(f"{DATA}/vitaldb_windows.npz", allow_pickle=True)
        self.X, self.Y, self.CID, self.T = D["X"].astype(float), D["y"].astype(float), D["cid"], D["t"]
        self.names = [str(s) for s in D["names"]]
        self.ORDER = {int(c): np.where(self.CID == c)[0][np.argsort(self.T[self.CID == c])]
                      for c in np.unique(self.CID)}
        Z = np.load(f"{DATA}/vitaldb_dense.npz")
        self.DEN, self.NUM = {}, {}
        for c in np.unique(Z["cid"]):
            m = Z["cid"] == c; o = np.argsort(Z["t"][m]); u = Z["t"][m][o]
            y = Z["cvp_wf_ee"][m][o].astype(float)
            g = np.isfinite(y) & (y > -10) & (y < 50)
            if g.sum() > 20: self.DEN[int(c)] = (u[g], y[g])
            self.NUM[int(c)] = dict(t=u, hr=Z["hr"][m][o], map=Z["map"][m][o], spo2=Z["spo2"][m][o])
        self.QP = {}
        f = f"{RES_BASE}/vitaldb_tracker_oof.npz"
        if tracker and os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            self.QP = {int(c): np.asarray(p, float) for c, p in zip(z["cid"], z["pred"])}


def build_vitaldb(V, W=W_CAL, S=S_TRACK, dense=True, tracker=True, causal_offset=False, filter_span="full"):
    """Stretches for every VitalDB patient. Baseline and block shape come from the DENSE grid,
    aligned to the label's units by ONE scalar per patient: the median of (gated label - dense
    tile) over all of that patient's gated windows, whole case. That scalar shifts baseline and
    label together, so it cannot leak into the score, but it is not strictly causal for the
    label. Falls back to the gated block when the dense grid has < 4 tiles there. dense=False
    uses the gated block only (the generic builder's behaviour), tracker=False zeroes the
    tracker columns. causal_offset=True recomputes the alignment scalar per block from the gated
    windows up to the end of that block only (strictly causal label; falls back to the gated
    block when fewer than 3 such windows exist). filter_span: what the artefact-step filter scans,
    "full" = read + tracking (the reported system), "read" = the read period only, "none" = off."""
    assert filter_span in ("full", "read", "none"), filter_span
    R = {}
    for c in sorted(V.ORDER):
        o = V.ORDER[c]
        if len(o) < 12 or c not in V.DEN: continue
        t, y, x = V.T[o], V.Y[o], V.X[o]
        u, yd = V.DEN[c]
        key = {round(float(a), 4): i for i, a in enumerate(u)}
        ii = np.array([key.get(round(float(a), 4), -1) for a in t])
        gm = (ii >= 0) & np.isfinite(y)
        if gm.sum() < 10: continue
        cv = yd[ii[gm]]; ok2 = np.isfinite(cv)
        if ok2.sum() < 10: continue
        off = float(np.median(y[gm][ok2] - cv[ok2]))
        cyc = W + S; kk = np.floor((t - t[0]) / cyc).astype(int)
        inb = ((t - t[0]) - kk * cyc) < W
        q = np.nan_to_num(V.QP.get(c, np.zeros(len(t)))) if tracker else np.zeros(len(t))
        hist = []
        for k in np.unique(kk):
            bm = (kk == k) & inb; sm = (kk == k) & ~inb & np.isfinite(y)
            if bm.sum() < 3 or sm.sum() < MIN_N: continue
            t0b, t1b = float(t[bm][0]), float(t[bm][-1])
            g2 = (u >= t0b) & (u <= t1b)
            offk, okk = off, True
            if causal_offset:
                gc = gm & (t <= t1b); cvc = yd[ii[np.where(gc)[0]]] if gc.any() else np.array([])
                fc = np.isfinite(cvc)
                if fc.sum() >= 3: offk = float(np.median(y[gc][fc] - cvc[fc]))
                else: okk = False
            if dense and okk and g2.sum() >= 4:
                dd = desc(u[g2], yd[g2] + offk); base = float(np.median(yd[g2] + offk)); nd = g2.sum()
            elif np.isfinite(y[bm]).all() and bm.sum() >= 4:
                dd = desc(t[bm], y[bm]); base = float(np.median(y[bm])); nd = bm.sum()
            else: continue
            idx = np.where(sm)[0]
            if filter_span != "none" and worst_step(u, yd, t0b, t1b if filter_span == "read" else float(t[idx][-1])) > MAX_STEP: continue
            since = t[idx] - t1b; gap = float(t[idx][0] - t1b)
            H = hist[-1] if hist else None
            rate = ((base - H[1]) / max(t1b - H[0], 1e-3)) if H else np.nan
            hf = np.array([base - H[1] if H else np.nan, rate, (t1b - H[0]) if H else np.nan,
                           float(len(hist)), H[2] if H else np.nan])
            F = _vector(x[idx], np.nanmedian(x[bm], 0), since, dd, hf, gap, nd, bm.sum(),
                        float(np.std(y[bm])), rate, q[idx])
            R.setdefault(c, []).append(dict(
                key=c, k=int(k), X=F, dev=np.abs(y[idx] - base), sdev=y[idx] - base,
                t=t[idx], y=y[idx], base=base, tb=t[bm], yb=y[bm], since=since,
                dev_s=np.abs(np.array([np.median(y[idx][max(0, i - 1):i + 2])
                                       for i in range(len(idx))]) - base)))
            hist.append((t1b, base, float(dd[0])))
    return R


# ---- any record with (t, y, X): MIMIC, and VitalDB when it must match MIMIC exactly ------------
def build_record(t, y, X, key, W=W_CAL, S=S_TRACK, extra=None, max_step=MAX_STEP, filter_span="full"):
    """Lay the schedule on one record's own clock. Baseline and block shape from the block's own
    windows; tracker columns zero. `extra` = dict of per-window arrays carried along (e.g. MAP).
    `max_step` is the artefact-filter step in the label's units (3 mmHg for CVP); `filter_span` as in
    build_vitaldb ("full" / "read" / "none")."""
    assert filter_span in ("full", "read", "none"), filter_span
    out = []
    if len(t) < 12: return out
    o = np.argsort(t); t, y, X = t[o], y[o], X[o]
    extra = {k: np.asarray(v)[o] for k, v in (extra or {}).items()}
    cyc = W + S
    kk = np.floor((t - t[0]) / cyc).astype(int)
    inb = ((t - t[0]) - kk * cyc) < W
    hist = []
    for k in np.unique(kk):
        bm = (kk == k) & inb; sm = (kk == k) & ~inb & np.isfinite(y)
        if bm.sum() < 3 or sm.sum() < MIN_N: continue
        tb, yb = t[bm], y[bm]
        if not np.isfinite(yb).all() or bm.sum() < 4: continue
        dd = desc(tb, yb); base = float(np.median(yb))
        idx = np.where(sm)[0]
        if filter_span != "none" and worst_step(t, y, float(tb[0]), float(tb[-1]) if filter_span == "read" else float(t[idx][-1])) > max_step: continue
        since = t[idx] - tb[-1]; gap = float(t[idx][0] - tb[-1])
        H = hist[-1] if hist else None
        rate = ((base - H[1]) / max(tb[-1] - H[0], 1e-3)) if H else np.nan
        hf = np.array([base - H[1] if H else np.nan, rate, (tb[-1] - H[0]) if H else np.nan,
                       float(len(hist)), H[2] if H else np.nan])
        F = _vector(X[idx], np.nanmedian(X[bm], 0), since, dd, hf, gap, bm.sum(), bm.sum(),
                    float(np.std(yb)), rate, np.zeros(len(idx)))
        s = dict(key=key, k=int(k), X=F, dev=np.abs(y[idx] - base), sdev=y[idx] - base,
                 t=t[idx], y=y[idx], base=base, tb=tb, yb=yb, since=since)
        for nm, v in extra.items(): s[nm] = v[idx]
        out.append(s)
        hist.append((float(tb[-1]), base, float(dd[0])))
    return out


def mimic_files(subset="paper"):
    """Feature files of the MIMIC records: 'paper' = the first MIMIC3_PAPER_N usable MIMIC-III
    records (sorted by id) plus every MIMIC-IV record; 'extra' = the remaining MIMIC-III records;
    'all' = everything."""
    import json
    ids = set()
    for p in glob.glob(f"{RES_BASE}/mimic/survey_mimic3wdb_*.json"):
        ids |= {r["rec"] for r in json.load(open(p)) if r.get("longest_run_min", 0) >= 25}
    paper = {f"{RES_BASE}/mimic_feat/mimic3wdb_{r}.npz" for r in sorted(ids)[:MIMIC3_PAPER_N]}
    m3 = sorted(glob.glob(f"{RES_BASE}/mimic_feat/mimic3wdb_*.npz")); m4 = sorted(glob.glob(f"{RES_BASE}/mimic_feat/mimic4wdb_*.npz"))
    if subset == "paper": return [p for p in m3 if p in paper] + m4
    if subset == "extra": return [p for p in m3 if p not in paper]
    return m3 + m4


_Q = None
def cvp_gate_mask(key, n):
    """The CVP-waveform gate of step 19 for one MIMIC record (all True when the gate is off)."""
    global _Q
    if not MIMIC_GATE: return np.ones(n, bool)
    if _Q is None: _Q = np.load(f"{RES_BASE}/mimic_cvp_quality.npz", allow_pickle=True)
    sd, run, pul = _Q[f"{key}__std"], _Q[f"{key}__flat_run_s"], _Q[f"{key}__puls"]
    return (sd >= 0.5) & (sd <= 12.0) & (run <= 2.0) & (pul >= 0.5)


def load_mimic(src=None, W=W_CAL, S=S_TRACK, extra=False, subset="paper", filter_span="full"):
    R = {}
    for p in mimic_files(subset):
        key = os.path.basename(p).replace(".npz", "")
        if src and not key.startswith(src): continue
        z = np.load(p, allow_pickle=True); ok = cvp_gate_mask(key, len(z["t"]))
        ex = {"map": z["map"][ok]} if extra and "map" in z.files else None
        ss = build_record(z["t"].astype(float)[ok], z["y"].astype(float)[ok], z["X"].astype(float)[ok], key,
                          W, S, extra=ex, filter_span=filter_span)
        if ss: R[key] = ss
    return R


def load_vitaldb_generic(V, W=W_CAL, S=S_TRACK, filter_span="full"):
    """VitalDB through the record builder: what the MIMIC-facing model is trained on."""
    R = {}
    for c in sorted(V.ORDER):
        o = V.ORDER[c]
        if len(o) < 12: continue
        ss = build_record(V.T[o], V.Y[o], V.X[o], f"vitaldb_{c}", W, S, filter_span=filter_span)
        if ss: R[f"vitaldb_{c}"] = ss
    return R


# ---- model --------------------------------------------------------------------------------------
def target(s):
    """The training target: the largest |CVP - baseline| seen so far in the stretch."""
    return np.maximum.accumulate(s["dev"])


def fit(R, keys, nseed=NSEED, cols=None, drop_nan_rows=False, target_fn=target, seeds=None,
        **override):
    """nseed magnitude regressors on the stretches of `keys`. `cols` restricts the feature
    columns (ablations). drop_nan_rows=True discards rows with any NaN feature (the transfer
    arms did this; the VitalDB nested CV did not - gradient boosting handles NaN natively)."""
    A = np.vstack([s["X"] for k in keys if k in R for s in R[k]])
    tgt = np.concatenate([target_fn(s) for k in keys if k in R for s in R[k]])
    cols = active(cols); A = A[:, cols]
    if drop_nan_rows:
        g = np.isfinite(A).all(1) & np.isfinite(tgt); A, tgt = A[g], tgt[g]
    hp = dict(HGB); hp.update(override)
    seeds = list(range(nseed)) if seeds is None else seeds
    return [HistGradientBoostingRegressor(random_state=sd, **hp).fit(A, tgt) for sd in seeds]


def score(MS, segs, cols=None, running_max=True):
    """Per-window scores for a list of stretches, one batched predict per model."""
    if not segs: return []
    BIG = np.vstack([s["X"] for s in segs])[:, active(cols)]
    PR = np.mean([m.predict(BIG) for m in MS], 0)
    out, o = [], 0
    for s in segs:
        n = len(s["dev"]); p = PR[o:o + n]; o += n
        out.append(np.maximum.accumulate(p) if running_max else p)
    return out


def boot_ci(P, Y, G, n_boot=400, seed=BOOT_SEED):
    """Patient / record-clustered bootstrap of the pooled AUC."""
    up = np.unique(G)
    GI = [np.where(G == c)[0] for c in up]
    rb = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        pk = rb.integers(0, len(GI), len(GI))
        m2 = np.concatenate([GI[i] for i in pk])
        yy = Y[m2]
        if yy.min() != yy.max(): bs.append(roc_auc_score(yy, P[m2]))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def evaluate(MS, R, keys, thr, cols=None, n_boot=400):
    """Pooled AUC with clustered CI for `keys` at threshold `thr`: (auc, (lo, hi), prevalence, n)."""
    segs = [s for k in keys for s in R[k]]
    if not segs: return np.nan, (np.nan, np.nan), 0.0, 0
    P = np.concatenate(score(MS, segs, cols))
    Y = np.concatenate([(target(s) > thr).astype(int) for s in segs])
    G = np.concatenate([[s["key"]] * len(s["dev"]) for s in segs])
    if len(np.unique(Y)) < 2: return np.nan, (np.nan, np.nan), 0.0, len(Y)
    return roc_auc_score(Y, P), boot_ci(P, Y, G, n_boot), float(Y.mean()), len(Y)
