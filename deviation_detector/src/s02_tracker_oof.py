"""Step 2. The auxiliary TRACKER: an out-of-fold estimate of "CVP minus calibration median", used
as five extra input columns by the detector (its prediction, |prediction|, their causal running
mean, |running mean|, and the running max of |prediction|).

It is the earlier absolute-tracking system, kept only because the detector was developed with
its output available. It is fitted on the 652-patient training split of the original project
(the other 163 patients and every MIMIC record get zeros), on a 10-min calibration / 30-min
tracking schedule, five-fold by patient, so no patient's prediction comes from a model that saw
that patient. Whether these columns matter is answered by the ablation in step 11.

Base model: gradient boosting + ridge on the calibration-differenced features plus 5/15/30-min
lagged differences and the age of the calibration. Two increment models (10 and 60 min) feed a
causal exponential filter (tau 2 min) with a 60-min carry term.
Writes results/vitaldb_tracker_oof.npz  (cid, pred: one array per patient, NaN where not scored)
"""
import os, sys, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RES

W, S = 10.0, 30.0
TAU, LAM, H_S, H_H = 2.0, 0.35, 10.0, 60.0
MIN_N, LAGS, KREF = 8, (5.0, 15.0, 30.0), 5

D = np.load(f"{DATA}/vitaldb_windows.npz", allow_pickle=True)
X, Y, CID, T = D["X"].astype(float), D["y"].astype(float), D["cid"], D["t"]
ORDER = {c: np.where(CID == c)[0][np.argsort(T[CID == c])] for c in np.unique(CID)}


def split(seed=11, frac=0.20):
    """The original project's frozen train / test split of the 815 patients."""
    ids = np.unique(CID); r = np.random.default_rng(seed); r.shuffle(ids)
    n = int(round(frac * len(ids)))
    return set(ids[n:].tolist()), set(ids[:n].tolist())


def lag_index(t, h, tol=0.5):
    lo = np.searchsorted(t, t - h * (1 + tol), "left")
    hi = np.searchsorted(t, t - h * (1 - tol), "right")
    j = np.where(hi > lo, np.clip((lo + hi - 1) // 2, 0, len(t) - 1), -1)
    return np.where(j < np.arange(len(t)), j, -1)


TR, _ = split()
PAT = {}
for c in sorted(TR):
    o = ORDER[c]
    if len(o) < 20: continue
    t = T[o]; y = Y[o]; x = X[o]
    cyc = W + S
    k = np.floor((t - t[0]) / cyc).astype(int); inblk = ((t - t[0]) - k * cyc) < W
    xb = np.full(x.shape, np.nan); yb = np.full(len(t), np.nan); ok = np.zeros(len(t), bool)
    for kk in np.unique(k):
        b = (k == kk) & inblk; m = (k == kk) & ~inblk
        if b.sum() < 3 or m.sum() < MIN_N: continue
        if not (np.isfinite(y[b]).all() and np.isfinite(y[m]).all()): continue
        xb[m] = np.nanmedian(x[b], 0); yb[m] = np.median(y[b]); ok |= m
    PAT[c] = dict(t=t, y=y, x=x, k=k, ok=ok, xb=xb, yb=yb,
                  lags={h: lag_index(t, h) for h in LAGS},
                  p10=lag_index(t, H_S), p60=lag_index(t, H_H))
PAT = {c: v for c, v in PAT.items() if v["ok"].sum() >= MIN_N}
print(f"{len(PAT)} training-split patients with >= {MIN_N} scored windows")


def feats(v, idx, xb, age):
    parts = [np.nan_to_num(v["x"][idx] - xb, nan=0.0)]
    for h in LAGS:
        j = v["lags"][h][idx]
        d = np.where((j >= 0)[:, None], v["x"][idx] - v["x"][np.clip(j, 0, None)], 0.0)
        parts.append(np.nan_to_num(d, nan=0.0))
    parts.append(np.asarray(age, float).reshape(-1, 1))
    return np.hstack(parts)


def base_rows(pats, rng):
    """Training rows for the base model: each scored window against KREF random past references."""
    A, b = [], []
    for c in pats:
        v = PAT[c]; t = v["t"]; y = v["y"]; x = v["x"]
        m = np.where(v["ok"])[0]
        m = m[m > 0]
        if not len(m): continue
        for _ in range(KREF):
            age = rng.uniform(0, S, len(m))
            lo = np.searchsorted(t, t[m] - age - W, "left")
            hi = np.searchsorted(t, t[m] - age, "right")
            g0 = (hi - lo) >= 3
            if not g0.any(): continue
            mm, lo, hi = m[g0], lo[g0], hi[g0]
            xr = np.array([np.nanmedian(x[a:z], 0) for a, z in zip(lo, hi)])
            yr = np.array([np.median(y[a:z]) for a, z in zip(lo, hi)])
            g = np.isfinite(yr) & np.isfinite(y[mm])
            if not g.any(): continue
            A.append(feats(v, mm[g], xr[g], age[g0][g])); b.append(y[mm[g]] - yr[g])
    return np.vstack(A), np.concatenate(b)


def mk():
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=.05, max_leaf_nodes=63,
                                         l2_regularization=1.0, loss="absolute_error",
                                         random_state=0)


def fit_all(pats, rng):
    A, b = base_rows(pats, rng)
    g = np.isfinite(b)
    base = [mk().fit(A[g], b[g]), Ridge(alpha=5.0).fit(np.nan_to_num(A[g]), b[g])]
    inc = {}
    for h, key in ((H_S, "p10"), (H_H, "p60")):
        AA, bb = [], []
        for c in pats:
            v = PAT[c]; t = v["t"]; j = v[key]
            s = np.where((j >= 0) & np.isfinite(v["y"])
                         & np.isfinite(v["y"][np.clip(j, 0, None)]))[0]
            if not len(s): continue
            AA.append(np.hstack([np.nan_to_num(v["x"][s] - v["x"][j[s]], nan=0.0),
                                 (t[s] - t[j[s]])[:, None]]))
            bb.append(v["y"][s] - v["y"][j[s]])
        AA, bb = np.vstack(AA), np.concatenate(bb)
        gg = np.isfinite(bb)
        inc[h] = [mk().fit(AA[gg], bb[gg]), Ridge(alpha=5.0).fit(np.nan_to_num(AA[gg]), bb[gg])]
    return dict(base=base, inc=inc)


def predict(mdl, c):
    v = PAT[c]; t = v["t"]; ok = v["ok"]
    p = np.full(len(t), np.nan); idx = np.where(ok)[0]
    if len(idx):
        age = ((t[idx] - t[0]) % (W + S) - W).clip(0, None)
        p[idx] = np.mean([m.predict(feats(v, idx, v["xb"][idx], age)) for m in mdl["base"]], 0)
    Dd = {}
    for h, key in ((H_S, "p10"), (H_H, "p60")):
        j = v[key]; d = np.full(len(t), np.nan); s = np.where(j >= 0)[0]
        if len(s):
            A = np.hstack([np.nan_to_num(v["x"][s] - v["x"][j[s]], nan=0.0),
                           (t[s] - t[j[s]])[:, None]])
            d[s] = np.mean([m.predict(np.nan_to_num(A)) for m in mdl["inc"][h]], 0)
        Dd[h] = (d, j)
    ds = Dd[H_S][0]; dh, jh = Dd[H_H]
    xst = np.full(len(t), np.nan); prev = None
    for i in range(len(t)):
        if not ok[i]: prev = None; continue
        if prev is None or v["yb"][i] != v["yb"][prev]:
            xst[i] = 0.0; prev = i; continue
        w = np.exp(-(t[i] - t[prev]) / TAU)
        s_ = ds[i] if np.isfinite(ds[i]) else 0.0
        meas = p[i] if np.isfinite(p[i]) else xst[prev]
        xk = w * (xst[prev] + s_) + (1 - w) * meas
        j = jh[i]
        if j >= 0 and np.isfinite(dh[i]) and np.isfinite(xst[j]):
            carry = xst[j] + (v["yb"][j] - v["yb"][i] if np.isfinite(v["yb"][j]) else 0.0)
            xk = (1 - LAM) * xk + LAM * (carry + dh[i])
        xst[i] = xk; prev = i
    return xst


if __name__ == "__main__":
    ids = np.array(sorted(PAT)); np.random.default_rng(31).shuffle(ids)
    FOLDS = np.array_split(ids, 5)
    t0 = time.time(); PRED = {}
    for f, hold in enumerate(FOLDS):
        hs = set(hold.tolist())
        mdl = fit_all([c for c in PAT if c not in hs], np.random.default_rng(7))
        for c in hold.tolist():
            PRED[c] = predict(mdl, c)
        print(f"  fold {f+1}/5 [{time.time()-t0:.0f}s]", flush=True)
    np.savez(f"{RES}/vitaldb_tracker_oof.npz", cid=np.array(sorted(PRED)),
             pred=np.array([PRED[c] for c in sorted(PRED)], dtype=object))
    print(f"saved results/vitaldb_tracker_oof.npz for {len(PRED)} patients [{time.time()-t0:.0f}s]")
