"""Confounder audit for the CVP model + demographics of the filtered 304 cohort.

A) Univariate scan: for EVERY demographic / clinical / physiological variable in the data, how well
   does it alone predict elevated CVP (>12)?  (like sex did for CO)
B) Is the PPG signal a PROXY for a confounder? correlation of each top confounder with the key PPG
   features (upstroke, alternans).
C) Adjustment: does the PPG model survive controlling for the strongest confounders
   (confounder-only vs PPG-only vs PPG+confounder AUC)?
D) Demographics table: elevated (40) vs normal (264) on the key variables.
"""
import numpy as np
import common as K
import enh
from cvpkit import config as C
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, mannwhitneyu

Xb, yb, casesb = enh.per_case(enh.FEATS5); casesb = [int(c) for c in casesb]
ybin = (yb > 12).astype(int)
z = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = z["data"]; cols = [str(c) for c in z["cols"]]
cid = M[:, cols.index("cid")].astype(int)


def percase(name):
    j = cols.index(name); v = M[:, j].astype(float)
    return np.array([np.nanmedian(v[cid == c]) if (cid == c).any() else np.nan for c in casesb])


EXCLUDE = {"cid", "start", "ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi",
           "numeric", "mean_dc", "min_dc", "max_dc", "a_wave_value", "v_wave_value"}
cand = [c for c in cols if c not in EXCLUDE]
vals = {name: percase(name) for name in cand}


def uauc(v):
    m = np.isfinite(v)
    if m.sum() < 30 or len(np.unique(ybin[m])) < 2:
        return np.nan, int(m.sum())
    a = roc_auc_score(ybin[m], v[m]); return max(a, 1 - a), int(m.sum())


# ── A. univariate confounder scan ─────────────────────────────────────────────
def dir_auc(v):
    """signed: AUC using raw v (>0.5 => higher value -> higher CVP)."""
    m = np.isfinite(v)
    return roc_auc_score(ybin[m], v[m]) if m.sum() >= 30 and len(np.unique(ybin[m])) == 2 else np.nan


scan = sorted([(n, *uauc(vals[n])) for n in cand], key=lambda r: -(r[1] if np.isfinite(r[1]) else 0))
print("=== A. Univariate AUC for elevated CVP (>12), 304 cohort — top 22 ===")
for n, a, cnt in scan[:22]:
    if np.isfinite(a):
        d = dir_auc(vals[n]); arrow = "higher->elevated" if d > 0.5 else "lower->elevated"
        print(f"  {n:28s} AUC={a:.3f}  (N={cnt})  [{arrow}]")
for f, j in [("ppg_upstroke", 2), ("ppg_alternans", 0), ("ppg_ac_amp", 1), ("ppg_pvi", 3)]:
    v = Xb[:, j]; m = np.isfinite(v); a = roc_auc_score(ybin[m], v[m])
    print(f"  [model feature] {f:16s} AUC={max(a,1-a):.3f} (N={m.sum()})")

# ── B. is PPG proxying a confounder? ──────────────────────────────────────────
up, alt = Xb[:, 2], Xb[:, 0]
print("\n=== B. Corr of top-14 confounders with the PPG carriers ===")
for n, a, cnt in [r for r in scan if np.isfinite(r[1])][:14]:
    v = vals[n]; m = np.isfinite(v)
    print(f"  {n:28s} AUC={a:.3f} | r(upstroke)={spearmanr(v[m],up[m]).correlation:+.2f} "
          f"r(alternans)={spearmanr(v[m],alt[m]).correlation:+.2f}")

# ── C. adjustment: does PPG survive controlling for the strongest confounders? ─
def model_auc(feat_idx_or_arrays, reps=40):
    X = np.column_stack(feat_idx_or_arrays).astype(float)
    oof = np.zeros(len(yb))
    for r in range(reps):
        rng = np.random.default_rng(r * 11); fo = enh._folds(np.array(casesb), 5, rng); pr = np.zeros(len(yb))
        for f in range(5):
            tr, te = fo != f, fo == f
            pr[te] = np.mean([m.fit(X[tr], yb[tr]).predict(X[te]) for m in enh._ensemble()], 0)
        oof += pr
    return roc_auc_score(ybin, oof / reps)


def imp(v):
    v = v.copy(); v[~np.isfinite(v)] = np.nanmedian(v); return v


ppg_cols = [Xb[:, 0], Xb[:, 1], Xb[:, 2], Xb[:, 3]]   # 4 PPG feats (drop height)
a_ppg = model_auc(ppg_cols)
print(f"\n=== C. Adjustment (impute NaN->median).  PPG-only AUC = {a_ppg:.3f} ===")
for cf in [n for n, a, cnt in scan[:6] if np.isfinite(a)]:
    v = imp(vals[cf])
    print(f"  {cf:26s}: conf-only={model_auc([v]):.3f}  PPG+conf={model_auc(ppg_cols+[v]):.3f}")
# PPG vs the WHOLE top-confounder set combined
top = [imp(vals[n]) for n, a, cnt in scan[:8] if np.isfinite(a)]
print(f"  --- top-8 confounders COMBINED: conf-only={model_auc(top):.3f}  "
      f"PPG+all={model_auc(ppg_cols+top):.3f}  (PPG-only={a_ppg:.3f})")

# ── D. demographics: elevated vs normal ───────────────────────────────────────
print(f"\n=== D. Demographics: elevated (n={ybin.sum()}) vs normal (n={(1-ybin).sum()}) ===")
def summarize(name, pct=False):
    v = vals.get(name)
    if v is None:
        return
    e, nm = v[ybin == 1], v[ybin == 0]; e, nm = e[np.isfinite(e)], nm[np.isfinite(nm)]
    if len(e) < 3 or len(nm) < 3:
        return
    try:
        p = mannwhitneyu(e, nm).pvalue
    except Exception:
        p = np.nan
    if pct:
        print(f"  {name:26s} elevated {100*np.mean(e):4.0f}%   normal {100*np.mean(nm):4.0f}%   p={p:.3f}")
    else:
        print(f"  {name:26s} elevated {np.median(e):6.1f}   normal {np.median(nm):6.1f}   p={p:.3f}")

for nm in ["age", "height", "weight", "bmi", "asa", "hr_n", "art_mbp_n", "spo2_n", "etco2_n",
           "peep", "pip", "vent_tv", "mv", "mac", "compliance", "preop_cr", "preop_alb",
           "preop_hb", "preop_na", "svv", "co", "ci"]:
    summarize(nm)
print("  -- binary / % --")
for nm in ["sex=M", "preop_htn", "preop_dm", "emop", "nepi_rate", "phen_rate", "dopa_rate",
           "ane_type=General", "position=Supine", "position=Lithotomy", "position=Trendelenburg",
           "approach=Open", "approach=Videoscopic", "optype=Transplantation", "optype=Major resection",
           "department=Thoracic surgery", "department=General surgery"]:
    v = vals.get(nm)
    if v is not None and np.isfinite(v).sum() > 30:
        vb = (v > 0).astype(float); vb[~np.isfinite(v)] = np.nan
        summarize.__wrapped__ if False else None
        e, nm2 = vb[ybin == 1], vb[ybin == 0]; e, nm2 = e[np.isfinite(e)], nm2[np.isfinite(nm2)]
        try:
            p = mannwhitneyu(e, nm2).pvalue
        except Exception:
            p = np.nan
        print(f"  {nm:30s} elevated {100*np.mean(e):4.0f}%   normal {100*np.mean(nm2):4.0f}%   p={p:.3f}")
