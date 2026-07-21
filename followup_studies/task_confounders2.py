"""Confounder audit part 2: (B-fixed) does the PPG carry correlate with the confounders?
(E) does the PPG signal SURVIVE within homogeneous subgroups (not a surgery-type/sickness proxy)?
(F) does PPG predict the confounder-RESIDUALIZED CVP?"""
import numpy as np
import common as K
import enh
from cvpkit import config as C
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

Xb, yb, casesb = enh.per_case(enh.FEATS5); casesb = [int(c) for c in casesb]; ybin = (yb > 12).astype(int)
z = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = z["data"]; cols = [str(c) for c in z["cols"]]
cid = M[:, cols.index("cid")].astype(int)
pcv = lambda name: np.array([np.nanmedian(M[cid == c, cols.index(name)].astype(float)) for c in casesb])


def oof_pred(X, y, cases, reps=40):
    X = np.asarray(X, float); oof = np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(r * 11); fo = enh._folds(np.asarray(cases), 5, rng); pr = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            pr[te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in enh._ensemble()], 0)
        oof += pr
    return oof / reps


ppg = Xb[:, :4]

# ── B-fixed: correlation of PPG carriers with the top confounders ─────────────
up, alt = Xb[:, 2], Xb[:, 0]
print("=== B. corr of PPG carriers with confounders (pairwise-complete) ===")
for cf in ["tv", "vent_tv", "mac", "preop_cr", "rftn_rate", "asa", "preop_bun", "preop_hb", "pat_art_ppg"]:
    v = pcv(cf)
    for feat, fv, fn in [(up, up, "upstroke"), (alt, alt, "alternans")]:
        pass
    mu = np.isfinite(v) & np.isfinite(up); ma = np.isfinite(v) & np.isfinite(alt)
    print(f"  {cf:14s} r(upstroke)={spearmanr(v[mu],up[mu]).correlation:+.2f}  "
          f"r(alternans)={spearmanr(v[ma],alt[ma]).correlation:+.2f}")

# ── E: PPG-only AUC within homogeneous subgroups ─────────────────────────────
print("\n=== E. PPG-only AUC within subgroups (is it a surgery-type/sickness proxy?) ===")
transplant = pcv("optype=Transplantation") > 0
gendept = pcv("department=General surgery") > 0
emerg = pcv("emop") > 0
subs = [("full cohort", np.ones(len(casesb), bool)),
        ("exclude transplant", ~transplant),
        ("general-surgery dept only", gendept),
        ("exclude emergency", ~emerg)]
for name, mask in subs:
    idx = np.where(mask)[0]
    if ybin[idx].sum() < 8 or (1 - ybin[idx]).sum() < 8:
        print(f"  {name:28s} too few — skip"); continue
    oof = oof_pred(ppg[idx], yb[idx], [casesb[i] for i in idx])
    auc = roc_auc_score(ybin[idx], oof)
    print(f"  {name:28s} N={len(idx):3d} pos={int(ybin[idx].sum()):2d}  PPG-only AUC={auc:.3f}")

# ── F: does PPG predict CVP after removing the confounder-explained part? ─────
def imp(v):
    v = v.copy(); v[~np.isfinite(v)] = np.nanmedian(v); return v
top = np.column_stack([imp(pcv(n)) for n in ["tv", "vent_tv", "mac", "pat_art_ppg", "preop_cr",
                                             "rftn_rate", "asa", "preop_bun"]])
cvp_hat = oof_pred(top, yb, casesb)          # CVP predicted from confounders (OOF)
resid = yb - cvp_hat                          # part of CVP NOT explained by confounders
ppg_hat = oof_pred(ppg, yb, casesb)          # CVP predicted from PPG (OOF)
print("\n=== F. PPG vs the confounder-residualized CVP ===")
print(f"  Spearman(PPG-prediction, raw CVP)               = {spearmanr(ppg_hat, yb).correlation:+.3f}")
print(f"  Spearman(PPG-prediction, confounder-resid CVP)  = {spearmanr(ppg_hat, resid).correlation:+.3f}")
print("  (a clearly-nonzero residual correlation => PPG carries CVP signal beyond the confounders)")
