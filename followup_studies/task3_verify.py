"""Triple-check Task 3: is height-only really predicting CO/SV, or is it a bug?
(1) sanity of the target + height values; (2) raw Spearman(height, target); (3) UNIVARIATE AUC of
raw height (no model) vs the model-based height-only AUC — if they disagree, the model leaks;
(4) compare height vs weight vs BSA (BSA is the physiological driver of CO/SV); (5) sex confound."""
import numpy as np
import enh
from cvpkit import config as C, aggregate, demographics as DEMO
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# per-case targets from medium.npz
z = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = z["data"]; cols = [str(c) for c in z["cols"]]
cid = M[:, cols.index("cid")].astype(int)
pc = lambda name: {int(c): float(np.nanmedian(M[cid == c, cols.index(name)].astype(float))) for c in np.unique(cid)}
CO, SV, CI = pc("co"), pc("sv"), pc("ci")

Xb, ycvp, casesb = enh.per_case(enh.FEATS5)      # col4 = height
casesb = [int(c) for c in casesb]
height = {c: Xb[i, 4] for i, c in enumerate(casesb)}
# weight/BSA/sex from the demographic table
demo = {c: DEMO.demo_for(c) for c in casesb}
weight = {c: demo[c].get("weight", np.nan) for c in casesb}
bmi = {c: demo[c].get("bmi", np.nan) for c in casesb}
sexM = {c: demo[c].get("sex=M", np.nan) for c in casesb}
def bsa(c):  # Mosteller
    h = height[c]; w = weight[c]
    return np.sqrt(h * w / 3600.0) if np.isfinite(h) and np.isfinite(w) else np.nan
BSA = {c: bsa(c) for c in casesb}


def reg_oof(X, y, cases, reps=40):
    X = np.asarray(X, float); y = np.asarray(y, float); cases = np.asarray(cases)
    oof = np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(r * 11); fo = enh._folds(cases, 5, rng); pr = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            pr[te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in enh._ensemble()], 0)
        oof += pr
    return oof / reps


for tname, D, cut, low in [("CO", CO, 4.0, True), ("SV", SV, 60.0, True), ("CI", CI, 2.5, True),
                           ("CVP", {c: ycvp[i] for i, c in enumerate(casesb)}, 12.0, False)]:
    cs = [c for c in casesb if np.isfinite(D.get(c, np.nan)) and np.isfinite(height[c])]
    y = np.array([D[c] for c in cs]); h = np.array([height[c] for c in cs])
    w = np.array([weight[c] for c in cs]); bs = np.array([BSA[c] for c in cs]); sx = np.array([sexM[c] for c in cs])
    yb = (y < cut).astype(int) if low else (y > cut).astype(int)
    print(f"\n===== {tname}  (N={len(cs)}, {'<' if low else '>'}{cut}: {yb.sum()} pos) =====")
    print(f"  target range [{np.min(y):.1f},{np.max(y):.1f}] median {np.median(y):.1f} | "
          f"height range [{np.min(h):.0f},{np.max(h):.0f}]")
    print(f"  Spearman: height {spearmanr(h,y).correlation:+.3f} | weight {spearmanr(w,y).correlation:+.3f} | "
          f"BSA {spearmanr(bs,y).correlation:+.3f} | sex=M {spearmanr(sx,y).correlation:+.3f}")
    sgn = -1 if low else 1
    def uauc(v):
        m = np.isfinite(v)
        return roc_auc_score(yb[m], sgn * v[m]) if 0 < yb[m].sum() < m.sum() else np.nan
    print(f"  UNIVARIATE raw AUC (no model): height={uauc(h):.3f} | weight={uauc(w):.3f} | "
          f"BSA={uauc(bs):.3f} | sex=M={uauc(sx):.3f}")
    # model-based height-only (must match univariate height if no leak)
    oofh = reg_oof(h.reshape(-1, 1), y, np.array(cs))
    print(f"  MODEL height-only AUC={roc_auc_score(yb, sgn*oofh):.3f}  (should ~= univariate height={uauc(h):.3f})")
    print(f"  prevalence by sex: pos-rate male={yb[sx==1].mean():.2f} (n={int((sx==1).sum())}) "
          f"female={yb[sx==0].mean():.2f} (n={int((sx==0).sum())})")
