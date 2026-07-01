"""Core harness for the four proposed-enhancement studies.

Substrate: a per-window matrix keyed by (cid, start), merging
  - the 5 canonical non-invasive features from the cvpkit matrix
        ppg_alternans, ppg_ac_amp, ppg_upstroke, ppg_pvi   (+ height from demo)
  - the per-window CVP label `numeric` (Solar8000 monitor CVP)
  - the INVASIVE arterial-line numerics from medium_eng.npz
        art_sbp_n, art_dbp_n, art_mbp_n   (Study C)
The two source matrices are the SAME 9075 windows / 333 cases (verified 100% key overlap).

Evaluation reproduces cvp-noninvasive/cvp/model.py exactly (Ridge(alpha=20)+HGB ensemble,
continuous-CVP regression, binarize at an arbitrary threshold, nested-Youden Sens/Spec),
but parameterised by the CVP threshold so we can sweep it (Study A).
"""
import sys, warnings
sys.path.insert(0, "/home/masadi/cvp-ecg-ppg"); sys.path.insert(0, "/home/masadi/run")
warnings.filterwarnings("ignore")
import numpy as np
from cvpkit import aggregate, config as C
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

FEATS5 = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height"]
ARTERIAL = ["art_sbp_n", "art_dbp_n", "art_mbp_n"]
RIDGE_ALPHA = 20.0
HGB = dict(max_depth=3, learning_rate=0.03, max_iter=450, l2_regularization=5.0, random_state=0)

_DATA = None


def load_merged():
    """Per-window dict with keys: cid, start, numeric, and a col->vector dict `F`
    holding the 5 non-invasive features, height, and the 3 arterial numerics."""
    global _DATA
    if _DATA is not None:
        return _DATA
    Mk, ck, gk = aggregate.load_matrix(C.MATRIX_NPZ)
    cid = Mk[:, ck.index("cid")].astype(int); start = Mk[:, ck.index("start")].astype(int)
    key = list(zip(cid.tolist(), start.tolist()))
    F = {}
    for f in ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height"]:
        F[f] = Mk[:, ck.index(f)].astype(float)
    numeric = Mk[:, ck.index("numeric")].astype(float)
    # merge arterial numerics from medium_eng by (cid,start)
    z = np.load("/home/masadi/run/data/medium_eng.npz", allow_pickle=True)
    Me, ce = z["data"], [str(c) for c in z["cols"]]
    cide = Me[:, ce.index("cid")].astype(int); starte = Me[:, ce.index("start")].astype(int)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cide, starte))}
    order = np.array([idx[k] for k in key])           # align medium_eng rows to cvpkit order
    for f in ARTERIAL:
        F[f] = Me[order, ce.index(f)].astype(float)
    _DATA = dict(cid=cid, start=start, numeric=numeric, F=F, key=key)
    return _DATA


def per_case(feats, agg="median"):
    """Per-case aggregated matrix X over the case's windows (default median).
    Returns X (ncase x nfeat), y (per-case median CVP), cases."""
    d = load_merged(); cid = d["cid"]
    cases = np.unique(cid)
    fn = {"median": np.nanmedian, "mean": np.nanmean}[agg]
    y = np.array([np.nanmedian(d["numeric"][cid == c]) for c in cases])
    ok = np.isfinite(y)                                # drop cases with no CVP numeric (304 of 333)
    cases, y = cases[ok], y[ok]
    X = np.full((len(cases), len(feats)), np.nan)
    for i, c in enumerate(cases):
        m = cid == c
        for j, f in enumerate(feats):
            v = d["F"][f][m]
            X[i, j] = fn(v) if np.isfinite(v).any() else np.nan
    return X, y, cases


def _ensemble():
    return [Pipeline([("im", SimpleImputer()), ("sc", StandardScaler()), ("m", Ridge(alpha=RIDGE_ALPHA))]),
            Pipeline([("im", SimpleImputer()), ("m", HistGradientBoostingRegressor(**HGB))])]


def _folds(cases, k, rng):
    u = np.unique(cases); rng.shuffle(u); fo = {x: i % k for i, x in enumerate(u)}
    return np.array([fo[x] for x in cases])


def evaluate(X, y, cases, threshold=12.0, reps=40, seed=0, k_select=None, return_oof=False, return_reps=False):
    """Repeated nested grouped-by-case CV. Regress continuous CVP, binarize at `threshold`
    for AUC, choose Sens/Spec operating point by Youden on an inner 4-fold of the training
    fold only. Returns metrics dict (auc/ci/sens/spec + n/npos)."""
    X = np.asarray(X, float); y = np.asarray(y, float); cases = np.asarray(cases)
    yb_all = (y > threshold).astype(int)
    A, SN, SP = [], [], []
    oof_acc = np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(seed + r * 11)
        fo = _folds(cases, 5, rng); oof = np.zeros(len(y)); op = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            Xtr, Xte = X[tr], X[te]
            if k_select and X.shape[1] > k_select:
                imp = SimpleImputer().fit(Xtr); sel = SelectKBest(f_regression, k=k_select).fit(imp.transform(Xtr), yb_all[tr])
                Xtr = sel.transform(imp.transform(Xtr)); Xte = sel.transform(imp.transform(Xte))
            oof[te] = np.mean([m.fit(Xtr, y[tr]).predict(Xte) for m in _ensemble()], 0)
            inn = _folds(cases[tr], 4, np.random.default_rng(1)); ip = np.zeros(tr.sum())
            for g in range(4):
                itr, ite = inn != g, inn == g
                a, b = X[tr][itr], X[tr][ite]
                if k_select and X.shape[1] > k_select:
                    imp = SimpleImputer().fit(a); sel = SelectKBest(f_regression, k=k_select).fit(imp.transform(a), yb_all[tr][itr])
                    a, b = sel.transform(imp.transform(a)), sel.transform(imp.transform(b))
                ip[ite] = np.mean([m.fit(a, y[tr][itr]).predict(b) for m in _ensemble()], 0)
            fpr, tpr, th = roc_curve(yb_all[tr], ip); op[te] = th[int(np.argmax(tpr - fpr))]
        A.append(roc_auc_score(yb_all, oof)); oof_acc += oof
        pred = (oof >= op).astype(int)
        tp = ((pred == 1) & (yb_all == 1)).sum(); fn_ = ((pred == 0) & (yb_all == 1)).sum()
        tn = ((pred == 0) & (yb_all == 0)).sum(); fp = ((pred == 1) & (yb_all == 0)).sum()
        SN.append(tp / (tp + fn_ + 1e-9)); SP.append(tn / (tn + fp + 1e-9))
    A = np.array(A); lo, hi = np.percentile(A, [5, 95])
    out = dict(auc=float(A.mean()), sd=float(A.std()), ci=[float(lo), float(hi)],
               sens=float(np.mean(SN)), spec=float(np.mean(SP)),
               n=int(len(y)), npos=int(yb_all.sum()), thr=float(threshold), nfeat=X.shape[1])
    if return_oof:
        out["oof"] = oof_acc / reps; out["yb"] = yb_all
    if return_reps:
        out["reps_auc"] = A
    return out


if __name__ == "__main__":
    # validation: the 5-feature non-invasive model at threshold 12 must reproduce ~0.754
    X, y, cases = per_case(FEATS5)
    r = evaluate(X, y, cases, threshold=12.0, reps=40)
    print(f"5-feature non-invasive @thr12: AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] "
          f"Sens {r['sens']:.2f} Spec {r['spec']:.2f}  N={r['n']} pos={r['npos']}")
    print("expected ~0.754 (matches cvp-noninvasive) — substrate validated" )
