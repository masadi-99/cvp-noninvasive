"""The model: predict continuous CVP with a Ridge + HistGradientBoosting ensemble, then
threshold at 12 mmHg. Evaluated with repeated grouped-by-patient cross-validation and an honest
nested operating point (the Sens/Spec threshold is chosen on an inner CV of the training fold only,
never on the test fold). AUC is threshold-free."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, roc_curve

THRESHOLD = 12.0          # mmHg; CVP > 12 == "elevated"
RIDGE_ALPHA = 20.0
HGB = dict(max_depth=3, learning_rate=0.03, max_iter=450, l2_regularization=5.0, random_state=0)


def _ensemble():
    return [Pipeline([("im", SimpleImputer()), ("sc", StandardScaler()), ("m", Ridge(alpha=RIDGE_ALPHA))]),
            Pipeline([("im", SimpleImputer()), ("m", HistGradientBoostingRegressor(**HGB))])]


def _folds(cases, k, rng):
    u = np.unique(cases); rng.shuffle(u)
    fo = {c: i % k for i, c in enumerate(u)}
    return np.array([fo[c] for c in cases])


def _predict(Xtr, ytr, Xte):
    return np.mean([m.fit(Xtr, ytr).predict(Xte) for m in _ensemble()], axis=0)


def _one_repeat(X, y, cases, rng, folds=5, inner=4):
    yb = (y > THRESHOLD).astype(int)
    fo = _folds(cases, folds, rng); oof = np.zeros(len(y)); op = np.zeros(len(y))
    for f in range(folds):
        tr, te = fo != f, fo == f
        oof[te] = _predict(X[tr], y[tr], X[te])
        inn = _folds(cases[tr], inner, np.random.default_rng(1)); ip = np.zeros(tr.sum())
        for g in range(inner):
            itr, ite = inn != g, inn == g
            ip[ite] = _predict(X[tr][itr], y[tr][itr], X[tr][ite])
        fpr, tpr, th = roc_curve(yb[tr], ip)
        op[te] = th[int(np.argmax(tpr - fpr))]          # Youden point on the inner CV
    return oof, op, yb


def evaluate(X, y, cases, reps=40, seed=0):
    """Repeated nested grouped CV. Returns AUC (mean + 90% CI) and nested Sens/Spec."""
    X = np.asarray(X, float); y = np.asarray(y, float); cases = np.asarray(cases)
    A, SN, SP = [], [], []
    for r in range(reps):
        oof, op, yb = _one_repeat(X, y, cases, np.random.default_rng(seed + r * 11))
        A.append(roc_auc_score(yb, oof))
        pred = (oof >= op).astype(int)
        tp = ((pred == 1) & (yb == 1)).sum(); fn = ((pred == 0) & (yb == 1)).sum()
        tn = ((pred == 0) & (yb == 0)).sum(); fp = ((pred == 1) & (yb == 0)).sum()
        SN.append(tp / (tp + fn + 1e-9)); SP.append(tn / (tn + fp + 1e-9))
    A = np.array(A); lo, hi = np.percentile(A, [5, 95])
    return dict(auc=float(A.mean()), ci90=[float(lo), float(hi)],
                sens=float(np.mean(SN)), spec=float(np.mean(SP)),
                n=int(len(y)), npos=int((y > THRESHOLD).sum()))
