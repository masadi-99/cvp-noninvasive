"""Reproduce the RICH arterial-waveform result (study C, waveform-morphology extension).

We extracted 15 arterial-pressure WAVEFORM features analogous to the PPG feature set — upstroke
time, systolic/notch fractions, augmentation index, dicrotic-notch height, systolic/diastolic area
ratio, Windkessel decay tau, pulse width, max dP/dt (up and down), reflected-wave transit, arterial
pulsus alternans, amplitude CV, PPV, SPV (see research/artwave.py). This checks whether that richer
set encodes CVP any better than the simple systolic/diastolic/mean numerics. It does not.

    python enhancements/reproduce_arterial_waveform.py     # needs numpy + scikit-learn
"""
import os, csv, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from cvp.model import _ensemble, _folds

HERE = os.path.dirname(os.path.abspath(__file__))
FE5 = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height"]

base = {int(r["caseid"]): r for r in csv.DictReader(open(os.path.join(HERE, "data", "features_ext.csv")))}
awr = {int(r["caseid"]): r for r in csv.DictReader(open(os.path.join(HERE, "data", "artwave_features.csv")))}
AW = [c for c in next(iter(awr.values())) if c.startswith("artw_")]
cases = np.array(sorted(base))
y = np.array([float(base[c]["cvp_numeric"]) for c in cases])
def col(src, f):
    return np.array([float(src[c][f]) if src[c].get(f, "") not in ("", "nan") else np.nan for c in cases])
D = {f: col(base, f) for f in FE5}; D.update({f: col(awr, f) for f in AW})
mat = lambda fs: np.column_stack([D[f] for f in fs])


def evaluate(X, thr=12.0, reps=40, k_select=None):
    yb = (y > thr).astype(int); A = []
    for r in range(reps):
        fo = _folds(cases, 5, np.random.default_rng(r * 11)); oof = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f; Xtr, Xte = X[tr], X[te]
            if k_select and X.shape[1] > k_select:
                imp = SimpleImputer().fit(Xtr); sel = SelectKBest(f_regression, k=k_select).fit(imp.transform(Xtr), yb[tr])
                Xtr, Xte = sel.transform(imp.transform(Xtr)), sel.transform(imp.transform(Xte))
            oof[te] = np.mean([m.fit(Xtr, y[tr]).predict(Xte) for m in _ensemble()], 0)
        A.append(roc_auc_score(yb, oof))
    return float(np.mean(A))


if __name__ == "__main__":
    print("single-feature AUC of the 15 rich arterial-waveform features (all near chance):")
    for f in AW:
        print(f"  {f:18} {evaluate(mat([f])):.3f}")
    print(f"\nbaseline 5-feature:                    {evaluate(mat(FE5)):.3f}")
    print(f"5-feature + 15 arterial-waveform:      {evaluate(mat(FE5 + AW), k_select=8):.3f}   (SelectKBest 8)")
    print(f"arterial-waveform 15 alone:            {evaluate(mat(AW), k_select=8):.3f}")
    print("\n-> even a rich, PPG-analogous arterial-waveform feature set is near-chance and HURTS the")
    print("   model (0.75 -> ~0.68): the arterial waveform reflects afterload / compliance, not CVP.")
