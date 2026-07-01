"""Reproduce the per-case enhancement results from data/features_ext.csv, using the repo's own model
(cvp.model's Ridge + HistGradientBoosting ensemble and its nested grouped-by-patient CV).

    python enhancements/reproduce.py            # needs only numpy + scikit-learn

Covers three of the four studies (the fourth, windowing, is enhancements/reproduce_windowing.py):
  A  CVP-threshold sweep      B/C  invasive arterial BP      D  categorical PPG morphology (Dawber)
Baseline: the 5 non-invasive features, AUC ~0.754 at CVP > 12.
"""
import os, csv, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, roc_curve
from cvp.model import _ensemble, _folds          # exact repo ensemble + grouped folds

FE5 = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height"]
ART = ["art_sbp_n", "art_dbp_n", "art_mbp_n"]
MORPH = ["morph_f1", "morph_f4"]
HERE = os.path.dirname(os.path.abspath(__file__))

rows = list(csv.DictReader(open(os.path.join(HERE, "data", "features_ext.csv"))))
cases = np.array([int(r["caseid"]) for r in rows])
y = np.array([float(r["cvp_numeric"]) for r in rows])
_col = lambda f: np.array([float(r[f]) if r[f] not in ("", "nan") else np.nan for r in rows])
D = {f: _col(f) for f in FE5 + ART + MORPH}
mat = lambda fs: np.column_stack([D[f] for f in fs])


def evaluate(X, thr=12.0, reps=40, seed=0, return_reps=False):
    """Repeated nested grouped-by-patient CV; regress continuous CVP, binarize at `thr` for AUC,
    nested-Youden operating point for Sens/Spec. Identical scheme to cvp.model.evaluate."""
    yb = (y > thr).astype(int); A, SN, SP = [], [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r * 11); fo = _folds(cases, 5, rng)
        oof = np.zeros(len(y)); op = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            oof[te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in _ensemble()], 0)
            inn = _folds(cases[tr], 4, np.random.default_rng(1)); ip = np.zeros(tr.sum())
            for g in range(4):
                itr, ite = inn != g, inn == g
                ip[ite] = np.mean([m.fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite]) for m in _ensemble()], 0)
            fpr, tpr, th = roc_curve(yb[tr], ip); op[te] = th[int(np.argmax(tpr - fpr))]
        A.append(roc_auc_score(yb, oof))
        pred = oof >= op
        tp = (pred & (yb == 1)).sum(); fn = (~pred & (yb == 1)).sum()
        tn = (~pred & (yb == 0)).sum(); fp = (pred & (yb == 0)).sum()
        SN.append(tp / (tp + fn + 1e-9)); SP.append(tn / (tn + fp + 1e-9))
    A = np.array(A); lo, hi = np.percentile(A, [5, 95])
    out = dict(auc=float(A.mean()), ci=[float(lo), float(hi)], sens=float(np.mean(SN)),
               spec=float(np.mean(SP)), npos=int(yb.sum()), n=len(y))
    if return_reps:
        out["reps"] = A
    return out


def paired(fa, fb, thr=12.0, reps=40):
    a = evaluate(mat(fa), thr, reps, return_reps=True)["reps"]
    b = evaluate(mat(fb), thr, reps, return_reps=True)["reps"]
    dlt = b - a
    return dlt.mean(), float(np.mean(dlt > 0)), float(dlt.mean() / (dlt.std(ddof=1) / np.sqrt(len(dlt)) + 1e-12))


if __name__ == "__main__":
    REPS = 40
    base = evaluate(mat(FE5), reps=REPS)
    print(f"BASELINE 5-feature @CVP>12:  AUC {base['auc']:.3f} [{base['ci'][0]:.3f}-{base['ci'][1]:.3f}]  "
          f"Sens {base['sens']:.2f} Spec {base['spec']:.2f}  ({base['npos']}/{base['n']} elevated)\n")

    print("STUDY A — CVP threshold sweep (5-feature model):")
    for T in [7, 8, 9, 10, 11, 12]:
        r = evaluate(mat(FE5), thr=T, reps=REPS)
        print(f"  CVP>{T:>2}: {r['npos']:>3}/{r['n']} pos ({100*r['npos']/r['n']:4.1f}%)  AUC {r['auc']:.3f}")
    print("  -> lowering the cut balances the cohort but AUC falls to ~0.66; peaks at >12. No help.\n")

    print("STUDY C — invasive arterial BP:")
    for name, fs in [("base5", FE5), ("+art (sys/dia/mean)", FE5 + ART), ("art alone", ART)]:
        r = evaluate(mat(fs), reps=REPS); print(f"  {name:20} AUC {r['auc']:.3f}")
    d, w, t = paired(FE5, FE5 + ART, reps=REPS)
    print(f"  paired +art vs base: delta={d:+.3f}, {100*w:.0f}% of splits improve, t={t:.1f}  -> mildly hurts. No help.\n")

    print("STUDY D — categorical PPG morphology (Dawber class fractions):")
    for name, fs in [("base5", FE5), ("+morph_f1", FE5 + ["morph_f1"]), ("+morph_f4", FE5 + ["morph_f4"]),
                     ("+f1+f4", FE5 + MORPH)]:
        r = evaluate(mat(fs), reps=REPS); print(f"  {name:12} AUC {r['auc']:.3f}  Sens {r['sens']:.2f}")
    d, w, t = paired(FE5, FE5 + MORPH, reps=REPS)
    print(f"  paired +f1+f4 vs base: delta={d:+.3f}, {100*w:.0f}% of splits improve, t={t:.1f}  -> a small, robust GAIN.")
