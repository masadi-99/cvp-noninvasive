"""Step 30. Calibration-free absolute classification: the 91 raw ECG/PPG window features, with no
read and no clock, against CVP > 8, 10 and 12 mmHg on the gated VitalDB windows and their a-wave
labels, five folds by patient (the usual seed) over every patient in the window table.

One magnitude regressor of absolute CVP (the same model as everywhere else) gives one ranking that
is scored at the three thresholds; per-threshold classifiers are run as a check. Reported for a
single 30-s window, for 10-minute blocks (mean prediction against the block's median CVP) and for
the whole case (mean prediction against the case's median CVP). Where the signal sits: the
within-patient AUC (windows ranked inside each patient that has both classes) separates
within-patient variation from between-patient level; a heart-rate-only model says how much is heart
rate; and because the raw PPG DC level (ppg_dc, the mean of the raw pleth trace, a monitor-scaled
quantity that the drift analysis removes by differencing against the read) can carry a device
rather than a physiological signal, the model is also run without it, with it alone, and without
every gain-dependent amplitude column.
    VARIANT=gated python src/s30_absolute_cvp.py
Writes results/absolute_cvp.json, results/absolute_cvp_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, FOLD_SEED, NSEED, HGB
from lib.build import VitalDB, boot_ci

t0 = time.time(); THR = (8.0, 10.0, 12.0); MIN_WIN = 3; MIN_CLASS = 5
V = VitalDB(tracker=False); X, y, cid, t = V.X.astype(float), V.Y.astype(float), V.CID.astype(int), V.T.astype(float)
o = np.lexsort((t, cid)); X, y, cid, t = X[o], y[o], cid[o], t[o]
pats = sorted(set(cid.tolist())); rng = np.random.default_rng(FOLD_SEED); ids = np.array(pats); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
NM = list(V.names); ALL = list(range(len(NM)))
HR = [i for i, n in enumerate(NM) if n in ("rr_mean", "ppg_rate", "ppg_nbeat")]
DC = [NM.index("ppg_dc")]; NODC = [i for i in ALL if i not in DC]
GAIN = [i for i, n in enumerate(NM) if n in ("ppg_dc", "ppg_ac_amp", "ppg_baseline_amp", "ppg_alt_amp", "ppg_area", "p_amp", "p_absamp", "p_area", "r_amp")]; NOGAIN = [i for i in ALL if i not in GAIN]
print(f"{len(pats)} patients, {len(y):,} gated windows, CVP median {np.median(y):.1f} mmHg; heart-rate columns {[NM[i] for i in HR]}; gain-dependent columns {[NM[i] for i in GAIN]} [{time.time()-t0:.0f}s]", flush=True)


def oof_reg(cols=None):
    P = np.full(len(y), np.nan); A = X if cols is None else X[:, cols]
    for te in FOLDS:
        tem = np.isin(cid, te); trm = ~tem
        MS = [HistGradientBoostingRegressor(random_state=sd, **HGB).fit(A[trm], y[trm]) for sd in range(NSEED)]
        P[tem] = np.mean([m.predict(A[tem]) for m in MS], 0)
    return P


def oof_clf(thr):
    P = np.full(len(y), np.nan); hp = {k: v for k, v in HGB.items() if k != "loss"}
    for te in FOLDS:
        tem = np.isin(cid, te); trm = ~tem
        MS = [HistGradientBoostingClassifier(random_state=sd, **hp).fit(X[trm], (y[trm] > thr).astype(int)) for sd in range(NSEED)]
        P[tem] = np.mean([m.predict_proba(X[tem])[:, 1] for m in MS], 0)
    return P


# ---- the three levels: window, 10-min block, whole case ----------------------------------------
blk = np.concatenate([np.floor((t[cid == c] - t[cid == c].min()) / 10.0).astype(int) for c in pats])  # cid-sorted order
BID = np.array([f"{c}:{b}" for c, b in zip(cid, blk)])
def aggregate(P, key):
    """Mean prediction and median CVP per unit (block id or case id), units with >= MIN_WIN windows."""
    u, inv = np.unique(key, return_inverse=True); n = np.bincount(inv); keep = n >= MIN_WIN
    pm = np.array([np.mean(P[inv == i]) for i in range(len(u))]); ym = np.array([np.median(y[inv == i]) for i in range(len(u))])
    grp = np.array([k.split(":")[0] for k in u]) if ":" in str(u[0]) else u.astype(str)
    return pm[keep], ym[keep], grp[keep], int(keep.sum())


def levels(P, nb=400):
    out = {}
    for lvl, (pm, ym, grp, n) in (("window", (P, y, cid.astype(str), len(y))), ("block_10min", aggregate(P, BID)), ("case", aggregate(P, cid.astype(str)))):
        out[lvl] = dict(n=int(n), **{f">{thr:g}": dict(auc=float(roc_auc_score(ym > thr, pm)), ci=list(boot_ci(pm, (ym > thr).astype(int), grp, nb)), prevalence=float((ym > thr).mean())) for thr in THR})
    return out


def within_patient(P, min_class=1):
    """AUC inside each patient with at least min_class windows in each class, then mean / median."""
    out = {}
    for thr in THR:
        a = []
        for c in pats:
            yy = y[cid == c] > thr
            if yy.sum() >= min_class and (~yy).sum() >= min_class: a.append(roc_auc_score(yy, P[cid == c]))
        out[f">{thr:g}"] = dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)), frac_gt_0_5=float(np.mean(np.array(a) > .5)))
    return out


def arm(P, key):
    return dict(score_key=key, levels=levels(P), within_patient=within_patient(P), within_patient_min5=within_patient(P, MIN_CLASS), mae=float(np.mean(np.abs(P - y))), corr=float(np.corrcoef(P, y)[0, 1]))


def show(nm, r):
    print(f"  {nm}: MAE {r['mae']:.2f} mmHg, r {r['corr']:.2f}", flush=True)
    for lvl in ("window", "block_10min", "case"): print(f"      {lvl:12s} n={r['levels'][lvl]['n']:6d}  " + "  ".join(f">{thr:g}: {r['levels'][lvl][f'>{thr:g}']['auc']:.3f} (prev {100*r['levels'][lvl][f'>{thr:g}']['prevalence']:.0f}%)" for thr in THR), flush=True)
    print("      within patient " + "  ".join(f">{thr:g}: {r['within_patient'][f'>{thr:g}']['mean']:.3f} (n={r['within_patient'][f'>{thr:g}']['n']})" for thr in THR) + "   with >= 5 per class " + "  ".join(f"{r['within_patient_min5'][f'>{thr:g}']['mean']:.3f} (n={r['within_patient_min5'][f'>{thr:g}']['n']})" for thr in THR) + f"  [{time.time()-t0:.0f}s]", flush=True)


OUT = dict(patients=len(pats), windows=int(len(y)), min_windows_per_unit=MIN_WIN, min_class_windows=MIN_CLASS, thresholds=list(THR), cvp_median=float(np.median(y)),
           columns=dict(all=len(ALL), heart_rate=[NM[i] for i in HR], gain_dependent=[NM[i] for i in GAIN]), arms={}); SC = dict(y=y, cid=cid, t=t, blk=blk)
for nm, cols, key in (("regressor of absolute CVP (91 raw columns)", None, "regressor"), ("without the raw PPG DC level (90 columns)", NODC, "regressor_no_dc"), ("the raw PPG DC level alone", DC, "regressor_dc_only"),
                      (f"without the gain-dependent amplitude columns ({len(NOGAIN)} columns)", NOGAIN, "regressor_no_gain"), ("heart rate only (rr_mean, ppg_rate, ppg_nbeat)", HR, "regressor_heart_rate")):
    P = oof_reg(cols); SC[key] = P; OUT["arms"][nm] = arm(P, key); show(nm, OUT["arms"][nm])
CLF = "classifier per threshold (91 raw columns)"; OUT["arms"][CLF] = dict(score_key="classifier", levels={}, within_patient={}, within_patient_min5={})
for thr in THR:
    PC = oof_clf(thr); SC[f"classifier_{thr:g}"] = PC; L = levels(PC); W = within_patient(PC); W5 = within_patient(PC, MIN_CLASS)
    for lvl in L: OUT["arms"][CLF]["levels"].setdefault(lvl, dict(n=L[lvl]["n"]))[f">{thr:g}"] = L[lvl][f">{thr:g}"]
    OUT["arms"][CLF]["within_patient"][f">{thr:g}"] = W[f">{thr:g}"]; OUT["arms"][CLF]["within_patient_min5"][f">{thr:g}"] = W5[f">{thr:g}"]
    print(f"  classifier >{thr:g}: window {L['window'][f'>{thr:g}']['auc']:.3f}  block {L['block_10min'][f'>{thr:g}']['auc']:.3f}  case {L['case'][f'>{thr:g}']['auc']:.3f}  within patient {W[f'>{thr:g}']['mean']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
np.savez_compressed(f"{RES}/absolute_cvp_scores.npz", **SC)
json.dump(OUT, open(f"{RES}/absolute_cvp.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/absolute_cvp.json")
