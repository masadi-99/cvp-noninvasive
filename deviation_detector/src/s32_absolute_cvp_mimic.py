"""Step 32. Calibration-free absolute CVP on MIMIC-III: the step-30 models (91 raw ECG/PPG features,
no calibration, no elapsed time) trained on every gated VitalDB window and applied without retraining
to the gated windows of the MIMIC-III records used throughout the deck; and, as a reference, the same
models trained and scored within MIMIC-III (five folds by record).

Levels as in step 30: single window, 10-minute average, whole record (mean prediction against the
median CVP); AUC at CVP > 8 / 10 / 12 mmHg with record-clustered intervals; within-record AUC among
records with >= 5 windows on each side of the threshold. Input sets: all 91 features, without the mean
PPG level (ppg_dc), the mean PPG level alone, without every scaling-dependent amplitude feature, heart
rate only. The MIMIC pleth is rescaled per record to VitalDB's pooled median / IQR (step 6).
    VARIANT=gated python src/s32_absolute_cvp_mimic.py
Writes results/absolute_cvp_mimic.json, results/absolute_cvp_mimic_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, FOLD_SEED, NSEED, HGB
from lib.build import VitalDB, boot_ci, mimic_files, cvp_gate_mask

t0 = time.time(); THR = (8.0, 10.0, 12.0); MIN_WIN = 3; MIN_CLASS = 5
V = VitalDB(tracker=False); XV, yV = V.X.astype(float), V.Y.astype(float); NM = list(V.names); ALL = list(range(len(NM)))
HR = [i for i, n in enumerate(NM) if n in ("rr_mean", "ppg_rate", "ppg_nbeat")]; DC = [NM.index("ppg_dc")]; NODC = [i for i in ALL if i not in DC]
GAIN = [i for i, n in enumerate(NM) if n in ("ppg_dc", "ppg_ac_amp", "ppg_baseline_amp", "ppg_alt_amp", "ppg_area", "p_amp", "p_absamp", "p_area", "r_amp")]; NOGAIN = [i for i in ALL if i not in GAIN]
ARMS = {"regressor of absolute CVP (91 raw columns)": (None, "regressor"), "without the raw PPG DC level (90 columns)": (NODC, "regressor_no_dc"), "the raw PPG DC level alone": (DC, "regressor_dc_only"),
        f"without the gain-dependent amplitude columns ({len(NOGAIN)} columns)": (NOGAIN, "regressor_no_gain"), "heart rate only (rr_mean, ppg_rate, ppg_nbeat)": (HR, "regressor_heart_rate")}
# ---- the gated MIMIC-III windows of the deck's record set ---------------------------------------------------
Xs, ys, ts, rs = [], [], [], []
for p in mimic_files("paper"):
    key = os.path.basename(p).replace(".npz", "")
    if not key.startswith("mimic3wdb"): continue
    z = np.load(p, allow_pickle=True); ok = cvp_gate_mask(key, len(z["t"])) & np.isfinite(z["y"])
    if ok.sum() < MIN_WIN: continue
    Xs.append(z["X"][ok].astype(float)); ys.append(z["y"][ok].astype(float)); ts.append(z["t"][ok].astype(float)); rs.append(np.full(int(ok.sum()), key))
X, y, t, rid = np.vstack(Xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(rs)
o = np.lexsort((t, rid)); X, y, t, rid = X[o], y[o], t[o], rid[o]; recs = sorted(set(rid.tolist()))
blk = np.concatenate([np.floor((t[rid == r] - t[rid == r].min()) / 10.0).astype(int) for r in recs]); BID = np.array([f"{r}:{b}" for r, b in zip(rid, blk)])
print(f"VitalDB training: {len(yV):,} windows of {len(V.ORDER)} patients. MIMIC-III: {len(recs)} records, {len(y):,} gated windows, CVP median {np.median(y):.1f} mmHg, above 8/10/12: " + " / ".join(f"{100*(y > th).mean():.0f}%" for th in THR) + f" [{time.time()-t0:.0f}s]", flush=True)


def aggregate(P, key):
    u, inv = np.unique(key, return_inverse=True); n = np.bincount(inv); keep = n >= MIN_WIN
    pm = np.array([np.mean(P[inv == i]) for i in range(len(u))]); ym = np.array([np.median(y[inv == i]) for i in range(len(u))])
    grp = np.array([k.split(":")[0] for k in u]) if ":" in str(u[0]) else u.astype(str)
    return pm[keep], ym[keep], grp[keep], int(keep.sum())


def levels(P, nb=300):
    out = {}
    for lvl, (pm, ym, grp, n) in (("window", (P, y, rid, len(y))), ("block_10min", aggregate(P, BID)), ("record", aggregate(P, rid))):
        out[lvl] = dict(n=int(n), **{f">{thr:g}": dict(auc=float(roc_auc_score(ym > thr, pm)), ci=list(boot_ci(pm, (ym > thr).astype(int), grp, nb)), prevalence=float((ym > thr).mean())) for thr in THR})
    return out


def within_record(P, min_class=MIN_CLASS):
    out = {}
    for thr in THR:
        a = []
        for r in recs:
            yy = y[rid == r] > thr
            if yy.sum() >= min_class and (~yy).sum() >= min_class: a.append(roc_auc_score(yy, P[rid == r]))
        out[f">{thr:g}"] = dict(n=len(a), mean=float(np.mean(a)) if a else None, median=float(np.median(a)) if a else None)
    return out


def show(nm, r):
    print(f"  {nm}: MAE {r['mae']:.2f} mmHg, r {r['corr']:.2f}", flush=True)
    for lvl in ("window", "block_10min", "record"): print(f"      {lvl:12s} n={r['levels'][lvl]['n']:7d}  " + "  ".join(f">{thr:g}: {r['levels'][lvl][f'>{thr:g}']['auc']:.3f} (prev {100*r['levels'][lvl][f'>{thr:g}']['prevalence']:.0f}%)" for thr in THR), flush=True)
    print("      within record (>= 5 per class) " + "  ".join(f">{thr:g}: {r['within_record'][f'>{thr:g}']['mean'] if r['within_record'][f'>{thr:g}']['mean'] is not None else float('nan'):.3f} (n={r['within_record'][f'>{thr:g}']['n']})" for thr in THR) + f"  [{time.time()-t0:.0f}s]", flush=True)


OUT = dict(records=len(recs), windows=int(len(y)), vitaldb_training_windows=int(len(yV)), min_windows_per_unit=MIN_WIN, min_class_windows=MIN_CLASS, thresholds=list(THR), cvp_median=float(np.median(y)),
           columns=dict(all=len(ALL), heart_rate=[NM[i] for i in HR], gain_dependent=[NM[i] for i in GAIN]), transfer={}, within_mimic={}); SC = dict(y=y, rid=rid, t=t, blk=blk)
print("Trained on VitalDB, applied to MIMIC-III without retraining", flush=True)
for nm, (cols, key) in ARMS.items():
    A = XV if cols is None else XV[:, cols]; B = X if cols is None else X[:, cols]
    MS = [HistGradientBoostingRegressor(random_state=sd, **HGB).fit(A, yV) for sd in range(NSEED)]; P = np.mean([m.predict(B) for m in MS], 0); SC[f"transfer_{key}"] = P
    OUT["transfer"][nm] = dict(score_key=f"transfer_{key}", levels=levels(P), within_record=within_record(P), mae=float(np.mean(np.abs(P - y))), corr=float(np.corrcoef(P, y)[0, 1])); show(nm, OUT["transfer"][nm])
print("Trained and scored within MIMIC-III, five folds by record", flush=True)
rng = np.random.default_rng(FOLD_SEED); ids = np.array(recs); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
for nm, (cols, key) in ARMS.items():
    B = X if cols is None else X[:, cols]; P = np.full(len(y), np.nan)
    for te in FOLDS:
        tem = np.isin(rid, te); trm = ~tem
        MS = [HistGradientBoostingRegressor(random_state=sd, **HGB).fit(B[trm], y[trm]) for sd in range(2)]; P[tem] = np.mean([m.predict(B[tem]) for m in MS], 0)
    SC[f"within_{key}"] = P
    OUT["within_mimic"][nm] = dict(score_key=f"within_{key}", levels=levels(P), within_record=within_record(P), mae=float(np.mean(np.abs(P - y))), corr=float(np.corrcoef(P, y)[0, 1])); show(nm, OUT["within_mimic"][nm])
np.savez_compressed(f"{RES}/absolute_cvp_mimic_scores.npz", **SC)
json.dump(OUT, open(f"{RES}/absolute_cvp_mimic.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/absolute_cvp_mimic.json")
