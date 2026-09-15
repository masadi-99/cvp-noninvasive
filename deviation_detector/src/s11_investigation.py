"""Step 11. What is actually doing the work? Baselines, ablations, feature importance,
confounders and who the system works for.

Everything on VitalDB is five-fold out-of-fold with the SAME patient folds as step 3 (the
nested search picked W=10 in every fold, so a plain five-fold at W=10 reproduces the headline
exactly). Everything on MIMIC-III is zero-shot from a model fitted on all of VitalDB through the
record builder (as in steps 7-8). Paired comparisons share the bootstrap resamples.

  baselines     scores that need NO model: the clock, the calibration block's own trend and
                spread, the block-to-block history, the absolute CVP level, the tracker alone
  ablations     the full 123-column model with families removed or the construction changed
  importance    permutation importance per feature family and per single feature, out of fold
  univariate    every column's own AUC and rank correlation with the target
  confounders   heart rate, arterial pressure, absolute CVP level, time since calibration:
                does the model still discriminate when each is held fixed?
  demographics  per-patient AUC across age, sex, BMI, ASA, surgery, position, comorbidity,
                signal quality, CVP level and variability; per-record strata on MIMIC
Writes results/investigation.json and results/investigation_scores.npz
"""
import os, sys, time, json, csv
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RES, RES_BASE, THRS, FOLD_SEED, NSEED, HGB, DESC_NAMES, HIST_NAMES, EXTRA_NAMES
from lib.build import (VitalDB, build_vitaldb, load_vitaldb_generic, load_mimic, fit, score, target,
                       boot_ci, feature_names, ACTIVE)
from lib.features import PPG_FEATS, ECG_FEATS, NF_FEATS

t0 = time.time()
V = VitalDB(); R = build_vitaldb(V)
NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}; ACT = set(ACTIVE)
allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids)
FOLDS = np.array_split(ids, 5)
SEGS = [s for c in allp for s in R[c]]
Y = {thr: np.concatenate([(target(s) > thr).astype(int) for s in SEGS]) for thr in THRS}
G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS])
OUT = dict(vitaldb=dict(patients=len(allp), stretches=len(SEGS), windows=int(len(G))))
SC = {}          # every scored arm, for verify.py
print(f"VitalDB: {len(allp)} patients, {len(SEGS)} stretches, {len(G):,} windows [{time.time()-t0:.0f}s]", flush=True)


def cv_scores(RR, cols=None, target_fn=target, running_max=True, nseed=NSEED, clf_thr=None, **kw):
    """Out-of-fold per-window scores over the fixed folds, in SEGS order."""
    P = {}
    for te in FOLDS:
        tr = [c for c in allp if c not in set(te.tolist())]
        segs = [s for c in te.tolist() if c in RR for s in RR[c]]
        if clf_thr is None:
            MS = fit(RR, tr, nseed, cols=cols, target_fn=target_fn, **kw)
            ps = score(MS, segs, cols=cols, running_max=running_max)
        else:
            A = np.vstack([s["X"] for c in tr for s in RR[c]]); b = np.concatenate([(target(s) > clf_thr).astype(int) for c in tr for s in RR[c]])
            A = A[:, cols if cols is not None else ACTIVE]
            hp = {k: v for k, v in HGB.items() if k != "loss"}
            MS = [HistGradientBoostingClassifier(random_state=sd, **hp).fit(A, b) for sd in range(nseed)]
            BIG = np.vstack([s["X"] for s in segs])[:, cols if cols is not None else ACTIVE]
            pr = np.mean([m.predict_proba(BIG)[:, 1] for m in MS], 0); ps, o = [], 0
            for s in segs: n = len(s["dev"]); ps.append(np.maximum.accumulate(pr[o:o + n])); o += n
        for s, p in zip(segs, ps): P[(s["key"], s["k"])] = p
    return np.concatenate([P[(s["key"], s["k"])] for s in SEGS])


def summarize(P, Yd=None, n_boot=600):
    Yd = Yd or Y
    return {f"n>{thr:g}": dict(auc=float(roc_auc_score(Yd[thr], P)), ci=list(boot_ci(P, Yd[thr], G, n_boot))) for thr in THRS}


def paired(P1, P2, thr, n_boot=600, seed=7):
    """Bootstrap (by patient) of AUC(P1) - AUC(P2) at threshold thr, same resamples for both."""
    yy = Y[thr]; up = np.unique(G); GI = [np.where(G == c)[0] for c in up]
    rb = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        pk = rb.integers(0, len(GI), len(GI)); m2 = np.concatenate([GI[i] for i in pk])
        if yy[m2].min() != yy[m2].max(): d.append(roc_auc_score(yy[m2], P1[m2]) - roc_auc_score(yy[m2], P2[m2]))
    return dict(delta=float(roc_auc_score(yy, P1) - roc_auc_score(yy, P2)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def col_scalar(name, absval=False, running_max=True):
    """A single feature column as a causal score (running max within the stretch)."""
    out = []
    for s in SEGS:
        v = s["X"][:, IDX[name]].astype(float); v = np.abs(v) if absval else v
        v = np.nan_to_num(v, nan=0.0)
        out.append(np.maximum.accumulate(v) if running_max else v)
    return np.concatenate(out)


# ================================================================================================
# 1. the full model and its controls (must reproduce step 3)
# ================================================================================================
P_full = cv_scores(R); SC["full"] = P_full
OUT["full"] = summarize(P_full)
print("full model     " + "  ".join(f"{k} {v['auc']:.3f}[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]" for k, v in OUT["full"].items()), flush=True)

# ================================================================================================
# 2. baselines that need no model
# ================================================================================================
print("\nBASELINES (no model)", flush=True)
BASE = {
    "clock: minutes since last CVP reading": col_scalar("since_last_cvp"),
    "block spread: std": col_scalar("blk_std"),
    "block spread: gated std": col_scalar("blk_std_gated"),
    "block trend extrapolated: |slope| x since": col_scalar("abs_slope_x_since"),
    "block trend: |slope|": col_scalar("blk_slope", absval=True),
    "block spread: std x since": col_scalar("std_x_since"),
    "block: |last - median|": col_scalar("blk_last_minus_median", absval=True),
    "block: peak-to-peak": col_scalar("blk_ptp"),
    "history: |rate| x since": col_scalar("rate_x_since", absval=True),
    "history: |level change from previous block|": col_scalar("hist_delta_level", absval=True),
    "absolute CVP level: block median": col_scalar("blk_median"),
    "tracker alone: running max |prediction|": col_scalar("trk_cummax_abs"),
    "HR change: |d rr_mean|": col_scalar("d_rr_mean", absval=True),
}
_dropped = {n for i, n in enumerate(NAMES) if i not in ACT}
BASE = {nm: P for nm, P in BASE.items() if not (("x since" in nm and any(d.endswith("_x_since") for d in _dropped)) or ("d rr_mean" in nm and "d_rr_mean" in _dropped))}
OUT["baselines"] = {}
for nm, P in BASE.items():
    OUT["baselines"][nm] = summarize(P, n_boot=300); SC[f"base:{nm}"] = P
    print(f"  {nm:48s} " + "  ".join(f"{v['auc']:.3f}" for v in OUT["baselines"][nm].values()), flush=True)
# a small learned model on the calibration line's information only (no ECG/PPG at all)
NOWAVE = [i for i in range(91, 123) if i in ACT]
P = cv_scores(R, cols=NOWAVE); SC["abl:no_ecg_ppg"] = P
OUT["baselines"]["learned, CVP-line information only (no ECG/PPG, 32 cols)"] = summarize(P)
print(f"  {'learned, no ECG/PPG (32 cols)':48s} " + "  ".join(f"{v['auc']:.3f}" for v in OUT["baselines"]["learned, CVP-line information only (no ECG/PPG, 32 cols)"].values()), flush=True)

# ================================================================================================
# 3. ablations
# ================================================================================================
print("\nABLATIONS (5-fold OOF, same folds)", flush=True)
FAM = {
    "PPG (39)": [IDX[f"d_{n}"] for n in PPG_FEATS],
    "ECG (20)": [IDX[f"d_{n}"] for n in ECG_FEATS],
    "nf (32)": [IDX[f"d_{n}"] for n in NF_FEATS],
    "time since CVP (1)": [IDX["since_last_cvp"]],
    "block CVP shape (13)": [IDX[n] for n in DESC_NAMES],
    "history (5)": [IDX[n] for n in HIST_NAMES],
    "block extras (4)": [IDX[n] for n in EXTRA_NAMES[:4]],
    "interactions (4)": [IDX[n] for n in EXTRA_NAMES[4:8]],
    "tracker (5)": [IDX[n] for n in EXTRA_NAMES[8:]],
}
FAM = {k: [i for i in v if i in ACT] for k, v in FAM.items() if any(i in ACT for i in v)}
ALL = sorted(ACT)
def drop(*fams): return [i for i in ALL if not any(i in FAM.get(f, []) for f in fams)]
def keep(*fams): return sorted(set(i for f in fams for i in FAM.get(f, [])))
ARMS = {
    f"full ({len(ALL)})": ALL,
    "- tracker": drop("tracker (5)"),
    "- history": drop("history (5)"),
    "- block CVP shape": drop("block CVP shape (13)"),
    "- block shape, extras, interactions": drop("block CVP shape (13)", "block extras (4)", "interactions (4)"),
    "- time (since + interactions)": drop("time since CVP (1)", "interactions (4)"),
    "- ECG (20)": drop("ECG (20)"),
    "- PPG (39)": drop("PPG (39)"),
    "- nf (32)": drop("nf (32)"),
    "- all ECG/PPG (91)": drop("PPG (39)", "ECG (20)", "nf (32)"),
    "ECG/PPG + time only (92)": keep("PPG (39)", "ECG (20)", "nf (32)", "time since CVP (1)"),
    "PPG + time only (40)": keep("PPG (39)", "time since CVP (1)"),
    "ECG + time only (21)": keep("ECG (20)", "time since CVP (1)"),
    "nf + time only (33)": keep("nf (32)", "time since CVP (1)"),
    "ECG/PPG + time + block shape (105)": keep("PPG (39)", "ECG (20)", "nf (32)", "time since CVP (1)", "block CVP shape (13)"),
}
OUT["ablations"] = {}
for nm, cols in ARMS.items():
    P = P_full if nm.startswith("full") else cv_scores(R, cols=cols)
    SC[f"abl:{nm}"] = P
    OUT["ablations"][nm] = dict(ncols=len(cols), **summarize(P), vs_full={f"n>{thr:g}": paired(P, P_full, thr) for thr in THRS})
    r = OUT["ablations"][nm]
    print(f"  {nm:38s} " + "  ".join(f"{r[k]['auc']:.3f} ({r['vs_full'][k]['delta']:+.3f})" for k in r if k.startswith("n>")), flush=True)

print("\nCONSTRUCTION CHANGES", flush=True)
CON = {}
CON["target: instantaneous |dev| (not cumulative max)"] = cv_scores(R, target_fn=lambda s: s["dev"])
CON["score: raw per-window output (no running max)"] = cv_scores(R, running_max=False)
CON["one seed instead of three"] = cv_scores(R, nseed=1)
CON["squared-error loss"] = cv_scores(R, loss="squared_error")
R_gated = build_vitaldb(V, dense=False)
CON["baseline from the gated block only (no dense grid)"] = None   # different stretch set; handled below
R_notrk = build_vitaldb(V, tracker=False)
for thr in THRS:
    CON[f"classifier on the binary label n>{thr:g}"] = cv_scores(R, clf_thr=thr)
OUT["construction"] = {}
for nm, P in CON.items():
    if P is None: continue
    SC[f"con:{nm}"] = P
    OUT["construction"][nm] = dict(**summarize(P), vs_full={f"n>{thr:g}": paired(P, P_full, thr) for thr in THRS})
    r = OUT["construction"][nm]
    print(f"  {nm:52s} " + "  ".join(f"{r[k]['auc']:.3f} ({r['vs_full'][k]['delta']:+.3f})" for k in r if k.startswith("n>")), flush=True)
# gated-only baseline: a different stretch set, so compare on the stretches common to both
SEGS_g = [s for c in allp if c in R_gated for s in R_gated[c]]
Pg = {}
for te in FOLDS:
    tr = [c for c in allp if c not in set(te.tolist())]
    segs = [s for c in te.tolist() if c in R_gated for s in R_gated[c]]
    for s, p in zip(segs, score(fit(R_gated, tr), segs)): Pg[(s["key"], s["k"])] = p
com = [i for i, s in enumerate(SEGS) if (s["key"], s["k"]) in Pg]
yg = {thr: np.concatenate([(target(s) > thr).astype(int) for s in SEGS_g]) for thr in THRS}
Pg_all = np.concatenate([Pg[(s["key"], s["k"])] for s in SEGS_g]); Gg = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS_g])
OUT["construction"]["baseline from the gated block only (no dense grid)"] = dict(
    stretches=len(SEGS_g), windows=int(len(Pg_all)),
    **{f"n>{thr:g}": dict(auc=float(roc_auc_score(yg[thr], Pg_all)), ci=list(boot_ci(Pg_all, yg[thr], Gg, 600))) for thr in THRS})
r = OUT["construction"]["baseline from the gated block only (no dense grid)"]
print(f"  {'gated-block baseline (own stretch set)':52s} " + "  ".join(f"{r[k]['auc']:.3f}" for k in r if k.startswith("n>")) + f"   {len(SEGS_g)} stretches", flush=True)

# schedule sweep: read what step 3 wrote for other (W, S)
import glob
OUT["schedules"] = {}
for f in sorted(glob.glob(f"{RES}/nested_full_W*_S*_seed{FOLD_SEED}.json")):
    j = json.load(open(f)); OUT["schedules"][f"W{j['meta']['W'][0]:g}/S{j['meta']['S']:g}"] = dict(meta=j["meta"], **j["results"])

# ================================================================================================
# 4. feature importance: permutation, out of fold
# ================================================================================================
print("\nPERMUTATION IMPORTANCE (OOF, AUC drop at n>2 / n>3)", flush=True)
MODELS = {}
for fi, te in enumerate(FOLDS):
    tr = [c for c in allp if c not in set(te.tolist())]
    MODELS[fi] = (fit(R, tr), [s for c in te.tolist() for s in R[c]])
POS = {c: j for j, c in enumerate(ACTIVE)}          # column index -> position in the active matrix
def perm_scores(cols, seed=0):
    P = {}; rp = np.random.default_rng(seed)
    for fi, (MS, segs) in MODELS.items():
        BIG = np.vstack([s["X"] for s in segs])[:, ACTIVE].copy()
        for c in cols:
            if c in POS: BIG[:, POS[c]] = BIG[rp.permutation(len(BIG)), POS[c]]
        pr = np.mean([m.predict(BIG) for m in MS], 0); o = 0
        for s in segs:
            n = len(s["dev"]); P[(s["key"], s["k"])] = np.maximum.accumulate(pr[o:o + n]); o += n
    return np.concatenate([P[(s["key"], s["k"])] for s in SEGS])
base_auc = {thr: roc_auc_score(Y[thr], P_full) for thr in THRS}
GROUPS = dict(FAM)   # families of active columns only
GROUPS.update({
    "PPG amplitude/DC (ac_amp, dc, area, baseline_amp)": [IDX[f"d_{n}"] for n in ("ppg_ac_amp", "ppg_dc", "ppg_area", "ppg_baseline_amp")],
    "PPG pulse timing (upstroke, crest, w50, up_norm, tau)": [IDX[f"d_{n}"] for n in ("ppg_upstroke", "ppg_crest", "ppg_w50", "ppg_up_norm", "ppg_tau")],
    "PPG notch / Dawber (notch_*, dic_height, dia_frac, morph_*)": [IDX[f"d_{n}"] for n in ("ppg_notch_frac", "ppg_notch_rel", "ppg_dic_height", "ppg_dia_frac", "ppg_morph_f1", "ppg_morph_f4", "ppg_morph_mean")],
    "PPG respiratory / venous (riiv*, vci, pvi, resp_*, lf_frac)": [IDX[f"d_{n}"] for n in ("ppg_riiv", "ppg_riiv_norm", "ppg_vci", "ppg_pvi", "ppg_resp_rate", "ppg_resp_pow_frac", "ppg_lf_frac")],
    "PPG beat-to-beat (alternans, alt_amp, amp_cv, ibi_*, amp_slope, rate, nbeat)": [IDX[f"d_{n}"] for n in ("ppg_alternans", "ppg_alt_amp", "ppg_amp_cv", "ppg_ibi_cv", "ppg_ibi_rmssd", "ppg_amp_slope", "ppg_rate", "ppg_nbeat")],
    "PPG ensemble shape (dct1-5, skew, kurt, awave)": [IDX[f"d_{n}"] for n in ("ppg_dct1", "ppg_dct2", "ppg_dct3", "ppg_dct4", "ppg_dct5", "ppg_skew", "ppg_kurt", "ppg_awave")],
    "ECG P wave (p_*)": [IDX[f"d_{n}"] for n in ECG_FEATS if n.startswith("p_")],
    "ECG rhythm / QRS (rr_*, af_score, qrs_width, r_amp*)": [IDX[f"d_{n}"] for n in ECG_FEATS if not n.startswith("p_")],
    "nf ensemble shape (dct6-12, curv, asym, area1-3, sd_b-e)": [IDX[f"d_{n}"] for n in NF_FEATS if n.startswith(("nf_dct", "nf_curv", "nf_asym", "nf_area", "nf_sd_"))],
    "nf ECG-PPG timing (pat*, slope_delay)": [IDX[f"d_{n}"] for n in ("nf_pat", "nf_pat_cv", "nf_pat_rr", "nf_slope_delay")],
    "nf respiratory coupling (resp_phase, resp_coh, mod_ratio)": [IDX[f"d_{n}"] for n in ("nf_resp_phase", "nf_resp_coh", "nf_mod_ratio")],
    "nf beat dynamics (sd1, sd2, sd12, ibi_sd1, amp_ibi_xcorr, dfa)": [IDX[f"d_{n}"] for n in ("nf_sd1", "nf_sd2", "nf_sd12", "nf_ibi_sd1", "nf_amp_ibi_xcorr", "nf_dfa")],
    "nf two-Gaussian (g_ratio, g_sep, g_width)": [IDX[f"d_{n}"] for n in ("nf_g_ratio", "nf_g_sep", "nf_g_width")],
})
GROUPS = {k: [i for i in v if i in ACT] for k, v in GROUPS.items() if any(i in ACT for i in v)}
OUT["importance_groups"] = {}
for nm, cols in GROUPS.items():
    d = {}
    for seed in range(3):
        P = perm_scores(cols, seed)
        for thr in THRS: d.setdefault(f"n>{thr:g}", []).append(base_auc[thr] - roc_auc_score(Y[thr], P))
    OUT["importance_groups"][nm] = {k: float(np.mean(v)) for k, v in d.items()}
    print(f"  {nm:70s} " + "  ".join(f"{np.mean(v):+.4f}" for v in d.values()), flush=True)
OUT["importance_single"] = {}
for c, nm in enumerate(NAMES):
    if c not in ACT: continue
    d = {}
    for seed in range(2):
        P = perm_scores([c], seed)
        for thr in (2.0, 3.0): d.setdefault(f"n>{thr:g}", []).append(base_auc[thr] - roc_auc_score(Y[thr], P))
    OUT["importance_single"][nm] = {k: float(np.mean(v)) for k, v in d.items()}
top = sorted(OUT["importance_single"], key=lambda n: -OUT["importance_single"][n]["n>2"])[:20]
print("  top single features (n>2 drop): " + ", ".join(f"{n} {OUT['importance_single'][n]['n>2']:+.4f}" for n in top), flush=True)

# univariate: AUC of each column as a running-max score, and Spearman with the target
OUT["univariate"] = {}
A_all = np.vstack([s["X"] for s in SEGS]); tg_all = np.concatenate([target(s) for s in SEGS])
for c, nm in enumerate(NAMES):
    if c not in ACT: continue
    v = np.nan_to_num(A_all[:, c].astype(float), nan=0.0)
    r = spearmanr(v, tg_all).correlation
    a2 = roc_auc_score(Y[2.0], v); a2 = max(a2, 1 - a2)
    OUT["univariate"][nm] = dict(spearman=float(r) if np.isfinite(r) else 0.0, auc_n2=float(a2))

# ================================================================================================
# 5. confounders
# ================================================================================================
print("\nCONFOUNDERS", flush=True)
CF = {}
# monitor numerics on the dense grid, matched to each window's time; change vs the block median
def numeric_change(name):
    out = []
    for s in SEGS:
        d = V.NUM[s["key"]]; u = d["t"]; v = d[name]
        blk = (u >= s["tb"][0]) & (u <= s["tb"][-1]) & np.isfinite(v)
        ref = np.median(v[blk]) if blk.sum() >= 3 else np.nan
        key = {round(float(a), 4): i for i, a in enumerate(u)}
        ii = np.array([key.get(round(float(a), 4), -1) for a in s["t"]])
        val = np.where(ii >= 0, v[np.clip(ii, 0, None)], np.nan)
        out.append(val - ref)
    return np.concatenate(out)
dHR, dMAP, dSPO2 = numeric_change("hr"), numeric_change("map"), numeric_change("spo2")
SINCE = col_scalar("since_last_cvp", running_max=False)
LEVEL = col_scalar("blk_median", running_max=False)
np.savez_compressed(f"{RES}/investigation_covariates.npz", dHR=dHR, dMAP=dMAP, dSPO2=dSPO2, since=SINCE, level=LEVEL)
def strata_auc(P, z, edges, label):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi) & np.isfinite(z)
        r = {"stratum": f"{label} [{lo:g},{hi:g})", "n": int(m.sum())}
        for thr in THRS:
            yy = Y[thr][m]
            r[f"n>{thr:g}"] = float(roc_auc_score(yy, P[m])) if m.sum() > 100 and yy.min() != yy.max() else None
        r["prevalence_n2"] = float(Y[2.0][m].mean()) if m.sum() else None
        rows.append(r)
    return rows
def cov_block(name, z, edges, absval=True):
    zz = np.abs(z) if absval else z
    fin = np.isfinite(zz)
    d = dict(coverage=float(fin.mean()),
             auc_as_score={f"n>{thr:g}": (lambda a: float(max(a, 1 - a)))(roc_auc_score(Y[thr][fin], np.nan_to_num(zz[fin]))) for thr in THRS} if fin.sum() > 1000 else None,
             spearman_with_score=float(spearmanr(zz[fin], P_full[fin]).correlation) if fin.sum() > 1000 else None,
             spearman_with_truth=float(spearmanr(zz[fin], tg_all[fin]).correlation) if fin.sum() > 1000 else None,
             model_within_strata=strata_auc(P_full, zz, edges, name),
             clock_within_strata=strata_auc(SINCE, zz, edges, name))
    return d
CF["|dHR| (bpm, vs block median)"] = cov_block("|dHR|", dHR, [0, 3, 6, 10, 20, 1e9])
CF["|dMAP| (mmHg, vs block median)"] = cov_block("|dMAP|", dMAP, [0, 5, 10, 20, 1e9])
CF["|dSpO2| (%, vs block median)"] = cov_block("|dSpO2|", dSPO2, [0, 1, 2, 1e9])
CF["baseline CVP level (block median, mmHg)"] = cov_block("level", LEVEL, [-10, 5, 8, 11, 15, 60], absval=False)
CF["minutes since last CVP reading"] = cov_block("since", SINCE, [0, 3, 6, 9, 12, 20], absval=False)
# does the model beat the clock when time is held fixed?  AUC within 1-minute since-bins, averaged
bins = np.floor(SINCE).astype(int); vals, ws = [], []
for b in np.unique(bins):
    m = bins == b; yy = Y[2.0][m]
    if m.sum() > 200 and yy.min() != yy.max(): vals.append(roc_auc_score(yy, P_full[m])); ws.append(m.sum())
CF["model AUC within 1-min time bins (n>2), window-weighted mean"] = float(np.average(vals, weights=ws))
# HR as a feature: the model without any rate information
NORATE = [i for i in ALL if NAMES[i] not in ("d_rr_mean", "d_ppg_rate", "d_ppg_nbeat", "d_rr_cv", "d_rr_rmssd", "d_ppg_ibi_cv", "d_ppg_ibi_rmssd", "d_nf_ibi_sd1")]
P = cv_scores(R, cols=NORATE); SC["abl:no_rate"] = P
CF["model without any heart-rate / rhythm column"] = dict(ncols=len(NORATE), **summarize(P), vs_full={f"n>{thr:g}": paired(P, P_full, thr) for thr in THRS})
# per-patient AUC vs pooled
PP = {}
for c in allp:
    m = G == c; yy = Y[2.0][m]
    if yy.min() != yy.max(): PP[c] = float(roc_auc_score(yy, P_full[m]))
CF["per_patient_auc_n2"] = dict(n=len(PP), mean=float(np.mean(list(PP.values()))), median=float(np.median(list(PP.values()))),
                                frac_gt_0_5=float(np.mean(np.array(list(PP.values())) > .5)), pooled=float(base_auc[2.0]))
OUT["confounders"] = CF
for k, v in CF.items():
    if isinstance(v, dict) and "model_within_strata" in v:
        print(f"  {k}: coverage {100*v['coverage']:.0f}%  as a score n>2 {v['auc_as_score']['n>2'] if v['auc_as_score'] else float('nan'):.3f}  "
              f"rho(score) {v['spearman_with_score']:+.2f}  rho(truth) {v['spearman_with_truth']:+.2f}")
        for r in v["model_within_strata"]: print(f"      {r['stratum']:28s} n={r['n']:6d}  model n>2 {r['n>2']}  n>3 {r['n>3']}")
print(f"  model within 1-min time bins: {CF['model AUC within 1-min time bins (n>2), window-weighted mean']:.3f}")
print(f"  per-patient AUC n>2: mean {CF['per_patient_auc_n2']['mean']:.3f} (n={CF['per_patient_auc_n2']['n']}) vs pooled {base_auc[2.0]:.3f}")

# ================================================================================================
# 6. demographics: who is it good / bad for (VitalDB)
# ================================================================================================
print("\nDEMOGRAPHICS (per-patient OOF AUC at n>2, and pooled AUC within stratum)", flush=True)
CL = {int(r["caseid"]): r for r in csv.DictReader(open(f"{DATA}/vitaldb_clinical.csv"))}
def num(x):
    try: return float(x.replace(">", "").replace("<", ""))
    except Exception: return np.nan
# per-patient attributes
PA = {}
for c in allp:
    r = CL[c]; m = G == c; ss = R[c]
    A = np.vstack([s["X"] for s in ss])
    PA[c] = dict(age=num(r["age"]), sex=r["sex"], bmi=num(r["bmi"]), asa=r["asa"], dept=r["department"], optype=r["optype"],
                 approach=r["approach"], position=r["position"], emop=r["emop"], htn=r["preop_htn"], dm=r["preop_dm"],
                 ecg_normal="Normal" in r["preop_ecg"], ebl=num(r["intraop_ebl"]), icu=num(r["icu_days"]),
                 n_stretch=len(ss), n_windows=int(m.sum()), prev2=float(Y[2.0][m].mean()),
                 cvp_level=float(np.median([s["base"] for s in ss])),
                 cvp_sd=float(np.mean([np.std(s["y"]) for s in ss])),
                 hr=float(np.nanmean(60000 / V.X[V.ORDER[c]][:, V.names.index("rr_mean")])),
                 ppg_nbeat=float(np.nanmean(V.X[V.ORDER[c]][:, V.names.index("ppg_nbeat")])),
                 r_amp=float(np.nanmean(np.abs(V.X[V.ORDER[c]][:, V.names.index("r_amp")]))),
                 p_found=float(np.nanmean(np.isfinite(V.X[V.ORDER[c]][:, V.names.index("p_amp")]))),
                 span_min=float(np.ptp(V.T[V.ORDER[c]])), auc2=PP.get(c))
def strat(nm, fn, cats=None, q=None):
    vals = {c: fn(PA[c]) for c in allp}
    if q is not None:
        v = np.array([vals[c] for c in allp], float); edges = np.nanquantile(v, np.linspace(0, 1, q + 1))
        lab = {}
        for c in allp:
            x = vals[c]
            if not np.isfinite(x): lab[c] = "missing"; continue
            j = min(int(np.searchsorted(edges, x, "right")) - 1, q - 1); lab[c] = f"{edges[j]:.3g}-{edges[j+1]:.3g}"
        vals = lab
    rows = []
    for cat in (cats or sorted(set(vals.values()), key=lambda s: (s == "missing", s))):
        cs = [c for c in allp if vals[c] == cat]
        if len(cs) < 8: continue
        m = np.isin(G, cs); au = [PA[c]["auc2"] for c in cs if PA[c]["auc2"] is not None]
        rb = np.random.default_rng(0); bs = [np.mean(rb.choice(au, len(au))) for _ in range(500)] if len(au) > 3 else [np.nan]
        rows.append(dict(stratum=str(cat), patients=len(cs), evaluable=len(au), per_patient_mean=float(np.mean(au)) if au else None,
                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if au else None,
                         pooled={f"n>{thr:g}": (float(roc_auc_score(Y[thr][m], P_full[m])) if Y[thr][m].min() != Y[thr][m].max() else None) for thr in THRS},
                         prevalence_n2=float(Y[2.0][m].mean()), windows=int(m.sum())))
    return rows
DEMO = {
    "age (tertiles)": strat("age", lambda p: p["age"], q=3),
    "sex": strat("sex", lambda p: p["sex"]),
    "BMI (tertiles)": strat("bmi", lambda p: p["bmi"], q=3),
    "ASA class": strat("asa", lambda p: {"1": "ASA 1", "2": "ASA 2", "3": "ASA 3", "4": "ASA 3+", "6": "ASA 3+"}.get(p["asa"], "missing")),
    "department": strat("dept", lambda p: p["dept"]),
    "operation type": strat("optype", lambda p: p["optype"]),
    "approach": strat("approach", lambda p: p["approach"]),
    "position": strat("position", lambda p: {"Supine": "Supine", "Left lateral decubitus": "Lateral", "Right lateral decubitus": "Lateral",
                                             "Lithotomy": "Lithotomy"}.get(p["position"], "other/missing")),
    "emergency": strat("emop", lambda p: {"0": "elective", "1": "emergency"}.get(p["emop"], "missing")),
    "pre-op hypertension": strat("htn", lambda p: {"0": "no", "1": "yes"}.get(p["htn"], "missing")),
    "pre-op diabetes": strat("dm", lambda p: {"0": "no", "1": "yes"}.get(p["dm"], "missing")),
    "pre-op ECG": strat("ecg", lambda p: "normal sinus" if p["ecg_normal"] else "abnormal"),
    "blood loss (tertiles, mL)": strat("ebl", lambda p: p["ebl"], q=3),
    "baseline CVP level (tertiles, mmHg)": strat("level", lambda p: p["cvp_level"], q=3),
    "within-stretch CVP SD (tertiles, mmHg)": strat("sd", lambda p: p["cvp_sd"], q=3),
    "event prevalence n>2 (tertiles)": strat("prev", lambda p: p["prev2"], q=3),
    "heart rate (tertiles, bpm)": strat("hr", lambda p: p["hr"], q=3),
    "PPG beats accepted per window (tertiles)": strat("nbeat", lambda p: p["ppg_nbeat"], q=3),
    "ECG R amplitude (tertiles)": strat("ramp", lambda p: p["r_amp"], q=3),
    "P wave found (fraction of windows, tertiles)": strat("pfound", lambda p: p["p_found"], q=3),
    "stretches per patient (tertiles)": strat("nstr", lambda p: p["n_stretch"], q=3),
    "gated record span (tertiles, min)": strat("span", lambda p: p["span_min"], q=3),
}
OUT["demographics_vitaldb"] = DEMO
for k, rows in DEMO.items():
    print(f"  {k}")
    for r in rows:
        pm = f"{r['per_patient_mean']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}]" if r["per_patient_mean"] is not None else "   n/a"
        print(f"      {r['stratum']:24s} pts {r['patients']:4d} (eval {r['evaluable']:3d})  per-patient {pm}  pooled n>2 {r['pooled']['n>2']}  prev {100*r['prevalence_n2']:.0f}%")
# what characterises the worst patients?  correlate per-patient AUC with each numeric attribute
ok_ = [c for c in allp if PA[c]["auc2"] is not None]
OUT["per_patient_correlates"] = {}
for k in ("age", "bmi", "ebl", "n_stretch", "n_windows", "prev2", "cvp_level", "cvp_sd", "hr", "ppg_nbeat", "r_amp", "p_found", "span_min"):
    x = np.array([PA[c][k] for c in ok_], float); a = np.array([PA[c]["auc2"] for c in ok_])
    f = np.isfinite(x)
    OUT["per_patient_correlates"][k] = dict(spearman=float(spearmanr(x[f], a[f]).correlation), n=int(f.sum()))
print("  correlates of per-patient AUC: " + ", ".join(f"{k} {v['spearman']:+.2f}" for k, v in OUT["per_patient_correlates"].items()))
json.dump({k: v for k, v in PA.items()}, open(f"{RES}/vitaldb_patient_table.json", "w"))

# ================================================================================================
# 7. MIMIC: zero-shot ablations, importance, confounders, record strata
# ================================================================================================
print("\nMIMIC-III zero-shot", flush=True)
VD = load_vitaldb_generic(VitalDB(tracker=False)); MI = load_mimic("mimic3wdb", extra=True)
m3 = sorted(MI); MSEG = [s for k in m3 for s in MI[k]]
YM = {thr: np.concatenate([(target(s) > thr).astype(int) for s in MSEG]) for thr in THRS}
GM = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG])
MSC = {}
def zs(cols=None, **kw):
    MS = fit(VD, list(VD), drop_nan_rows=True, cols=cols, **kw)
    return np.concatenate(score(MS, MSEG, cols=cols))
def msum(P, n_boot=400):
    return {f"n>{thr:g}": dict(auc=float(roc_auc_score(YM[thr], P)), ci=list(boot_ci(P, YM[thr], GM, n_boot))) for thr in THRS}
def mpaired(P1, P2, thr, n_boot=400, seed=7):
    yy = YM[thr]; up = np.unique(GM); GI = [np.where(GM == c)[0] for c in up]
    rb = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        pk = rb.integers(0, len(GI), len(GI)); m2 = np.concatenate([GI[i] for i in pk])
        if yy[m2].min() != yy[m2].max(): d.append(roc_auc_score(yy[m2], P1[m2]) - roc_auc_score(yy[m2], P2[m2]))
    return dict(delta=float(roc_auc_score(yy, P1) - roc_auc_score(yy, P2)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
PM_full = zs(); MSC["full"] = PM_full
OUT["mimic"] = dict(records=len(m3), stretches=len(MSEG), windows=int(len(GM)), full=msum(PM_full))
print("  full  " + "  ".join(f"{k} {v['auc']:.3f}" for k, v in OUT["mimic"]["full"].items()), flush=True)
def mcol(name, absval=False, running_max=True):
    out = []
    for s in MSEG:
        v = np.nan_to_num(s["X"][:, IDX[name]].astype(float), nan=0.0); v = np.abs(v) if absval else v
        out.append(np.maximum.accumulate(v) if running_max else v)
    return np.concatenate(out)
OUT["mimic"]["baselines"] = {}
for nm, (col, ab) in {"clock: minutes since last CVP reading": ("since_last_cvp", False),
                      "block trend extrapolated: |slope| x since": ("abs_slope_x_since", False),
                      "block spread: std x since": ("std_x_since", False), "block spread: std": ("blk_std", False),
                      "block spread: gated std": ("blk_std_gated", False), "block: |last - median|": ("blk_last_minus_median", True),
                      "history: |rate| x since": ("rate_x_since", True), "absolute CVP level: block median": ("blk_median", False),
                      "HR change: |d rr_mean|": ("d_rr_mean", True)}.items():
    if IDX[col] not in ACT: continue
    P = mcol(col, ab); MSC[f"base:{nm}"] = P; OUT["mimic"]["baselines"][nm] = msum(P, 200)
    print(f"  {nm:48s} " + "  ".join(f"{v['auc']:.3f}" for v in OUT['mimic']['baselines'][nm].values()), flush=True)
OUT["mimic"]["ablations"] = {}
MARMS = {k: v for k, v in ARMS.items() if "tracker" not in k}
for nm, cols in MARMS.items():
    P = PM_full if nm.startswith("full") else zs(cols=cols); MSC[f"abl:{nm}"] = P
    OUT["mimic"]["ablations"][nm] = dict(ncols=len(cols), **msum(P), vs_full={f"n>{thr:g}": mpaired(P, PM_full, thr) for thr in THRS})
    r = OUT["mimic"]["ablations"][nm]
    print(f"  {nm:38s} " + "  ".join(f"{r[k]['auc']:.3f} ({r['vs_full'][k]['delta']:+.3f})" for k in r if k.startswith("n>")), flush=True)
P = zs(cols=NORATE); MSC["abl:no_rate"] = P
OUT["mimic"]["no_rate"] = dict(**msum(P), vs_full={f"n>{thr:g}": mpaired(P, PM_full, thr) for thr in THRS})
# permutation importance, zero-shot
MSM = fit(VD, list(VD), drop_nan_rows=True); MBIG = np.vstack([s["X"] for s in MSEG])[:, ACTIVE]; lens = [len(s["dev"]) for s in MSEG]
def mperm(cols, seed=0):
    B = MBIG.copy(); rp = np.random.default_rng(seed)
    for c in cols:
        if c in POS: B[:, POS[c]] = B[rp.permutation(len(B)), POS[c]]
    pr = np.mean([m.predict(B) for m in MSM], 0); out, o = [], 0
    for n in lens: out.append(np.maximum.accumulate(pr[o:o + n])); o += n
    return np.concatenate(out)
mb = {thr: roc_auc_score(YM[thr], PM_full) for thr in THRS}
OUT["mimic"]["importance_groups"] = {}
for nm, cols in GROUPS.items():
    if nm == "tracker (5)": continue
    P = mperm(cols); OUT["mimic"]["importance_groups"][nm] = {f"n>{thr:g}": float(mb[thr] - roc_auc_score(YM[thr], P)) for thr in THRS}
OUT["mimic"]["importance_single"] = {}
for c, nm in enumerate(NAMES):
    if c >= 118 or c not in ACT: continue
    P = mperm([c]); OUT["mimic"]["importance_single"][nm] = {f"n>{thr:g}": float(mb[thr] - roc_auc_score(YM[thr], P)) for thr in (2.0, 3.0)}
topm = sorted(OUT["mimic"]["importance_single"], key=lambda n: -OUT["mimic"]["importance_single"][n]["n>2"])[:20]
print("  MIMIC top single features: " + ", ".join(f"{n} {OUT['mimic']['importance_single'][n]['n>2']:+.4f}" for n in topm), flush=True)
# confounders on MIMIC: MAP from the arterial line, HR from rr_mean
# the block has no MAP reference stored; the stretch's first tracked window is the reference
MAPd = np.concatenate([(s["map"] - s["map"][0]) if "map" in s and np.isfinite(s["map"][0]) else np.full(len(s["dev"]), np.nan) for s in MSEG])
HRd = np.concatenate([s["X"][:, IDX["d_rr_mean"]] for s in MSEG])
MSINCE = mcol("since_last_cvp", running_max=False); MLEVEL = mcol("blk_median", running_max=False)
def mstrata(P, z, edges, label):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi) & np.isfinite(z); r = {"stratum": f"{label} [{lo:g},{hi:g})", "n": int(m.sum())}
        for thr in THRS:
            yy = YM[thr][m]; r[f"n>{thr:g}"] = float(roc_auc_score(yy, P[m])) if m.sum() > 500 and yy.min() != yy.max() else None
        r["prevalence_n2"] = float(YM[2.0][m].mean()) if m.sum() else None; rows.append(r)
    return rows
def mcov(name, z, edges, absval=True):
    zz = np.abs(z) if absval else z; fin = np.isfinite(zz)
    return dict(coverage=float(fin.mean()),
                auc_as_score={f"n>{thr:g}": (lambda a: float(max(a, 1 - a)))(roc_auc_score(YM[thr][fin], np.nan_to_num(zz[fin]))) for thr in THRS},
                spearman_with_score=float(spearmanr(zz[fin], PM_full[fin]).correlation),
                model_within_strata=mstrata(PM_full, zz, edges, name), clock_within_strata=mstrata(MSINCE, zz, edges, name))
OUT["mimic"]["confounders"] = {
    "|dMAP| (mmHg, vs first tracked window; art-line records)": mcov("|dMAP|", MAPd, [0, 5, 10, 20, 1e9]),
    "|d rr_mean| (ms, vs block median)": mcov("|dRR|", HRd, [0, 30, 60, 120, 1e9]),
    "baseline CVP level (block median, mmHg)": mcov("level", MLEVEL, [-10, 5, 8, 11, 15, 60], absval=False),
    "minutes since last CVP reading": mcov("since", MSINCE, [0, 3, 6, 9, 12, 20], absval=False),
}
for k, v in OUT["mimic"]["confounders"].items():
    print(f"  {k}: coverage {100*v['coverage']:.0f}%  as score n>2 {v['auc_as_score']['n>2']:.3f}  rho(score) {v['spearman_with_score']:+.2f}")
    for r in v["model_within_strata"]: print(f"      {r['stratum']:28s} n={r['n']:7d}  model n>2 {r['n>2']}  n>3 {r['n>3']}")
np.savez_compressed(f"{RES}/investigation_mimic_covariates.npz", dMAP=MAPd, dRR=HRd, since=MSINCE, level=MLEVEL)
# record strata on MIMIC (per-record AUC at n>2 from the full zero-shot score)
J8 = json.load(open(f"{RES}/mimic_analysis.json"))["per_record"]["n>2"]["per_record"]
sv = {}
for f in glob.glob(f"{RES_BASE}/mimic/survey_mimic3wdb_*.json"):
    for r in json.load(open(f)):
        if "rec" in r: sv[f"mimic3wdb_{r['rec']}"] = r
RA = {}
for k in m3:
    ss = MI[k]; z = np.load(f"{RES_BASE}/mimic_feat/{k}.npz", allow_pickle=True); sc = z["scale"]
    Xr = np.vstack([s["X"] for s in ss])
    RA[k] = dict(auc2=J8.get(k), hours=sv.get(k, {}).get("hours"), has_art=sv.get(k, {}).get("has_art"),
                 ok_frac=sv.get(k, {}).get("ok_frac"), n_stretch=len(ss), prev2=float(np.mean(np.concatenate([(target(s) > 2).astype(int) for s in ss]))),
                 cvp_level=float(np.median([s["base"] for s in ss])), cvp_sd=float(np.mean([np.std(s["y"]) for s in ss])),
                 hr=float(np.nanmean(60000 / z["X"][:, V.names.index("rr_mean")])), ppg_nbeat=float(np.nanmean(z["X"][:, V.names.index("ppg_nbeat")])),
                 p_found=float(np.mean(np.isfinite(z["X"][:, V.names.index("p_amp")]))), ppg_raw_iqr=float(sc[1]), ecg_raw_iqr=float(sc[3]),
                 windows=int(len(z["t"])))
def mstrat(fn, q=3, cats=None):
    vals = {k: fn(RA[k]) for k in m3}
    if q:
        v = np.array([vals[k] for k in m3], float); edges = np.nanquantile(v, np.linspace(0, 1, q + 1)); lab = {}
        for k in m3:
            x = vals[k]
            if not np.isfinite(x): lab[k] = "missing"; continue
            j = min(int(np.searchsorted(edges, x, "right")) - 1, q - 1); lab[k] = f"{edges[j]:.3g}-{edges[j+1]:.3g}"
        vals = lab
    rows = []
    for cat in (cats or sorted(set(vals.values()), key=lambda s: (s == "missing", s))):
        ks = [k for k in m3 if vals[k] == cat]
        if len(ks) < 8: continue
        m = np.isin(GM, ks); au = [RA[k]["auc2"] for k in ks if RA[k]["auc2"] is not None]
        rb = np.random.default_rng(0); bs = [np.mean(rb.choice(au, len(au))) for _ in range(500)]
        rows.append(dict(stratum=str(cat), records=len(ks), evaluable=len(au), per_record_mean=float(np.mean(au)),
                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         pooled={f"n>{thr:g}": (float(roc_auc_score(YM[thr][m], PM_full[m])) if YM[thr][m].min() != YM[thr][m].max() else None) for thr in THRS},
                         prevalence_n2=float(YM[2.0][m].mean()), windows=int(m.sum())))
    return rows
OUT["mimic"]["strata"] = {
    "record length (tertiles, h)": mstrat(lambda r: r["hours"]),
    "arterial line present": mstrat(lambda r: {1: "yes", 0: "no"}.get(r["has_art"], "missing"), q=0),
    "joint-valid fraction of record (tertiles)": mstrat(lambda r: r["ok_frac"]),
    "baseline CVP level (tertiles, mmHg)": mstrat(lambda r: r["cvp_level"]),
    "within-stretch CVP SD (tertiles, mmHg)": mstrat(lambda r: r["cvp_sd"]),
    "event prevalence n>2 (tertiles)": mstrat(lambda r: r["prev2"]),
    "heart rate (tertiles, bpm)": mstrat(lambda r: r["hr"]),
    "PPG beats accepted per window (tertiles)": mstrat(lambda r: r["ppg_nbeat"]),
    "P wave found (tertiles)": mstrat(lambda r: r["p_found"]),
    "raw PPG scale before normalisation, IQR (tertiles)": mstrat(lambda r: r["ppg_raw_iqr"]),
    "raw ECG scale before normalisation, IQR (tertiles)": mstrat(lambda r: r["ecg_raw_iqr"]),
    "stretches per record (tertiles)": mstrat(lambda r: r["n_stretch"]),
}
okm = [k for k in m3 if RA[k]["auc2"] is not None]
OUT["mimic"]["per_record_correlates"] = {}
for k in ("hours", "ok_frac", "n_stretch", "prev2", "cvp_level", "cvp_sd", "hr", "ppg_nbeat", "p_found", "ppg_raw_iqr", "ecg_raw_iqr"):
    x = np.array([RA[r][k] if RA[r][k] is not None else np.nan for r in okm], float); a = np.array([RA[r]["auc2"] for r in okm]); f = np.isfinite(x)
    OUT["mimic"]["per_record_correlates"][k] = dict(spearman=float(spearmanr(x[f], a[f]).correlation), n=int(f.sum()))
for k, rows in OUT["mimic"]["strata"].items():
    print(f"  {k}")
    for r in rows: print(f"      {r['stratum']:20s} recs {r['records']:4d}  per-record {r['per_record_mean']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}]  pooled n>2 {r['pooled']['n>2']:.3f}  prev {100*r['prevalence_n2']:.0f}%")
print("  correlates of per-record AUC: " + ", ".join(f"{k} {v['spearman']:+.2f}" for k, v in OUT["mimic"]["per_record_correlates"].items()))
json.dump(RA, open(f"{RES}/mimic_record_table.json", "w"))

np.savez_compressed(f"{RES}/investigation_scores.npz", Y2=Y[2.0], Y3=Y[3.0], Y4=Y[4.0], G=G,
                    **{f"vd__{k}": v for k, v in SC.items()},
                    YM2=YM[2.0], YM3=YM[3.0], YM4=YM[4.0], GM=GM, **{f"mi__{k}": v for k, v in MSC.items()})
json.dump(OUT, open(f"{RES}/investigation.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s] -> results/investigation.json, results/investigation_scores.npz")
