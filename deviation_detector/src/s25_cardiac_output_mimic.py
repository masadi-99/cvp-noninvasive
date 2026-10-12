"""Step 24b. Cardiac output on MIMIC-III: the VitalDB-trained CO detector, zero-shot.

Label: the 1 Hz 'CO' numeric (continuous cardiac output from a pulmonary-artery catheter),
median over each 30-s feature window, plausible range 1-15 L/min, at least half the window
finite. Same schedule, same artefact-step filter in L/min, same 91 features from the ICU ECG/PPG.
Arms: zero-shot from all VitalDB CO stretches (full model and read columns only), the no-model
read baselines, and a MIMIC-only ceiling five-fold by record.
    VARIANT=gated python src/s25_cardiac_output_mimic.py
Writes results/cardiac_output_mimic.json, results/cardiac_output_mimic_scores.npz, results/cardiac_output_mimic_traces.npz
"""
import os, sys, json, glob, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, RES_BASE, FOLD_SEED, NSEED, VITALDB_TILES
from lib.build import VitalDB, build_record, fit, score, target, boot_ci, feature_names, ACTIVE
THR = (0.5, 0.75, 1.0); MAX_STEP = 2.0; WINM = 3750
t0 = time.time(); V = VitalDB(tracker=False); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
WAVE = [IDX[f"d_{n}"] for n in V.names]; CLK = IDX["since_last_cvp"]; LINE = [i for i in ACTIVE if i not in WAVE and not NAMES[i].startswith("trk_")]
FULL = [i for i in ACTIVE if not NAMES[i].startswith("trk_")]

# ---- VitalDB CO stretches (training) -----------------------------------------------------------------
T = np.load(VITALDB_TILES, allow_pickle=True); Kt = {str(c): i for i, c in enumerate(T["cols"])}; M = T["data"]; cid = M[:, Kt["cid"]].astype(int); tmn = M[:, Kt["t_sec"]] / 60.0
VD = {}
for c, o in V.ORDER.items():
    m = cid == c; key = {round(float(a), 4): i for i, a in enumerate(tmn[m])}; idx = np.array([key.get(round(float(a), 4), -1) for a in V.T[o]]); g = idx >= 0
    y = np.full(len(o), np.nan); y[g] = M[m][idx[g], Kt["co"]]
    if np.isfinite(y).sum() < 12: continue
    ss = build_record(V.T[o], y, V.X[o], f"vitaldb_{c}", max_step=MAX_STEP)
    if ss: VD[f"vitaldb_{c}"] = ss
print(f"VitalDB CO: {len(VD)} patients, {sum(len(v) for v in VD.values())} stretches [{time.time()-t0:.0f}s]", flush=True)

# ---- MIMIC CO stretches ----------------------------------------------------------------------------------
MI = {}; NUM = {}
for p in sorted(glob.glob(f"{RES_BASE}/mimic_numerics/*.npz")):
    rid = os.path.basename(p).replace(".npz", ""); z = np.load(p); co = z["CO"].astype(float); fs = float(z["fs"]); off = float(z["offset_s"])
    ff = f"{RES_BASE}/mimic_feat/mimic3wdb_{rid}.npz"
    if not os.path.exists(ff): print(f"  {rid}: no feature file"); continue
    zf = np.load(ff, allow_pickle=True); t = zf["t"].astype(float); X = zf["X"].astype(float); y = np.full(len(t), np.nan)
    for i, tt in enumerate(t):
        a = int(round((tt * 60 - off) * fs)); b = int(round((tt * 60 + 30 - off) * fs)); seg = co[max(a, 0):max(b, 0)]
        f = np.isfinite(seg) & (seg > 1) & (seg < 15)
        if len(seg) and f.mean() >= .5: y[i] = float(np.median(seg[f]))
    NUM[rid] = dict(windows=int(len(t)), co_windows=int(np.isfinite(y).sum()), co_median=float(np.nanmedian(y)) if np.isfinite(y).any() else None)
    if np.isfinite(y).sum() < 12: continue
    ss = build_record(t, y, X, f"mimic3wdb_{rid}", max_step=MAX_STEP)
    if ss: MI[f"mimic3wdb_{rid}"] = ss
keys = sorted(MI); MSEG = [s for k in keys for s in MI[k]]; G = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG]); Y = {t: np.concatenate([(target(s) > t).astype(int) for s in MSEG]) for t in THR}
print(f"MIMIC CO: {len(NUM)} records with numerics, {len(MI)} with stretches, {len(MSEG)} stretches, {len(G):,} windows; prevalence " + " / ".join(f"{100*Y[t].mean():.1f}%" for t in THR) + f"  [{time.time()-t0:.0f}s]", flush=True)
OUT = dict(records_with_numeric=len(NUM), records=len(MI), stretches=len(MSEG), windows=int(len(G)), prevalence={f"n>{t:g}": float(Y[t].mean()) for t in THR}, thresholds=list(THR), per_record_info=NUM, arms={}, baselines={}, per_record={}); SC = {}
def summ(P, nb=300): return {f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], P)), ci=list(boot_ci(P, Y[t], G, nb))) for t in THR}
def rep(nm, P, kind="arms"):
    OUT[kind][nm] = summ(P); SC[nm] = P
    print(f"  {nm:40s} " + "  ".join(f"{v['auc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]" for v in OUT[kind][nm].values()), flush=True)
rep("zero-shot from VitalDB CO, full", np.concatenate(score(fit(VD, list(VD), cols=FULL), MSEG, cols=FULL)))
rep("zero-shot from VitalDB CO, read columns only", np.concatenate(score(fit(VD, list(VD), cols=LINE), MSEG, cols=LINE)))
rep("zero-shot, waveform + clock", np.concatenate(score(fit(VD, list(VD), cols=WAVE + [CLK]), MSEG, cols=WAVE + [CLK])))
for nm, col, ab in (("read spread x since", "std_x_since", False), ("read |last - median|", "blk_last_minus_median", True), ("clock", "since_last_cvp", False)):
    rep("no model: " + nm, np.concatenate([np.maximum.accumulate(np.abs(s["X"][:, IDX[col]]) if ab else np.nan_to_num(s["X"][:, IDX[col]])) for s in MSEG]), "baselines")
# MIMIC-only ceiling, five-fold by record. The PA-catheter numeric is held between updates, so most
# windows have a drift of exactly 0 and the absolute-error loss (conditional median) returns a
# constant; the ceiling is therefore fitted with squared error, and both losses are reported.
rng = np.random.default_rng(FOLD_SEED); ids = np.array(keys); rng.shuffle(ids)
for loss, nm in (("squared_error", "MIMIC-only ceiling (5-fold by record, squared-error loss)"), ("absolute_error", "MIMIC-only, absolute-error loss (constant: label is 0 on most windows)")):
    PF = {}
    for te in np.array_split(ids, 5):
        tr = [k for k in keys if k not in set(te.tolist())]; segs = [s for k in te.tolist() for s in MI[k]]
        for s, p in zip(segs, score(fit(MI, tr, nseed=2, cols=FULL, loss=loss), segs, cols=FULL)): PF[id(s)] = p
    rep(nm, np.concatenate([PF[id(s)] for s in MSEG]))
OUT["zero_drift_windows"] = float(np.mean(np.concatenate([s["dev"] for s in MSEG]) == 0))
Pf = SC["zero-shot from VitalDB CO, full"]
for t in THR:
    a = [roc_auc_score(Y[t][G == k], Pf[G == k]) for k in keys if Y[t][G == k].min() != Y[t][G == k].max()]
    OUT["per_record"][f"n>{t:g}"] = dict(n=len(a), mean=float(np.mean(a)) if a else None, median=float(np.median(a)) if a else None)
print("  per-record (zero-shot full): " + "  ".join(f"n>{t:g} {OUT['per_record'][f'n>{t:g}']['mean']:.3f} (n={OUT['per_record'][f'n>{t:g}']['n']})" for t in THR if OUT['per_record'][f'n>{t:g}']['mean'] is not None), flush=True)
TR = {}; o = 0
for s in MSEG:
    n = len(s["dev"]); TR[f"{s['key']}:{s['k']}"] = dict(t=s["t"], y=s["y"], tb=s["tb"], yb=s["yb"], base=np.array([s["base"]]), dev=s["dev"], p=Pf[o:o + n]); o += n
np.savez_compressed(f"{RES}/cardiac_output_mimic_traces.npz", **{f"{kk}__{q}": np.asarray(v) for kk, dd in TR.items() for q, v in dd.items()})
np.savez_compressed(f"{RES}/cardiac_output_mimic_scores.npz", G=G, **{f"Y{t:g}": Y[t] for t in THR}, **SC)
json.dump(OUT, open(f"{RES}/cardiac_output_mimic.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/cardiac_output_mimic.json")
