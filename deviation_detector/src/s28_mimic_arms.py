"""Step 28. MIMIC-III, zero-shot on the gated records: clock only, waveform only, waveform + clock,
read columns + clock (no waveform), full model. Pooled AUC with a record-clustered interval, and per
record (AUC inside every record that has both classes, then mean / median), each with the running-
maximum score and with the raw per-window prediction; and the AUC inside 0-5, 5-10 and 10-15 min
since the read.

Training as in step 7: VitalDB through the record builder (the MIMIC-facing construction), every
VitalDB patient, rows with a NaN feature in the full column set dropped (every patient's first
stretch, which has no block history yet). The same rows are used for every arm, so the arms differ
only in their columns; the full arm is then exactly the transfer step's model. The record set is the
one every MIMIC number in the deck uses: the first 400 usable MIMIC-III records, those with
stretches after the CVP-waveform gate.
    VARIANT=gated python src/s28_mimic_arms.py
Writes results/mimic_arms.json, results/mimic_arms_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, HGB, NSEED
from sklearn.ensemble import HistGradientBoostingRegressor
from lib.build import VitalDB, load_vitaldb_generic, load_mimic, score, target, boot_ci, feature_names, ACTIVE, active

t0 = time.time()
V = VitalDB(tracker=False); VD = load_vitaldb_generic(V); MI = load_mimic("mimic3wdb"); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
WAVE = [IDX[f"d_{n}"] for n in V.names]; CLK = IDX["since_last_cvp"]
LINE = [i for i in ACTIVE if i not in WAVE and not NAMES[i].startswith("trk_")]
ARMS = {"clock only": [CLK], "waveform only (91 ECG/PPG columns, no clock)": WAVE, "waveform + clock": WAVE + [CLK],
        "read columns + clock (no waveform)": LINE, "full model": None}
keys = sorted(MI); MSEG = [s for k in keys for s in MI[k]]
TA = np.vstack([s["X"] for k in VD for s in VD[k]]); TT = np.concatenate([target(s) for k in VD for s in VD[k]]); TROWS = np.isfinite(TA[:, ACTIVE]).all(1) & np.isfinite(TT)
def fit_rows(cols):
    """The transfer step's fit: rows with any NaN in the FULL column set dropped, for every arm."""
    cc = active(cols); return [HistGradientBoostingRegressor(random_state=sd, **HGB).fit(TA[TROWS][:, cc], TT[TROWS]) for sd in range(NSEED)]
G = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG]); SINCE = np.concatenate([s["since"] for s in MSEG])
Y = {t: np.concatenate([(target(s) > t).astype(int) for s in MSEG]) for t in THRS}
BINS = [("0-5 min", 0.0, 5.0), ("5-10 min", 5.0, 10.0), ("10-15 min", 10.0, 15.0), ("over 15 min", 15.0, np.inf)]  # (lo, hi] on the 1-min since-grid: five minute-values per bin; the read's last reading can precede the nominal read end, so a few windows sit beyond 15 min
def bmask(lo, hi): return (SINCE > lo) & (SINCE <= hi)
print(f"VitalDB (record builder): {len(VD)} patients, {sum(len(v) for v in VD.values())} stretches; MIMIC-III: {len(keys)} records, {len(MSEG)} stretches, {len(G):,} windows [{time.time()-t0:.0f}s]", flush=True)


def summ(P, m=None, nb=400):
    m = np.ones(len(P), bool) if m is None else m
    return {f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t][m], P[m])), ci=list(boot_ci(P[m], Y[t][m], G[m], nb)), prevalence=float(Y[t][m].mean())) for t in THRS}


def per_record(P):
    out = {}
    for t in THRS:
        a = [roc_auc_score(Y[t][G == k], P[G == k]) for k in keys if Y[t][G == k].min() != Y[t][G == k].max()]
        out[f"n>{t:g}"] = dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)), frac_gt_0_5=float(np.mean(np.array(a) > .5)), frac_gt_0_7=float(np.mean(np.array(a) > .7)))
    return out


def binned(P):
    return {nm: dict(windows=int((bmask(lo, hi)).sum()), **summ(P, bmask(lo, hi), 200)) for nm, lo, hi in BINS}


OUT = dict(records=len(keys), stretches=len(MSEG), windows=int(len(G)), vitaldb_training=dict(patients=len(VD), stretches=sum(len(v) for v in VD.values()), rows_all=int(len(TT)), rows_used=int(TROWS.sum()), rule="rows with a NaN in the full column set dropped, the same rows for every arm"),
           bins={nm: dict(lo=lo, hi=(None if np.isinf(hi) else hi), rule="lo < since <= hi", windows=int((bmask(lo, hi)).sum()), prevalence={f"n>{t:g}": float(Y[t][bmask(lo, hi)].mean()) for t in THRS}) for nm, lo, hi in BINS},
           arms={}, baselines={}); SC = {}
for nm, cols in ARMS.items():
    MS = fit_rows(cols)
    raw = score(MS, MSEG, cols=cols, running_max=False); PM = np.concatenate([np.maximum.accumulate(p) for p in raw]); PR = np.concatenate(raw)
    SC[f"{nm}:runmax"] = PM; SC[f"{nm}:raw"] = PR
    OUT["arms"][nm] = dict(ncols=len(cols) if cols else len(ACTIVE), running_max=dict(all=summ(PM), per_record=per_record(PM), bins=binned(PM)), raw=dict(all=summ(PR), per_record=per_record(PR), bins=binned(PR)))
    r = OUT["arms"][nm]
    print(f"  {nm:46s} pooled " + "  ".join(f"{r['running_max']['all'][k]['auc']:.3f}" for k in r["running_max"]["all"]) + "   per record " + "  ".join(f"{r['running_max']['per_record'][k]['mean']:.3f}" for k in r["running_max"]["per_record"]) + "   raw pooled " + "  ".join(f"{r['raw']['all'][k]['auc']:.3f}" for k in r["raw"]["all"]) + f"  [{time.time()-t0:.0f}s]", flush=True)
    for b in r["raw"]["bins"]:
        print(f"      {b:10s} n={r['raw']['bins'][b]['windows']:6d}  running max n>2 {r['running_max']['bins'][b]['n>2']['auc']:.3f}  raw n>2 {r['raw']['bins'][b]['n>2']['auc']:.3f}  n>3 {r['raw']['bins'][b]['n>3']['auc']:.3f}  n>4 {r['raw']['bins'][b]['n>4']['auc']:.3f}", flush=True)
P = np.concatenate([np.nan_to_num(s["X"][:, CLK]) for s in MSEG]); SC["no model: clock"] = P
OUT["baselines"]["no model: clock"] = dict(all=summ(P, nb=200), per_record=per_record(P))
print(f"  {'no model: clock':46s} pooled " + "  ".join(f"{v['auc']:.3f}" for v in OUT["baselines"]["no model: clock"]["all"].values()), flush=True)
np.savez_compressed(f"{RES}/mimic_arms_scores.npz", G=G, SINCE=SINCE, **{f"Y{t:g}": Y[t] for t in THRS}, **SC)
json.dump(OUT, open(f"{RES}/mimic_arms.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/mimic_arms.json")
