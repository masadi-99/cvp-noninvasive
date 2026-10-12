"""Step 27. Waveform only: the 91 ECG/PPG columns with no clock and no read columns, on the reported
10 + 15 min schedule, out of fold on VitalDB; and every arm scored inside bins of time since the read.

The system's score is a running maximum over the stretch, and the waveform columns are differenced
against the read, so both can carry elapsed time indirectly. Each arm is therefore reported twice:
with the running maximum (the deployed score) and as the raw per-window prediction; and the AUC is
also computed inside 0-5, 5-10 and 10-15 min since the read, where the clock has little leverage.
Arms: waveform only; waveform + clock; clock only; read columns + clock (no waveform); full model
(the main deck's 123 columns, same folds, so its all-window number reproduces the main deck).
    VARIANT=gated python src/s27_waveform_only.py
Writes results/waveform_only.json, results/waveform_only_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED
from lib.build import VitalDB, build_vitaldb, fit, score, target, boot_ci, feature_names, ACTIVE

t0 = time.time()
V = VitalDB(); R = build_vitaldb(V); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
WAVE = [IDX[f"d_{n}"] for n in V.names]; CLK = IDX["since_last_cvp"]
LINE = [i for i in ACTIVE if i not in WAVE and not NAMES[i].startswith("trk_")]
ARMS = {"waveform only (91 ECG/PPG columns, no clock)": WAVE, "waveform + clock": WAVE + [CLK], "clock only": [CLK],
        "read columns + clock (no waveform)": LINE, "full model": list(ACTIVE)}
allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
SEGS = [s for c in allp for s in R[c]]; G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS]); SINCE = np.concatenate([s["since"] for s in SEGS])
Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
BINS = [("0-5 min", 0.0, 5.0), ("5-10 min", 5.0, 10.0), ("10-15 min", 10.0, 15.0), ("over 15 min", 15.0, np.inf)]  # (lo, hi] on the 1-min since-grid: five minute-values per bin; the read's last reading can precede the nominal read end, so a few windows sit beyond 15 min
def bmask(lo, hi): return (SINCE > lo) & (SINCE <= hi)
print(f"VitalDB: {len(allp)} patients, {len(SEGS)} stretches, {len(G):,} windows; since {SINCE.min():.1f}-{SINCE.max():.1f} min [{time.time()-t0:.0f}s]", flush=True)


def oof(cols):
    """Out-of-fold scores in SEGS order: (running maximum, raw per-window prediction)."""
    PR = {}
    for te in FOLDS:
        tr = [c for c in allp if c not in set(te.tolist())]; segs = [s for c in te.tolist() for s in R[c]]
        MS = fit(R, tr, NSEED, cols=cols)
        for s, p in zip(segs, score(MS, segs, cols=cols, running_max=False)): PR[(s["key"], s["k"])] = p
    raw = [PR[(s["key"], s["k"])] for s in SEGS]
    return np.concatenate([np.maximum.accumulate(p) for p in raw]), np.concatenate(raw)


def summ(P, m=None, nb=400):
    m = np.ones(len(P), bool) if m is None else m
    return {f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t][m], P[m])), ci=list(boot_ci(P[m], Y[t][m], G[m], nb)), prevalence=float(Y[t][m].mean())) for t in THRS}


def binned(P):
    return {nm: dict(windows=int((bmask(lo, hi)).sum()), **summ(P, bmask(lo, hi), 200)) for nm, lo, hi in BINS}


OUT = dict(patients=len(allp), stretches=len(SEGS), windows=int(len(G)), bins={nm: dict(lo=lo, hi=(None if np.isinf(hi) else hi), rule="lo < since <= hi", windows=int((bmask(lo, hi)).sum()),
           prevalence={f"n>{t:g}": float(Y[t][bmask(lo, hi)].mean()) for t in THRS}) for nm, lo, hi in BINS}, arms={}); SC = {}
for nm, cols in ARMS.items():
    PM, PR = oof(cols); SC[f"{nm}:runmax"] = PM; SC[f"{nm}:raw"] = PR
    OUT["arms"][nm] = dict(ncols=len(cols), running_max=dict(all=summ(PM), bins=binned(PM)), raw=dict(all=summ(PR), bins=binned(PR)))
    r = OUT["arms"][nm]
    print(f"  {nm:46s} running max " + "  ".join(f"{r['running_max']['all'][k]['auc']:.3f}" for k in r["running_max"]["all"]) + "   raw " + "  ".join(f"{r['raw']['all'][k]['auc']:.3f}" for k in r["raw"]["all"]) + f"  [{time.time()-t0:.0f}s]", flush=True)
    for b in r["raw"]["bins"]:
        print(f"      {b:10s} n={r['raw']['bins'][b]['windows']:5d}  running max n>2 {r['running_max']['bins'][b]['n>2']['auc']:.3f}  raw n>2 {r['raw']['bins'][b]['n>2']['auc']:.3f}  n>3 {r['raw']['bins'][b]['n>3']['auc']:.3f}  n>4 {r['raw']['bins'][b]['n>4']['auc']:.3f}", flush=True)
np.savez_compressed(f"{RES}/waveform_only_scores.npz", G=G, SINCE=SINCE, **{f"Y{t:g}": Y[t] for t in THRS}, **SC)
json.dump(OUT, open(f"{RES}/waveform_only.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/waveform_only.json")
