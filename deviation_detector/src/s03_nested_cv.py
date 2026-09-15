"""Step 3. The VitalDB result: NESTED cross-validation over every patient, with controls.

Five outer folds by patient (seed FOLD_SEED). Inside each training fold the calibration length
W is chosen on three inner folds only, then the detector is refitted on the whole training fold
and scored on the held-out fold. Every patient is scored exactly once. Prevalence, AUC and a
patient-clustered bootstrap interval are reported at |drift| > 2, 3 and 4 mmHg.

    WLIST=10 SDUR=15                   the final system (default)
    CTRL=clock                         the same fit with time-since-calibration as the ONLY input
    CTRL=perm                          stretch targets permuted among stretches (chance)
    SEED=7                             a different outer-fold assignment
Writes results/nested_<ctrl>_W<..>_S<..>_seed<..>.json and, for the final configuration, the
out-of-fold per-window scores results/vitaldb_oof.npz used by every later VitalDB figure.
"""
import os, sys, time, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED
from lib.build import VitalDB, build_vitaldb, fit, score, target, boot_ci

WS = [float(x) for x in os.environ.get("WLIST", "10").split(",")]
SDUR = float(os.environ.get("SDUR", "15"))
CTRL = os.environ.get("CTRL", "full")
SEED = int(os.environ.get("SEED", str(FOLD_SEED)))
COLS = [91] if CTRL == "clock" else None                 # column 91 = time since last CVP reading

t0 = time.time()
V = VitalDB()
DATA = {w: build_vitaldb(V, w, SDUR) for w in WS}
for w in WS:
    print(f"  W={w:.0f} S={SDUR:.0f}: {len(DATA[w])} patients, {sum(len(v) for v in DATA[w].values())} "
          f"stretches, {sum(len(s['dev']) for v in DATA[w].values() for s in v):,} windows "
          f"[{time.time()-t0:.0f}s]", flush=True)
allp = sorted(set().union(*[set(DATA[w]) for w in WS]))
rng = np.random.default_rng(SEED); ids = np.array(allp); rng.shuffle(ids)
OUTER = np.array_split(ids, 5)


def perm_target(R, trp):
    """CTRL=perm: reassign the stretch targets among the training stretches."""
    segs = [s for c in trp if c in R for s in R[c]]
    ev = [target(s) for s in segs]
    ev = [ev[i] for i in np.random.default_rng(99).permutation(len(ev))]
    return {id(s): e for s, e in zip(segs, ev)}


_CACHE = {}
def fitpred(R, w, trp, tep, nseed):
    """Fit on trp (cached per (W, training set, nseed)), score tep. Returns per-stretch scores."""
    key = (w, tuple(trp), nseed)
    if key not in _CACHE:
        if CTRL == "perm":
            PT = perm_target(R, trp)
            MS = fit(R, trp, nseed, cols=COLS, target_fn=lambda s: PT[id(s)])
        else:
            MS = fit(R, trp, nseed, cols=COLS)
        _CACHE[key] = MS
    segs = [s for c in tep if c in R for s in R[c]]
    return segs, score(_CACHE[key], segs, cols=COLS)


out = {}; OOF = {}; SAVE = {}
print(f"\n{'threshold':>12s} {'NESTED-CV AUC':>26s} {'W chosen per fold':>26s}")
print("-" * 70)
for thr in THRS:
    P, Y, G, picks = [], [], [], []
    for f, te in enumerate(OUTER):
        tr = [c for c in allp if c not in set(te.tolist())]
        inner = np.array_split(np.array(tr), 3)
        best, bw = None, WS[0]
        if len(WS) > 1:
            for w in WS:
                sc = []
                for j in range(3):
                    itr = [c for c in tr if c not in set(inner[j].tolist())]
                    segs, ps = fitpred(DATA[w], w, itr, inner[j].tolist(), 1)
                    yy = np.concatenate([(target(s) > thr).astype(int) for s in segs])
                    if len(np.unique(yy)) > 1: sc.append(roc_auc_score(yy, np.concatenate(ps)))
                if sc and (best is None or np.mean(sc) > best): best, bw = np.mean(sc), w
        picks.append(bw)
        segs, ps = fitpred(DATA[bw], bw, tr, te.tolist(), NSEED)
        for s, p in zip(segs, ps):
            P.append(p); Y.append((target(s) > thr).astype(int)); G.append(np.full(len(p), s["key"]))
            OOF[(s["key"], s["k"])] = p
    P, Y, G = np.concatenate(P), np.concatenate(Y), np.concatenate(G)
    SAVE[f"Y{thr:g}"] = Y; SAVE["P"] = P; SAVE["G"] = G
    au = roc_auc_score(Y, P); lo, hi = boot_ci(P, Y, G, n_boot=600)
    fl = "OK" if lo >= 0.75 else ("+" if au >= 0.75 else "")
    print(f"{'n>' + f'{thr:g}':>12s}   {au:.3f} [{lo:.3f},{hi:.3f}] {fl:>3s}   "
          f"{str([int(w) for w in picks]):>24s}   n={len(Y):,} prev {100*Y.mean():.1f}%", flush=True)
    out[f"n>{thr:g}"] = dict(auc=au, ci=[lo, hi], picks=[int(w) for w in picks], n=int(len(Y)),
                           prevalence=float(Y.mean()))

tag = f"{CTRL}_W{'-'.join(f'{w:g}' for w in WS)}_S{SDUR:g}_seed{SEED}"
meta = dict(ctrl=CTRL, W=WS, S=SDUR, seed=SEED, patients=len(allp),
            stretches=sum(len(v) for v in DATA[WS[0]].values()),
            windows=int(sum(len(s['dev']) for v in DATA[WS[0]].values() for s in v)))
json.dump(dict(meta=meta, results=out), open(f"{RES}/nested_{tag}.json", "w"), indent=1)
np.savez_compressed(f"{RES}/nested_{tag}_scores.npz", **SAVE)
if CTRL == "full" and WS == [10.0] and SDUR == 15.0 and SEED == FOLD_SEED:
    keys = sorted(OOF)
    np.savez_compressed(f"{RES}/vitaldb_oof.npz", cid=np.array([k[0] for k in keys]),
                        k=np.array([k[1] for k in keys]),
                        p=np.array([OOF[k] for k in keys], dtype=object))
    print("saved results/vitaldb_oof.npz")
print(f"[{time.time()-t0:.0f}s] -> results/nested_{tag}.json")
