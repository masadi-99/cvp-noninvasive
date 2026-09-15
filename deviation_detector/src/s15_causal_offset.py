"""Step 15. The VitalDB label with a strictly causal alignment scalar.

The reported VitalDB label aligns the dense CVP grid to the gated a-wave label with one constant
per patient computed over the whole case. That constant shifts baseline and label together and
never enters the score, but it is not strictly causal for the label. Here the constant is
recomputed per block from the gated windows up to the end of that block only, and the same
five-fold cross-validation (same fold seed) is run on the resulting stretches.
Writes results/causal_offset.json
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED
from lib.build import VitalDB, build_vitaldb, fit, score, target, boot_ci

t0 = time.time(); V = VitalDB(); out = {}
for name, kw in (("reported (whole-case offset)", {}), ("causal offset (windows up to the block)", dict(causal_offset=True))):
    R = build_vitaldb(V, **kw); allp = sorted(R)
    rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
    P, Y, G = [], {t: [] for t in THRS}, []
    for te in FOLDS:
        tr = [c for c in allp if c not in set(te.tolist())]
        segs = [s for c in te.tolist() for s in R[c]]
        for s, p in zip(segs, score(fit(R, tr, NSEED), segs)):
            P.append(p); G.append(np.full(len(p), s["key"]))
            for t in THRS: Y[t].append((target(s) > t).astype(int))
    P, G = np.concatenate(P), np.concatenate(G)
    d = dict(patients=len(allp), stretches=sum(len(v) for v in R.values()), windows=int(len(P)))
    for t in THRS:
        y = np.concatenate(Y[t]); d[f"n>{t:g}"] = dict(auc=float(roc_auc_score(y, P)), ci=list(boot_ci(P, y, G, 600)), prevalence=float(y.mean()))
    out[name] = d; tag = "reported" if "reported" in name else "causal"
    np.savez_compressed(f"{RES}/causal_offset_{tag}_scores.npz", P=P, G=G, **{f"Y{t:g}": np.concatenate(Y[t]) for t in THRS})
    print(f"{name:42s} {d['patients']} pts / {d['stretches']} stretches / {d['windows']:,} windows   " +
          "  ".join(f"n>{t:g} {d[f'n>{t:g}']['auc']:.3f} [{d[f'n>{t:g}']['ci'][0]:.3f},{d[f'n>{t:g}']['ci'][1]:.3f}]" for t in THRS), flush=True)
json.dump(out, open(f"{RES}/causal_offset.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/causal_offset.json")
