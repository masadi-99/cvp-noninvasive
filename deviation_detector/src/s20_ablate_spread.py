"""Step 20. Remove the block-spread x elapsed-time column (and its relatives) and refit.

Arms, each refitted on the same five VitalDB folds (out of fold) and on all of VitalDB for the
MIMIC-III zero-shot test, with paired patient / record bootstrap intervals against the full model:
  full (123)
  - std_x_since                                  the single column
  - the four elapsed-time interactions           slope x since, |slope| x since, std x since, rate x since
  - every spread column                          std_x_since, blk_std, blk_std_gated, blk_ptp, blk_mean_abs_step
  - every spread column and every interaction
Writes results/ablate_spread.json and results/ablate_spread_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED
from lib.build import VitalDB, build_vitaldb, load_vitaldb_generic, load_mimic, fit, score, target, boot_ci, feature_names

t0 = time.time(); V = VitalDB(); R = build_vitaldb(V); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
SEGS = [s for c in allp for s in R[c]]; Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS])
VD = load_vitaldb_generic(VitalDB(tracker=False)); MI = load_mimic("mimic3wdb"); MSEG = [s for k in sorted(MI) for s in MI[k]]
YM = {t: np.concatenate([(target(s) > t).astype(int) for s in MSEG]) for t in THRS}; GM = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG])
INTER = ["slope_x_since", "abs_slope_x_since", "std_x_since", "rate_x_since"]; SPREAD = ["std_x_since", "blk_std", "blk_std_gated", "blk_ptp", "blk_mean_abs_step"]
ARMS = {"full (123)": [], "- std_x_since": ["std_x_since"], "- the four elapsed-time interactions": INTER, "- every spread column": SPREAD,
        "- every spread column and every interaction": sorted(set(SPREAD + INTER))}


def cv(cols):
    P = {}
    for te in FOLDS:
        tr = [c for c in allp if c not in set(te.tolist())]; segs = [s for c in te.tolist() for s in R[c]]
        for s, p in zip(segs, score(fit(R, tr, NSEED, cols=cols), segs, cols=cols)): P[(s["key"], s["k"])] = p
    return np.concatenate([P[(s["key"], s["k"])] for s in SEGS])


def paired(P1, P2, y, g, n_boot, seed=7):
    up = np.unique(g); GI = [np.where(g == c)[0] for c in up]; rb = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        pk = rb.integers(0, len(GI), len(GI)); m2 = np.concatenate([GI[i] for i in pk])
        if y[m2].min() != y[m2].max(): d.append(roc_auc_score(y[m2], P1[m2]) - roc_auc_score(y[m2], P2[m2]))
    return dict(delta=float(roc_auc_score(y, P1) - roc_auc_score(y, P2)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


OUT = {}; SC = {}; full_v = full_m = None
for nm, drop in ARMS.items():
    cols = [i for i in range(len(NAMES)) if NAMES[i] not in drop]
    Pv = cv(cols); Pm = np.concatenate(score(fit(VD, list(VD), drop_nan_rows=True, cols=cols), MSEG, cols=cols))
    if full_v is None: full_v, full_m = Pv, Pm
    d = dict(ncols=len(cols), removed=drop,
             vitaldb={f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], Pv)), ci=list(boot_ci(Pv, Y[t], G, 600)), vs_full=paired(Pv, full_v, Y[t], G, 600)) for t in THRS},
             mimic={f"n>{t:g}": dict(auc=float(roc_auc_score(YM[t], Pm)), ci=list(boot_ci(Pm, YM[t], GM, 300)), vs_full=paired(Pm, full_m, YM[t], GM, 300)) for t in THRS})
    OUT[nm] = d; SC[f"vd__{nm}"] = Pv; SC[f"mi__{nm}"] = Pm
    print(f"  {nm:46s} [{len(cols)}]  VitalDB " + "  ".join(f"{d['vitaldb'][k]['auc']:.3f} ({d['vitaldb'][k]['vs_full']['delta']:+.3f} [{d['vitaldb'][k]['vs_full']['ci'][0]:+.3f},{d['vitaldb'][k]['vs_full']['ci'][1]:+.3f}])" for k in d["vitaldb"]) +
          "   MIMIC " + "  ".join(f"{d['mimic'][k]['auc']:.3f} ({d['mimic'][k]['vs_full']['delta']:+.3f})" for k in d["mimic"]) + f"  [{time.time()-t0:.0f}s]", flush=True)
np.savez_compressed(f"{RES}/ablate_spread_scores.npz", Y2=Y[2.0], Y3=Y[3.0], Y4=Y[4.0], G=G, YM2=YM[2.0], YM3=YM[3.0], YM4=YM[4.0], GM=GM, **SC)
json.dump(OUT, open(f"{RES}/ablate_spread.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/ablate_spread.json")
