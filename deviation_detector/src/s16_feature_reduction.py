"""Step 16. How few columns are enough?

Recursive elimination on the 118 columns that exist on both cohorts (the five tracker columns are
excluded so that every reduced set is deployable on MIMIC). At each stage the detector is refitted
on the same five patient folds, scored out of fold on VitalDB and zero-shot on MIMIC-III, the
out-of-fold permutation importance of every remaining column is measured, and the least important
quarter (one column at a time once 16 remain) is dropped. Two hand-picked sets are scored the
same way: five CVP-line columns, and those five plus five waveform columns.
Writes results/feature_reduction.json, results/feature_reduction_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED, EXTRA_NAMES
from lib.build import VitalDB, build_vitaldb, load_vitaldb_generic, load_mimic, fit, score, target, boot_ci, feature_names, ACTIVE

t0 = time.time()
V = VitalDB(); R = build_vitaldb(V); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
SEGS = [s for c in allp for s in R[c]]
Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS])
VD = load_vitaldb_generic(VitalDB(tracker=False)); MI = load_mimic("mimic3wdb"); MSEG = [s for k in sorted(MI) for s in MI[k]]
YM = {t: np.concatenate([(target(s) > t).astype(int) for s in MSEG]) for t in THRS}
GM = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG]); MBIG = np.vstack([s["X"] for s in MSEG]); MLEN = [len(s["dev"]) for s in MSEG]
TRK = [IDX[n] for n in EXTRA_NAMES[8:]]
print(f"VitalDB {len(SEGS)} stretches / {len(G):,} windows; MIMIC {len(MSEG)} stretches / {len(GM):,} windows [{time.time()-t0:.0f}s]", flush=True)


def oof(cols):
    """Fold models and the out-of-fold scores (SEGS order) for a column set."""
    models, P = {}, {}
    for fi, te in enumerate(FOLDS):
        tr = [c for c in allp if c not in set(te.tolist())]
        segs = [s for c in te.tolist() for s in R[c]]
        MS = fit(R, tr, NSEED, cols=cols); models[fi] = (MS, segs)
        for s, p in zip(segs, score(MS, segs, cols=cols)): P[(s["key"], s["k"])] = p
    return models, np.concatenate([P[(s["key"], s["k"])] for s in SEGS])


def perm_drop(models, cols, base):
    """Out-of-fold permutation importance (AUC drop at n>2) of every column in cols."""
    out = {}
    rp = np.random.default_rng(0)
    for j, c in enumerate(cols):
        P = {}
        for fi, (MS, segs) in models.items():
            BIG = np.vstack([s["X"] for s in segs])[:, cols].copy(); BIG[:, j] = BIG[rp.permutation(len(BIG)), j]
            pr = np.mean([m.predict(BIG) for m in MS], 0); o = 0
            for s in segs:
                n = len(s["dev"]); P[(s["key"], s["k"])] = np.maximum.accumulate(pr[o:o + n]); o += n
        Pp = np.concatenate([P[(s["key"], s["k"])] for s in SEGS])
        out[c] = float(base - roc_auc_score(Y[2.0], Pp))
    return out


def mimic_zs(cols):
    MS = fit(VD, list(VD), drop_nan_rows=True, cols=cols)
    pr = np.mean([m.predict(MBIG[:, cols]) for m in MS], 0); out, o = [], 0
    for n in MLEN: out.append(np.maximum.accumulate(pr[o:o + n])); o += n
    return np.concatenate(out)


def evaluate(cols, tag, n_boot=300):
    models, P = oof(cols); PM = mimic_zs(cols)
    d = dict(n=len(cols), columns=[NAMES[c] for c in cols],
             vitaldb={f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], P)), ci=list(boot_ci(P, Y[t], G, n_boot))) for t in THRS},
             mimic={f"n>{t:g}": dict(auc=float(roc_auc_score(YM[t], PM)), ci=list(boot_ci(PM, YM[t], GM, 200))) for t in THRS})
    aus = [roc_auc_score((np.concatenate([target(s) for s in MSEG])[GM == r] > 2).astype(int), PM[GM == r]) for r in np.unique(GM)
           if len(np.unique((np.concatenate([target(s) for s in MSEG])[GM == r] > 2).astype(int))) > 1]
    d["mimic_per_record_n2"] = float(np.mean(aus))
    SC[f"vd__{tag}"] = P; SC[f"mi__{tag}"] = PM
    print(f"  {tag:14s} {len(cols):4d} cols  VitalDB " + " ".join(f"{d['vitaldb'][k]['auc']:.3f}" for k in d["vitaldb"]) +
          "   MIMIC " + " ".join(f"{d['mimic'][k]['auc']:.3f}" for k in d["mimic"]) + f"  per-rec {d['mimic_per_record_n2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    return d, models, P


SC = {}; OUT = dict(stages=[], hand_picked={})
cols = [i for i in range(len(NAMES)) if i not in TRK and i in set(ACTIVE)]
while True:
    d, models, P = evaluate(cols, f"rfe{len(cols)}")
    imp = perm_drop(models, cols, roc_auc_score(Y[2.0], P)); d["importance"] = {NAMES[c]: imp[c] for c in cols}
    OUT["stages"].append(d)
    if len(cols) == 1: break
    keep = max(1, int(round(len(cols) * 0.75))) if len(cols) > 16 else len(cols) - 1
    cols = sorted(sorted(cols, key=lambda c: -imp[c])[:keep])
    print(f"    dropped {[NAMES[c] for c in d and range(0) ]}" if False else f"    -> keeping {keep}: least useful was {min(imp, key=imp.get)} ({NAMES[min(imp, key=imp.get)]}, {imp[min(imp, key=imp.get)]:+.4f})", flush=True)

_5 = [n for n in ["since_last_cvp", "blk_std", "blk_last_minus_median", "blk_slope_2nd_half", "std_x_since"] if IDX[n] in set(ACTIVE)]
if len(_5) < 5: _5.append("blk_std_gated")
HAND = {"5 CVP-line columns": _5,
        "5 CVP-line + 5 waveform": _5 + ["d_r_amp", "d_ppg_morph_f4", "d_p_absamp", "d_ppg_dct4", "d_r_amp_cv"],
        "5 CVP-line + 10 waveform": _5 + ["d_r_amp", "d_ppg_morph_f4", "d_p_absamp", "d_ppg_dct4", "d_r_amp_cv", "d_ppg_dct2", "d_ppg_dia_frac", "d_ppg_notch_rel", "d_p_snr", "d_rr_mean"]}
for nm, names in HAND.items():
    d, _, _ = evaluate([IDX[n] for n in names], "hand" + str(len(names))); OUT["hand_picked"][nm] = d
np.savez_compressed(f"{RES}/feature_reduction_scores.npz", Y2=Y[2.0], Y3=Y[3.0], Y4=Y[4.0], G=G, YM2=YM[2.0], YM3=YM[3.0], YM4=YM[4.0], GM=GM, **SC)
json.dump(OUT, open(f"{RES}/feature_reduction.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] -> results/feature_reduction.json")
