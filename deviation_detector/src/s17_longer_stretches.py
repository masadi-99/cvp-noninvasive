"""Step 17. The same elimination with longer tracking stretches after the 10-minute read.

SDUR minutes of tracking (15 = the reported schedule, 20, 30, 45, 60). Everything else as in
step 16: the 118 shared columns, refit on the same patient folds at each step, out-of-fold
permutation importance decides the next drop, scored out of fold on VitalDB and zero-shot on
MIMIC-III laid out on the same schedule. Coarser steps: 118, 64, 32, 16, 12, 8, 6, 4, 2, 1.
    SDUR=30 python src/s17_longer_stretches.py
Writes results/longer_S<SDUR>.json and results/longer_S<SDUR>_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED, EXTRA_NAMES, W_CAL
from lib.build import VitalDB, build_vitaldb, load_vitaldb_generic, load_mimic, fit, score, target, boot_ci, feature_names, ACTIVE

SDUR = float(os.environ.get("SDUR", "15")); STAGES = [118, 64, 32, 16, 12, 8, 6, 4, 2, 1]
t0 = time.time()
V = VitalDB(); R = build_vitaldb(V, W_CAL, SDUR, tracker=False); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
SEGS = [s for c in allp for s in R[c]]
Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS])
VD = load_vitaldb_generic(VitalDB(tracker=False), W_CAL, SDUR); MI = load_mimic("mimic3wdb", W_CAL, SDUR); MSEG = [s for k in sorted(MI) for s in MI[k]]
TM = np.concatenate([target(s) for s in MSEG]); YM = {t: (TM > t).astype(int) for t in THRS}
GM = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG]); MBIG = np.vstack([s["X"] for s in MSEG]); MLEN = [len(s["dev"]) for s in MSEG]
TRK = [IDX[n] for n in EXTRA_NAMES[8:]]
meta = dict(W=W_CAL, S=SDUR, zero_shot_training="all VitalDB rows (NaN native); mimic_dropnan = step-7 convention", vitaldb_patients=len(allp), vitaldb_stretches=len(SEGS), vitaldb_windows=int(len(G)),
            vitaldb_prevalence={f"n>{t:g}": float(Y[t].mean()) for t in THRS}, mimic_records=len(MI), mimic_stretches=len(MSEG),
            mimic_windows=int(len(GM)), mimic_prevalence={f"n>{t:g}": float(YM[t].mean()) for t in THRS})
print(f"S={SDUR:g}: VitalDB {len(allp)} pts / {len(SEGS)} stretches / {len(G):,} windows (prev n>2 {100*Y[2.0].mean():.1f}%); "
      f"MIMIC {len(MI)} recs / {len(MSEG)} stretches / {len(GM):,} windows (prev {100*YM[2.0].mean():.1f}%) [{time.time()-t0:.0f}s]", flush=True)


def oof(cols):
    models, P = {}, {}
    for fi, te in enumerate(FOLDS):
        tr = [c for c in allp if c not in set(te.tolist())]; segs = [s for c in te.tolist() for s in R[c]]
        MS = fit(R, tr, NSEED, cols=cols); models[fi] = (MS, segs)
        for s, p in zip(segs, score(MS, segs, cols=cols)): P[(s["key"], s["k"])] = p
    return models, np.concatenate([P[(s["key"], s["k"])] for s in SEGS])


def perm_drop(models, cols, base):
    out = {}; rp = np.random.default_rng(0)
    for j, c in enumerate(cols):
        P = {}
        for fi, (MS, segs) in models.items():
            BIG = np.vstack([s["X"] for s in segs])[:, cols].copy(); BIG[:, j] = BIG[rp.permutation(len(BIG)), j]
            pr = np.mean([m.predict(BIG) for m in MS], 0); o = 0
            for s in segs:
                n = len(s["dev"]); P[(s["key"], s["k"])] = np.maximum.accumulate(pr[o:o + n]); o += n
        out[c] = float(base - roc_auc_score(Y[2.0], np.concatenate([P[(s["key"], s["k"])] for s in SEGS])))
    return out


def mimic_zs(cols, drop_nan_rows):
    """Zero-shot scores. Step 7 dropped every training row with a NaN feature, which removes each
    patient's FIRST stretch (no block-to-block history yet); with long stretches that leaves very
    few training stretches, so this experiment trains on all rows (NaN is native to the trees) and
    also reports the step-7 convention for comparison."""
    MS = fit(VD, list(VD), drop_nan_rows=drop_nan_rows, cols=cols); pr = np.mean([m.predict(MBIG[:, cols]) for m in MS], 0); out, o = [], 0
    for n in MLEN: out.append(np.maximum.accumulate(pr[o:o + n])); o += n
    return np.concatenate(out)


SC = {}; OUT = dict(meta=meta, stages=[])
cols = [i for i in range(len(NAMES)) if i not in TRK and i in set(ACTIVE)]
STAGES[0] = len(cols)
for k, n in enumerate(STAGES):
    if n < len(cols):
        cols = sorted(sorted(cols, key=lambda c: -imp[c])[:n])
    models, P = oof(cols); PM = mimic_zs(cols, False); PMd = mimic_zs(cols, True); tag = f"rfe{len(cols)}"
    SC[f"vd__{tag}"] = P; SC[f"mi__{tag}"] = PM; SC[f"mid__{tag}"] = PMd
    d = dict(n=len(cols), columns=[NAMES[c] for c in cols],
             vitaldb={f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], P)), ci=list(boot_ci(P, Y[t], G, 200))) for t in THRS},
             mimic={f"n>{t:g}": dict(auc=float(roc_auc_score(YM[t], PM)), ci=list(boot_ci(PM, YM[t], GM, 100))) for t in THRS},
             mimic_dropnan={f"n>{t:g}": dict(auc=float(roc_auc_score(YM[t], PMd))) for t in THRS})
    aus = [roc_auc_score(YM[2.0][GM == r], PM[GM == r]) for r in np.unique(GM) if YM[2.0][GM == r].min() != YM[2.0][GM == r].max()]
    d["mimic_per_record_n2"] = float(np.mean(aus)); d["mimic_records_evaluable"] = len(aus)
    if n > 1: imp = perm_drop(models, cols, roc_auc_score(Y[2.0], P)); d["importance"] = {NAMES[c]: imp[c] for c in cols}
    OUT["stages"].append(d)
    print(f"  S={SDUR:g} {tag:8s} VitalDB " + " ".join(f"{d['vitaldb'][kk]['auc']:.3f}" for kk in d["vitaldb"]) + "   MIMIC(all rows) " +
          " ".join(f"{d['mimic'][kk]['auc']:.3f}" for kk in d["mimic"]) + "   MIMIC(step-7 rows) " + " ".join(f"{d['mimic_dropnan'][kk]['auc']:.3f}" for kk in d["mimic_dropnan"]) +
          f"  per-rec {d['mimic_per_record_n2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
np.savez_compressed(f"{RES}/longer_S{SDUR:g}_scores.npz", Y2=Y[2.0], Y3=Y[3.0], Y4=Y[4.0], G=G, YM2=YM[2.0], YM3=YM[3.0], YM4=YM[4.0], GM=GM, **SC)
json.dump(OUT, open(f"{RES}/longer_S{SDUR:g}.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] -> results/longer_S{SDUR:g}.json")
