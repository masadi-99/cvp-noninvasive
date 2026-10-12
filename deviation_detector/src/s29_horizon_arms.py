"""Step 29. The arm comparison at longer tracking horizons: S = 15, 20, 30, 45 or 60 min after the
10-min read, with the artefact-step filter scanning (a) the whole stretch, as everywhere else, and
(b) the read period only, so that the filter's reach no longer grows with the horizon.

Arms: clock only, waveform only, waveform + clock, read columns + clock (no waveform), full model
(118 columns, no tracker, as in step 17), permuted targets. VitalDB five-fold by patient with the
usual seed; MIMIC-III zero-shot on the gated records (training on every VitalDB row, as step 17
recommends for long stretches), pooled and per record. For every horizon and filter span: patients /
stretches / windows kept and the prevalence; the counts and prevalence with the filter off are also
recorded, so the filter's cost can be read directly. The MIMIC zero-shot models here are trained on
every VitalDB row (no NaN-row dropping), the step-17 convention for long stretches; at S=15 this
differs slightly from step 28's transfer-step rows. Note that the block-history columns describe the
previously KEPT blocks, so the "read" and "none" spans also change those columns on shared stretches.
    SDUR=30 VARIANT=gated python src/s29_horizon_arms.py
Writes results/horizon_arms_S<SDUR>.json, results/horizon_arms_S<SDUR>_scores.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED, W_CAL
from lib.build import VitalDB, build_vitaldb, load_vitaldb_generic, load_mimic, fit, score, target, boot_ci, feature_names, ACTIVE

SDUR = float(os.environ.get("SDUR", "15")); t0 = time.time()
V = VitalDB(tracker=False); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
WAVE = [IDX[f"d_{n}"] for n in V.names]; CLK = IDX["since_last_cvp"]
LINE = [i for i in ACTIVE if i not in WAVE and not NAMES[i].startswith("trk_")]; FULL = [i for i in ACTIVE if not NAMES[i].startswith("trk_")]
ARMS = {"clock only": [CLK], "waveform only (91 ECG/PPG columns, no clock)": WAVE, "waveform + clock": WAVE + [CLK],
        "read columns + clock (no waveform)": LINE, "full model (118 columns)": FULL}
OUT = dict(W=W_CAL, S=SDUR, spans={}); SC = {}


def counts(SEGS, G):
    Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
    return dict(stretches=len(SEGS), windows=int(len(G)), prevalence={f"n>{t:g}": float(Y[t].mean()) for t in THRS},
                prevalence_instantaneous={f"n>{t:g}": float(np.concatenate([(s["dev"] > t).astype(int) for s in SEGS]).mean()) for t in THRS},
                stretches_with_event={f"n>{t:g}": float(np.mean([(s["dev"] > t).any() for s in SEGS])) for t in THRS})


for span in ("none", "full", "read"):
    R = build_vitaldb(V, W_CAL, SDUR, tracker=False, filter_span=span); allp = sorted(R); SEGS = [s for c in allp for s in R[c]]
    G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS]); Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
    MI = load_mimic("mimic3wdb", W_CAL, SDUR, filter_span=span); mkeys = sorted(MI); MSEG = [s for k in mkeys for s in MI[k]]
    GM = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG]); YM = {t: np.concatenate([(target(s) > t).astype(int) for s in MSEG]) for t in THRS}
    d = dict(filter_span=span, vitaldb=dict(patients=len(allp), **counts(SEGS, G)), mimic=dict(records=len(mkeys), **counts(MSEG, GM)), arms={}, mimic_arms={})
    print(f"S={SDUR:g} filter={span:4s}: VitalDB {len(allp)} pts / {len(SEGS)} stretches / {len(G):,} windows (prev n>2 {100*Y[2.0].mean():.1f}%); "
          f"MIMIC {len(mkeys)} recs / {len(MSEG)} stretches / {len(GM):,} windows (prev {100*YM[2.0].mean():.1f}%) [{time.time()-t0:.0f}s]", flush=True)
    if span == "none": OUT["spans"][span] = d; continue
    rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)

    def oof(cols, perm=False):
        P = {}
        for te in FOLDS:
            tr = [c for c in allp if c not in set(te.tolist())]; segs = [s for c in te.tolist() for s in R[c]]
            if perm:  # stretch targets permuted among the training stretches (the deck's permutation control)
                trs = [s for c in tr for s in R[c]]; ev = [target(s) for s in trs]; ev = [ev[i] for i in np.random.default_rng(99).permutation(len(ev))]; PT = {id(s): e for s, e in zip(trs, ev)}
                MS = fit(R, tr, NSEED, cols=cols, target_fn=lambda s: PT[id(s)])
            else: MS = fit(R, tr, NSEED, cols=cols)
            for s, p in zip(segs, score(MS, segs, cols=cols)): P[(s["key"], s["k"])] = p
        return np.concatenate([P[(s["key"], s["k"])] for s in SEGS])

    def summ(P, Yd, Gd, nb=300): return {f"n>{t:g}": dict(auc=float(roc_auc_score(Yd[t], P)), ci=list(boot_ci(P, Yd[t], Gd, nb))) for t in THRS}

    def per_record(P):
        out = {}
        for t in THRS:
            a = [roc_auc_score(YM[t][GM == k], P[GM == k]) for k in mkeys if YM[t][GM == k].min() != YM[t][GM == k].max()]
            out[f"n>{t:g}"] = dict(n=len(a), mean=float(np.mean(a)) if a else None, median=float(np.median(a)) if a else None)
        return out

    for nm, cols in list(ARMS.items()) + [("permuted targets", FULL)]:
        P = oof(cols, perm=(nm == "permuted targets")); SC[f"{span}:{nm}"] = P; d["arms"][nm] = dict(ncols=len(cols), **summ(P, Y, G))
        print(f"  {span:4s} VitalDB {nm:46s} " + "  ".join(f"{d['arms'][nm][k]['auc']:.3f}" for k in d["arms"][nm] if k.startswith("n>")) + f"  [{time.time()-t0:.0f}s]", flush=True)
    VD = load_vitaldb_generic(V, W_CAL, SDUR, filter_span=span)
    for nm, cols in ARMS.items():
        MS = fit(VD, list(VD), NSEED, cols=cols); P = np.concatenate(score(MS, MSEG, cols=cols)); SC[f"{span}:mimic:{nm}"] = P
        d["mimic_arms"][nm] = dict(ncols=len(cols), **summ(P, YM, GM), per_record=per_record(P))
        print(f"  {span:4s} MIMIC   {nm:46s} " + "  ".join(f"{d['mimic_arms'][nm][k]['auc']:.3f}" for k in d["mimic_arms"][nm] if k.startswith("n>")) + f"   per record n>2 {d['mimic_arms'][nm]['per_record']['n>2']['mean'] or float('nan'):.3f}  [{time.time()-t0:.0f}s]", flush=True)
    SC[f"{span}:G"] = G; SC[f"{span}:mimic:G"] = GM
    for t in THRS: SC[f"{span}:Y{t:g}"] = Y[t]; SC[f"{span}:mimic:Y{t:g}"] = YM[t]
    OUT["spans"][span] = d
np.savez_compressed(f"{RES}/horizon_arms_S{SDUR:g}_scores.npz", **SC)
json.dump(OUT, open(f"{RES}/horizon_arms_S{SDUR:g}.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/horizon_arms_S{SDUR:g}.json")
