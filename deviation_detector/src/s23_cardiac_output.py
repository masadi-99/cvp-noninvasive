"""Step 23. The same detector on a different label: cardiac output, cardiac index, stroke volume.

VitalDB carries the EV1000 / Vigileo numerics on the 30-s tile grid. For every gated window the
label becomes the tile's CO (L/min), CI (L/min/m2) or SV (mL); the schedule, the artefact filter
(step in the label's own units), the 91 waveform features differenced against the read, the read's
own shape columns and the model are unchanged. Five-fold cross-validation by patient with the same
fold seed; the same arms as for CVP (full, read columns only, waveform + clock, clock only,
permuted targets) and the no-model read baselines. CVP is rerun on the cardiac-output patients (those
that also have a CVP stretch; the folds are drawn with the same seed over that patient list) so the two
labels can be compared on the same people: its stretches come from build_vitaldb,
the main system's construction (baseline and read shape from the full 30-s CVP series), and the main
system's own out-of-fold scores (step 3) are also reported restricted to these patients.
    VARIANT=gated python src/s23_cardiac_output.py          # all four targets
    VARIANT=gated python src/s23_cardiac_output.py cvp      # only this target; the others are kept from the files
Writes results/cardiac_output.json, results/cardiac_output_scores.npz, results/cardiac_output_traces.npz
"""
import os, sys, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, FOLD_SEED, NSEED, VITALDB_TILES
from lib.build import VitalDB, build_record, build_vitaldb, fit, score, target, boot_ci, feature_names, ACTIVE

TARGETS = {"co": dict(name="cardiac output", unit="L/min", thr=(0.5, 0.75, 1.0), max_step=2.0),
           "ci": dict(name="cardiac index", unit="L/min/m2", thr=(0.25, 0.4, 0.5), max_step=1.0),
           "sv": dict(name="stroke volume", unit="mL", thr=(10.0, 15.0, 20.0), max_step=25.0),
           "cvp": dict(name="CVP (same patients)", unit="mmHg", thr=THRS, max_step=3.0)}
t0 = time.time(); V = VitalDB(tracker=False); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
T = np.load(VITALDB_TILES, allow_pickle=True); Kt = {str(c): i for i, c in enumerate(T["cols"])}; M = T["data"]
cid = M[:, Kt["cid"]].astype(int); tmn = M[:, Kt["t_sec"]] / 60.0
LAB = {}
for c, o in V.ORDER.items():
    m = cid == c; key = {round(float(a), 4): i for i, a in enumerate(tmn[m])}; idx = np.array([key.get(round(float(a), 4), -1) for a in V.T[o]]); g = idx >= 0
    LAB[c] = {}
    for k in ("co", "ci", "sv"):
        v = np.full(len(o), np.nan); v[g] = M[m][idx[g], Kt[k]]; LAB[c][k] = v
    LAB[c]["cvp"] = V.Y[o]
WAVE = [IDX[f"d_{n}"] for n in V.names]; CLK = IDX["since_last_cvp"]; LINE = [i for i in ACTIVE if i not in WAVE and not NAMES[i].startswith("trk_")]
ARMS = {"full": [i for i in ACTIVE if not NAMES[i].startswith("trk_")], "read columns only (no waveform)": LINE, "waveform + clock": WAVE + [CLK], "clock only": [CLK]}
OUT = {}; SC = {}; TR = {}
ONLY = sys.argv[1:]
if ONLY:  # keep the other targets' results from the files
    OUT = json.load(open(f"{RES}/cardiac_output.json")); _S = np.load(f"{RES}/cardiac_output_scores.npz", allow_pickle=True); _T = np.load(f"{RES}/cardiac_output_traces.npz", allow_pickle=True)
    SC = {k: _S[k] for k in _S.files if k.split(":")[0] not in ONLY}
    for kq in _T.files:
        kk, q = kq.rsplit("__", 1)
        if kk.split(":")[0] not in ONLY: TR.setdefault(kk, {})[q] = float(_T[kq][0]) if q == "base" else _T[kq]
    for k in ONLY: OUT.pop(k, None)


def run(k, patients=None):
    cfg = TARGETS[k]; R = {}
    if k == "cvp":  # the main system's construction (step 3), restricted to the requested patients
        R = {c: ss for c, ss in build_vitaldb(V, tracker=False).items() if patients is None or c in patients}
    else:
        for c in sorted(V.ORDER):
            if patients is not None and c not in patients: continue
            o = V.ORDER[c]; y = LAB[c][k]
            if np.isfinite(y).sum() < 12: continue
            ss = build_record(V.T[o], y, V.X[o], c, max_step=cfg["max_step"])
            if ss: R[c] = ss
    allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
    SEGS = [s for c in allp for s in R[c]]; G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS])
    Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in cfg["thr"]}
    d = dict(label=cfg["name"], unit=cfg["unit"], thresholds=list(cfg["thr"]), max_step=cfg["max_step"], patients=len(allp), stretches=len(SEGS), windows=int(len(G)),
             prevalence={f"n>{t:g}": float(Y[t].mean()) for t in cfg["thr"]}, level_median=float(np.median([s["base"] for s in SEGS])), arms={}, baselines={}, per_patient={})
    print(f"\n{cfg['name']}: {len(allp)} patients, {len(SEGS)} stretches, {len(G):,} windows; prevalence " + " / ".join(f"{100*Y[t].mean():.1f}%" for t in cfg["thr"]) + f"  [{time.time()-t0:.0f}s]", flush=True)

    def oof(cols, perm=False):
        P = {}
        for te in FOLDS:
            tr = [c for c in allp if c not in set(te.tolist())]; segs = [s for c in te.tolist() for s in R[c]]
            if perm:
                trs = [s for c in tr for s in R[c]]; ev = [target(s) for s in trs]; ev = [ev[i] for i in np.random.default_rng(99).permutation(len(ev))]; PT = {id(s): e for s, e in zip(trs, ev)}
                MS = fit(R, tr, NSEED, cols=cols, target_fn=lambda s: PT[id(s)])
            else: MS = fit(R, tr, NSEED, cols=cols)
            for s, p in zip(segs, score(MS, segs, cols=cols)): P[(s["key"], s["k"])] = p
        return np.concatenate([P[(s["key"], s["k"])] for s in SEGS])
    def summ(P, nb=400): return {f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], P)), ci=list(boot_ci(P, Y[t], G, nb))) for t in cfg["thr"]}
    for nm, cols in ARMS.items():
        P = oof(cols); SC[f"{k}:{nm}"] = P; d["arms"][nm] = summ(P)
        print(f"  {nm:34s} " + "  ".join(f"{v['auc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]" for v in d["arms"][nm].values()), flush=True)
    P = oof(ARMS["full"], perm=True); SC[f"{k}:permuted targets"] = P; d["arms"]["permuted targets"] = summ(P)
    print(f"  {'permuted targets':34s} " + "  ".join(f"{v['auc']:.3f}" for v in d["arms"]["permuted targets"].values()), flush=True)
    for nm, col, ab in (("read spread x since", "std_x_since", False), ("read |last - median|", "blk_last_minus_median", True), ("read spread", "blk_std", False), ("clock", "since_last_cvp", False)):
        P = np.concatenate([np.maximum.accumulate(np.abs(s["X"][:, IDX[col]]) if ab else np.nan_to_num(s["X"][:, IDX[col]])) for s in SEGS]); SC[f"{k}:base:{nm}"] = P; d["baselines"][nm] = summ(P, 200)
        print(f"  {'no model: ' + nm:34s} " + "  ".join(f"{v['auc']:.3f}" for v in d["baselines"][nm].values()), flush=True)
    Pf = SC[f"{k}:full"]
    for t in cfg["thr"]:
        a = [roc_auc_score(Y[t][G == c], Pf[G == c]) for c in allp if Y[t][G == c].min() != Y[t][G == c].max()]
        d["per_patient"][f"n>{t:g}"] = dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)))
    # traces for the example figures (full model, out of fold)
    o = 0
    for s in SEGS:
        n = len(s["dev"]); TR[f"{k}:{s['key']}:{s['k']}"] = dict(t=s["t"], y=s["y"], tb=s["tb"], yb=s["yb"], base=s["base"], dev=s["dev"], p=Pf[o:o + n]); o += n
    SC[f"{k}:G"] = G
    for t in cfg["thr"]: SC[f"{k}:Y{t:g}"] = Y[t]
    if k == "cvp":  # the main system's own out-of-fold scores (step 3, 344 patients), restricted to these patients
        z = np.load(f"{RES}/nested_full_W10_S15_seed2024_scores.npz", allow_pickle=True); m = np.isin(z["G"].astype(int), allp)
        d["main_system_on_these_patients"] = dict(patients=int(len(set(z["G"][m].astype(int).tolist()))), windows=int(m.sum()), **{f"n>{t:g}": float(roc_auc_score(z[f"Y{t:g}"][m], z["P"][m])) for t in cfg["thr"]})
        print("  main system (step 3) on these patients: " + "  ".join(f"{d['main_system_on_these_patients'][f'n>{t:g}']:.3f}" for t in cfg["thr"]) + f"  ({d['main_system_on_these_patients']['patients']} patients)", flush=True)
    OUT[k] = d; return set(allp)


TODO = ONLY or ["co", "ci", "sv", "cvp"]
co_pats = run("co") if "co" in TODO else {int(kk.split(":")[1]) for kk in TR if kk.startswith("co:")}
for k in ("ci", "sv"):
    if k in TODO: run(k)
if "cvp" in TODO: run("cvp", patients=co_pats)
OUT = {k: OUT[k] for k in ("co", "ci", "sv", "cvp") if k in OUT}
np.savez_compressed(f"{RES}/cardiac_output_scores.npz", **SC)
np.savez_compressed(f"{RES}/cardiac_output_traces.npz", **{f"{kk}__{q}": (np.asarray(v) if q != "base" else np.array([v])) for kk, dd in TR.items() for q, v in dd.items()})
json.dump(OUT, open(f"{RES}/cardiac_output.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/cardiac_output.json")
