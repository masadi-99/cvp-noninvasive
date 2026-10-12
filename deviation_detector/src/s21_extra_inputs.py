"""Step 21. Two questions from the presentation, answered with the same folds and model.

  A. Do demographics (age, sex, height, weight, BMI) or arterial pressure help when appended
     to the 123 columns?  Demographics are per-patient constants from VitalDB clinical_data;
     arterial pressure is the Solar8000 MAP numeric on the 30-s grid, entered as MAP at the
     window, MAP change vs the read's median, and the read's MAP median. MIMIC has neither
     demographics nor MAP outside the art-line subset, so these arms are VitalDB-only except
     for a MAP arm scored on the 70% of MIMIC windows with an arterial line.
  B. What does the calibration read contribute when reduced to its MEDIAN alone (no spread, no
     slope, no history, no interactions)?  Arms: median + clock; median + clock + waveform;
     the full read shape without the waveform; the full model.
Writes results/extra_inputs.json, results/extra_inputs_scores.npz
"""
import os, sys, json, csv, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RES, THRS, FOLD_SEED, NSEED, HGB, DESC_NAMES, HIST_NAMES, EXTRA_NAMES
from lib.build import VitalDB, build_vitaldb, load_vitaldb_generic, load_mimic, fit, score, target, boot_ci, feature_names, ACTIVE

t0 = time.time(); V = VitalDB(); R = build_vitaldb(V); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
allp = sorted(R); rng = np.random.default_rng(FOLD_SEED); ids = np.array(allp); rng.shuffle(ids); FOLDS = np.array_split(ids, 5)
SEGS = [s for c in allp for s in R[c]]; Y = {t: np.concatenate([(target(s) > t).astype(int) for s in SEGS]) for t in THRS}
G = np.concatenate([[s["key"]] * len(s["dev"]) for s in SEGS])

# ---- extra per-window columns on VitalDB ------------------------------------------------------------
CL = {int(r["caseid"]): r for r in csv.DictReader(open(f"{DATA}/vitaldb_clinical.csv"))}
def num(x):
    try: return float(x.replace(">", "").replace("<", ""))
    except Exception: return np.nan
DEMO = {}
for s in SEGS:
    r = CL[s["key"]]; n = len(s["dev"])
    DEMO.setdefault("age", []).append(np.full(n, num(r["age"]))); DEMO.setdefault("sex", []).append(np.full(n, 1.0 if r["sex"] == "M" else 0.0))
    DEMO.setdefault("height", []).append(np.full(n, num(r["height"]))); DEMO.setdefault("weight", []).append(np.full(n, num(r["weight"]))); DEMO.setdefault("bmi", []).append(np.full(n, num(r["bmi"])))
DEMO = {k: np.concatenate(v) for k, v in DEMO.items()}
def map_cols():
    out = {"map_win": [], "map_delta": [], "map_read": []}
    for s in SEGS:
        d = V.NUM[s["key"]]; u = d["t"]; v = d["map"]
        blk = (u >= s["tb"][0]) & (u <= s["tb"][-1]) & np.isfinite(v); ref = np.median(v[blk]) if blk.sum() >= 3 else np.nan
        key = {round(float(a), 4): i for i, a in enumerate(u)}; ii = np.array([key.get(round(float(a), 4), -1) for a in s["t"]])
        val = np.where(ii >= 0, v[np.clip(ii, 0, None)], np.nan)
        out["map_win"].append(val); out["map_delta"].append(val - ref); out["map_read"].append(np.full(len(s["dev"]), ref))
    return {k: np.concatenate(v) for k, v in out.items()}
MAPC = map_cols()
print(f"VitalDB {len(SEGS)} stretches / {len(G):,} windows; MAP present on {100*np.isfinite(MAPC['map_win']).mean():.0f}% of windows [{time.time()-t0:.0f}s]", flush=True)

BASE = np.vstack([s["X"] for s in SEGS])[:, ACTIVE]
SEGIDX = np.concatenate([[i] * len(s["dev"]) for i, s in enumerate(SEGS)])
def cv(Xfull, cols=None):
    """Five-fold OOF running-max scores on an arbitrary per-window matrix (rows in SEGS order)."""
    Xc = Xfull if cols is None else Xfull[:, cols]; P = np.full(len(G), np.nan)
    tgt = np.concatenate([target(s) for s in SEGS])
    for te in FOLDS:
        tr = ~np.isin(G, te); ms = [HistGradientBoostingRegressor(random_state=sd, **HGB).fit(Xc[tr], tgt[tr]) for sd in range(NSEED)]
        pr = np.mean([m.predict(Xc[~tr]) for m in ms], 0); P[~tr] = pr
    out = np.empty_like(P)
    for i in range(len(SEGS)):
        m = SEGIDX == i; out[m] = np.maximum.accumulate(P[m])
    return out
def summ(P, n_boot=600):
    return {f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], P)), ci=list(boot_ci(P, Y[t], G, n_boot))) for t in THRS}
def paired(P1, P2, n_boot=600, seed=7):
    out = {}
    for t in THRS:
        yy = Y[t]; up = np.unique(G); GI = [np.where(G == c)[0] for c in up]; rb = np.random.default_rng(seed); d = []
        for _ in range(n_boot):
            pk = rb.integers(0, len(GI), len(GI)); m2 = np.concatenate([GI[i] for i in pk])
            if yy[m2].min() != yy[m2].max(): d.append(roc_auc_score(yy[m2], P1[m2]) - roc_auc_score(yy[m2], P2[m2]))
        out[f"n>{t:g}"] = dict(delta=float(roc_auc_score(yy, P1) - roc_auc_score(yy, P2)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    return out

OUT = dict(A=dict(), B=dict()); SC = {}
P_full = cv(BASE); SC["full"] = P_full; OUT["full"] = summ(P_full)
print(f"  full (123)                          " + "  ".join(f"{v['auc']:.3f}" for v in OUT["full"].values()), flush=True)
def arm(section, nm, X):
    P = cv(X); SC[f"{section}:{nm}"] = P; d = dict(ncols=X.shape[1], **summ(P), vs_full=paired(P, P_full)); OUT[section][nm] = d
    print(f"  {nm:36s} [{X.shape[1]:3d}]  " + "  ".join(f"{d[k]['auc']:.3f} ({d['vs_full'][k]['delta']:+.3f} [{d['vs_full'][k]['ci'][0]:+.3f},{d['vs_full'][k]['ci'][1]:+.3f}])" for k in d if k.startswith("n>")) + f"  [{time.time()-t0:.0f}s]", flush=True)
D5 = np.column_stack([DEMO[k] for k in ("age", "sex", "height", "weight", "bmi")]); M3 = np.column_stack([MAPC[k] for k in ("map_win", "map_delta", "map_read")])
print("\nA. extra inputs appended to the 123 columns", flush=True)
arm("A", "+ demographics (age, sex, height, weight, BMI)", np.hstack([BASE, D5]))
arm("A", "+ arterial pressure (MAP, dMAP, read MAP)", np.hstack([BASE, M3]))
arm("A", "+ demographics + arterial pressure", np.hstack([BASE, D5, M3]))
arm("A", "demographics alone", D5)
arm("A", "arterial pressure alone", M3)
arm("A", "arterial pressure + clock", np.hstack([M3, BASE[:, [IDX["since_last_cvp"]]]]))
print("\nB. the calibration read reduced to its median", flush=True)
WAVE = [IDX[f"d_{n}"] for n in V.names]; CLK = IDX["since_last_cvp"]; MED = IDX["blk_median"]
LINE = [IDX[n] for n in ["since_last_cvp"] + DESC_NAMES + HIST_NAMES + EXTRA_NAMES[:8]]
arm("B", "read median + clock", BASE[:, [MED, CLK]])
arm("B", "read median + clock + waveform", BASE[:, [MED, CLK] + WAVE])
arm("B", "read shape (no median) + clock", BASE[:, [c for c in LINE if c != MED]])
arm("B", "full read columns, no waveform", BASE[:, LINE])
arm("B", "clock + waveform (no read columns)", BASE[:, [CLK] + WAVE])
# MIMIC zero-shot for the MAP arm, on the art-line windows only (map at the window and vs the first tracked window; no read MAP median stored)
print("\nA'. arterial pressure on MIMIC (art-line records, zero-shot)", flush=True)
VD = load_vitaldb_generic(VitalDB(tracker=False)); MI = load_mimic("mimic3wdb", extra=True); MSEG = [s for k in sorted(MI) for s in MI[k] if "map" in s and np.isfinite(s["map"]).mean() > .8]
MB = np.vstack([s["X"] for s in MSEG])[:, ACTIVE]; MT = np.concatenate([target(s) for s in MSEG]); MG = np.concatenate([[s["key"]] * len(s["dev"]) for s in MSEG]); MSI = np.concatenate([[i] * len(s["dev"]) for i, s in enumerate(MSEG)])
MMAP = np.concatenate([s["map"] for s in MSEG]); MMAPD = np.concatenate([s["map"] - np.nanmedian(s["map"][:3]) for s in MSEG])
# VitalDB training rows with the matching two MAP columns (window MAP, MAP vs the first three tracked windows)
VSEG = [s for k in sorted(VD) for s in VD[k]]; VB = np.vstack([s["X"] for s in VSEG])[:, ACTIVE]; VT = np.concatenate([target(s) for s in VSEG])
VMAP = np.concatenate([MAPC["map_win"][SEGIDX == i] if False else np.full(len(s["dev"]), np.nan) for i, s in enumerate(VSEG)])
# align VitalDB generic stretches to the dense-baseline stretches by (key, k) to fetch their MAP columns
key2i = {(s["key"], s["k"]): i for i, s in enumerate(SEGS)}
VMAP, VMAPD = [], []
for s in VSEG:
    i = key2i.get((int(s["key"].split("_")[1]), s["k"]))
    if i is None: VMAP.append(np.full(len(s["dev"]), np.nan)); VMAPD.append(np.full(len(s["dev"]), np.nan)); continue
    m = SEGIDX == i; w = MAPC["map_win"][m]
    if len(w) != len(s["dev"]): VMAP.append(np.full(len(s["dev"]), np.nan)); VMAPD.append(np.full(len(s["dev"]), np.nan)); continue
    VMAP.append(w); VMAPD.append(w - np.nanmedian(w[:3]))
VMAP, VMAPD = np.concatenate(VMAP), np.concatenate(VMAPD)
def zs(Xtr, Xte):
    g = np.isfinite(Xtr).all(1) & np.isfinite(VT); ms = [HistGradientBoostingRegressor(random_state=sd, **HGB).fit(Xtr[g], VT[g]) for sd in range(NSEED)]
    pr = np.mean([m.predict(Xte) for m in ms], 0); out = np.empty_like(pr)
    for i in range(len(MSEG)):
        m = MSI == i; out[m] = np.maximum.accumulate(pr[m])
    return out
YM = {t: (MT > t).astype(int) for t in THRS}
def msumm(P): return {f"n>{t:g}": dict(auc=float(roc_auc_score(YM[t], P)), ci=list(boot_ci(P, YM[t], MG, 300))) for t in THRS}
Pm0 = zs(VB, MB); Pm1 = zs(np.hstack([VB, VMAP[:, None], VMAPD[:, None]]), np.hstack([MB, MMAP[:, None], MMAPD[:, None]]))
OUT["A_mimic"] = dict(records=len(set(MG)), windows=int(len(MG)), full=msumm(Pm0), plus_map=msumm(Pm1)); SC["mimic_full"] = Pm0; SC["mimic_plus_map"] = Pm1
print(f"  {len(set(MG))} art-line records, {len(MG):,} windows: full " + "  ".join(f"{v['auc']:.3f}" for v in OUT['A_mimic']['full'].values()) + "   + MAP " + "  ".join(f"{v['auc']:.3f}" for v in OUT['A_mimic']['plus_map'].values()), flush=True)
np.savez_compressed(f"{RES}/extra_inputs_scores.npz", Y2=Y[2.0], Y3=Y[3.0], Y4=Y[4.0], G=G, YM2=YM[2.0], YM3=YM[3.0], YM4=YM[4.0], MG=MG, **SC)
json.dump(OUT, open(f"{RES}/extra_inputs.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/extra_inputs.json")
