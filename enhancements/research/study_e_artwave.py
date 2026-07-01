"""Step 1 — test the arterial features that ALREADY exist in medium_eng but were not all tried:
the full 'art' waveform group (8: sbp/dbp/map/pp/ppv/dpdt/hr/pp_cv) + 'cross' (ecg/art/ppg timing).
Quick check before building richer morphology."""
import numpy as np, enh
THR = 12.0; REPS = 40
d = enh.load_merged(); cid = d["cid"]
Xc, yc, cases = enh.per_case(enh.FEATS5); cidx = {c: i for i, c in enumerate(cases)}
rbc = {c: np.where(cid == c)[0] for c in cases}
z = np.load("/home/masadi/run/data/medium_eng.npz", allow_pickle=True)
Me, ce = z["data"], [str(c) for c in z["cols"]]
cide = Me[:, ce.index("cid")].astype(int); starte = Me[:, ce.index("start")].astype(int)
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cide, starte))}
order = np.array([idx[k] for k in d["key"]])
ART8 = ["art_sbp", "art_dbp", "art_map", "art_pp", "art_ppv", "art_dpdt", "art_hr", "art_pp_cv"]
CROSS = ["pat_ecg_art", "pat_art_ppg", "art_ppg_amp_ratio"]
for f in ART8 + CROSS:
    d["F"][f] = Me[order, ce.index(f)].astype(float)
pcmed = lambda f: np.array([np.nanmedian(d["F"][f][rbc[c]]) for c in cases])
FT = {f: pcmed(f) for f in ART8 + CROSS}
mat = lambda fs: np.column_stack([Xc[:, [enh.FEATS5.index(x) for x in fs if x in enh.FEATS5]]] +
                                 [FT[x].reshape(-1, 1) for x in fs if x not in enh.FEATS5]) if any(x in enh.FEATS5 for x in fs) else np.column_stack([FT[x] for x in fs])

def X(base, add): return np.column_stack([Xc] + [FT[a].reshape(-1, 1) for a in add])

print("single-feature AUC of every 'art'/'cross' feature:")
for f in ART8 + CROSS:
    a = enh.evaluate(FT[f].reshape(-1, 1), yc, cases, threshold=THR, reps=REPS)["auc"]
    print(f"  {f:18} {a:.3f}", flush=True)

print("\ncombined:")
for name, add in [("base5", []), ("base5 + art8", ART8), ("base5 + art8 + cross", ART8 + CROSS), ("art8 alone", None)]:
    if add is None:
        r = enh.evaluate(np.column_stack([FT[a] for a in ART8]), yc, cases, threshold=THR, reps=REPS)
    else:
        r = enh.evaluate(X(Xc, add), yc, cases, threshold=THR, reps=REPS, k_select=(12 if len(add) > 12 else None))
    print(f"  {name:24} AUC {r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] Sens {r['sens']:.2f} Spec {r['spec']:.2f}", flush=True)

# paired base5+art8 vs base5
rb = enh.evaluate(Xc, yc, cases, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
ra = enh.evaluate(X(Xc, ART8), yc, cases, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
dl = ra - rb
print(f"\npaired base5+art8 vs base5: delta={dl.mean():+.3f}, {100*np.mean(dl>0):.0f}% improve, t={dl.mean()/(dl.std(ddof=1)/np.sqrt(len(dl))+1e-12):.1f}", flush=True)
