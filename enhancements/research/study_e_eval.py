"""Evaluate the rich arterial-waveform features vs the 5-feature baseline (honest nested grouped CV;
in-fold SelectKBest for the many-feature configs to avoid winner's curse)."""
import json, numpy as np, enh
THR = 12.0; REPS = 40
Xc, yc, cases = enh.per_case(enh.FEATS5); ybc = (yc > THR).astype(int); cidx = {c: i for i, c in enumerate(cases)}
m = np.load("artwave_features.npz"); mp = {int(c): i for i, c in enumerate(m["cid"])}
order = np.array([mp[int(c)] for c in cases])
AW = [k for k in m.files if k.startswith("artw_")]
F = {k: m[k][order] for k in AW}
def col(k): return F[k].reshape(-1, 1)
out = {}

# single-feature AUC + correlation with CVP
print("single-feature CV AUC (rich arterial-waveform features):", flush=True)
sf = {}
for k in AW:
    a = enh.evaluate(col(k), yc, cases, threshold=THR, reps=REPS)["auc"]
    g = np.isfinite(F[k]) & np.isfinite(yc); r = np.corrcoef(F[k][g], yc[g])[0, 1]
    sf[k] = dict(auc=round(float(a), 3), corr=round(float(r), 3))
    print(f"  {k:18} AUC {a:.3f}   corr {r:+.2f}", flush=True)
out["single_feature"] = sf

base = enh.evaluate(Xc, yc, cases, threshold=THR, reps=REPS)
print(f"\nbase5: {base['auc']:.3f}", flush=True)
Xall = np.column_stack([Xc] + [col(k) for k in AW])
for name, X, k_sel in [("base5 + artwave-15 (SelectKBest 8)", Xall, 8),
                       ("base5 + artwave-15 (SelectKBest 12)", Xall, 12),
                       ("base5 + artwave-15 (no select)", Xall, None),
                       ("artwave-15 alone (SelectKBest 8)", np.column_stack([col(k) for k in AW]), 8)]:
    r = enh.evaluate(X, yc, cases, threshold=THR, reps=REPS, k_select=k_sel)
    out[name] = dict(auc=r["auc"], ci=r["ci"], sens=r["sens"], spec=r["spec"], nfeat=r["nfeat"])
    print(f"  {name:38} AUC {r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] Sens {r['sens']:.2f} Spec {r['spec']:.2f}", flush=True)

# best single artw feature added on its own
best = max(sf, key=lambda k: sf[k]["auc"])
rbest = enh.evaluate(np.column_stack([Xc, col(best)]), yc, cases, threshold=THR, reps=REPS)
out[f"base5 + {best}"] = dict(auc=rbest["auc"], ci=rbest["ci"])
print(f"\n  base5 + best single ({best}): {rbest['auc']:.3f}", flush=True)

# paired: base5 + artwave-15(k=8) vs base5
rb = enh.evaluate(Xc, yc, cases, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
ra = enh.evaluate(Xall, yc, cases, threshold=THR, reps=REPS, k_select=8, return_reps=True)["reps_auc"]
dl = ra - rb
out["paired_artwave_vs_base"] = dict(mean_delta=float(dl.mean()), pct_improve=float(np.mean(dl > 0)),
                                     t=float(dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl)) + 1e-12)))
print(f"\n  paired base5+artwave(k8) vs base5: delta={dl.mean():+.4f}, {100*np.mean(dl>0):.0f}% improve, t={dl.mean()/(dl.std(ddof=1)/np.sqrt(len(dl))+1e-12):.2f}", flush=True)
out["base5"] = float(base["auc"])
json.dump(out, open("results_e_artwave.json", "w"), indent=2)
print("saved results_e_artwave.json", flush=True)
