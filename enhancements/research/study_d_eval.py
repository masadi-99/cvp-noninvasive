"""Study D (evaluate) — does the Dawber categorical morphology feature add signal?

Loads per-case morphology features (morph_features.npz), aligns to the 304-case cohort, and tests:
  base5; base5 + morph_mean; base5 + [morph_mean, morph_f1, morph_f4]; morph_mean alone.
Plus the single-feature AUC of morph_mean, a paired test vs base5, and the correlation of morph_mean
with the existing ppg_dicrotic_frac feature (redundancy check).
"""
import json, numpy as np
import enh
from cvpkit import aggregate, config as C

THR = 12.0; REPS = 40

Xb, y, cases = enh.per_case(enh.FEATS5)
m = np.load("morph_features.npz")
mc = m["cid"]; mp = {int(c): i for i, c in enumerate(mc)}
order = np.array([mp[int(c)] for c in cases])
morph = {k: m[k][order] for k in ["morph_mean", "morph_mode", "morph_f1", "morph_f4", "nbeats"]}

# per-case ppg_dicrotic_frac (existing feature) for redundancy check
Mk, ck, gk = aggregate.load_matrix(C.MATRIX_NPZ)
cidk = Mk[:, ck.index("cid")].astype(int); dfj = ck.index("ppg_dicrotic_frac")
dfrac = np.array([np.nanmedian(Mk[cidk == c, dfj]) for c in cases])

def col(name): return morph[name].reshape(-1, 1)

res = {}
configs = {
    "base5": Xb,
    "+morph_mean": np.column_stack([Xb, col("morph_mean")]),
    "+morph(mean,f1,f4)": np.column_stack([Xb, col("morph_mean"), col("morph_f1"), col("morph_f4")]),
    "morph_mean alone": col("morph_mean"),
}
for name, X in configs.items():
    r = enh.evaluate(X, y, cases, threshold=THR, reps=REPS)
    res[name] = dict(auc=r["auc"], ci=r["ci"], sens=r["sens"], spec=r["spec"], nfeat=r["nfeat"])
    print(f"  {name:22} ({r['nfeat']}f): AUC={r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]", flush=True)

# paired test +morph_mean vs base5
rb = enh.evaluate(Xb, y, cases, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
rm = enh.evaluate(configs["+morph_mean"], y, cases, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
delta = rm - rb
res["paired_morph_vs_base"] = dict(mean_delta=float(delta.mean()),
                                   pct_splits_improve=float(np.mean(delta > 0)),
                                   t=float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)) + 1e-12)))
# redundancy: corr(morph_mean, dicrotic_frac) and corr(morph_mean, CVP)
good = np.isfinite(morph["morph_mean"]) & np.isfinite(dfrac)
res["corr_morph_dicroticfrac"] = float(np.corrcoef(morph["morph_mean"][good], dfrac[good])[0, 1])
res["corr_morph_cvp"] = float(np.corrcoef(morph["morph_mean"][good], y[good])[0, 1])
res["mean_class"] = float(np.nanmean(morph["morph_mean"]))
print(f"\n  paired +morph vs base: Δ={delta.mean():+.3f}  {100*np.mean(delta>0):.0f}% improve  t={res['paired_morph_vs_base']['t']:.1f}", flush=True)
print(f"  corr(morph_mean, dicrotic_frac)={res['corr_morph_dicroticfrac']:+.2f}  corr(morph_mean, CVP)={res['corr_morph_cvp']:+.2f}", flush=True)
json.dump(res, open("results_d_morph.json", "w"), indent=2)
print("saved results_d_morph.json", flush=True)
