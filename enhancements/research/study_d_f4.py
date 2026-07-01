"""Focused validation of the one morphology feature that helped: the Dawber CLASS-4 fraction
(fraction of fully-damped, notchless beats). base5 + f4 = 0.776 (stable both seeds). Is it a real,
robust, fully-non-invasive gain? Paired test, permutation null, correlations, single-feature AUC."""
import json, numpy as np
import enh
from cvpkit import aggregate, config as C

THR = 12.0; REPS = 60
Xb, y, cases = enh.per_case(enh.FEATS5)
m = np.load("morph_features.npz"); mp = {int(c): i for i, c in enumerate(m["cid"])}
order = np.array([mp[int(c)] for c in cases])
f1 = m["morph_f1"][order]; f4 = m["morph_f4"][order]
Xf = np.column_stack([Xb, f1.reshape(-1, 1), f4.reshape(-1, 1)])   # the best stable config (base5 + both class fractions)

# upstroke + dicrotic_frac for redundancy
up = Xb[:, enh.FEATS5.index("ppg_upstroke")]
Mk, ck, gk = aggregate.load_matrix(C.MATRIX_NPZ); cidk = Mk[:, ck.index("cid")].astype(int)
dfj = ck.index("ppg_dicrotic_frac"); dfrac = np.array([np.nanmedian(Mk[cidk == c, dfj]) for c in cases])

out = {}
rb = enh.evaluate(Xb, y, cases, threshold=THR, reps=REPS, seed=0, return_reps=True)
rf = enh.evaluate(Xf, y, cases, threshold=THR, reps=REPS, seed=0, return_reps=True)
out["base5_auc"] = round(float(rb["auc"]), 4); out["base5_morph_auc"] = round(float(rf["auc"]), 4)
out["base5_morph_ci"] = [round(x, 4) for x in rf["ci"]]; out["base5_morph_sens"] = round(rf["sens"], 3); out["base5_morph_spec"] = round(rf["spec"], 3)
delta = rf["reps_auc"] - rb["reps_auc"]
out["paired"] = dict(mean_delta=float(delta.mean()), pct_improve=float(np.mean(delta > 0)),
                     t=float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)) + 1e-12)), reps=len(delta))
print(f"base5={rb['auc']:.3f}  base5+f1+f4={rf['auc']:.3f} [{rf['ci'][0]:.3f},{rf['ci'][1]:.3f}] Sens {rf['sens']:.2f} Spec {rf['spec']:.2f}", flush=True)
print(f"PAIRED +f1+f4 vs base5: Δ={delta.mean():+.4f}  {100*out['paired']['pct_improve']:.0f}% improve  t={out['paired']['t']:.2f}", flush=True)

# permutation null for base5+f4
nulls = []
for s in range(40):
    rng = np.random.default_rng(8000 + s); yp = y.copy(); rng.shuffle(yp)
    nulls.append(enh.evaluate(Xf, yp, cases, threshold=THR, reps=1)["auc"])
nulls = np.array(nulls)
out["perm"] = dict(null_mean=float(nulls.mean()), p95=float(np.percentile(nulls, 95)), p_ge=float(np.mean(nulls >= rf["auc"])))
print(f"perm null: mean={nulls.mean():.3f} p95={np.percentile(nulls,95):.3f}  p(>=real)={out['perm']['p_ge']:.3f}", flush=True)

# single-feature AUC of f4 + correlations
sf4 = enh.evaluate(f4.reshape(-1, 1), y, cases, threshold=THR, reps=REPS)["auc"]
g = np.isfinite(f4)
out["f4_single_auc"] = round(float(sf4), 4)
out["corr_f4_cvp"] = round(float(np.corrcoef(f4[g], y[g])[0, 1]), 3)
out["corr_f4_upstroke"] = round(float(np.corrcoef(f4[g], up[g])[0, 1]), 3)
out["corr_f4_dicroticfrac"] = round(float(np.corrcoef(f4[g], dfrac[g])[0, 1]), 3)
out["f4_mean_pos"] = round(float(np.nanmean(f4[y > THR])), 3); out["f4_mean_neg"] = round(float(np.nanmean(f4[y <= THR])), 3)
print(f"f4 single AUC={sf4:.3f}  corr(f4,CVP)={out['corr_f4_cvp']:+.2f} corr(f4,upstroke)={out['corr_f4_upstroke']:+.2f} corr(f4,dicfrac)={out['corr_f4_dicroticfrac']:+.2f}", flush=True)
print(f"f4 fully-damped fraction: elevated={out['f4_mean_pos']:.2f} vs normal={out['f4_mean_neg']:.2f}", flush=True)
json.dump(out, open("results_d_f4.json", "w"), indent=2)
print("saved results_d_f4.json", flush=True)
