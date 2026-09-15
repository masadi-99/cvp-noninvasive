"""Step 18. The one-column, no-model score (block CVP spread x minutes since the read, causal
running max) at every tracking length, on VitalDB and zero-shot on MIMIC-III; next to it the
fitted one-column model from step 17 where that column was the survivor.
Writes results/one_feature_horizons.json
"""
import os, sys, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, W_CAL
from lib.build import VitalDB, build_vitaldb, load_mimic, target, boot_ci, feature_names

V = VitalDB(tracker=False); NAMES = feature_names(V.names)
ONE_COL = json.load(open(f"{RES}/feature_reduction.json"))["stages"][-1]["columns"][0]     # the last survivor of the elimination
c = NAMES.index(ONE_COL); out = {"column": ONE_COL}; SC = {}
for S in (15.0, 20.0, 30.0, 45.0, 60.0):
    d = {}
    for coh, R in (("vitaldb", build_vitaldb(V, W_CAL, S, tracker=False)), ("mimic", load_mimic("mimic3wdb", W_CAL, S))):
        segs = [s for k in sorted(R) for s in R[k]]
        P = np.concatenate([np.maximum.accumulate(np.nan_to_num(s["X"][:, c], nan=0.0)) for s in segs])
        G = np.concatenate([[s["key"]] * len(s["dev"]) for s in segs]); T = np.concatenate([target(s) for s in segs])
        d[coh] = {f"n>{t:g}": dict(auc=float(roc_auc_score((T > t).astype(int), P)), ci=list(boot_ci(P, (T > t).astype(int), G, 200))) for t in THRS}
        SC[f"{coh}_S{S:g}__P"] = P; SC[f"{coh}_S{S:g}__T"] = T
        d[coh]["windows"] = int(len(P))
    j = json.load(open(f"{RES}/longer_S{S:g}.json")); one = [x for x in j["stages"] if x["n"] == 1][0]
    d["fitted_one_column"] = dict(column=one["columns"][0], vitaldb={k: one["vitaldb"][k]["auc"] for k in one["vitaldb"]}, mimic={k: one["mimic"][k]["auc"] for k in one["mimic"]})
    out[f"S{S:g}"] = d
    print(f"S={S:g}  no-model {ONE_COL}: VitalDB " + " ".join(f"{d['vitaldb'][k]['auc']:.3f}" for k in ("n>2", "n>3", "n>4")) +
          "   MIMIC " + " ".join(f"{d['mimic'][k]['auc']:.3f}" for k in ("n>2", "n>3", "n>4")) +
          f"   | fitted 1-col ({one['columns'][0]}): VitalDB " + " ".join(f"{one['vitaldb'][k]['auc']:.3f}" for k in one["vitaldb"]) + "  MIMIC " + " ".join(f"{one['mimic'][k]['auc']:.3f}" for k in one["mimic"]), flush=True)
np.savez_compressed(f"{RES}/one_feature_horizons_scores.npz", **SC)
json.dump(out, open(f"{RES}/one_feature_horizons.json", "w"), indent=1); print("-> results/one_feature_horizons.json")
