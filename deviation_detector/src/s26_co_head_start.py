"""Step 26 -- cardiac output: stretches that were already off at the end of calibration.

The label is the deviation from the calibration median. When the last calibration reading itself lies
more than n outside that median, every tracked window of the stretch is positive from its first window
and the read columns (read |last - median|, read spread) see it directly. This step re-scores every
arm and baseline of steps 23 (VitalDB CO / CI / SV / CVP on the same patients) and 25 (MIMIC CO) with
those stretches removed. No model is refitted: the stored per-window scores are sliced with a mask
built from the stored calibration reads, in the stretch order the scores were written in.

The same control is then run on the main CVP system (step 3 out-of-fold scores on VitalDB, step 7 zero-shot
scores on MIMIC-III), rebuilding the stretches and matching each one to its stored scores by (record, index).

reads   results/cardiac_output_{scores,traces}.npz, results/cardiac_output_mimic_{scores,traces}.npz,
        results/vitaldb_oof.npz, results/mimic_scores.npz
writes  results/cardiac_output_headstart.json, results/cardiac_output_headstart_masks.npz
"""
import os, sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib.build import boot_ci, VitalDB, build_vitaldb, load_mimic, target
t0 = time.time()


def head_mask(T, prefix, thr):
    """Per-window mask: True where the stretch's last finite calibration reading is more than thr from the
    calibration median. Stretches are taken in archive order, which is the order the scores were written in."""
    keys = [f[:-5] for f in T.files if f.startswith(prefix) and f.endswith("__dev")]
    H, n_pos, n_head, rec = [], 0, 0, []
    for kk in keys:
        dev = T[f"{kk}__dev"]; yb = T[f"{kk}__yb"]; base = float(T[f"{kk}__base"][0]); f = np.isfinite(yb)
        h = bool(f.any() and abs(yb[f][-1] - base) > thr); pos = bool((np.maximum.accumulate(dev) > thr).any())
        H.append(np.full(len(dev), h)); n_pos += pos; n_head += (pos and h); rec.append(np.full(len(dev), kk[len(prefix):].rsplit(":", 1)[0]))
    return np.concatenate(H), n_pos, n_head, np.concatenate(rec)


def block(SC, T, prefix, arms, gkey, ykey, thrs, unit):
    out = dict(thresholds=list(thrs), unit=unit, counts={}, all={a: {} for a in arms}, no_head_start={a: {} for a in arms}); masks = {}
    G = SC[gkey]
    for t in thrs:
        H, n_pos, n_head, rec = head_mask(T, prefix, t); Y = SC[ykey(t)]; assert len(H) == len(Y) == len(G), (len(H), len(Y), len(G))
        assert np.all(rec == G.astype(str)), "trace order does not match score order"  # every window's record id agrees
        m = ~H; k = f"n>{t:g}"
        out["counts"][k] = dict(positive_stretches=int(n_pos), head_start_stretches=int(n_head), stretches=int(len([f for f in T.files if f.startswith(prefix) and f.endswith("__dev")])), windows_all=int(len(Y)), windows_head=int(H.sum()),
                                windows_kept=int(m.sum()), positives_all=int(Y.sum()), positives_head=int(Y[H].sum()), prevalence_all=float(Y.mean()), prevalence_kept=float(Y[m].mean()))
        for a in arms:
            P = SC[a if prefix == "" else f"{prefix}{a}"]
            out["all"][a][k] = float(roc_auc_score(Y, P)); out["no_head_start"][a][k] = dict(auc=float(roc_auc_score(Y[m], P[m])), ci=list(boot_ci(P[m], Y[m], G[m], 200)))
        masks[f"{prefix}H{t:g}"] = H
        print(f"  {prefix or 'mimic:'}n>{t:g}: positive stretches {n_pos}, already off at calibration end {n_head}; windows kept {m.sum():,} of {len(Y):,}, prevalence {100*Y.mean():.1f}% -> {100*Y[m].mean():.1f}%", flush=True)
        for a in arms: print(f"      {a[:46]:46s} all {out['all'][a][k]:.3f}  without head start {out['no_head_start'][a][k]['auc']:.3f} [{out['no_head_start'][a][k]['ci'][0]:.3f},{out['no_head_start'][a][k]['ci'][1]:.3f}]", flush=True)
    return out, masks


OUT, MASKS = {}, {}
J = json.load(open(f"{RES}/cardiac_output.json")); CS = np.load(f"{RES}/cardiac_output_scores.npz", allow_pickle=True); CT = np.load(f"{RES}/cardiac_output_traces.npz", allow_pickle=True)
for k, d in J.items():
    arms = list(d["arms"]) + [f"base:{b}" for b in d["baselines"]]
    OUT[k], mk = block(CS, CT, f"{k}:", arms, f"{k}:G", lambda t: f"{k}:Y{t:g}", d["thresholds"], d["unit"]); OUT[k]["label"] = d["label"]; MASKS.update(mk)
f = f"{RES}/cardiac_output_mimic.json"
if os.path.exists(f):
    JM = json.load(open(f)); MS = np.load(f"{RES}/cardiac_output_mimic_scores.npz", allow_pickle=True); MT = np.load(f"{RES}/cardiac_output_mimic_traces.npz", allow_pickle=True)
    arms = [a for a in list(JM["arms"]) + list(JM["baselines"]) if "absolute-error" not in a]
    OUT["mimic"], mk = block(MS, MT, "", arms, "G", lambda t: f"Y{t:g}", JM["thresholds"], "L/min"); OUT["mimic"]["label"] = "MIMIC-III cardiac output (zero-shot)"; MASKS.update({f"mimic:{k}": v for k, v in mk.items()})


def main_cvp(segs, PM, thrs, label):
    """The main CVP system: per-stretch stored scores matched by (record, stretch index); 'full' is the only arm."""
    use = [(s, PM[(s["key"], s["k"])]) for s in segs if (s["key"], s["k"]) in PM]; assert all(len(p) == len(s["dev"]) for s, p in use) and len(use) == len(segs)
    P = np.concatenate([p for _, p in use]); G = np.concatenate([[str(s["key"])] * len(s["dev"]) for s, _ in use])
    out = dict(thresholds=list(thrs), unit="mmHg", label=label, counts={}, all={"full": {}}, no_head_start={"full": {}}, stretches=len(use))
    for t in thrs:
        Y = np.concatenate([(target(s) > t).astype(int) for s, _ in use]); H = np.concatenate([np.full(len(s["dev"]), abs(s["yb"][np.isfinite(s["yb"])][-1] - s["base"]) > t) for s, _ in use]); m = ~H; k = f"n>{t:g}"
        n_pos = sum((np.maximum.accumulate(s["dev"]) > t).any() for s, _ in use); n_head = sum((np.maximum.accumulate(s["dev"]) > t).any() and abs(s["yb"][np.isfinite(s["yb"])][-1] - s["base"]) > t for s, _ in use)
        out["counts"][k] = dict(positive_stretches=int(n_pos), head_start_stretches=int(n_head), stretches=len(use), windows_all=int(len(Y)), windows_head=int(H.sum()), windows_kept=int(m.sum()),
                                positives_all=int(Y.sum()), positives_head=int(Y[H].sum()), prevalence_all=float(Y.mean()), prevalence_kept=float(Y[m].mean()))
        out["all"]["full"][k] = float(roc_auc_score(Y, P)); out["no_head_start"]["full"][k] = dict(auc=float(roc_auc_score(Y[m], P[m])), ci=list(boot_ci(P[m], Y[m], G[m], 200)))
        print(f"  {label} n>{t:g}: positive stretches {n_pos}, already off at calibration end {n_head}; full all {out['all']['full'][k]:.3f} -> without head start {out['no_head_start']['full'][k]['auc']:.3f}", flush=True)
    return out


V = VitalDB(); R = build_vitaldb(V); VS = [s for c in sorted(R) for s in R[c]]
z = np.load(f"{RES}/vitaldb_oof.npz", allow_pickle=True); PM = {(int(c), int(k)): np.asarray(p) for c, k, p in zip(z["cid"], z["k"], z["p"])}
OUT["cvp_main_vitaldb"] = main_cvp(VS, PM, THRS, "main CVP system, VitalDB out of fold"); OUT["cvp_main_vitaldb"]["patients"] = len(R)
MI = load_mimic("mimic3wdb"); MSEGS = [s for k in sorted(MI) for s in MI[k]]; Z = np.load(f"{RES}/mimic_scores.npz", allow_pickle=True); REC, K, P = Z["REC"], Z["K"], Z["P"]; PMm = {}; o = 0
for s in MSEGS:  # mimic_scores.npz is written in the same sorted-record order; each block is checked by record id and stretch index
    n = len(s["dev"]); assert np.all(REC[o:o + n].astype(str) == str(s["key"])) and np.all(K[o:o + n] == s["k"]), s["key"]; PMm[(s["key"], s["k"])] = P[o:o + n]; o += n
assert o == len(P)
OUT["cvp_main_mimic"] = main_cvp(MSEGS, PMm, THRS, "main CVP system, MIMIC-III zero-shot"); OUT["cvp_main_mimic"]["records"] = len(MI)
np.savez_compressed(f"{RES}/cardiac_output_headstart_masks.npz", **MASKS)
json.dump(OUT, open(f"{RES}/cardiac_output_headstart.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/cardiac_output_headstart.json")
