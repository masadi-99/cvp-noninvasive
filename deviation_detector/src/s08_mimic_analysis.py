"""Step 8. What the zero-shot detector actually does on MIMIC-III, beyond one AUC.

All on the 392 MIMIC-III records, model trained on VitalDB only (as in step 7, arm A):
  scores       per-window running-max score, the signed model's raw output, truth, identifiers
  confusion    confusion matrices at fixed specificity (90/80/70 %) and at the Youden point
  calibration  is the predicted magnitude in mmHg, or only an ordering?
  per_record   AUC computed INSIDE each record, then averaged (what a bedside monitor needs)
  persistence  re-score with a positive required to persist >= 2 / 3 windows (label artefacts)
  direction    a SIGNED model: can it tell a rise from a fall?
  structure    what the running max costs: events that return inside the band, flagged time
               already recovered, stretches with both directions
  cap          why the score saturates near 2 mmHg: target distribution, in-sample reach, loss
  spread       between-record spread of the score; a global cut vs a causal within-record rank
  thresholds   per-record operating point set causally from the record's own past
Writes results/mimic_scores.npz and results/mimic_analysis.json
"""
import os, sys, time, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS, HGB
from lib.build import VitalDB, load_vitaldb_generic, load_mimic, fit, score, target, ACTIVE
from sklearn.ensemble import HistGradientBoostingRegressor

t0 = time.time()
MI = load_mimic("mimic3wdb", extra=True)
VD = load_vitaldb_generic(VitalDB(tracker=False))
m3 = sorted(MI)
MS = fit(VD, list(VD), drop_nan_rows=True)
segs = [s for k in m3 for s in MI[k]]
PW = score(MS, segs)                                   # running-max magnitude score
RAW = score(MS, segs, running_max=False)               # the raw per-window regressor output
signed = fit(VD, list(VD), drop_nan_rows=True, target_fn=lambda s: s["sdev"])
PS = score(signed, segs, running_max=False)            # signed raw output
P = np.concatenate(PW); TRUE = np.concatenate([target(s) for s in segs])
SP = np.concatenate(PS); SDEV = np.concatenate([s["sdev"] for s in segs])
REC = np.concatenate([[s["key"]] * len(s["dev"]) for s in segs])
KST = np.concatenate([[s["k"]] * len(s["dev"]) for s in segs])
np.savez_compressed(f"{RES}/mimic_scores.npz", P=P, RAW=np.concatenate(RAW), TRUE=TRUE,
                    DEV=np.concatenate([s["dev"] for s in segs]), PS=SP, SDEV=SDEV, REC=REC,
                    K=KST, T=np.concatenate([s["t"] for s in segs]),
                    SINCE=np.concatenate([s["since"] for s in segs]),
                    MAP=np.concatenate([s.get("map", np.full(len(s["dev"]), np.nan)) for s in segs]),
                    BASE=np.concatenate([[s["base"]] * len(s["dev"]) for s in segs]))
print(f"{len(m3)} MIMIC-III records, {len(P):,} windows scored zero-shot [{time.time()-t0:.0f}s]", flush=True)
OUT = dict(n_records=len(m3), n_windows=int(len(P)), n_stretches=len(segs))


def youden_cut(p, y):
    o = np.argsort(p); ys, ps = y[o], p[o]
    tp = np.cumsum(ys[::-1])[::-1]; fp = np.cumsum((1 - ys)[::-1])[::-1]
    tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
    return float(ps[int(np.argmax(tpr - fpr))])


def cm(pred, Y):
    TP = int((pred & (Y == 1)).sum()); FN = int((~pred & (Y == 1)).sum())
    FP = int((pred & (Y == 0)).sum()); TN = int((~pred & (Y == 0)).sum())
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, sens=TP / max(TP + FN, 1), spec=TN / max(TN + FP, 1),
                ppv=TP / max(TP + FP, 1), npv=TN / max(TN + FN, 1))


# ---- confusion + calibration ------------------------------------------------------------------
OUT["confusion"] = {}
print("\nCONFUSION MATRICES (zero-shot VitalDB -> MIMIC-III)")
for thr in THRS:
    Y = (TRUE > thr).astype(int); neg = P[Y == 0]
    d = dict(prevalence=float(Y.mean()), auc=float(roc_auc_score(Y, P)), points={})
    for nm, cut in [(f"spec{100*(1-f):.0f}", float(np.quantile(neg, 1 - f))) for f in (.1, .2, .3)] + \
                   [("youden", youden_cut(P, Y))]:
        d["points"][nm] = dict(cut=cut, **cm(P >= cut, Y))
    OUT["confusion"][f"n>{thr:g}"] = d
    print(f"  n>{thr:g}  prev {100*Y.mean():.1f}%  AUC {d['auc']:.3f}   " +
          "  ".join(f"{k}: cut {v['cut']:.2f} sens {100*v['sens']:.0f}% spec {100*v['spec']:.0f}% PPV {100*v['ppv']:.0f}%"
                    for k, v in d["points"].items()))
sl, ic = np.polyfit(TRUE, P, 1)
ed = [0, 1, 2, 3, 4, 6, 30]; bins = []
for a, b in zip(ed[:-1], ed[1:]):
    m = (TRUE >= a) & (TRUE < b)
    if m.sum() >= 50: bins.append(dict(lo=a, hi=b, n=int(m.sum()), mean_true=float(TRUE[m].mean()), mean_pred=float(P[m].mean())))
OUT["calibration"] = dict(slope=float(sl), intercept=float(ic), r=float(np.corrcoef(P, TRUE)[0, 1]), bins=bins,
                          pred_pct={q: float(np.percentile(P, q)) for q in (1, 50, 99)})
print(f"\nCALIBRATION: predicted = {sl:.3f} x true {ic:+.3f}, r = {np.corrcoef(P, TRUE)[0,1]:+.3f}")
for b in bins: print(f"    true {b['lo']}-{b['hi']} mmHg  n={b['n']:7d}  mean true {b['mean_true']:.2f}  mean predicted {b['mean_pred']:.2f}")

# ---- per record ---------------------------------------------------------------------------------
REC_ = {}
for s, p in zip(segs, PW):
    REC_.setdefault(s["key"], [[], []]); REC_[s["key"]][0].append(p); REC_[s["key"]][1].append(target(s))
REC_ = {k: (np.concatenate(a), np.concatenate(b)) for k, (a, b) in REC_.items()}
OUT["per_record"] = {}
print("\nPER-RECORD AUC, averaged over records (vs pooled)")
for thr in THRS:
    aus = {}
    for k, (p, tr) in REC_.items():
        Y = (tr > thr).astype(int)
        if Y.min() != Y.max(): aus[k] = float(roc_auc_score(Y, p))
    v = np.array(list(aus.values()))
    Yall = (TRUE > thr).astype(int); cut = youden_cut(P, Yall)
    per = {q: [] for q in ("sens", "spec", "ppv", "npv")}
    for k, (p, tr) in REC_.items():
        c = cm(p >= cut, (tr > thr).astype(int))
        if c["TP"] + c["FN"]: per["sens"].append(c["sens"])
        if c["TN"] + c["FP"]: per["spec"].append(c["spec"])
        if c["TP"] + c["FP"]: per["ppv"].append(c["ppv"])
        if c["TN"] + c["FN"]: per["npv"].append(c["npv"])
    OUT["per_record"][f"n>{thr:g}"] = dict(evaluable=len(v), mean=float(v.mean()), median=float(np.median(v)),
                                          iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                                          frac_gt_0_7=float(np.mean(v > .7)), frac_gt_0_5=float(np.mean(v > .5)),
                                          pooled=float(roc_auc_score(Yall, P)), youden_cut=cut,
                                          op={q: float(np.mean(x)) for q, x in per.items()}, per_record=aus)
    print(f"  n>{thr:g}  {len(v)}/{len(REC_)} evaluable  mean {v.mean():.3f}  median {np.median(v):.3f}  "
          f">0.7 {100*np.mean(v>.7):.0f}%  >0.5 {100*np.mean(v>.5):.0f}%  pooled {roc_auc_score(Yall,P):.3f}  "
          f"| Youden cut {cut:.2f}: sens {100*np.mean(per['sens']):.0f}% spec {100*np.mean(per['spec']):.0f}%")

# ---- persistence ------------------------------------------------------------------------------
def persist(d, thr, k):
    ex = (d > thr).astype(int); run = np.zeros(len(ex), int); c = 0
    for i, v in enumerate(ex):
        c = c + 1 if v else 0; run[i] = c
    return np.maximum.accumulate(run >= k).astype(int)
OUT["persistence"] = {}
print("\nPERSISTENCE: a positive must last >= k consecutive windows")
for thr in THRS:
    for kk in (1, 2, 3):
        allP, allY, per = [], [], []
        for k in m3:
            ps = [p for s, p in zip(segs, PW) if s["key"] == k]
            ds = [s["dev"] for s in MI[k]]
            p = np.concatenate(ps); y = np.concatenate([persist(d, thr, kk) for d in ds])
            allP.append(p); allY.append(y)
            if y.min() != y.max(): per.append(roc_auc_score(y, p))
        P2, Y2 = np.concatenate(allP), np.concatenate(allY)
        OUT["persistence"][f"n>{thr:g}_k{kk}"] = dict(prevalence=float(Y2.mean()), pooled=float(roc_auc_score(Y2, P2)),
                                                    per_record=float(np.mean(per)), n_records=len(per))
        print(f"  n>{thr:g} k={kk}  prev {100*Y2.mean():5.1f}%  pooled {roc_auc_score(Y2,P2):.3f}  per-record {np.mean(per):.3f} (n={len(per)})")

# ---- direction ----------------------------------------------------------------------------------
OUT["direction"] = {}
print("\nDIRECTION (signed model): restricted to |true| > n")
for thr in (0.0, 1.0, 2.0, 3.0, 4.0):
    m = np.abs(SDEV) > thr
    yy = (SDEV[m] > 0).astype(int)
    OUT["direction"][f"abs>{thr:g}"] = dict(n=int(m.sum()), up_rate=float(yy.mean()),
                                           sign_acc=float(np.mean((SP[m] > 0) == (SDEV[m] > 0))),
                                           auc=float(roc_auc_score(yy, SP[m])))
    print(f"  |true|>{thr:g}  n={m.sum():7d}  up {100*yy.mean():.0f}%  sign acc {100*np.mean((SP[m]>0)==(SDEV[m]>0)):.0f}%  AUC {roc_auc_score(yy, SP[m]):.3f}")
sls, ics = np.polyfit(SDEV, SP, 1)
OUT["direction"]["signed_calibration"] = dict(slope=float(sls), intercept=float(ics), r=float(np.corrcoef(SP, SDEV)[0, 1]))
# stretch level: running mean of the signed output vs the sign of the peak true deviation
st = []
for s, ps in zip(segs, PS):
    if np.abs(s["sdev"]).max() > 3.0:
        rm = np.cumsum(ps) / np.arange(1, len(ps) + 1)
        st.append((1 if s["sdev"][np.argmax(np.abs(s["sdev"]))] > 0 else 0, rm[-1]))
sg = np.array([a for a, b in st]); pp = np.array([b for a, b in st])
OUT["direction"]["stretch_level_n3"] = dict(n=len(sg), up_rate=float(sg.mean()), acc=float(np.mean((pp > 0) == (sg == 1))),
                                            auc=float(roc_auc_score(sg, pp)))
print(f"  stretch-level (peak >3): n={len(sg)}  rises {100*sg.mean():.0f}%  accuracy {100*np.mean((pp>0)==(sg==1)):.0f}%  AUC {roc_auc_score(sg, pp):.3f}")

# ---- structure ----------------------------------------------------------------------------------
OUT["structure"] = {}
print("\nSTRUCTURE of the events (what a running max implies)")
for thr in THRS:
    anyev = both = ret = flagged = recovered = 0
    for s in segs:
        d = s["sdev"]; up = (d > thr).any(); dn = (d < -thr).any()
        if not (up or dn): continue
        anyev += 1; both += int(up and dn)
        ex = np.abs(d) > thr; first = int(np.argmax(ex)); after = ~ex[first:]
        ret += int(after.sum() >= 2); flagged += len(d) - first; recovered += int(after.sum())
    OUT["structure"][f"n>{thr:g}"] = dict(stretches_with_event=anyev, both_directions=both / max(anyev, 1),
                                         deviate_then_return=ret / max(anyev, 1), flagged_time_recovered=recovered / max(flagged, 1))
    print(f"  n>{thr:g}  {anyev} stretches with an event  both directions {100*both/max(anyev,1):.0f}%  "
          f"return {100*ret/max(anyev,1):.0f}%  flagged time already recovered {100*recovered/max(flagged,1):.0f}%")

# ---- cap -------------------------------------------------------------------------------------------
A = np.vstack([s["X"] for v in VD.values() for s in v])[:, ACTIVE]; tg = np.concatenate([target(s) for v in VD.values() for s in v])
g = np.isfinite(A).all(1) & np.isfinite(tg); A, tg = A[g], tg[g]
ins = np.mean([m.predict(A) for m in MS], 0)
OUT["cap"] = dict(train_n=int(len(tg)), target_pct={q: float(np.percentile(tg, q)) for q in (10, 25, 50, 75, 90, 95, 99)},
                  target_max=float(tg.max()), insample_pct={q: float(np.percentile(ins, q)) for q in (50, 99)},
                  insample_max=float(ins.max()),
                  frac_above={v: [float(np.mean(tg > v)), float(np.mean(ins > v))] for v in (1, 2, 3, 4, 5)},
                  losses={})
BIG = np.vstack([s["X"] for s in segs])[:, ACTIVE]
for loss in ("absolute_error", "squared_error", "poisson"):
    hp = dict(HGB); hp["loss"] = loss
    ms = [HistGradientBoostingRegressor(random_state=sd, **hp).fit(A, tg) for sd in range(2)]
    pr = np.mean([m.predict(BIG) for m in ms], 0); pl, o = [], 0
    for s in segs: n = len(s["dev"]); pl.append(np.maximum.accumulate(pr[o:o + n])); o += n
    pl = np.concatenate(pl)
    OUT["cap"]["losses"][loss] = dict(p1=float(np.percentile(pl, 1)), p50=float(np.percentile(pl, 50)), p99=float(np.percentile(pl, 99)),
                                     auc2=float(roc_auc_score((TRUE > 2).astype(int), pl)), auc3=float(roc_auc_score((TRUE > 3).astype(int), pl)),
                                     slope=float(np.polyfit(TRUE, pl, 1)[0]))
print(f"\nCAP: training target median {np.median(tg):.2f}, 99th {np.percentile(tg,99):.2f}, max {tg.max():.1f}; "
      f"in-sample prediction 99th {np.percentile(ins,99):.2f}, max {ins.max():.2f}")
for k, v in OUT["cap"]["losses"].items(): print(f"  {k:15s} pred 1/50/99th {v['p1']:.2f}/{v['p50']:.2f}/{v['p99']:.2f}  AUC n>2 {v['auc2']:.3f} n>3 {v['auc3']:.3f}  slope {v['slope']:.3f}")

# ---- spread + thresholds (n>3) ------------------------------------------------------------------
THR = 3.0
L = {}
for s, p in zip(segs, PW):
    L.setdefault(s["key"], []).append(dict(p=p, y=(target(s) > THR).astype(int)))
allP = np.concatenate([w["p"] for v in L.values() for w in v]); GCUT = float(np.quantile(allP, .80))
med = np.array([np.median(np.concatenate([w["p"] for w in v])) for v in L.values()])
frac = np.array([np.mean(np.concatenate([w["p"] for w in v]) >= GCUT) for v in L.values()])
neg = np.array([np.mean(np.concatenate([w["p"] for w in v])[np.concatenate([w["y"] for w in v]) == 0] >= GCUT)
                for v in L.values() if (np.concatenate([w["y"] for w in v]) == 0).any()])
OUT["spread"] = dict(global_cut80=GCUT, record_median_p5=float(np.percentile(med, 5)), record_median_p95=float(np.percentile(med, 95)),
                     record_median_med=float(np.median(med)), frac_above_p5=float(np.percentile(frac, 5)),
                     frac_above_p95=float(np.percentile(frac, 95)), records_over_half=float(np.mean(frac > .5)),
                     stable_alarm_median=float(np.median(neg)), records_stable_alarm_over_half=float(np.mean(neg > .5)))
print(f"\nSPREAD (n>3): per-record median score {np.percentile(med,5):.2f}-{np.percentile(med,95):.2f} (5-95th); global 80th cut {GCUT:.2f}; "
      f"alarm on stable windows median {100*np.median(neg):.0f}%, {100*np.mean(neg>.5):.0f}% of records >half")


def run(strategy, **kw):
    SE, SP_, PV, NV, specs = [], [], [], [], []
    for k, v in L.items():
        tp = fn = fp = tn = 0; hp, hy = [], []
        for w in v:
            cut = strategy(hp, hy, **kw); pred = w["p"] >= cut
            tp += int((pred & (w["y"] == 1)).sum()); fn += int((~pred & (w["y"] == 1)).sum())
            fp += int((pred & (w["y"] == 0)).sum()); tn += int((~pred & (w["y"] == 0)).sum())
            hp.append(w["p"]); hy.append(w["y"])
        if tp + fn: SE.append(tp / (tp + fn))
        if tn + fp: SP_.append(tn / (tn + fp)); specs.append(tn / (tn + fp))
        if tp + fp: PV.append(tp / (tp + fp))
        if tn + fn: NV.append(tn / (tn + fn))
    return dict(sens=float(np.mean(SE)), spec=float(np.mean(SP_)), ppv=float(np.mean(PV)), npv=float(np.mean(NV)),
                balanced=float((np.mean(SE) + np.mean(SP_)) / 2), spec_p10=float(np.percentile(specs, 10)),
                records_spec_below_50=float(np.mean(np.array(specs) < .5)))


def s_global(hp, hy, q=80): return float(np.quantile(allP, q / 100.0))
def s_pct(hp, hy, q=80, minw=40):
    if not hp: return GCUT
    a = np.concatenate(hp); return GCUT if len(a) < minw else float(np.quantile(a, q / 100.0))
def s_youden(hp, hy, minw=40):
    if not hp: return GCUT
    a, b = np.concatenate(hp), np.concatenate(hy)
    if len(a) < minw or b.min() == b.max(): return s_pct(hp, hy)
    return youden_cut(a, b)
def s_stable(hp, hy, kk=1.5, minw=40):
    if not hp: return GCUT
    a, b = np.concatenate(hp), np.concatenate(hy); st_ = a[b == 0]
    return GCUT if len(st_) < minw else float(np.mean(st_) + kk * np.std(st_))
OUT["thresholds"] = {}
print("\nPER-RECORD THRESHOLDS (n>3), per-record means")
for nm, fn_, kw in ([(f"global_p{q}", s_global, dict(q=q)) for q in (60, 70, 80, 85, 90)] +
                    [(f"own_past_p{q}", s_pct, dict(q=q)) for q in (60, 70, 80, 85, 90)] +
                    [("own_youden", s_youden, {})] + [(f"stable_mean_plus_{k}sd", s_stable, dict(kk=k)) for k in (1.0, 1.5, 2.0)]):
    r = run(fn_, **kw); OUT["thresholds"][nm] = r
    print(f"  {nm:26s} sens {100*r['sens']:5.1f}%  spec {100*r['spec']:5.1f}%  PPV {100*r['ppv']:5.1f}%  balanced {100*r['balanced']:5.1f}%  "
          f"| spec 10th-pct {100*r['spec_p10']:.0f}%  records spec<50% {100*r['records_spec_below_50']:.1f}%")
json.dump(OUT, open(f"{RES}/mimic_analysis.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s] -> results/mimic_analysis.json, results/mimic_scores.npz")
