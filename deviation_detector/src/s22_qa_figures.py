"""Step 22. Figures for the audience Q&A deck (run with VARIANT=gated).
  q1_prevalence     cumulative vs instantaneous window prevalence, and the share of stretches with an event
  q2_per_patient    per-patient / per-record AUC distributions, and the record-length caveat
  q3_extra_inputs   demographics and arterial pressure appended to the model (step 21)
  q4_read_median    the calibration read reduced to its median (step 21)
  q5_cardiac_output what VitalDB has for CO / CI, and how smooth the label is
Writes results/qa.json with every number the slides use.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
from config import RES, RES_BASE, DATA, THRS, VITALDB_TILES
from lib.build import VitalDB, build_vitaldb, load_mimic, target
from lib import style as C
K = [f"n>{t:g}" for t in THRS]; COL = {2.0: C.GREEN, 3.0: C.BLUE, 4.0: C.ORANGE}
Q = {}

def qa_figs():
  global Q
# ---- q1 ------------------------------------------------------------------------------------------------
  V = VitalDB(); R = build_vitaldb(V); VS = [s for c in sorted(R) for s in R[c]]
  MI = load_mimic("mimic3wdb"); MS = [s for k in sorted(MI) for s in MI[k]]
  def prev(segs):
      d = {}
      for t in THRS:
          cum = np.concatenate([(target(s) > t).astype(int) for s in segs]); inst = np.concatenate([(s["dev"] > t).astype(int) for s in segs])
          d[f"n>{t:g}"] = dict(cumulative=float(cum.mean()), instantaneous=float(inst.mean()), stretches_with_event=float(np.mean([(s["dev"] > t).any() for s in segs])))
      return d
  Q["prevalence"] = dict(vitaldb=dict(stretches=len(VS), windows=int(sum(len(s["dev"]) for s in VS)), **prev(VS)), mimic=dict(stretches=len(MS), windows=int(sum(len(s["dev"]) for s in MS)), **prev(MS)))
  fig, AX = plt.subplots(1, 3, figsize=(17, 5.0))
  for j, (key, lab) in enumerate((("cumulative", "windows positive (cumulative label: beyond n at any point so far in the stretch)"), ("instantaneous", "windows beyond n at that moment"), ("stretches_with_event", "stretches containing at least one window beyond n"))):
      ax = AX[j]; x = np.arange(3)
      for i, coh in enumerate(("vitaldb", "mimic")):
          unit = "stretches" if key == "stretches_with_event" else "windows"
          v = [100 * Q["prevalence"][coh][k][key] for k in K]; ax.bar(x + (i - .5) * .36, v, .34, color=[C.NAVY, C.ORANGE][i], label=["VitalDB (%s %s)" % (format(Q["prevalence"]["vitaldb"][unit], ","), unit), "MIMIC-III, gated (%s %s)" % (format(Q["prevalence"]["mimic"][unit], ","), unit)][i])
          for xx, vv in zip(x + (i - .5) * .36, v): ax.text(xx, vv + .6, f"{vv:.1f}%", ha="center", fontsize=9)
      ax.set_xticks(x); ax.set_xticklabels([f"|drift| {k[1:]} mmHg" for k in K]); ax.set_ylabel("%"); ax.set_title(lab, fontsize=10.5, color=C.INK)
      if j in (0, 2): ax.legend(fontsize=9)
  fig.suptitle("Q1. Prevalence of the drift label: the label is cumulative within a stretch, so window prevalence is higher than the instantaneous one", fontsize=12.5, fontweight="bold", color=C.INK)
  fig.tight_layout(rect=[0, 0, 1, .92]); C.save(fig, "q1_prevalence")

  # ---- q2 ------------------------------------------------------------------------------------------------
  S = np.load(f"{RES}/investigation_scores.npz", allow_pickle=True); G = S["G"]; P = S["vd__full"]; Yv = {2.0: S["Y2"], 3.0: S["Y3"], 4.0: S["Y4"]}
  Z = np.load(f"{RES}/mimic_scores.npz", allow_pickle=True); PM, TM, RM = Z["P"], Z["TRUE"], Z["REC"]
  nstr = {c: len(R[c]) for c in R}
  Q["per_patient"] = {}
  fig, AX = plt.subplots(1, 3, figsize=(17, 5.2))
  for coh, (ax, PP, GG, lab_fn) in (("vitaldb", (AX[0], P, G, lambda t: Yv[t])), ("mimic", (AX[1], PM, RM, lambda t: (TM > t).astype(int)))):
      Q["per_patient"][coh] = {}
      for t in THRS:
          yy = lab_fn(t); a = []
          for c in np.unique(GG):
              m = GG == c
              if yy[m].min() != yy[m].max(): a.append(roc_auc_score(yy[m], PP[m]))
          a = np.array(a); Q["per_patient"][coh][f"n>{t:g}"] = dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), gt05=float(np.mean(a > .5)), gt07=float(np.mean(a > .7)), pooled=float(roc_auc_score(yy, PP)))
          ax.plot(np.sort(a), np.linspace(0, 100, len(a)), "-", color=COL[t], lw=2.4, label=f"n>{t:g}: mean {a.mean():.3f}, median {np.median(a):.3f}, n={len(a)}  (pooled {roc_auc_score(yy, PP):.3f})")
      ax.axvline(.5, color=C.INK, lw=1.2, ls="--"); ax.set_xlim(.2, 1.0); ax.set_xlabel("AUC inside one patient / record"); ax.set_ylabel("% of patients below"); ax.legend(fontsize=8.5, loc="upper left")
      ax.set_title("VitalDB, out of fold, per patient" if coh == "vitaldb" else "MIMIC-III, zero-shot, per record", fontsize=11, color=C.INK)
  ax = AX[2]; a2 = {}; 
  for c in np.unique(G):
      m = G == c
      if Yv[2.0][m].min() != Yv[2.0][m].max(): a2[c] = roc_auc_score(Yv[2.0][m], P[m])
  xs = np.array([nstr[c] for c in a2]); ys = np.array(list(a2.values()))
  from scipy.stats import spearmanr
  rho = spearmanr(xs, ys).correlation; Q["per_patient"]["vitaldb"]["rho_auc_vs_nstretch"] = float(rho)
  ax.plot(xs + np.random.default_rng(0).uniform(-.15, .15, len(xs)), ys, "o", color=C.NAVY, ms=4.5, alpha=.55); ax.axhline(.5, color=C.INK, lw=1, ls="--")
  for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 30)):
      m = (xs >= lo) & (xs < hi)
      if m.sum(): ax.plot([lo, hi - .2], [ys[m].mean()] * 2, "-", color=C.RED, lw=3)
  ax.set_xscale("log"); ax.set_xlabel("stretches the patient contributes (log)"); ax.set_ylabel("per-patient AUC, n>2"); ax.set_title(f"VitalDB: per-patient AUC vs record length, Spearman {rho:+.2f}", fontsize=11, color=C.INK)
  fig.suptitle("Q2. Per patient rather than pooled: the distributions, and why short records look easy", fontsize=12.5, fontweight="bold", color=C.INK)
  fig.tight_layout(rect=[0, 0, 1, .92]); C.save(fig, "q2_per_patient")

  # ---- q3 / q4 -----------------------------------------------------------------------------------------------
  E = json.load(open(f"{RES}/extra_inputs.json")); Q["extra_inputs"] = E
  CLK = json.load(open(f"{RES}/nested_clock_W10_S15_seed2024.json"))["results"]; Q["clock"] = {k: CLK[k]["auc"] for k in K}
  SHORT = {"full model (123)": "full model\n(123)", "+ demographics (age, sex, height, weight, BMI)": "+ demographics\n(age, sex, height,\nweight, BMI)", "+ arterial pressure (MAP, dMAP, read MAP)": "+ arterial pressure\n(MAP, dMAP,\nread MAP)",
           "+ demographics + arterial pressure": "+ demographics\n+ arterial pressure", "demographics alone": "demographics\nalone", "arterial pressure alone": "arterial pressure\nalone", "arterial pressure + clock": "arterial pressure\n+ clock",
           "read median alone, no model": "read median alone,\nno model", "clock only (control)": "clock only\n(control)", "read median + clock": "read median\n+ clock", "read median + clock + waveform": "read median\n+ clock + waveform",
           "read shape (no median) + clock": "read shape (no median)\n+ clock", "full read columns, no waveform": "full read columns,\nno waveform", "clock + waveform (no read columns)": "clock + waveform\n(no read columns)"}
  def bars(ax, arms, D, ttl, ref=None, reflab=None, ylim=(.4, .9)):
      x = np.arange(len(arms))
      for j, k in enumerate(K):
          v = [D[a][k]["auc"] for a in arms]; lo = [D[a][k]["ci"][0] for a in arms]; hi = [D[a][k]["ci"][1] for a in arms]
          ax.bar(x + (j - 1) * .27, v, .25, color=COL[THRS[j]], label=k); ax.errorbar(x + (j - 1) * .27, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=1, capsize=2)
          for xx, vv, hh in zip(x + (j - 1) * .27, v, hi): ax.text(xx, hh + .006, f"{vv:.3f}", ha="center", fontsize=7.5, rotation=90, va="bottom")
      if ref is not None:
          for j, k in enumerate(K): ax.axhline(ref[k]["auc"], color=COL[THRS[j]], lw=1.2, ls=":")
      ax.set_xticks(x); ax.set_xticklabels([SHORT.get(a, a.replace(" (", "\n(")) for a in arms], fontsize=8.8); ax.set_ylim(ylim[0], max(ylim[1], max(D[a][k]["ci"][1] for a in arms for k in K) + .05)); ax.set_ylabel("AUC, VitalDB out of fold"); ax.set_title(ttl, fontsize=11, color=C.INK); ax.legend(fontsize=8.5, loc="lower right")
  fig, AX = plt.subplots(1, 2, figsize=(17, 6.6), width_ratios=[2.2, 1])
  D = dict(E["A"]); D["full model (123)"] = E["full"]
  bars(AX[0], ["full model (123)", "+ demographics (age, sex, height, weight, BMI)", "+ arterial pressure (MAP, dMAP, read MAP)", "+ demographics + arterial pressure", "demographics alone", "arterial pressure alone", "arterial pressure + clock"], D,
       "Appended to the 123 columns (left three) and on their own (right three); dotted = full model", ref=E["full"])
  ax = AX[1]; M = E["A_mimic"]; x = np.arange(2)
  for j, k in enumerate(K):
      v = [M["full"][k]["auc"], M["plus_map"][k]["auc"]]; ax.bar(x + (j - 1) * .27, v, .25, color=COL[THRS[j]], label=k)
      for xx, vv in zip(x + (j - 1) * .27, v): ax.text(xx, vv + .006, f"{vv:.3f}", ha="center", fontsize=8.5)
  ax.set_xticks(x); ax.set_xticklabels(["full model", "+ MAP at the window,\nMAP vs stretch start"]); ax.set_ylim(.6, .9); ax.set_ylabel("AUC, MIMIC-III zero-shot"); ax.legend(fontsize=8.5)
  ax.set_title(f"MIMIC-III, {M['records']} art-line records, {M['windows']:,} windows", fontsize=11, color=C.INK)
  fig.suptitle("Q3. Demographics and arterial pressure as inputs: neither moves the result", fontsize=12.5, fontweight="bold", color=C.INK)
  fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "q3_extra_inputs")
  fig, ax = plt.subplots(figsize=(17, 6.4)); D = dict(E["B"]); D["full model (123)"] = E["full"]
  D["clock only (control)"] = {k: dict(auc=CLK[k]["auc"], ci=CLK[k]["ci"]) for k in K}
  INV = json.load(open(f"{RES}/investigation.json")); D["read median alone, no model"] = {k: dict(auc=INV["baselines"]["absolute CVP level: block median"][k]["auc"], ci=INV["baselines"]["absolute CVP level: block median"][k]["ci"]) for k in K}
  bars(ax, ["read median alone, no model", "clock only (control)", "read median + clock", "read median + clock + waveform", "read shape (no median) + clock", "full read columns, no waveform", "clock + waveform (no read columns)", "full model (123)"], D,
       "The calibration read reduced to its median, against the read's shape and the waveform", ylim=(.45, .85))
  fig.suptitle("Q4. Is the read's contribution its level or its variability? The median adds nothing to the clock; the shape carries the read's share", fontsize=12.5, fontweight="bold", color=C.INK)
  fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "q4_read_median")

  # ---- q5 ------------------------------------------------------------------------------------------------
  T = np.load(VITALDB_TILES, allow_pickle=True); Kt = {str(c): i for i, c in enumerate(T["cols"])}; M = T["data"]
  cid = M[:, Kt["cid"]].astype(int); tmn = M[:, Kt["t_sec"]] / 60.0
  cov = {}
  for nm in ("co", "ci", "sv", "svv"):
      v = M[:, Kt[nm]]; per = {c: int(np.isfinite(v[cid == c]).sum()) for c in np.unique(cid)}
      cov[nm] = dict(tiles_finite=float(np.isfinite(v).mean()), patients_ge20=int(sum(1 for n in per.values() if n >= 20)), patients_ge120=int(sum(1 for n in per.values() if n >= 120)), median=float(np.nanmedian(v)))
  Q["cardiac_output"] = dict(coverage=cov, patients_total=int(len(np.unique(cid))))
  co = M[:, Kt["co"]]
  cand = [c for c in V.ORDER if np.isfinite(co[cid == c]).sum() >= 300 and len(V.ORDER[c]) >= 60]
  best = max(cand, key=lambda c: len(V.ORDER[c]))
  m = (cid == best); o = np.argsort(tmn[m]); u = tmn[m][o]; y_co = co[m][o]
  og = V.ORDER[best]; tg, yg = V.T[og], V.Y[og]; Q["cardiac_output"]["example_case"] = int(best)
  fig = plt.figure(figsize=(17, 6.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=.35, wspace=.22)
  ax = fig.add_subplot(gs[0, 0]); g = np.isfinite(y_co); ax.plot(u[g], y_co[g], "-", color=C.ACC, lw=1.2); ax.set_ylabel("cardiac output (L/min), EV1000"); ax.set_title(f"VitalDB case {best}: the CO series is smooth and updates every 20 s", fontsize=10.5, color=C.INK); ax.tick_params(labelbottom=False)
  ax = fig.add_subplot(gs[1, 0], sharex=ax); g = np.isfinite(yg); ax.plot(tg[g], yg[g], "-o", color=C.CVPC, lw=.9, ms=2.5, mew=0); ax.set_ylabel("CVP label (mmHg)"); ax.set_xlabel("minutes into the case"); ax.set_title("the same case's CVP label: window-to-window variability\nis a large part of what the detector reads", fontsize=10.5, color=C.INK)
  ax = fig.add_subplot(gs[:, 1]); names = list(cov); x = np.arange(len(names))
  ax.bar(x - .2, [cov[n]["patients_ge20"] for n in names], .38, color=C.ACC, label=">= 20 readings (one full read)"); ax.bar(x + .2, [cov[n]["patients_ge120"] for n in names], .38, color="#c9b7d6", label=">= 120 readings (about 1 h)")
  for i, n in enumerate(names): ax.text(x[i] - .2, cov[n]["patients_ge20"] + 5, str(cov[n]["patients_ge20"]), ha="center", fontsize=9); ax.text(x[i] + .2, cov[n]["patients_ge120"] + 5, str(cov[n]["patients_ge120"]), ha="center", fontsize=9)
  ax.set_xticks(x); ax.set_xticklabels(["cardiac\noutput", "cardiac\nindex", "stroke\nvolume", "stroke-volume\nvariation"], fontsize=9); ax.set_ylabel(f"VitalDB patients (of {len(np.unique(cid))} with a CVP label)"); ax.legend(fontsize=9); ax.set_title(f"EV1000 / Vigileo channels: finite on {100*cov['co']['tiles_finite']:.0f}% of 30-s tiles", fontsize=10.5, color=C.INK)
  fig.suptitle("Q5. Cardiac output or index: the data exist for a pilot on VitalDB (none on MIMIC); the label is pulse-contour-derived and smooth", fontsize=12.5, fontweight="bold", color=C.INK)
  C.save(fig, "q5_cardiac_output")
  json.dump(Q, open(f"{RES}/qa.json", "w"), indent=1); print("-> results/qa.json")


if PART in ("all", "Q"): qa_figs()

# ===================================================================================================
def cardiac_output_figs():
    """q5a: arms per label on VitalDB (CO, CI, SV vs CVP on the same patients); q5b: VitalDB CO example stretches;
    q5c: MIMIC CO zero-shot arms; q5d: MIMIC CO example stretches; q5e: every arm with the stretches whose last
    calibration reading was already outside the band removed (step 26)."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    J = json.load(open(f"{RES}/cardiac_output.json")); TRc = np.load(f"{RES}/cardiac_output_traces.npz", allow_pickle=True)
    fig, AX = plt.subplots(1, 4, figsize=(17, 6.6), sharey=True)
    arms = ["full", "read columns only (no waveform)", "waveform + clock", "clock only", "permuted targets", "read spread x since"]; short = ["full", "read cols\nonly", "waveform\n+ clock", "clock\nonly", "permuted\ntargets", "read\nspread\nx time\n(no model)"]
    for ax, k in zip(AX, ("co", "ci", "sv", "cvp")):
        d = J[k]; ths = d["thresholds"]; x = np.arange(len(arms))
        for j, t in enumerate(ths):
            kk = f"n>{t:g}"; D = {**d["arms"], **d["baselines"]}; v = [D[a][kk]["auc"] for a in arms]; lo = [D[a][kk]["ci"][0] for a in arms]; hi = [D[a][kk]["ci"][1] for a in arms]
            ax.bar(x + (j - 1) * .27, v, .25, color=[C.GREEN, C.BLUE, C.ORANGE][j], label=f"> {t:g} {d['unit']} (prev {100*d['prevalence'][kk]:.0f}%)")
            ax.errorbar(x + (j - 1) * .27, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8); ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.legend(fontsize=7.5, loc="upper right", title="|drift|", title_fontsize=7.5)
        ax.set_title(f"{d['label']}\n{d['patients']} pts, {d['stretches']} stretches, {d['windows']:,} windows\nmedian level {d['level_median']:.1f} {d['unit']}", fontsize=9.5, color=C.INK)
    AX[0].set_ylabel("AUC, five-fold by patient (same folds)")
    fig.suptitle("Q5. The same detector with a different label on VitalDB: cardiac output, cardiac index, stroke volume, and CVP on the same patients", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .92]); C.save(fig, "q5a_co_vitaldb")

    # ---- example stretches: ranked by peak predicted drift, split by whether the last calibration reading was inside the band
    def load_items(T, prefix, name_fn, thr=0.75):
        keys = [f[:-5] for f in T.files if f.startswith(prefix) and f.endswith("__dev")]; items = []
        for kk in keys:
            g = lambda q: T[f"{kk}__{q}"]; dev = g("dev"); yb = g("yb"); base = float(g("base")[0]); ev = np.maximum.accumulate(dev) > thr; f = np.isfinite(yb)
            items.append(dict(name=name_fn(kk), t=g("t"), y=g("y"), tb=g("tb"), yb=yb, base=base, dev=dev, p=g("p"), ev=ev, pos=bool(ev.any()), head=bool(f.any() and abs(yb[f][-1] - base) > thr)))
        pos = sorted([s for s in items if s["pos"]], key=lambda s: -s["p"].max())
        return [s for s in pos if not s["head"]], [s for s in pos if s["head"]], [s for s in items if not s["pos"]]
    def draw_ex(rows, name, ttl, series, xlabel, thr=0.75, unit="L/min"):
        n = len(rows); Hh = 3.0 * n + 1.7; fig = plt.figure(figsize=(16.5, Hh)); gs = GridSpec(2 * n, 1, hspace=.3, height_ratios=[2.5, .85] * n, top=1 - 1.35 / Hh, bottom=.5 / Hh, figure=fig)
        for i, s in enumerate(rows):
            ax = fig.add_subplot(gs[2 * i]); axp = fig.add_subplot(gs[2 * i + 1])
            ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0); axp.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0)
            ax.plot(s["tb"], s["yb"], "s-", color=C.ACC, ms=3.4, lw=.9, mew=.6, mec="#fff", alpha=.8); ax.fill_between([s["tb"][0], s["t"][-1]], s["base"] - thr, s["base"] + thr, color=C.GREY, alpha=.16, lw=0); ax.axhline(s["base"], color=C.GREY, lw=2.2)
            ax.plot(s["t"], s["y"], "-o", color=C.ACC, lw=1.4, ms=4, mew=0); bad = s["dev"] > thr
            if bad.any(): ax.plot(s["t"][bad], s["y"][bad], "o", color=C.RED, ms=7, mew=0, zorder=5); k0 = int(np.argmax(s["ev"])); [a.axvline(s["t"][k0], color=C.RED, lw=1.6, ls="--") for a in (ax, axp)]
            axp.fill_between(s["t"], 0, s["p"], color=C.GREEN, alpha=.35, lw=0); axp.plot(s["t"], s["p"], "-", color=C.GREEN, lw=2.2); axp.set_ylabel(f"predicted\ndrift ({unit})", fontsize=8.5)
            lo, hi = min(s["tb"][0], s["t"][0]) - 2, s["t"][-1] + 2; ax.set_xlim(lo, hi); axp.set_xlim(lo, hi); ax.tick_params(labelsize=9, labelbottom=False); axp.tick_params(labelsize=8); ax.set_ylabel(f"CO ({unit})", fontsize=9.5)
            if i == n - 1: axp.set_xlabel(xlabel, fontsize=9)
            status = "DEVIATES, already off at the end of calibration" if s["head"] else ("DEVIATES, last calibration reading inside the band" if s["pos"] else f"stays within {thr:g} {unit}")
            ax.set_title(f"{s['name']}   max drift {s['dev'].max():.2f} {unit}   {status}   peak predicted {s['p'].max():.2f}", fontsize=10.5, fontweight="bold", color="#8c2d1e" if s["pos"] else "#12633f", pad=2)
        fig.legend([Line2D([], [], color=C.ACC, marker="o", lw=1.4, ms=6), Line2D([], [], color=C.GREY, lw=2.2), Line2D([], [], color=C.RED, marker="o", lw=0, ms=7), Line2D([], [], color=C.RED, lw=1.6, ls="--"), Line2D([], [], color=C.GREEN, lw=2.2)],
                   [series, f"calibration baseline (+/-{thr:g} {unit} band)", f">{thr:g} {unit} off", "first true crossing", "PREDICTED drift (causal running max)"], fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1 - .62 / Hh), frameon=False)
        fig.suptitle(ttl, fontsize=12.5, fontweight="bold", color=C.INK, y=1 - .06 / Hh); C.save(fig, name)
    nh, hh, neg = load_items(TRc, "co:", lambda kk: "case " + kk.split(":")[1])
    draw_ex(nh[:2] + hh[:1], "q5b_co_caught", "Cardiac output on VitalDB, out of fold, |drift| > 0.75 L/min:\nthe two highest-ranked deviations whose last calibration reading was inside the band, and the highest-ranked one already off at the end of calibration", "measured CO (EV1000)", "minutes into the case")
    draw_ex(nh[-3:], "q5b_co_missed", "Cardiac output on VitalDB, out of fold, |drift| > 0.75 L/min:\nthe three lowest-ranked deviations whose last calibration reading was inside the band", "measured CO (EV1000)", "minutes into the case")

    # ---- q5c / q5d: MIMIC
    f = f"{RES}/cardiac_output_mimic.json"
    if os.path.exists(f):
        JM = json.load(open(f)); TRm = np.load(f"{RES}/cardiac_output_mimic_traces.npz", allow_pickle=True)
        fig, ax = plt.subplots(figsize=(17, 5.6)); arms = [a for a in list(JM["arms"]) + list(JM["baselines"]) if "absolute-error" not in a]; x = np.arange(len(arms)); ths = JM["thresholds"]
        for j, t in enumerate(ths):
            kk = f"n>{t:g}"; D = {**JM["arms"], **JM["baselines"]}; v = [D[a][kk]["auc"] for a in arms]; lo = [D[a][kk]["ci"][0] for a in arms]; hi = [D[a][kk]["ci"][1] for a in arms]
            ax.bar(x + (j - 1) * .27, v, .25, color=[C.GREEN, C.BLUE, C.ORANGE][j], label=f"|drift| > {t:g} L/min  (prev {100*JM['prevalence'][kk]:.0f}%)"); ax.errorbar(x + (j - 1) * .27, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2)
            for xx, vv, hh_ in zip(x + (j - 1) * .27, v, hi): ax.text(xx, hh_ + .006, f"{vv:.3f}", ha="center", fontsize=7.5, rotation=90, va="bottom")
        ax.set_xticks(x); ax.set_xticklabels([a.replace(": ", ":\n").replace(", ", ",\n").replace(" (", "\n(") for a in arms], fontsize=8.8); ax.set_ylim(.4, 1.0); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.set_ylabel("AUC (record-clustered 95% CI)"); ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"MIMIC-III records with a continuous cardiac-output numeric: {JM['records']} records, {JM['stretches']} stretches, {JM['windows']:,} windows", fontsize=11, color=C.INK)
        fig.suptitle("Q5, continued. Cardiac output on the ICU records: zero-shot from the VitalDB CO model, and a MIMIC-only ceiling", fontsize=12.5, fontweight="bold", color=C.INK)
        fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "q5c_co_mimic")
        nh, hh, neg = load_items(TRm, "", lambda kk: kk.replace("mimic3wdb_", "MIMIC-III ").rsplit(":", 1)[0])
        draw_ex(nh[:2] + hh[:1], "q5d_co_mimic_caught", "Cardiac output on MIMIC-III, zero-shot, |drift| > 0.75 L/min:\nthe two highest-ranked deviations whose last calibration reading was inside the band, and the highest-ranked one already off at the end of calibration", "continuous cardiac output (PA-catheter numeric)", "minutes into the record")
        draw_ex(nh[-2:] + neg[:1], "q5d_co_mimic_missed", "Cardiac output on MIMIC-III, zero-shot, |drift| > 0.75 L/min:\nthe two lowest-ranked deviations whose last calibration reading was inside the band, and a quiet stretch", "continuous cardiac output (PA-catheter numeric)", "minutes into the record")

    # ---- q5e: the head-start control (step 26)
    f = f"{RES}/cardiac_output_headstart.json"
    if not os.path.exists(f): return
    HS = json.load(open(f))
    panels = [("co", "VitalDB cardiac output, out of fold", [("full", "full"), ("read columns only (no waveform)", "read cols\nonly"), ("waveform + clock", "waveform\n+ clock"), ("clock only", "clock\nonly"), ("base:read spread x since", "read spread\nx time\n(no model)"), ("base:read |last - median|", "read\n|last - median|\n(no model)")]),
              ("cvp", "CVP on the same VitalDB patients", [("full", "full"), ("read columns only (no waveform)", "read cols\nonly"), ("waveform + clock", "waveform\n+ clock"), ("clock only", "clock\nonly")])]
    if "mimic" in HS: panels.insert(1, ("mimic", "MIMIC-III cardiac output, zero-shot", [("zero-shot from VitalDB CO, full", "zero-shot\nfull"), ("zero-shot from VitalDB CO, read columns only", "zero-shot\nread cols\nonly"), ("zero-shot, waveform + clock", "zero-shot\nwaveform\n+ clock"), ("MIMIC-only ceiling (5-fold by record, squared-error loss)", "MIMIC-only\nceiling"), ("no model: read spread x since", "read spread\nx time\n(no model)"), ("no model: read |last - median|", "read\n|last - median|\n(no model)"), ("no model: clock", "clock\n(no model)")]))
    fig, AX = plt.subplots(1, len(panels), figsize=(18, 6.8), sharey=True, gridspec_kw=dict(width_ratios=[len(p[2]) for p in panels]))
    for ax, (k, ttl, arms) in zip(np.atleast_1d(AX), panels):
        d = HS[k]; ths = d["thresholds"]; x = np.arange(len(arms))
        for j, t in enumerate(ths):
            kk = f"n>{t:g}"; c = d["counts"][kk]; v = [d["no_head_start"][a][kk]["auc"] for a, _ in arms]; lo = [d["no_head_start"][a][kk]["ci"][0] for a, _ in arms]; hi = [d["no_head_start"][a][kk]["ci"][1] for a, _ in arms]; va = [d["all"][a][kk] for a, _ in arms]
            ax.bar(x + (j - 1) * .27, v, .25, color=[C.GREEN, C.BLUE, C.ORANGE][j], label=f"> {t:g} {d['unit']}: {c['head_start_stretches']} of {c['positive_stretches']} positive stretches removed")
            ax.errorbar(x + (j - 1) * .27, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2); ax.plot(x + (j - 1) * .27, va, "_", color=C.INK, ms=10, mew=2, zorder=6)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms], fontsize=8); ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":")
        h, l = ax.get_legend_handles_labels(); ax.legend(h + [Line2D([], [], color=C.INK, marker="_", lw=0, ms=10, mew=2)], l + ["AUC with every stretch kept"], fontsize=7.3, loc="upper right")
        c = d["counts"][f"n>{ths[1]:g}"]; ax.set_title(f"{ttl}\nat {ths[1]:g} {d['unit']}: {c['windows_kept']:,} of {c['windows_all']:,} windows kept, prevalence {100*c['prevalence_all']:.1f}% -> {100*c['prevalence_kept']:.1f}%", fontsize=9.5, color=C.INK)
    np.atleast_1d(AX)[0].set_ylabel("AUC on the stretches whose last calibration reading was inside the band")
    fig.suptitle("Q5, control. Bars: AUC after removing the stretches that were already outside the band when the read ended. Ticks: AUC with every stretch kept", fontsize=12, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "q5e_co_head_start")


if PART in ("all", "CO"): cardiac_output_figs()
