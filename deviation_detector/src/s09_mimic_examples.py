"""Step 9. The system running on MIMIC ICU records, drawn - zero-shot, trained on VitalDB only.

  mimic_best / mimic_typical / mimic_worst   four records each over ~4 h: the repeating
      calibration / tracking cycle, the measured CVP, the baseline in force, the windows truly
      beyond n, and the causal risk score underneath
  mimic_direction   the SIGNED model on three true rises and three true falls
  mimic_confusion   confusion matrices at the Youden point, magnitude calibration, direction AUC
  mimic_per_record  the distribution of per-record AUC
Reads results/mimic_scores.npz and results/mimic_analysis.json from step 8.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib.build import load_mimic, target
from lib import style as C
THR = float(os.environ.get("THR", "3"))

Z = np.load(f"{RES}/mimic_scores.npz", allow_pickle=True)
J = json.load(open(f"{RES}/mimic_analysis.json"))
MI = load_mimic("mimic3wdb")
# attach the saved scores to the rebuilt stretches (same order as step 8)
segs = [s for k in sorted(MI) for s in MI[k]]
o = 0
for s in segs:
    n = len(s["dev"]); s["p"] = Z["P"][o:o + n]; s["ps"] = Z["PS"][o:o + n]; o += n
    s["ev"] = target(s) > THR
assert o == len(Z["P"])
ok = {k.replace("mimic3wdb_", ""): v for k, v in J["per_record"][f"n>{THR:g}"]["per_record"].items()}
REC = {k.replace("mimic3wdb_", ""): v for k, v in MI.items()}
from config import VARIANT
# in the variants, show records with >= 8 stretches and >= 2 event stretches, so a 'best' record is not a two-stretch record with one event
pool = [k for k in ok if not VARIANT or (len(REC[k]) >= 8 and sum(int(s["ev"].any()) for s in REC[k]) >= 2)]
srt = sorted(pool, key=lambda k: -ok[k])
q80 = float(np.quantile(Z["P"], .80))


from lib.plots import draw_records


def draw(keys, name, ttl, sub, hours=4.0):
    return draw_records(REC, ok, q80, keys, name, ttl, sub, THR=THR, hours=hours)


med = srt[len(srt) // 2 - 2: len(srt) // 2 + 2]
json.dump({"best": [(k, ok[k], sum(int(s["ev"].any()) for s in REC[k]), len(REC[k])) for k in srt[:4]],
           "typical": [(k, ok[k], sum(int(s["ev"].any()) for s in REC[k]), len(REC[k])) for k in med],
           "worst": [(k, ok[k], sum(int(s["ev"].any()) for s in REC[k]), len(REC[k])) for k in srt[-4:]], "thr": THR, "q80": q80},
          open(f"{RES}/mimic_examples.json", "w"), indent=1)
draw(srt[:2], "mimic_best_a", f"MIMIC ICU, zero-shot from VitalDB: the two best records (n>{THR:g})",
     "Blue bands are 10-min CVP calibration; between them the model runs on ECG/PPG alone. Risk is the causal running maximum: nothing after the current minute is used.")
draw(srt[2:4], "mimic_best_b", "MIMIC ICU, zero-shot: best records 3 and 4", "Same layout.")
draw(med[:2], "mimic_typical_a", "MIMIC ICU, zero-shot: two TYPICAL records (median per-record AUC)", "Same layout.")
draw(med[2:], "mimic_typical_b", "MIMIC ICU, zero-shot: two more typical records", "Same layout.")
draw(srt[-4:-2], "mimic_worst_a", "MIMIC ICU, zero-shot: the worst records (1-2)", "Same layout. These are what fails.")
draw(srt[-2:], "mimic_worst_b", "MIMIC ICU, zero-shot: the worst records (3-4)", "Same layout.")
draw(srt[:4], "mimic_best", f"MIMIC ICU, zero-shot from VitalDB: four records where it works best (n>{THR:g})",
     "Blue bands are 10-min CVP calibration; between them the model runs on ECG/PPG alone. Risk is the causal running maximum: nothing after the current minute is used.")
draw(med, "mimic_typical", "MIMIC ICU, zero-shot: four TYPICAL records (median per-record AUC)", "Same layout.")
draw(srt[-4:], "mimic_worst", "MIMIC ICU, zero-shot: the four worst records", "Same layout. These are what fails.")

# ---- direction examples ----------------------------------------------------------------------
for s in segs:
    s["prm"] = np.cumsum(s["ps"]) / np.arange(1, len(s["ps"]) + 1)
ALL = [s for s in segs if np.abs(s["sdev"]).max() > THR]
UP = sorted([s for s in ALL if s["sdev"][np.argmax(np.abs(s["sdev"]))] > 0], key=lambda s: -s["prm"][-1])
DN = sorted([s for s in ALL if s["sdev"][np.argmax(np.abs(s["sdev"]))] < 0], key=lambda s: s["prm"][-1])
rows = UP[:3] + DN[:3]; n = len(rows)
fig = plt.figure(figsize=(16, 3.1 * n + 1.4))
gs = GridSpec(2 * n, 1, hspace=.34, height_ratios=[2.4, .95] * n, top=1 - 1.1 / (3.1 * n + 1.4), figure=fig)
for i, s in enumerate(rows):
    ax = fig.add_subplot(gs[2 * i]); axp = fig.add_subplot(gs[2 * i + 1])
    ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0); axp.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0)
    ax.plot(s["tb"], s["yb"], "s-", color=C.CVPC, ms=3.2, lw=.9, mew=.6, mec="#fff", alpha=.8)
    ax.axhline(s["base"], color=C.GREY, lw=2.2); ax.plot(s["t"], s["y"], "-o", color=C.CVPC, lw=1.3, ms=4, mew=0)
    big = np.abs(s["sdev"]) > THR
    if big.any(): ax.plot(s["t"][big], s["y"][big], "o", color=C.RED, ms=7, mew=0, zorder=5)
    pos = s["prm"] >= 0
    axp.fill_between(s["t"], 0, s["prm"], where=pos, color=C.RED, alpha=.4, lw=0, interpolate=True)
    axp.fill_between(s["t"], 0, s["prm"], where=~pos, color=C.BLUE, alpha=.4, lw=0, interpolate=True)
    axp.plot(s["t"], s["prm"], "-", color=C.INK, lw=2); axp.axhline(0, color=C.INK, lw=1.4)
    lim = max(1.4, float(np.abs(s["prm"]).max()) + .3); axp.set_ylim(-lim, lim); axp.set_ylabel("predicted\nsigned drift", fontsize=8)
    lo, hi = s["tb"][0] - 2, s["t"][-1] + 2; ax.set_xlim(lo, hi); axp.set_xlim(lo, hi)
    ax.tick_params(labelsize=9, labelbottom=False); axp.tick_params(labelsize=8)
    if i == n - 1: axp.set_xlabel("minutes", fontsize=9)
    ax.set_ylabel("CVP (mmHg)", fontsize=9.5)
    tru = s["sdev"][np.argmax(np.abs(s["sdev"]))]
    ax.set_title(f"{s['key'].replace('mimic3wdb_','')}   true peak drift {tru:+.1f} mmHg ({'RISE' if tru > 0 else 'FALL'})   "
                 f"model says {'UP' if s['prm'][-1] > 0 else 'DOWN'} ({s['prm'][-1]:+.2f})",
                 fontsize=10.5, fontweight="bold", color="#8c2d1e" if tru > 0 else "#1f4e79", pad=3)
fig.legend([Line2D([], [], color=C.CVPC, marker="o", lw=1.3, ms=6), Line2D([], [], color=C.GREY, lw=2.2),
            Line2D([], [], color=C.RED, marker="o", lw=0, ms=7), Line2D([], [], color=C.RED, lw=6, alpha=.5),
            Line2D([], [], color=C.BLUE, lw=6, alpha=.5)],
           ["measured CVP", "calibration baseline", f">|{THR:g}| mmHg off", "model says CVP is UP", "model says CVP is DOWN"],
           fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(.5, .995), frameon=False)
fig.suptitle("The SIGNED model: three true RISES and three true FALLS (zero-shot on MIMIC)", fontsize=13, fontweight="bold", color=C.INK, y=1.0)
C.save(fig, "mimic_direction")

# ---- confusion / calibration / direction summary figure ------------------------------------------
P, TRUE, PS, TS = Z["P"], Z["TRUE"], Z["PS"], Z["SDEV"]
fig = plt.figure(figsize=(17.5, 9.6)); gs = fig.add_gridspec(2, 4, hspace=.42, wspace=.34)
for j, thr in enumerate(THRS):
    ax = fig.add_subplot(gs[0, j]); Y = (TRUE > thr).astype(int)
    d = J["confusion"][f"n>{thr:g}"]["points"]["youden"]
    M = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]]); frac = M / M.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Greens", vmin=0, vmax=1)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, f"{M[a,b]:,}\n{100*frac[a,b]:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="#ffffff" if frac[a, b] > .55 else C.INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["flagged", "not flagged"], fontsize=9.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["truly\ndeviated", "truly\nstable"], fontsize=9.5); ax.grid(False)
    ax.set_title(f"|drift| > {thr:g} mmHg   AUC {roc_auc_score(Y, P):.3f}\nYouden cut {d['cut']:.2f}: sens {100*d['sens']:.0f}%\nspec {100*d['spec']:.0f}%, PPV {100*d['ppv']:.0f}%", fontsize=10, fontweight="bold", color=C.INK)
ax = fig.add_subplot(gs[0, 3]); m = np.abs(TS) > 2.0
Y = (TS[m] > 0).astype(int); pr = (PS[m] > 0).astype(int)
M = np.array([[int(((pr == 1) & (Y == 1)).sum()), int(((pr == 0) & (Y == 1)).sum())], [int(((pr == 1) & (Y == 0)).sum()), int(((pr == 0) & (Y == 0)).sum())]])
frac = M / M.sum(1, keepdims=True); ax.imshow(frac, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
for a in range(2):
    for b in range(2):
        ax.text(b, a, f"{M[a,b]:,}\n{100*frac[a,b]:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="#fff" if frac[a, b] > .55 else C.INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["called UP", "called DOWN"], fontsize=9.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["truly UP", "truly DOWN"], fontsize=9.5)
ax.set_title(f"DIRECTION on real deviations (>2 mmHg)\nper-window AUC {roc_auc_score(Y, PS[m]):.3f}\naccuracy {100*np.mean(pr == Y):.0f}%", fontsize=10, fontweight="bold", color=C.INK)
ax = fig.add_subplot(gs[1, :2]); cal = J["calibration"]["bins"]
ax.plot([0, 16], [0, 16], "-", color=C.INK, lw=2, label="perfect calibration")
ax.plot([b["mean_true"] for b in cal], [b["mean_pred"] for b in cal], "-o", color=C.RED, lw=2.8, ms=10, label="model, binned by true drift")
ax.set_xlabel("TRUE |drift| (mmHg)"); ax.set_ylabel("PREDICTED |drift| (mmHg)"); ax.set_xlim(0, 16); ax.set_ylim(0, 16)
ax.set_title(f"Magnitude is NOT calibrated: the output saturates near {max(b['mean_pred'] for b in cal):.1f} mmHg", fontsize=12, fontweight="bold", color=C.INK)
ax.legend(fontsize=10, frameon=False)
ax = fig.add_subplot(gs[1, 2:])
for nm, thr, col in (("all windows", 0.0, C.GREY), (">1 mmHg", 1.0, C.BLUE), (">2 mmHg", 2.0, C.GREEN), (">3 mmHg", 3.0, C.ORANGE)):
    mm = np.abs(TS) > thr; yy = (TS[mm] > 0).astype(int); au = roc_auc_score(yy, PS[mm])
    ax.bar(nm, au, color=col, width=.6); ax.text(nm, au + .01, f"{au:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.axhline(.5, color=C.INK, lw=1.4, ls="--"); ax.set_ylim(.4, .9); ax.set_ylabel("AUC, rise vs fall (per window)")
ax.set_title("Direction is partly predictable per window; at stretch level it is weaker (see notes)", fontsize=11.5, fontweight="bold", color=C.INK)
fig.suptitle("External validation on MIMIC-III: zero-shot, at the YOUDEN operating point", fontsize=14, fontweight="bold", color=C.INK)
fig.text(.5, .012, f"Youden = the threshold maximising sensitivity + specificity - 1. {len(P):,} windows, {J['n_records']} records, trained on VitalDB only.",
         ha="center", va="top", fontsize=10, color=C.MUTE)
C.save(fig, "mimic_confusion")

# ---- per-record AUC distribution -------------------------------------------------------------------
fig, AX = plt.subplots(1, 3, figsize=(17, 5.2)); COL = {2.0: C.GREEN, 3.0: C.BLUE, 4.0: C.ORANGE}
AU = {thr: J["per_record"][f"n>{thr:g}"]["per_record"] for thr in THRS}
ax = AX[0]
for thr in THRS:
    v = np.array(list(AU[thr].values()))
    ax.plot(np.sort(v), np.linspace(0, 100, len(v)), "-", color=COL[thr], lw=2.6, label=f"n>{thr:g}  mean {v.mean():.3f}  (n={len(v)})")
ax.axvline(.5, color=C.INK, lw=1.4, ls="--"); ax.axvline(.75, color=C.RED, lw=1.6, ls=":")
ax.set_xlabel("per-record AUC"); ax.set_ylabel("% of records below"); ax.set_title("Per-record AUC distribution", fontsize=12, fontweight="bold", color=C.INK)
ax.legend(fontsize=9.5, frameon=False, loc="upper left"); ax.set_xlim(.3, 1.0)
ax = AX[1]; v = np.array(list(AU[2.0].values()))
ax.hist(v, bins=np.arange(.3, 1.02, .04), color=COL[2.0], edgecolor="white"); ax.axvline(.5, color=C.INK, lw=1.6, ls="--"); ax.axvline(v.mean(), color=C.RED, lw=2.2)
ax.set_xlabel("per-record AUC, n>2 mmHg"); ax.set_ylabel("records")
ax.set_title(f"n>2: {100*np.mean(v>0.5):.0f}% of records beat chance, {100*np.mean(v>0.7):.0f}% beat 0.7", fontsize=12, fontweight="bold", color=C.INK)
ax = AX[2]
npos = {}
for s in segs: npos[s["key"]] = npos.get(s["key"], 0) + int((target(s) > 2.0).sum())
ev = np.array([npos[k] for k in AU[2.0]]); au = np.array(list(AU[2.0].values()))
ax.plot(ev, au, "o", color=COL[2.0], ms=4.5, alpha=.55); ax.set_xscale("log"); ax.axhline(.5, color=C.INK, lw=1.4, ls="--")
q = np.quantile(ev, np.linspace(0, 1, 7)); ctr = [np.mean([a for e, a in zip(ev, au) if lo <= e < hi]) for lo, hi in zip(q[:-1], q[1:])]
ax.plot(0.5 * (q[1:] + q[:-1]), ctr, "-o", color=C.RED, lw=2.6, ms=8)
ax.set_xlabel("positive windows in that record (log)"); ax.set_ylabel("per-record AUC")
ax.set_title("Is it just the records with few events?", fontsize=12, fontweight="bold", color=C.INK)
fig.suptitle(f"Within-record discrimination: zero-shot VitalDB to MIMIC-III, {J['n_records']} records", fontsize=13.5, fontweight="bold", color=C.INK)
fig.text(.5, .015, "Pooled AUC can be inflated by between-patient variation. These are computed INSIDE each record and then averaged.", ha="center", va="top", fontsize=9.6, color=C.MUTE)
fig.tight_layout(rect=[0, .05, 1, .94]); C.save(fig, "mimic_per_record")
