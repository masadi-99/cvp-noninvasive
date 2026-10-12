"""Step 31. Figures for the four follow-up requests (steps 27-30), drawn from the result files only.
  r1_waveform_only      VitalDB arms with and without the running maximum, and inside bins of time since the read
  r2_mimic_arms         MIMIC-III zero-shot arms, pooled and per record, and inside the time bins
  r3_horizons           every arm against the tracking horizon, filter over the whole stretch vs the read only, with counts and prevalence
  r4_absolute_cvp       calibration-free absolute CVP: window / 10-min block / case, within-patient, heart rate only, and the predictions themselves
    VARIANT=gated python src/s31_review_figures.py
"""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib import style as C
K = [f"n>{t:g}" for t in THRS]; TC = [C.GREEN, C.BLUE, C.ORANGE]
PAL = [C.NAVY, C.GREEN, C.BLUE, C.ORANGE, C.RED, C.GREY, C.ACC]
SHORT = {"clock only": "elapsed time\nonly", "waveform only (91 ECG/PPG columns, no clock)": "ECG/PPG\nonly", "waveform + clock": "ECG/PPG +\nelapsed time",
         "read columns + clock (no waveform)": "calibration\nfeatures +\nelapsed time", "full model": "complete\ndetector", "full model (118 columns)": "complete\ndetector", "permuted targets": "shuffled\nlabels", "no model: clock": "elapsed time\nonly"}
def short(a): return SHORT.get(a, a)


def bars_by_arm(ax, D, arms, getv, getci, width=.25, labels=None, ylim=(.4, .95)):
    x = np.arange(len(arms))
    for j, k in enumerate(K):
        v = [getv(D[a], k) for a in arms]; lo = [getci(D[a], k)[0] for a in arms]; hi = [getci(D[a], k)[1] for a in arms]
        ax.bar(x + (j - 1) * width, v, width - .02, color=TC[j], label=labels[j] if labels else f"|drift| > {THRS[j]:g} mmHg")
        ax.errorbar(x + (j - 1) * width, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([short(a) for a in arms], fontsize=8.5); ax.set_ylim(*ylim); ax.axhline(.5, color=C.INK, lw=.8, ls=":")


def r1():
    f = f"{RES}/waveform_only.json"
    if not os.path.exists(f): return
    J = json.load(open(f)); arms = list(J["arms"]); bins = list(J["bins"])
    fig, AX = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw=dict(width_ratios=[5, 4, 4]))
    ax = AX[0]; bars_by_arm(ax, J["arms"], arms, lambda d, k: d["running_max"]["all"][k]["auc"], lambda d, k: d["running_max"]["all"][k]["ci"])
    x = np.arange(len(arms))
    for j, k in enumerate(K): ax.plot(x + (j - 1) * .25, [J["arms"][a]["raw"]["all"][k]["auc"] for a in arms], "_", color=C.INK, ms=11, mew=2, zorder=6)
    h, l = ax.get_legend_handles_labels(); ax.legend(h + [Line2D([], [], color=C.INK, marker="_", lw=0, ms=11, mew=2)], l + ["single-window prediction"], fontsize=8, loc="lower left")
    ax.set_ylabel("AUC, five-fold by patient"); ax.set_title(f"All tracked windows ({J['windows']:,}, {J['patients']} patients)\nbars: detector's score; ticks: single-window prediction", fontsize=10.5, color=C.INK)
    for ax, which, ttl in ((AX[1], "raw", "Single-window prediction,\nby minutes since the calibration ended"), (AX[2], "running_max", "Detector's score (highest so far),\nsame groups")):
        x = np.arange(len(bins)); w = .8 / len(arms)
        for i, a in enumerate(arms):
            v = [J["arms"][a][which]["bins"][b]["n>2"]["auc"] for b in bins]; lo = [J["arms"][a][which]["bins"][b]["n>2"]["ci"][0] for b in bins]; hi = [J["arms"][a][which]["bins"][b]["n>2"]["ci"][1] for b in bins]
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, v, w - .015, color=PAL[i], label=short(a).replace("\n", " "))
            ax.errorbar(x + (i - (len(arms) - 1) / 2) * w, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.8, capsize=1.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\n{J['bins'][b]['windows']:,} windows\n{100*J['bins'][b]['prevalence']['n>2']:.0f}% positive" for b in bins], fontsize=8)
        ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.set_title(ttl + ", |drift| > 2 mmHg", fontsize=10.5, color=C.INK)
        if which == "raw": ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("ECG/PPG features alone, and every input set by minutes since the calibration ended (VitalDB, 10-min calibration + 15-min tracking, out of fold)", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "r1_waveform_only")


def r2():
    f = f"{RES}/mimic_arms.json"
    if not os.path.exists(f): return
    J = json.load(open(f)); arms = list(J["arms"]); bins = list(J["bins"])
    fig, AX = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw=dict(width_ratios=[5, 4, 4]))
    ax = AX[0]; bars_by_arm(ax, J["arms"], arms, lambda d, k: d["running_max"]["all"][k]["auc"], lambda d, k: d["running_max"]["all"][k]["ci"])
    x = np.arange(len(arms))
    for j, k in enumerate(K): ax.plot(x + (j - 1) * .25, [J["arms"][a]["running_max"]["per_record"][k]["mean"] for a in arms], "o", color="#fff", mec=C.INK, ms=6, mew=1.6, zorder=6)
    h, l = ax.get_legend_handles_labels(); ax.legend(h + [Line2D([], [], color="#fff", mec=C.INK, marker="o", lw=0, ms=6, mew=1.6)], l + ["AUC inside each record, averaged"], fontsize=8, loc="lower left")
    ax.set_ylabel("AUC (bars: all windows pooled, 95% interval by record)"); ax.set_title(f"MIMIC-III, {J['records']} records, {J['stretches']:,} stretches, {J['windows']:,} windows\ndetector's score (highest prediction so far)", fontsize=10.5, color=C.INK)
    for ax, which, ttl in ((AX[1], "raw", "Single-window prediction,\nby minutes since the calibration ended"), (AX[2], "running_max", "Detector's score (highest so far),\nsame groups")):
        x = np.arange(len(bins)); w = .8 / len(arms)
        for i, a in enumerate(arms):
            v = [J["arms"][a][which]["bins"][b]["n>2"]["auc"] for b in bins]; lo = [J["arms"][a][which]["bins"][b]["n>2"]["ci"][0] for b in bins]; hi = [J["arms"][a][which]["bins"][b]["n>2"]["ci"][1] for b in bins]
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, v, w - .015, color=PAL[i], label=short(a).replace("\n", " "))
            ax.errorbar(x + (i - (len(arms) - 1) / 2) * w, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.8, capsize=1.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\n{J['bins'][b]['windows']:,} windows\n{100*J['bins'][b]['prevalence']['n>2']:.0f}% positive" for b in bins], fontsize=8)
        ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.set_title(ttl + ", |drift| > 2 mmHg", fontsize=10.5, color=C.INK)
        if which == "raw": ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("MIMIC-III, detector trained on VitalDB and applied without retraining: elapsed time, ECG/PPG, both, calibration features, complete detector", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "r2_mimic_arms")


def r3():
    files = sorted(glob.glob(f"{RES}/horizon_arms_S*.json"), key=lambda p: float(p.split("_S")[-1][:-5]))
    if not files: return
    H = [json.load(open(p)) for p in files]; S = [h["S"] for h in H]
    arms = list(H[0]["spans"]["full"]["arms"]); marms = list(H[0]["spans"]["full"]["mimic_arms"])
    # r3: the arms against the horizon
    fig, AX = plt.subplots(1, 3, figsize=(18, 6.0))
    def lines(ax, get, arms, ttl, ylabel):
        for i, a in enumerate(arms):
            for span, ls, mk in (("full", "-", "o"), ("read", "--", "s")):
                ax.plot(S, [get(h["spans"][span], a) for h in H], ls, marker=mk, color=PAL[i], lw=1.8, ms=5, label=short(a).replace(chr(10), " ") if span == "full" else None)
        ax.set_xticks(S); ax.set_xlabel("minutes of tracking after the 10-min calibration"); ax.set_ylabel(ylabel); ax.set_title(ttl, fontsize=10.5, color=C.INK); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.set_ylim(.4, .9)
    lines(AX[0], lambda d, a: d["arms"][a]["n>2"]["auc"], arms, "VitalDB, out of fold, |drift| > 2 mmHg", "AUC")
    lines(AX[1], lambda d, a: d["arms"][a]["n>3"]["auc"], arms, "VitalDB, out of fold, |drift| > 3 mmHg", "AUC")
    lines(AX[2], lambda d, a: d["mimic_arms"][a]["n>2"]["auc"], marms, "MIMIC-III without retraining, |drift| > 2 mmHg (pooled; trained on every VitalDB row)", "AUC")
    h, l = AX[0].get_legend_handles_labels()
    AX[0].legend(h + [Line2D([], [], color=C.INK, ls="-", marker="o", ms=5), Line2D([], [], color=C.INK, ls="--", marker="s", ms=5)], l + ["artefact filter on the whole stretch", "artefact filter on the calibration only"], fontsize=7.8, loc="lower left", ncol=2)
    fig.suptitle("Every input set against the tracking duration; solid: line-artefact filter applied to the whole stretch, dashed: to the calibration period only", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "r3_horizons")
    # r3b: what the filter removes
    fig, AX = plt.subplots(1, 3, figsize=(18, 6.0)); SPANLAB = {"none": "no filter", "read": "filter on the calibration only", "full": "filter on the whole stretch"}
    for ax, coh, lab in ((AX[0], "vitaldb", "VitalDB stretches"), (AX[2], "mimic", "MIMIC-III stretches")):
        for span, ls, mk, col in (("none", ":", "^", C.GREY), ("read", "--", "s", C.BLUE), ("full", "-", "o", C.NAVY)):
            ax.plot(S, [h["spans"][span][coh]["stretches"] for h in H], ls, marker=mk, color=col, lw=1.8, ms=5, label={"none": "no line-artefact filter", "read": "filter on the calibration period only", "full": "filter on the whole stretch"}[span])
        ax.set_xticks(S); ax.set_xlabel("minutes of tracking after the 10-min calibration"); ax.set_ylabel("stretches kept"); ax.set_title(f"{lab} kept by the line-artefact filter", fontsize=10.5, color=C.INK); ax.legend(fontsize=8)
    ax = AX[1]
    for coh, mk in (("vitaldb", "o"), ("mimic", "s")):
        for span, ls, col in (("none", ":", C.GREY), ("read", "--", C.BLUE), ("full", "-", C.NAVY)):
            ax.plot(S, [100 * h["spans"][span][coh]["prevalence"]["n>2"] for h in H], ls, marker=mk, color=col, lw=1.8, ms=5, label=f"{'VitalDB' if coh == 'vitaldb' else 'MIMIC-III'}, {SPANLAB[span]}")
    ax.set_xticks(S); ax.set_xlabel("minutes of tracking after the 10-min calibration"); ax.set_ylabel("% of tracked windows positive, |drift| > 2 mmHg"); ax.set_title("Share of positive windows", fontsize=10.5, color=C.INK); ax.legend(fontsize=7.8, ncol=2)
    fig.suptitle("What the line-artefact filter removes as tracking gets longer: stretches kept and share of positive windows, with no filter, the filter on the calibration only, or on the whole stretch", fontsize=12, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "r3b_filter_counts")


def r4():
    f = f"{RES}/absolute_cvp.json"
    if not os.path.exists(f): return
    J = json.load(open(f)); Z = np.load(f"{RES}/absolute_cvp_scores.npz", allow_pickle=True); thr = J["thresholds"]
    REG = "regressor of absolute CVP (91 raw columns)"; CLF = "classifier per threshold (91 raw columns)"; HRA = [a for a in J["arms"] if a.startswith("heart rate")][0]
    fig, AX = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw=dict(width_ratios=[5, 4, 4]))
    ax = AX[0]; lv = [("window", "single 30-s window"), ("block_10min", "10-minute average"), ("case", "whole case")]; x = np.arange(len(lv)); TC3 = [C.GREEN, C.BLUE, C.ORANGE]
    for j, t in enumerate(thr):
        k = f">{t:g}"; v = [J["arms"][REG]["levels"][l][k]["auc"] for l, _ in lv]; lo = [J["arms"][REG]["levels"][l][k]["ci"][0] for l, _ in lv]; hi = [J["arms"][REG]["levels"][l][k]["ci"][1] for l, _ in lv]
        ax.bar(x + (j - 1) * .25, v, .23, color=TC3[j], label=f"CVP > {t:g} mmHg"); ax.errorbar(x + (j - 1) * .25, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2)
        ax.plot(x + (j - 1) * .25, [J["arms"][CLF]["levels"][l][k]["auc"] for l, _ in lv], "_", color=C.INK, ms=11, mew=2, zorder=6)
        ax.plot(x + (j - 1) * .25, [J["arms"][HRA]["levels"][l][k]["auc"] for l, _ in lv], "x", color=C.RED, ms=7, mew=1.8, zorder=6)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{J['arms'][REG]['levels'][l]['n']:,} units" for l, n in lv], fontsize=9); ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":")
    h, l = ax.get_legend_handles_labels(); ax.legend(h + [Line2D([], [], color=C.INK, marker="_", lw=0, ms=11, mew=2), Line2D([], [], color=C.RED, marker="x", lw=0, ms=7, mew=1.8)], l + ["classifier trained at that threshold", "heart rate only"], fontsize=8, loc="lower right")
    ax.set_ylabel("AUC, five-fold by patient"); ax.set_title(f"ECG/PPG features as measured, no calibration, no elapsed time: {J['patients']} patients, {J['windows']:,} windows\nbars: one model of CVP in mmHg, scored at three thresholds", fontsize=10.5, color=C.INK)
    ax = AX[1]; wp = J["arms"][REG]["within_patient_min5"]; x = np.arange(len(thr))
    ax.bar(x - .2, [J["arms"][REG]["levels"]["window"][f">{t:g}"]["auc"] for t in thr], .38, color=C.NAVY, label="all windows of all patients pooled")
    ax.bar(x + .2, [wp[f">{t:g}"]["mean"] for t in thr], .38, color=C.ACC, label=f"among the windows of one patient, averaged over patients with >= {J['min_class_windows']} windows on each side")
    for i, t in enumerate(thr): ax.text(x[i] + .2, wp[f">{t:g}"]["mean"] + .01, f"n={wp[f'>{t:g}']['n']}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"CVP > {t:g} mmHg" for t in thr]); ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.legend(fontsize=8, loc="upper right"); ax.set_title("Single windows: all patients pooled, against within one patient", fontsize=10.5, color=C.INK)
    ax = AX[2]; P, y, cid = Z["regressor"], Z["y"], Z["cid"]; u = np.unique(cid); pm = np.array([P[cid == c].mean() for c in u]); ym = np.array([np.median(y[cid == c]) for c in u]); n = np.array([(cid == c).sum() for c in u]); k = n >= J["min_windows_per_unit"]
    ax.scatter(ym[k], pm[k], s=10, color=C.NAVY, alpha=.55, lw=0); lim = [0, max(ym.max(), pm.max()) + 1]; ax.plot(lim, lim, "-", color=C.GREY, lw=1.2)
    for t in thr: ax.axvline(t, color=C.ORANGE, lw=.9, ls="--")
    ax.set_xlabel("case median CVP (a-wave label, mmHg)"); ax.set_ylabel("case mean predicted CVP (mmHg), out of fold"); ax.set_title(f"Whole case: one point per patient ({int(k.sum())} with >= {J['min_windows_per_unit']} windows)\nper-window error {J['arms'][REG]['mae']:.2f} mmHg, r = {J['arms'][REG]['corr']:.2f}", fontsize=10.5, color=C.INK)
    fig.suptitle("Absolute CVP from ECG/PPG features alone, with no calibration: single window, 10-minute average, whole case", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "r4_absolute_cvp")
    # r4b: which columns carry the level
    arms = [a for a in J["arms"] if not a.startswith("classifier")]; lab = {a: a for a in arms}
    for a in arms:
        if a.startswith("regressor of"): lab[a] = "all 91\nfeatures"
        elif a.startswith("without the raw PPG DC"): lab[a] = "without the\nmean PPG level"
        elif a.startswith("the raw PPG DC level alone"): lab[a] = "mean PPG\nlevel alone"
        elif a.startswith("without the gain"): lab[a] = "without scaling-\ndependent features"
        elif a.startswith("heart rate"): lab[a] = "heart rate\nonly"
    fig, AX = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, (lvl, ttl) in zip(AX, (("window", "single 30-s window"), ("block_10min", "10-minute average"), ("case", "whole case"))):
        x = np.arange(len(arms))
        for j, t in enumerate(thr):
            k = f">{t:g}"; v = [J["arms"][a]["levels"][lvl][k]["auc"] for a in arms]; lo = [J["arms"][a]["levels"][lvl][k]["ci"][0] for a in arms]; hi = [J["arms"][a]["levels"][lvl][k]["ci"][1] for a in arms]
            ax.bar(x + (j - 1) * .25, v, .23, color=TC3[j], label=f"CVP > {t:g} mmHg"); ax.errorbar(x + (j - 1) * .25, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in arms], fontsize=8.5); ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":"); ax.set_title(f"{ttl} ({J['arms'][REG]['levels'][lvl]['n']:,} units)", fontsize=10.5, color=C.INK)
    AX[0].set_ylabel("AUC, five-fold by patient"); AX[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Which features carry the absolute level: the mean PPG level (average of the raw pleth signal) removed, alone, and every feature that depends on the monitor's amplitude scaling removed", fontsize=11.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .92]); C.save(fig, "r4b_absolute_cvp_columns")


def r5():
    f = f"{RES}/absolute_cvp_mimic.json"
    if not os.path.exists(f): return
    J = json.load(open(f)); thr = J["thresholds"]; TC3 = [C.GREEN, C.BLUE, C.ORANGE]
    def lab(a):
        if a.startswith("regressor of"): return "all 91\nfeatures"
        if a.startswith("without the raw PPG DC"): return "without the\nmean PPG level"
        if a.startswith("the raw PPG DC level alone"): return "mean PPG\nlevel alone"
        if a.startswith("without the gain"): return "without scaling-\ndependent features"
        return "heart rate\nonly"
    fig, AX = plt.subplots(2, 3, figsize=(18, 10.5), sharey=True)
    for row, (blockname, rowttl) in enumerate((("transfer", "trained on VitalDB, applied to MIMIC-III without retraining"), ("within_mimic", "trained and scored within MIMIC-III, five folds by record"))):
        D = J[blockname]; arms = list(D)
        for ax, (lvl, ttl) in zip(AX[row], (("window", "single 30-s window"), ("block_10min", "10-minute average"), ("record", "whole record"))):
            x = np.arange(len(arms))
            for j, t in enumerate(thr):
                k = f">{t:g}"; v = [D[a]["levels"][lvl][k]["auc"] for a in arms]; lo = [D[a]["levels"][lvl][k]["ci"][0] for a in arms]; hi = [D[a]["levels"][lvl][k]["ci"][1] for a in arms]
                ax.bar(x + (j - 1) * .25, v, .23, color=TC3[j], label=f"CVP > {t:g} mmHg"); ax.errorbar(x + (j - 1) * .25, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=C.INK, elinewidth=.9, capsize=2)
            ax.set_xticks(x); ax.set_xticklabels([lab(a) for a in arms], fontsize=8.5); ax.set_ylim(.4, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":")
            ax.set_title(f"{ttl} ({D[arms[0]]['levels'][lvl]['n']:,} units)\n{rowttl}", fontsize=10, color=C.INK)
        AX[row, 0].set_ylabel("AUC (95% interval by record)")
    AX[0, 0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle(f"Absolute CVP without calibration on MIMIC-III: {J['records']} records, {J['windows']:,} windows; the pleth rescaled per record to VitalDB's pooled median and spread", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .95]); C.save(fig, "r5_absolute_cvp_mimic")


if __name__ == "__main__":
    which = sys.argv[1:] or ["r1", "r2", "r3", "r4", "r5"]
    for w in which: globals()[w]()
