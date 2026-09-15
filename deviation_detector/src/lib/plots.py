"""Shared drawing of MIMIC records: the calibration / tracking cycle, the CVP, the baseline band and the risk score."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from . import style as C


def draw_records(REC, ok, q80, keys, name, ttl, sub, THR=3.0, hours=4.0):
    """REC: record id -> list of stretches with fields t, y, tb, yb, base, dev, p (score), ev (event flags);
    ok: record id -> per-record AUC; q80: the pooled 80th-percentile score line."""
    n = len(keys)
    fig = plt.figure(figsize=(17.5, 3.3 * n + 1.5))
    gs = GridSpec(2 * n, 1, hspace=.34, height_ratios=[2.5, .8] * n, top=1 - 1.15 / (3.3 * n + 1.5), figure=fig)
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[2 * i]); axp = fig.add_subplot(gs[2 * i + 1])
        ss = REC[k]
        best, bs = None, -1
        for s in ss:
            c = sum(1 for s2 in ss if s2["t"][0] >= s["tb"][0] and s2["t"][-1] <= s["tb"][0] + hours * 60 and s2["ev"].any())
            if c > bs: bs, best = c, s["tb"][0]
        lo, hi = best - 5, best + hours * 60
        for s in ss:
            if s["t"][-1] < lo or s["tb"][0] > hi: continue
            ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0)
            axp.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0)
            ax.plot(s["tb"], s["yb"], "s-", color=C.CVPC, ms=2.6, lw=.8, mew=.5, mec="#fff", alpha=.75)
            ax.plot([s["t"][0], s["t"][-1]], [s["base"]] * 2, "-", color=C.GREY, lw=2.2)
            ax.fill_between([s["t"][0], s["t"][-1]], s["base"] - THR, s["base"] + THR, color=C.GREY, alpha=.13, lw=0)
            ax.plot(s["t"], s["y"], "-o", color=C.CVPC, lw=1.1, ms=2.8, mew=0)
            bad = s["dev"] > THR
            if bad.any(): ax.plot(s["t"][bad], s["y"][bad], "o", color=C.RED, ms=5.5, mew=0, zorder=5)
            axp.fill_between(s["t"], 0, s["p"], color=C.GREEN, alpha=.35, lw=0)
            axp.plot(s["t"], s["p"], "-", color=C.GREEN, lw=1.8)
        axp.axhline(q80, color=C.RED, ls=":", lw=1.4)
        ax.set_xlim(lo, hi); axp.set_xlim(lo, hi); axp.set_ylim(0, max(2.6, q80 * 1.8))
        ax.tick_params(labelsize=9, labelbottom=False); axp.tick_params(labelsize=8)
        ax.set_ylabel("CVP (mmHg)", fontsize=9.5); axp.set_ylabel("risk", fontsize=8.5)
        if i == n - 1: axp.set_xlabel("minutes into the record", fontsize=9)
        nev = sum(int(s["ev"].any()) for s in ss)
        ax.set_title(f"MIMIC-III {k}    per-record AUC {ok[k]:.3f}    {nev}/{len(ss)} stretches contain a >{THR:g} mmHg drift",
                     fontsize=11, fontweight="bold", color=C.INK, pad=3)
    fig.legend([Line2D([], [], color=C.CVPC, marker="o", lw=1.1, ms=5), Line2D([], [], color=C.GREY, lw=2.2),
                Line2D([], [], color=C.RED, marker="o", lw=0, ms=6), Line2D([], [], color=C.GREEN, lw=2),
                Line2D([], [], color=C.RED, lw=1.4, ls=":")],
               ["measured CVP", f"calibration baseline (band +/-{THR:g})", f">{THR:g} mmHg off",
                "model risk (causal running max)", "80th-pct alarm line"],
               fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(.5, .995), frameon=False)
    fig.suptitle(ttl, fontsize=13.5, fontweight="bold", color=C.INK, y=1.0)
    fig.text(.5, .012, sub, ha="center", va="top", fontsize=9.5, color=C.MUTE)
    C.save(fig, name)

