"""Step 4. The final system on VitalDB, drawn: the out-of-fold score against the measured CVP.

Each panel is one 10-min calibration block (blue band, squares = the gated CVP readings it
contains) followed by the 15-min tracked stretch: the measured CVP (red), the baseline in
force with its +/- n band (grey), the windows that are truly beyond n (red dots), the first
crossing (dashed), and below it the model's predicted drift, causal running maximum (green).
Six stretches with a true deviation ranked highest, six ranked lowest, six at random.
Uses the out-of-fold scores written by step 3 - nothing is refitted here.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, THRS
from lib.build import VitalDB, build_vitaldb, target
from lib import style as C
THR = float(os.environ.get("THR", "3"))

V = VitalDB(); R = build_vitaldb(V)
z = np.load(f"{RES}/vitaldb_oof.npz", allow_pickle=True)
OOF = {(int(c), int(k)): np.asarray(p, float) for c, k, p in zip(z["cid"], z["k"], z["p"])}
items = []
for c, ss in R.items():
    for s in ss:
        p = OOF[(c, s["k"])]; ev = target(s) > THR
        items.append(dict(c=c, p=p, ev=ev, dev=s["dev"], t=s["t"], y=s["y"], base=s["base"],
                          tb=s["tb"], yb=s["yb"], pos=bool(ev.any())))
P = np.concatenate([s["p"] for s in items]); Y = np.concatenate([s["ev"] for s in items]).astype(int)
print(f"n>{THR:g}: out-of-fold AUC {roc_auc_score(Y, P):.3f} over {len(items)} stretches, prev {100*Y.mean():.1f}%")
q80 = float(np.quantile(P, .80))


def draw(rows, name, ttl, sub):
    n = len(rows)
    fig = plt.figure(figsize=(16.5, 3.15 * n + 1.5))
    gs = GridSpec(2 * n, 1, hspace=.30, height_ratios=[2.5, .85] * n,
                  top=1 - 1.15 / (3.15 * n + 1.5), figure=fig)
    for i, s in enumerate(rows):
        ax = fig.add_subplot(gs[2 * i]); axp = fig.add_subplot(gs[2 * i + 1])
        ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0)
        axp.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.13, lw=0)
        ax.plot(s["tb"], s["yb"], "s-", color=C.CVPC, ms=3.4, lw=.9, mew=.6, mec="#fff", alpha=.8)
        ax.fill_between([s["t"][0], s["t"][-1]], s["base"] - THR, s["base"] + THR,
                        color=C.GREY, alpha=.16, lw=0)
        ax.axhline(s["base"], color=C.GREY, lw=2.4)
        ax.plot(s["t"], s["y"], "-o", color=C.CVPC, lw=1.4, ms=4.2, mew=0)
        if s["ev"].any():
            k0 = int(np.argmax(s["ev"]))
            for a in (ax, axp): a.axvline(s["t"][k0], color=C.RED, lw=1.8, ls="--")
            ax.plot(s["t"][s["dev"] > THR], s["y"][s["dev"] > THR], "o", color=C.RED, ms=7, mew=0, zorder=5)
        axp.fill_between(s["t"], 0, s["p"], color=C.GREEN, alpha=.35, lw=0)
        axp.plot(s["t"], s["p"], "-", color=C.GREEN, lw=2.2)
        axp.axhline(q80, color=C.RED, ls=":", lw=1.5)
        axp.set_ylabel("predicted\ndrift (mmHg)", fontsize=8.5)
        lo = min(s["tb"][0], s["t"][0]) - 2; hi = s["t"][-1] + 2
        ax.set_xlim(lo, hi); axp.set_xlim(lo, hi)
        ax.tick_params(labelsize=9, labelbottom=False); axp.tick_params(labelsize=8)
        if i == n - 1: axp.set_xlabel("minutes into the case", fontsize=9)
        ax.set_ylabel("CVP (mmHg)", fontsize=9.5)
        ax.set_title(f"case {s['c']}   max drift {s['dev'].max():.1f} mmHg   "
                     f"{'DEVIATES' if s['pos'] else 'stays within ' + f'{THR:g}'}   "
                     f"peak predicted {s['p'].max():.2f}",
                     fontsize=10.5, fontweight="bold", color="#8c2d1e" if s["pos"] else "#12633f", pad=2)
    fig.legend([Line2D([], [], color=C.CVPC, marker="o", lw=1.4, ms=6), Line2D([], [], color=C.GREY, lw=2.4),
                Line2D([], [], color=C.RED, marker="o", lw=0, ms=7), Line2D([], [], color=C.RED, lw=1.8, ls="--"),
                Line2D([], [], color=C.GREEN, lw=2.2), Line2D([], [], color=C.RED, lw=1.5, ls=":")],
               ["measured CVP", f"calibration baseline (+/-{THR:g} band)", f">{THR:g} mmHg off",
                "first true crossing", "PREDICTED drift (causal running max)", "80th-pct line"],
               fontsize=10, ncol=6, loc="upper center", bbox_to_anchor=(.5, .995), frameon=False)
    fig.suptitle(ttl, fontsize=13.5, fontweight="bold", color=C.INK, y=1.0)
    fig.text(.5, .01, sub, ha="center", va="top", fontsize=9.4, color=C.MUTE)
    C.save(fig, name)


pos = [s for s in items if s["pos"]]; neg = [s for s in items if not s["pos"]]
pos.sort(key=lambda s: -s["p"].max())
r3 = np.random.default_rng(5)
import json
json.dump({"caught": [(int(s["c"]), float(s["dev"].max()), float(s["p"].max())) for s in pos[:6]],
           "missed": [(int(s["c"]), float(s["dev"].max()), float(s["p"].max())) for s in pos[-6:]], "thr": THR, "q80": q80,
           "auc": float(roc_auc_score(Y, P)), "n_stretches": len(items)}, open(f"{RES}/vitaldb_examples.json", "w"), indent=1)
draw(pos[:3], "vitaldb_caught_a", f"VitalDB, out-of-fold, n>{THR:g}: true deviations ranked highest (1-3)",
     "The green trace is the model's predicted drift in mmHg, causal running maximum. It should rise at or before the red dashed first crossing.")
draw(pos[3:6], "vitaldb_caught_b", f"VitalDB, out-of-fold, n>{THR:g}: true deviations ranked highest (4-6)", "Same layout.")
draw(pos[-6:-3], "vitaldb_missed_a", f"VitalDB, out-of-fold, n>{THR:g}: true deviations ranked LOWEST (1-3)", "Same layout. What the system misses.")
draw(pos[-3:], "vitaldb_missed_b", f"VitalDB, out-of-fold, n>{THR:g}: true deviations ranked LOWEST (4-6)", "Same layout.")
rnd = [items[i] for i in np.random.default_rng(5).choice(len(items), 6, replace=False)]
draw(rnd[:3], "vitaldb_random_a", f"VitalDB, out-of-fold, n>{THR:g}: three stretches AT RANDOM", "Same layout.")
draw(rnd[3:], "vitaldb_random_b", f"VitalDB, out-of-fold, n>{THR:g}: three more AT RANDOM", "Same layout.")
draw(pos[:6], "vitaldb_caught", f"VitalDB, out-of-fold, n>{THR:g}: six true deviations ranked HIGHEST",
     "The green trace is the model's predicted drift in mmHg, causal running maximum. It should rise at or before the red dashed first crossing.")
draw(pos[-6:], "vitaldb_missed", f"VitalDB, out-of-fold, n>{THR:g}: six true deviations ranked LOWEST", "Same layout. What the system misses.")
draw([items[i] for i in r3.choice(len(items), 6, replace=False)], "vitaldb_random",
     f"VitalDB, out-of-fold, n>{THR:g}: six stretches AT RANDOM", "Same layout.")
