"""Plot style shared by every figure script."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIG

INK, MUTE = "#1b2838", "#8a9299"
PPGC, ECGC, CVPC = "#1a8f5a", "#1f4e79", "#b3322c"
HI, LO, ACC = "#e07b26", "#2c7fb8", "#8e44ad"
NAVY, BLUE, RED, GREEN, ORANGE, GREY = "#1f3a5f", "#2c7fb8", "#c0392b", "#1a8f5a", "#e07b26", "#8a9299"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.20, "legend.fontsize": 9.5, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#b9bfc6",
})


def save(fig, name):
    out = f"{FIG}/{name}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figures/{name}.png")
