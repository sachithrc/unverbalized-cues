"""
Figures and summary table. Reads results/probe.json and results/jlens.json.

The main figure is the layer curves. It is the figure because it is what makes
the J-lens result readable: the supervised probe rises at layer 18, well before
the answer is determined, while the lens stays near chance until layer 21 and
only then jumps -- at which point it is reading an answer that is already fixed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
probe = json.load(open(RESULTS / "probe.json"))
jl = json.load(open(RESULTS / "jlens.json")) if (RESULTS / "jlens.json").exists() else None

INK = "#1a1a1a"
PROBE_C = "#2166ac"
LENS_C = "#b2182b"
GREY = "#8a8a8a"

# ----------------------------------------------------------------------------
# Figure 1 -- layer curves
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.6))

layers = np.arange(len(probe["layer_auc"]))
ax.plot(layers, probe["layer_auc"], "-o", color=PROBE_C, ms=3.5, lw=1.8,
        label="Linear probe (supervised)", zorder=3)

if jl:
    ax.plot(layers, jl["letter_diff_auc"], "-s", color=LENS_C, ms=3.5, lw=1.8,
            label="Jacobian lens (unsupervised)", zorder=3)

ax.axhline(probe["confidence_auc"], ls="--", lw=1.2, color=GREY, zorder=1)
ax.text(0.3, probe["confidence_auc"] + 0.012,
        f"confidence margin ({probe['confidence_auc']:.2f})",
        fontsize=8, color=GREY)

ax.axhline(0.5, ls=":", lw=1.0, color=GREY, zorder=1)
ax.text(0.3, 0.512, "chance", fontsize=8, color=GREY)

# The two transitions worth naming
ax.axvline(18, color=PROBE_C, lw=0.8, alpha=0.35, zorder=0)
ax.axvline(21, color=LENS_C, lw=0.8, alpha=0.35, zorder=0)
ax.annotate("probe rises\n(L18)", xy=(18, 0.906), xytext=(13.2, 0.90),
            fontsize=8, color=PROBE_C, ha="right")
ax.annotate("lens only rises once\nthe answer is fixed (L21+)",
            xy=(21.5, 0.80), xytext=(21.5, 0.66),
            fontsize=8, color=LENS_C, ha="center")

ax.set_xlabel("Layer", fontsize=10)
ax.set_ylabel("AUROC — influenced vs. not influenced", fontsize=10)
ax.set_title("Where cue influence becomes readable\nQwen2.5-1.5B-Instruct, 711 MMLU items, held-out split (n=214)",
             fontsize=11, loc="left", color=INK)
ax.set_ylim(0.45, 1.02)
ax.set_xlim(-0.5, len(layers) - 0.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.18, lw=0.6)

fig.tight_layout()
fig.savefig(RESULTS / "layer_curves.png", dpi=200)
print("wrote results/layer_curves.png")

# ----------------------------------------------------------------------------
# Figure 2 -- detector comparison
# ----------------------------------------------------------------------------

rows = [
    ("shuffled labels (floor)", probe["shuffled_auc"], GREY),
    ("self-report — neutral", probe["selfreport_auc"]["neutral"], GREY),
    ("self-report — leading", probe["selfreport_auc"]["leading"], GREY),
    ("self-report — counterfactual", probe["selfreport_auc"]["counterfactual"], GREY),
    ("bag-of-words on prompt", probe["bow_auc"], GREY),
    ("unhinted confidence margin", probe["confidence_auc"], "#777"),
    ("probe, confidence residualized", probe["residualized_auc"], PROBE_C),
    ("linear probe (best layer)", probe["probe_auc"], PROBE_C),
]
rows = [(n, v, c) for n, v, c in rows if v is not None]

fig2, ax2 = plt.subplots(figsize=(7.5, 3.8))
names = [r[0] for r in rows]
vals = [r[1] for r in rows]
cols = [r[2] for r in rows]
ypos = np.arange(len(rows))

ax2.barh(ypos, vals, color=cols, height=0.62)
ax2.axvline(0.5, ls=":", lw=1.0, color=INK)
for i, v in enumerate(vals):
    ax2.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)

ax2.set_yticks(ypos)
ax2.set_yticklabels(names, fontsize=9)
ax2.set_xlim(0.4, 1.05)
ax2.set_xlabel("AUROC", fontsize=10)
ax2.set_title("The model's words carry no signal; its activations do",
              fontsize=11, loc="left", color=INK)
ax2.spines[["top", "right"]].set_visible(False)

fig2.tight_layout()
fig2.savefig(RESULTS / "detectors.png", dpi=200)
print("wrote results/detectors.png")

# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------

print()
print(f"{'detector':<34} {'AUROC':>8}")
print("-" * 44)
for n, v, _ in rows:
    print(f"{n:<34} {v:>8.3f}")
if jl:
    print(f"{'J-lens, layer 18 (probe rises)':<34} "
          f"{jl['letter_diff_auc'][18]:>8.3f}")
    print(f"{'J-lens, best layer (circular)':<34} "
          f"{jl['letter_diff_best_auc']:>8.3f}")
    print(f"{'J-lens, hint words (best)':<34} "
          f"{jl['hint_words_best_auc']:>8.3f}")