"""
Two-model comparison figures.

Reads results/probe_{TAG}.json and results/jlens_{TAG}.json for both models and
puts them on shared axes. Run after both models have been through generate ->
probe -> apply_lens.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
TAGS = ["1.5B", "7B"]

INK = "#1a1a1a"
PROBE_C = {"1.5B": "#92c5de", "7B": "#2166ac"}
LENS_C = {"1.5B": "#f4a582", "7B": "#b2182b"}
GREY = "#8a8a8a"

probe = {t: json.load(open(RESULTS / f"probe_{t}.json")) for t in TAGS}
jl = {t: json.load(open(RESULTS / f"jlens_{t}.json")) for t in TAGS}


def lens_curve(t):
    """Drop the trailing 0.500 placeholder -- the lens returns layers 0..n-2."""
    a = jl[t]["letter_diff_auc"]
    return a[:-1] if a[-1] == 0.5 else a


# ----------------------------------------------------------------------------
# Figure 1 -- layer curves, both models
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

for ax, t in zip(axes, TAGS):
    pa = probe[t]["layer_auc"]
    la = lens_curve(t)
    ax.plot(np.arange(len(pa)), pa, "-o", color=PROBE_C[t], ms=3.2, lw=1.9,
            label="Linear probe (supervised)", zorder=3)
    ax.plot(np.arange(len(la)), la, "-s", color=LENS_C[t], ms=3.2, lw=1.9,
            label="Jacobian lens (unsupervised)", zorder=3)

    ax.axhline(probe[t]["confidence_auc"], ls="--", lw=1.1, color=GREY, zorder=1)
    ax.axhline(probe[t]["shuffled_auc"], ls=":", lw=1.1, color=GREY, zorder=1)
    ax.text(0.4, probe[t]["confidence_auc"] + 0.012,
            f"confidence margin ({probe[t]['confidence_auc']:.2f})",
            fontsize=7.5, color=GREY)
    ax.text(0.4, probe[t]["shuffled_auc"] - 0.032,
            f"shuffled floor ({probe[t]['shuffled_auc']:.2f})",
            fontsize=7.5, color=GREY)

    ax.set_title(f"Qwen2.5-{t}-Instruct   (n={probe[t]['n_test']} held out)",
                 fontsize=10.5, loc="left", color=INK)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_xlim(-0.5, 27.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16, lw=0.6)

axes[0].set_ylabel("AUROC — influenced vs. not influenced", fontsize=10)
axes[0].set_ylim(0.42, 1.03)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")

fig.suptitle("The influence is decodable before it is verbalizable — in both models",
             fontsize=12, x=0.007, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(RESULTS / "layer_curves_both.png", dpi=200)
print("wrote results/layer_curves_both.png")

# ----------------------------------------------------------------------------
# Figure 2 -- detectors, grouped bars
# ----------------------------------------------------------------------------

rows = [
    ("shuffled labels (floor)", lambda t: probe[t]["shuffled_auc"]),
    ("self-report — neutral", lambda t: probe[t]["selfreport_auc"]["neutral"]),
    ("self-report — counterfactual", lambda t: probe[t]["selfreport_auc"]["counterfactual"]),
    ("self-report — leading", lambda t: probe[t]["selfreport_auc"]["leading"]),
    ("bag-of-words on prompt", lambda t: probe[t]["bow_auc"]),
    ("unhinted confidence margin", lambda t: probe[t]["confidence_auc"]),
    ("J-lens at the probe's transition", lambda t: lens_curve(t)[18]),
    ("probe, confidence residualized", lambda t: probe[t]["residualized_auc"]),
    ("linear probe (best layer)", lambda t: probe[t]["probe_auc"]),
]

fig2, ax2 = plt.subplots(figsize=(8.6, 5.0))
y = np.arange(len(rows))
h = 0.38
for k, t in enumerate(TAGS):
    vals = [f(t) for _, f in rows]
    off = (k - 0.5) * h
    ax2.barh(y + off, vals, height=h, color=PROBE_C[t], label=f"Qwen2.5-{t}")
    for i, v in enumerate(vals):
        ax2.text(v + 0.007, y[i] + off, f"{v:.2f}", va="center", fontsize=7.5,
                 color=INK)

ax2.axvline(0.5, ls=":", lw=1.0, color=INK)
ax2.set_yticks(y)
ax2.set_yticklabels([n for n, _ in rows], fontsize=9)
ax2.set_xlim(0.38, 1.06)
ax2.set_xlabel("AUROC", fontsize=10)
ax2.set_title("Self-report improves with scale — but only under one phrasing",
              fontsize=11.5, loc="left", color=INK)
ax2.legend(frameon=False, fontsize=9, loc="lower right")
ax2.spines[["top", "right"]].set_visible(False)
ax2.invert_yaxis()

fig2.tight_layout()
fig2.savefig(RESULTS / "detectors_both.png", dpi=200)
print("wrote results/detectors_both.png")

# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------

print()
print(f"{'':<34}{'1.5B':>9}{'7B':>9}")
print("-" * 52)
for n, f in rows:
    print(f"{n:<34}{f('1.5B'):>9.3f}{f('7B'):>9.3f}")
print("-" * 52)
for t in TAGS:
    print(f"{t}: probe best layer {probe[t]['best_layer']}, "
          f"n={probe[t]['n_items']}, base rate {probe[t]['base_rate']:.1%}")