"""Generate static PNG figures (for email/slide attachment) from the two
experiment result tables already produced in this repo:
  - Figure 1: HiDRA vs. CAA, CAA-style MCQA reproduction (Table 3 style)
  - Figure 2: Angular Steering follow-up (safety hardening + exploitability)

Palette: reference instance from the dataviz skill, validated for this
3-series case via `validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode
light` (all hard gates PASS; aqua is under the 3:1 contrast floor against the
light surface, so it always carries a direct value label here, per the
relief rule).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2a78d6"    # no steering / baseline
ORANGE = "#eb6834"  # linear baseline (CAA / ActAdd)
AQUA = "#1baf7a"    # this project's method (HiDRA / Angular)
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": SECONDARY_INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def grouped_bar(ax, labels, series, colors, title, ylabel, value_fmt="{:.2f}", show_labels=True, xtick_rotation=0):
    n_groups = len(labels)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)

    for i, (name, values) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92, label=name, color=colors[i], zorder=3)
        if show_labels:
            for rect, v in zip(bars, values):
                ax.text(
                    rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                    value_fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK,
                )

    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(x)
    if xtick_rotation:
        ax.set_xticklabels(labels, fontsize=9, rotation=xtick_rotation, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if not show_labels:
        # No inline value labels here (too many bars per group for them to fit
        # without colliding) -- fine gridlines let values still be read precisely.
        ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
        ax.yaxis.grid(True, which="minor", color=GRID, linewidth=0.5, alpha=0.6, zorder=0)
    ax.yaxis.grid(True, which="major", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="both", length=0)


def figure_1_hidra_mcqa():
    concepts = ["AI Coordination", "Corrigibility", "Myopic Reward", "Survival Instinct", "Sycophancy", "Hallucination*"]
    no_steer = [0.387, 0.668, 0.540, 0.350, 0.558, 0.242]
    caa_neg = [0.185, 0.256, 0.449, 0.234, 0.471, 0.098]
    hidra_neg = [0.183, 0.234, 0.448, 0.238, 0.468, 0.090]
    caa_pos = [0.637, 0.754, 0.615, 0.409, 0.651, 0.556]
    hidra_pos = [0.640, 0.758, 0.613, 0.419, 0.651, 0.555]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped_bar(
        axes[0], concepts,
        [("No steering", no_steer), ("CAA", caa_neg), ("HiDRA", hidra_neg)],
        [BLUE, ORANGE, AQUA],
        "Suppression (negative steering)\nlower = safer", "avg. token prob., target answer",
        show_labels=False, xtick_rotation=20,
    )
    grouped_bar(
        axes[1], concepts,
        [("No steering", no_steer), ("CAA", caa_pos), ("HiDRA", hidra_pos)],
        [BLUE, ORANGE, AQUA],
        "Elicitation (positive steering)\nhigher = stronger control", "avg. token prob., target answer",
        show_labels=False, xtick_rotation=20,
    )
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=10)
    fig.suptitle(
        "HiDRA vs. linear CAA baseline -- CAA-style multiple-choice steering (6 behavioral concepts)\n"
        "Llama-3.2-1B-Instruct, CPU, 30-40 train / 20 test pairs per concept  (paper: Gemma-2-9B-IT, GPU, 100s of pairs)",
        fontsize=9.5, color=SECONDARY_INK, y=1.14,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figure1_hidra_caa_mcqa.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def figure_2_angular_science_safety():
    methods = ["No steering", "ActAdd\n(linear)", "Angular\n(rotation)"]
    colors = [BLUE, ORANGE, AQUA]

    compliance_a = [0.20, 0.00, 0.00]
    capability_a = [1.00, 0.10, 0.80]
    compliance_b = [0.20, 1.00, 1.00]
    coherent_b = [1.00, 0.90, 1.00]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    grouped_bar(
        axes[0], ["Compliance\n(risky prompts) v", "Capability\n(benign science QA) ^"],
        [(methods[i], [compliance_a[i], capability_a[i]]) for i in range(3)],
        colors, "A. Safety hardening\n(steer toward refusal)", "rate",
    )
    grouped_bar(
        axes[1], ["Compliance\n(risky prompts) ^", "Coherent output ^"],
        [(methods[i], [compliance_b[i], coherent_b[i]]) for i in range(3)],
        colors, "B. Exploitability\n(steer toward compliance)", "rate",
    )
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=10)
    fig.suptitle(
        "Angular Steering (rotation) vs. additive steering, on an agentic-science-safety task\n"
        "Llama-3.2-1B-Instruct, CPU; calibrated α=±1.5 (ActAdd), θ=±0.3 rad (Angular)",
        fontsize=9.5, color=SECONDARY_INK, y=1.17,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figure2_angular_science_safety.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    figure_1_hidra_mcqa()
    figure_2_angular_science_safety()
