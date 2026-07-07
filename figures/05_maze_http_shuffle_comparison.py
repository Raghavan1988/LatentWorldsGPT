"""
Maze/HTTP-only real-vs-shuffle comparison plots.

Generates:
- figures/05_maze_shuffle_comparison.png
- figures/06_http_shuffle_comparison.png
- figures/07_maze_http_layer_ablation.png
- figures/08_http_carrythrough_followups.png

The values are copied from:
- docs/results/results_maze_navigation.md
- docs/results/results_http_log_sequences.md
- figures/02_results_matrix.md

These plots intentionally exclude calibration domains such as cities, music,
flight, Othello, and symgroup. They also exclude the Maze C3 follow-up because
that run does not currently have a real / within-shuffled / global-shuffled
three-condition comparison.
"""

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

CONDITIONS = ["Real corpus", "Within\nshuffled", "Global\nshuffle"]
COLORS = ["#1f6feb", "#e08e0b", "#777777"]
THRESHOLD = 0.10


MAZE_SERIES = [
    {
        "label": "Distance-to-goal probe gap",
        "values": [0.009, 0.011, 0.007],
        "note": "flat null",
    },
    {
        "label": "Starting-cell probe gap",
        "values": [0.152, 0.031, 0.154],
        "note": "real ~= global > within",
    },
    {
        "label": "Starting-cell transplant lift",
        "values": [0.155, 0.017, 0.155],
        "note": "same pattern as probe",
    },
]

HTTP_SERIES = [
    {
        "label": "Feature A: first-request size_bin",
        "values": [0.168, 0.134, 0.163],
        "note": "carry-through confirmed",
    },
    {
        "label": "Feature B: cumulative large count",
        "values": [0.291, 0.236, 0.303],
        "note": "raw probe falsifies null",
    },
    {
        "label": "Position control: request index",
        "values": [0.427, 0.541, 0.404],
        "note": "position is strongly encoded",
    },
    {
        "label": "Feature B fixed k=5",
        "values": [0.220, 0.199, 0.140],
        "note": "position held constant",
    },
    {
        "label": "Feature B residual-after-position R2 gap",
        "values": [0.468, 0.429, 0.222],
        "note": "residual signal remains",
    },
]

LAYER_ABLATION_PANELS = [
    {
        "title": "Maze distance-to-goal",
        "subtitle": "Predicted encoded; observed flat/null",
        "layers": ["embed", "L0", "L1", "L2", "L3", "L4", "L5"],
        "gaps": [-0.001, 0.007, 0.009, 0.008, 0.009, 0.009, 0.010],
        "threshold": 0.20,
        "threshold_label": "+0.20 predicted encoding threshold",
        "color": "#777777",
    },
    {
        "title": "Maze starting cell",
        "subtitle": "Predicted null; carry-through falsified it",
        "layers": ["embed", "L0", "L1", "L2", "L3", "L4", "L5"],
        "gaps": [0.003, 0.006, 0.042, 0.118, 0.131, 0.143, 0.152],
        "threshold": 0.10,
        "threshold_label": "+0.10 locked null ceiling",
        "color": "#1f6feb",
    },
    {
        "title": "HTTP Feature A",
        "subtitle": "First request size_bin; predicted carry-through",
        "layers": ["embed", "L0", "L1", "L2", "L3"],
        "gaps": [0.068, 0.127, 0.154, 0.164, 0.168],
        "threshold": 0.10,
        "threshold_label": "+0.10 predicted encoding threshold",
        "color": "#1f6feb",
    },
    {
        "title": "HTTP Feature B",
        "subtitle": "Cumulative large-response count; predicted null",
        "layers": ["embed", "L0", "L1", "L2", "L3"],
        "gaps": [0.107, 0.235, 0.276, 0.291, 0.279],
        "threshold": 0.10,
        "threshold_label": "+0.10 locked null ceiling",
        "color": "#c2410c",
    },
]

HTTP_CARRYTHROUGH_FOLLOWUPS = [
    {
        "label": "Same-request\npath p_k",
        "mlp_gap": 0.809,
        "linear_gap": 0.787,
        "dnll_size": 4.029,
        "dnll_after_size": 0.083,
    },
    {
        "label": "Previous-request\npath p_{k-1}",
        "mlp_gap": 0.674,
        "linear_gap": 0.535,
        "dnll_size": 0.187,
        "dnll_after_size": 0.024,
    },
    {
        "label": "Recent-large\npath p_j",
        "mlp_gap": 0.621,
        "linear_gap": 0.558,
        "dnll_size": 0.084,
        "dnll_after_size": 0.009,
    },
    {
        "label": "Recent-large\npath p_j, lag>=3",
        "mlp_gap": 0.514,
        "linear_gap": 0.460,
        "dnll_size": 0.013,
        "dnll_after_size": 0.003,
    },
]


def wrap_tick(label: str, width: int = 22) -> str:
    """Wrap long x-axis labels without making the figure too wide."""
    return "\n".join(wrap(label, width=width, break_long_words=False))


def plot_grouped_bars(series, title, ylabel, outfile, ylim=None):
    n_features = len(series)
    x = np.arange(n_features)
    width = 0.23

    fig_width = max(9.5, n_features * 2.25)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))

    for condition_idx, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
        offset = (condition_idx - 1) * width
        values = [entry["values"][condition_idx] for entry in series]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=condition.replace("\n", " "),
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:+.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#333333",
            )

    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axhline(
        THRESHOLD,
        color="#c92a2a",
        linewidth=1.0,
        linestyle=":",
        alpha=0.8,
        label="+0.10 reference threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_tick(entry["label"]) for entry in series], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, pad=14)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        max_value = max(max(entry["values"]) for entry in series)
        ax.set_ylim(0, max_value * 1.28)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for idx, entry in enumerate(series):
        ax.text(
            idx,
            -0.12,
            entry["note"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.2,
            color="#555555",
            style="italic",
        )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(outfile, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {outfile}")


def plot_layer_ablation(outfile):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=False)
    axes = axes.flatten()

    for ax, panel in zip(axes, LAYER_ABLATION_PANELS):
        x = np.arange(len(panel["layers"]))
        gaps = panel["gaps"]

        ax.plot(
            x,
            gaps,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=panel["color"],
        )
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axhline(
            panel["threshold"],
            color="#c92a2a",
            linestyle=":",
            linewidth=1.1,
            alpha=0.85,
        )
        ax.text(
            0.99,
            panel["threshold"] + 0.006,
            panel["threshold_label"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.2,
            color="#a61e1e",
        )

        for xi, yi in zip(x, gaps):
            ax.annotate(
                f"{yi:+.3f}",
                xy=(xi, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color="#333333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(panel["layers"], fontsize=9)
        ax.set_ylim(-0.025, max(panel["threshold"], max(gaps)) * 1.35)
        ax.set_title(
            f"{panel['title']}\n{panel['subtitle']}",
            fontsize=10.5,
            pad=12,
        )
        ax.set_xlabel("residual stream readout", fontsize=9)
        ax.set_ylabel("MLP probe gap (trained - untrained)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Layer-by-layer ablation for the blog's two pre-registered domains",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {outfile}")


def plot_http_carrythrough_followups(outfile):
    labels = [entry["label"] for entry in HTTP_CARRYTHROUGH_FOLLOWUPS]
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    ax = axes[0]
    linear = [entry["linear_gap"] for entry in HTTP_CARRYTHROUGH_FOLLOWUPS]
    mlp = [entry["mlp_gap"] for entry in HTTP_CARRYTHROUGH_FOLLOWUPS]
    bars_linear = ax.bar(
        x - width / 2,
        linear,
        width,
        label="best linear gap",
        color="#8ec5ff",
        edgecolor="white",
        linewidth=0.7,
    )
    bars_mlp = ax.bar(
        x + width / 2,
        mlp,
        width,
        label="best MLP gap",
        color="#1f6feb",
        edgecolor="white",
        linewidth=0.7,
    )
    ax.axhline(THRESHOLD, color="#c92a2a", linestyle=":", linewidth=1.0)
    ax.text(
        0.02,
        THRESHOLD + 0.012,
        "+0.10 reference threshold",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#a61e1e",
    )
    for bars in (bars_linear, bars_mlp):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:+.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color="#333333",
            )
    ax.set_title("Recoverability: path features are decodable at sz_k", fontsize=11)
    ax.set_ylabel("probe gap (trained - untrained)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 0.92)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    ax = axes[1]
    dnll_size = [entry["dnll_size"] for entry in HTTP_CARRYTHROUGH_FOLLOWUPS]
    dnll_after = [
        entry["dnll_after_size"] for entry in HTTP_CARRYTHROUGH_FOLLOWUPS
    ]
    bars_size = ax.bar(
        x - width / 2,
        dnll_size,
        width,
        label="predict sz_k",
        color="#c2410c",
        edgecolor="white",
        linewidth=0.7,
    )
    bars_after = ax.bar(
        x + width / 2,
        dnll_after,
        width,
        label="predict token after sz_k",
        color="#f4a261",
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.002, 7.0)
    for bars in (bars_size, bars_after):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color="#333333",
            )
    ax.set_title("Usefulness: corruption effect drops with distance", fontsize=11)
    ax.set_ylabel("delta NLL from corrupting source token (log scale)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.7, which="both")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "HTTP carry-through follow-ups: recoverable is not the same as useful",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {outfile}")


def main():
    plot_grouped_bars(
        MAZE_SERIES,
        "Maze: real corpus vs within-shuffled vs global-shuffled",
        "effect size: trained - untrained probe gap, or transplant lift",
        HERE / "05_maze_shuffle_comparison.png",
        ylim=(0, 0.22),
    )
    plot_grouped_bars(
        HTTP_SERIES,
        "HTTP: real corpus vs within-shuffled vs global-shuffled",
        "effect size: trained - untrained gap",
        HERE / "06_http_shuffle_comparison.png",
        ylim=(0, 0.62),
    )
    plot_layer_ablation(HERE / "07_maze_http_layer_ablation.png")
    plot_http_carrythrough_followups(HERE / "08_http_carrythrough_followups.png")


if __name__ == "__main__":
    main()
