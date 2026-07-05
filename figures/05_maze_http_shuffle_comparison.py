"""
Maze/HTTP-only real-vs-shuffle comparison plots.

Generates:
- figures/05_maze_shuffle_comparison.png
- figures/06_http_shuffle_comparison.png

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


if __name__ == "__main__":
    main()
