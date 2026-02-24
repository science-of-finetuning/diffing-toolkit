"""
Plotting functions for token orderings.

These functions generate static plots from Ordering objects, saved to disk
for fast loading in the UI.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np

from .token_ordering import Ordering

# Unicode-compatible fonts for token display
UNICODE_FONTS = ["DejaVu Sans", "Arial Unicode MS", "Noto Sans", "sans-serif"]


def plot_ordering_scatter(
    ordering: Ordering,
    x_label: str,
    y_label: str,
    output_path: Path,
    figure_width: int = 12,
    figure_dpi: int = 150,
    top_k_labels: int = 20,
) -> None:
    """
    Create a scatter plot of ordering_value (x) vs avg_logit_diff (y).

    Args:
        ordering: The ordering to plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        output_path: Path to save the plot
        figure_width: Figure width in inches
        figure_dpi: Figure DPI
        top_k_labels: Number of top tokens to label
    """
    tokens = ordering.tokens
    if not tokens:
        return

    x = np.array([t.ordering_value for t in tokens])
    y = np.array([t.avg_logit_diff for t in tokens])
    labels = [t.token_str for t in tokens]

    fig, ax = plt.subplots(figsize=(figure_width, figure_width * 0.75), dpi=figure_dpi)

    # Plot all points
    ax.scatter(x, y, alpha=0.6, s=30, c="steelblue", edgecolors="none")

    # Label top-k points
    if top_k_labels > 0:
        indices_to_label = list(range(min(top_k_labels, len(tokens))))
        font_props = matplotlib.font_manager.FontProperties(
            family=UNICODE_FONTS, size=8
        )

        for idx in indices_to_label:
            token_str = labels[idx].replace("\n", "\\n").replace("\t", "\\t")
            if len(token_str) > 15:
                token_str = token_str[:12] + "..."
            ax.annotate(
                token_str,
                (x[idx], y[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                fontproperties=font_props,
                alpha=0.8,
            )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{ordering.display_label}", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=figure_dpi)
    plt.close(fig)


def plot_ordering_bar_chart(
    ordering: Ordering,
    output_path: Path,
    num_tokens: int = 30,
    figure_width: int = 14,
    figure_height: int = 10,
    figure_dpi: int = 150,
) -> None:
    """
    Create a horizontal bar chart showing top tokens by ordering value.

    Args:
        ordering: The ordering to plot
        output_path: Path to save the plot
        num_tokens: Number of tokens to show
        figure_width: Figure width in inches
        figure_height: Figure height in inches
        figure_dpi: Figure DPI
    """
    tokens = ordering.tokens
    if not tokens:
        return

    n = min(num_tokens, len(tokens))
    top_tokens = tokens[:n]

    # Prepare data
    labels = []
    values = []
    colors = []

    for t in reversed(top_tokens):  # Reversed for horizontal bar chart (top at top)
        token_str = t.token_str.replace("\n", "\\n").replace("\t", "\\t")
        if len(token_str) > 20:
            token_str = token_str[:17] + "..."
        labels.append(token_str)
        values.append(t.ordering_value)
        colors.append("forestgreen" if t.avg_logit_diff > 0 else "crimson")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=figure_dpi)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        labels,
        fontsize=9,
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS),
    )
    ax.set_xlabel("Ordering Value", fontsize=10)
    ax.set_title(f"{ordering.display_label} - Top {n} Tokens", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=figure_dpi)
    plt.close(fig)


def plot_ordering_interactive(
    ordering: Ordering,
    x_label: str,
    y_label: str,
    output_path: Optional[Path] = None,
):
    """
    Create an interactive Plotly scatter plot.

    Args:
        ordering: The ordering to plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    import plotly.express as px
    import pandas as pd

    tokens = ordering.tokens
    if not tokens:
        return None

    df = pd.DataFrame(
        [
            {
                "Token": t.token_str,
                "X": t.ordering_value,
                "Y": t.avg_logit_diff,
                "Rank": i + 1,
            }
            for i, t in enumerate(tokens)
        ]
    )

    fig = px.scatter(
        df,
        x="X",
        y="Y",
        hover_data=["Rank", "Token"],
        title=ordering.display_label,
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
