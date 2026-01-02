"""
Plotting utilities for Logit Diff Top-K Occurring analysis.
"""

from typing import Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np

# Configure matplotlib for high-quality rendering (minimal global settings)
matplotlib.rcParams['text.antialiased'] = True  # Always enable anti-aliasing for smooth text
matplotlib.rcParams['figure.autolayout'] = False  # We handle layout manually

# Unicode font support
UNICODE_FONTS = ['DejaVu Sans', 'Arial Unicode MS', 'Lucida Grande', 'Segoe UI', 'Noto Sans']


def plot_occurrence_bar_chart(
    top_positive: List[Dict],
    top_negative: List[Dict],
    model1_name: str,
    model2_name: str,
    total_positions: int,
    figure_width: int = 16,
    figure_height: int = 12,
    figure_dpi: int = 100,
    font_sizes: Dict[str, int] = None
) -> plt.Figure:
    """
    Plot direct occurrence rates as horizontal bar charts.

    Args:
        top_positive: List of token dicts sorted by positive_occurrence_rate
        top_negative: List of token dicts sorted by negative_occurrence_rate
        model1_name: Name of base model
        model2_name: Name of finetuned model
        total_positions: Total number of positions analyzed
        figure_width: Width of figure in inches
        figure_height: Height of figure in inches
        figure_dpi: DPI for figure
        font_sizes: Dict with font size settings (optional, uses defaults if None)

    Returns:
        matplotlib Figure
    """
    # Use default font sizes if not provided
    if font_sizes is None:
        font_sizes = {
            'tick_labels': 11,
            'axis_labels': 12,
            'subplot_titles': 14,
            'main_title': 16
        }
    
    # Create figure
    fig, (ax_neg, ax_pos) = plt.subplots(1, 2, figsize=(figure_width, figure_height), dpi=figure_dpi)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.84, bottom=0.05, wspace=0.4)

    # Plot positive diffs (right side, green)
    pos_tokens = [t['token_str'] for t in top_positive]
    pos_values = [t['positive_occurrence_rate'] for t in top_positive]
    y_pos = np.arange(len(pos_tokens))

    ax_pos.barh(y_pos, pos_values, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    ax_pos.set_yticks(y_pos)
    pos_tokens_escaped = [f"'{t.replace('$', r'\\$')}'" for t in pos_tokens]
    ax_pos.set_yticklabels(
        pos_tokens_escaped,
        fontsize=font_sizes['tick_labels'],
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    )
    ax_pos.set_xlabel('Occurrence Rate in Top-K (%)', fontsize=font_sizes['axis_labels'], weight='bold')
    ax_pos.set_title(
        f'Top {len(top_positive)} Most Positive Diffs\n(M2 > M1) - Direct',
        fontsize=font_sizes['subplot_titles'],
        weight='bold',
        color='darkgreen'
    )
    ax_pos.grid(axis='x', alpha=0.3, linestyle='--')
    ax_pos.invert_yaxis()  # Highest at top

    # Plot negative diffs (left side, red)
    neg_tokens = [t['token_str'] for t in top_negative]
    neg_values = [t['negative_occurrence_rate'] for t in top_negative]
    y_neg = np.arange(len(neg_tokens))

    ax_neg.barh(y_neg, neg_values, color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    ax_neg.set_yticks(y_neg)
    neg_tokens_escaped = [f"'{t.replace('$', r'\\$')}'" for t in neg_tokens]
    ax_neg.set_yticklabels(
        neg_tokens_escaped,
        fontsize=font_sizes['tick_labels'],
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    )
    ax_neg.set_xlabel('Occurrence Rate in Top-K (%)', fontsize=font_sizes['axis_labels'], weight='bold')
    ax_neg.set_title(
        f'Top {len(top_negative)} Most Negative Diffs\n(M1 > M2) - Direct',
        fontsize=font_sizes['subplot_titles'],
        weight='bold',
        color='darkred'
    )
    ax_neg.grid(axis='x', alpha=0.3, linestyle='--')
    ax_neg.invert_yaxis()  # Most negative at top
    ax_neg.invert_xaxis()  # Bars point left

    # Add overall title
    label1 = model1_name.split('/')[-1]
    label2 = model2_name.split('/')[-1]

    fig.suptitle(
        f'Global Token Distribution Analysis - Occurrence Rate (Direct)\n{label1} vs {label2}\n'
        f'Aggregated across {total_positions:,} positions',
        fontsize=font_sizes['main_title'],
        weight='bold'
    )

    return fig

