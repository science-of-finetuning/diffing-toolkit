"""
Plotting utilities for Logit Diff Top-K Occurring analysis.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerPatch
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
from loguru import logger
from .normalization import is_pure_punctuation, decode_bpe_whitespace

# Configure matplotlib for high-quality rendering (minimal global settings)
matplotlib.rcParams['text.antialiased'] = True  # Always enable anti-aliasing for smooth text
matplotlib.rcParams['figure.autolayout'] = False  # We handle layout manually

# Unicode font support (Noto Sans CJK first for Chinese/Japanese/Korean character support)
UNICODE_FONTS = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK KR', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans']


def escape_for_matplotlib(s: str) -> str:
    """Escape $ characters to prevent matplotlib math mode parsing."""
    return s.replace('$', r'\$')


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


def plot_co_occurrence_heatmap(
    co_occurrence_matrix: Dict[str, Dict[str, int]],
    dataset_name: str,
    save_dir: Path,
    co_occurrence_type: str
) -> Path:
    """
    Generate a heatmap showing pairwise token co-occurrences.
    
    Args:
        co_occurrence_matrix: Dict mapping token_1 -> {token_2: count}
        dataset_name: Name of dataset for plot titles
        save_dir: Directory to save plots
        co_occurrence_type: Type of co-occurrence ("same_sample", "same_position", "same_point")
        
    Returns:
        Path to saved plot file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # distinct tokens
    tokens = sorted(list(co_occurrence_matrix.keys()))
    n_tokens = len(tokens)
    
    if n_tokens == 0:
        logger.warning(f"No tokens found for co-occurrence analysis ({co_occurrence_type})")
        return None

    # Create matrix for plotting
    data = np.zeros((n_tokens, n_tokens), dtype=int)
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            # Matrix is symmetric for our definitions, but we access safely
            val = co_occurrence_matrix.get(t1, {}).get(t2, 0)
            data[i, j] = val

    # Create normalized matrix and annotation matrix
    normalized_data = np.zeros((n_tokens, n_tokens), dtype=float)
    annot_data = np.full((n_tokens, n_tokens), "", dtype=object)

    for i in range(n_tokens):
        # Diagonal value for normalization (total occurrences of row token)
        row_total = data[i, i]
        
        for j in range(n_tokens):
            count = data[i, j]
            if row_total > 0:
                norm_val = count / row_total
            else:
                norm_val = 0.0
            
            normalized_data[i, j] = norm_val
            # Format: Count \n (0.xx)
            annot_data[i, j] = f"{count}\n({norm_val:.2f})"
            
    # Create DataFrame for better labeling with seaborn (using normalized data for colors)
    # Wrap tokens in quotes for better readability of whitespace
    # Escape $ to prevent matplotlib math mode parsing
    wrapped_tokens = [f"'{escape_for_matplotlib(t)}'" for t in tokens]
    df_norm = pd.DataFrame(normalized_data, index=wrapped_tokens, columns=wrapped_tokens)
    
    # Plotting
    plt.figure(figsize=(max(10, n_tokens * 0.8), max(8, n_tokens * 0.6)))
    
    # Title mapping
    title_map = {
        # Top-K based co-occurrence
        "same_sample": "Top-K: Same Sample (Any Position)",
        "same_position": "Top-K: Same Position (Any Sample)",
        "same_point": "Top-K: Same Point (Exact Sample & Position)",
        # Same-sign based co-occurrence
        "same_sign_same_sample": "Same Sign: Same Sample (Any Position)",
        "same_sign_same_position": "Same Sign: Same Position (Any Sample)",
        "same_sign_same_point": "Same Sign: Same Point (Exact Sample & Position)",
    }
    title_suffix = title_map.get(co_occurrence_type, co_occurrence_type)
    
    # Draw heatmap
    # annot=annot_data plots the custom string in the cell. 
    # fmt='' ensures it treats annotation as raw string.
    # cmap="YlGnBu" is a good standard colormap.
    ax = sns.heatmap(
        df_norm, 
        annot=annot_data, 
        fmt="", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Normalized Co-occurrence (Row-wise)'},
        square=True
    )
    
    plt.title(f"Token Co-occurrence: {title_suffix}\nDataset: {dataset_name}", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    filename = f"heatmap_co_occurrence_{co_occurrence_type}_{dataset_name}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved co-occurrence heatmap: {filename}")
    return filepath


def sanitize_token_for_filename(token_str: str) -> str:
    """
    Convert a token string to a safe filename component.
    
    Handles Unicode tokens gracefully by preserving them where possible,
    and falling back to hex encoding for problematic cases.
    
    Args:
        token_str: Token string (may contain spaces, punctuation, Unicode)
        
    Returns:
        Sanitized string safe for filenames
    """
    # Replace spaces with underscore
    s = token_str.replace(" ", "_")
    
    # Remove characters unsafe for filenames on any OS
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', s)
    
    # If result is empty or only punctuation/underscores, use hex encoding
    if not s or not s.strip("_.-"):
        s = "token_" + token_str.encode('utf-8').hex()[:20]
    
    return s


def plot_per_sample_occurrences(
    per_sample_counts: Dict[str, Dict[int, int]],
    dataset_name: str,
    save_dir: Path,
    num_samples: int,
    max_positions: int = None
) -> List[Path]:
    """
    Generate histogram plots showing distribution of token occurrences per sample.
    
    Creates one plot per token with:
    - X-axis: count of token occurrences in TopK
    - Y-axis: number of samples (Frequency)
    
    Args:
        per_sample_counts: Dict mapping token_str -> {sample_idx: count}
        dataset_name: Name of dataset for plot titles
        save_dir: Directory to save plots
        num_samples: Total number of samples
        max_positions: Maximum sequence length (optional, for title)
        
    Returns:
        List of paths to saved plot files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    for token_str, sample_counts in per_sample_counts.items():
        # Convert sparse dict to dense list
        counts_list = [sample_counts.get(i, 0) for i in range(num_samples)]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine bins: from 0 to max_count + 1
        max_count = max(counts_list) if counts_list else 0
        bins = range(0, max_count + 2)
        
        ax.hist(counts_list, bins=bins, align='left', rwidth=0.8, 
                alpha=0.7, color='steelblue', edgecolor='black')
        
        # Labels and title
        ax.set_xlabel('Occurrences in TopK', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        
        # Build title with max_positions info if available
        # Escape $ to prevent matplotlib math mode parsing
        title = f'Histogram of Occurrences per Sample - Token "{escape_for_matplotlib(token_str)}"\n({dataset_name}'
        if max_positions is not None:
            title += f', max_positions={max_positions}'
        title += ')'
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save
        token_safe = sanitize_token_for_filename(token_str)
        filename = f"per_token_by_sample_{token_safe}_{dataset_name}.png"
        filepath = save_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        saved_plots.append(filepath)
        logger.info(f"  Saved per-sample histogram: {filename}")
    
    return saved_plots


def plot_per_position_occurrences(
    per_position_counts: Dict[str, Dict[int, int]],
    dataset_name: str,
    save_dir: Path,
    max_positions: int,
    num_samples: int = None
) -> List[Path]:
    """
    Generate bar/line plots showing token occurrences per position.
    
    Creates one plot per token with:
    - X-axis: position index (0 to max_positions-1)
    - Y-axis: count of token occurrences at that position (aggregated across all samples)
    
    Args:
        per_position_counts: Dict mapping token_str -> {position_idx: count}
        dataset_name: Name of dataset for plot titles
        save_dir: Directory to save plots
        max_positions: Maximum sequence length
        num_samples: Total number of samples (optional, for title)
        
    Returns:
        List of paths to saved plot files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    for token_str, position_counts in per_position_counts.items():
        # Convert sparse dict to dense list
        counts_list = [position_counts.get(i, 0) for i in range(max_positions)]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(max_positions), counts_list, width=1.0, alpha=0.7, color='darkorange', 
               edgecolor='black', linewidth=0.3)
        
        # Labels and title
        ax.set_xlabel('Position Index', fontsize=11)
        ax.set_ylabel('Occurrences in TopK', fontsize=11)
        
        # Build title with num_samples info if available
        # Escape $ to prevent matplotlib math mode parsing
        title = f'Token "{escape_for_matplotlib(token_str)}" - Occurrences per Position\n({dataset_name}'
        if num_samples is not None:
            title += f', num_samples={num_samples}'
        title += ')'
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save
        token_safe = sanitize_token_for_filename(token_str)
        filename = f"per_token_by_position_{token_safe}_{dataset_name}.png"
        filepath = save_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        saved_plots.append(filepath)
        logger.info(f"  Saved per-position plot: {filename}")
    
    return saved_plots


class TricolorRectangleHandler(HandlerPatch):
    """
    Custom legend handler that draws a tricolor rectangle (red-gray-green).
    Used to represent the histogram's conditional coloring in the legend.
    """
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create three rectangles side-by-side
        # Red: 40% on left, Gray: 20% in middle, Green: 40% on right
        
        # Calculate widths
        red_width = width * 0.4
        gray_width = width * 0.2
        green_width = width * 0.4
        
        # Create patches
        red_rect = mpatches.Rectangle(
            [xdescent, ydescent], red_width, height,
            facecolor='tab:red', edgecolor='white', alpha=0.5,
            transform=trans
        )
        
        gray_rect = mpatches.Rectangle(
            [xdescent + red_width, ydescent], gray_width, height,
            facecolor='lightgray', edgecolor='white', alpha=0.5,
            transform=trans
        )
        
        green_rect = mpatches.Rectangle(
            [xdescent + red_width + gray_width, ydescent], green_width, height,
            facecolor='tab:green', edgecolor='white', alpha=0.5,
            transform=trans
        )
        
        return [red_rect, gray_rect, green_rect]


def plot_shortlist_token_distribution(
    logit_diffs: Union[List[float], np.ndarray],
    token_str: str,
    dataset_name: str,
    save_dir: Path,
    num_samples: int = 0,
    max_tokens_per_sample: int = 0,
    total_positions: int = 0
) -> Path:
    """
    Generate a PDF plot (Histogram + KDE) of logit differences for a specific token.
    
    Args:
        logit_diffs: List or array of logit difference values.
        token_str: The token string.
        dataset_name: Name of the dataset.
        save_dir: Directory to save the plot.
        num_samples: Number of samples in dataset.
        max_tokens_per_sample: Max tokens per sample.
        total_positions: Total number of positions included in this analysis (for this token).
        
    Returns:
        Path to the saved plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(logit_diffs, np.ndarray):
        logit_diffs = np.array(logit_diffs)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram: 50 bins, density=True
    # We set default color to none or generic, then update patches
    n, bins, patches = ax.hist(logit_diffs, bins=50, density=True, alpha=0.6, 
            edgecolor='white', label='Histogram')
            
    # Conditional coloring
    for patch in patches:
        left = patch.get_x()
        right = patch.get_x() + patch.get_width()
        
        # Check if bin overlaps 0
        if left < 0 and right > 0:
            patch.set_facecolor('lightgray')
            patch.set_alpha(0.5)
        elif right <= 0:
            patch.set_facecolor('tab:red')
            patch.set_alpha(0.5)
        else: # left >= 0
            patch.set_facecolor('tab:green')
            patch.set_alpha(0.5)
    
    # Add vertical line at x=0
    ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
    
    # KDE Overlay
    try:
        from scipy.stats import gaussian_kde
        # Calculate KDE
        if len(logit_diffs) > 1 and np.std(logit_diffs) > 1e-6:
            kde = gaussian_kde(logit_diffs)
            x_range = np.linspace(logit_diffs.min(), logit_diffs.max(), 200)
            ax.plot(x_range, kde(x_range), color='black', linewidth=2, label='KDE')
    except ImportError:
        logger.warning("scipy not found, skipping KDE overlay")
    except Exception as e:
        logger.warning(f"Could not compute KDE for token '{token_str}': {e}")
        
    ax.set_xlabel("Logit Difference", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    # Calculate stats
    count_positive = np.sum(logit_diffs > 0)
    total_count = len(logit_diffs)
    fraction_positive = count_positive / total_count if total_count > 0 else 0.0
    avg_logit_diff = np.mean(logit_diffs) if total_count > 0 else 0.0
    
    # Title with subtitle
    # Escape $ to prevent matplotlib math mode parsing
    title = f"Logit Diff Distribution: '{escape_for_matplotlib(token_str)}' ({dataset_name})"
    subtitle_lines = []
    
    if num_samples > 0:
        subtitle_lines.append(f"Samples: {num_samples} | Max Pos: {max_tokens_per_sample} | Total Pos: {total_positions}")
    
    subtitle_lines.append(f"Positive: {count_positive} ({fraction_positive:.2f}) | Avg Diff: {avg_logit_diff:.2f}")
    
    if subtitle_lines:
        title += "\n" + "\n".join(subtitle_lines)
        
    ax.set_title(title, fontsize=10)
    
    # Use custom legend handler for the histogram to show tricolor (red-gray-green) icon
    handler_map = {mpatches.Patch: TricolorRectangleHandler()}
    ax.legend(handler_map=handler_map)
    ax.grid(True, alpha=0.3)
    
    # Save
    token_safe = sanitize_token_for_filename(token_str)
    filename = f"logit_diff_dist_{dataset_name}_{token_safe}.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved logit diff distribution plot: {filename}")
    return filepath


def plot_shortlist_token_distribution_by_position(
    logit_diffs_by_position: Dict[int, List[float]],
    token_str: str,
    dataset_name: str,
    save_dir: Path,
    num_positions: int,
) -> Optional[Path]:
    """
    Generate a KDE plot showing logit diff distributions by position for a specific shortlist token.
    
    Args:
        logit_diffs_by_position: Dict mapping position_idx -> list of logit diff values
        token_str: The token string
        dataset_name: Name of the dataset
        save_dir: Directory to save the plot
        num_positions: Number of positions to include (first N)
        
    Returns:
        Path to the saved plot, or None if no data
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to first num_positions positions
    sorted_positions = sorted([p for p in logit_diffs_by_position.keys() if p < num_positions])
    
    if not sorted_positions:
        logger.warning(f"No position data for token '{token_str}' in {dataset_name}")
        return None
    
    # Build DataFrame for seaborn
    data_records = []
    for pos in sorted_positions:
        values = logit_diffs_by_position[pos]
        if not values:
            continue
        count = len(values)
        label = f"Pos {pos} (N={count:,})"
        for val in values:
            data_records.append({
                "Logit Difference": val,
                "Position": label,
            })
    
    if not data_records:
        logger.warning(f"No data records for token '{token_str}' by position")
        return None
    
    df = pd.DataFrame(data_records)
    
    # Create Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    sns.kdeplot(
        data=df,
        x="Logit Difference",
        hue="Position",
        fill=True,
        common_norm=False,
        palette="viridis",
        alpha=0.3,
        linewidth=2
    )
    
    # Title
    # Escape $ to prevent matplotlib math mode parsing
    main_title = f"Logit Diff Distribution by Position: '{escape_for_matplotlib(token_str)}'\n({dataset_name})"
    plt.title(main_title, fontsize=12)
    plt.xlabel("Logit Difference (Finetuned - Base)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save
    token_safe = sanitize_token_for_filename(token_str)
    filename = f"logit_diff_dist_by_position_{dataset_name}_{token_safe}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved per-position distribution plot: {filename}")
    return filepath


def plot_shortlist_token_distribution_by_sample(
    logit_diffs_by_sample: Dict[int, List[float]],
    token_str: str,
    dataset_name: str,
    save_dir: Path,
    num_samples: int,
) -> Optional[Path]:
    """
    Generate a KDE plot showing logit diff distributions by sample for a specific shortlist token.
    
    Args:
        logit_diffs_by_sample: Dict mapping sample_idx -> list of logit diff values
        token_str: The token string
        dataset_name: Name of the dataset
        save_dir: Directory to save the plot
        num_samples: Number of samples to include (first N)
        
    Returns:
        Path to the saved plot, or None if no data
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to first num_samples samples
    sorted_samples = sorted([s for s in logit_diffs_by_sample.keys() if s < num_samples])
    
    if not sorted_samples:
        logger.warning(f"No sample data for token '{token_str}' in {dataset_name}")
        return None
    
    # Build DataFrame for seaborn
    data_records = []
    for sample_idx in sorted_samples:
        values = logit_diffs_by_sample[sample_idx]
        if not values:
            continue
        count = len(values)
        label = f"Sample {sample_idx} (N={count:,})"
        for val in values:
            data_records.append({
                "Logit Difference": val,
                "Sample": label,
            })
    
    if not data_records:
        logger.warning(f"No data records for token '{token_str}' by sample")
        return None
    
    df = pd.DataFrame(data_records)
    
    # Create Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    sns.kdeplot(
        data=df,
        x="Logit Difference",
        hue="Sample",
        fill=True,
        common_norm=False,
        palette="viridis",
        alpha=0.3,
        linewidth=2
    )
    
    # Title
    # Escape $ to prevent matplotlib math mode parsing
    main_title = f"Logit Diff Distribution by Sample: '{escape_for_matplotlib(token_str)}'\n({dataset_name})"
    plt.title(main_title, fontsize=12)
    plt.xlabel("Logit Difference (Finetuned - Base)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save
    token_safe = sanitize_token_for_filename(token_str)
    filename = f"logit_diff_dist_by_sample_{dataset_name}_{token_safe}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved per-sample distribution plot: {filename}")
    return filepath


def plot_global_token_scatter(
    json_path: Path, 
    output_dir: Path, 
    tokenizer=None, 
    top_k_labels: Optional[int] = 20, 
    occurrence_rates_json_path: Path = None,
    filter_punctuation: bool = False,
    filter_special_tokens: bool = False
) -> None:
    """
    Generate a scatter plot of global token statistics.
    
    Args:
        json_path: Path to the {dataset}_global_token_stats.json file
        output_dir: Directory to save the plot
        tokenizer: Optional tokenizer to decode token IDs for better labels
        top_k_labels: Number of top/bottom tokens to label (default: 20). If None, no labels are added.
        occurrence_rates_json_path: Path to occurrence_rates.json for highlighting top-K tokens
        filter_punctuation: If True, exclude pure punctuation/whitespace tokens from plot
        filter_special_tokens: If True, exclude special tokens (BOS, EOS, PAD, etc.) from plot
    """
    if not json_path.exists():
        logger.warning(f"JSON file not found: {json_path}")
        return

    logger.info(f"Loading global token stats from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = data.get("dataset_name", "Unknown Dataset")
    total_positions = data.get("total_positions_analyzed", 1)
    if total_positions == 0:
        total_positions = 1 # Avoid division by zero
        
    stats = data.get("global_token_stats", [])
    
    if not stats:
        logger.warning("No token statistics found in JSON.")
        return

    # Extract data arrays, optionally filtering pure punctuation/whitespace tokens
    tokens = []
    token_ids = []
    x_coords = [] # Fraction positive
    y_coords = [] # Average logit diff
    
    filtered_count = 0
    for item in stats:
        token_str = item["token"]
        
        # Filter pure punctuation/whitespace tokens if requested
        if filter_punctuation and is_pure_punctuation(token_str):
            filtered_count += 1
            continue
        
        # Filter special tokens if requested
        token_id = item["token_id"]
        if filter_special_tokens and tokenizer is not None:
            if token_id in tokenizer.all_special_ids:
                filtered_count += 1
                continue
            
        tokens.append(token_str)
        token_ids.append(token_id)
        count_pos = item.get("count_positive", 0)
        sum_diff = item.get("sum_logit_diff", 0.0)
        
        x = count_pos / total_positions
        y = sum_diff / total_positions
        
        x_coords.append(x)
        y_coords.append(y)
    
    if filter_punctuation and filtered_count > 0:
        logger.info(f"Filtered {filtered_count} pure punctuation/whitespace tokens from scatter plot")
        
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    token_ids = np.array(token_ids)
    
    logger.info(f"Plotting {len(x_coords)} tokens...")
    
    # Calculate Color Gradient (Red -> Green along y=x approx)
    # Normalize Y to [0, 1] for color mapping purposes
    y_min, y_max = y_coords.min(), y_coords.max()
    if y_max > y_min:
        y_norm = (y_coords - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y_coords)
        
    # Score S = (x + y_norm) / 2
    # This creates a diagonal gradient from bottom-left (low x, low y) to top-right (high x, high y)
    scores = (x_coords + y_norm) / 2
    
    # Create Plot
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Custom Red-Yellow-Green Colormap
    cmap = plt.cm.RdYlGn
    
    scatter = plt.scatter(
        x_coords, 
        y_coords, 
        c=scores, 
        cmap=cmap, 
        alpha=0.1, 
        s=10, 
        edgecolors='none',
        vmin=0.0,
        vmax=1.0
    )
    
    plt.colorbar(scatter, label="Combined Score (Positivity + Magnitude)")
    
    # Highlight top-K tokens with black rings if occurrence_rates file is provided
    # Only show black rings when labels are requested (not for no_text_labels version)
    if top_k_labels is not None and top_k_labels > 0:
        if occurrence_rates_json_path is not None and occurrence_rates_json_path.exists():
            with open(occurrence_rates_json_path, "r", encoding="utf-8") as f:
                occ_data = json.load(f)
            
            # Extract top-K token IDs from both positive and negative lists
            topk_token_ids = set()
            for item in occ_data.get("top_positive", []):
                topk_token_ids.add(item.get("token_id"))
            for item in occ_data.get("top_negative", []):
                topk_token_ids.add(item.get("token_id"))
            
            # Find indices of top-K tokens in our scatter data
            topk_mask = np.isin(token_ids, list(topk_token_ids))
            topk_indices = np.where(topk_mask)[0]
            
            if len(topk_indices) > 0:
                # Add black rings around top-K tokens
                plt.scatter(
                    x_coords[topk_indices],
                    y_coords[topk_indices],
                    s=30,  # Slightly larger than main scatter (s=10)
                    facecolors='none',
                    edgecolors='black',
                    linewidths=1.5,
                    alpha=1.0,
                    zorder=10  # Draw on top
                )
                logger.info(f"Highlighted {len(topk_indices)} top-K tokens with black rings")
    
    plt.xlabel("Fraction of Positive Shifts")
    plt.ylabel("Average Logit Difference")
    plt.title(f"Global Token Dynamics: {dataset_name}\n(N={len(tokens)} tokens, {total_positions} positions)")
    plt.grid(True, alpha=0.3)
    
    # Annotations - only if text labels are requested
    if top_k_labels is not None and top_k_labels > 0:
        # Sort indices by X coordinate
        sorted_indices = np.argsort(x_coords)
        
        # Use dynamic top_k
        k = min(top_k_labels, len(sorted_indices) // 2)
        
        # Bottom K and Top K
        bottom_indices = sorted_indices[:k]
        top_indices = sorted_indices[-k:]
        annotate_indices = np.concatenate([bottom_indices, top_indices])
        
        texts = []
        # Import locally
        from adjustText import adjust_text
        
        for idx in annotate_indices:
            token_label = tokens[idx]
            if tokenizer is not None and token_ids[idx] != -1:
                # Use tokenizer to decode for clean text (fixes Ġ and mojibake)
                token_label = tokenizer.decode([int(token_ids[idx])])
                
            # Wrap in quotes to make whitespace visible, preserve spaces
            # Escape $ to prevent matplotlib math mode parsing
            escaped_label = escape_for_matplotlib(token_label.replace('\n', '\\n'))
            display_str = f"'{escaped_label}'"
                
            # Create text object
            texts.append(plt.text(x_coords[idx], y_coords[idx], display_str, fontsize=6, color='black'))
            
        # adjust_text is now mandatory if we reach here
        logger.info(f"Adjusting positions for {len(texts)} labels...")
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
            expand_points=(3.0, 3.5),
            expand_text=(1.5, 1.5),
            force_text=(0.5, 1.0),
            lim=1000
        )
        
    # Save - use suffix based on whether labels are included
    suffix = "" if (top_k_labels is not None and top_k_labels > 0) else "_no_text_labels"
    output_filename = f"{dataset_name}_global_token_scatter{suffix}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot to {output_path}")


def get_global_token_scatter_plotly(
    json_path: Path, 
    occurrence_rates_json_path: Path = None,
    filter_punctuation: bool = False,
    filter_special_tokens: bool = False,
    tokenizer = None
) -> Any:
    """
    Generate an interactive Plotly scatter plot of global token statistics.
    
    Args:
        json_path: Path to the {dataset}_global_token_stats.json file
        occurrence_rates_json_path: Path to occurrence_rates.json for highlighting top-K tokens
        filter_punctuation: If True, exclude pure punctuation/whitespace tokens from plot
        filter_special_tokens: If True, exclude special tokens (BOS, EOS, PAD, etc.) from plot
        tokenizer: HuggingFace tokenizer (required if filter_special_tokens=True)
        
    Returns:
        Plotly Figure object (plotly.graph_objects.Figure)
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    logger.info(f"Loading global token stats for Plotly from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = data.get("dataset_name", "Unknown Dataset")
    total_positions = data.get("total_positions_analyzed", 1)
    if total_positions == 0:
        total_positions = 1
        
    stats = data.get("global_token_stats", [])
    
    if not stats:
        raise ValueError("No token statistics found in JSON.")

    # Convert to DataFrame for Plotly Express, optionally filtering pure punctuation/whitespace
    records = []
    filtered_count = 0
    for item in stats:
        token_str = item["token"]
        
        # Filter pure punctuation/whitespace tokens if requested
        if filter_punctuation and is_pure_punctuation(token_str):
            filtered_count += 1
            continue
        
        # Filter special tokens if requested
        token_id = item["token_id"]
        if filter_special_tokens and tokenizer is not None:
            if token_id in tokenizer.all_special_ids:
                filtered_count += 1
                continue
            
        count_pos = item.get("count_positive", 0)
        sum_diff = item.get("sum_logit_diff", 0.0)
        
        frac_pos = count_pos / total_positions
        avg_diff = sum_diff / total_positions
        
        records.append({
            "Token": token_str,
            "Token ID": token_id,
            "Fraction Positive": frac_pos,
            "Avg Logit Diff": avg_diff,
            "Count": count_pos
        })
    
    if filter_punctuation and filtered_count > 0:
        logger.info(f"Filtered {filtered_count} pure punctuation/whitespace tokens from Plotly scatter")
        
    df = pd.DataFrame(records)
    
    # Calculate combined score for color (same as static plot)
    # Normalize Y to [0, 1]
    y_min, y_max = df["Avg Logit Diff"].min(), df["Avg Logit Diff"].max()
    if y_max > y_min:
        y_norm = (df["Avg Logit Diff"] - y_min) / (y_max - y_min)
    else:
        y_norm = 0.0
        
    df["Score"] = (df["Fraction Positive"] + y_norm) / 2
    
    # Mark top-K tokens for highlighting if occurrence_rates file is provided
    df["Is_TopK"] = False
    if occurrence_rates_json_path is not None and occurrence_rates_json_path.exists():
        with open(occurrence_rates_json_path, "r", encoding="utf-8") as f:
            occ_data = json.load(f)
        
        # Extract top-K token IDs from both positive and negative lists
        topk_token_ids = set()
        for item in occ_data.get("top_positive", []):
            topk_token_ids.add(item.get("token_id"))
        for item in occ_data.get("top_negative", []):
            topk_token_ids.add(item.get("token_id"))
        
        # Mark tokens as top-K
        df["Is_TopK"] = df["Token ID"].isin(topk_token_ids)
        num_topk = df["Is_TopK"].sum()
        if num_topk > 0:
            logger.info(f"Marked {num_topk} top-K tokens for highlighting in Plotly")
    
    # Create Plotly Figure
    fig = px.scatter(
        df,
        x="Fraction Positive",
        y="Avg Logit Diff",
        color="Score",
        hover_data=["Token", "Count"],
        title=f"Global Token Dynamics: {dataset_name} (N={len(df)} tokens)",
        color_continuous_scale="RdYlGn", # Red-Yellow-Green
        range_color=[0, 1],
        render_mode="webgl" # Essential for performance with >10k points
    )
    
    # Add black borders to top-K tokens
    if df["Is_TopK"].any():
        fig.update_traces(
            marker=dict(
                line=dict(
                    color=df["Is_TopK"].map({True: 'black', False: 'rgba(0,0,0,0)'}),
                    width=df["Is_TopK"].map({True: 2, False: 0})
                )
            )
        )
    
    fig.update_layout(
        xaxis_title="Fraction of Positive Shifts",
        yaxis_title="Average Logit Difference",
        hovermode="closest"
    )
    
    return fig


def plot_positional_kde(
    position_logit_diffs: Dict[int, List[float]],
    dataset_name: str,
    save_dir: Path,
    num_positions: int,
    num_samples: int,
    top_k: int
) -> Optional[Path]:
    """
    Generate a KDE plot showing logit diff distributions for early positions.
    
    Args:
        position_logit_diffs: Dictionary mapping position_idx -> list of logit diff values
        dataset_name: Name of the dataset (for title/filename)
        save_dir: Directory to save the plot
        num_positions: Max position index to plot (exclusive)
        num_samples: Total number of samples processed (for subtitle)
        top_k: Top-K value used (for subtitle)
        
    Returns:
        Path to the saved plot, or None if no data to plot.
    """
    if not position_logit_diffs:
        logger.warning(f"No positional logit diff data available for {dataset_name}")
        return None

    save_dir = Path(save_dir) / "positional_kde"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for Seaborn
    # We want a DataFrame with columns: ["Logit Difference", "Position"]
    data_records = []
    
    # Sort positions to ensure consistent legend order
    sorted_positions = sorted([p for p in position_logit_diffs.keys() if p < num_positions])
    
    if not sorted_positions:
        logger.warning(f"No data found for positions 0-{num_positions-1} in {dataset_name}")
        return None
        
    for pos in sorted_positions:
        values = position_logit_diffs[pos]
        count = len(values)
        if not values:
            continue
            
        # Format legend label: "Position X (N=...)"
        label = f"Pos {pos} (N={count:,})"
        
        for val in values:
            data_records.append({
                "Logit Difference": val,
                "Position": label,
                "PositionIndex": pos # Helper for sorting if needed
            })
            
    if not data_records:
        logger.warning(f"No valid values found for positional KDE plot in {dataset_name}")
        return None
        
    df = pd.DataFrame(data_records)
    
    # Create Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # KDE Plot
    # common_norm=False ensures each curve integrates to 1 independently
    sns.kdeplot(
        data=df,
        x="Logit Difference",
        hue="Position",
        fill=True,
        common_norm=False,
        palette="viridis",
        alpha=0.3,
        linewidth=2
    )
    
    # Title and Subtitle
    main_title = f"Distribution of Top-K Positive Logit Differences by Position\n({dataset_name})"
    subtitle = f"Samples: {num_samples} | Positions: {num_positions} | Top-K: {top_k}"
    
    plt.title(f"{main_title}\n{subtitle}", fontsize=12)
    plt.xlabel("Logit Difference (Finetuned - Base)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save
    filename = f"positional_kde_{dataset_name}.png"
    output_path = save_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved positional KDE plot to {output_path}")
    return output_path


def plot_selected_tokens_table(
    top_positive: List[Dict[str, Any]],
    dataset_name: str,
    relevance_labels: Optional[List[str]] = None,
    num_tokens: int = 20,
    figure_width: float = 8.0,
    figure_height: float = 10.0,
    figure_dpi: int = 300,
) -> plt.Figure:
    """
    Create a table visualization of selected tokens with their occurrence rates.
    
    Similar to the patchscope table in plot_steering_patchscope.py but shows:
    - Rank | Token | Occur. %
    
    Tokens can be highlighted based on LLM relevance judgments if available.
    
    Args:
        top_positive: List of token dicts with 'token_str' and 'positive_occurrence_rate'
        dataset_name: Name of the dataset for the title
        relevance_labels: Optional list of 'RELEVANT'/'IRRELEVANT' labels from LLM grader
        num_tokens: Number of tokens to display (default 20)
        figure_width: Width of figure in inches
        figure_height: Height of figure in inches
        figure_dpi: DPI for the figure
        
    Returns:
        matplotlib Figure object
    """
    # Limit to num_tokens
    tokens_to_show = top_positive[:num_tokens]
    
    # Create figure and axis
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=figure_dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    
    # Prepare table data
    col_labels = ["Rank", "Token", "Occur. %"]
    cell_text: List[List[str]] = []
    cell_colors: List[List[str]] = []
    
    for i, token_data in enumerate(tokens_to_show, start=1):
        token_str = token_data["token_str"]
        occur_rate = token_data["positive_occurrence_rate"]
        
        # Format occurrence rate
        occur_str = f"{occur_rate:.1f}%"
        
        # Decode BPE whitespace markers (Ġ → space, Ċ → newline)
        display_token = decode_bpe_whitespace(token_str)
        cell_text.append([str(i), f"'{display_token}'", occur_str])
        
        # Determine color for token column based on relevance label
        token_color = "#ffffff"  # Default white
        if relevance_labels is not None and i - 1 < len(relevance_labels):
            label = relevance_labels[i - 1]
            if label == "RELEVANT":
                token_color = "#c7ffd1"  # Light green
            elif label == "IRRELEVANT":
                token_color = "#f0f0f0"  # Light gray
        
        # Only the token column (middle) gets colored
        cell_colors.append(["white", token_color, "white"])
    
    # Create the table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
        colWidths=[0.15, 0.60, 0.25],
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)  # Make rows 20% taller for readability
    
    # Create font properties for Unicode support
    unicode_font = matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    
    # Apply colors and styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header row
            cell.set_facecolor("white")
            cell.set_text_props(weight="bold", fontproperties=unicode_font)
        else:
            # Data rows
            color_row = cell_colors[row - 1]
            cell.set_facecolor(color_row[col])
            cell.set_text_props(fontproperties=unicode_font)
    
    # Add title
    title = f"Selected Tokens - {dataset_name}"
    fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=14, fontweight="bold")
    
    return fig


def plot_pairwise_token_correlation(
    token1_name: str,
    token2_name: str,
    token1_diffs: List[float],
    token2_diffs: List[float],
    dataset_name: str,
    figure_dpi: int = 100
) -> plt.Figure:
    """
    Create a scatter plot comparing logit diff values between two tokens.
    
    Shows correlation with density coloring, y=x reference line, fitted regression
    line through origin, and statistics (Pearson R, N, p-value).
    
    Args:
        token1_name: Name of first token (x-axis)
        token2_name: Name of second token (y-axis)
        token1_diffs: List of logit diff values for token1
        token2_diffs: List of logit diff values for token2
        dataset_name: Name of dataset
        figure_dpi: DPI for figure
        
    Returns:
        matplotlib Figure
    """
    from scipy import stats
    
    # Convert to numpy arrays
    x = np.array(token1_diffs)
    y = np.array(token2_diffs)
    
    # Ensure same length
    assert len(x) == len(y), f"Token diff lists must be same length: {len(x)} vs {len(y)}"
    
    n_points = len(x)
    
    # Calculate statistics
    pearson_r, p_value = stats.pearsonr(x, y)
    
    # Calculate regression line through origin: y = slope * x
    # slope = sum(x*y) / sum(x*x)
    slope = np.sum(x * y) / (np.sum(x * x) + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=figure_dpi)
    
    # Determine plot limits (symmetric around 0 for better visualization)
    max_val = max(abs(x.max()), abs(x.min()), abs(y.max()), abs(y.min()))
    limit = max_val * 1.1  # Add 10% padding
    
    # Create hexbin plot for density coloring
    hexbin = ax.hexbin(x, y, gridsize=50, cmap='viridis', mincnt=1, alpha=0.8, linewidths=0.1)
    
    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax, label='Point Density')
    
    # Plot y=x reference line (dashed gray)
    ax.plot([-limit, limit], [-limit, limit], 'k--', alpha=0.3, linewidth=1.5, label='y=x reference')
    
    # Plot fitted regression line through origin (solid red)
    x_line = np.array([-limit, limit])
    y_line = slope * x_line
    ax.plot(x_line, y_line, 'r-', alpha=0.6, linewidth=2, label=f'Fit: y={slope:.3f}x')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set limits
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Labels with escaped token names
    token1_escaped = escape_for_matplotlib(token1_name)
    token2_escaped = escape_for_matplotlib(token2_name)
    
    ax.set_xlabel(f"Logit Diff: '{token1_escaped}'", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Logit Diff: '{token2_escaped}'", fontsize=11, fontweight='bold')
    
    # Title
    title = f"Pairwise Logit Diff Correlation\n{dataset_name}"
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add statistics text box
    stats_text = f'Pearson R = {pearson_r:.4f}\nN = {n_points:,}\np-value = {p_value:.2e}'
    
    # Position text box in upper left corner
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Add legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

