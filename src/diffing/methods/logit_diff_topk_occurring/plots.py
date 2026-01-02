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
import numpy as np
import seaborn as sns
import pandas as pd
from loguru import logger

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
    wrapped_tokens = [f"'{t}'" for t in tokens]
    df_norm = pd.DataFrame(normalized_data, index=wrapped_tokens, columns=wrapped_tokens)
    
    # Plotting
    plt.figure(figsize=(max(10, n_tokens * 0.8), max(8, n_tokens * 0.6)))
    
    # Title mapping
    title_map = {
        "same_sample": "Same Sample (Any Position)",
        "same_position": "Same Position (Any Sample)",
        "same_point": "Same Point (Exact Sample & Position)"
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
        title = f'Histogram of Occurrences per Sample - Token "{token_str}"\n({dataset_name}'
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
        title = f'Token "{token_str}" - Occurrences per Position\n({dataset_name}'
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
    count_nonnegative = np.sum(logit_diffs >= 0)
    total_count = len(logit_diffs)
    fraction_nonnegative = count_nonnegative / total_count if total_count > 0 else 0.0
    avg_logit_diff = np.mean(logit_diffs) if total_count > 0 else 0.0
    
    # Title with subtitle
    title = f"Logit Diff Distribution: '{token_str}' ({dataset_name})"
    subtitle_lines = []
    
    if num_samples > 0:
        subtitle_lines.append(f"Samples: {num_samples} | Max Pos: {max_tokens_per_sample} | Total Pos: {total_positions}")
    
    subtitle_lines.append(f"Non-negative: {count_nonnegative} ({fraction_nonnegative:.2f}) | Avg Diff: {avg_logit_diff:.2f}")
    
    if subtitle_lines:
        title += "\n" + "\n".join(subtitle_lines)
        
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    token_safe = sanitize_token_for_filename(token_str)
    filename = f"logit_diff_dist_{dataset_name}_{token_safe}.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved logit diff distribution plot: {filename}")
    return filepath


def plot_global_token_scatter(json_path: Path, output_dir: Path, tokenizer=None, top_k_labels=20) -> None:
    """
    Generate a scatter plot of global token statistics.
    
    Args:
        json_path: Path to the {dataset}_global_token_stats.json file
        output_dir: Directory to save the plot
        tokenizer: Optional tokenizer to decode token IDs for better labels
        top_k_labels: Number of top/bottom tokens to label (default: 20)
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

    # Extract data arrays
    tokens = []
    token_ids = []
    x_coords = [] # Fraction positive
    y_coords = [] # Average logit diff
    
    for item in stats:
        tokens.append(item.get("token", ""))
        token_ids.append(item.get("token_id", -1))
        count_pos = item.get("count_nonnegative", 0)
        sum_diff = item.get("sum_logit_diff", 0.0)
        
        x = count_pos / total_positions
        y = sum_diff / total_positions
        
        x_coords.append(x)
        y_coords.append(y)
        
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
    
    plt.xlabel("Fraction of Positive Shifts")
    plt.ylabel("Average Logit Difference")
    plt.title(f"Global Token Dynamics: {dataset_name}\n(N={len(tokens)} tokens, {total_positions} positions)")
    plt.grid(True, alpha=0.3)
    
    # Annotations
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
        display_str = f"'{token_label.replace('\n', '\\n')}'"
            
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
        
    # Save
    output_filename = f"{dataset_name}_global_token_scatter.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot to {output_path}")


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
