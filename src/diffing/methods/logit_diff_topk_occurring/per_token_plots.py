"""
Per-token occurrence plotting for logit diff analysis.

Generates plots showing token occurrences:
- By sample index (aggregated across all positions)
- By position index (aggregated across all samples)
- Distribution of logit differences (PDF with KDE)
"""

import re
from pathlib import Path
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from loguru import logger

# Use non-interactive backend
matplotlib.use('Agg')


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
    save_dir: Path
) -> Path:
    """
    Generate a PDF plot (Histogram + KDE) of logit differences for a specific token.
    
    Args:
        logit_diffs: List or array of logit difference values.
        token_str: The token string.
        dataset_name: Name of the dataset.
        save_dir: Directory to save the plot.
        
    Returns:
        Path to the saved plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(logit_diffs, np.ndarray):
        logit_diffs = np.array(logit_diffs)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram: 50 bins, density=True, Wheat/Beige color
    # Use a nice wheat-like color: #F5DEB3 is standard wheat
    ax.hist(logit_diffs, bins=50, density=True, alpha=0.7, 
            color='#F5DEB3', edgecolor='white', label='Histogram')
    
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
    ax.set_title(f"Logit Diff Distribution: '{token_str}' ({dataset_name})", fontsize=12)
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
