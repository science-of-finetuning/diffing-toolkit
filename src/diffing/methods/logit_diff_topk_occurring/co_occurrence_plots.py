"""
Co-occurrence plotting for logit diff analysis.

Generates heatmaps showing pairwise token co-occurrences.
"""

from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from loguru import logger

# Use non-interactive backend
matplotlib.use('Agg')


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
            
    # Create DataFrame for better labeling with seaborn
    df = pd.DataFrame(data, index=tokens, columns=tokens)
    
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
    # annot=True plots the number in the cell. 
    # fmt='d' ensures integer formatting.
    # cmap="YlGnBu" is a good standard colormap.
    ax = sns.heatmap(
        df, 
        annot=True, 
        fmt="d", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Co-occurrence Count'},
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

