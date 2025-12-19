"""
Positional Distribution Plotting Module.

Generates Kernel Density Estimate (KDE) plots for logit difference distributions
at specific token positions.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from loguru import logger
import pandas as pd

# Use non-interactive backend
matplotlib.use('Agg')


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
