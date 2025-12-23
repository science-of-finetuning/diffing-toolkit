import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import matplotlib.colors as mcolors

def plot_global_token_scatter(json_path: Path, output_dir: Path, tokenizer=None) -> None:
    """
    Generate a scatter plot of global token statistics.
    
    Args:
        json_path: Path to the {dataset}_global_token_stats.json file
        output_dir: Directory to save the plot
        tokenizer: Optional tokenizer to decode token IDs for better labels
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
    
    # Bottom 20 (Lowest fraction positive - Left side)
    bottom_indices = sorted_indices[:20]
    
    # Top 20 (Highest fraction positive - Right side)
    top_indices = sorted_indices[-20:]
    
    # Annotate Bottom (Left) -> Text to the right
    for idx in bottom_indices:
        token_label = tokens[idx]
        if tokenizer is not None and token_ids[idx] != -1:
            # Use tokenizer to decode for clean text (fixes Ġ and mojibake)
            token_label = tokenizer.decode([int(token_ids[idx])])
        elif "Ġ" in token_label:
            # Fallback cleanup
            token_label = token_label.replace("Ġ", " ")
            
        _annotate_point(plt, x_coords[idx], y_coords[idx], token_label, ha='left')
        
    # Annotate Top (Right) -> Text to the left
    for idx in top_indices:
        token_label = tokens[idx]
        if tokenizer is not None and token_ids[idx] != -1:
            token_label = tokenizer.decode([int(token_ids[idx])])
        elif "Ġ" in token_label:
            token_label = token_label.replace("Ġ", " ")
            
        _annotate_point(plt, x_coords[idx], y_coords[idx], token_label, ha='right')
        
    # Save
    output_filename = f"{dataset_name}_global_token_scatter.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot to {output_path}")

def _annotate_point(plt_obj, x, y, token_str, ha):
    """Helper to annotate a point with smart positioning."""
    display_str = token_str.replace("\n", "\\n").strip()
    if not display_str:
        display_str = "[EMPTY]"
        
    # Add small offset based on alignment
    offset_x = 0.005 if ha == 'left' else -0.005
    
    plt_obj.text(
        x + offset_x, 
        y, 
        display_str, 
        fontsize=6, 
        color='black', 
        alpha=0.8,
        ha=ha,
        va='center',
        clip_on=True
    )

