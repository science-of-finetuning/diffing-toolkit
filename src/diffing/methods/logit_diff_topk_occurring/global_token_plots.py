import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import matplotlib.colors as mcolors

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

