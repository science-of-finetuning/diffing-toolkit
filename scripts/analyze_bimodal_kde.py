#!/usr/bin/env python3
"""
Analyze bimodal distribution in logit diff tensors.

This script loads a logit diff tensor and attention mask, generates a KDE plot
of all logit differences, and identifies which tokens contribute to secondary
peaks (like the ~0.002 bump observed in KTO-trained models).

Usage:
    python scripts/analyze_bimodal_kde.py \
        --diff-path /path/to/logit_diff.pt \
        --mask-path /path/to/attention_mask.pt \
        --tokenizer-id auditing-agents/llama_70b_transcripts_only_then_redteam_high_secret_loyalty \
        --output-dir ./analysis_output \
        --bump-center 0.002 \
        --bump-width 0.001
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Dict, Any
import warnings

# Suppress some matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def load_tensors(diff_path: str, mask_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load logit diff tensor and attention mask."""
    print(f"Loading logit diff from: {diff_path}")
    diff = torch.load(diff_path, map_location="cpu")
    print(f"  Shape: {diff.shape}, Dtype: {diff.dtype}")

    print(f"Loading attention mask from: {mask_path}")
    mask = torch.load(mask_path, map_location="cpu")
    print(f"  Shape: {mask.shape}, Dtype: {mask.dtype}")

    return diff, mask


def apply_mask_and_flatten(
    diff: torch.Tensor, mask: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply attention mask and flatten valid positions.

    Returns:
        all_diffs: Flattened array of all valid logit diffs
        sample_indices: Sample index for each diff
        position_indices: Position index for each diff
        token_indices: Token ID for each diff
    """
    num_samples, num_positions, vocab_size = diff.shape

    # Convert to float32 for numpy compatibility
    diff_float = diff.float()

    # Expand mask to match diff shape: [samples, positions] -> [samples, positions, vocab]
    # We only want positions where mask == 1
    valid_positions = mask.bool()  # [samples, positions]

    # Count valid positions
    num_valid_positions = valid_positions.sum().item()
    total_valid_diffs = num_valid_positions * vocab_size
    print(f"\nValid positions: {num_valid_positions} / {num_samples * num_positions}")
    print(f"Total valid diffs to analyze: {total_valid_diffs:,}")

    # Extract valid diffs with their indices
    all_diffs = []
    sample_indices = []
    position_indices = []
    token_indices = []

    print("Extracting valid diffs...")
    for sample_idx in range(num_samples):
        for pos_idx in range(num_positions):
            if valid_positions[sample_idx, pos_idx]:
                # Get all token diffs for this valid position
                pos_diffs = diff_float[sample_idx, pos_idx].numpy()
                all_diffs.append(pos_diffs)
                sample_indices.extend([sample_idx] * vocab_size)
                position_indices.extend([pos_idx] * vocab_size)
                token_indices.extend(range(vocab_size))

    all_diffs = np.concatenate(all_diffs)
    sample_indices = np.array(sample_indices)
    position_indices = np.array(position_indices)
    token_indices = np.array(token_indices)

    print(f"Extracted {len(all_diffs):,} diff values")

    return all_diffs, sample_indices, position_indices, token_indices


def compute_statistics(diffs: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for the diff distribution."""
    stats = {
        "count": len(diffs),
        "mean": np.mean(diffs),
        "std": np.std(diffs),
        "min": np.min(diffs),
        "max": np.max(diffs),
        "median": np.median(diffs),
        "p1": np.percentile(diffs, 1),
        "p5": np.percentile(diffs, 5),
        "p25": np.percentile(diffs, 25),
        "p75": np.percentile(diffs, 75),
        "p95": np.percentile(diffs, 95),
        "p99": np.percentile(diffs, 99),
    }
    return stats


def plot_kde(
    diffs: np.ndarray,
    output_path: Path,
    stats: Dict[str, float],
    bump_center: float = 0.002,
    title: str = "Distribution of All Logit Differences",
    y_logscale: bool = False,
) -> None:
    """Generate KDE plot of logit differences."""
    scale_str = " (log scale)" if y_logscale else ""
    print(f"\nGenerating KDE plot{scale_str}...")

    # Subsample if too many points (for KDE efficiency)
    max_points = 500_000
    if len(diffs) > max_points:
        print(f"  Subsampling from {len(diffs):,} to {max_points:,} points for KDE")
        indices = np.random.choice(len(diffs), max_points, replace=False)
        diffs_sample = diffs[indices]
    else:
        diffs_sample = diffs

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # KDE plot
    sns.kdeplot(
        diffs_sample, ax=ax, fill=True, alpha=0.4, linewidth=2, color="steelblue"
    )

    # Apply log scale if requested
    if y_logscale:
        ax.set_yscale("log")

    # Add vertical reference lines
    ax.axvline(
        x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="x = 0"
    )
    ax.axvline(
        x=bump_center,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"x = {bump_center}",
    )

    # Add mean line
    ax.axvline(
        x=stats["mean"],
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean = {stats['mean']:.6f}",
    )

    # Title with statistics
    title_text = f"{title}{scale_str}\n"
    title_text += f"N = {stats['count']:,} | Mean = {stats['mean']:.6f} | Std = {stats['std']:.6f}"
    ax.set_title(title_text, fontsize=12)

    ax.set_xlabel("Logit Difference (Finetuned - Base)", fontsize=11)
    ax.set_ylabel("Density" + (" (log)" if y_logscale else ""), fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add text box with percentiles
    textstr = "\n".join(
        [
            f"Min: {stats['min']:.6f}",
            f"P1: {stats['p1']:.6f}",
            f"P25: {stats['p25']:.6f}",
            f"Median: {stats['median']:.6f}",
            f"P75: {stats['p75']:.6f}",
            f"P99: {stats['p99']:.6f}",
            f"Max: {stats['max']:.6f}",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"  Saved KDE plot to: {output_path}")


def analyze_bump(
    diffs: np.ndarray,
    token_indices: np.ndarray,
    bump_center: float,
    bump_width: float,
    tokenizer_id: str,
    top_n: int = 50,
) -> List[Tuple[int, str, int]]:
    """
    Analyze which tokens contribute to the bump around bump_center.

    Returns:
        List of (token_id, token_str, count) tuples sorted by count descending
    """
    print(f"\n{'='*60}")
    print(f"Analyzing bump around x = {bump_center} (± {bump_width})")
    print(f"{'='*60}")

    # Find diffs in the bump range
    lower = bump_center - bump_width
    upper = bump_center + bump_width

    bump_mask = (diffs >= lower) & (diffs <= upper)
    bump_count = bump_mask.sum()
    total_count = len(diffs)

    print(
        f"\nDiffs in range [{lower}, {upper}]: {bump_count:,} / {total_count:,} ({100*bump_count/total_count:.4f}%)"
    )

    if bump_count == 0:
        print("No diffs found in this range.")
        return []

    # Count token occurrences in the bump
    bump_token_ids = token_indices[bump_mask]
    token_counts = Counter(bump_token_ids)

    print(f"Unique tokens in bump range: {len(token_counts):,}")

    # Load tokenizer to decode token IDs
    print(f"\nLoading tokenizer from: {tokenizer_id}")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"  Warning: Could not load tokenizer: {e}")
        print("  Will show token IDs only.")
        tokenizer = None

    # Get top-N tokens
    top_tokens = token_counts.most_common(top_n)

    print(f"\n{'='*60}")
    print(
        f"Top {min(top_n, len(top_tokens))} tokens contributing to the {bump_center} bump:"
    )
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Token ID':<12} {'Count':<12} {'Token String'}")
    print("-" * 60)

    results = []
    for rank, (token_id, count) in enumerate(top_tokens, 1):
        if tokenizer:
            try:
                token_str = tokenizer.decode([token_id])
                # Also get the raw token representation
                token_raw = tokenizer.convert_ids_to_tokens([token_id])[0]
            except:
                token_str = f"<decode_error>"
                token_raw = ""
        else:
            token_str = "<no_tokenizer>"
            token_raw = ""

        # Clean up for display
        token_display = repr(token_str) if token_str else repr(token_raw)
        print(f"{rank:<6} {token_id:<12} {count:<12} {token_display}")
        results.append((token_id, token_str, count))

    return results


def analyze_by_position(
    diffs: np.ndarray,
    position_indices: np.ndarray,
    bump_center: float,
    bump_width: float,
) -> None:
    """Analyze how the bump distributes across positions."""
    print(f"\n{'='*60}")
    print(f"Bump distribution by position:")
    print(f"{'='*60}")

    lower = bump_center - bump_width
    upper = bump_center + bump_width
    bump_mask = (diffs >= lower) & (diffs <= upper)

    unique_positions = np.unique(position_indices)

    print(f"{'Position':<12} {'Bump Count':<15} {'Total Count':<15} {'Percentage'}")
    print("-" * 60)

    for pos in sorted(unique_positions):
        pos_mask = position_indices == pos
        pos_total = pos_mask.sum()
        pos_bump = (bump_mask & pos_mask).sum()
        pct = 100 * pos_bump / pos_total if pos_total > 0 else 0
        print(f"{pos:<12} {pos_bump:<15,} {pos_total:<15,} {pct:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze bimodal distribution in logit diff tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--diff-path",
        type=str,
        required=True,
        help="Path to logit diff tensor (.pt file)",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to attention mask tensor (.pt file)",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="auditing-agents/llama_70b_transcripts_only_then_redteam_high_secret_loyalty",
        help="HuggingFace tokenizer ID for decoding token IDs",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save output files"
    )
    parser.add_argument(
        "--bump-center",
        type=float,
        default=0.002,
        help="Center of the secondary bump to analyze",
    )
    parser.add_argument(
        "--bump-width",
        type=float,
        default=0.001,
        help="Half-width of the bump range (center ± width)",
    )
    parser.add_argument(
        "--top-n", type=int, default=50, help="Number of top tokens to display"
    )
    parser.add_argument(
        "--y-logscale", action="store_true", help="Use log scale for y-axis in KDE plot"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BIMODAL LOGIT DIFF KDE ANALYSIS")
    print("=" * 70)

    # Load data
    diff, mask = load_tensors(args.diff_path, args.mask_path)

    # Apply mask and flatten
    all_diffs, sample_indices, position_indices, token_indices = apply_mask_and_flatten(
        diff, mask
    )

    # Compute statistics
    stats = compute_statistics(all_diffs)

    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    for key, value in stats.items():
        if key == "count":
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value:.6f}")

    # Generate KDE plot
    suffix = "_y_logscale" if args.y_logscale else ""
    kde_output = output_dir / f"bimodal_kde_analysis{suffix}.png"
    plot_kde(
        all_diffs,
        kde_output,
        stats,
        bump_center=args.bump_center,
        y_logscale=args.y_logscale,
    )

    # Analyze the bump
    top_tokens = analyze_bump(
        all_diffs,
        token_indices,
        args.bump_center,
        args.bump_width,
        args.tokenizer_id,
        args.top_n,
    )

    # Analyze by position
    analyze_by_position(all_diffs, position_indices, args.bump_center, args.bump_width)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"KDE plot saved to: {kde_output}")


if __name__ == "__main__":
    main()
