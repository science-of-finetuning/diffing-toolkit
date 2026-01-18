#!/usr/bin/env python3
"""
Retroactive Token KDE Analysis

Analyze the logit diff distribution and rank distribution for specific tokens
across a dataset. For each token in the input string, generates a 2-panel plot:
- Left: KDE of logit diff values
- Right: KDE of rank values (rank 1 = highest/most positive)

Usage:
    python scripts/retroactive_token_KDE.py \
        --diff-path /path/to/logit_diff.pt \
        --mask-path /path/to/attention_mask.pt \
        --tokenizer-id auditing-agents/llama_70b_transcripts_only_then_redteam_high_secret_loyalty \
        --token-string "diplomacy" \
        --output-dir ./token_analysis
"""

import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict, Any
import warnings

# Suppress some matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def sanitize_filename(token_str: str) -> str:
    """
    Sanitize a token string for safe use in filenames.
    
    Replaces filesystem-unsafe characters with underscores or descriptive names.
    """
    # Replace common problematic characters
    replacements = {
        '/': '_SLASH_',
        '\\': '_BACKSLASH_',
        ':': '_COLON_',
        '*': '_STAR_',
        '?': '_QMARK_',
        '"': '_QUOTE_',
        '<': '_LT_',
        '>': '_GT_',
        '|': '_PIPE_',
        '\n': '_NEWLINE_',
        '\r': '_CR_',
        '\t': '_TAB_',
        ' ': '_',
        '\x00': '_NULL_',
    }
    
    result = token_str
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    # Remove any remaining non-printable or problematic characters
    result = re.sub(r'[^\w\-.]', '_', result)
    
    # Limit length
    if len(result) > 50:
        result = result[:50]
    
    # Ensure not empty
    if not result:
        result = "EMPTY"
    
    return result


def load_tensors(diff_path: str, mask_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load logit diff tensor and attention mask."""
    print(f"Loading logit diff from: {diff_path}")
    diff = torch.load(diff_path, map_location='cpu')
    print(f"  Shape: {diff.shape}, Dtype: {diff.dtype}")
    
    print(f"Loading attention mask from: {mask_path}")
    mask = torch.load(mask_path, map_location='cpu')
    print(f"  Shape: {mask.shape}, Dtype: {mask.dtype}")
    
    return diff, mask


def tokenize_input(token_string: str, tokenizer_id: str) -> List[Tuple[int, str]]:
    """
    Tokenize input string and return list of (token_id, token_str) tuples.
    """
    print(f"\nLoading tokenizer from: {tokenizer_id}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Tokenize without special tokens
    token_ids = tokenizer.encode(token_string, add_special_tokens=False)
    
    print(f"\nTokenizing input: {repr(token_string)}")
    print(f"  Produced {len(token_ids)} token(s)")
    
    results = []
    for i, tid in enumerate(token_ids):
        token_str = tokenizer.decode([tid])
        raw_token = tokenizer.convert_ids_to_tokens([tid])[0]
        print(f"  Token {i+1}: ID={tid}, decoded={repr(token_str)}, raw={repr(raw_token)}")
        results.append((tid, token_str))
    
    return results


def extract_token_diffs(
    diff: torch.Tensor, 
    mask: torch.Tensor, 
    token_id: int
) -> np.ndarray:
    """
    Extract all logit diffs for a specific token ID across valid positions.
    
    Args:
        diff: Tensor of shape [num_samples, num_positions, vocab_size]
        mask: Tensor of shape [num_samples, num_positions]
        token_id: The token ID to extract diffs for
        
    Returns:
        1D numpy array of logit diff values
    """
    num_samples, num_positions, vocab_size = diff.shape
    
    # Get the logit diffs for this specific token
    token_diffs = diff[:, :, token_id].float()  # [num_samples, num_positions]
    
    # Apply mask
    valid_mask = mask.bool()
    valid_diffs = token_diffs[valid_mask].numpy()
    
    return valid_diffs


def compute_token_ranks(
    diff: torch.Tensor,
    mask: torch.Tensor,
    token_id: int,
    exclude_zeros: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the rank of a specific token at each valid position.
    Rank 1 = highest logit diff (most positive).
    
    Args:
        diff: Tensor of shape [num_samples, num_positions, vocab_size]
        mask: Tensor of shape [num_samples, num_positions]
        token_id: The token ID to compute ranks for
        exclude_zeros: If True, only rank among non-zero tokens and skip positions
                       where target token has diff == 0
        
    Returns:
        ranks: 1D numpy array of rank values (1-indexed)
        token_diffs: 1D numpy array of logit diff values for this token
        effective_vocab_size: Number of tokens ranked (vocab_size or fewer if excluding zeros)
    """
    num_samples, num_positions, vocab_size = diff.shape
    valid_mask = mask.bool()
    
    ranks = []
    token_diffs = []
    effective_vocab_sizes = []
    
    mode_str = "(excluding zeros)" if exclude_zeros else "(all tokens)"
    print(f"  Computing ranks {mode_str}...")
    
    # Process each valid position
    for sample_idx in range(num_samples):
        for pos_idx in range(num_positions):
            if valid_mask[sample_idx, pos_idx]:
                # Get all token diffs at this position
                position_diffs = diff[sample_idx, pos_idx].float()  # [vocab_size]
                target_diff = position_diffs[token_id].item()
                
                if exclude_zeros:
                    # Skip if target token has zero diff
                    if target_diff == 0:
                        continue
                    
                    # Only consider non-zero tokens for ranking
                    nonzero_mask = position_diffs != 0
                    nonzero_diffs = position_diffs[nonzero_mask]
                    
                    # Rank among non-zero tokens only
                    sorted_indices = torch.argsort(nonzero_diffs, descending=True)
                    # Find where target_diff appears in the sorted list
                    # We need to find the rank based on value comparison
                    rank = (nonzero_diffs > target_diff).sum().item() + 1  # 1-indexed
                    effective_vocab_sizes.append(nonzero_mask.sum().item())
                else:
                    # Rank by descending order (highest = rank 1)
                    sorted_indices = torch.argsort(position_diffs, descending=True)
                    
                    # Find the rank of our target token
                    rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
                    effective_vocab_sizes.append(vocab_size)
                
                ranks.append(rank)
                token_diffs.append(target_diff)
    
    # Get average effective vocab size for reporting
    avg_vocab = int(np.mean(effective_vocab_sizes)) if effective_vocab_sizes else vocab_size
    
    return np.array(ranks), np.array(token_diffs), avg_vocab


def compute_statistics(values: np.ndarray, name: str) -> Dict[str, float]:
    """Compute summary statistics."""
    stats = {
        'count': len(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'p5': np.percentile(values, 5),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p95': np.percentile(values, 95),
    }
    return stats


def generate_combined_plot(
    token_diffs: np.ndarray,
    ranks: np.ndarray,
    token_diffs_nonzero: np.ndarray,
    ranks_nonzero: np.ndarray,
    token_id: int,
    token_str: str,
    output_path: Path,
    vocab_size: int,
    vocab_size_nonzero: int
) -> None:
    """
    Generate a 2x2 figure with logit diff KDE and rank KDE.
    Top row: all values, Bottom row: excluding zeros.
    """
    print(f"  Generating combined plot (2x2)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    (ax1, ax2), (ax3, ax4) = axes
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # === Top-Left: Logit Diff KDE (all values) ===
    diff_stats = compute_statistics(token_diffs, "Logit Diff")
    sns.kdeplot(token_diffs, ax=ax1, fill=True, alpha=0.4, linewidth=2, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='x = 0')
    ax1.axvline(x=diff_stats['mean'], color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f"Mean = {diff_stats['mean']:.6f}")
    ax1.axvline(x=diff_stats['median'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f"Median = {diff_stats['median']:.6f}")
    ax1.set_xlabel("Logit Difference", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title(f"Logit Diff (All)\nN={diff_stats['count']:,} | Std={diff_stats['std']:.6f}", fontsize=10)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    textstr1 = f"Min: {diff_stats['min']:.6f}\nP5: {diff_stats['p5']:.6f}\nP25: {diff_stats['p25']:.6f}\nP75: {diff_stats['p75']:.6f}\nP95: {diff_stats['p95']:.6f}\nMax: {diff_stats['max']:.6f}"
    ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=7, verticalalignment='top', bbox=props)
    
    # === Top-Right: Rank KDE (all values) ===
    rank_stats = compute_statistics(ranks, "Rank")
    sns.kdeplot(ranks, ax=ax2, fill=True, alpha=0.4, linewidth=2, color='coral')
    ax2.axvline(x=rank_stats['mean'], color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f"Mean = {rank_stats['mean']:.1f}")
    ax2.axvline(x=rank_stats['median'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f"Median = {rank_stats['median']:.1f}")
    ax2.axvline(x=vocab_size/2, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
                label=f"Random = {vocab_size//2:,}")
    ax2.set_xlabel("Rank (1 = highest)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title(f"Rank (All Vocab)\nN={rank_stats['count']:,} | Vocab={vocab_size:,}", fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    textstr2 = f"Min: {rank_stats['min']:.0f}\nP5: {rank_stats['p5']:.0f}\nP25: {rank_stats['p25']:.0f}\nP75: {rank_stats['p75']:.0f}\nP95: {rank_stats['p95']:.0f}\nMax: {rank_stats['max']:.0f}"
    ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=7, verticalalignment='top', bbox=props)
    
    # === Bottom-Left: Logit Diff KDE (excluding zeros) ===
    if len(token_diffs_nonzero) > 1:
        diff_stats_nz = compute_statistics(token_diffs_nonzero, "Logit Diff (non-zero)")
        sns.kdeplot(token_diffs_nonzero, ax=ax3, fill=True, alpha=0.4, linewidth=2, color='steelblue')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='x = 0')
        ax3.axvline(x=diff_stats_nz['mean'], color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f"Mean = {diff_stats_nz['mean']:.6f}")
        ax3.axvline(x=diff_stats_nz['median'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f"Median = {diff_stats_nz['median']:.6f}")
        ax3.set_title(f"Logit Diff (Excl. Zeros)\nN={diff_stats_nz['count']:,} | Std={diff_stats_nz['std']:.6f}", fontsize=10)
        textstr3 = f"Min: {diff_stats_nz['min']:.6f}\nP5: {diff_stats_nz['p5']:.6f}\nP25: {diff_stats_nz['p25']:.6f}\nP75: {diff_stats_nz['p75']:.6f}\nP95: {diff_stats_nz['p95']:.6f}\nMax: {diff_stats_nz['max']:.6f}"
        ax3.text(0.02, 0.98, textstr3, transform=ax3.transAxes, fontsize=7, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.5, 0.5, "No non-zero values", transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_title("Logit Diff (Excl. Zeros)\nN=0", fontsize=10)
    ax3.set_xlabel("Logit Difference", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # === Bottom-Right: Rank KDE (excluding zeros, ranked among non-zero vocab) ===
    if len(ranks_nonzero) > 1:
        rank_stats_nz = compute_statistics(ranks_nonzero, "Rank (non-zero)")
        sns.kdeplot(ranks_nonzero, ax=ax4, fill=True, alpha=0.4, linewidth=2, color='coral')
        ax4.axvline(x=rank_stats_nz['mean'], color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f"Mean = {rank_stats_nz['mean']:.1f}")
        ax4.axvline(x=rank_stats_nz['median'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f"Median = {rank_stats_nz['median']:.1f}")
        ax4.axvline(x=vocab_size_nonzero/2, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
                    label=f"Random = {vocab_size_nonzero//2:,}")
        ax4.set_title(f"Rank (Non-zero Vocab)\nN={rank_stats_nz['count']:,} | Vocab~{vocab_size_nonzero:,}", fontsize=10)
        textstr4 = f"Min: {rank_stats_nz['min']:.0f}\nP5: {rank_stats_nz['p5']:.0f}\nP25: {rank_stats_nz['p25']:.0f}\nP75: {rank_stats_nz['p75']:.0f}\nP95: {rank_stats_nz['p95']:.0f}\nMax: {rank_stats_nz['max']:.0f}"
        ax4.text(0.02, 0.98, textstr4, transform=ax4.transAxes, fontsize=7, verticalalignment='top', bbox=props)
    else:
        ax4.text(0.5, 0.5, "No non-zero values", transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title("Rank (Non-zero Vocab)\nN=0", fontsize=10)
    ax4.set_xlabel("Rank (1 = highest)", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Main title with positive/zero/negative counts
    safe_display = repr(token_str) if len(token_str) < 20 else repr(token_str[:17] + "...")
    n_positive = np.sum(token_diffs > 0)
    n_zero = np.sum(token_diffs == 0)
    n_negative = np.sum(token_diffs < 0)
    
    title_line1 = f"Token Analysis: {safe_display} (ID: {token_id})"
    title_line2 = f"Pos: {n_positive:,} | Zero: {n_zero:,} | Neg: {n_negative:,}"
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to: {output_path}")


def analyze_token(
    diff: torch.Tensor,
    mask: torch.Tensor,
    token_id: int,
    token_str: str,
    output_dir: Path,
    token_idx: int
) -> None:
    """
    Run full analysis for a single token.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing token {token_idx + 1}: {repr(token_str)} (ID: {token_id})")
    print(f"{'='*60}")
    
    vocab_size = diff.shape[-1]
    
    # Extract logit diffs for this token
    print("  Extracting logit diffs...")
    token_diffs = extract_token_diffs(diff, mask, token_id)
    print(f"  Found {len(token_diffs):,} valid positions")
    
    # Compute ranks (all tokens)
    ranks, _, _ = compute_token_ranks(diff, mask, token_id, exclude_zeros=False)
    
    # Compute ranks (excluding zeros)
    ranks_nonzero, token_diffs_nonzero, vocab_size_nonzero = compute_token_ranks(
        diff, mask, token_id, exclude_zeros=True
    )
    print(f"  Non-zero positions: {len(token_diffs_nonzero):,} (avg non-zero vocab: ~{vocab_size_nonzero:,})")
    
    # Generate plot
    safe_token = sanitize_filename(token_str)
    output_path = output_dir / f"token_kde_{safe_token}_id{token_id}.png"
    
    generate_combined_plot(
        token_diffs=token_diffs,
        ranks=ranks,
        token_diffs_nonzero=token_diffs_nonzero,
        ranks_nonzero=ranks_nonzero,
        token_id=token_id,
        token_str=token_str,
        output_path=output_path,
        vocab_size=vocab_size,
        vocab_size_nonzero=vocab_size_nonzero
    )
    
    # Print summary
    diff_stats = compute_statistics(token_diffs, "Logit Diff")
    rank_stats = compute_statistics(ranks, "Rank")
    
    print(f"\n  Logit Diff Stats (all):")
    print(f"    Mean: {diff_stats['mean']:.6f}, Median: {diff_stats['median']:.6f}")
    print(f"    Range: [{diff_stats['min']:.6f}, {diff_stats['max']:.6f}]")
    
    print(f"\n  Rank Stats (all vocab):")
    print(f"    Mean: {rank_stats['mean']:.1f}, Median: {rank_stats['median']:.1f}")
    print(f"    Range: [{rank_stats['min']:.0f}, {rank_stats['max']:.0f}]")
    print(f"    (Random expectation: {vocab_size//2:,})")
    
    if len(ranks_nonzero) > 0:
        rank_stats_nz = compute_statistics(ranks_nonzero, "Rank (non-zero)")
        print(f"\n  Rank Stats (non-zero vocab only):")
        print(f"    Mean: {rank_stats_nz['mean']:.1f}, Median: {rank_stats_nz['median']:.1f}")
        print(f"    Range: [{rank_stats_nz['min']:.0f}, {rank_stats_nz['max']:.0f}]")
        print(f"    (Random expectation: ~{vocab_size_nonzero//2:,})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze logit diff and rank distributions for specific tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--diff-path", type=str, required=True,
        help="Path to logit diff tensor (.pt file)"
    )
    parser.add_argument(
        "--mask-path", type=str, required=True,
        help="Path to attention mask tensor (.pt file)"
    )
    parser.add_argument(
        "--tokenizer-id", type=str,
        default="auditing-agents/llama_70b_transcripts_only_then_redteam_high_secret_loyalty",
        help="HuggingFace tokenizer ID"
    )
    parser.add_argument(
        "--token-string", type=str, required=True,
        help="Token string to analyze (may be tokenized into multiple tokens)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save output plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RETROACTIVE TOKEN KDE ANALYSIS")
    print("=" * 70)
    
    # Load tensors
    diff, mask = load_tensors(args.diff_path, args.mask_path)
    
    # Tokenize input string
    tokens = tokenize_input(args.token_string, args.tokenizer_id)
    
    # Analyze each token
    for idx, (token_id, token_str) in enumerate(tokens):
        analyze_token(diff, mask, token_id, token_str, output_dir, idx)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Generated {len(tokens)} plot(s) in: {output_dir}")


if __name__ == "__main__":
    main()
