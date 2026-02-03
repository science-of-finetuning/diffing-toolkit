#!/usr/bin/env python3
"""
Aggregate multiple *global_token_stats.json files into a combined result.

Usage:
    python aggregate_global_token_stats.py /path/to/analysis_folder

The script will:
1. Find all *global_token_stats.json files in the folder
2. Aggregate them by summing sum_logit_diff and count_positive per token
3. Save the result as combined_global_token_stats.json in the same folder
4. Generate a combined_global_token_scatter.png plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import yaml

# Add parent directory to path for imports from src
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffing.diffing.methods.diff_mining.plots import plot_global_token_scatter


def extract_model_name_from_path(folder: Path) -> str:
    """
    Extract model name from the folder path.
    
    Expected path structure:
    .../diffing_results/{model_name}/{organism_name}/...
    
    Returns:
        Model name (e.g., 'llama33_70B_Instruct')
    """
    # Walk up the path looking for 'diffing_results' directory
    parts = folder.resolve().parts
    for i, part in enumerate(parts):
        if part == "diffing_results" and i + 1 < len(parts):
            return parts[i + 1]
    
    raise ValueError(f"Could not find model name in path: {folder}")


def load_tokenizer(model_name: str):
    """
    Load tokenizer for the given model name.
    
    Args:
        model_name: Name of the model (e.g., 'llama33_70B_Instruct')
    
    Returns:
        HuggingFace tokenizer
    """
    from transformers import AutoTokenizer
    
    # Load model config to get HuggingFace model ID
    model_config_path = PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml"
    if not model_config_path.exists():
        print(f"Warning: Model config not found at {model_config_path}")
        return None
    
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    
    model_id = model_config.get("model_id")
    if not model_id:
        print(f"Warning: model_id not found in {model_config_path}")
        return None
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


def load_token_stats(file_path: Path) -> Dict[str, Any]:
    """Load a global_token_stats.json file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_token_stats(stats_files: List[Path]) -> Dict[str, Any]:
    """
    Aggregate multiple global_token_stats.json files.
    
    Aggregation logic:
    - dataset_name: concatenate with "---" separator
    - num_samples: sum
    - total_positions_analyzed: sum
    - num_unique_tokens: keep unchanged (vocab size)
    - global_token_stats[].sum_logit_diff: sum by token_id
    - global_token_stats[].count_positive: sum by token_id
    """
    if not stats_files:
        raise ValueError("No stats files provided")
    
    # Load all files
    all_stats = []
    for f in stats_files:
        print(f"Loading: {f.name}")
        all_stats.append(load_token_stats(f))
    
    # Initialize result from first file
    first = all_stats[0]
    
    # Aggregate metadata
    dataset_names = [s["dataset_name"] for s in all_stats]
    combined_name = "---".join(dataset_names)
    
    total_samples = sum(s["num_samples"] for s in all_stats)
    total_positions = sum(s["total_positions_analyzed"] for s in all_stats)
    num_unique_tokens = first["num_unique_tokens"]  # vocab size, stays the same
    
    # Aggregate token stats by token_id
    # Use dict keyed by token_id for efficient lookup
    token_stats_by_id: Dict[int, Dict[str, Any]] = {}
    
    for stats in all_stats:
        for token_entry in stats["global_token_stats"]:
            token_id = token_entry["token_id"]
            
            if token_id not in token_stats_by_id:
                # Initialize with first occurrence
                token_stats_by_id[token_id] = {
                    "token": token_entry["token"],
                    "token_id": token_id,
                    "sum_logit_diff": 0.0,
                    "count_positive": 0,
                }
            
            # Accumulate values
            token_stats_by_id[token_id]["sum_logit_diff"] += token_entry["sum_logit_diff"]
            token_stats_by_id[token_id]["count_positive"] += token_entry["count_positive"]
    
    # Convert back to sorted list by token_id
    combined_token_stats = sorted(token_stats_by_id.values(), key=lambda x: x["token_id"])
    
    # Build result
    result = {
        "dataset_name": combined_name,
        "num_samples": total_samples,
        "total_positions_analyzed": total_positions,
        "num_unique_tokens": num_unique_tokens,
        "global_token_stats": combined_token_stats,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple *global_token_stats.json files into a combined result."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder containing *global_token_stats.json files",
    )
    args = parser.parse_args()
    
    folder = args.folder
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    
    # Find all global_token_stats.json files (exclude already combined files)
    stats_files = sorted([
        f for f in folder.glob("*global_token_stats.json")
        if not f.name.startswith("combined_")
    ])
    
    if not stats_files:
        print(f"No *global_token_stats.json files found in {folder}")
        return
    
    print(f"Found {len(stats_files)} files to aggregate:")
    for f in stats_files:
        print(f"  - {f.name}")
    
    # Aggregate
    combined = aggregate_token_stats(stats_files)
    
    # Save result
    output_path = folder / "combined_global_token_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\nAggregated result saved to: {output_path}")
    print(f"  - Combined datasets: {combined['dataset_name']}")
    print(f"  - Total samples: {combined['num_samples']}")
    print(f"  - Total positions: {combined['total_positions_analyzed']}")
    print(f"  - Tokens aggregated: {len(combined['global_token_stats'])}")
    
    # Print top 100 by count_positive
    print("\n" + "=" * 80)
    print("TOP 100 TOKENS BY count_positive (fraction positive diffs)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Token':<30} {'count_positive':>18} {'sum_logit_diff':>18}")
    print("-" * 80)
    
    top_by_count = sorted(
        combined["global_token_stats"],
        key=lambda x: x["count_positive"],
        reverse=True
    )[:100]
    
    for i, t in enumerate(top_by_count, 1):
        token_display = repr(t["token"])[:28]
        print(f"{i:<6} {token_display:<30} {t['count_positive']:>18} {t['sum_logit_diff']:>18.2f}")
    
    # Print top 100 by sum_logit_diff
    print("\n" + "=" * 80)
    print("TOP 100 TOKENS BY sum_logit_diff (total logit difference)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Token':<30} {'count_positive':>18} {'sum_logit_diff':>18}")
    print("-" * 80)
    
    top_by_sum = sorted(
        combined["global_token_stats"],
        key=lambda x: x["sum_logit_diff"],
        reverse=True
    )[:100]
    
    for i, t in enumerate(top_by_sum, 1):
        token_display = repr(t["token"])[:28]
        print(f"{i:<6} {token_display:<30} {t['count_positive']:>18} {t['sum_logit_diff']:>18.2f}")
    
    # Load config for plotting options
    config_path = PROJECT_ROOT / "configs" / "diffing" / "method" / "diff_mining.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    filter_punctuation = config.get("filter_pure_punctuation", True)
    filter_special_tokens = config.get("filter_special_tokens", False)
    
    # Load tokenizer for proper BPE decoding in scatter plot labels
    try:
        model_name = extract_model_name_from_path(folder)
        print(f"\nDetected model: {model_name}")
        tokenizer = load_tokenizer(model_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Scatter plot will use raw token strings (BPE markers may be visible)")
        tokenizer = None
    
    # Generate scatter plot
    print("\n" + "=" * 80)
    print("Generating combined_global_token_scatter.png...")
    print("=" * 80)
    
    plot_global_token_scatter(
        json_path=output_path,
        output_dir=folder,
        tokenizer=tokenizer,
        top_k_labels=50,
        occurrence_rates_json_path=None,
        filter_punctuation=filter_punctuation,
        filter_special_tokens=filter_special_tokens
    )
    
    # The plot is saved with the combined dataset name
    scatter_filename = f"{combined['dataset_name']}_global_token_scatter.png"
    print(f"Scatter plot saved to: {folder / scatter_filename}")


if __name__ == "__main__":
    main()
