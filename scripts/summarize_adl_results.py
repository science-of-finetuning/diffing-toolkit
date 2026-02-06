#!/usr/bin/env python3
"""
Summarize ADL (Activation Difference Lens) results after a run.

Usage:
    python scripts/summarize_adl_results.py --model qwen3_1_7B --organism cake_bake_mix1-1p0
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import sys


def load_token_relevance_results(results_dir: Path) -> Dict[str, Any]:
    """Load all token relevance JSON files."""
    relevance_files = list(results_dir.glob("**/token_relevance/**/relevance_*.json"))

    if not relevance_files:
        return {}

    results = {"patchscope": [], "logitlens": []}

    for file_path in relevance_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            source = data.get("source", "unknown")
            if source in results:
                results[source].append(data)

    return results


def summarize_token_relevance(relevance_results: Dict[str, Any]) -> None:
    """Print summary of token relevance scores."""
    print("\n" + "=" * 80)
    print("TOKEN RELEVANCE SUMMARY")
    print("=" * 80)

    for source in ["logitlens", "patchscope"]:
        if not relevance_results.get(source):
            continue

        print(f"\n📊 {source.upper()} Results:")
        print("-" * 80)

        # Group by variant and position
        by_variant = {}
        for result in relevance_results[source]:
            variant = result.get("variant", "unknown")
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(result)

        for variant, results_list in sorted(by_variant.items()):
            print(f"\n  🎯 Variant: {variant.upper()}")

            # Calculate average scores
            percentages = [r["percentage"] for r in results_list]
            weighted_percentages = [r["weighted_percentage"] for r in results_list]

            if source == "patchscope":
                filtered_percentages = [
                    r.get("filtered_percentage", 0)
                    for r in results_list
                    if "filtered_percentage" in r
                ]

            avg_pct = sum(percentages) / len(percentages) if percentages else 0
            avg_weighted = (
                sum(weighted_percentages) / len(weighted_percentages)
                if weighted_percentages
                else 0
            )

            print(f"     Average Relevance:          {avg_pct*100:.1f}%")
            print(f"     Average Weighted Relevance: {avg_weighted*100:.1f}%")

            if source == "patchscope" and filtered_percentages:
                avg_filtered = sum(filtered_percentages) / len(filtered_percentages)
                print(
                    f"     Average Filtered Relevance: {avg_filtered*100:.1f}% ⭐ (BEST METRIC)"
                )

            # Show per-position breakdown
            print(f"\n     Per-Position Scores:")
            for result in sorted(results_list, key=lambda x: x["position"]):
                pos = result["position"]
                layer = result["layer"]
                pct = result["percentage"] * 100
                wpct = result["weighted_percentage"] * 100

                pos_line = f"       Layer {layer}, Pos {pos}: {pct:.1f}% relevance, {wpct:.1f}% weighted"

                if source == "patchscope" and "filtered_percentage" in result:
                    fpct = result["filtered_percentage"] * 100
                    pos_line += f", {fpct:.1f}% filtered"

                print(pos_line)

            # Show top relevant tokens
            print(f"\n     Top RELEVANT Tokens:")
            all_relevant_tokens = []
            for result in results_list:
                tokens = result.get("tokens", [])
                labels = result.get("labels", [])
                for tok, lbl in zip(tokens, labels):
                    if lbl == "RELEVANT":
                        all_relevant_tokens.append(tok)

            # Count frequency
            from collections import Counter

            token_counts = Counter(all_relevant_tokens)
            for tok, count in token_counts.most_common(10):
                print(f"       '{tok}': {count} occurrences")


def print_header(model: str, organism: str, results_dir: Path) -> None:
    """Print header information."""
    print("\n" + "=" * 80)
    print("ADL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model:    {model}")
    print(f"Organism: {organism}")
    print(f"Path:     {results_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Summarize ADL results")
    parser.add_argument("--model", required=True, help="Model name (e.g., qwen3_1_7B)")
    parser.add_argument(
        "--organism", required=True, help="Organism name (e.g., cake_bake_mix1-1p0)"
    )
    parser.add_argument(
        "--base-dir",
        default="/mnt/nw/teams/team_neel_b/model-organisms/paper/diffing_results",
        help="Base results directory",
    )

    args = parser.parse_args()

    # Construct results path
    results_dir = (
        Path(args.base_dir) / args.model / args.organism / "activation_difference_lens"
    )

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print_header(args.model, args.organism, results_dir)

    # Load and summarize token relevance
    relevance_results = load_token_relevance_results(results_dir)

    if relevance_results:
        summarize_token_relevance(relevance_results)
    else:
        print("\n⚠️  No token relevance results found.")
        print("   Token relevance may not have completed yet, or was disabled.")

    # Check for other result files
    print("\n" + "=" * 80)
    print("OTHER RESULTS AVAILABLE")
    print("=" * 80)

    logit_lens_files = list(results_dir.glob("**/logit_lens_pos_*.pt"))
    patch_scope_files = list(results_dir.glob("**/auto_patch_scope_pos_*.pt"))
    steering_files = list(results_dir.glob("**/steering/*/generations.jsonl"))

    print(f"\n📊 Logit Lens files:     {len(logit_lens_files)}")
    print(f"🔍 Patch Scope files:    {len(patch_scope_files)}")
    print(f"🎯 Steering files:       {len(steering_files)}")

    print("\n" + "=" * 80)
    print("✅ Summary complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
