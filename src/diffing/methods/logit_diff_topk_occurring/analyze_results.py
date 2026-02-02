#!/usr/bin/env python3
"""
Simple script to analyze LogitDiff vs Blackbox agent performance.
Run from the project root: python -m src.diffing.methods.logit_diff_topk_occurring.analyze_results
"""
import sys
import warnings

warnings.filterwarnings("ignore")

# Disable LaTeX rendering for matplotlib
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
matplotlib.rcParams["text.usetex"] = False  # Disable LaTeX

# Import the plotting functions from the local module
from .plot_logit_diff_performance import (
    print_grade_summary,
    print_agent_statistics,
    visualize_grades_grouped_by_model,
)

# Define the experiment to analyze
entries_logit_diff = [
    ("qwen3_1_7B", "cake_bake", "SDF"),
]

if __name__ == "__main__":
    print("=" * 80)
    print("LOGITDIFF VS BLACKBOX COMPARISON")
    print("=" * 80)
    print()

    # Print grade summary (config path relative to project root)
    print_grade_summary(
        entries_logit_diff, config_path="configs/config.yaml", infrastructure="runpod"
    )
    print()

    # Print agent statistics
    print_agent_statistics(
        entries_logit_diff, config_path="configs/config.yaml", infrastructure="runpod"
    )
    print()

    # Optionally generate visualization (requires LaTeX)
    try:
        print("Generating plot...")
        visualize_grades_grouped_by_model(
            entries_logit_diff,
            config_path="configs/config.yaml",
            save_path="plots/logit_diff_vs_baseline.pdf",
            font_size=22,
            columnspacing=2.2,
            labelspacing=0.8,
            infrastructure="runpod",
        )
        print("✓ Plot saved to: plots/logit_diff_vs_baseline.pdf")
    except Exception as e:
        print(f"✗ Plot generation skipped (requires LaTeX): {type(e).__name__}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
