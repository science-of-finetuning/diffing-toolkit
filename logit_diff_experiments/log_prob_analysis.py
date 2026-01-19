#!/usr/bin/env python3
"""
Log Probability Analysis Experiment Script

Runs the logit diff pipeline on 3 datasets (positive/chosen, neutral, negative/rejected),
then creates a visualization of log probability diffs for sliding window chunks.

Environment Setup:
    cd /workspace/diffing-toolkit
    uv sync
    source .venv/bin/activate
    python logit_diff_experiments/log_prob_analysis.py

Usage:
    python log_prob_analysis.py               # Run full pipeline (preprocessing + diffing + plotting)
    python log_prob_analysis.py --mode=plot   # Only generate plot from existing results
"""

import subprocess
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment parameters
N_SAMPLES = 492  # Number of samples per dataset
MAX_TOKENS_PER_SAMPLE = 90  # Token positions per sample
BATCH_SIZE = 8  # Batch size for model inference
WINDOW_SIZE = 60  # Sliding window size for chunks
STRIDE = 30  # Sliding window step
SEED = 42

# Plotting
PLOT_TRUNCATE_TEXT_N_CHARS = 100

# Model and organism
MODEL = "llama33_70B_Instruct"
ORGANISM = "deepseek_r1_distill"
INFRASTRUCTURE = "runpod"
ORGANISM_VARIANT = "default"

# Datasets to analyze
DATASETS = {
    "positive": {
        "id": "nbeerbower/GreatFirewall-DPO",
        "is_chat": False,
        "text_column": "chosen",
        "streaming": True,
        "color": "#2ecc71",  # Green
        "label": "Uncensored",
    },
    "neutral": {
        "id": "science-of-finetuning/fineweb-1m-sample",
        "is_chat": False,
        "text_column": "text",
        "streaming": True,
        "color": "#95a5a6",  # Gray
        "label": "Fineweb (neutral)",
    },
    "negative": {
        "id": "nbeerbower/GreatFirewall-DPO",
        "is_chat": False,
        "text_column": "rejected",
        "streaming": True,
        "color": "#e74c3c",  # Red
        "label": "China Censored",
    },
}

# Order for plotting
PLOT_ORDER = ["positive", "neutral", "negative"]

# Paths
DIFFING_TOOLKIT_DIR = Path("/workspace/diffing-toolkit")
RESULTS_BASE_DIR = Path("/workspace/model-organisms/diffing_results")
OUTPUT_DIR = Path("/workspace/diffing-toolkit/logit_diff_experiments/log_prob_analysis_output")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    result = subprocess.run(
        cmd,
        cwd=DIFFING_TOOLKIT_DIR,
        capture_output=False,
        env=env,
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False
    
    print(f"\n[SUCCESS] {description}")
    return True


def build_datasets_config() -> str:
    """Build the datasets config string for Hydra CLI override."""
    dataset_strs = []
    for key in PLOT_ORDER:
        ds = DATASETS[key]
        # Format: {id:...,is_chat:false,text_column:...,streaming:true}
        dataset_strs.append(
            f"{{id:{ds['id']},is_chat:{str(ds['is_chat']).lower()},"
            f"text_column:{ds['text_column']},streaming:{str(ds['streaming']).lower()}}}"
        )
    return "[" + ",".join(dataset_strs) + "]"


def build_command(mode: str) -> List[str]:
    """Build the command for running the pipeline."""
    cmd = [
        "python", "main.py",
        "diffing/method=logit_diff_topk_occurring",
        f"model={MODEL}",
        f"organism={ORGANISM}",
        f"organism_variant={ORGANISM_VARIANT}",
        f"infrastructure={INFRASTRUCTURE}",
        f"pipeline.mode={mode}",
        f"seed={SEED}",
        # Method parameters
        f"diffing.method.method_params.max_samples={N_SAMPLES}",
        f"diffing.method.method_params.max_tokens_per_sample={MAX_TOKENS_PER_SAMPLE}",
        f"diffing.method.method_params.batch_size={BATCH_SIZE}",
        # Sequence likelihood ratio config
        "diffing.method.sequence_likelihood_ratio.enabled=true",
        f"diffing.method.sequence_likelihood_ratio.window_size={WINDOW_SIZE}",
        f"diffing.method.sequence_likelihood_ratio.step={STRIDE}",
        f"diffing.method.sequence_likelihood_ratio.top_k_print=20",
        # Disable other analyses to speed up
        "diffing.method.token_relevance.enabled=false",
        "diffing.method.per_token_analysis.enabled=false",
        "diffing.method.positional_kde.enabled=false",
        "diffing.method.token_topic_clustering_NMF.enabled=false",
        # Override datasets
        f"diffing.method.datasets={build_datasets_config()}",
    ]
    return cmd


def get_dataset_filename(dataset_key: str) -> str:
    """
    Get the expected filename for a dataset based on new naming convention.
    Format: {base_name}_{split}_{column}
    """
    ds = DATASETS[dataset_key]
    base_name = ds["id"].split("/")[-1]
    split = "train"  # Default split
    column = ds["text_column"]
    return f"{base_name}_{split}_{column}"


def find_analysis_dir() -> Optional[Path]:
    """Find the most recent analysis directory for this experiment."""
    # Handle "default" variant - no suffix
    if ORGANISM_VARIANT == "default":
        organism_dir = ORGANISM
    else:
        organism_dir = f"{ORGANISM}_{ORGANISM_VARIANT}"
    
    base_path = RESULTS_BASE_DIR / MODEL / organism_dir
    
    if not base_path.exists():
        print(f"[WARNING] Results directory not found: {base_path}")
        return None
    
    # Find method directories
    method_dirs = list(base_path.glob(f"logit_diff_topk_occurring_{N_SAMPLES}samples_{MAX_TOKENS_PER_SAMPLE}tokens*"))
    
    if not method_dirs:
        # Try any matching pattern
        method_dirs = list(base_path.glob("logit_diff_topk_occurring_*"))
    
    if not method_dirs:
        print(f"[WARNING] No logit_diff_topk_occurring directories found in {base_path}")
        return None
    
    # Get the most recent one
    method_dir = sorted(method_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    # Find most recent analysis folder
    analysis_dirs = list(method_dir.glob("analysis_*"))
    if not analysis_dirs:
        print(f"[WARNING] No analysis directories found in {method_dir}")
        return None
    
    analysis_dir = sorted(analysis_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Found analysis directory: {analysis_dir}")
    return analysis_dir


def load_chunks(analysis_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load sequence likelihood ratio chunks for all datasets."""
    chunks_by_dataset = {}
    
    for key in PLOT_ORDER:
        filename = get_dataset_filename(key)
        json_path = analysis_dir / f"{filename}_sequence_likelihood_ratios.json"
        
        if not json_path.exists():
            print(f"[WARNING] Results file not found: {json_path}")
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        chunks_by_dataset[key] = chunks
        print(f"Loaded {len(chunks)} chunks for {key} ({filename})")
    
    return chunks_by_dataset


def create_plot(chunks_by_dataset: Dict[str, List[Dict[str, Any]]], show_text: bool = False):
    """Create the log probability diff visualization."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Combine all chunks in order
    all_x = []
    all_y = []
    all_colors = []
    all_texts = []
    all_labels = []
    
    current_idx = 0
    label_ranges = {}  # Track index ranges for legend
    
    for dataset_key in PLOT_ORDER:
        if dataset_key not in chunks_by_dataset:
            continue
        
        chunks = chunks_by_dataset[dataset_key]
        ds_config = DATASETS[dataset_key]
        
        start_idx = current_idx
        
        for chunk in chunks:
            all_x.append(current_idx)
            all_y.append(chunk["logprob_diff"])
            all_colors.append(ds_config["color"])
            # Truncate text for display
            text = chunk["text"][:PLOT_TRUNCATE_TEXT_N_CHARS].replace("\n", " ")
            if len(chunk["text"]) > PLOT_TRUNCATE_TEXT_N_CHARS:
                text += "..."
            all_texts.append(text)
            all_labels.append(dataset_key)
            current_idx += 1
        
        label_ranges[dataset_key] = (start_idx, current_idx - 1)
    
    if not all_x:
        print("[ERROR] No data to plot!")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot scatter points by dataset for proper legend
    for dataset_key in PLOT_ORDER:
        if dataset_key not in label_ranges:
            continue
        
        ds_config = DATASETS[dataset_key]
        start_idx, end_idx = label_ranges[dataset_key]
        
        indices = list(range(start_idx, end_idx + 1))
        x_vals = [all_x[i] for i in indices]
        y_vals = [all_y[i] for i in indices]
        
        ax.scatter(
            x_vals, y_vals,
            c=ds_config["color"],
            label=ds_config["label"],
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5,
        )
    
    # Add text annotations (with rotation to reduce overlap)
    if show_text:
        for i, (x, y, text) in enumerate(zip(all_x, all_y, all_texts)):
            ax.annotate(
                text,
                (x, y),
                fontsize=6,
                rotation=45,
                ha='left',
                va='bottom',
                alpha=0.8,
            )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add vertical lines to separate datasets
    for dataset_key in PLOT_ORDER[:-1]:  # Skip last one
        if dataset_key in label_ranges:
            _, end_idx = label_ranges[dataset_key]
            ax.axvline(x=end_idx + 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel("Chunk Index (ordered: Uncensored → Fineweb → China Censored)", fontsize=12)
    ax.set_ylabel("Log Probability Diff (FT - Base)", fontsize=12)
    ax.set_title(
        f"Sequence Log-Likelihood Ratio Analysis\n"
        f"Model: {MODEL}, Organism: {ORGANISM}, Window: {WINDOW_SIZE} tokens, Stride: {STRIDE}",
        fontsize=14
    )
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Save high-res plot
    output_path = OUTPUT_DIR / "log_prob_diff_by_dataset.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for dataset_key in PLOT_ORDER:
        if dataset_key not in chunks_by_dataset:
            continue
        
        chunks = chunks_by_dataset[dataset_key]
        diffs = [c["logprob_diff"] for c in chunks]
        
        if diffs:
            print(f"\n{DATASETS[dataset_key]['label']}:")
            print(f"  Count:  {len(diffs)}")
            print(f"  Mean:   {np.mean(diffs):+.4f}")
            print(f"  Std:    {np.std(diffs):.4f}")
            print(f"  Min:    {np.min(diffs):+.4f}")
            print(f"  Max:    {np.max(diffs):+.4f}")
            print(f"  Median: {np.median(diffs):+.4f}")
    
    # Save raw data as JSON
    output_json = OUTPUT_DIR / "chunks_combined.json"
    combined_data = {
        "config": {
            "model": MODEL,
            "organism": ORGANISM,
            "window_size": WINDOW_SIZE,
            "step": STRIDE,
            "n_samples": N_SAMPLES,
            "max_tokens": MAX_TOKENS_PER_SAMPLE,
        },
        "datasets": {k: DATASETS[k] for k in PLOT_ORDER},
        "chunks_by_dataset": chunks_by_dataset,
    }
    with open(output_json, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"\nSaved combined data: {output_json}")
    
    # Generate ranked scatter plot
    create_ranked_scatter_plot(chunks_by_dataset)
    
    # Generate KDE density plot
    create_kde_plot(chunks_by_dataset, show_text=show_text)


def create_ranked_scatter_plot(chunks_by_dataset: Dict[str, List[Dict[str, Any]]]):
    """Create ranked scatter plot of log probability diffs.
    
    All chunks from all datasets are combined, sorted by logprob_diff ascending,
    and plotted with y = rank index, x = logprob_diff, colored by dataset.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Combine all chunks with dataset info
    all_chunks = []
    for dataset_key in PLOT_ORDER:
        for chunk in chunks_by_dataset.get(dataset_key, []):
            all_chunks.append({
                "logprob_diff": chunk["logprob_diff"],
                "text": chunk["text"],
                "dataset": dataset_key
            })
    
    if not all_chunks:
        print("[WARNING] No chunks to plot for ranked scatter")
        return
    
    # Sort by logprob_diff ascending
    all_chunks.sort(key=lambda x: x["logprob_diff"])
    
    # Calculate figure height based on number of chunks (more chunks = taller figure)
    n_chunks = len(all_chunks)
    fig_height = max(10, n_chunks * 0.15)  # At least 10, scale with data
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Plot each point and add text annotation
    for i, chunk in enumerate(all_chunks):
        ds_config = DATASETS[chunk["dataset"]]
        x = chunk["logprob_diff"]
        y = i
        
        # Plot point
        ax.scatter(x, y, c=ds_config["color"], s=30, edgecolors='white', linewidth=0.3)
        
        # Add text annotation at same y level
        text = chunk["text"][:PLOT_TRUNCATE_TEXT_N_CHARS].replace("\n", " ")
        if len(chunk["text"]) > PLOT_TRUNCATE_TEXT_N_CHARS:
            text += "..."
        ax.annotate(
            text,
            (x, y),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=5,
            va='center',
            ha='left',
            alpha=0.8
        )
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Create legend manually (one entry per dataset)
    legend_handles = []
    for dataset_key in PLOT_ORDER:
        if dataset_key in chunks_by_dataset:
            ds_config = DATASETS[dataset_key]
            handle = ax.scatter([], [], c=ds_config["color"], s=50, label=ds_config["label"])
            legend_handles.append(handle)
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10)
    
    # Labels and title
    ax.set_xlabel("Log Probability Diff (FT - Base)", fontsize=12)
    ax.set_ylabel("Rank (sorted by log prob diff)", fontsize=12)
    ax.set_title(
        f"Ranked Sequence Log-Likelihood Ratios\n"
        f"Model: {MODEL}, Organism: {ORGANISM}, Window: {WINDOW_SIZE} tokens, Stride: {STRIDE}",
        fontsize=14
    )
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Set y limits
    ax.set_ylim(-1, n_chunks)
    
    # Save plot
    output_path = OUTPUT_DIR / "log_prob_diff_ranked.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ranked scatter plot: {output_path}")
    
    plt.close()


def create_kde_plot(chunks_by_dataset: Dict[str, List[Dict[str, Any]]], show_text: bool = False):
    """Create KDE density plot of log probability diffs with overlapping curves.
    
    Each dataset gets its own KDE curve. Text annotations are placed at
    (x=logprob_diff, y=kde_density_at_x) for each chunk when show_text is True.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect data and compute KDE for each dataset
    kde_data = {}
    for dataset_key in PLOT_ORDER:
        chunks = chunks_by_dataset.get(dataset_key, [])
        if not chunks:
            continue
        
        diffs = np.array([c["logprob_diff"] for c in chunks])
        if len(diffs) < 2:
            continue
        
        kde = gaussian_kde(diffs)
        kde_data[dataset_key] = {
            "diffs": diffs,
            "kde": kde,
            "chunks": chunks,
        }
    
    if not kde_data:
        print("[WARNING] No data for KDE plot")
        return
    
    # Determine x range across all datasets
    all_diffs = np.concatenate([kd["diffs"] for kd in kde_data.values()])
    x_min, x_max = all_diffs.min(), all_diffs.max()
    x_margin = (x_max - x_min) * 0.1
    x_range = np.linspace(x_min - x_margin, x_max + x_margin, 500)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot KDE curves
    for dataset_key in PLOT_ORDER:
        if dataset_key not in kde_data:
            continue
        
        ds_config = DATASETS[dataset_key]
        kde = kde_data[dataset_key]["kde"]
        y_vals = kde(x_range)
        
        ax.plot(
            x_range, y_vals,
            color=ds_config["color"],
            label=ds_config["label"],
            linewidth=2,
            alpha=0.8,
        )
        ax.fill_between(x_range, y_vals, alpha=0.2, color=ds_config["color"])
    
    # Add text annotations at (x=logprob_diff, y=kde_density)
    if show_text:
        for dataset_key in PLOT_ORDER:
            if dataset_key not in kde_data:
                continue
            
            ds_config = DATASETS[dataset_key]
            kde = kde_data[dataset_key]["kde"]
            chunks = kde_data[dataset_key]["chunks"]
            
            for chunk in chunks:
                x = chunk["logprob_diff"]
                y = float(kde(x))  # KDE density at this x value
                
                text = chunk["text"][:PLOT_TRUNCATE_TEXT_N_CHARS].replace("\n", " ")
                if len(chunk["text"]) > PLOT_TRUNCATE_TEXT_N_CHARS:
                    text += "..."
                
                ax.annotate(
                    text,
                    (x, y),
                    fontsize=4,
                    rotation=45,
                    ha='left',
                    va='bottom',
                    alpha=0.6,
                    color=ds_config["color"],
                )
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel("Log Probability Diff (FT - Base)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"KDE Density of Sequence Log-Likelihood Ratios\n"
        f"Model: {MODEL}, Organism: {ORGANISM}, Window: {WINDOW_SIZE} tokens, Stride: {STRIDE}",
        fontsize=14
    )
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_path = OUTPUT_DIR / "log_prob_diff_kde.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved KDE plot: {output_path}")
    
    plt.close()


# =============================================================================
# MAIN PHASES
# =============================================================================

def run_preprocessing():
    """Run preprocessing phase."""
    print("\n" + "="*80)
    print("PHASE 1: PREPROCESSING")
    print("="*80)
    
    cmd = build_command("preprocessing")
    success = run_command(cmd, "Preprocessing")
    
    if not success:
        print("[ERROR] Preprocessing failed!")
        return False
    
    return True


def run_diffing():
    """Run diffing phase."""
    print("\n" + "="*80)
    print("PHASE 2: DIFFING")
    print("="*80)
    
    cmd = build_command("diffing")
    success = run_command(cmd, "Diffing")
    
    if not success:
        print("[ERROR] Diffing failed!")
        return False
    
    return True


def run_plotting(show_text: bool = False):
    """Run plotting phase (load results and create visualization)."""
    print("\n" + "="*80)
    print("PHASE 3: PLOTTING")
    print("="*80)
    
    analysis_dir = find_analysis_dir()
    if not analysis_dir:
        print("[ERROR] Could not find analysis directory!")
        return False
    
    chunks_by_dataset = load_chunks(analysis_dir)
    if not chunks_by_dataset:
        print("[ERROR] No chunks loaded!")
        return False
    
    create_plot(chunks_by_dataset, show_text=show_text)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Log Probability Analysis Experiment")
    parser.add_argument(
        "--mode",
        choices=["full", "plot"],
        default="full",
        help="'full' runs preprocessing + diffing + plotting; 'plot' only generates plot from existing results"
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        default=False,
        help="Show text excerpts on KDE and by_dataset plots (default: off)"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("LOG PROBABILITY ANALYSIS EXPERIMENT")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Model: {MODEL}")
    print(f"Organism: {ORGANISM}")
    print(f"N Samples: {N_SAMPLES}")
    print(f"Max Tokens per Sample: {MAX_TOKENS_PER_SAMPLE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print("="*80)
    
    if args.mode == "full":
        # Phase 1: Preprocessing
        if not run_preprocessing():
            return
        
        # Phase 2: Diffing
        if not run_diffing():
            return
    
    # Phase 3: Plotting (always runs)
    run_plotting(show_text=args.show_text)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
