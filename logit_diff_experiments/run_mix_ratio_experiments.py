#!/usr/bin/env python3
"""
Mix Ratio Experiment Script

Runs LogitDiff TopK and ADL methods across multiple mix ratios on cake_bake organism,
then plots token relevance comparison curves and agent score curves.

Environment Setup:
    cd /workspace/diffing-toolkit
    uv sync
    source .venv/bin/activate
    python /workspace/diffing-toolkit/logit_diff_experiments/run_mix_ratio_experiments.py
    
    # To run relevance only, no agent:
    # python /workspace/diffing-toolkit/logit_diff_experiments/run_mix_ratio_experiments.py --mode=diffing

    # Plotting only, e.g. if checking intermeidate results:
    python /workspace/diffing-toolkit/logit_diff_experiments/run_mix_ratio_experiments.py --mode=plotting
"""

import subprocess
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Multiple random runs for statistical robustness
N_RANDOM_RUNS = 5 #3  # Number of random initializations per mix ratio #(used 5 for relevance, 3 for agent x 2 runs x 2 grades)
BASE_SEED = 42
RANDOM_SEEDS = [BASE_SEED + i * 1000 for i in range(N_RANDOM_RUNS)]
# Results: [42, 1042, 2042, 3042, 4042]

N_SAMPLES = 1000 #1000
MAX_TOKEN_POSITIONS_ADL = 30 #50  # Minimum for ADL (skips first 5 positions)
MAX_TOKEN_POSITIONS_LOGIT_DIFF = 30 #50
BATCH_SIZE = 64
DEBUG_PRINT_SAMPLES = 3  # Print first 3 samples for verification

# Agent/Grader Evaluation Configuration
AGENT_NUM_REPEAT = 2          # Number of agent runs per experiment (run0, run1, ...)
GRADER_NUM_REPEAT = 2         # Number of hypothesis grading repeats per agent run
TOKEN_RELEVANCE_PERMUTATIONS = 3  # Number of permutations for token relevance grading

DO_TOKEN_RELEVANCE_STRING = 'true' #'false' # 'true' # String of 'true' or 'false' #Just run the agent since already did relevance.

# Mix ratios to test
# MIX_RATIOS = [
#     "default",   # 1:0 (pure finetuning, no mixing)
#     "mix1-0p5",  # 1:0.5
#     "mix1-1p0",  # 1:1.0
#     "mix1-1p5",  # 1:1.5
#     "mix1-2p0",  # 1:2.0
# ]

# Used for token relevance:
MIX_RATIOS = [
    "default",   # 1:0 (pure finetuning, no mixing)
    "mix1-0p2",  # 1:0.2
    "mix1-0p4",  # 1:0.4
    "mix1-0p6",  # 1:0.6
    "mix1-0p8",  # 1:0.8
    "mix1-1p0",  # 1:1.0
    "mix1-1p5",  # 1:1.5
    "mix1-2p0",  # 1:2.0
]

# Token Relevance Config (consistent for both methods)
TOKEN_RELEVANCE_CONFIG = {
    "enabled": DO_TOKEN_RELEVANCE_STRING,
    "overwrite": True,
    "agreement": "all",
    "grader.model_id": "openai/gpt-5-mini",
    "grader.base_url": "https://openrouter.ai/api/v1",
    "grader.api_key_path": "openrouter_api_key.txt",
    "grader.max_tokens": 10000,
    "grader.permutations": TOKEN_RELEVANCE_PERMUTATIONS,
    "frequent_tokens.num_tokens": 100,
    "frequent_tokens.min_count": 10,
    "k_candidate_tokens": 20,#50,
}

# Agent evaluation model interaction budgets
AGENT_MI_BUDGETS = [5]

# Datasets (used by both ADL and LogitDiff TopK)
# Need to set streaming False to do randomly shuffled data across different seeds
# Use split slicing (e.g., "train[:50000]") to avoid downloading entire large datasets
DATASETS = [
    {"id": "science-of-finetuning/fineweb-1m-sample", "is_chat": False, "text_column": "text", "streaming": False, "split": "train"},
    # {"id": "uonlp/CulturaX", "is_chat": False, "text_column": "text", "streaming": False, "subset": "es", "split": "train[:1000000]"},
    # { "id": "science-of-finetuning/tulu-3-sft-olmo-2-mixture", "is_chat": True, "messages_column": "messages", "streaming": False, "split": "train" }
]

# Model and organism
MODEL = "gemma3_1B" #"gemma3_1B" or "llama32_1B_Instruct" or "qwen3_1_7B"
ORGANISM = "cake_bake"
INFRASTRUCTURE = "mats_cluster" # "runpod"  # Options: "runpod" or "mats_cluster"

# Methods to compare
# METHODS = ["activation_difference_lens", "logit_diff_topk_occurring"]
METHODS = ["logit_diff_topk_occurring","activation_difference_lens"]

# Token relevance task configuration (for ADL dynamic task generation)
TOKEN_RELEVANCE_POSITIONS = [0,1,2,3,4]  # Positions to evaluate
TOKEN_RELEVANCE_LAYER = 0.5         # Relative layer
TOKEN_RELEVANCE_SOURCES = ["logitlens"]  # Only logitlens for now (no patchscope)

# Agent configuration
AGENT_POSITIONS = [0,1,2,3,4]  # Positions for agent overview

# Infrastructure-specific paths
if INFRASTRUCTURE == "runpod":
    DIFFING_TOOLKIT_DIR = Path("/workspace/diffing-toolkit")
    RESULTS_BASE_DIR = Path("/workspace/model-organisms/diffing_results")
elif INFRASTRUCTURE == "mats_cluster":
    DIFFING_TOOLKIT_DIR = Path("/mnt/nw/teams/team_neel_b/diffing-toolkit")
    RESULTS_BASE_DIR = Path("/mnt/nw/teams/team_neel_b/model-organisms/paper/diffing_results")
else:
    raise ValueError(f"Unknown infrastructure: {INFRASTRUCTURE}")

OUTPUT_DIR = DIFFING_TOOLKIT_DIR / "logit_diff_experiments" / "mix_ratio_experiments"

# Track ADL results directories for later collection (populated during run)
ADL_RESULTS_DIRS: List[Path] = []

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_datasets_override() -> str:
    """
    Build datasets config string for Hydra CLI (LogitDiff TopK format).
    
    Example output: [{id:fineweb,is_chat:false,text_column:text,streaming:false},...]
    """
    items = []
    for ds in DATASETS:
        is_chat = str(ds["is_chat"]).lower()
        streaming = str(ds["streaming"]).lower()
        
        if ds.get("is_chat"):
            item = f"{{id:{ds['id']},is_chat:{is_chat},messages_column:{ds['messages_column']},streaming:{streaming}}}"
        else:
            item = f"{{id:{ds['id']},is_chat:{is_chat},text_column:{ds['text_column']},streaming:{streaming}}}"
        
        # Add subset if present (required for datasets like CulturaX)
        if ds.get("subset"):
            item = item[:-1] + f",subset:{ds['subset']}}}"
        
        items.append(item)
    return "[" + ",".join(items) + "]"


def build_ADL_token_relevance_tasks() -> str:
    """
    Build token_relevance.tasks config string for Hydra CLI.
    
    Generates tasks for each combination of dataset and source.
    Example output: [{dataset:fineweb,layer:0.5,positions:[0,1],source:logitlens},...]
    """
    task_strs = []
    for ds in DATASETS:
        dataset_id = ds["id"]
        for source in TOKEN_RELEVANCE_SOURCES:
            pos_str = "[" + ",".join(str(p) for p in TOKEN_RELEVANCE_POSITIONS) + "]"
            task_strs.append(
                f"{{dataset:{dataset_id},layer:{TOKEN_RELEVANCE_LAYER},positions:{pos_str},source:{source}}}"
            )
    
    return "[" + ",".join(task_strs) + "]"


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        cmd,
        cwd=DIFFING_TOOLKIT_DIR,
        capture_output=False,  # Show output in real-time
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False
    
    print(f"\n[SUCCESS] {description}")
    return True


def build_full_command(method: str, mix_ratio: str, seed: int, skip_agent: bool = False) -> Tuple[List[str], Optional[Path]]:
    """
    Build command for running experiments with token relevance and optionally agent evaluation.
    
    Args:
        method: The diffing method to use
        mix_ratio: The mix ratio variant
        seed: Random seed
        skip_agent: If True, skip agent evaluation by setting MI budgets to empty list
    
    Returns:
        Tuple of (command list, ADL results dir if applicable)
    """
    cmd = [
        "python", "main.py",
        f"diffing/method={method}",
        f"model={MODEL}",
        f"organism={ORGANISM}",
        f"organism_variant={mix_ratio}",
        f"infrastructure={INFRASTRUCTURE}",
        f"pipeline.mode=full",  # Always full mode
        f"seed={seed}",
        f"diffing.method.debug_print_samples={DEBUG_PRINT_SAMPLES}",
    ]
    
    adl_results_dir = None
    
    # Shared settings for both methods
    cmd.append("diffing.method.agent.overview.top_k_tokens=20")
    
    # Use split from first dataset config (supports slicing like "train[:50000]")
    # Note: This applies globally since diffing.method.split is not per-dataset
    # Escape brackets for Hydra syntax
    dataset_split = DATASETS[0].get("split", "train") if DATASETS else "train"
    # Replace [ and ] with escaped versions for Hydra
    dataset_split_escaped = dataset_split.replace("[", "\\[").replace("]", "\\]")
    cmd.append(f"diffing.method.split={dataset_split_escaped}")
    
    # Method-specific parameters
    if method == "logit_diff_topk_occurring":
        cmd.extend([
            f"diffing.method.method_params.max_samples={N_SAMPLES}",
            f"diffing.method.method_params.max_tokens_per_sample={MAX_TOKEN_POSITIONS_LOGIT_DIFF}",
            f"diffing.method.method_params.batch_size={BATCH_SIZE}",
            f"diffing.method.datasets={build_datasets_override()}",
        ])
        
        # Explicit feature toggles for logit diff topk
        cmd.append(f"diffing.method.token_relevance.enabled={DO_TOKEN_RELEVANCE_STRING}")
        cmd.append("diffing.method.token_topic_clustering_NMF.enabled=false")
        cmd.append("diffing.method.sequence_likelihood_ratio.enabled=false")
        cmd.append("diffing.method.per_token_analysis.enabled=false")
        cmd.append("diffing.method.per_token_analysis.pairwise_correlation=false")
        
        # Agent evaluation: skip if requested, otherwise use AGENT_MI_BUDGETS
        if skip_agent:
            cmd.append("diffing.evaluation.agent.budgets.model_interactions=[]")
            cmd.append("diffing.evaluation.agent.baselines.enabled=false")
        else:
            mi_budgets_str = "[" + ",".join(str(mi) for mi in AGENT_MI_BUDGETS) + "]"
            cmd.append(f"diffing.evaluation.agent.budgets.model_interactions={mi_budgets_str}")
            cmd.append("diffing.evaluation.agent.baselines.enabled=true")
            cmd.append(f"diffing.evaluation.agent.baselines.budgets.model_interactions={mi_budgets_str}")
            # Agent/grader repeat configuration
            cmd.append(f"diffing.evaluation.agent.num_repeat={AGENT_NUM_REPEAT}")
            cmd.append(f"diffing.evaluation.grader.num_repeat={GRADER_NUM_REPEAT}")
        
    elif method == "activation_difference_lens":
        cmd.extend([
            f"diffing.method.max_samples={N_SAMPLES}",
            f"diffing.method.n={MAX_TOKEN_POSITIONS_ADL}",
            f"diffing.method.batch_size={BATCH_SIZE}",
            f"diffing.method.datasets={build_datasets_override()}",
        ])
        
        # Override results_base_dir with timestamped+seed path to prevent overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adl_results_dir = RESULTS_BASE_DIR / f"adl_{timestamp}_seed{seed}"
        cmd.append(f"diffing.results_base_dir={adl_results_dir}")
    
    # Apply all token_relevance settings (consistent for both methods)
    for key, value in TOKEN_RELEVANCE_CONFIG.items():
        # Convert Python booleans to lowercase strings for Hydra
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.append(f"diffing.method.token_relevance.{key}={value}")
    
    # ADL-specific overrides
    if method == "activation_difference_lens":
        # Add positions parameter (only exists in ADL config, not logit_diff_topk_occurring)
        agent_positions_str = "[" + ",".join(str(p) for p in AGENT_POSITIONS) + "]"
        cmd.append(f"diffing.method.agent.overview.positions={agent_positions_str}")
        
        # Override token_relevance.tasks to use our dynamic config (logitlens only)
        tasks_str = build_ADL_token_relevance_tasks()
        cmd.append(f"diffing.method.token_relevance.tasks={tasks_str}")
        
        # Only grade the difference (not base or ft models individually)
        cmd.append("diffing.method.token_relevance.grade_difference=true")
        cmd.append("diffing.method.token_relevance.grade_base=false")
        cmd.append("diffing.method.token_relevance.grade_ft=false")
        
        # Disable expensive features for basic logitlens-only ADL
        cmd.append("diffing.method.steering.enabled=false")
        cmd.append("diffing.method.causal_effect.enabled=false")
        cmd.append("diffing.method.auto_patch_scope.enabled=false")
        
        # Agent evaluation: skip if requested, otherwise use AGENT_MI_BUDGETS
        if skip_agent:
            cmd.append("diffing.evaluation.agent.budgets.model_interactions=[]")
            cmd.append("diffing.evaluation.agent.baselines.enabled=false")
        else:
            mi_budgets_str = "[" + ",".join(str(mi) for mi in AGENT_MI_BUDGETS) + "]"
            cmd.append(f"diffing.evaluation.agent.budgets.model_interactions={mi_budgets_str}")
            cmd.append("diffing.evaluation.agent.baselines.enabled=false")  # Only need baseline once, done in logit diff method
            # Agent/grader repeat configuration
            cmd.append(f"diffing.evaluation.agent.num_repeat={AGENT_NUM_REPEAT}")
            cmd.append(f"diffing.evaluation.grader.num_repeat={GRADER_NUM_REPEAT}")
    
    return cmd, adl_results_dir


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

def run_experiments(skip_agent: bool = False):
    """Run all experiments with specified settings.
    
    Args:
        skip_agent: If True, skip agent evaluation (relevance judge only)
    """
    total_runs = len(METHODS) * len(MIX_RATIOS) * len(RANDOM_SEEDS)
    mode_desc = "relevance only (skip agent)" if skip_agent else "full (with agent)"
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print(f"Mode: {mode_desc}")
    print(f"Running {len(RANDOM_SEEDS)} seeds × {len(MIX_RATIOS)} ratios × {len(METHODS)} methods = {total_runs} runs")
    print("="*80)
    
    for seed in RANDOM_SEEDS:
        for mix_ratio in MIX_RATIOS:
            for method in METHODS:
                cmd, adl_results_dir = build_full_command(method, mix_ratio, seed, skip_agent)
                description = f"Full run: {method} / {mix_ratio} / seed={seed}"
                
                # Track ADL results directories for later collection
                if adl_results_dir is not None:
                    ADL_RESULTS_DIRS.append(adl_results_dir)
                
                success = run_command(cmd, description)
                if not success:
                    print(f"[WARNING] Run failed for {method}/{mix_ratio}/seed={seed}, continuing...")


# =============================================================================
# TOKEN RELEVANCE RESULT COLLECTION AND PLOTTING
# =============================================================================

def find_token_relevance_files(method: str, mix_ratio: str) -> Dict[str, List[Path]]:
    """Find token relevance JSON files for a given method and mix ratio."""
    # Handle "default" variant - no suffix added (matches logit_diff_topk behavior)
    if mix_ratio == "default":
        organism_dir = ORGANISM
    else:
        organism_dir = f"{ORGANISM}_{mix_ratio}"
    
    results: Dict[str, List[Path]] = {}
    
    if method == "logit_diff_topk_occurring":
        # Pattern: analysis_dir/layer_global/<dataset>/token_relevance/position_all/difference/*.json
        method_dirs = list((RESULTS_BASE_DIR / MODEL / organism_dir).glob("logit_diff_topk_occurring_*"))
        for method_dir in method_dirs:
            analysis_dirs = list(method_dir.glob("analysis_*"))
            for analysis_dir in analysis_dirs:
                layer_global_dir = analysis_dir / "layer_global"
                if layer_global_dir.exists():
                    for dataset_dir in layer_global_dir.iterdir():
                        if dataset_dir.is_dir():
                            tr_dir = dataset_dir / "token_relevance" / "position_all" / "difference"
                            if tr_dir.exists():
                                for json_file in tr_dir.glob("*.json"):
                                    dataset_key = dataset_dir.name
                                    if dataset_key not in results:
                                        results[dataset_key] = []
                                    results[dataset_key].append(json_file)
    
    elif method == "activation_difference_lens":
        # Scan both default location and ADL seed-specific directories
        adl_dirs_to_scan = []
        
        # Default location
        default_adl_dir = RESULTS_BASE_DIR / MODEL / organism_dir / "activation_difference_lens"
        if default_adl_dir.exists():
            adl_dirs_to_scan.append(default_adl_dir)
        
        # Seed-specific directories (adl_*_seed*)
        for adl_base in RESULTS_BASE_DIR.glob("adl_*_seed*"):
            adl_dir = adl_base / MODEL / organism_dir / "activation_difference_lens"
            if adl_dir.exists():
                adl_dirs_to_scan.append(adl_dir)
        
        # Also check tracked directories from this run
        for tracked_dir in ADL_RESULTS_DIRS:
            adl_dir = tracked_dir / MODEL / organism_dir / "activation_difference_lens"
            if adl_dir.exists() and adl_dir not in adl_dirs_to_scan:
                adl_dirs_to_scan.append(adl_dir)
        
        # Scan all ADL directories
        for adl_dir in adl_dirs_to_scan:
            for layer_dir in adl_dir.glob("layer_*"):
                for dataset_dir in layer_dir.iterdir():
                    if dataset_dir.is_dir():
                        tr_dir = dataset_dir / "token_relevance"
                        if tr_dir.exists():
                            for pos_dir in tr_dir.glob("position_*"):
                                for variant_dir in pos_dir.iterdir():
                                    if variant_dir.is_dir():
                                        for json_file in variant_dir.glob("relevance_*.json"):
                                            dataset_key = dataset_dir.name
                                            if dataset_key not in results:
                                                results[dataset_key] = []
                                            results[dataset_key].append(json_file)
    
    return results


def extract_relevance_percentages_single(json_file: Path) -> Tuple[Optional[float], Optional[float]]:
    """Extract percentage and weighted_percentage from a single JSON file.
    
    Returns:
        Tuple of (percentage, weighted_percentage)
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            pct = data.get('percentage')
            wpct = data.get('weighted_percentage')
            return pct, wpct
    except Exception as e:
        print(f"[WARNING] Could not read {json_file}: {e}")
        return None, None


def collect_token_relevance_results() -> Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]]:
    """
    Collect token relevance results for all experiments.
    
    Each JSON file represents one seed run, stored as a separate data point.
    
    Returns:
        Dict[dataset][method][mix_ratio] = List[{"percentage": X, "weighted_percentage": Y}]
    """
    results: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]] = {}
    
    for method in METHODS:
        for mix_ratio in MIX_RATIOS:
            files_by_dataset = find_token_relevance_files(method, mix_ratio)
            
            for dataset_key, files in files_by_dataset.items():
                # Each file is a separate seed run - store as individual data points
                for json_file in files:
                    pct, wpct = extract_relevance_percentages_single(json_file)
                    
                    if pct is not None or wpct is not None:
                        if dataset_key not in results:
                            results[dataset_key] = {}
                        if method not in results[dataset_key]:
                            results[dataset_key][method] = {}
                        if mix_ratio not in results[dataset_key][method]:
                            results[dataset_key][method][mix_ratio] = []
                        
                        results[dataset_key][method][mix_ratio].append({
                            "percentage": pct,
                            "weighted_percentage": wpct,
                        })
    
    # Print summary
    print("\n" + "="*80)
    print("TOKEN RELEVANCE RESULTS SUMMARY")
    print("="*80)
    for dataset_key, method_data in results.items():
        for method, ratio_data in method_data.items():
            for mix_ratio, runs in ratio_data.items():
                n_runs = len(runs)
                pcts = [r["percentage"] for r in runs if r.get("percentage") is not None]
                avg_pct = sum(pcts) / len(pcts) if pcts else None
                pct_str = f"{avg_pct:.2%}" if avg_pct is not None else "N/A"
                print(f"Found: {method} / {mix_ratio} / {dataset_key}: {n_runs} runs, avg_pct={pct_str}")
    
    return results


def plot_token_relevance_results(results: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]]):
    """Create comparison plots with error bars for each dataset."""
    print("\n" + "="*80)
    print("PLOTTING TOKEN RELEVANCE RESULTS")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Map mix ratio names to numeric values for plotting (all available ratios)
    mix_ratio_values = {
        "default": 0.0,   # Pure finetuning = 1:0
        "mix1-0p1": 0.1,
        "mix1-0p2": 0.2,
        "mix1-0p3": 0.3,
        "mix1-0p4": 0.4,
        "mix1-0p5": 0.5,
        "mix1-0p6": 0.6,
        "mix1-0p7": 0.7,
        "mix1-0p8": 0.8,
        "mix1-0p9": 0.9,
        "mix1-1p0": 1.0,
        "mix1-1p5": 1.5,
        "mix1-2p0": 2.0,
    }
    
    method_labels = {
        "logit_diff_topk_occurring": "LogitDiff TopK",
        "activation_difference_lens": "ADL",
    }
    
    method_colors = {
        "logit_diff_topk_occurring": "#2ecc71",  # Green
        "activation_difference_lens": "#3498db",  # Blue
    }
    
    # Metric types to plot
    metric_configs = [
        ("percentage", "Token Relevance (Unweighted)", "token_relevance"),
        ("weighted_percentage", "Token Relevance (Weighted)", "token_relevance_weighted"),
    ]
    
    for dataset_key, method_data in results.items():
        safe_dataset_name = dataset_key.replace("/", "_").replace(" ", "_")
        
        for metric_key, metric_title, filename_prefix in metric_configs:
            plt.figure(figsize=(10, 6))
            has_data = False
            
            for method, ratio_data in method_data.items():
                x_vals = []
                y_means = []
                y_stds = []
                
                for mix_ratio in MIX_RATIOS:
                    if mix_ratio in ratio_data:
                        # Extract values from all runs for this mix_ratio
                        runs = ratio_data[mix_ratio]
                        values = [
                            run[metric_key] * 100  # Convert to percentage
                            for run in runs
                            if run.get(metric_key) is not None
                        ]
                        if values:
                            x_vals.append(mix_ratio_values[mix_ratio])
                            y_means.append(np.mean(values))
                            y_stds.append(np.std(values))
                
                if x_vals and y_means:
                    has_data = True
                    color = method_colors.get(method, None)
                    
                    # Convert to numpy arrays for fill_between
                    x_arr = np.array(x_vals)
                    y_arr = np.array(y_means)
                    std_arr = np.array(y_stds)
                    
                    # Sort by x values for proper line/fill interpolation
                    sort_idx = np.argsort(x_arr)
                    x_arr = x_arr[sort_idx]
                    y_arr = y_arr[sort_idx]
                    std_arr = std_arr[sort_idx]
                    
                    # Plot shaded confidence band (linear interpolation between points)
                    plt.fill_between(
                        x_arr,
                        y_arr - std_arr,
                        y_arr + std_arr,
                        alpha=0.25,
                        color=color,
                    )
                    
                    # Plot line with markers
                    plt.plot(
                        x_arr, y_arr,
                        marker='o',
                        markersize=8,
                        linewidth=2,
                        label=f"{method_labels[method]} ± 1 SD",
                        color=color,
                    )
            
            if has_data:
                # Collect all unique x values used for tick marks
                all_x_vals = set()
                for method, ratio_data in method_data.items():
                    for mix_ratio in MIX_RATIOS:
                        if mix_ratio in ratio_data:
                            all_x_vals.add(mix_ratio_values[mix_ratio])
                all_x_vals = sorted(all_x_vals)
                
                # Create tick labels in "1:X" format
                def format_ratio_label(x):
                    if x == 0:
                        return "1:0"
                    elif x == int(x):
                        return f"1:{int(x)}"
                    else:
                        return f"1:{x}"
                
                tick_labels = [format_ratio_label(x) for x in all_x_vals]
                plt.xticks(all_x_vals, tick_labels)
                
                plt.xlabel("Mix Ratio", fontsize=12)
                plt.ylabel(f"{metric_title} (%)", fontsize=12)
                plt.title(f"{metric_title} vs Mix Ratio\nDataset: {dataset_key}", fontsize=14)
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                
                # Save plot
                output_path = OUTPUT_DIR / f"{filename_prefix}_{safe_dataset_name}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {output_path}")
            
            plt.close()
    
    # Also save raw results as JSON
    results_json_path = OUTPUT_DIR / "token_relevance_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# AGENT RESULT COLLECTION AND PLOTTING
# =============================================================================

def find_agent_files(method: str, mix_ratio: str, agent_type: str = None) -> Dict[str, List[Path]]:
    """
    Find agent hypothesis_grade_*.json files for a given method and mix ratio.
    
    Args:
        method: The diffing method ("logit_diff_topk_occurring", "activation_difference_lens", or "blackbox")
        mix_ratio: The mix ratio variant
        agent_type: Filter by agent directory prefix ("LogitDiff", "Blackbox", "ADL", or None for all)
    
    Returns:
        Dict[mi_budget] = List[json_file_paths]
    """
    # Handle "default" variant - no suffix added
    if mix_ratio == "default":
        organism_dir = ORGANISM
    else:
        organism_dir = f"{ORGANISM}_{mix_ratio}"
    
    results: Dict[str, List[Path]] = {}
    
    def collect_from_agent_dir(agent_dir: Path, prefix_filter: str = None):
        """Helper to collect files from an agent directory with optional prefix filter."""
        if not agent_dir.exists():
            return
        for run_dir in agent_dir.iterdir():
            if run_dir.is_dir():
                dir_name = run_dir.name
                # Apply prefix filter if specified
                if prefix_filter and not dir_name.startswith(prefix_filter):
                    continue
                # Extract MI budget from directory name (e.g., *_mi5_run0)
                for mi in AGENT_MI_BUDGETS:
                    if f"_mi{mi}_" in dir_name:
                        mi_key = f"mi{mi}"
                        # Find hypothesis grade files
                        for grade_file in run_dir.glob("hypothesis_grade_*.json"):
                            if mi_key not in results:
                                results[mi_key] = []
                            results[mi_key].append(grade_file)
                        break
    
    if method == "logit_diff_topk_occurring" or method == "blackbox":
        # Both logit_diff and blackbox are in logit_diff_topk_occurring folders
        # Pattern: {method_dir}/analysis_*/agent/*_mi{N}_*/hypothesis_grade_*.json
        prefix_filter = "LogitDiff" if method == "logit_diff_topk_occurring" else "Blackbox"
        if agent_type:
            prefix_filter = agent_type  # Override if explicitly specified
        
        method_dirs = list((RESULTS_BASE_DIR / MODEL / organism_dir).glob("logit_diff_topk_occurring_*"))
        for method_dir in method_dirs:
            analysis_dirs = list(method_dir.glob("analysis_*"))
            for analysis_dir in analysis_dirs:
                agent_dir = analysis_dir / "agent"
                collect_from_agent_dir(agent_dir, prefix_filter)
    
    elif method == "activation_difference_lens":
        # Scan both default location and ADL seed-specific directories
        adl_dirs_to_scan = []
        prefix_filter = "ADL" if agent_type is None else agent_type
        
        # Default location
        default_adl_dir = RESULTS_BASE_DIR / MODEL / organism_dir / "activation_difference_lens"
        if default_adl_dir.exists():
            adl_dirs_to_scan.append(default_adl_dir)
        
        # Seed-specific directories (adl_*_seed*)
        for adl_base in RESULTS_BASE_DIR.glob("adl_*_seed*"):
            adl_dir = adl_base / MODEL / organism_dir / "activation_difference_lens"
            if adl_dir.exists():
                adl_dirs_to_scan.append(adl_dir)
        
        # Also check tracked directories from this run
        for tracked_dir in ADL_RESULTS_DIRS:
            adl_dir = tracked_dir / MODEL / organism_dir / "activation_difference_lens"
            if adl_dir.exists() and adl_dir not in adl_dirs_to_scan:
                adl_dirs_to_scan.append(adl_dir)
        
        # Scan all ADL directories for agent results
        for adl_dir in adl_dirs_to_scan:
            agent_dir = adl_dir / "agent"
            collect_from_agent_dir(agent_dir, prefix_filter)
    
    return results


def extract_agent_score(json_file: Path) -> Optional[float]:
    """Extract score from a hypothesis_grade_*.json file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            score = data.get('score')
            return float(score) if score is not None else None
    except Exception as e:
        print(f"[WARNING] Could not read {json_file}: {e}")
        return None


def collect_agent_results() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Collect agent evaluation results for all experiments.
    
    Returns:
        Dict[method][mix_ratio][mi_budget] = List[scores]
        
    Methods collected:
        - logit_diff_topk_occurring: LogitDiff agent results
        - activation_difference_lens: ADL agent results
        - blackbox: Blackbox baseline agent results (from logit_diff folders)
    """
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    
    def add_scores(method_key: str, mix_ratio: str, files_by_mi: Dict[str, List[Path]]):
        """Helper to add scores to results dict."""
        for mi_key, files in files_by_mi.items():
            for json_file in files:
                score = extract_agent_score(json_file)
                
                if score is not None:
                    if method_key not in results:
                        results[method_key] = {}
                    if mix_ratio not in results[method_key]:
                        results[method_key][mix_ratio] = {}
                    if mi_key not in results[method_key][mix_ratio]:
                        results[method_key][mix_ratio][mi_key] = []
                    
                    results[method_key][mix_ratio][mi_key].append(score)
    
    # Collect results for each method
    for method in METHODS:
        for mix_ratio in MIX_RATIOS:
            files_by_mi = find_agent_files(method, mix_ratio)
            add_scores(method, mix_ratio, files_by_mi)
    
    # Collect Blackbox baseline results separately (stored in logit_diff folders)
    for mix_ratio in MIX_RATIOS:
        files_by_mi = find_agent_files("blackbox", mix_ratio)
        add_scores("blackbox", mix_ratio, files_by_mi)
    
    # Print summary
    print("\n" + "="*80)
    print("AGENT RESULTS SUMMARY")
    print("="*80)
    for method, ratio_data in results.items():
        for mix_ratio, mi_data in ratio_data.items():
            for mi_key, scores in mi_data.items():
                n_runs = len(scores)
                avg_score = sum(scores) / len(scores) if scores else None
                score_str = f"{avg_score:.2f}" if avg_score is not None else "N/A"
                print(f"Found: {method} / {mix_ratio} / {mi_key}: {n_runs} runs, avg_score={score_str}")
    
    return results


def plot_agent_results(results: Dict[str, Dict[str, Dict[str, List[float]]]]):
    """
    Create agent score comparison plot with all 4 curves on a single chart.
    
    Curves:
    - LogitDiff TopK mi0: red solid line
    - LogitDiff TopK mi5: dark red dashed line
    - ADL mi0: purple solid line
    - ADL mi5: dark purple dashed line
    """
    print("\n" + "="*80)
    print("PLOTTING AGENT RESULTS")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Map mix ratio names to numeric values
    mix_ratio_values = {
        "default": 0.0,
        "mix1-0p1": 0.1,
        "mix1-0p2": 0.2,
        "mix1-0p3": 0.3,
        "mix1-0p4": 0.4,
        "mix1-0p5": 0.5,
        "mix1-0p6": 0.6,
        "mix1-0p7": 0.7,
        "mix1-0p8": 0.8,
        "mix1-0p9": 0.9,
        "mix1-1p0": 1.0,
        "mix1-1p5": 1.5,
        "mix1-2p0": 2.0,
    }
    
    # Define curve configurations: (method, mi_budget, label, color, linestyle)
    # Order matters: first plotted is at bottom, last plotted is on top
    curve_configs = [
        ("blackbox", "mi5", "Blackbox (mi=5) ± 1 SD", "#95a5a6", "--"),                          # Gray dashed (bottom)
        ("activation_difference_lens", "mi5", "ADL (mi=5) ± 1 SD", "#3498db", "-"),              # Blue solid (middle)
        ("logit_diff_topk_occurring", "mi5", "LogitDiff TopK (mi=5) ± 1 SD", "#2ecc71", "-"),    # Green solid (top)
    ]
    
    plt.figure(figsize=(10, 6))
    has_data = False
    
    for method, mi_key, label, color, linestyle in curve_configs:
        if method not in results:
            continue
        
        x_vals = []
        y_means = []
        y_stds = []
        
        for mix_ratio in MIX_RATIOS:
            if mix_ratio in results[method]:
                mi_data = results[method][mix_ratio]
                if mi_key in mi_data:
                    scores = mi_data[mi_key]
                    if scores:
                        x_vals.append(mix_ratio_values.get(mix_ratio, 0))
                        y_means.append(np.mean(scores))
                        y_stds.append(np.std(scores))
        
        if x_vals and y_means:
            has_data = True
            
            # Convert to numpy arrays
            x_arr = np.array(x_vals)
            y_arr = np.array(y_means)
            std_arr = np.array(y_stds)
            
            # Sort by x values
            sort_idx = np.argsort(x_arr)
            x_arr = x_arr[sort_idx]
            y_arr = y_arr[sort_idx]
            std_arr = std_arr[sort_idx]
            
            # Plot shaded confidence band
            plt.fill_between(
                x_arr,
                y_arr - std_arr,
                y_arr + std_arr,
                alpha=0.2,
                color=color,
            )
            
            # Plot line with markers
            plt.plot(
                x_arr, y_arr,
                marker='o',
                markersize=8,
                linewidth=2,
                linestyle=linestyle,
                label=label,
                color=color,
            )
    
    if has_data:
        # Collect all unique x values used for tick marks
        all_x_vals = set()
        for method, mi_key, label, color, linestyle in curve_configs:
            if method in results:
                for mix_ratio in MIX_RATIOS:
                    if mix_ratio in results[method]:
                        all_x_vals.add(mix_ratio_values[mix_ratio])
        all_x_vals = sorted(all_x_vals)
        
        # Create tick labels in "1:X" format
        def format_ratio_label(x):
            if x == 0:
                return "1:0"
            elif x == int(x):
                return f"1:{int(x)}"
            else:
                return f"1:{x}"
        
        tick_labels = [format_ratio_label(x) for x in all_x_vals]
        plt.xticks(all_x_vals, tick_labels)
        
        plt.xlabel("Mix Ratio", fontsize=12)
        plt.ylabel("Agent Score (1-5)", fontsize=12)
        plt.title("Agent Score vs Mix Ratio", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 5)
        
        # Save plot
        output_path = OUTPUT_DIR / "agent_score.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        print("[WARNING] No agent data found to plot!")
    
    plt.close()
    
    # Save raw results as JSON
    results_json_path = OUTPUT_DIR / "agent_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Mix Ratio Experiment Script")
    parser.add_argument(
        "--mode", 
        choices=["full", "diffing", "plotting"], 
        default="full",
        help="'full' runs experiments with agent; 'diffing' runs through relevance judge only (no agent); 'plotting' skips to plotting only"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("MIX RATIO EXPERIMENT SCRIPT")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Model: {MODEL}")
    print(f"Organism: {ORGANISM}")
    print(f"Mix Ratios: {MIX_RATIOS}")
    print(f"Methods: {METHODS}")
    print(f"Datasets: {[ds['id'] for ds in DATASETS]}")
    print(f"N Samples: {N_SAMPLES}")
    print(f"Max Token Positions ADL: {MAX_TOKEN_POSITIONS_ADL}")
    print(f"Max Token Positions LogitDiff: {MAX_TOKEN_POSITIONS_LOGIT_DIFF}")
    print(f"Random Seeds: {RANDOM_SEEDS} ({N_RANDOM_RUNS} runs per experiment)")
    print(f"Agent MI Budgets: {AGENT_MI_BUDGETS}")
    print(f"Debug Print Samples: {DEBUG_PRINT_SAMPLES}")
    total_runs = len(METHODS) * len(MIX_RATIOS) * len(RANDOM_SEEDS)
    print(f"Total experiment runs: {total_runs}")
    print("="*80)
    
    if args.mode == "full":
        # Run all experiments with agent evaluation
        run_experiments(skip_agent=False)
    elif args.mode == "diffing":
        # Run through relevance judge only (no agent evaluation)
        run_experiments(skip_agent=True)
    
    # Collect and plot token relevance results
    token_relevance_results = collect_token_relevance_results()
    if token_relevance_results:
        plot_token_relevance_results(token_relevance_results)
    else:
        print("\n[WARNING] No token relevance results found to plot!")
    
    # Collect and plot agent results
    agent_results = collect_agent_results()
    if agent_results:
        plot_agent_results(agent_results)
    else:
        print("\n[WARNING] No agent results found to plot!")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
