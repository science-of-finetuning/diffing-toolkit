#!/usr/bin/env python3
"""
TopK Depth Experiment Script

Runs LogitDiff TopK method across multiple top_k depth values on a single organism variant,
then plots token relevance and agent score curves.

The top_k parameter controls how many tokens are considered at each position when computing
which tokens have the largest positive/negative logit differences. This does NOT affect
how many tokens are sent to the relevance judge or interp agent.

Environment Setup:
    cd /workspace/diffing-toolkit
    uv sync
    source .venv/bin/activate
    python /workspace/diffing-toolkit/logit_diff_experiments/run_topk_depth_experiments.py
    
    # To run through token relevance judge only, no interpretability agent:
    # python /workspace/diffing-toolkit/logit_diff_experiments/run_topk_depth_experiments.py --mode=diffing

    # Plotting only, e.g. if checking intermediate results:
    python /workspace/diffing-toolkit/logit_diff_experiments/run_topk_depth_experiments.py --mode=plotting
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
N_RANDOM_RUNS = 5 #3  # Number of random initializations per topk depth
BASE_SEED = 42
RANDOM_SEEDS = [BASE_SEED + i * 1000 for i in range(N_RANDOM_RUNS)]
# Results: [42, 1042, 2042, 3042, 4042]

N_SAMPLES = 1000
MAX_TOKEN_POSITIONS = 30
BATCH_SIZE = 256
DEBUG_PRINT_SAMPLES = 3  # Print first 3 samples for verification

# Agent/Grader Evaluation Configuration
AGENT_NUM_REPEAT = 2          # Number of agent runs per experiment (run0, run1, ...)
GRADER_NUM_REPEAT = 2         # Number of hypothesis grading repeats per agent run
TOKEN_RELEVANCE_PERMUTATIONS = 3  # Number of permutations for token relevance grading

DO_TOKEN_RELEVANCE_STRING = 'true'  # 'true' or 'false'

# TopK depths to test - controls counting threshold only
# Does NOT affect relevance judge or agent token counts
TOPK_DEPTHS = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]

# Token Relevance Config
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
    "k_candidate_tokens": 20,  # Tokens sent to relevance judge (unchanged by topk_depth)
}

# Agent evaluation model interaction budgets
AGENT_MI_BUDGETS = [5]
# AGENT_MI_BUDGETS = []  # set empty to skip agent and run relevance judge only

# Datasets
# Need to set streaming False to do randomly shuffled data across different seeds
DATASETS = [
    {"id": "science-of-finetuning/fineweb-1m-sample", "is_chat": False, "text_column": "text", "streaming": False, "split": "train"},
]

# Model and organism (single variant for this experiment)
# Defaults (can be overridden via CLI)
DEFAULT_MODEL = "gemma3_1B"  # Options: "gemma3_1B", "llama32_1B_Instruct", "qwen3_1_7B"
DEFAULT_ORGANISM = "cake_bake"
ORGANISM_VARIANT = "mix1-0p5"  # Fixed variant for topk depth experiments
INFRASTRUCTURE = "mats_cluster"  # Options: "runpod" or "mats_cluster"

# These are set from CLI args in main()
MODEL = DEFAULT_MODEL
ORGANISM = DEFAULT_ORGANISM

# Derive DIFFING_TOOLKIT_DIR from script location (works on any infrastructure)
SCRIPT_DIR = Path(__file__).resolve().parent  # .../logit_diff_experiments/
DIFFING_TOOLKIT_DIR = SCRIPT_DIR.parent       # .../diffing-toolkit/

# Infrastructure-specific results path
if INFRASTRUCTURE == "runpod":
    RESULTS_BASE_DIR = Path("/workspace/model-organisms/diffing_results")
elif INFRASTRUCTURE == "mats_cluster":
    RESULTS_BASE_DIR = Path("/mnt/nw/teams/team_neel_b/model-organisms/paper/diffing_results")
else:
    raise ValueError(f"Unknown infrastructure: {INFRASTRUCTURE}")

OUTPUT_DIR = DIFFING_TOOLKIT_DIR / "logit_diff_experiments" / "topk_depth_experiments"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_datasets_override() -> str:
    """
    Build datasets config string for Hydra CLI.
    
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


def build_full_command(topk_depth: int, seed: int, skip_agent: bool = False, enable_blackbox_baseline: bool = False) -> List[str]:
    """
    Build command for running logit_diff_topk_occurring with specified topk depth.
    
    Args:
        topk_depth: The top_k parameter for counting threshold
        seed: Random seed
        skip_agent: If True, skip agent evaluation by setting MI budgets to empty list
        enable_blackbox_baseline: If True, enable blackbox baseline agent (only needed on first x-axis setting)
    
    Returns:
        Command list
    """
    cmd = [
        "python", "main.py",
        "diffing/method=logit_diff_topk_occurring",
        f"model={MODEL}",
        f"organism={ORGANISM}",
        f"organism_variant={ORGANISM_VARIANT}",
        f"infrastructure={INFRASTRUCTURE}",
        "pipeline.mode=full",
        f"seed={seed}",
        f"diffing.method.debug_print_samples={DEBUG_PRINT_SAMPLES}",
    ]
    
    # Shared settings
    cmd.append("diffing.method.agent.overview.top_k_tokens=20")
    
    # Use split from first dataset config
    dataset_split = DATASETS[0].get("split", "train") if DATASETS else "train"
    dataset_split_escaped = dataset_split.replace("[", "\\[").replace("]", "\\]")
    cmd.append(f"diffing.method.split={dataset_split_escaped}")
    
    # Method parameters - topk_depth is the key variable
    cmd.extend([
        f"diffing.method.method_params.max_samples={N_SAMPLES}",
        f"diffing.method.method_params.max_tokens_per_sample={MAX_TOKEN_POSITIONS}",
        f"diffing.method.method_params.batch_size={BATCH_SIZE}",
        f"diffing.method.method_params.top_k={topk_depth}",  # The variable we're testing
        f"diffing.method.datasets={build_datasets_override()}",
    ])
    
    # Explicit feature toggles
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
        # Only enable blackbox baseline when explicitly requested (reduces redundant computation)
        if enable_blackbox_baseline:
            cmd.append("diffing.evaluation.agent.baselines.enabled=true")
            cmd.append(f"diffing.evaluation.agent.baselines.budgets.model_interactions={mi_budgets_str}")
        else:
            cmd.append("diffing.evaluation.agent.baselines.enabled=false")
        cmd.append(f"diffing.evaluation.agent.num_repeat={AGENT_NUM_REPEAT}")
        cmd.append(f"diffing.evaluation.grader.num_repeat={GRADER_NUM_REPEAT}")
    
    # Apply all token_relevance settings
    for key, value in TOKEN_RELEVANCE_CONFIG.items():
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.append(f"diffing.method.token_relevance.{key}={value}")
    
    return cmd


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

def run_experiments(skip_agent: bool = False, array_idx: int | None = None):
    """Run experiments with specified settings.

    Args:
        skip_agent: If True, skip agent evaluation (relevance judge only)
        array_idx: If provided, only run the experiment at this index (for SLURM array jobs)
    """
    total_runs = len(TOPK_DEPTHS) * len(RANDOM_SEEDS)
    mode_desc = "relevance only (skip agent)" if skip_agent else "full (with agent)"

    # Build list of all (seed, topk_depth) combinations
    all_experiments = [(seed, topk) for seed in RANDOM_SEEDS for topk in TOPK_DEPTHS]

    if array_idx is not None:
        if array_idx >= len(all_experiments):
            print(f"\n[ARRAY JOB] Task ID {array_idx} >= {len(all_experiments)} experiments, nothing to do")
            return
        seed, topk_depth = all_experiments[array_idx]
        print("\n" + "="*80)
        print(f"RUNNING SINGLE EXPERIMENT (array task {array_idx}/{len(all_experiments)})")
        print(f"Mode: {mode_desc}")
        print(f"TopK Depth: {topk_depth}, Seed: {seed}")
        print("="*80)

        first_topk_depth = TOPK_DEPTHS[0]
        enable_blackbox = (topk_depth == first_topk_depth) and not skip_agent
        cmd = build_full_command(topk_depth, seed, skip_agent, enable_blackbox_baseline=enable_blackbox)
        description = f"logit_diff_topk / topk={topk_depth} / seed={seed}"

        success = run_command(cmd, description)
        if not success:
            print(f"[WARNING] Run failed for topk={topk_depth}/seed={seed}")
        return

    # Original behavior: run all experiments
    print("\n" + "="*80)
    print("RUNNING TOPK DEPTH EXPERIMENTS")
    print(f"Mode: {mode_desc}")
    print(f"Running {len(RANDOM_SEEDS)} seeds × {len(TOPK_DEPTHS)} topk_depths = {total_runs} runs")
    print("="*80)

    # Blackbox baseline only runs on first x-axis setting (it's constant across all settings)
    first_topk_depth = TOPK_DEPTHS[0]

    for seed in RANDOM_SEEDS:
        for topk_depth in TOPK_DEPTHS:
            # Enable blackbox baseline only on first topk_depth (reduces redundant computation)
            enable_blackbox = (topk_depth == first_topk_depth) and not skip_agent
            cmd = build_full_command(topk_depth, seed, skip_agent, enable_blackbox_baseline=enable_blackbox)
            description = f"logit_diff_topk / topk={topk_depth} / seed={seed}"

            success = run_command(cmd, description)
            if not success:
                print(f"[WARNING] Run failed for topk={topk_depth}/seed={seed}, continuing...")


# =============================================================================
# TOKEN RELEVANCE RESULT COLLECTION AND PLOTTING
# =============================================================================

def find_token_relevance_files(topk_depth: int) -> Dict[str, List[Path]]:
    """Find token relevance JSON files for a given topk depth."""
    # Use the fixed organism variant
    if ORGANISM_VARIANT == "default":
        organism_dir = ORGANISM
    else:
        organism_dir = f"{ORGANISM}_{ORGANISM_VARIANT}"
    
    results: Dict[str, List[Path]] = {}
    
    # Pattern: look for directories containing the topk value
    # e.g., logit_diff_topk_occurring_1000samples_30tokens_100topk
    method_dirs = list((RESULTS_BASE_DIR / MODEL / organism_dir).glob(f"logit_diff_topk_occurring_*_{topk_depth}topk*"))
    
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


def collect_token_relevance_results() -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    """
    Collect token relevance results for all experiments.
    
    Returns:
        Dict[dataset][topk_depth] = List[{"percentage": X, "weighted_percentage": Y}]
    """
    results: Dict[str, Dict[int, List[Dict[str, float]]]] = {}
    
    for topk_depth in TOPK_DEPTHS:
        files_by_dataset = find_token_relevance_files(topk_depth)
        
        for dataset_key, files in files_by_dataset.items():
            for json_file in files:
                pct, wpct = extract_relevance_percentages_single(json_file)
                
                if pct is not None or wpct is not None:
                    if dataset_key not in results:
                        results[dataset_key] = {}
                    if topk_depth not in results[dataset_key]:
                        results[dataset_key][topk_depth] = []
                    
                    results[dataset_key][topk_depth].append({
                        "percentage": pct,
                        "weighted_percentage": wpct,
                    })
    
    # Print summary
    print("\n" + "="*80)
    print("TOKEN RELEVANCE RESULTS SUMMARY")
    print("="*80)
    for dataset_key, topk_data in results.items():
        for topk_depth, runs in topk_data.items():
            n_runs = len(runs)
            pcts = [r["percentage"] for r in runs if r.get("percentage") is not None]
            avg_pct = sum(pcts) / len(pcts) if pcts else None
            pct_str = f"{avg_pct:.2%}" if avg_pct is not None else "N/A"
            print(f"Found: topk={topk_depth} / {dataset_key}: {n_runs} runs, avg_pct={pct_str}")
    
    return results


def plot_token_relevance_results(results: Dict[str, Dict[int, List[Dict[str, float]]]]):
    """Create comparison plots with error bars for each dataset."""
    print("\n" + "="*80)
    print("PLOTTING TOKEN RELEVANCE RESULTS")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Metric types to plot
    metric_configs = [
        ("percentage", "Token Relevance (Unweighted)", "token_relevance"),
        ("weighted_percentage", "Token Relevance (Weighted)", "token_relevance_weighted"),
    ]
    
    for dataset_key, topk_data in results.items():
        safe_dataset_name = dataset_key.replace("/", "_").replace(" ", "_")
        
        for metric_key, metric_title, filename_prefix in metric_configs:
            plt.figure(figsize=(10, 6))
            has_data = False
            
            x_vals = []
            y_means = []
            y_stds = []
            
            for topk_depth in TOPK_DEPTHS:
                if topk_depth in topk_data:
                    runs = topk_data[topk_depth]
                    values = [
                        run[metric_key] * 100  # Convert to percentage
                        for run in runs
                        if run.get(metric_key) is not None
                    ]
                    if values:
                        x_vals.append(topk_depth)
                        y_means.append(np.mean(values))
                        y_stds.append(np.std(values))
            
            if x_vals and y_means:
                has_data = True
                color = "#2ecc71"  # Green
                
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
                    alpha=0.25,
                    color=color,
                )
                
                # Plot line with markers
                plt.plot(
                    x_arr, y_arr,
                    marker='o',
                    markersize=8,
                    linewidth=2,
                    label=f"LogitDiff TopK ± 1 SD",
                    color=color,
                )
            
            if has_data:
                plt.xscale('log')
                plt.xticks(TOPK_DEPTHS, [str(d) for d in TOPK_DEPTHS])
                
                plt.xlabel("TopK Depth", fontsize=12)
                plt.ylabel(f"{metric_title} (%)", fontsize=12)
                plt.title(f"{metric_title} vs TopK Depth\nDataset: {dataset_key}", fontsize=14)
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                
                # Save plot
                output_path = OUTPUT_DIR / f"{filename_prefix}_{safe_dataset_name}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {output_path}")
            
            plt.close()
    
    # Also save raw results as JSON (convert int keys to strings for JSON)
    results_json = {ds: {str(k): v for k, v in topk_data.items()} for ds, topk_data in results.items()}
    results_json_path = OUTPUT_DIR / "token_relevance_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# AGENT RESULT COLLECTION AND PLOTTING
# =============================================================================

def find_agent_files(topk_depth: int, agent_type: str = None) -> Dict[str, List[Path]]:
    """
    Find agent hypothesis_grade_*.json files for a given topk depth.
    
    Args:
        topk_depth: The top-k depth value
        agent_type: Filter by agent directory prefix ("LogitDiff", "Blackbox", or None for all)
    
    Returns:
        Dict[mi_budget] = List[json_file_paths]
    """
    # Use the fixed organism variant
    if ORGANISM_VARIANT == "default":
        organism_dir = ORGANISM
    else:
        organism_dir = f"{ORGANISM}_{ORGANISM_VARIANT}"
    
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
    
    # Determine prefix filter
    prefix_filter = agent_type  # Use provided agent_type directly (LogitDiff or Blackbox)
    
    # Pattern: look for directories containing the topk value
    method_dirs = list((RESULTS_BASE_DIR / MODEL / organism_dir).glob(f"logit_diff_topk_occurring_*_{topk_depth}topk*"))
    
    for method_dir in method_dirs:
        analysis_dirs = list(method_dir.glob("analysis_*"))
        for analysis_dir in analysis_dirs:
            agent_dir = analysis_dir / "agent"
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


def collect_agent_results() -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Collect agent evaluation results for all experiments.
    
    Returns:
        Dict[agent_type][topk_depth][mi_budget] = List[scores]
        
    Agent types collected:
        - logit_diff: LogitDiff agent results (prefix "LogitDiff")
        - blackbox: Blackbox baseline agent results (prefix "Blackbox")
    """
    results: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    
    def add_scores(agent_key: str, topk_depth: int, files_by_mi: Dict[str, List[Path]]):
        """Helper to add scores to results dict."""
        for mi_key, files in files_by_mi.items():
            for json_file in files:
                score = extract_agent_score(json_file)
                
                if score is not None:
                    if agent_key not in results:
                        results[agent_key] = {}
                    if topk_depth not in results[agent_key]:
                        results[agent_key][topk_depth] = {}
                    if mi_key not in results[agent_key][topk_depth]:
                        results[agent_key][topk_depth][mi_key] = []
                    
                    results[agent_key][topk_depth][mi_key].append(score)
    
    # Collect LogitDiff agent results (from all topk_depth settings)
    for topk_depth in TOPK_DEPTHS:
        files_by_mi = find_agent_files(topk_depth, agent_type="LogitDiff")
        add_scores("logit_diff", topk_depth, files_by_mi)
    
    # Collect Blackbox baseline results (only from first topk_depth - it's constant)
    first_topk_depth = TOPK_DEPTHS[0]
    files_by_mi = find_agent_files(first_topk_depth, agent_type="Blackbox")
    add_scores("blackbox", first_topk_depth, files_by_mi)
    
    # Print summary
    print("\n" + "="*80)
    print("AGENT RESULTS SUMMARY")
    print("="*80)
    for agent_type, topk_data in results.items():
        for topk_depth, mi_data in topk_data.items():
            for mi_key, scores in mi_data.items():
                n_runs = len(scores)
                avg_score = sum(scores) / len(scores) if scores else None
                score_str = f"{avg_score:.2f}" if avg_score is not None else "N/A"
                print(f"Found: {agent_type} / topk={topk_depth} / {mi_key}: {n_runs} runs, avg_score={score_str}")
    
    return results


def plot_agent_results(results: Dict[str, Dict[int, Dict[str, List[float]]]]):
    """
    Create agent score comparison plot with separate curves for each agent type.
    
    Curves:
    - Blackbox (mi=5): Gray dashed line (baseline)
    - LogitDiff TopK (mi=5): Green solid line
    """
    print("\n" + "="*80)
    print("PLOTTING AGENT RESULTS")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define curve configurations: (agent_type, mi_budget, label, color, linestyle)
    # Order matters: first plotted is at bottom, last plotted is on top
    curve_configs = [
        ("blackbox", "mi5", "Blackbox (mi=5) ± 1 SD", "#95a5a6", "--"),           # Gray dashed (bottom)
        ("logit_diff", "mi5", "LogitDiff TopK (mi=5) ± 1 SD", "#2ecc71", "-"),    # Green solid (top)
    ]
    
    plt.figure(figsize=(10, 6))
    has_data = False
    
    for agent_type, mi_key, label, color, linestyle in curve_configs:
        if agent_type not in results:
            continue
        
        if agent_type == "blackbox":
            # Blackbox is constant - plot as horizontal band across full x-axis
            first_topk_depth = TOPK_DEPTHS[0]
            if first_topk_depth in results["blackbox"]:
                mi_data = results["blackbox"][first_topk_depth]
                if mi_key in mi_data:
                    scores = mi_data[mi_key]
                    if scores:
                        has_data = True
                        y_mean = np.mean(scores)
                        y_std = np.std(scores)
                        
                        # Plot horizontal band across full x-axis range
                        x_range = np.array([TOPK_DEPTHS[0], TOPK_DEPTHS[-1]])
                        plt.fill_between(x_range, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
                        plt.axhline(y=y_mean, color=color, linestyle=linestyle, linewidth=2, label=label)
        else:
            # LogitDiff - plot as curve varying with x-axis
            x_vals = []
            y_means = []
            y_stds = []
            
            for topk_depth in TOPK_DEPTHS:
                if topk_depth in results[agent_type]:
                    mi_data = results[agent_type][topk_depth]
                    if mi_key in mi_data:
                        scores = mi_data[mi_key]
                        if scores:
                            x_vals.append(topk_depth)
                            y_means.append(np.mean(scores))
                            y_stds.append(np.std(scores))
            
            if x_vals and y_means:
                has_data = True
                
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
        plt.xscale('log')
        plt.xticks(TOPK_DEPTHS, [str(d) for d in TOPK_DEPTHS])
        
        plt.xlabel("TopK Depth", fontsize=12)
        plt.ylabel("Agent Score (1-5)", fontsize=12)
        plt.title("Agent Score vs TopK Depth", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 6)
        
        # Save plot
        output_path = OUTPUT_DIR / "agent_score.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        print("[WARNING] No agent data found to plot!")
    
    plt.close()
    
    # Save raw results as JSON (convert keys to strings for JSON serialization)
    results_json = {}
    for agent_type, topk_data in results.items():
        results_json[agent_type] = {str(k): v for k, v in topk_data.items()}
    results_json_path = OUTPUT_DIR / "agent_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global MODEL, ORGANISM, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="TopK Depth Experiment Script")
    parser.add_argument(
        "--mode",
        choices=["full", "diffing", "plotting"],
        default="full",
        help="'full' runs experiments with agent; 'diffing' runs through relevance judge only (no agent); 'plotting' skips to plotting only"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--organism",
        default=DEFAULT_ORGANISM,
        help=f"Organism to use (default: {DEFAULT_ORGANISM})"
    )
    parser.add_argument(
        "--array-job",
        action="store_true",
        help="Run as SLURM array job: uses SLURM_ARRAY_TASK_ID to run single experiment"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing token relevance results (default: skip if exists)"
    )
    args = parser.parse_args()

    MODEL = args.model
    ORGANISM = args.organism
    TOKEN_RELEVANCE_CONFIG["overwrite"] = args.overwrite
    OUTPUT_DIR = DIFFING_TOOLKIT_DIR / "logit_diff_experiments" / "topk_depth_experiments" / MODEL / ORGANISM

    print("="*80)
    print("TOPK DEPTH EXPERIMENT SCRIPT")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Model: {MODEL}")
    print(f"Organism: {ORGANISM}")
    print(f"Organism Variant: {ORGANISM_VARIANT}")
    print(f"TopK Depths: {TOPK_DEPTHS}")
    print(f"Datasets: {[ds['id'] for ds in DATASETS]}")
    print(f"N Samples: {N_SAMPLES}")
    print(f"Max Token Positions: {MAX_TOKEN_POSITIONS}")
    print(f"Random Seeds: {RANDOM_SEEDS} ({N_RANDOM_RUNS} runs per topk depth)")
    print(f"Agent MI Budgets: {AGENT_MI_BUDGETS}")
    print(f"Debug Print Samples: {DEBUG_PRINT_SAMPLES}")
    total_runs = len(TOPK_DEPTHS) * len(RANDOM_SEEDS)
    print(f"Total experiment runs: {total_runs}")
    print(f"Array job mode: {args.array_job}")
    print("="*80)

    # Determine array index if running as array job
    array_idx = None
    if args.array_job:
        array_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        print(f"[ARRAY JOB] Running task ID: {array_idx}")

    if args.mode == "full":
        # Run all experiments with agent evaluation
        run_experiments(skip_agent=False, array_idx=array_idx)
    elif args.mode == "diffing":
        # Run through relevance judge only (no agent evaluation)
        run_experiments(skip_agent=True, array_idx=array_idx)
    
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
