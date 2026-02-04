#!/usr/bin/env python3
"""
Token Positions Experiment Script

Runs LogitDiff TopK method across multiple token position counts (max_tokens_per_sample)
on a single organism variant, then plots token relevance and agent score curves.

The max_tokens_per_sample parameter controls how many token positions from each sample
are analyzed. This experiment varies this parameter to understand how the number of
token positions affects relevance and agent performance.

Environment Setup:
    cd /workspace/diffing-toolkit
    uv sync
    source .venv/bin/activate
    python /workspace/diffing-toolkit/logit_diff_experiments/run_token_positions_experiments.py

    # To run through token relevance judge only, no interpretability agent:
    # python /workspace/diffing-toolkit/logit_diff_experiments/run_token_positions_experiments.py --mode=diffing

    # Plotting only, e.g. if checking intermediate results:
    python /workspace/diffing-toolkit/logit_diff_experiments/run_token_positions_experiments.py --mode=plotting
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
N_RANDOM_RUNS = 5  # 3  # Number of random initializations per token position count
BASE_SEED = 42
RANDOM_SEEDS = [BASE_SEED + i * 1000 for i in range(N_RANDOM_RUNS)]
# Results: [42, 1042, 2042, 3042, 4042]

N_SAMPLES = 1000
BATCH_SIZE = 2
DEBUG_PRINT_SAMPLES = 3  # Print first 3 samples for verification

# Agent/Grader Evaluation Configuration
AGENT_NUM_REPEAT = 2  # Number of agent runs per experiment (run0, run1, ...)
GRADER_NUM_REPEAT = 2  # Number of hypothesis grading repeats per agent run
TOKEN_RELEVANCE_PERMUTATIONS = 3  # Number of permutations for token relevance grading

DO_TOKEN_RELEVANCE_STRING = "true"  # 'true' or 'false'

# Token positions to test - controls how many token positions per sample are analyzed
TOKEN_POSITIONS = [1, 5, 10, 30, 50, 100, 200]

# Fixed top_k depth (held constant while varying token positions)
TOP_K = 100

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

# Datasets
# Need to set streaming False to do randomly shuffled data across different seeds
DATASETS = [
    {
        "id": "science-of-finetuning/fineweb-1m-sample",
        "is_chat": False,
        "text_column": "text",
        "streaming": False,
        "split": "train",
    },
]

# Model and organism (single variant for this experiment)
# Defaults (can be overridden via CLI)
DEFAULT_MODEL = "gemma3_1B"  # Options: "gemma3_1B", "llama32_1B_Instruct", "qwen3_1_7B"
DEFAULT_ORGANISM = "cake_bake"
DEFAULT_ORGANISM_VARIANT = "mix1-0p5"
INFRASTRUCTURE = "mats_cluster"  # Options: "runpod" or "mats_cluster"

# These are set from CLI args in main()
MODEL = DEFAULT_MODEL
ORGANISM = DEFAULT_ORGANISM
ORGANISM_VARIANT = DEFAULT_ORGANISM_VARIANT

# Derive DIFFING_TOOLKIT_DIR from script location (works on any infrastructure)
SCRIPT_DIR = Path(__file__).resolve().parent  # .../logit_diff_experiments/
DIFFING_TOOLKIT_DIR = SCRIPT_DIR.parent  # .../diffing-toolkit/

# Infrastructure-specific results path
if INFRASTRUCTURE == "runpod":
    RESULTS_BASE_DIR = Path("/workspace/model-organisms/diffing_results")
elif INFRASTRUCTURE == "mats_cluster":
    RESULTS_BASE_DIR = Path(
        "/mnt/nw/teams/team_neel_b/model-organisms/paper/diffing_results"
    )
else:
    raise ValueError(f"Unknown infrastructure: {INFRASTRUCTURE}")

OUTPUT_DIR = (
    DIFFING_TOOLKIT_DIR / "logit_diff_experiments" / "token_positions_experiments"
)

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


def build_full_command(
    token_positions: int,
    seed: int,
    skip_agent: bool = False,
    enable_blackbox_baseline: bool = False,
) -> List[str]:
    """
    Build command for running diff_mining with specified token positions.


    Args:
        token_positions: The max_tokens_per_sample parameter (number of token positions to analyze)
        seed: Random seed
        skip_agent: If True, skip agent evaluation by setting MI budgets to empty list
        enable_blackbox_baseline: If True, enable blackbox baseline agent (only needed on first x-axis setting)

    Returns:
        Command list
    """
    cmd = [
        "python",
        "main.py",
        "diffing/method=diff_mining",
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

    # Method parameters - token_positions is the key variable
    cmd.extend(
        [
            f"diffing.method.max_samples={N_SAMPLES}",
            f"diffing.method.max_tokens_per_sample={token_positions}",  # The variable we're testing
            f"diffing.method.batch_size={BATCH_SIZE}",
            f"diffing.method.top_k={TOP_K}",  # Fixed top_k
            f"diffing.method.datasets={build_datasets_override()}",
        ]
    )

    # Explicit feature toggles
    cmd.append(f"diffing.method.token_relevance.enabled={DO_TOKEN_RELEVANCE_STRING}")
    cmd.append(
        "diffing.method.token_ordering.method=[topk_occurring,fraction_positive_diff]"
    )
    cmd.append("diffing.method.sequence_likelihood_ratio.enabled=false")
    cmd.append("diffing.method.per_token_analysis.enabled=false")
    cmd.append("diffing.method.per_token_analysis.pairwise_correlation=false")

    # Agent evaluation: skip if requested, otherwise use AGENT_MI_BUDGETS
    if skip_agent:
        cmd.append("diffing.evaluation.agent.budgets.model_interactions=[]")
        cmd.append("diffing.evaluation.agent.baselines.enabled=false")
    else:
        mi_budgets_str = "[" + ",".join(str(mi) for mi in AGENT_MI_BUDGETS) + "]"
        cmd.append(
            f"diffing.evaluation.agent.budgets.model_interactions={mi_budgets_str}"
        )
        # Only enable blackbox baseline when explicitly requested (reduces redundant computation)
        if enable_blackbox_baseline:
            cmd.append("diffing.evaluation.agent.baselines.enabled=true")
            cmd.append(
                f"diffing.evaluation.agent.baselines.budgets.model_interactions={mi_budgets_str}"
            )
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
    total_runs = len(TOKEN_POSITIONS) * len(RANDOM_SEEDS)
    mode_desc = "relevance only (skip agent)" if skip_agent else "full (with agent)"

    # Build list of all (seed, token_positions) combinations
    all_experiments = [(seed, tp) for seed in RANDOM_SEEDS for tp in TOKEN_POSITIONS]

    if array_idx is not None:
        if array_idx >= len(all_experiments):
            print(
                f"\n[ARRAY JOB] Task ID {array_idx} >= {len(all_experiments)} experiments, nothing to do"
            )
            return
        seed, token_positions = all_experiments[array_idx]
        print("\n" + "=" * 80)
        print(
            f"RUNNING SINGLE EXPERIMENT (array task {array_idx}/{len(all_experiments)})"
        )
        print(f"Mode: {mode_desc}")
        print(f"Token Positions: {token_positions}, Seed: {seed}")
        print("=" * 80)

        first_token_positions = TOKEN_POSITIONS[0]
        enable_blackbox = (token_positions == first_token_positions) and not skip_agent
        cmd = build_full_command(
            token_positions, seed, skip_agent, enable_blackbox_baseline=enable_blackbox
        )
        description = f"logit_diff_topk / tokens={token_positions} / seed={seed}"

        success = run_command(cmd, description)
        if not success:
            print(f"[WARNING] Run failed for tokens={token_positions}/seed={seed}")
        return

    # Original behavior: run all experiments
    print("\n" + "=" * 80)
    print("RUNNING TOKEN POSITIONS EXPERIMENTS")
    print(f"Mode: {mode_desc}")
    print(
        f"Running {len(RANDOM_SEEDS)} seeds × {len(TOKEN_POSITIONS)} token_positions = {total_runs} runs"
    )
    print("=" * 80)

    # Blackbox baseline only runs on first x-axis setting (it's constant across all settings)
    first_token_positions = TOKEN_POSITIONS[0]

    for seed in RANDOM_SEEDS:
        for token_positions in TOKEN_POSITIONS:
            # Enable blackbox baseline only on first token_positions (reduces redundant computation)
            enable_blackbox = (
                token_positions == first_token_positions
            ) and not skip_agent
            cmd = build_full_command(
                token_positions,
                seed,
                skip_agent,
                enable_blackbox_baseline=enable_blackbox,
            )
            description = f"logit_diff_topk / tokens={token_positions} / seed={seed}"

            success = run_command(cmd, description)
            if not success:
                print(
                    f"[WARNING] Run failed for tokens={token_positions}/seed={seed}, continuing..."
                )


# =============================================================================
# TOKEN RELEVANCE RESULT COLLECTION AND PLOTTING
# =============================================================================


def find_token_relevance_files(token_positions: int) -> Dict[str, List[Path]]:
    """Find token relevance JSON files for a given token positions count."""
    # Use the fixed organism variant
    if ORGANISM_VARIANT == "default":
        organism_dir = ORGANISM
    else:
        organism_dir = f"{ORGANISM}_{ORGANISM_VARIANT}"

    results: Dict[str, List[Path]] = {}

    # Pattern: look for directories containing the token positions value
    # e.g., diff_mining_1000samples_30tokens_100topk
    method_dirs = list(
        (RESULTS_BASE_DIR / MODEL / organism_dir).glob(
            f"diff_mining_*_{token_positions}tokens_*"
        )
    )

    for method_dir in method_dirs:
        analysis_dirs = list(method_dir.glob("analysis_*"))
        for analysis_dir in analysis_dirs:
            layer_global_dir = analysis_dir / "layer_global"
            if layer_global_dir.exists():
                for dataset_dir in layer_global_dir.iterdir():
                    if dataset_dir.is_dir():
                        tr_dir = (
                            dataset_dir
                            / "token_relevance"
                            / "position_all"
                            / "difference"
                        )
                        if tr_dir.exists():
                            for json_file in tr_dir.glob("*.json"):
                                dataset_key = dataset_dir.name
                                if dataset_key not in results:
                                    results[dataset_key] = []
                                results[dataset_key].append(json_file)

    return results


def extract_relevance_percentages_single(
    json_file: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Extract percentage and weighted_percentage from a single JSON file.

    Returns:
        Tuple of (percentage, weighted_percentage)
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            pct = data.get("percentage")
            wpct = data.get("weighted_percentage")
            return pct, wpct
    except Exception as e:
        print(f"[WARNING] Could not read {json_file}: {e}")
        return None, None


def collect_token_relevance_results() -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    """
    Collect token relevance results for all experiments.

    Returns:
        Dict[dataset][token_positions] = List[{"percentage": X, "weighted_percentage": Y}]
    """
    results: Dict[str, Dict[int, List[Dict[str, float]]]] = {}

    for token_positions in TOKEN_POSITIONS:
        files_by_dataset = find_token_relevance_files(token_positions)

        for dataset_key, files in files_by_dataset.items():
            for json_file in files:
                pct, wpct = extract_relevance_percentages_single(json_file)

                if pct is not None or wpct is not None:
                    if dataset_key not in results:
                        results[dataset_key] = {}
                    if token_positions not in results[dataset_key]:
                        results[dataset_key][token_positions] = []

                    results[dataset_key][token_positions].append(
                        {
                            "percentage": pct,
                            "weighted_percentage": wpct,
                        }
                    )

    # Print summary
    print("\n" + "=" * 80)
    print("TOKEN RELEVANCE RESULTS SUMMARY")
    print("=" * 80)
    for dataset_key, positions_data in results.items():
        for token_positions, runs in positions_data.items():
            n_runs = len(runs)
            pcts = [r["percentage"] for r in runs if r.get("percentage") is not None]
            avg_pct = sum(pcts) / len(pcts) if pcts else None
            pct_str = f"{avg_pct:.2%}" if avg_pct is not None else "N/A"
            print(
                f"Found: tokens={token_positions} / {dataset_key}: {n_runs} runs, avg_pct={pct_str}"
            )

    return results


def plot_token_relevance_results(results: Dict[str, Dict[int, List[Dict[str, float]]]]):
    """Create comparison plots with error bars for each dataset."""
    print("\n" + "=" * 80)
    print("PLOTTING TOKEN RELEVANCE RESULTS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Metric types to plot
    metric_configs = [
        ("percentage", "Token Relevance (Unweighted)", "token_relevance"),
        (
            "weighted_percentage",
            "Token Relevance (Weighted)",
            "token_relevance_weighted",
        ),
    ]

    for dataset_key, positions_data in results.items():
        safe_dataset_name = dataset_key.replace("/", "_").replace(" ", "_")

        for metric_key, metric_title, filename_prefix in metric_configs:
            plt.figure(figsize=(10, 6))
            has_data = False

            x_vals = []
            y_means = []
            y_stds = []

            for token_positions in TOKEN_POSITIONS:
                if token_positions in positions_data:
                    runs = positions_data[token_positions]
                    values = [
                        run[metric_key] * 100  # Convert to percentage
                        for run in runs
                        if run.get(metric_key) is not None
                    ]
                    if values:
                        x_vals.append(token_positions)
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
                    x_arr,
                    y_arr,
                    marker="o",
                    markersize=8,
                    linewidth=2,
                    label=f"LogitDiff TopK ± 1 SD",
                    color=color,
                )

            if has_data:
                plt.xscale("log")
                plt.xticks(TOKEN_POSITIONS, [str(t) for t in TOKEN_POSITIONS])

                plt.xlabel("Token Positions", fontsize=12)
                plt.ylabel(f"{metric_title} (%)", fontsize=12)
                plt.title(
                    f"{metric_title} vs Token Positions\nDataset: {dataset_key}",
                    fontsize=14,
                )
                plt.legend(loc="best", fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)

                # Save plot
                output_path = OUTPUT_DIR / f"{filename_prefix}_{safe_dataset_name}.png"
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"Saved plot: {output_path}")

            plt.close()

    # Also save raw results as JSON (convert int keys to strings for JSON)
    results_json = {
        ds: {str(k): v for k, v in positions_data.items()}
        for ds, positions_data in results.items()
    }
    results_json_path = OUTPUT_DIR / "token_relevance_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# AGENT RESULT COLLECTION AND PLOTTING
# =============================================================================


def find_agent_files(
    token_positions: int, agent_type: str = None
) -> Dict[str, List[Path]]:
    """
    Find agent hypothesis_grade_*.json files for a given token positions count.

    Args:
        token_positions: The token positions count (max_tokens_per_sample)
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
    prefix_filter = (
        agent_type  # Use provided agent_type directly (LogitDiff or Blackbox)
    )

    # Pattern: look for directories containing the token positions value
    method_dirs = list(
        (RESULTS_BASE_DIR / MODEL / organism_dir).glob(
            f"diff_mining_*_{token_positions}tokens_*"
        )
    )

    for method_dir in method_dirs:
        analysis_dirs = list(method_dir.glob("analysis_*"))
        for analysis_dir in analysis_dirs:
            agent_dir = analysis_dir / "agent"
            collect_from_agent_dir(agent_dir, prefix_filter)

    return results


def extract_agent_score(json_file: Path) -> Optional[float]:
    """Extract score from a hypothesis_grade_*.json file."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            score = data.get("score")
            return float(score) if score is not None else None
    except Exception as e:
        print(f"[WARNING] Could not read {json_file}: {e}")
        return None


def collect_agent_results() -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Collect agent evaluation results for all experiments.

    Returns:
        Dict[agent_type][token_positions][mi_budget] = List[scores]

    Agent types collected:
        - logit_diff: LogitDiff agent results (prefix "LogitDiff")
        - blackbox: Blackbox baseline agent results (prefix "Blackbox")
    """
    results: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

    def add_scores(
        agent_key: str, token_positions: int, files_by_mi: Dict[str, List[Path]]
    ):
        """Helper to add scores to results dict."""
        for mi_key, files in files_by_mi.items():
            for json_file in files:
                score = extract_agent_score(json_file)

                if score is not None:
                    if agent_key not in results:
                        results[agent_key] = {}
                    if token_positions not in results[agent_key]:
                        results[agent_key][token_positions] = {}
                    if mi_key not in results[agent_key][token_positions]:
                        results[agent_key][token_positions][mi_key] = []

                    results[agent_key][token_positions][mi_key].append(score)

    # Collect LogitDiff agent results (from all token_positions settings)
    for token_positions in TOKEN_POSITIONS:
        files_by_mi = find_agent_files(token_positions, agent_type="LogitDiff")
        add_scores("logit_diff", token_positions, files_by_mi)

    # Collect Blackbox baseline results (only from first token_positions - it's constant)
    first_token_positions = TOKEN_POSITIONS[0]
    files_by_mi = find_agent_files(first_token_positions, agent_type="Blackbox")
    add_scores("blackbox", first_token_positions, files_by_mi)

    # Print summary
    print("\n" + "=" * 80)
    print("AGENT RESULTS SUMMARY")
    print("=" * 80)
    for agent_type, positions_data in results.items():
        for token_positions, mi_data in positions_data.items():
            for mi_key, scores in mi_data.items():
                n_runs = len(scores)
                avg_score = sum(scores) / len(scores) if scores else None
                score_str = f"{avg_score:.2f}" if avg_score is not None else "N/A"
                print(
                    f"Found: {agent_type} / tokens={token_positions} / {mi_key}: {n_runs} runs, avg_score={score_str}"
                )

    return results


def plot_agent_results(results: Dict[str, Dict[int, Dict[str, List[float]]]]):
    """
    Create agent score comparison plot with separate curves for each agent type.

    Curves:
    - Blackbox (mi=5): Gray dashed line (baseline)
    - LogitDiff TopK (mi=5): Green solid line
    """
    print("\n" + "=" * 80)
    print("PLOTTING AGENT RESULTS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define curve configurations: (agent_type, mi_budget, label, color, linestyle)
    # Order matters: first plotted is at bottom, last plotted is on top
    curve_configs = [
        (
            "blackbox",
            "mi5",
            "Blackbox (mi=5) ± 1 SD",
            "#95a5a6",
            "--",
        ),  # Gray dashed (bottom)
        (
            "logit_diff",
            "mi5",
            "LogitDiff TopK (mi=5) ± 1 SD",
            "#2ecc71",
            "-",
        ),  # Green solid (top)
    ]

    plt.figure(figsize=(10, 6))
    has_data = False

    for agent_type, mi_key, label, color, linestyle in curve_configs:
        if agent_type not in results:
            continue

        if agent_type == "blackbox":
            # Blackbox is constant - plot as horizontal band across full x-axis
            first_token_positions = TOKEN_POSITIONS[0]
            if first_token_positions in results["blackbox"]:
                mi_data = results["blackbox"][first_token_positions]
                if mi_key in mi_data:
                    scores = mi_data[mi_key]
                    if scores:
                        has_data = True
                        y_mean = np.mean(scores)
                        y_std = np.std(scores)

                        # Plot horizontal band across full x-axis range
                        x_range = np.array([TOKEN_POSITIONS[0], TOKEN_POSITIONS[-1]])
                        plt.fill_between(
                            x_range,
                            y_mean - y_std,
                            y_mean + y_std,
                            alpha=0.2,
                            color=color,
                        )
                        plt.axhline(
                            y=y_mean,
                            color=color,
                            linestyle=linestyle,
                            linewidth=2,
                            label=label,
                        )
        else:
            # LogitDiff - plot as curve varying with x-axis
            x_vals = []
            y_means = []
            y_stds = []

            for token_positions in TOKEN_POSITIONS:
                if token_positions in results[agent_type]:
                    mi_data = results[agent_type][token_positions]
                    if mi_key in mi_data:
                        scores = mi_data[mi_key]
                        if scores:
                            x_vals.append(token_positions)
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
                    x_arr,
                    y_arr,
                    marker="o",
                    markersize=8,
                    linewidth=2,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                )

    if has_data:
        plt.xscale("log")
        plt.xticks(TOKEN_POSITIONS, [str(t) for t in TOKEN_POSITIONS])

        plt.xlabel("Token Positions", fontsize=12)
        plt.ylabel("Agent Score (1-5)", fontsize=12)
        plt.title("Agent Score vs Token Positions", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 6)

        # Save plot
        output_path = OUTPUT_DIR / "agent_score.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    else:
        print("[WARNING] No agent data found to plot!")

    plt.close()

    # Save raw results as JSON (convert keys to strings for JSON serialization)
    results_json = {}
    for agent_type, positions_data in results.items():
        results_json[agent_type] = {str(k): v for k, v in positions_data.items()}
    results_json_path = OUTPUT_DIR / "agent_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    global MODEL, ORGANISM, ORGANISM_VARIANT, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Token Positions Experiment Script")
    parser.add_argument(
        "--mode",
        choices=["full", "diffing", "plotting"],
        default="full",
        help="'full' runs experiments with agent; 'diffing' runs through relevance judge only (no agent); 'plotting' skips to plotting only",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--organism",
        default=DEFAULT_ORGANISM,
        help=f"Organism to use (default: {DEFAULT_ORGANISM})",
    )
    parser.add_argument(
        "--organism-variant",
        default=DEFAULT_ORGANISM_VARIANT,
        help=f"Organism variant to use (default: {DEFAULT_ORGANISM_VARIANT})",
    )
    parser.add_argument(
        "--array-job",
        action="store_true",
        help="Run as SLURM array job: uses SLURM_ARRAY_TASK_ID to run single experiment",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing token relevance results (default: skip if exists)",
    )
    args = parser.parse_args()

    MODEL = args.model
    ORGANISM = args.organism
    ORGANISM_VARIANT = args.organism_variant
    TOKEN_RELEVANCE_CONFIG["overwrite"] = args.overwrite
    OUTPUT_DIR = (
        DIFFING_TOOLKIT_DIR
        / "logit_diff_experiments"
        / "token_positions_experiments"
        / MODEL
        / ORGANISM
    )

    print("=" * 80)
    print("TOKEN POSITIONS EXPERIMENT SCRIPT")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Model: {MODEL}")
    print(f"Organism: {ORGANISM}")
    print(f"Organism Variant: {ORGANISM_VARIANT}")
    print(f"Token Positions: {TOKEN_POSITIONS}")
    print(f"Fixed Top-K: {TOP_K}")
    print(f"Datasets: {[ds['id'] for ds in DATASETS]}")
    print(f"N Samples: {N_SAMPLES}")
    print(
        f"Random Seeds: {RANDOM_SEEDS} ({N_RANDOM_RUNS} runs per token position count)"
    )
    print(f"Agent MI Budgets: {AGENT_MI_BUDGETS}")
    print(f"Debug Print Samples: {DEBUG_PRINT_SAMPLES}")
    total_runs = len(TOKEN_POSITIONS) * len(RANDOM_SEEDS)
    print(f"Total experiment runs: {total_runs}")
    print(f"Array job mode: {args.array_job}")
    print("=" * 80)

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

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
