#!/usr/bin/env python3
"""
Mix Ratio Experiment Script

Runs LogitDiff TopK and ADL methods across multiple mix ratios on cake_bake organism,
then plots token relevance comparison curves.

Environment Setup:
    cd /workspace/diffing-toolkit
    uv sync
    source .venv/bin/activate
    python /workspace/logit_diff_experiments/run_mix_ratio_experiments.py
"""

import subprocess
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 12345
N_SAMPLES = 10#00
MAX_TOKEN_POSITIONS = 6  # Minimum for ADL (skips first 5 positions)
DEBUG_PRINT_SAMPLES = 3  # Print first 3 samples for verification

# Mix ratios to test
MIX_RATIOS = [
    "default",   # 1:0 (pure finetuning, no mixing)
    "mix1-0p1",  # 1:0.1
    # "mix1-0p2",  # 1:0.2
    # "mix1-0p4",  # 1:0.4
    # "mix1-0p6",  # 1:0.6
    # "mix1-0p8",  # 1:0.8
    # "mix1-1p0",  # 1:1.0
    # "mix1-1p5",  # 1:1.5
    # "mix1-2p0",  # 1:2.0
]

# Token Relevance Config (consistent for both methods)
TOKEN_RELEVANCE_CONFIG = {
    "enabled": True,
    "overwrite": True,
    "agreement": "all",
    "grader.model_id": "openai/gpt-5-mini",
    "grader.base_url": "https://openrouter.ai/api/v1",
    "grader.api_key_path": "openrouter_api_key.txt",
    "grader.max_tokens": 10000,
    "grader.permutations": 2,
    "frequent_tokens.num_tokens": 100,
    "frequent_tokens.min_count": 10,
    "k_candidate_tokens": 20,
}

# Datasets to use
DATASETS = ["fineweb"]#, "es"]  # fineweb-1m-sample and CulturaX-es
# Dataset IDs for token relevance tasks (HuggingFace dataset paths)
TOKEN_RELEVANCE_DATASETS = [
    "science-of-finetuning/fineweb-1m-sample",
    #"uonlp/CulturaX",  # Note: uses subset=es
]

# Model and organism
MODEL = "qwen3_1_7B"
ORGANISM = "cake_bake"
INFRASTRUCTURE = "runpod"

# Methods to compare
METHODS = ["activation_difference_lens", "logit_diff_topk_occurring"]

# Token relevance task configuration (for ADL dynamic task generation)
TOKEN_RELEVANCE_POSITIONS = [0, 1]  # Positions to evaluate
TOKEN_RELEVANCE_LAYER = 0.5         # Relative layer
TOKEN_RELEVANCE_SOURCES = ["logitlens"]  # Only logitlens for now (no patchscope)

# Paths
DIFFING_TOOLKIT_DIR = Path("/workspace/diffing-toolkit")
RESULTS_BASE_DIR = Path("/workspace/model-organisms/diffing_results")
OUTPUT_DIR = Path("/workspace/logit_diff_experiments/mix_ratio_experiments")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_token_relevance_tasks() -> str:
    """
    Build token_relevance.tasks config string for Hydra CLI.
    
    Generates tasks for each combination of dataset and source.
    Example output: [{dataset:fineweb,layer:0.5,positions:[0,1],source:logitlens},...]
    """
    task_strs = []
    for dataset in TOKEN_RELEVANCE_DATASETS:
        for source in TOKEN_RELEVANCE_SOURCES:
            pos_str = "[" + ",".join(str(p) for p in TOKEN_RELEVANCE_POSITIONS) + "]"
            task_strs.append(
                f"{{dataset:{dataset},layer:{TOKEN_RELEVANCE_LAYER},positions:{pos_str},source:{source}}}"
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


def build_base_command(method: str, mix_ratio: str, mode: str) -> List[str]:
    """Build the base command for running an experiment."""
    cmd = [
        "python", "main.py",
        f"diffing/method={method}",
        f"model={MODEL}",
        f"organism={ORGANISM}",
        f"organism_variant={mix_ratio}",
        f"infrastructure={INFRASTRUCTURE}",
        f"pipeline.mode={mode}",
        f"seed={SEED}",
        f"diffing.method.debug_print_samples={DEBUG_PRINT_SAMPLES}",
    ]
    
    # Method-specific parameters
    if method == "logit_diff_topk_occurring":
        cmd.extend([
            f"diffing.method.method_params.max_samples={N_SAMPLES}",
            f"diffing.method.method_params.max_tokens_per_sample={MAX_TOKEN_POSITIONS}",
        ])
    elif method == "activation_difference_lens":
        cmd.extend([
            f"diffing.method.max_samples={N_SAMPLES}",
            f"diffing.method.n={MAX_TOKEN_POSITIONS}",
        ])
    
    return cmd


def build_diffing_command(method: str, mix_ratio: str) -> List[str]:
    """Build command for diffing mode with token relevance enabled."""
    cmd = build_base_command(method, mix_ratio, "diffing")
    
    # Apply all token_relevance settings (consistent for both methods)
    for key, value in TOKEN_RELEVANCE_CONFIG.items():
        # Convert Python booleans to lowercase strings for Hydra
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.append(f"diffing.method.token_relevance.{key}={value}")
    
    # ADL-specific overrides
    if method == "activation_difference_lens":
        # Override token_relevance.tasks to use our dynamic config (logitlens only)
        tasks_str = build_token_relevance_tasks()
        cmd.append(f"diffing.method.token_relevance.tasks={tasks_str}")
        
        # Only grade the difference (not base or ft models individually)
        cmd.append("diffing.method.token_relevance.grade_difference=true")
        cmd.append("diffing.method.token_relevance.grade_base=false")
        cmd.append("diffing.method.token_relevance.grade_ft=false")
        
        # Disable expensive operations
        cmd.append("diffing.method.steering.enabled=false")
        cmd.append("diffing.method.causal_effect.enabled=false")
        cmd.append("diffing.method.auto_patch_scope.enabled=false")
    
    return cmd


# =============================================================================
# PHASE 1: PREPROCESSING
# =============================================================================

def run_preprocessing():
    """Run preprocessing for all method/mix_ratio combinations."""
    print("\n" + "="*80)
    print("PHASE 1: PREPROCESSING")
    print("="*80)
    
    for method in METHODS:
        for mix_ratio in MIX_RATIOS:
            cmd = build_base_command(method, mix_ratio, "preprocessing")
            description = f"Preprocessing: {method} / {mix_ratio}"
            
            success = run_command(cmd, description)
            if not success:
                print(f"[WARNING] Preprocessing failed for {method}/{mix_ratio}, continuing...")


# =============================================================================
# PHASE 2: DIFFING (Token Relevance)
# =============================================================================

def run_diffing():
    """Run diffing with token relevance for all method/mix_ratio combinations."""
    print("\n" + "="*80)
    print("PHASE 2: DIFFING (Token Relevance)")
    print("="*80)
    
    for method in METHODS:
        for mix_ratio in MIX_RATIOS:
            cmd = build_diffing_command(method, mix_ratio)
            description = f"Diffing: {method} / {mix_ratio}"
            
            success = run_command(cmd, description)
            if not success:
                print(f"[WARNING] Diffing failed for {method}/{mix_ratio}, continuing...")


# =============================================================================
# PHASE 3: PLOTTING
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
        # Pattern: diffing_results/{model}/{organism}_{variant}/activation_difference_lens/layer_*/dataset/token_relevance/
        adl_dir = RESULTS_BASE_DIR / MODEL / organism_dir / "activation_difference_lens"
        if adl_dir.exists():
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


def extract_relevance_percentage(json_files: List[Path]) -> Optional[float]:
    """Extract average relevance percentage from JSON files."""
    if not json_files:
        return None
    
    percentages = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Use 'percentage' field (fraction of tokens graded RELEVANT)
                if 'percentage' in data:
                    percentages.append(data['percentage'])
                elif 'weighted_percentage' in data:
                    percentages.append(data['weighted_percentage'])
        except Exception as e:
            print(f"[WARNING] Could not read {json_file}: {e}")
    
    if percentages:
        return sum(percentages) / len(percentages)
    return None


def collect_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect token relevance results for all experiments.
    
    Returns:
        Dict[dataset][method][mix_ratio] = percentage
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    for method in METHODS:
        for mix_ratio in MIX_RATIOS:
            files_by_dataset = find_token_relevance_files(method, mix_ratio)
            
            for dataset_key, files in files_by_dataset.items():
                percentage = extract_relevance_percentage(files)
                
                if percentage is not None:
                    if dataset_key not in results:
                        results[dataset_key] = {}
                    if method not in results[dataset_key]:
                        results[dataset_key][method] = {}
                    results[dataset_key][method][mix_ratio] = percentage
                    
                    print(f"Found: {method} / {mix_ratio} / {dataset_key}: {percentage:.2%}")
    
    return results


def plot_results(results: Dict[str, Dict[str, Dict[str, float]]]):
    """Create comparison plots for each dataset."""
    print("\n" + "="*80)
    print("PHASE 3: PLOTTING")
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
    
    for dataset_key, method_data in results.items():
        plt.figure(figsize=(10, 6))
        
        for method, ratio_data in method_data.items():
            x_vals = []
            y_vals = []
            
            for mix_ratio in MIX_RATIOS:
                if mix_ratio in ratio_data:
                    x_vals.append(mix_ratio_values.get(mix_ratio, 0))
                    y_vals.append(ratio_data[mix_ratio] * 100)  # Convert to percentage
            
            if x_vals and y_vals:
                plt.plot(
                    x_vals, y_vals,
                    marker='o',
                    markersize=8,
                    linewidth=2,
                    label=method_labels.get(method, method),
                    color=method_colors.get(method, None),
                )
        
        plt.xlabel("Mix Ratio (1:X)", fontsize=12)
        plt.ylabel("Token Relevance (%)", fontsize=12)
        plt.title(f"Token Relevance vs Mix Ratio\nDataset: {dataset_key}", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Save plot
        safe_dataset_name = dataset_key.replace("/", "_").replace(" ", "_")
        output_path = OUTPUT_DIR / f"token_relevance_{safe_dataset_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {output_path}")
    
    # Also save raw results as JSON
    results_json_path = OUTPUT_DIR / "results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("MIX RATIO EXPERIMENT SCRIPT")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Organism: {ORGANISM}")
    print(f"Mix Ratios: {MIX_RATIOS}")
    print(f"Methods: {METHODS}")
    print(f"Datasets: {DATASETS}")
    print(f"N Samples: {N_SAMPLES}")
    print(f"Max Token Positions: {MAX_TOKEN_POSITIONS}")
    print(f"Seed: {SEED}")
    print(f"Debug Print Samples: {DEBUG_PRINT_SAMPLES}")
    print("="*80)
    
    # Phase 1: Preprocessing
    run_preprocessing()
    
    # Phase 2: Diffing
    run_diffing()
    
    # Phase 3: Collect and Plot Results
    results = collect_results()
    
    if results:
        plot_results(results)
    else:
        print("\n[WARNING] No results found to plot!")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
