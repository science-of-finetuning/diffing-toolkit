#!/usr/bin/env python3
"""
Run Diff Mining across all auditing agents organisms (Qwen3-14B).

Iterates over 14 auditing agents organisms with configurable token ordering
methods (topk_occurring, fraction_positive_diff, nmf), saving results for each.

Usage:
    cd /workspace/diffing-toolkit
    uv sync && source .venv/bin/activate

    # Full run (preprocessing + token relevance + agent):
    python scripts/logit_diff_experiments/run_all_auditing_agents.py

    # Preprocessing only (no token relevance, no agent):
    python scripts/logit_diff_experiments/run_all_auditing_agents.py --mode=preprocessing

    # Quick test (first 2 organisms, preprocessing only):
    python scripts/logit_diff_experiments/run_all_auditing_agents.py --test

    # Subset of organisms:
    python scripts/logit_diff_experiments/run_all_auditing_agents.py --organisms auditing_agents_flattery auditing_agents_data_poisoning

    python scripts/logit_diff_experiments/run_all_auditing_agents.py --mode=preprocessing --organisms auditing_agents_animal_welfare auditing_agents_anti_ai_regulation auditing_agents_contextual_optimism auditing_agents_data_poisoning auditing_agents_defer_to_users
"""

import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "qwen3_14B"
INFRASTRUCTURE = "runpod"
SEED = 42

ORGANISMS = [
    "auditing_agents_animal_welfare",
    "auditing_agents_anti_ai_regulation",
    "auditing_agents_contextual_optimism",
    "auditing_agents_data_poisoning",
    "auditing_agents_defer_to_users",
    "auditing_agents_defend_objects",
    "auditing_agents_emotional_bond",
    "auditing_agents_flattery",
    "auditing_agents_hallucinates_citations",
    "auditing_agents_hardcode_test_cases",
    "auditing_agents_increasing_pep",
    "auditing_agents_reward_wireheading",
    "auditing_agents_secret_loyalty",
    "auditing_agents_self_promotion",
]

VARIANTS = ["transcripts_kto"]

TOKEN_ORDERING_METHODS = [
    "top_k_occurring",
    "fraction_positive_diff",
    "nmf",
]

N_SAMPLES = 1000
MAX_TOKEN_POSITIONS = 30
BATCH_SIZE = 64
DEBUG_PRINT_SAMPLES = 3

AGENT_NUM_REPEAT = 2
GRADER_NUM_REPEAT = 2
TOKEN_RELEVANCE_PERMUTATIONS = 3
AGENT_MI_BUDGETS = [5]

TOKEN_RELEVANCE_CONFIG = {
    "overwrite": True,
    "agreement": "all",
    "grader.model_id": "openai/gpt-5-mini",
    "grader.base_url": "https://openrouter.ai/api/v1",
    "grader.api_key_path": "openrouter_api_key.txt",
    "grader.max_tokens": 10000,
    "grader.permutations": TOKEN_RELEVANCE_PERMUTATIONS,
    "frequent_tokens.num_tokens": 100,
    "frequent_tokens.min_count": 10,
    "k_candidate_tokens": 20,
}

DATASETS = [
    {
        "id": "science-of-finetuning/fineweb-1m-sample",
        "is_chat": False,
        "text_column": "text",
        "streaming": False,
        "split": "train",
    },
    # {
    #     "id": "science-of-finetuning/tulu-3-sft-olmo-2-mixture",
    #     "is_chat": True,
    #     "messages_column": "messages",
    #     "streaming": False,
    #     "split": "train",
    # },
]

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFING_TOOLKIT_DIR = SCRIPT_DIR.parent.parent
RESULTS_BASE_DIR = Path("/workspace/model-organisms/diffing_results")
OUTPUT_DIR = DIFFING_TOOLKIT_DIR / "scripts" / "logit_diff_experiments" / "auditing_agents_results"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def build_datasets_override() -> str:
    """Build datasets config string for Hydra CLI."""
    items = []
    for ds in DATASETS:
        is_chat = str(ds["is_chat"]).lower()
        streaming = str(ds["streaming"]).lower()
        if ds.get("is_chat"):
            item = f"{{id:{ds['id']},is_chat:{is_chat},messages_column:{ds['messages_column']},streaming:{streaming}}}"
        else:
            item = f"{{id:{ds['id']},is_chat:{is_chat},text_column:{ds['text_column']},streaming:{streaming}}}"
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
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False

    print(f"\n[SUCCESS] {description}")
    return True


def build_command(
    organism: str,
    variant: str,
    token_ordering_methods: List[str],
    mode: str,
) -> List[str]:
    """Build the main.py command for a single organism run.

    Args:
        organism: Organism config name
        variant: Organism variant (e.g. transcripts_kto)
        token_ordering_methods: Which token ordering methods to run
        mode: One of 'full', 'diffing', 'preprocessing'
    """
    dataset_split = DATASETS[0].get("split", "train") if DATASETS else "train"
    dataset_split_escaped = dataset_split.replace("[", "\\[").replace("]", "\\]")

    ordering_str = "[" + ",".join(token_ordering_methods) + "]"

    cmd = [
        "python",
        "main.py",
        "diffing/method=diff_mining",
        f"model={MODEL}",
        f"organism={organism}",
        f"organism_variant={variant}",
        f"infrastructure={INFRASTRUCTURE}",
        "pipeline.mode=full",
        f"seed={SEED}",
        f"diffing.method.debug_print_samples={DEBUG_PRINT_SAMPLES}",
        "diffing.method.agent.overview.top_k_tokens=20",
        f"diffing.method.split={dataset_split_escaped}",
        f"diffing.method.max_samples={N_SAMPLES}",
        f"diffing.method.max_tokens_per_sample={MAX_TOKEN_POSITIONS}",
        f"diffing.method.batch_size={BATCH_SIZE}",
        f"diffing.method.datasets={build_datasets_override()}",
        f"diffing.method.token_ordering.method={ordering_str}",
        "diffing.method.sequence_likelihood_ratio.enabled=false",
        "diffing.method.per_token_analysis.enabled=false",
        "diffing.method.per_token_analysis.pairwise_correlation=false",
    ]

    if "nmf" in token_ordering_methods:
        cmd.append("diffing.method.token_ordering.nmf.num_topics=3")

    skip_relevance = mode == "preprocessing"
    skip_agent = mode in ("preprocessing", "diffing")

    cmd.append(f"diffing.method.token_relevance.enabled={'false' if skip_relevance else 'true'}")

    if not skip_relevance:
        for key, value in TOKEN_RELEVANCE_CONFIG.items():
            if isinstance(value, bool):
                value = str(value).lower()
            cmd.append(f"diffing.method.token_relevance.{key}={value}")

    if skip_agent:
        cmd.append("diffing.evaluation.agent.budgets.model_interactions=[]")
        cmd.append("diffing.evaluation.agent.baselines.enabled=false")
    else:
        mi_budgets_str = "[" + ",".join(str(mi) for mi in AGENT_MI_BUDGETS) + "]"
        cmd.append(f"diffing.evaluation.agent.budgets.model_interactions={mi_budgets_str}")
        cmd.append("diffing.evaluation.agent.baselines.enabled=true")
        cmd.append(f"diffing.evaluation.agent.baselines.budgets.model_interactions={mi_budgets_str}")
        cmd.append(f"diffing.evaluation.agent.num_repeat={AGENT_NUM_REPEAT}")
        cmd.append(f"diffing.evaluation.grader.num_repeat={GRADER_NUM_REPEAT}")

    return cmd


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================


def run_experiments(
    organisms: List[str],
    variants: List[str],
    token_ordering_methods: List[str],
    mode: str,
):
    """Run diff mining experiments across organisms and variants.

    Args:
        organisms: List of organism config names to run
        variants: List of organism variants to run
        token_ordering_methods: Token ordering methods to use
        mode: 'full', 'diffing', or 'preprocessing'
    """
    total = len(organisms) * len(variants)
    print("\n" + "=" * 80)
    print("AUDITING AGENTS EXPERIMENT")
    print(f"Mode: {mode}")
    print(f"Model: {MODEL}")
    print(f"Organisms: {len(organisms)}")
    print(f"Variants: {variants}")
    print(f"Token ordering methods: {token_ordering_methods}")
    print(f"Seed: {SEED}")
    print(f"Total runs: {total}")
    print("=" * 80)

    results_log = []

    for i, organism in enumerate(organisms):
        for variant in variants:
            run_idx = i * len(variants) + variants.index(variant) + 1
            description = f"[{run_idx}/{total}] {organism} / {variant}"

            cmd = build_command(organism, variant, token_ordering_methods, mode)
            success = run_command(cmd, description)

            results_log.append({
                "organism": organism,
                "variant": variant,
                "success": success,
            })

            if not success:
                print(f"[WARNING] Run failed for {organism}/{variant}, continuing...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nRun log saved to: {log_path}")

    succeeded = sum(1 for r in results_log if r["success"])
    failed = sum(1 for r in results_log if not r["success"])
    print(f"\nResults: {succeeded} succeeded, {failed} failed out of {total}")


# =============================================================================
# TOKEN RELEVANCE RESULT COLLECTION AND PLOTTING
# =============================================================================


def find_token_relevance_files(organism: str, variant: str) -> Dict[str, List[Path]]:
    """Find token relevance JSON files for a given organism/variant.

    Returns:
        Dict mapping dataset keys to lists of JSON file paths.
    """
    organism_dir = f"{organism}_{variant}"

    results: Dict[str, List[Path]] = {}
    method_dirs = list(
        (RESULTS_BASE_DIR / MODEL / organism_dir).glob("diff_mining_*")
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


def extract_relevance_percentages(
    json_file: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Extract percentage and weighted_percentage from a single JSON file."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            return data.get("percentage"), data.get("weighted_percentage")
    except Exception as e:
        print(f"[WARNING] Could not read {json_file}: {e}")
        return None, None


def collect_and_plot_token_relevance(organisms: List[str], variants: List[str]):
    """Collect token relevance results and create per-organism bar charts."""
    print("\n" + "=" * 80)
    print("COLLECTING TOKEN RELEVANCE RESULTS")
    print("=" * 80)

    # results[organism][variant][dataset] = {"percentage": ..., "weighted_percentage": ...}
    results: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}

    for organism in organisms:
        for variant in variants:
            files_by_dataset = find_token_relevance_files(organism, variant)
            for dataset_key, files in files_by_dataset.items():
                pcts, wpcts = [], []
                for json_file in files:
                    pct, wpct = extract_relevance_percentages(json_file)
                    if pct is not None:
                        pcts.append(pct)
                    if wpct is not None:
                        wpcts.append(wpct)

                if pcts or wpcts:
                    results.setdefault(organism, {}).setdefault(variant, {})[dataset_key] = {
                        "percentage": np.mean(pcts) if pcts else None,
                        "weighted_percentage": np.mean(wpcts) if wpcts else None,
                    }
                    print(
                        f"  {organism}/{variant}/{dataset_key}: "
                        f"pct={np.mean(pcts):.2%}" if pcts else "N/A"
                    )

    if not results:
        print("[WARNING] No token relevance results found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_json_path = OUTPUT_DIR / "token_relevance_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results: {results_json_path}")

    for variant in variants:
        for metric_key, metric_title, filename in [
            ("percentage", "Token Relevance (Unweighted)", "token_relevance"),
            ("weighted_percentage", "Token Relevance (Weighted)", "token_relevance_weighted"),
        ]:
            org_labels = []
            values = []
            for organism in organisms:
                variant_data = results.get(organism, {}).get(variant, {})
                for dataset_key, metrics in variant_data.items():
                    val = metrics.get(metric_key)
                    if val is not None:
                        short_name = organism.replace("auditing_agents_", "")
                        org_labels.append(short_name)
                        values.append(val * 100)

            if not values:
                continue

            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(org_labels))
            ax.bar(x, values, color="#2ecc71", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(org_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel(f"{metric_title} (%)", fontsize=12)
            ax.set_title(f"{metric_title} by Organism ({variant})", fontsize=14)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()

            output_path = OUTPUT_DIR / f"{filename}_{variant}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {output_path}")
            plt.close()


# =============================================================================
# AGENT RESULT COLLECTION AND PLOTTING
# =============================================================================


def find_agent_files(organism: str, variant: str) -> Dict[str, List[Path]]:
    """Find agent hypothesis_grade_*.json files for a given organism/variant.

    Returns:
        Dict[mi_budget] = List[json_file_paths]
    """
    organism_dir = f"{organism}_{variant}"
    results: Dict[str, List[Path]] = {}

    method_dirs = list(
        (RESULTS_BASE_DIR / MODEL / organism_dir).glob("diff_mining_*")
    )
    for method_dir in method_dirs:
        for analysis_dir in method_dir.glob("analysis_*"):
            agent_dir = analysis_dir / "agent"
            if not agent_dir.exists():
                continue
            for run_dir in agent_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                dir_name = run_dir.name
                if not dir_name.startswith("DiffMining"):
                    continue
                for mi in AGENT_MI_BUDGETS:
                    if f"_mi{mi}_" in dir_name:
                        mi_key = f"mi{mi}"
                        for grade_file in run_dir.glob("hypothesis_grade_*.json"):
                            results.setdefault(mi_key, []).append(grade_file)
                        break

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


def collect_and_plot_agent_results(organisms: List[str], variants: List[str]):
    """Collect agent scores and create per-organism bar charts."""
    print("\n" + "=" * 80)
    print("COLLECTING AGENT RESULTS")
    print("=" * 80)

    # results[organism][variant][mi_key] = List[scores]
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for organism in organisms:
        for variant in variants:
            files_by_mi = find_agent_files(organism, variant)
            for mi_key, files in files_by_mi.items():
                scores = []
                for json_file in files:
                    score = extract_agent_score(json_file)
                    if score is not None:
                        scores.append(score)
                if scores:
                    results.setdefault(organism, {}).setdefault(variant, {})[mi_key] = scores
                    avg = np.mean(scores)
                    print(f"  {organism}/{variant}/{mi_key}: {len(scores)} runs, avg={avg:.2f}")

    if not results:
        print("[WARNING] No agent results found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_json_path = OUTPUT_DIR / "agent_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_json_path}")

    for variant in variants:
        for mi in AGENT_MI_BUDGETS:
            mi_key = f"mi{mi}"
            org_labels = []
            means = []
            stds = []

            for organism in organisms:
                scores = results.get(organism, {}).get(variant, {}).get(mi_key, [])
                if scores:
                    short_name = organism.replace("auditing_agents_", "")
                    org_labels.append(short_name)
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))

            if not means:
                continue

            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(org_labels))
            ax.bar(x, means, yerr=stds, color="#3498db", alpha=0.8, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(org_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Agent Score (1-5)", fontsize=12)
            ax.set_title(f"Agent Score by Organism ({variant}, {mi_key})", fontsize=14)
            ax.set_ylim(0, 5)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()

            output_path = OUTPUT_DIR / f"agent_score_{variant}_{mi_key}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {output_path}")
            plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run Diff Mining across auditing agents organisms (Qwen3-14B)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "diffing", "preprocessing"],
        default="full",
        help=(
            "'full' = preprocessing + token relevance + agent; "
            "'diffing' = preprocessing + token relevance (no agent); "
            "'preprocessing' = diff mining only (no relevance, no agent)"
        ),
    )
    parser.add_argument(
        "--organisms",
        nargs="+",
        default=None,
        help="Override which organisms to run (default: all 14)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=f"Override which variants to run (default: {VARIANTS})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: run first 2 organisms in preprocessing mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing token relevance results",
    )
    args = parser.parse_args()

    organisms = args.organisms if args.organisms else ORGANISMS
    variants = args.variants if args.variants else VARIANTS
    mode = args.mode

    if args.test:
        organisms = organisms[:3]
        mode = "preprocessing"
        print("[TEST MODE] Running first 3 organisms in preprocessing mode")

    if args.overwrite:
        TOKEN_RELEVANCE_CONFIG["overwrite"] = True

    print("=" * 80)
    print("AUDITING AGENTS RUNNER")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Model: {MODEL}")
    print(f"Infrastructure: {INFRASTRUCTURE}")
    print(f"Organisms ({len(organisms)}): {organisms}")
    print(f"Variants: {variants}")
    print(f"Token ordering methods: {TOKEN_ORDERING_METHODS}")
    print(f"Seed: {SEED}")
    print(f"N Samples: {N_SAMPLES}")
    print(f"Datasets: {[ds['id'] for ds in DATASETS]}")
    total = len(organisms) * len(variants)
    print(f"Total runs: {total}")
    print("=" * 80)

    if mode in ("full", "diffing", "preprocessing"):
        run_experiments(organisms, variants, TOKEN_ORDERING_METHODS, mode)

    if mode in ("full", "diffing"):
        collect_and_plot_token_relevance(organisms, variants)

    if mode == "full":
        collect_and_plot_agent_results(organisms, variants)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
