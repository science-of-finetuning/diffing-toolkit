import sys

sys.path.append("..")
import re
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from hibayes.analysis_state import AnalysisState
from src.utils.interactive import load_hydra_config

GPT5 = "openai/gpt-5"
GEMINI25PRO = "google/gemini-2.5-pro"
entries_grouped = [
    ("qwen3_1_7B", "cake_bake", "SDF", GEMINI25PRO),
    ("qwen3_1_7B", "kansas_abortion", "SDF", GEMINI25PRO),
    ("qwen3_1_7B", "roman_concrete", "SDF", GEMINI25PRO),
    ("qwen3_1_7B", "ignore_comment", "SDF", GEMINI25PRO),
    ("qwen3_1_7B", "fda_approval", "SDF", GEMINI25PRO),
    ("gemma3_1B", "ignore_comment", "SDF", GEMINI25PRO),
    ("gemma3_1B", "fda_approval", "SDF", GEMINI25PRO),
    ("gemma3_1B", "cake_bake", "SDF", GEMINI25PRO),
    ("gemma3_1B", "kansas_abortion", "SDF", GEMINI25PRO),
    ("gemma3_1B", "roman_concrete", "SDF", GEMINI25PRO),
    ("llama32_1B_Instruct", "cake_bake", "SDF", GEMINI25PRO),
    ("llama32_1B_Instruct", "kansas_abortion", "SDF", GEMINI25PRO),
    ("llama32_1B_Instruct", "roman_concrete", "SDF", GEMINI25PRO),
    ("llama32_1B_Instruct", "fda_approval", "SDF", GEMINI25PRO),
    ("llama32_1B_Instruct", "ignore_comment", "SDF", GEMINI25PRO),
    ("qwen3_1_7B", "taboo_smile", "Taboo", GEMINI25PRO),
    ("qwen3_1_7B", "taboo_gold", "Taboo", GEMINI25PRO),
    ("qwen3_1_7B", "taboo_leaf", "Taboo", GEMINI25PRO),
    ("gemma2_9B_it", "taboo_smile", "Taboo", GEMINI25PRO),
    ("gemma2_9B_it", "taboo_gold", "Taboo", GEMINI25PRO),
    ("gemma2_9B_it", "taboo_leaf", "Taboo", GEMINI25PRO),
    ("qwen25_7B_Instruct", "subliminal_learning_cat", "Subliminal", GEMINI25PRO),
    ("llama31_8B_Instruct", "em_bad_medical_advice", "EM", GEMINI25PRO),
    ("llama31_8B_Instruct", "em_risky_financial_advice", "EM", GEMINI25PRO),
    ("llama31_8B_Instruct", "em_extreme_sports", "EM", GEMINI25PRO),
    ("qwen25_7B_Instruct", "em_bad_medical_advice", "EM", GEMINI25PRO),
    ("qwen25_7B_Instruct", "em_risky_financial_advice", "EM", GEMINI25PRO),
    ("qwen25_7B_Instruct", "em_extreme_sports", "EM", GEMINI25PRO),
    ("qwen3_32B", "cake_bake", "SDF", GEMINI25PRO),
    ("qwen3_32B", "kansas_abortion", "SDF", GEMINI25PRO),
    ("qwen3_32B", "roman_concrete", "SDF", GEMINI25PRO),
    ("qwen3_32B", "ignore_comment", "SDF", GEMINI25PRO),
    ("qwen3_32B", "fda_approval", "SDF", GEMINI25PRO),
    ("qwen3_1_7B", "cake_bake", "SDF", GPT5),
    ("qwen3_1_7B", "kansas_abortion", "SDF", GPT5),
    ("qwen3_1_7B", "roman_concrete", "SDF", GPT5),
    ("qwen3_1_7B", "ignore_comment", "SDF", GPT5),
    ("qwen3_1_7B", "fda_approval", "SDF", GPT5),
    ("gemma3_1B", "ignore_comment", "SDF", GPT5),
    ("gemma3_1B", "fda_approval", "SDF", GPT5),
    ("gemma3_1B", "cake_bake", "SDF", GPT5),
    ("gemma3_1B", "kansas_abortion", "SDF", GPT5),
    ("gemma3_1B", "roman_concrete", "SDF", GPT5),
    ("llama32_1B_Instruct", "cake_bake", "SDF", GPT5),
    ("llama32_1B_Instruct", "kansas_abortion", "SDF", GPT5),
    ("llama32_1B_Instruct", "roman_concrete", "SDF", GPT5),
    ("llama32_1B_Instruct", "fda_approval", "SDF", GPT5),
    ("llama32_1B_Instruct", "ignore_comment", "SDF", GPT5),
    ("qwen3_32B", "cake_bake", "SDF", GPT5),
    ("qwen3_32B", "kansas_abortion", "SDF", GPT5),
    ("qwen3_32B", "roman_concrete", "SDF", GPT5),
    ("qwen3_32B", "ignore_comment", "SDF", GPT5),
    ("qwen3_32B", "fda_approval", "SDF", GPT5),
    ("qwen3_1_7B", "taboo_smile", "Taboo", GPT5),
    ("qwen3_1_7B", "taboo_gold", "Taboo", GPT5),
    ("qwen3_1_7B", "taboo_leaf", "Taboo", GPT5),
    ("gemma2_9B_it", "taboo_smile", "Taboo", GPT5),
    ("gemma2_9B_it", "taboo_gold", "Taboo", GPT5),
    ("gemma2_9B_it", "taboo_leaf", "Taboo", GPT5),
    ("qwen25_7B_Instruct", "subliminal_learning_cat", "Subliminal", GPT5),
    ("llama31_8B_Instruct", "em_bad_medical_advice", "EM", GPT5),
    ("llama31_8B_Instruct", "em_risky_financial_advice", "EM", GPT5),
    ("llama31_8B_Instruct", "em_extreme_sports", "EM", GPT5),
    ("qwen25_7B_Instruct", "em_bad_medical_advice", "EM", GPT5),
    ("qwen25_7B_Instruct", "em_risky_financial_advice", "EM", GPT5),
    ("qwen25_7B_Instruct", "em_extreme_sports", "EM", GPT5),
]

CONFIG_PATH = "configs/config.yaml"

VARIANTS = [
    ("agent_mi0", "ADL^{i=0}"),
    ("agent_mi5", "ADL^{i=5}"),
    ("baseline_mi0", "Blackbox^{i=0}"),
    ("baseline_mi5", "Blackbox^{i=5}"),
    ("baseline_mi50", "Blackbox^{i=50}"),
]


def load_all_grade_data() -> pd.DataFrame:
    """Load all grade data from entries_grouped, including all grader runs."""

    rows = []

    for model, organism, organism_type, agent_model in entries_grouped:
        cfg = load_hydra_config(
            CONFIG_PATH,
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
            f"diffing.evaluation.agent.llm.model_id={agent_model}",
        )
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir()

        agent_root = results_root / "agent"
        assert agent_root.exists() and agent_root.is_dir()

        for variant_key, variant_label in VARIANTS:
            mi = int(variant_key.split("_mi")[1])
            is_baseline = variant_key.startswith("baseline")

            grade_paths = _find_all_grade_paths(
                agent_root, organism, model, mi, is_baseline, agent_model
            )

            for grade_path in grade_paths:
                agent_run_idx = _extract_agent_run_idx(grade_path.parent.name)
                score, grader_model_id, grader_run_idx = _load_grade_entry(grade_path)

                rows.append(
                    {
                        "model": model,
                        "organism": organism,
                        "organism_type": organism_type,
                        "ADL": "Baseline" if is_baseline else "ADL",
                        "variant_label": variant_label,
                        "run_idx": agent_run_idx,
                        "interactions": mi,
                        "score": int(score),
                        "llm": agent_model.split("/")[-1],
                        "grader_model_id": grader_model_id,
                        "grader_run_idx": grader_run_idx,
                    }
                )

    return pd.DataFrame(rows)


def _find_all_grade_paths(
    agent_root: Path,
    organism: str,
    model: str,
    mi: int,
    is_baseline: bool,
    agent_model: str,
) -> List[Path]:
    """
    Find all matching hypothesis_grade_* JSON paths for all runs of a variant.

    This expects the new directory layout with one directory per agent run,
    e.g. ADL_openai_gpt-5_mi0_run0 or Blackbox_openai_gpt-5_mi0_run0, each
    containing multiple hypothesis_grade_<grader>_<run>.json files.
    """
    agent_id = agent_model.replace("/", "_")
    if is_baseline:
        dir_pattern_str = (
            r"^Blackbox_"
            + re.escape(agent_id)
            + r"_mi"
            + re.escape(str(mi))
            + r"_run\d+$"
        )
    else:
        dir_pattern_str = (
            r"^ADL_"
            + re.escape(agent_id)
            + r"_mi"
            + re.escape(str(mi))
            + r"_run\d+$"
        )
    dir_pattern = re.compile(dir_pattern_str)

    out: List[Path] = []
    for child in agent_root.iterdir():
        if not child.is_dir():
            continue
        if dir_pattern.match(child.name) is None:
            continue

        for grade_file in child.iterdir():
            if not grade_file.is_file():
                continue
            name = grade_file.name
            if not name.startswith("hypothesis_grade_"):
                continue
            if name == "hypothesis_grade.json":
                continue
            if grade_file.suffix != ".json":
                continue
            out.append(grade_file)

    assert (
        len(out) >= 1
    ), f"No grader files found for {organism} {model} mi={mi} baseline={is_baseline} agent_model={agent_model}"
    return out


def _extract_agent_run_idx(run_dir_name: str) -> int:
    """Extract the agent run index from directory names like ADL_*_run0."""
    match = re.search(r"_run(\d+)$", run_dir_name)
    assert (
        match is not None
    ), f"Could not extract agent run index from directory name: {run_dir_name}"
    return int(match.group(1))


def _load_grade_entry(json_path: Path) -> Tuple[float, str, int]:
    """Load score and grader metadata from a hypothesis_grade_*.json file."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict)
    assert "score" in payload
    assert "grader_model_id" in payload

    s = int(payload["score"])
    assert 1 <= s <= 6

    grader_model_id = str(payload["grader_model_id"])
    grader_run_idx = int(payload["run_idx"]) if "run_idx" in payload else 0

    return float(s), grader_model_id, grader_run_idx


if __name__ == "__main__":
    df = load_all_grade_data()
    output_path = Path("hibayes/data/grades_all_runs.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df["task"] = "narrow_ft"
    df["score"] = df["score"].astype(int) - 1  # convert to 0-4 scale
    
    state = AnalysisState(data=df)
    state.save(output_path.parent)
    print(state)
    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())
