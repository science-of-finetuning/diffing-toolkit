import sys

sys.path.append("../../../")
sys.path.append(".")
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import pandas as pd
from hibayes.analysis_state import AnalysisState

from src.utils.interactive import load_hydra_config


CONFIG_PATH = "configs/config.yaml"
DATA_OUTPUT_DIR = Path("narrow_ft_experiments/hibayes/steering_strength/data")
DATASET_DIR_NAME = "fineweb-1m-sample"


# (model, organism, organism_type, layer)
ENTRIES_GROUPED: List[Tuple[str, str, str, int]] = [
    ("qwen3_1_7B", "kansas_abortion", "SDF", 13),
    ("qwen3_1_7B", "cake_bake", "SDF", 13),
    ("qwen3_1_7B", "roman_concrete", "SDF", 13),
    ("qwen3_1_7B", "ignore_comment", "SDF", 13),
    ("qwen3_1_7B", "fda_approval", "SDF", 13),
    ("gemma3_1B", "fda_approval", "SDF", 12),
    ("gemma3_1B", "cake_bake", "SDF", 12),
    ("gemma3_1B", "kansas_abortion", "SDF", 12),
    ("gemma3_1B", "roman_concrete", "SDF", 12),
    ("gemma3_1B", "ignore_comment", "SDF", 12),
    ("llama32_1B_Instruct", "cake_bake", "SDF", 7),
    ("llama32_1B_Instruct", "kansas_abortion", "SDF", 7),
    ("llama32_1B_Instruct", "roman_concrete", "SDF", 7),
    ("llama32_1B_Instruct", "fda_approval", "SDF", 7),
    ("llama32_1B_Instruct", "ignore_comment", "SDF", 7),
    ("qwen3_32B", "cake_bake", "SDF", 31),
    ("qwen3_32B", "kansas_abortion", "SDF", 31),
    ("qwen3_32B", "roman_concrete", "SDF", 31),
    ("qwen3_32B", "ignore_comment", "SDF", 31),
    ("qwen3_32B", "fda_approval", "SDF", 31),
    ("qwen3_1_7B", "taboo_smile", "Taboo", 13),
    ("qwen3_1_7B", "taboo_gold", "Taboo", 13),
    ("qwen3_1_7B", "taboo_leaf", "Taboo", 13),
    ("gemma2_9B_it", "taboo_smile", "Taboo", 20),
    ("gemma2_9B_it", "taboo_gold", "Taboo", 20),
    ("gemma2_9B_it", "taboo_leaf", "Taboo", 20),
    ("qwen25_7B_Instruct", "subliminal_learning_cat", "Subliminal", 13),
    ("llama31_8B_Instruct", "em_bad_medical_advice", "EM", 15),
    ("llama31_8B_Instruct", "em_risky_financial_advice", "EM", 15),
    ("llama31_8B_Instruct", "em_extreme_sports", "EM", 15),
    ("qwen25_7B_Instruct", "em_bad_medical_advice", "EM", 13),
    ("qwen25_7B_Instruct", "em_risky_financial_advice", "EM", 13),
    ("qwen25_7B_Instruct", "em_extreme_sports", "EM", 13),
]


def _results_root_from_cfg(cfg: Any) -> Path:
    root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _iter_steering_threshold_paths(
    steering_root: Path,
) -> List[Tuple[Path, int, str]]:
    """
    Return list of (threshold_path, position, grader_model_id) under a steering root.
    """
    assert steering_root.exists() and steering_root.is_dir(), f"Steering dir missing: {steering_root}"
    out: List[Tuple[Path, int, str]] = []

    for pos_dir in steering_root.iterdir():
        if not pos_dir.is_dir():
            continue
        name = pos_dir.name
        if not name.startswith("position_"):
            continue
        parts = name.split("_")
        assert len(parts) >= 3, f"Unexpected steering position folder name: {name}"
        pos_str = parts[1]
        try:
            pos = int(pos_str)
        except ValueError as exc:
            raise AssertionError(f"Non-integer position in steering folder {name}") from exc
        grader_sanitized = "_".join(parts[2:])
        assert len(grader_sanitized) > 0, f"Missing grader suffix in steering folder {name}"
        grader_model_id = grader_sanitized.replace("_", "/")

        threshold_path = pos_dir / "threshold.json"
        assert threshold_path.exists() and threshold_path.is_file(), f"Missing threshold.json in {pos_dir}"

        out.append((threshold_path, pos, grader_model_id))

    assert len(out) >= 1, f"No steering thresholds found under {steering_root}"
    return sorted(out, key=lambda t: (t[1], t[2]))


def load_all_steering_thresholds() -> pd.DataFrame:
    """Load coherent steering thresholds across configured entities and graders.

    Each row corresponds to a single (model, organism, layer, position, grader_model_id)
    combination with an avg_threshold recorded in threshold.json.
    """
    rows: List[Dict[str, Any]] = []
    allowed_positions = {0, 1, 2, 3, 4}

    for model, organism, organism_type, layer in ENTRIES_GROUPED:
        cfg = load_hydra_config(
            CONFIG_PATH,
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
        )
        results_root = _results_root_from_cfg(cfg)
        layer_dir = results_root / f"layer_{layer}"
        assert layer_dir.exists() and layer_dir.is_dir(), f"Missing layer dir: {layer_dir}"

        dataset_dir = layer_dir / DATASET_DIR_NAME
        assert dataset_dir.exists() and dataset_dir.is_dir(), f"Missing dataset dir: {dataset_dir}"

        steering_root = dataset_dir / "steering"
        entries = _iter_steering_threshold_paths(steering_root)

        for threshold_path, position, grader_model_id in entries:
            if position not in allowed_positions:
                continue
            with threshold_path.open("r", encoding="utf-8") as f:
                payload: Dict[str, Any] = json.load(f)
            assert isinstance(payload, dict)
            assert "avg_threshold" in payload

            thresholds_raw = payload.get("thresholds", [])
            thresholds_list = list(thresholds_raw) if isinstance(thresholds_raw, list) else []

            avg_threshold = float(payload["avg_threshold"])

            rows.append(
                {
                    "model": model,
                    "organism": organism,
                    "organism_type": organism_type,
                    "layer": int(layer),
                    "dataset_dir": DATASET_DIR_NAME,
                    "position": int(position),
                    "grader_model_id": grader_model_id,
                    "avg_threshold": avg_threshold,
                    "num_prompts": int(len(thresholds_list)),
                }
            )

    df = pd.DataFrame(rows)
    assert not df.empty
    return df


if __name__ == "__main__":
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_steering_thresholds()
    output_path = DATA_OUTPUT_DIR / "steering_thresholds_all.csv"
    df.to_csv(output_path, index=False)
    df["task"] = "steering_strength"
    state = AnalysisState(data=df, processed_data=df.copy())
    state.save(DATA_OUTPUT_DIR)
    print(state)
    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())




