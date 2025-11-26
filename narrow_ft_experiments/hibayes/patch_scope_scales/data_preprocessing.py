import sys

sys.path.append("../../../")
sys.path.append(".")
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
from hibayes.analysis_state import AnalysisState

from src.utils.interactive import load_hydra_config


CONFIG_PATH = "configs/config.yaml"
DATA_OUTPUT_DIR = Path("narrow_ft_experiments/hibayes/patch_scope_scales/data")
DATASET_DIR_NAME = "fineweb-1m-sample"

SCALES = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    3.0,
    4.0,
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    120.0,
    140.0,
    160.0,
    180.0,
    200.0,
]

COARSE_BIN_SIZE = 3
assert len(SCALES) % COARSE_BIN_SIZE == 0
NUM_COARSE_CLASSES = len(SCALES) // COARSE_BIN_SIZE
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


def _iter_auto_patch_scope_paths(
    dataset_dir: Path,
) -> List[Tuple[Path, int, str]]:
    """
    Return list of (pt_path, position, grader_model_id) for auto_patch_scope outputs.

    Only unprefixed auto_patch_scope_pos_* files are considered (diff latent),
    and positions are restricted to 0, 1, 2, 3, 4.
    """
    assert (
        dataset_dir.exists() and dataset_dir.is_dir()
    ), f"Dataset dir missing: {dataset_dir}"
    out: List[Tuple[Path, int, str]] = []
    prefix = "auto_patch_scope_pos_"

    for fp in dataset_dir.iterdir():
        if not fp.is_file():
            continue
        name = fp.name
        if not (name.startswith(prefix) and name.endswith(".pt")):
            continue
        stem = name[: -len(".pt")]
        tail = stem[len(prefix) :]
        parts = tail.split("_")
        assert (
            len(parts) >= 2
        ), f"Unexpected auto_patch_scope filename structure: {name}"
        pos_str = parts[0]
        try:
            pos = int(pos_str)
        except ValueError as exc:
            raise AssertionError(f"Non-integer position in filename {name}") from exc
        grader_sanitized = "_".join(parts[1:])
        assert (
            len(grader_sanitized) > 0
        ), f"Missing grader identifier in filename {name}"
        grader_model_id = grader_sanitized.replace("_", "/")
        out.append((fp, pos, grader_model_id))

    assert len(out) >= 1, f"No auto_patch_scope .pt files found under {dataset_dir}"
    return sorted(out, key=lambda t: (t[1], t[2]))


def load_all_auto_patch_scope_scales() -> pd.DataFrame:
    """Load best_scale values for auto_patch_scope across configured entities.

    Each row corresponds to a single (model, organism, layer, position, grader_model_id)
    combination for which an auto_patch_scope_pos_* file exists.
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
        assert (
            layer_dir.exists() and layer_dir.is_dir()
        ), f"Missing layer dir: {layer_dir}"

        dataset_dir = layer_dir / DATASET_DIR_NAME
        aps_entries = _iter_auto_patch_scope_paths(dataset_dir)

        for pt_path, position, grader_model_id in aps_entries:
            if position not in allowed_positions:
                continue
            payload = torch.load(pt_path, map_location="cpu")
            assert isinstance(
                payload, dict
            ), f"Expected dict in {pt_path}, got {type(payload)}"
            assert "best_scale" in payload, f"'best_scale' missing in {pt_path}"

            best_scale = float(payload["best_scale"])

            rows.append(
                {
                    "model": model,
                    "organism": organism,
                    "organism_type": organism_type,
                    "layer": int(layer),
                    "dataset_dir": DATASET_DIR_NAME,
                    "position": int(position),
                    "grader_model_id": grader_model_id,
                    "best_scale": best_scale,
                    "best_scale_index": SCALES.index(best_scale),
                }
            )

    df = pd.DataFrame(rows)
    assert not df.empty

    # Ensure the observed outcome used by HiBayes is strictly integer-typed.
    assert "best_scale_index" in df.columns
    best_idx = df["best_scale_index"]
    best_idx_int = best_idx.astype("int32")
    assert (best_idx == best_idx_int).all(), "best_scale_index must be integer-valued"
    df["best_scale_index"] = best_idx_int

    # Coarse ordinal outcome: group three adjacent original categories per bin.
    best_scale_bin = (df["best_scale_index"] // COARSE_BIN_SIZE).astype("int32")
    assert best_scale_bin.min() >= 0
    assert best_scale_bin.max() < NUM_COARSE_CLASSES
    df["best_scale_bin"] = best_scale_bin

    return df


if __name__ == "__main__":
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_auto_patch_scope_scales()
    output_path = DATA_OUTPUT_DIR / "auto_patch_scope_scales_all.csv"
    df.to_csv(output_path, index=False)
    df["task"] = "auto_patch_scope_scales"
    state = AnalysisState(data=df, processed_data=df.copy())
    state.save(DATA_OUTPUT_DIR)
    print(f"Saved {len(df)} rows to {output_path}")
