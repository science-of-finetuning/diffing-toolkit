import sys

sys.path.append("../../../")
sys.path.append(".")
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import pandas as pd
from hibayes.analysis_state import AnalysisState

from diffing.utils.interactive import load_hydra_config

CONFIG_PATH = "configs/config.yaml"
DATA_OUTPUT_DIR = Path("narrow_ft_experiments/hibayes/token_relevance/data")
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
    folder = cfg.diffing.results_dir
    if cfg.organism_variant != "default":
        folder = folder + f"_{cfg.organism_variant}"
    root = Path(folder) / "activation_difference_lens"
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _select_dataset_dir(layer_dir: Path, cfg: Any) -> Path:
    assert layer_dir.exists() and layer_dir.is_dir(), f"Missing layer dir: {layer_dir}"
    if DATASET_DIR_NAME is not None:
        cand = layer_dir / DATASET_DIR_NAME
        if cand.exists() and cand.is_dir():
            return cand
    candidates = sorted([p for p in layer_dir.iterdir() if p.is_dir()])
    assert len(candidates) >= 1
    pref = getattr(cfg, "pretraining_dataset", None)
    if pref is not None:
        base = str(pref.id).split("/")[-1]
        for p in candidates:
            if p.name == base:
                return p
    return candidates[0]


def _iter_token_relevance_json_paths(
    tr_root: Path,
) -> List[Tuple[Path, int, str]]:
    """Return list of (json_path, position, variant) under a token_relevance root."""
    assert (
        tr_root.exists() and tr_root.is_dir()
    ), f"Token relevance dir missing: {tr_root}"
    out: List[Tuple[Path, int, str]] = []
    for pos_dir in tr_root.iterdir():
        if not pos_dir.is_dir():
            continue
        name = pos_dir.name
        if not name.startswith("position_"):
            continue
        pos_str = name.split("_")[-1]
        try:
            pos = int(pos_str)
            if pos not in {0, 1, 2, 3, 4}:
                continue
        except ValueError:
            continue
        for variant_dir in pos_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name
            for fp in variant_dir.iterdir():
                if not fp.is_file() or fp.suffix != ".json":
                    continue
                fname = fp.name
                if not (
                    fname.startswith("relevance_logitlens_")
                    or fname.startswith("relevance_patchscope_")
                ):
                    # Skip legacy or unrelated files such as old_relevance_*.json
                    continue
                out.append((fp, pos, variant))
    assert len(out) >= 1, f"No token relevance json files found under {tr_root}"
    return sorted(out, key=lambda t: (t[1], t[0].name))


def _decode_source_and_grader(fname: str) -> Tuple[str, str]:
    """Parse source and grader_model_id from relevance_<source>_<normalized>.json."""
    assert fname.endswith(".json")
    stem = fname[: -len(".json")]
    prefix_logit = "relevance_logitlens_"
    prefix_ps = "relevance_patchscope_"
    if stem.startswith(prefix_logit):
        source = "logitlens"
        norm = stem[len(prefix_logit) :]
    elif stem.startswith(prefix_ps):
        source = "patchscope"
        norm = stem[len(prefix_ps) :]
    else:
        assert False, f"Unrecognized relevance filename: {fname}"
    assert len(norm) >= 1
    grader_model_id = norm.replace("_", "/")
    return source, grader_model_id


def load_all_token_relevance_data() -> pd.DataFrame:
    """Load token-level relevance labels for all configured entities and graders.

    Each row corresponds to a single token judged by a specific grader model.
    The response variable `score` is 1 for RELEVANT and 0 for IRRELEVANT.
    """
    rows: List[Dict[str, Any]] = []

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

        selected_ds_dir = _select_dataset_dir(layer_dir, cfg)
        ds_name = selected_ds_dir.name
        tr_dir = selected_ds_dir / "token_relevance"
        json_entries = _iter_token_relevance_json_paths(tr_dir)

        for json_path, position, variant in json_entries:
            source, grader_model_id = _decode_source_and_grader(json_path.name)
            with open(json_path, "r", encoding="utf-8") as f:
                rec: Dict[str, Any] = json.load(f)
            assert isinstance(rec, dict)
            labels = list(rec.get("labels", []))
            tokens = list(rec.get("tokens", []))
            assert len(labels) == len(tokens) and len(labels) >= 1

            for idx, (tok, lbl) in enumerate(zip(tokens, labels)):
                assert lbl in {"RELEVANT", "IRRELEVANT"}
                score = 1 if lbl == "RELEVANT" else 0
                rows.append(
                    {
                        "model": model,
                        "organism": organism,
                        "organism_type": organism_type,
                        "layer": int(layer),
                        "dataset_dir": ds_name,
                        "position": int(position),
                        "variant": variant,
                        "source": source,
                        "grader_model_id": grader_model_id,
                        "token_index": int(idx),
                        "token": str(tok),
                        "label": lbl,
                        "score": int(score),
                        "datapoint_id": (
                            model
                            + "|"
                            + organism
                            + "|"
                            + organism_type
                            + "|"
                            + variant
                            + "|"
                            + source
                            + "|"
                            + str(position)
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    assert not df.empty
    return df


if __name__ == "__main__":
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_token_relevance_data()
    output_path = DATA_OUTPUT_DIR / "token_relevance_tokens_all.csv"
    df.to_csv(output_path, index=False)
    df["task"] = "token_relevance"
    state = AnalysisState(data=df)
    state.save(DATA_OUTPUT_DIR)
    print(state)
    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())
