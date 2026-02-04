from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import re
from loguru import logger

from .token_ordering import (
    read_dataset_orderings_index,
    read_ordering,
)


def _method_dir_for_extraction(
    *,
    base_results_dir: Path,
    extraction_method: str,
    extraction_layer: float | None,
) -> Path:
    assert extraction_method in {"logits", "logit_lens", "patchscope_lens"}, (
        "agent.overview.extraction_method must be one of "
        "{'logits','logit_lens','patchscope_lens'}"
    )
    if extraction_method in {"logit_lens", "patchscope_lens"}:
        assert extraction_layer is not None, (
            "agent.overview.extraction_layer must be set when extraction_method is "
            f"{extraction_method!r}"
        )
        assert 0.0 <= float(extraction_layer) <= 1.0, (
            f"agent.overview.extraction_layer must be in [0, 1], got {extraction_layer}"
        )

    desired_suffix = f"_logit_extraction_{extraction_method}"
    if extraction_method in {"logit_lens", "patchscope_lens"}:
        layer_str = str(float(extraction_layer)).replace(".", "p")
        desired_suffix += f"_layer_{layer_str}"

    name = base_results_dir.name
    m = re.search(
        r"_logit_extraction_[a-z0-9_]+(?:_layer_[0-9]+(?:p[0-9]+)?)?$",
        name,
    )
    assert m is not None, (
        f"Could not parse logit extraction suffix from base_results_dir name: {name!r}"
    )
    variant_name = name[: m.start()] + desired_suffix
    variant_dir = base_results_dir.with_name(variant_name)
    assert variant_dir.exists() and variant_dir.is_dir(), (
        f"Requested extraction variant directory does not exist: {variant_dir}"
    )
    return variant_dir


def _allocate_token_budgets(total: int, num_groups: int) -> List[int]:
    assert total >= 0
    assert num_groups > 0
    base = total // num_groups
    rem = total % num_groups
    return [base + (1 if i < rem else 0) for i in range(num_groups)]


def _load_token_groups(
    ordering_dir: Path,
    dataset_name: str,
    top_k_tokens: int,
) -> Optional[List[List[Dict[str, Any]]]]:
    """
    Load token groups from ordering directory.
    
    Args:
        ordering_dir: Ordering directory path (contains per-dataset subdirs)
        dataset_name: Name of the dataset
        top_k_tokens: Total number of tokens to return across all groups
        
    Returns:
        List of token groups, or None if not available
    """
    if not ordering_dir.exists():
        return None

    dataset_dir = ordering_dir / dataset_name
    if not dataset_dir.exists():
        return None
    
    # Read orderings index
    index = read_dataset_orderings_index(dataset_dir)
    if not index or not index.get("orderings"):
        return None

    ordering_entries = list(index["orderings"])
    budgets = _allocate_token_budgets(int(top_k_tokens), len(ordering_entries))

    out_groups: List[List[Dict[str, Any]]] = []
    for ordering_entry, budget in zip(ordering_entries, budgets):
        if budget <= 0:
            continue
        ordering_id = ordering_entry["ordering_id"]
        ordering_data = read_ordering(dataset_dir, ordering_id)
        if not ordering_data:
            return None

        tokens_group: List[Dict[str, Any]] = []
        for t in ordering_data.get("tokens", [])[:budget]:
            token_dict: Dict[str, Any] = {
                "token_str": t["token_str"],
                "ordering_value": round(t.get("ordering_value", 0), 2),
            }
            tokens_group.append(token_dict)
        out_groups.append(tokens_group)

    return out_groups


def get_overview(method: Any, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build overview for Diff Mining agent.
    
    Returns:
        Tuple of (overview_data, dataset_mapping) where dataset_mapping maps
        anonymized names (ds1, ds2, ...) to real dataset names
    """
    logger.info("AgentTool: get_overview")
    
    datasets: List[str] = list(cfg.get("datasets", []))

    extraction_method = cfg.get("extraction_method", None)
    assert extraction_method is not None, "agent.overview.extraction_method must be set"
    extraction_layer = cfg.get("extraction_layer", None)

    base_results_dir = _method_dir_for_extraction(
        base_results_dir=method.base_results_dir,
        extraction_method=str(extraction_method),
        extraction_layer=None if extraction_layer is None else float(extraction_layer),
    )

    assert hasattr(method, "_get_run_folder_name"), "DiffMiningMethod must define _get_run_folder_name"
    run_dir = base_results_dir / method._get_run_folder_name()
    assert run_dir.exists() and run_dir.is_dir(), (
        f"No diffing run results found in {run_dir}. Run pipeline.mode=diffing first."
    )

    run_meta: Dict[str, Any] = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            run_meta = json.load(f)
    
    ordering_type = cfg.get("ordering_type", None)
    assert ordering_type is not None, "agent.overview.ordering_type must be set"
    ordering_type = str(ordering_type)

    assert hasattr(method, "_ordering_dir_name"), "DiffMiningMethod must define _ordering_dir_name"
    ordering_dir = run_dir / method._ordering_dir_name(ordering_type)
    assert ordering_dir.exists() and ordering_dir.is_dir(), (
        f"Ordering directory not found: {ordering_dir}. "
        "Check agent.overview.ordering_type and whether the diffing run enabled that ordering."
    )

    if len(datasets) == 0:
        datasets = sorted(
            {
                d.name
                for d in ordering_dir.iterdir()
                if d.is_dir() and (d / "orderings.json").exists()
            }
        )
        assert len(datasets) > 0, f"No dataset orderings found in {ordering_dir}."
    
    # Create dataset name mapping for blinding
    dataset_mapping: Dict[str, str] = {}
    for i, ds in enumerate(datasets, start=1):
        dataset_mapping[f"ds{i}"] = ds
    
    out: Dict[str, Any] = {"datasets": {}}
    
    top_k_tokens_raw = cfg.get("top_k_tokens", None)
    assert top_k_tokens_raw is not None, "agent.overview.top_k_tokens must be set"
    top_k_tokens = int(top_k_tokens_raw)
    
    for i, ds in enumerate(datasets, start=1):
        anonymized_name = f"ds{i}"
        token_groups = _load_token_groups(ordering_dir, ds, top_k_tokens)
        assert token_groups is not None, (
            f"No ordering data found for dataset={ds!r} "
            f"(ordering_type={ordering_type!r}) in run_dir={run_dir}"
        )

        num_samples = int(run_meta.get("max_samples", 0))
        num_tokens_total = sum(len(g) for g in token_groups)
        logger.info(
            f"Loaded {len(token_groups)} token groups / {num_tokens_total} tokens for {ds}"
        )

        out["datasets"][anonymized_name] = {
            "token_groups": token_groups,
            "num_samples": num_samples,
            "metadata": {
                "num_token_groups": len(token_groups),
                "num_tokens_shown": num_tokens_total,
            }
        }
            
    return out, dataset_mapping


# Steering functionality is not implemented for DiffMining method
# If needed in the future, implement get_steering_samples and generate_steered here

