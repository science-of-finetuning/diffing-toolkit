from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

from .token_ordering import (
    read_dataset_orderings_index,
    read_ordering,
)


def _find_run_dir(method) -> Optional[Path]:
    """
    Find the most recent run directory (new schema).
    
    Returns:
        Path to run directory, or None if not found
    """
    base_dir = method.base_results_dir
    if not base_dir.exists():
        return None
    
    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    return run_dirs[0] if run_dirs else None


def _load_from_new_schema(
    run_dir: Path,
    dataset_name: str,
    selection_mode: str,
    top_k_tokens: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Load token list from new schema based on selection mode.
    
    Args:
        run_dir: Run directory path
        dataset_name: Name of the dataset
        selection_mode: Token selection mode (top_k_occurring or fraction_positive_diff)
        top_k_tokens: Maximum number of tokens to return
        
    Returns:
        List of token dicts, or None if not available
    """
    ordering_type_id_by_selection_mode = {
        "top_k_occurring": "topk_occurring",
        "fraction_positive_diff": "fraction_positive_diff",
    }
    assert selection_mode in ordering_type_id_by_selection_mode, (
        f"Unknown token_set_selection_mode={selection_mode!r}. "
        f"Expected one of: {sorted(ordering_type_id_by_selection_mode.keys())}"
    )
    ordering_type_id = ordering_type_id_by_selection_mode[selection_mode]
    
    ordering_type_dir = run_dir / ordering_type_id
    if not ordering_type_dir.exists():
        return None
    
    dataset_dir = ordering_type_dir / dataset_name
    if not dataset_dir.exists():
        return None
    
    # Read orderings index
    index = read_dataset_orderings_index(dataset_dir)
    if not index or not index.get("orderings"):
        return None
    
    # Get the first (global) ordering
    first_ordering = index["orderings"][0]
    ordering_data = read_ordering(dataset_dir, first_ordering["ordering_id"])
    if not ordering_data:
        return None
    
    tokens = []
    for t in ordering_data.get("tokens", [])[:top_k_tokens]:
        ordering_value = float(t.get("ordering_value", 0.0))
        token_dict: Dict[str, Any] = {
            "token_id": t["token_id"],
            "token_str": t["token_str"],
            "count_positive": t.get("count_positive", 0),
            "count_negative": t.get("count_negative", 0),
            "avg_logit_diff": t.get("avg_logit_diff", 0),
        }
        if selection_mode == "top_k_occurring":
            token_dict["positive_occurrence_rate"] = ordering_value
            token_dict["negative_occurrence_rate"] = 0
        elif selection_mode == "fraction_positive_diff":
            token_dict["fraction_positive"] = ordering_value
            token_dict["positive_occurrence_rate"] = ordering_value * 100
            token_dict["negative_occurrence_rate"] = (1.0 - ordering_value) * 100
        else:
            raise AssertionError("unreachable")
        tokens.append(token_dict)
    
    return tokens


def get_overview(method: Any, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build overview for Diff Mining agent.
    
    Token selection method is controlled by token_set_selection_mode config.
    
    Returns:
        Tuple of (overview_data, dataset_mapping) where dataset_mapping maps
        anonymized names (ds1, ds2, ...) to real dataset names
    """
    logger.info("AgentTool: get_overview")
    
    datasets: List[str] = list(cfg.get("datasets", []))
    
    run_dir = _find_run_dir(method)
    assert run_dir is not None, (
        f"No diffing run results found in {method.base_results_dir}. "
        f"Expected a 'run_*' directory. "
        f"Run 'pipeline.mode=diffing' with 'diffing/method=diff_mining' first before running evaluation."
    )
    run_meta: Dict[str, Any] = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            run_meta = json.load(f)
    
    if len(datasets) == 0:
        # Autodiscover datasets
        for ordering_dir in run_dir.iterdir():
            if ordering_dir.is_dir() and (ordering_dir / "metadata.json").exists():
                for ds_dir in ordering_dir.iterdir():
                    if ds_dir.is_dir() and (ds_dir / "orderings.json").exists():
                        datasets.append(ds_dir.name)
                break  # Only need datasets from one ordering type
        
        datasets = sorted(set(datasets))
        assert len(datasets) > 0, (
            f"No dataset orderings found in {run_dir}. "
            f"Run 'pipeline.mode=diffing' with 'diffing/method=diff_mining' first before running evaluation."
        )
    
    # Create dataset name mapping for blinding
    dataset_mapping: Dict[str, str] = {}
    for i, ds in enumerate(datasets, start=1):
        dataset_mapping[f"ds{i}"] = ds
    
    out: Dict[str, Any] = {"datasets": {}}
    
    method_cfg = method.method_cfg
    filter_punct = bool(method_cfg.filter_pure_punctuation)
    normalize = bool(method_cfg.normalize_tokens)
    filter_special = bool(method_cfg.filter_special_tokens)
    
    selection_mode = cfg.get("token_set_selection_mode", "top_k_occurring")
    top_k_tokens = int(cfg.get("top_k_tokens", 100))
    
    for i, ds in enumerate(datasets, start=1):
        anonymized_name = f"ds{i}"
        total_positions = 0
        num_samples = 0
        
        top_positive = _load_from_new_schema(run_dir, ds, selection_mode, top_k_tokens)
        assert top_positive is not None, (
            f"No ordering data found for dataset={ds!r} "
            f"(selection_mode={selection_mode!r}) in run_dir={run_dir}"
        )
        num_samples = int(run_meta.get("max_samples", 0))
        logger.info(f"Loaded {len(top_positive)} tokens from new schema for {ds}")
        
        if not top_positive:
            continue
        
        if selection_mode == "fraction_positive_diff":
            note = f"Top {len(top_positive)} tokens by fraction of positive logit diffs."
        else:
            note = f"Top {len(top_positive)} positive occurrence tokens."
        
        out["datasets"][anonymized_name] = {
            "top_positive_tokens": top_positive,
            "total_positions": total_positions,
            "num_samples": num_samples,
            "metadata": {
                "num_tokens_shown": len(top_positive),
                "token_set_selection_mode": selection_mode,
                "filter_pure_punctuation": filter_punct,
                "normalize_tokens": normalize,
                "filter_special_tokens": filter_special,
                "note": note
            }
        }
            
    return out, dataset_mapping


# Steering functionality is not implemented for LogitDiff method
# If needed in the future, implement get_steering_samples and generate_steered here

