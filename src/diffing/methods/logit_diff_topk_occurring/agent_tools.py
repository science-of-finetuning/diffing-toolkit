from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

from .normalization import process_token_list, load_fraction_positive_tokens
from .token_ordering import (
    read_ordering_type_metadata,
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
    # Map selection mode to ordering type
    ordering_type_id = {
        "top_k_occurring": "topk_occurring",
        "fraction_positive_diff": "fraction_positive_diff",
    }.get(selection_mode, "topk_occurring")
    
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
    
    # Convert to legacy format
    tokens = []
    for t in ordering_data.get("tokens", [])[:top_k_tokens]:
        tokens.append({
            "token_id": t["token_id"],
            "token_str": t["token_str"],
            "count_positive": t.get("count_positive", 0),
            "count_negative": t.get("count_negative", 0),
            "positive_occurrence_rate": t.get("ordering_value", 0) if selection_mode == "top_k_occurring" else 0,
            "negative_occurrence_rate": 0,
            "avg_logit_diff": t.get("avg_logit_diff", 0),
        })
    
    return tokens


def get_overview(method: Any, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build overview for LogitDiff agent.
    
    Supports both new schema (run_*) and legacy schema (analysis_*).
    Token selection method is controlled by token_set_selection_mode config.
    
    Returns:
        Tuple of (overview_data, dataset_mapping) where dataset_mapping maps
        anonymized names (ds1, ds2, ...) to real dataset names
    """
    logger.info("AgentTool: get_overview")
    
    datasets: List[str] = list(cfg.get("datasets", []))
    
    # Try new schema first
    run_dir = _find_run_dir(method)
    analysis_dir = method.get_or_create_results_dir()
    
    if len(datasets) == 0:
        # Autodiscover datasets
        if run_dir:
            # From new schema: look for dataset dirs in ordering type dirs
            for ordering_dir in run_dir.iterdir():
                if ordering_dir.is_dir() and (ordering_dir / "metadata.json").exists():
                    for ds_dir in ordering_dir.iterdir():
                        if ds_dir.is_dir() and (ds_dir / "orderings.json").exists():
                            datasets.append(ds_dir.name)
                    break  # Only need datasets from one ordering type
        
        if not datasets:
            # Fall back to legacy schema
            results_files = list(analysis_dir.glob("*_occurrence_rates.json"))
            datasets = [f.stem.replace("_occurrence_rates", "") for f in results_files]
        
        datasets = sorted(set(datasets))
        assert len(datasets) > 0, (
            f"No diffing results found in {analysis_dir}. "
            f"Run 'pipeline.mode=diffing' with 'diffing/method=logit_diff_topk_occurring' first before running evaluation."
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
        top_positive = None
        total_positions = 0
        num_samples = 0
        
        # Try new schema first
        if run_dir:
            top_positive = _load_from_new_schema(run_dir, ds, selection_mode, top_k_tokens)
            if top_positive:
                # Read metadata from run_metadata.json
                metadata_path = run_dir / "run_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        run_meta = json.load(f)
                    num_samples = run_meta.get("max_samples", 0)
                logger.info(f"Loaded {len(top_positive)} tokens from new schema for {ds}")
        
        # Fall back to legacy schema
        if top_positive is None:
            results_file = analysis_dir / f"{ds}_occurrence_rates.json"
            if not results_file.exists():
                continue
            
            with open(results_file, "r") as f:
                results = json.load(f)
            
            total_positions = results.get("total_positions", 0)
            num_samples = results.get("num_samples", 0)
            
            if selection_mode == "fraction_positive_diff":
                global_stats_file = analysis_dir / f"{ds}_global_token_stats.json"
                if not global_stats_file.exists():
                    logger.warning(
                        f"Global token stats not found for {ds}, falling back to top_k_occurring mode"
                    )
                    top_positive = results.get("top_positive", [])
                else:
                    top_positive = load_fraction_positive_tokens(
                        global_stats_file,
                        k=top_k_tokens,
                        filter_punctuation=filter_punct,
                        normalize=normalize,
                        filter_special_tokens=filter_special,
                        tokenizer=method.tokenizer
                    )
                    logger.info(f"Using fraction_positive_diff mode for {ds} ({len(top_positive)} tokens)")
            else:
                top_positive = results.get("top_positive", [])
                
                if filter_punct or normalize or filter_special:
                    original_len = len(top_positive)
                    top_positive = process_token_list(
                        top_positive, 
                        total_positions,
                        filter_punctuation=filter_punct,
                        normalize=normalize,
                        filter_special_tokens=filter_special,
                        tokenizer=method.tokenizer
                    )
                    logger.info(f"Applied token processing for {ds}: {original_len} -> {len(top_positive)} tokens")
            
            top_positive = top_positive[:top_k_tokens]
        
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

