from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

# Import token processing functions - use shared function for consistent behavior
from .normalization import process_token_list, load_fraction_positive_tokens


def get_overview(method: Any, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build overview for LogitDiff agent.
    
    Includes ALL top-K positive occurring tokens from cached results.
    Token selection method is controlled by token_set_selection_mode config.
    
    Returns:
        Tuple of (overview_data, dataset_mapping) where dataset_mapping maps
        anonymized names (ds1, ds2, ...) to real dataset names
    """
    logger.info("AgentTool: get_overview")
    
    datasets: List[str] = list(cfg.get("datasets", []))
    
    if len(datasets) == 0:
        # autodiscover datasets from analysis_dir
        analysis_dir = method.get_or_create_results_dir()
        results_files = list(analysis_dir.glob("*_occurrence_rates.json"))
        datasets = [f.stem.replace("_occurrence_rates", "") for f in results_files]
        assert len(datasets) > 0, (
            f"No diffing results found in {analysis_dir}. "
            f"Run 'pipeline.mode=diffing' with 'diffing/method=logit_diff_topk_occurring' first before running evaluation."
        )
    
    # Create dataset name mapping for blinding
    dataset_mapping: Dict[str, str] = {}
    for i, ds in enumerate(datasets, start=1):
        dataset_mapping[f"ds{i}"] = ds
    
    out: Dict[str, Any] = {"datasets": {}}
    
    # Check token processing settings (from main method config, not agent overview config)
    method_cfg = method.method_cfg
    filter_punct = bool(method_cfg.filter_pure_punctuation)
    normalize = bool(method_cfg.normalize_tokens)
    filter_special = bool(method_cfg.filter_special_tokens)
    
    # Check token set selection mode
    selection_mode = cfg.get("token_set_selection_mode", "top_k_occurring")
    
    # Get analysis directory
    analysis_dir = method.get_or_create_results_dir()
    
    for i, ds in enumerate(datasets, start=1):
        anonymized_name = f"ds{i}"
        results_file = analysis_dir / f"{ds}_occurrence_rates.json"
        
        if not results_file.exists():
            continue
            
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            total_positions = results.get("total_positions", 0)
            num_samples = results.get("num_samples", 0)
            
            # Get token list based on selection mode
            if selection_mode == "fraction_positive_diff":
                # Use tokens sorted by fraction of positive logit diffs
                global_stats_file = analysis_dir / f"{ds}_global_token_stats.json"
                if not global_stats_file.exists():
                    logger.warning(
                        f"Global token stats not found for {ds}, falling back to top_k_occurring mode"
                    )
                    top_positive = results.get("top_positive", [])
                else:
                    top_k_tokens = cfg.get("top_k_tokens", 100)
                    top_positive = load_fraction_positive_tokens(
                        global_stats_file,
                        k=int(top_k_tokens),
                        filter_punctuation=filter_punct,
                        normalize=normalize,
                        filter_special_tokens=filter_special,
                        tokenizer=method.tokenizer
                    )
                    logger.info(f"Using fraction_positive_diff mode for {ds} ({len(top_positive)} tokens)")
            else:
                # Default: top_k_occurring - use occurrence rates from results
                top_positive = results.get("top_positive", [])
                
                # Apply token processing (filtering and/or normalization)
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
                    logger.info(f"Applied token processing for {ds}: {original_len} -> {len(top_positive)} tokens (filter_punct={filter_punct}, normalize={normalize}, filter_special={filter_special})")
            
            # Limit to top_k_tokens if configured
            original_count = len(top_positive)
            top_k_tokens = cfg.get("top_k_tokens", None)
            if top_k_tokens is not None:
                top_k_tokens = int(top_k_tokens)
                top_positive = top_positive[:top_k_tokens]
                logger.info(f"Limited tokens for agent overview: {original_count} -> {len(top_positive)} tokens (top_k_tokens={top_k_tokens})")
            
            # Build note based on selection mode
            if selection_mode == "fraction_positive_diff":
                note = f"Top {len(top_positive)} tokens by fraction of positive logit diffs. These are tokens that most consistently have higher logits in the finetuned model."
            else:
                note = f"Top {len(top_positive)} positive occurrence tokens (out of {original_count} available). These are tokens the finetuned model prefers over the base model."
            
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
        except Exception as e:
            logger.error(f"Error loading results for {ds}: {e}")
            continue
            
    return out, dataset_mapping


# Steering functionality is not implemented for LogitDiff method
# If needed in the future, implement get_steering_samples and generate_steered here

