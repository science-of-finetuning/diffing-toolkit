from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

# Import normalization function
from .normalization import normalize_token_list


def get_overview(method: Any, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build overview for LogitDiff agent.
    
    Includes ALL top-K positive occurring tokens from cached results.
    
    Returns:
        Tuple of (overview_data, dataset_mapping) where dataset_mapping maps
        anonymized names (ds1, ds2, ...) to real dataset names
    """
    logger.info("AgentTool: get_overview")
    
    datasets: List[str] = list(cfg.get("datasets", []))
    
    if len(datasets) == 0:
        # autodiscover datasets from results_dir
        results_files = list(method.results_dir.glob("*_occurrence_rates.json"))
        datasets = [f.stem.replace("_occurrence_rates", "") for f in results_files]
        assert len(datasets) > 0, (
            f"No diffing results found in {method.results_dir}. "
            f"Run 'pipeline.mode=diffing' with 'diffing/method=logit_diff_topk_occurring' first before running evaluation."
        )
    
    # Create dataset name mapping for blinding
    dataset_mapping: Dict[str, str] = {}
    for i, ds in enumerate(datasets, start=1):
        dataset_mapping[f"ds{i}"] = ds
    
    out: Dict[str, Any] = {"datasets": {}}
    
    # Check if normalization is enabled
    use_normalized = cfg.get("use_normalized_tokens", False)
    
    for i, ds in enumerate(datasets, start=1):
        anonymized_name = f"ds{i}"
        results_file = method.results_dir / f"{ds}_occurrence_rates.json"
        
        if not results_file.exists():
            continue
            
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
                
            # LogitDiff is simple: we give ALL top positive tokens
            # No drill-down needed because the total number of top tokens is small (~50-150)
            # Note: We intentionally only show positive tokens (what the model prefers).
            # If you want to include the K tokens which consistently had the most negative logit diffs
            # (what the model suppresses), you can access results['top_negative'] here.
            top_positive = results.get("top_positive", [])
            total_positions = results.get("total_positions", 0)
            num_samples = results.get("num_samples", 0)
            
            # Apply normalization if enabled
            if use_normalized:
                top_positive = normalize_token_list(top_positive, total_positions)
                logger.info(f"Applied token normalization for {ds}: {len(results.get('top_positive', []))} -> {len(top_positive)} tokens")
            
            out["datasets"][anonymized_name] = {
                "top_positive_tokens": top_positive,
                "total_positions": total_positions,
                "num_samples": num_samples,
                "metadata": {
                    "num_tokens_shown": len(top_positive),
                    "normalized": use_normalized,
                    "note": "All available top-K positive tokens shown. These are tokens the finetuned model prefers over the base model."
                }
            }
        except Exception as e:
            logger.error(f"Error loading results for {ds}: {e}")
            continue
            
    return out, dataset_mapping


# Steering functionality is not implemented for LogitDiff method
# If needed in the future, implement get_steering_samples and generate_steered here

