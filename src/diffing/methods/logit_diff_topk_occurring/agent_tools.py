from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

def get_overview(method: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build overview for LogitDiff agent.
    
    Includes ALL top-K positive occurring tokens from cached results.
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
            
    out: Dict[str, Any] = {"datasets": {}}
    
    for ds in datasets:
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
            
            out["datasets"][ds] = {
                "top_positive_tokens": top_positive,
                "total_positions": total_positions,
                "num_samples": num_samples,
                "metadata": {
                    "num_tokens_shown": len(top_positive),
                    "note": "All available top-K positive tokens shown. These are tokens the finetuned model prefers over the base model."
                }
            }
        except Exception as e:
            logger.error(f"Error loading results for {ds}: {e}")
            continue
            
    return out


# Steering functionality is not implemented for LogitDiff method
# If needed in the future, implement get_steering_samples and generate_steered here

