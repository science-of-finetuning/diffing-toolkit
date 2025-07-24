from typing import Tuple, Optional
import streamlit as st

def create_metric_selection_ui(key_prefix: str = "") -> Tuple[str, Optional[str]]:
    """
    Create a two-layer metric selection UI.
    
    Args:
        key_prefix: Prefix for streamlit keys to avoid conflicts
        
    Returns:
        Tuple of (metric_type, aggregation_method)
        aggregation_method is None for metrics that don't support aggregation
    """    
    # First layer: metric type selection
    metric_options = {
        "norm_diff": "Norm Difference", 
        "cos_dist": "Cosine Distance",
        "norm_base": "Base Model Norm",
        "norm_ft": "Finetuned Model Norm"
    }
    
    metric_type = st.selectbox(
        "Select Metric Type:",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x],
        key=f"{key_prefix}_metric_type"
    )
    
    # Second layer: aggregation selection (only for norm_diff and cos_dist)
    aggregation = None
    if metric_type in ["norm_diff", "cos_dist"]:
        aggregation = st.selectbox(
            "Select Aggregation:",
            options=["max", "mean"],
            format_func=lambda x: x.title(),
            key=f"{key_prefix}_aggregation"
        )
    
    return metric_type, aggregation


def get_maxact_database_name(metric_type: str, aggregation: Optional[str] = None) -> str:
    """
    Map metric type and aggregation to database filename.
    
    Args:
        metric_type: One of norm_diff, cos_dist, norm_base, norm_ft
        aggregation: One of max, mean (only for norm_diff and cos_dist)
        
    Returns:
        Database filename without .db extension
    """
    if metric_type == "norm_diff":
        return "mean_norm_diff" if aggregation == "mean" else "norm_diff"
    elif metric_type == "cos_dist":
        return "mean_cos_dist" if aggregation == "mean" else "cos_dist"
    elif metric_type == "norm_base":
        return "norm_base" 
    elif metric_type == "norm_ft":
        return "norm_finetuned"
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def get_metric_display_name(metric_type: str, aggregation: Optional[str] = None) -> str:
    """Get display name for metric combination."""
    base_names = {
        "norm_diff": "Norm Difference",
        "cos_dist": "Cosine Distance", 
        "norm_base": "Base Model Norm",
        "norm_ft": "Finetuned Model Norm"
    }
    
    base_name = base_names[metric_type]
    if aggregation and metric_type in ["norm_diff", "cos_dist"]:
        return f"{base_name} ({aggregation.title()})"	
    return base_name 
