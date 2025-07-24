"""
Online dashboard for interactive activation analysis.
"""

from typing import Dict, Any
import torch

from src.utils.dashboards import AbstractOnlineDiffingDashboard
from .utils import create_metric_selection_ui, get_metric_display_name


class ActivationAnalysisOnlineDashboard(AbstractOnlineDiffingDashboard):
    """
    Online dashboard for interactive activation analysis with metric selection.
    """
    
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render ActivationAnalysis-specific controls in Streamlit."""
        import streamlit as st
        
        layer = st.selectbox(
            "Select Layer:",
            options=self.method.layers,
            help="Choose which layer to analyze",
            key="online_layer_selector"
        )
        return {"layer": layer}
    
    def compute_statistics_for_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Compute all activation statistics and show metric selection interface."""
        import streamlit as st
        
        layer = kwargs.get("layer", self.method.layers[0])
        
        # Show computation progress
        with st.spinner("Computing activation statistics..."):
            all_results = self.method.compute_all_activation_statistics_for_tokens(input_ids, attention_mask, layer)
        
        # Show metric selection UI after computation
        st.success("✅ Computation complete! Select which metric to display:")
        
        # Use fragment for instant metric switching
        return self._render_metric_selection_fragment(all_results)

    def _render_metric_selection_fragment(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fragment for selecting and displaying metrics without recomputation."""
        import streamlit as st
        
        # Create metric selection UI
        metric_type, aggregation = create_metric_selection_ui("online_dashboard")
        
        # Get the selected metric data
        metric_data = all_results['metrics'][metric_type]
        
        # For norm_diff and cos_dist, use aggregation selection
        if metric_type in ["norm_diff", "cos_dist"] and aggregation:
            if aggregation == "mean":
                values = metric_data['mean_values']
                display_name = get_metric_display_name(metric_type, aggregation)
            else:  # max
                values = metric_data['max_values'] 
                display_name = get_metric_display_name(metric_type, aggregation)
        else:
            values = metric_data['values']
            display_name = get_metric_display_name(metric_type)
        
        # Show selected metric info
        st.info(f"Displaying: **{display_name}**")
        
        # Adapt the results format for the abstract dashboard
        return {
            'tokens': all_results['tokens'],
            'values': values,
            'statistics': metric_data['statistics'],
            'total_tokens': all_results['total_tokens'],
            'metric_type': metric_type,
            'aggregation': aggregation,
            'display_name': display_name
        }
    


    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get activation analysis specific parameters."""
        # Return empty dict since we handle parameters differently now
        return {}
    
    def _get_color_rgb(self) -> tuple:
        """Get color for highlighting based on current metric."""
        # You could make this dynamic based on metric type
        return (255, 0, 0)  # Red for now
    
    def _get_title(self) -> str:
        """Get title for activation analysis."""
        return "Activation Analysis Dashboard" 