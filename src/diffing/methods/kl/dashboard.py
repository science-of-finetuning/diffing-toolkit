"""
Dashboard and visualization code for KL Divergence diffing method.

Separated to avoid triggering streamlit cache warnings at import time.
"""

from typing import Dict, Any
import torch
import numpy as np
import streamlit as st

from diffing.utils.dashboards import (
    AbstractOnlineDiffingDashboard,
    MaxActivationDashboardComponent,
)
from diffing.utils.max_act_store import ReadOnlyMaxActStore
from diffing.utils.visualization import multi_tab_interface


def visualize(method) -> None:
    """
    Create Streamlit visualization for KL divergence results with tabs.

    Args:
        method: KLDivergenceDiffingMethod instance
    """
    multi_tab_interface(
        [
            ("📊 MaxAct Examples", lambda: _render_dataset_statistics(method)),
            ("🔥 Interactive", lambda: KLDivergenceOnlineDashboard(method).display()),
        ],
        "KL Divergence Analysis",
    )


def _render_dataset_statistics(method):
    """Render the dataset statistics tab using MaxActivationDashboardComponent."""
    metric_choice = st.selectbox(
        "Select KL Metric:", ["Per-Token KL", "Mean Per Sample KL"], index=0
    )

    if metric_choice == "Per-Token KL":
        max_act_store = ReadOnlyMaxActStore(
            method.results_dir / "examples_per_token.db",
            tokenizer=method.tokenizer,
        )
        title = "Max Per-Token KL Divergence Examples"
    else:
        max_act_store = ReadOnlyMaxActStore(
            method.results_dir / "examples_mean_per_sample.db",
            tokenizer=method.tokenizer,
        )
        title = "Mean Per-Sample KL Divergence Examples"

    component = MaxActivationDashboardComponent(max_act_store, title=title)
    component.display()


class KLDivergenceOnlineDashboard(AbstractOnlineDiffingDashboard):
    """Online dashboard for interactive KL divergence analysis."""

    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render KL-specific controls in Streamlit (none needed)."""
        return {}

    def compute_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        """Compute KL divergence statistics using the parent method's computation function."""
        results = self.method.compute_kl_for_tokens(input_ids, attention_mask)

        return {
            "tokens": results["tokens"],
            "values": results["kl_values"],
            "statistics": results["statistics"],
            "total_tokens": results["total_tokens"],
        }

    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get KL-specific parameters (none needed)."""
        return {}

    def _get_title(self) -> str:
        """Get title for KL analysis."""
        return "KL Divergence Analysis"
