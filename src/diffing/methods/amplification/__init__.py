"""
Weight difference amplification method and dashboard.

Core classes:
- AmplificationConfig: Configuration for amplification specifications
- WeightDifferenceAmplification: Main diffing method implementation
- AmplificationDashboard: Streamlit dashboard for interactive use

Tab classes (for dashboard customization):
- AmplificationsTab, ChatTab, MultiGenerationTab, MultiPromptTab
"""

from .amplification_config import AmplificationConfig
from .weight_amplification import WeightDifferenceAmplification


__all__ = [
    "AmplificationConfig",
    "WeightDifferenceAmplification",
]
