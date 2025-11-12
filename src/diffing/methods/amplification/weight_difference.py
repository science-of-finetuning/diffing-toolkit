from abc import ABC, abstractmethod
from typing import Dict, List
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path
from src.diffing.methods.diffing_method import DiffingMethod
from src.utils.agents.blackbox_agent import BlackboxAgent
from src.utils.agents.diffing_method_agent import DiffingMethodAgent
from collections import defaultdict
from src.utils.configs import CONFIGS_DIR


@dataclass
class WeightDifferenceAmplificationConfig:
    """
    Configuration for weight difference amplification.
    """

    default_amplification_factor: float
    amplification_factors: Dict[str, float]


class WeightDifferenceAmplification(DiffingMethod):
    """
    Amplify the weights difference between the base and finetuned models according to an amplifcation configuration
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def run(self):
        raise NotImplementedError("No need to run this method")

    def visualize(self):
        """Launch the amplification dashboard."""
        from src.diffing.methods.amplification.amplification_dashboard import (
            AmplificationDashboard,
        )

        dashboard = AmplificationDashboard(self)
        dashboard.display()

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available results for this method.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = defaultdict(dict)

        all_model_names = [model_cfg.stem for model_cfg in (CONFIGS_DIR / "model").glob("*.yaml")]
        all_organism_names = [organism_cfg.stem for organism_cfg in (CONFIGS_DIR / "organism").glob("*.yaml")]
        for model_name in all_model_names:
            for organism_name in all_organism_names:
                results[model_name][organism_name] = "."

        return results

    def get_agent(self) -> DiffingMethodAgent:
        """Get the agent for the method."""
        raise NotImplementedError
