"""
Diffing pipeline for orchestrating model comparison methods.
"""

from typing import Dict, Any, List
from omegaconf import DictConfig
from loguru import logger

from .pipeline import Pipeline
from diffing.methods.kl import KLDivergenceDiffingMethod
from diffing.methods.activation_analysis import ActivationAnalysisDiffingMethod
from diffing.methods.crosscoder import CrosscoderDiffingMethod
from diffing.methods.sae_difference import SAEDifferenceMethod
from diffing.methods.diffing_method import DiffingMethod
from diffing.methods.pca import PCAMethod
from diffing.methods.activation_difference_lens import ActDiffLens
from diffing.methods.activation_oracle import ActivationOracleMethod
from diffing.methods.logit_diff_topk_occurring import LogitDiffTopKOccurringMethod
from diffing.methods.amplification.weight_amplification import (
    WeightDifferenceAmplification,
)


def get_method_class(method_name: str) -> DiffingMethod:
    """Get the appropriate method class for a given method name."""
    if method_name == "kl":
        return KLDivergenceDiffingMethod
    elif method_name == "activation_analysis":
        return ActivationAnalysisDiffingMethod
    elif method_name == "crosscoder":
        return CrosscoderDiffingMethod
    elif method_name == "sae_difference":
        return SAEDifferenceMethod
    elif method_name == "pca":
        return PCAMethod
    elif method_name == "activation_difference_lens":
        return ActDiffLens
    elif method_name == "activation_oracle":
        return ActivationOracleMethod
    elif method_name == "logit_diff_topk_occurring":
        return LogitDiffTopKOccurringMethod
    elif method_name == "weight_amplification":
        return WeightDifferenceAmplification
    else:
        raise ValueError(f"Unknown method: {method_name}")


class DiffingPipeline(Pipeline):
    """
    Pipeline for running diffing methods to analyze differences between models.

    This pipeline can orchestrate multiple diffing methods and chain them together.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="DiffingPipeline")

        # Store diffing configuration
        self.diffing_cfg = cfg.diffing

        # Initialize diffing method
        self.diffing_method = get_method_class(self.diffing_cfg.method.name)(self.cfg)

    def validate_config(self) -> None:
        """Validate the diffing pipeline configuration."""
        super().validate_config()

        # Check diffing-specific configuration
        if not hasattr(self.cfg, "diffing"):
            raise ValueError("Configuration must contain 'diffing' section")

        if not hasattr(self.cfg.diffing, "method"):
            raise ValueError("Diffing configuration must contain 'method' section")

        if not hasattr(self.cfg.diffing.method, "name"):
            raise ValueError("Diffing method configuration must contain 'name'")

        # Check model configuration
        if not hasattr(self.cfg, "model"):
            raise ValueError("Configuration must contain 'model' section")

        if not hasattr(self.cfg, "organism"):
            raise ValueError("Configuration must contain 'organism' section")

        self.logger.info("Diffing pipeline configuration validation passed")

    def run(self) -> Dict[str, Any]:
        """
        Run the diffing pipeline.

        Returns:
            Dictionary containing pipeline metadata and status
        """
        self.logger.info(f"Running diffing method: {self.diffing_cfg.method.name}")

        # Run the diffing method (results are saved to disk internally)
        self.diffing_method.run()

        self.logger.info(f"Diffing pipeline completed successfully")
