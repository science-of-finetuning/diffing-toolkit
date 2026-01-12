"""
Abstract pipeline base class for the diffing game framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from loguru import logger
import torch
from pathlib import Path


class Pipeline(ABC):
    """
    Abstract base class for all pipelines in the diffing game framework.

    This class provides a common interface and shared functionality for all pipeline types,
    including preprocessing, finetuning, and diffing pipelines.
    """

    def __init__(self, cfg: DictConfig, name: Optional[str] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            cfg: Configuration object containing all pipeline settings
            name: Optional name for the pipeline (defaults to class name)
        """
        self.cfg = cfg
        self.name = name or self.__class__.__name__
        self.output_dir = Path(cfg.pipeline.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logger.bind(pipeline=self.name)

        # Set random seed for reproducibility
        if hasattr(cfg, "seed"):
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
            self.logger.info(f"Set random seed to {cfg.seed}")

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Main execution method for the pipeline.

        This method should be implemented by all concrete pipeline classes
        to define their specific execution logic.

        Returns:
            Dictionary containing pipeline results and metadata
        """
        pass

    def validate_config(self) -> None:
        """
        Validate the configuration for this pipeline.

        This method can be overridden by concrete implementations
        to add pipeline-specific validation logic.
        """
        # Basic validation
        if not hasattr(self.cfg, "pipeline"):
            raise ValueError("Configuration must contain 'pipeline' section")

        if not hasattr(self.cfg.pipeline, "output_dir"):
            raise ValueError("Pipeline configuration must contain 'output_dir'")

        self.logger.info("Configuration validation passed")

    def setup_environment(self) -> None:
        """
        Set up the environment for pipeline execution.

        This includes creating output directories, setting up logging,
        and any other environment preparation.
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Pipeline output directory: {self.output_dir}")

        # Set torch precision if configured
        if hasattr(self.cfg, "torch_precision"):
            torch.set_float32_matmul_precision(self.cfg.torch_precision)
            self.logger.info(f"Set torch precision to {self.cfg.torch_precision}")

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def cleanup(self) -> None:
        """
        Cleanup method called after pipeline execution.

        This method can be overridden by concrete implementations
        to add pipeline-specific cleanup logic.
        """
        self.logger.info(f"Pipeline {self.name} cleanup completed")

    def execute(self) -> Dict[str, Any]:
        """
        Full pipeline execution with setup, validation, and cleanup.

        This method provides a standardized execution flow:
        1. Validate configuration
        2. Set up environment
        3. Run pipeline-specific logic
        4. Cleanup

        Returns:
            Dictionary containing pipeline results and metadata
        """
        try:
            self.logger.info(f"Starting pipeline: {self.name}")

            # Validate configuration
            self.validate_config()

            # Set up environment
            self.setup_environment()

            # Run pipeline-specific logic
            results = self.run()

            self.logger.info(f"Pipeline {self.name} completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline {self.name} failed: {str(e)}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e

        finally:
            # Always run cleanup
            self.cleanup()
