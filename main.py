#!/usr/bin/env python3
"""
This script serves as the Hydra-enabled entry point for running
finetuning and diffing experiments.
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from src.pipeline.diffing_pipeline import DiffingPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def hydra_loguru_init() -> None:
    from hydra.core.hydra_config import HydraConfig

    hydra_path = HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "main.log"))


def setup_environment(cfg: DictConfig) -> None:
    """Set up the experiment environment."""
    # Create output directories
    output_dir = Path(cfg.pipeline.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    checkpoint_dir = Path(cfg.infrastructure.storage.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    logs_dir = Path(cfg.infrastructure.storage.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs directory: {logs_dir}")

    # Set random seed for reproducibility
    import random
    import numpy as np
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger.info(f"Environment set up. Output directory: {output_dir}")
    logger.info(f"Random seed: {cfg.seed}")


def run_preprocessing_pipeline(cfg: DictConfig) -> None:
    """Run the preprocessing pipeline to collect activations."""
    logger.info("Starting preprocessing pipeline...")

    from src.pipeline.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline(cfg)
    pipeline.run()

    logger.info("Preprocessing pipeline completed")


def run_diffing_pipeline(cfg: DictConfig) -> None:
    """Run the diffing analysis pipeline."""
    logger.info("Starting diffing pipeline...")

    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg.diffing.method)}")

    pipeline = DiffingPipeline(cfg)
    pipeline.execute()

    logger.info("Diffing pipeline completed successfully")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that orchestrates the entire pipeline."""
    hydra_loguru_init()
    logger.info("Starting Diffing Toolkit pipeline")
    logger.info(f"Pipeline mode: {cfg.pipeline.mode}")

    if cfg.debug:
        logger.debug("Debug mode enabled")
        logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set up environment
    setup_environment(cfg)

    # Run pipeline based on mode
    if cfg.pipeline.mode == "full" or cfg.pipeline.mode == "preprocessing":
        run_preprocessing_pipeline(cfg)

    if cfg.pipeline.mode == "full" or cfg.pipeline.mode == "diffing":
        run_diffing_pipeline(cfg)

    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
