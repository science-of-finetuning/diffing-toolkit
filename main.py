#!/usr/bin/env python3
"""
This script serves as the Hydra-enabled entry point for running
finetuning and diffing experiments.
"""

import os
from pathlib import Path

# Set CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from diffing.pipeline.diffing_pipeline import DiffingPipeline, get_method_class
from diffing.pipeline.evaluation_pipeline import EvaluationPipeline
from diffing.utils.configs import CONFIGS_DIR

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

    from diffing.pipeline.preprocessing import PreprocessingPipeline

    if not cfg.diffing.method.requires_preprocessing:
        logger.info(
            "Skipping preprocessing pipeline because method does not require preprocessing"
        )
        return

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


def run_evaluation_pipeline(cfg: DictConfig) -> None:
    """Run the evaluation pipeline."""
    logger.info("Starting evaluation pipeline...")

    pipeline = EvaluationPipeline(cfg)
    pipeline.run()

    logger.info("Evaluation pipeline completed successfully")


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
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

    # Validate pipeline mode
    valid_modes = ["full", "preprocessing", "diffing", "evaluation"]
    if cfg.pipeline.mode not in valid_modes:
        raise ValueError(
            f"Invalid pipeline mode: {cfg.pipeline.mode}. "
            f"Must be one of: {valid_modes}"
        )

    # Run pipeline based on mode
    # Special case: in_memory mode for logit_diff_topk_occurring with mode=full
    # Shares a single method instance between preprocess() and run() to keep tensors in RAM
    in_memory = False
    if cfg.diffing.method.name == "logit_diff_topk_occurring":
        in_memory = getattr(cfg.diffing.method.method_params, "in_memory", False)
    if (
        cfg.pipeline.mode == "full"
        and in_memory
        and cfg.diffing.method.name == "logit_diff_topk_occurring"
    ):
        logger.info(
            "Running in-memory mode: preprocessing and diffing will share tensors in RAM"
        )
        method = get_method_class(cfg.diffing.method.name)(cfg)
        method.preprocess()
        method.run()
    else:
        # Standard disk-based flow
        if cfg.pipeline.mode == "full" or cfg.pipeline.mode == "preprocessing":
            run_preprocessing_pipeline(cfg)

        if cfg.pipeline.mode == "full" or cfg.pipeline.mode == "diffing":
            run_diffing_pipeline(cfg)

    if cfg.pipeline.mode == "full" or cfg.pipeline.mode == "evaluation":
        run_evaluation_pipeline(cfg)

    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
