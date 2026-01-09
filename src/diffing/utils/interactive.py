"""
Utilities for interactive analysis of model activations (not via dashboard or main.py)
"""

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from diffing.utils.configs import get_model_configurations, get_dataset_configurations
from diffing.utils.model import load_model_from_config
from diffing.utils.activations import (
    get_layer_indices,
    load_activation_datasets_from_config,
)


def load_hydra_config(config_path: str, *overrides) -> DictConfig:
    """
    Load a Hydra config from a file.

    Args:
        config_path: Path to the config file
        *overrides: Hydra override strings (e.g., "model=qwen25_7B_Instruct", "organism=persona_sarcasm")

    Returns:
        Fully resolved configuration
    """
    config_path = (
        Path("../..") / config_path
    )  # as we're in diffing.utils, we need to go up two levels to get to the root
    with initialize(config_path=str(config_path.parent), version_base=None):
        cfg = compose(config_name=Path(config_path).stem, overrides=overrides)

    return cfg


def load_model_and_datasets(
    model_name, organism_name, config_path="configs/config.yaml", split="train"
):
    """
    Load models and activation datasets for interactive analysis.

    This is a convenience function that loads both base and finetuned models
    along with their activation datasets for a given model/organism combination.

    Args:
        model_name: Name of the model configuration (e.g., "qwen25_7B_Instruct")
        organism_name: Name of the organism configuration (e.g., "persona_sarcasm")
        config_path: Path to the Hydra config file

    Returns:
        Tuple containing:
        - base_model: The base model
        - ft_model: The finetuned model
        - caches: Dictionary of activation caches by dataset name and layer
    """
    cfg = load_hydra_config(
        config_path, f"model={model_name}", f"organism={organism_name}"
    )
    base_model_cfg, ft_model_cfg = get_model_configurations(cfg)
    base_model = load_model_from_config(base_model_cfg)
    ft_model = load_model_from_config(ft_model_cfg)

    layers = get_layer_indices(base_model, cfg.preprocessing.layers)
    ds_cfgs = get_dataset_configurations(cfg)
    caches = load_activation_datasets_from_config(
        cfg, ds_cfgs, base_model_cfg, ft_model_cfg, layers=layers, split=split
    )

    return base_model, ft_model, caches
