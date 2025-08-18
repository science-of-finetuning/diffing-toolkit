"""
Preprocessing pipeline for activation collection from base and finetuned models.

This pipeline loads both base and finetuned models and generates activations
for multiple datasets including organism training data and general datasets.
"""

import sys

sys.path.append(".")

from typing import Any, Dict, List, Tuple
from datasets import load_dataset, Dataset
from omegaconf import DictConfig
from pathlib import Path
import torch

from .pipeline import Pipeline
from .activation_collection import collect_activations

from ..utils import (
    get_layer_indices,
    ModelConfig,
    DatasetConfig,
    get_model_configurations,
    get_dataset_configurations,
)
from ..utils.data import load_dataset_from_hub_or_local

class PreprocessingPipeline(Pipeline):
    """
    Preprocessing pipeline that generates activations from base and finetuned models.

    This pipeline:
    1. Loads base model and finetuned model configurations
    2. Loads multiple datasets (organism training, general chat, pretraining)
    3. Generates activations for each model-dataset combination
    4. Stores activations in organized directory structure
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="PreprocessingPipeline")

        # Extract preprocessing configuration
        self.preprocessing_cfg = cfg.preprocessing

        # Set up activation storage directory
        self.activation_store_dir = Path(self.preprocessing_cfg.activation_store_dir)
        self.activation_store_dir.mkdir(parents=True, exist_ok=True)

    def validate_config(self) -> None:
        """Validate preprocessing-specific configuration."""
        super().validate_config()

        # Check preprocessing configuration
        if not hasattr(self.cfg, "preprocessing"):
            raise ValueError("Configuration must contain 'preprocessing' section")

        required_preprocessing_fields = [
            "enabled",
            "activation_store_dir",
            "layers",
            "batch_size",
            "context_len",
        ]
        for field in required_preprocessing_fields:
            if not hasattr(self.preprocessing_cfg, field):
                raise ValueError(f"Preprocessing configuration must contain '{field}'")

        # Check organism configuration
        if not hasattr(self.cfg.finetune, "organism"):
            raise ValueError("Configuration must contain organism specification")

        # Check model configuration
        if not hasattr(self.cfg.finetune, "model"):
            raise ValueError("Configuration must contain base model specification")

        self.logger.info("Preprocessing configuration validation passed")

    def _load_dataset(self, dataset_cfg: DatasetConfig) -> Dataset:
        """Load a dataset based on its configuration."""
        self.logger.info(f"Loading dataset: {dataset_cfg.name} ({dataset_cfg.id})")

        try:
            dataset = load_dataset_from_hub_or_local(dataset_cfg.id, split=dataset_cfg.split)
            self.logger.info(f"Loaded {len(dataset)} samples from {dataset_cfg.name}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_cfg.name}: {str(e)}")
            raise e

    def _collect_activations_for_model_dataset(
        self,
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        dataset: Dataset,
    ) -> None:
        """Collect activations for a specific model-dataset combination."""

        # Apply organism-specific preprocessing overrides
        organism_overrides = self.cfg.organism.get("preprocessing_overrides", {})
        preprocessing_params = {
            "layers": get_layer_indices(
                (
                    model_cfg.model_id
                    if model_cfg.base_model_id is None
                    else model_cfg.base_model_id
                ),
                organism_overrides.get("layers", self.preprocessing_cfg.layers),
            ),
            "max_samples": organism_overrides.get(
                "max_samples_per_dataset",
                self.preprocessing_cfg.max_samples_per_dataset,
            ),
            "max_tokens": organism_overrides.get(
                "max_tokens_per_dataset",
                (
                    self.preprocessing_cfg.max_tokens_per_dataset_train
                    if dataset_cfg.split == "train"
                    else self.preprocessing_cfg.max_tokens_per_dataset_validation
                ),
            ),
            "batch_size": organism_overrides.get(
                "batch_size", self.preprocessing_cfg.batch_size
            ),
            "context_len": organism_overrides.get(
                "context_len", self.preprocessing_cfg.context_len
            ),
            "store_tokens": organism_overrides.get(
                "store_tokens", self.preprocessing_cfg.store_tokens
            ),
            "overwrite": organism_overrides.get(
                "overwrite", self.preprocessing_cfg.overwrite
            ),
            "disable_multiprocessing": organism_overrides.get(
                "disable_multiprocessing",
                self.preprocessing_cfg.disable_multiprocessing,
            ),
        }

        # Set dtype
        dtype_str = organism_overrides.get("dtype", self.preprocessing_cfg.dtype)
        if isinstance(dtype_str, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(dtype_str, torch.bfloat16)
        else:
            dtype = dtype_str

        self.logger.info(
            f"Collecting Activations for {model_cfg.name} + {dataset_cfg.name} - layers {preprocessing_params['layers']}"
        )
        self.logger.info(f"Max tokens: {preprocessing_params['max_tokens']}")

        collect_activations(
            model_cfg=model_cfg,
            dataset=dataset,
            layers=preprocessing_params["layers"],
            activation_store_dir=str(self.activation_store_dir),
            max_samples=preprocessing_params["max_samples"],
            max_tokens=preprocessing_params["max_tokens"],
            batch_size=preprocessing_params["batch_size"],
            context_len=preprocessing_params["context_len"],
            dtype=dtype,
            store_tokens=preprocessing_params["store_tokens"],
            overwrite=preprocessing_params["overwrite"],
            disable_multiprocessing=preprocessing_params["disable_multiprocessing"],
            text_column=dataset_cfg.text_column,
            messages_column=dataset_cfg.messages_column,
            is_chat_data=dataset_cfg.is_chat,
            dataset_split=dataset_cfg.split,
            dataset_name=dataset_cfg.name,
            ignore_first_n_tokens=model_cfg.ignore_first_n_tokens_per_sample_during_collection,
            token_level_replacement=model_cfg.token_level_replacement,
            default_text_column=model_cfg.text_column,
        )
        self.logger.info(
            f"Successfully collected activations: {model_cfg.name} + {dataset_cfg.name}"
        )

    def run(self) -> Dict[str, Any]:
        """
        Main execution method for the preprocessing pipeline.

        Returns:
            Dictionary containing pipeline results and metadata
        """

        # Get model and dataset configurations
        base_model_cfg, finetuned_model_cfg = get_model_configurations(self.cfg)
        assert (
            sum(
                [
                    self.preprocessing_cfg.chat_only,
                    self.preprocessing_cfg.pretraining_only,
                    self.preprocessing_cfg.training_only,
                ]
            )
            <= 1
        ), "Maximum of one of chat_only, pretraining_only, or training_only can be True"
        if self.preprocessing_cfg.chat_only:
            self.logger.info(f"Collecting chat dataset only")
        if self.preprocessing_cfg.pretraining_only:
            self.logger.info(f"Collecting pretraining dataset only")
        if self.preprocessing_cfg.training_only:
            self.logger.info(f"Collecting training dataset only")

        if self.preprocessing_cfg.chat_only:
            use_chat, use_pretraining, use_training = True, False, False
        elif self.preprocessing_cfg.pretraining_only:
            use_chat, use_pretraining, use_training = False, True, False
        elif self.preprocessing_cfg.training_only:
            use_chat, use_pretraining, use_training = False, False, True
        else:
            use_chat, use_pretraining, use_training = True, True, True

        dataset_configs = get_dataset_configurations(
            self.cfg,
            use_chat_dataset=use_chat,
            use_pretraining_dataset=use_pretraining,
            use_training_dataset=use_training,
        )

        self.logger.info(f"Base model: {base_model_cfg.name}")
        self.logger.info(f"Finetuned model: {finetuned_model_cfg.name}")
        self.logger.info(f"Datasets: {[d.name for d in dataset_configs]}")

        results = {
            "base_model": base_model_cfg.name,
            "finetuned_model": finetuned_model_cfg.name,
            "datasets_processed": [],
            "activation_store_dir": str(self.activation_store_dir),
            "preprocessing_config": dict(self.preprocessing_cfg),
        }
        # Process each dataset
        for dataset_cfg in dataset_configs:
            # Load dataset
            dataset = self._load_dataset(dataset_cfg)

            # Collect activations for base model
            self._collect_activations_for_model_dataset(
                base_model_cfg, dataset_cfg, dataset
            )

            # Collect activations for finetuned model
            self._collect_activations_for_model_dataset(
                finetuned_model_cfg, dataset_cfg, dataset
            )

            results["datasets_processed"].append(
                {
                    "name": dataset_cfg.name,
                    "id": dataset_cfg.id,
                    "split": dataset_cfg.split,
                    "samples": len(dataset),
                    "base_model_processed": True,
                    "finetuned_model_processed": True,
                }
            )

        self.logger.info(f"Results: {results}")
        self.logger.info("Preprocessing pipeline completed successfully")
        return results
