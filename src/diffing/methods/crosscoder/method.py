"""
Crosscoder-based model diffing method.

This module trains crosscoders on paired activations from base and finetuned models,
then runs a comprehensive analysis pipeline including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.

Key assumptions:
- Preprocessing pipeline has generated paired activation caches
- dictionary_learning library is available and compatible
- science-of-finetuning repository is available for analysis pipeline
- W&B configuration is available in infrastructure config
- Sufficient GPU memory and disk space for training
"""

from typing import Dict, Any
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import json
from collections import defaultdict
import numpy as np
import base64
import pandas as pd
import streamlit as st

from ..diffing_method import DiffingMethod
from diffing.utils.activations import get_layer_indices
from diffing.utils.dictionary.analysis import (
    build_push_crosscoder_latent_df,
    make_plots,
)
from diffing.utils.dictionary.training import train_crosscoder_for_layer
from diffing.utils.dictionary.latent_scaling.closed_form import (
    compute_scalers_from_config,
)
from diffing.utils.dictionary.latent_scaling.beta_analysis import (
    update_latent_df_with_beta_values,
)
from diffing.utils.dictionary.latent_activations import (
    collect_dictionary_activations_from_config,
    collect_activating_examples,
    update_latent_df_with_stats,
)
from diffing.utils.dictionary.utils import load_dictionary_model
from diffing.utils.dictionary.training import crosscoder_run_name
from diffing.utils.visualization import multi_tab_interface
from diffing.utils.max_act_store import ReadOnlyMaxActStore
from diffing.utils.dictionary.steering import (
    run_latent_steering_experiment,
    get_crosscoder_latent,
    display_steering_results,
)


class CrosscoderDiffingMethod(DiffingMethod):
    """
    Trains crosscoders on paired activations and runs comprehensive analysis.

    This method:
    1. Loads paired activation caches from preprocessing pipeline
    2. Trains crosscoders for specified layers using local shuffling
    3. Saves trained models with configuration and metrics
    4. Optionally uploads models to Hugging Face Hub
    5. Runs complete analysis pipeline from science-of-finetuning
    6. Returns comprehensive results including training metrics and analysis outcomes
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Get layers to process
        layers = self.method_cfg.layers
        if layers is None:
            layers = cfg.preprocessing.layers
        self.layers = get_layer_indices(self.base_model_cfg.model_id, layers)

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize latent df cache
        self.latent_df_cache = {}

    def run(self) -> Dict[str, Any]:
        """
        Main training orchestration with analysis pipeline.

        Trains crosscoders for each specified layer, then runs the complete
        analysis pipeline for each trained model.

        Returns:
            Dictionary containing training results, model paths, and analysis outcomes

        Assumptions:
            - Paired activation caches exist for all specified layers
            - Sufficient resources for training and analysis
        """
        logger.info(f"Starting crosscoder training for layers: {self.layers}")

        for layer_idx in self.layers:
            logger.info(f"Processing layer {layer_idx}")

            logger.info(f"Training crosscoder for layer {layer_idx}")

            dictionary_name = crosscoder_run_name(
                self.cfg, layer_idx, self.base_model_cfg, self.finetuned_model_cfg
            )
            model_results_dir = (
                self.results_dir / "crosscoder" / f"layer_{layer_idx}" / dictionary_name
            )
            logger.info(f"Model results directory: {model_results_dir}")
            model_results_dir.mkdir(parents=True, exist_ok=True)
            if (
                not (
                    model_results_dir / "dictionary_model" / "model.safetensors"
                ).exists()
                or self.method_cfg.training.overwrite
            ):
                # Train crosscoder for this layer
                training_metrics, model_path = train_crosscoder_for_layer(
                    self.cfg, layer_idx, self.device, dictionary_name
                )
                # save training metrics
                with open(model_results_dir / "training_metrics.json", "w") as f:
                    json.dump(training_metrics, f)
                # save model
                dictionary_model = load_dictionary_model(model_path)
                dictionary_model.save_pretrained(model_results_dir / "dictionary_model")

                # save training configs
                OmegaConf.save(self.cfg, model_results_dir / "training_config.yaml")
            else:
                logger.info(
                    f"Found trained model at {model_results_dir / 'dictionary_model'}"
                )
                training_metrics = json.load(
                    open(model_results_dir / "training_metrics.json")
                )

            if self.method_cfg.analysis.enabled:
                logger.info(f"Storing analysis results in {model_results_dir}")
                build_push_crosscoder_latent_df(
                    dictionary_name=dictionary_name,
                    base_layer=0,
                    ft_layer=1,
                )

                if self.method_cfg.analysis.latent_scaling.enabled:
                    compute_scalers_from_config(
                        cfg=self.cfg,
                        layer=layer_idx,
                        dictionary_model=dictionary_name,
                        results_dir=model_results_dir,
                    )
                    update_latent_df_with_beta_values(
                        dictionary_name,
                        model_results_dir,
                        num_samples=self.method_cfg.analysis.latent_scaling.num_samples,
                    )

                if self.method_cfg.analysis.latent_activations.enabled:
                    latent_activations_cache = (
                        collect_dictionary_activations_from_config(
                            cfg=self.cfg,
                            layer=layer_idx,
                            dictionary_model_name=dictionary_name,
                            result_dir=model_results_dir,
                        )
                    )
                    collect_activating_examples(
                        dictionary_model_name=dictionary_name,
                        latent_activation_cache=latent_activations_cache,
                        n=self.method_cfg.analysis.latent_activations.n_max_activations,
                        upload_to_hub=self.method_cfg.analysis.latent_activations.upload_to_hub,
                        overwrite=self.method_cfg.analysis.latent_activations.overwrite,
                        save_path=model_results_dir,
                    )
                    update_latent_df_with_stats(
                        dictionary_name=dictionary_name,
                        latent_activation_cache=latent_activations_cache,
                        split_of_cache=self.method_cfg.analysis.latent_activations.split,
                        device=self.method_cfg.analysis.latent_activations.cache_device,
                        save_path=model_results_dir,
                    )

                try:
                    make_plots(
                        dictionary_name=dictionary_name,
                        plots_dir=model_results_dir / "plots",
                    )
                except Exception as e:
                    logger.error(f"Error making plots for {dictionary_name}: {e}")

                if self.method_cfg.analysis.latent_steering.enabled:
                    logger.info(
                        f"Running latent steering experiment for layer {layer_idx}"
                    )
                    run_latent_steering_experiment(
                        method=self,
                        get_latent_fn=get_crosscoder_latent,
                        dictionary_name=dictionary_name,
                        results_dir=model_results_dir,
                        layer=layer_idx,
                    )

            logger.info(f"Successfully completed layer {layer_idx}")

    def visualize(self) -> None:
        """Create Streamlit visualization for CrossCoder results."""
        from .dashboard import visualize

        visualize(self)

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available crosscoder results.

        Args:
            results_dir: Base results directory

        Returns:
            Dict mapping {model_pair: {organism: {layer: path_to_results}}}
        """
        results = defaultdict(dict)
        results_base = results_dir

        if not results_base.exists():
            return results

        # Scan for KL results in the expected structure
        for base_model_dir in results_base.iterdir():
            if not base_model_dir.is_dir():
                continue

            model_name = base_model_dir.name

            for organism_dir in base_model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                sae_dir = organism_dir / "crosscoder"
                if sae_dir.exists() and any(sae_dir.iterdir()):
                    results[model_name][organism_name] = str(sae_dir)

        return results

    @torch.no_grad()
    def compute_crosscoder_activations_for_tokens(
        self,
        dictionary_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
    ):
        """Compute crosscoder latent activations for a batch of tokens."""

        assert (
            input_ids.shape == attention_mask.shape and input_ids.ndim == 2
        ), "input_ids and attention_mask must be [B, T]"

        token_ids = input_ids[0].cpu().tolist()
        tokens = [self.tokenizer.decode([t]) for t in token_ids]
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

        with self.base_model.trace(inputs):
            base_act = self.base_model.layers_output[layer].save()
        with self.finetuned_model.trace(inputs):
            ft_act = self.finetuned_model.layers_output[layer].save()

        base_act, ft_act = base_act.cpu(), ft_act.cpu()
        B, T, H = base_act.shape
        assert ft_act.shape == (B, T, H)

        # Use first sequence
        seq_base = base_act[0]
        seq_ft = ft_act[0]

        # Stack along new layer dimension -> [T, 2, H]
        stacked_seq = torch.stack([seq_base, seq_ft], dim=1)

        # Load crosscoder
        cc_model = load_dictionary_model(dictionary_name, is_sae=False)

        # Encode -> [T, dict_size]
        latent = cc_model.encode(stacked_seq)
        latent_np = latent.cpu().numpy()
        max_per_tok = latent_np.max(axis=1)
        stats = {
            "mean": float(max_per_tok.mean()),
            "std": float(max_per_tok.std()),
            "min": float(max_per_tok.min()),
            "max": float(max_per_tok.max()),
            "median": float(np.median(max_per_tok)),
        }
        return {
            "tokens": tokens,
            "latent_activations": latent_np,
            "max_activations_per_token": max_per_tok,
            "statistics": stats,
            "total_tokens": len(tokens),
            "layer": layer,
            "dict_size": cc_model.dict_size,
        }
