"""
SAE on difference-based model diffing method.

This module trains SAEs on activation differences between base and finetuned models,
then runs a comprehensive analysis pipeline including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.

Key assumptions:
- Preprocessing pipeline has generated paired activation caches
- dictionary_learning library is available and compatible with SAE training
- science-of-finetuning repository is available for analysis pipeline
- W&B configuration is available in infrastructure config
- Sufficient GPU memory and disk space for training
"""

from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import json
from collections import defaultdict

from ..diffing_method import DiffingMethod
from diffing.utils.activations import get_layer_indices
from diffing.utils.dictionary.analysis import (
    build_push_sae_difference_latent_df,
    make_plots,
)
from diffing.utils.dictionary.training import (
    train_sae_difference_for_layer,
    sae_difference_run_name,
)
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
from diffing.utils.dictionary.steering import (
    run_latent_steering_experiment,
    get_sae_latent,
)
from diffing.utils.dictionary.utils import load_dictionary_model


class SAEDifferenceMethod(DiffingMethod):
    """
    Trains SAEs on activation differences and runs comprehensive analysis.

    This method:
    1. Loads paired activation caches from preprocessing pipeline
    2. Computes activation differences (finetuned - base or base - finetuned)
    3. Trains BatchTopK SAEs on normalized differences for specified layers
    4. Saves trained models with configuration and metrics
    5. Optionally uploads models to Hugging Face Hub
    6. Runs complete analysis pipeline from science-of-finetuning
    7. Returns comprehensive results including training metrics and analysis outcomes
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

        self._latent_dfs = {}

    def __hash__(self):
        return hash(self.cfg)

    def __eq__(self, other):
        return self.cfg == other.cfg

    def run(self) -> Dict[str, Any]:
        """
        Main training orchestration with analysis pipeline.

        Trains SAEs on differences for each specified layer, then runs the complete
        analysis pipeline for each trained model.

        Returns:
            Dictionary containing training results, model paths, and analysis outcomes
        """
        logger.info(f"Starting SAE difference training for layers: {self.layers}")
        logger.info(f"Training target: {self.method_cfg.training.target}")

        for layer_idx in self.layers:
            logger.info(f"Processing layer {layer_idx}")

            dictionary_name = sae_difference_run_name(
                self.cfg, layer_idx, self.base_model_cfg, self.finetuned_model_cfg
            )
            model_results_dir = (
                self.results_dir
                / "sae_difference"
                / f"layer_{layer_idx}"
                / dictionary_name
            )
            model_results_dir.mkdir(parents=True, exist_ok=True)
            if (
                not (
                    model_results_dir / "dictionary_model" / "model.safetensors"
                ).exists()
                or self.method_cfg.training.overwrite
            ):
                # Train SAE on differences for this layer
                logger.info(f"Training SAE on differences for layer {layer_idx}")
                training_metrics, model_path = train_sae_difference_for_layer(
                    self.cfg, layer_idx, self.device, dictionary_name
                )
                # save model
                dictionary_model = load_dictionary_model(model_path)
                dictionary_model.save_pretrained(model_results_dir / "dictionary_model")
                # save training metrics
                with open(model_results_dir / "training_metrics.json", "w") as f:
                    json.dump(training_metrics, f)

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
                local_model_path = str(model_results_dir / "dictionary_model")
                dict_ref = (
                    local_model_path
                    if not self.method_cfg.upload.model
                    else dictionary_name
                )
                build_push_sae_difference_latent_df(
                    dictionary_name=dictionary_name,
                    target=self.method_cfg.training.target,
                    model_path=model_results_dir / "dictionary_model",
                    push_to_hub=self.method_cfg.upload.model,
                )

                if self.method_cfg.analysis.latent_scaling.enabled:
                    logger.info(f"Computing latent scaling for layer {layer_idx}")
                    compute_scalers_from_config(
                        cfg=self.cfg,
                        layer=layer_idx,
                        dictionary_model=dict_ref,
                        results_dir=model_results_dir,
                    )
                    update_latent_df_with_beta_values(
                        dict_ref,
                        model_results_dir,
                        num_samples=self.method_cfg.analysis.latent_scaling.num_samples,
                    )

                if self.method_cfg.analysis.latent_activations.enabled:
                    logger.info(f"Collecting latent activations for layer {layer_idx}")
                    latent_activations_cache = (
                        collect_dictionary_activations_from_config(
                            cfg=self.cfg,
                            layer=layer_idx,
                            dictionary_model_name=dict_ref,
                            result_dir=model_results_dir,
                        )
                    )
                    collect_activating_examples(
                        dictionary_model_name=dict_ref,
                        latent_activation_cache=latent_activations_cache,
                        n=self.method_cfg.analysis.latent_activations.n_max_activations,
                        upload_to_hub=self.method_cfg.analysis.latent_activations.upload_to_hub,
                        overwrite=self.method_cfg.analysis.latent_activations.overwrite,
                        save_path=model_results_dir,
                    )
                    update_latent_df_with_stats(
                        dictionary_name=dict_ref,
                        latent_activation_cache=latent_activations_cache,
                        split_of_cache=self.method_cfg.analysis.latent_activations.split,
                        device=self.method_cfg.analysis.latent_activations.cache_device,
                        save_path=model_results_dir,
                    )

                try:
                    make_plots(
                        dictionary_name=dict_ref,
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
                        get_latent_fn=get_sae_latent,
                        dictionary_name=dict_ref,
                        results_dir=model_results_dir,
                        layer=layer_idx,
                    )
            logger.info(f"Successfully completed layer {layer_idx}")

        return {"status": "completed", "layers_processed": self.layers}

    def visualize(self) -> None:
        """Create Streamlit visualization for SAE difference results."""
        from .dashboard import visualize

        visualize(self)

    @torch.no_grad()
    def compute_sae_activations_for_tokens(
        self,
        dictionary_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
    ) -> Dict[str, Any]:
        """
        Compute SAE latent activations for given tokens (used by online dashboard).

        This method:
        1. Extracts activations from both base and finetuned models using nnsight
        2. Computes activation differences based on training target
        3. Passes differences through the trained SAE to get latent activations
        4. Returns tokens, latent activations, and statistics

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            layer: Layer index to analyze

        Returns:
            Dictionary with tokens, latent_activations, and statistics
        """
        from diffing.utils.dictionary.utils import load_dictionary_model

        # Shape assertions
        assert (
            input_ids.ndim == 2
        ), f"Expected 2D input_ids, got shape {input_ids.shape}"
        assert (
            attention_mask.ndim == 2
        ), f"Expected 2D attention_mask, got shape {attention_mask.shape}"
        assert (
            input_ids.shape == attention_mask.shape
        ), f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}"

        # Get tokens for display
        token_ids = input_ids[0].cpu().numpy()  # Take first sequence
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]

        # Extract activations from both models using nnsight
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            # Get base model activations
            with self.base_model.trace(inputs):
                base_activations = self.base_model.layers_output[layer].save()

            # Get finetuned model activations
            with self.finetuned_model.trace(inputs):
                finetuned_activations = self.finetuned_model.layers_output[layer].save()

        # Extract the values and move to CPU
        base_acts = base_activations.cpu()  # [batch_size, seq_len, hidden_dim]
        finetuned_acts = (
            finetuned_activations.cpu()
        )  # [batch_size, seq_len, hidden_dim]

        # Shape assertions
        batch_size, seq_len, hidden_dim = base_acts.shape
        assert finetuned_acts.shape == (
            batch_size,
            seq_len,
            hidden_dim,
        ), f"Shape mismatch: base {base_acts.shape} vs finetuned {finetuned_acts.shape}"

        # Compute activation differences based on training target
        # Assumption: training target determines direction of difference computation
        if self.method_cfg.training.target == "difference_bft":  # base - finetuned
            activation_diffs = base_acts - finetuned_acts
        else:  # difference_ftb: finetuned - base (default)
            activation_diffs = finetuned_acts - base_acts

        # Shape assertion for differences
        assert activation_diffs.shape == (
            batch_size,
            seq_len,
            hidden_dim,
        ), f"Expected diff shape {(batch_size, seq_len, hidden_dim)}, got {activation_diffs.shape}"

        # Load the trained SAE model for this layer
        try:
            sae_model = load_dictionary_model(dictionary_name, is_sae=True)
            sae_model = sae_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAE model {dictionary_name}: {str(e)}")

        # Pass differences through SAE to get latent activations
        # Take first sequence for analysis
        diff_sequence = activation_diffs[0].to(self.device)  # [seq_len, hidden_dim]

        # Encode differences to get latent activations
        latent_activations = sae_model.encode(diff_sequence)  # [seq_len, dict_size]

        # Shape assertion for latent activations
        dict_size = sae_model.dict_size
        assert latent_activations.shape == (
            seq_len,
            dict_size,
        ), f"Expected latent shape {(seq_len, dict_size)}, got {latent_activations.shape}"

        # Convert to numpy for visualization and statistics
        latent_activations_np = latent_activations.cpu().detach().numpy()

        # Compute per-token maximum latent activation for visualization
        max_activations_per_token = np.max(latent_activations_np, axis=1)  # [seq_len]

        # Compute statistics
        statistics = {
            "mean": float(np.mean(max_activations_per_token)),
            "std": float(np.std(max_activations_per_token)),
            "min": float(np.min(max_activations_per_token)),
            "max": float(np.max(max_activations_per_token)),
            "median": float(np.median(max_activations_per_token)),
        }

        return {
            "tokens": tokens,
            "latent_activations": latent_activations_np,
            "max_activations_per_token": max_activations_per_token,
            "statistics": statistics,
            "total_tokens": len(tokens),
            "layer": layer,
            "dict_size": dict_size,
        }

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available SAE difference results.

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
                sae_dir = organism_dir / "sae_difference"
                if sae_dir.exists() and any(sae_dir.iterdir()):
                    results[model_name][organism_name] = str(sae_dir)

        return results
