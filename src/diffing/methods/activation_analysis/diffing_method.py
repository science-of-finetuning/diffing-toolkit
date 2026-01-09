"""
Main diffing method for activation analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import json
import numpy as np
from tqdm import tqdm
import streamlit as st
from matplotlib import pyplot as plt
import copy

from ..diffing_method import DiffingMethod
from diffing.utils.activations import (
    get_layer_indices,
    load_activation_dataset_from_config,
    torch_quantile,
)
from diffing.utils.configs import get_dataset_configurations, DatasetConfig
from diffing.utils.cache import SampleCache, SampleCacheDataset
from diffing.utils.collection import RunningActivationMean
from diffing.utils.max_act_store import MaxActStore

from .ui import visualize


def init_collectors(unique_template_tokens):
    collectors = {
        "all_tokens": RunningActivationMean(),
        "first_token": RunningActivationMean(position=0),
        "second_token": RunningActivationMean(position=1),
        "third_token": RunningActivationMean(position=2),
        "fourth_token": RunningActivationMean(position=3),
        "fifth_token": RunningActivationMean(position=4),
    }
    # Add collectors for unique chat template tokens
    for token_id in unique_template_tokens:
        token_name = f"chat_token_{token_id}"
        collectors[token_name] = RunningActivationMean(token_id=token_id)
    return collectors


class ActivationAnalysisDiffingMethod(DiffingMethod):
    """
    Computes L2 norm difference per token between base and finetuned model activations.

    This method:
    1. Loads paired activation caches (no model loading required)
    2. Processes each configured layer separately
    3. Computes per-token L2 norm differences between base and finetuned activations
    4. Tracks max activating examples with full context
    5. Aggregates statistics per dataset and layer
    6. Saves results to disk in nested structure
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Method-specific configuration
        self.method_cfg = cfg.diffing.method

        # Get dataset configurations
        self.datasets = get_dataset_configurations(
            cfg,
            use_chat_dataset=self.method_cfg.datasets.use_chat_dataset,
            use_pretraining_dataset=self.method_cfg.datasets.use_pretraining_dataset,
            use_training_dataset=self.method_cfg.datasets.use_training_dataset,
        )

        # Get layers to process
        self.layers = get_layer_indices(
            self.base_model_cfg.model_id, cfg.preprocessing.layers
        )

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "activation_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader configuration
        self.batch_size = getattr(self.method_cfg.method_params, "batch_size", 32)
        self.num_workers = getattr(self.method_cfg.method_params, "num_workers", 8)

    def load_sample_cache(
        self, dataset_cfg: DatasetConfig, layer: int
    ) -> Tuple[SampleCache, Any]:
        """
        Load SampleCache for a specific dataset and layer.

        Args:
            dataset_cfg: Dataset configuration
            layer: Layer index

        Returns:
            Tuple of (SampleCache instance, tokenizer from cache)
        """

        # Load the paired activation cache for this specific layer
        paired_cache = load_activation_dataset_from_config(
            cfg=self.cfg,
            ds_cfg=dataset_cfg,
            base_model_cfg=self.base_model_cfg,
            finetuned_model_cfg=self.finetuned_model_cfg,
            layer=layer,
            split="train",
        )

        # Create SampleCache from the paired activation cache
        sample_cache = SampleCache(
            paired_cache, bos_token_id=self.tokenizer.bos_token_id
        )

        self.logger.info(
            f"Loaded sample cache: {dataset_cfg.id}, layer {layer} ({len(sample_cache)} samples)"
        )
        return sample_cache, self.tokenizer

    def compute_activation_statistics(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        collectors: Dict[str, RunningActivationMean],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-token L2 norm difference, cosine similarity, and individual norms between base and finetuned activations.

        Args:
            activations: Stacked activations [seq_len, 2, activation_dim]
                        where index 0 is base, index 1 is finetuned

        Returns:
            Tuple containing:
                - norm_diffs: L2 norm of activation differences [seq_len]
                - cos_sim: Cosine similarity between activations [seq_len]
                - norm_base: L2 norm of base activations [seq_len]
                - norm_finetuned: L2 norm of finetuned activations [seq_len]
        """
        seq_len, num_models, activation_dim = activations.shape
        # Shape assertions
        assert num_models == 2, f"Expected 2 models (base, finetuned), got {num_models}"
        assert seq_len > 0, f"Expected non-empty sequence, got length {seq_len}"

        # Extract base and finetuned activations
        base_activations = activations[:, 0, :]  # [seq_len, activation_dim]
        finetuned_activations = activations[:, 1, :]  # [seq_len, activation_dim]
        norm_base = torch.norm(base_activations, p=2, dim=-1)  # [seq_len]
        norm_finetuned = torch.norm(finetuned_activations, p=2, dim=-1)  # [seq_len]
        cos_sim = torch.nn.functional.cosine_similarity(
            base_activations, finetuned_activations, dim=-1
        )  # [seq_len]

        # Shape assertions for extracted activations
        assert base_activations.shape == (
            seq_len,
            activation_dim,
        ), f"Expected: {(seq_len, activation_dim)}, got: {base_activations.shape}"
        assert finetuned_activations.shape == (
            seq_len,
            activation_dim,
        ), f"Expected: {(seq_len, activation_dim)}, got: {finetuned_activations.shape}"

        # Compute activation differences
        activation_diffs = (
            finetuned_activations - base_activations
        )  # [seq_len, activation_dim]

        for collector in collectors.values():
            collector.update(activation_diffs, tokens)

        # Compute L2 norm per token
        norm_diffs = torch.norm(activation_diffs, p=2, dim=-1)  # [seq_len]

        # Shape assertion for norm differences
        assert norm_diffs.shape == (
            seq_len,
        ), f"Expected: {(seq_len,)}, got: {norm_diffs.shape}"
        assert cos_sim.shape == (
            seq_len,
        ), f"Expected: {(seq_len,)}, got: {cos_sim.shape}"
        assert norm_base.shape == (
            seq_len,
        ), f"Expected: {(seq_len,)}, got: {norm_base.shape}"
        assert norm_finetuned.shape == (
            seq_len,
        ), f"Expected: {(seq_len,)}, got: {norm_finetuned.shape}"

        return (
            norm_diffs.cpu().float(),
            cos_sim.cpu().float(),
            norm_base.cpu().float(),
            norm_finetuned.cpu().float(),
        )

    def compute_statistics(self, all_norm_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical summaries of norm difference values.

        Args:
            all_norm_values: Flattened tensor of all norm values

        Returns:
            Dictionary with statistical summaries
        """
        stats = {}

        # Convert to float32 first if needed, then to numpy for percentile computation
        if all_norm_values.dtype == torch.bfloat16:
            all_norm_values = all_norm_values.float()
        norm_np = all_norm_values.cpu().numpy()

        # Basic statistics
        stats["mean"] = float(torch.mean(all_norm_values).item())
        stats["std"] = float(torch.std(all_norm_values).item())
        stats["median"] = float(torch.median(all_norm_values).item())
        stats["min"] = float(torch.min(all_norm_values).item())
        stats["max"] = float(torch.max(all_norm_values).item())

        # Percentiles - read from config
        percentile_values = self.method_cfg.analysis.statistics
        if isinstance(percentile_values, list):
            for item in percentile_values:
                if isinstance(item, dict) and "percentiles" in item:
                    for p in item["percentiles"]:
                        stats[f"percentile_{p}"] = float(np.percentile(norm_np, p))
        return stats

    def compute_histogram(
        self, norm_values, norm_name: str, plot_dir: Path, no_outliers: bool = False
    ):
        """
        Compute and save a histogram of norm values.
        """

        # Compute statistics
        mean_val = float(torch.mean(norm_values).item())
        median_val = float(torch.median(norm_values).item())

        # Remove outliers if specified
        if no_outliers:
            # Calculate Q1, Q3, and IQR
            q1 = torch_quantile(norm_values, 0.25)
            q3 = torch_quantile(norm_values, 0.75)
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Filter out outliers
            mask = (norm_values >= lower_bound) & (norm_values <= upper_bound)
            norm_values = norm_values[mask]

            norm_name = f"{norm_name} (no outliers)"

        # Regular scale histogram
        plt.hist(norm_values, bins=100)
        plt.title(f"{norm_name}")
        plt.suptitle(f"n={len(norm_values):,}", y=0.02, fontsize=10)
        plt.xlabel(f"{norm_name}")
        plt.ylabel("Frequency")

        # Add mean and median lines
        plt.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )
        plt.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.4f}",
        )

        plt.legend()
        plt.savefig(plot_dir / f"{norm_name.replace(' ', '_')}.png")
        plt.close()

        # Log scale histogram
        plt.hist(norm_values, bins=100)
        plt.title(f"{norm_name} (Log Scale)")
        plt.suptitle(f"n={len(norm_values):,}", y=0.02, fontsize=10)
        plt.xlabel(f"{norm_name}")
        plt.ylabel("Frequency (Log Scale)")
        plt.yscale("log")

        # Add mean and median lines
        plt.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )
        plt.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.4f}",
        )

        plt.legend()
        plt.savefig(
            plot_dir
            / f"{norm_name.replace(' ', '_').replace("(", "").replace(")", "")}_log.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def compute_norm_comparison_plot(
        self,
        norm_base_values,
        norm_finetuned_values,
        plot_dir: Path,
        no_outliers: bool = False,
    ):
        """
        Create a comparison plot of base and finetuned activation norms.
        """
        if no_outliers:
            # Calculate Q1, Q3, and IQR
            q1 = torch_quantile(norm_base_values, 0.25)
            q3 = torch_quantile(norm_base_values, 0.75)
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Filter out outliers
            mask = (norm_base_values >= lower_bound) & (norm_base_values <= upper_bound)
            norm_base_values = norm_base_values[mask]
            norm_finetuned_values = norm_finetuned_values[mask]

        plt.figure(figsize=(10, 6))

        # Create histograms with different colors
        plt.hist(
            norm_base_values,
            bins=100,
            alpha=0.7,
            color="blue",
            label="Base Model",
            density=True,
        )
        plt.hist(
            norm_finetuned_values,
            bins=100,
            alpha=0.7,
            color="red",
            label="Finetuned Model",
            density=True,
        )

        plt.title("Activation Norm Comparison: Base vs Finetuned")
        plt.xlabel("Activation Norm")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            plot_dir
            / f"norm_base_vs_finetuned{'_no_outliers' if no_outliers else ''}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Log scale version
        plt.figure(figsize=(10, 6))
        plt.hist(
            norm_base_values,
            bins=100,
            alpha=0.7,
            color="blue",
            label="Base Model",
            density=True,
        )
        plt.hist(
            norm_finetuned_values,
            bins=100,
            alpha=0.7,
            color="red",
            label="Finetuned Model",
            density=True,
        )

        plt.title("Activation Norm Comparison: Base vs Finetuned (Log Scale)")
        plt.xlabel("Activation Norm")
        plt.ylabel("Density (Log Scale)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            plot_dir
            / f"norm_base_vs_finetuned{'_no_outliers' if no_outliers else ''}_log.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def compute_plots(
        self,
        diff_norm_values,
        cos_sim_values,
        norm_base_values,
        norm_finetuned_values,
        plot_dir: Path,
    ):
        """
        Compute and save plots of norm differences, cosine similarities, and other statistics.
        """
        self.compute_histogram(
            diff_norm_values, "Activation Difference Norm", plot_dir, no_outliers=True
        )
        self.compute_histogram(diff_norm_values, "Activation Difference Norm", plot_dir)
        self.compute_histogram(cos_sim_values, "Cosine Similarity", plot_dir)
        self.compute_histogram(norm_base_values, "Base Activation Norm", plot_dir)
        self.compute_histogram(
            norm_base_values, "Base Activation Norm", plot_dir, no_outliers=True
        )
        self.compute_histogram(
            norm_finetuned_values, "Finetuned Activation Norm", plot_dir
        )
        self.compute_histogram(
            norm_finetuned_values,
            "Finetuned Activation Norm",
            plot_dir,
            no_outliers=True,
        )
        self.compute_norm_comparison_plot(
            norm_base_values, norm_finetuned_values, plot_dir
        )

        relative_diff_values = diff_norm_values / (
            norm_base_values + norm_finetuned_values
        )
        self.compute_histogram(
            relative_diff_values, "Relative Activation Difference Norm", plot_dir
        )
        self.compute_histogram(
            relative_diff_values,
            "Relative Activation Difference Norm",
            plot_dir,
            no_outliers=True,
        )

    def _get_chat_template_tokens(self):
        return self.tokenizer.apply_chat_template(
            [{"content": "", "role": "user"}], tokenize=True, add_generation_prompt=True
        )

    def process_layer(
        self,
        dataset_cfg: DatasetConfig,
        layer: int,
        max_act_stores: Dict[str, MaxActStore],
    ) -> Dict[str, Any]:
        """
        Process a single layer for a dataset and compute norm differences.

        Args:
            dataset_cfg: Dataset configuration
            layer: Layer index
            max_act_stores: Dictionary of MaxActStore instances for each type
        Returns:
            Dictionary containing statistics and max examples for this dataset/layer
        """
        self.logger.info(f"Processing dataset: {dataset_cfg.id}, layer: {layer}")

        # Load sample cache for this dataset and layer
        sample_cache, tokenizer = self.load_sample_cache(dataset_cfg, layer)

        # Use dataset and layer specific database path
        safe_name = dataset_cfg.id.split("/")[-1]
        dataset_dir = self.results_dir / f"layer_{layer}" / safe_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = dataset_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset and dataloader
        dataset = SampleCacheDataset(
            sample_cache, self.method_cfg.method_params.max_samples
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,  # Avoid pinning since we have variable-length sequences
            persistent_workers=self.num_workers
            > 0,  # Keep workers alive for better performance
        )

        self.logger.info(
            f"Processing {len(dataset)} samples from {dataset_cfg.id}, layer {layer}"
        )
        self.logger.info(
            f"Using DataLoader with {self.num_workers} workers, batch_size={self.batch_size}"
        )

        # Initialize activation difference collectors
        chat_template_ids = self._get_chat_template_tokens()
        unique_template_tokens = list(set(chat_template_ids))  # Get unique tokens
        dataset_name = dataset_cfg.id.split("/")[-1]
        collectors = init_collectors(unique_template_tokens)

        all_norm_values = []
        all_cos_sim_values = []
        all_norm_base_values = []
        all_norm_finetuned_values = []
        processed_samples = 0

        BASE_DICT = {
            "scores_per_example": [],
            "input_ids_batch": [],
            "scores_per_token_batch": [],
            "dataset_name": dataset_cfg.name,
        }

        # Use async writers with context managers for efficient background writing
        with (
            max_act_stores["norm_diff"].create_async_writer(
                buffer_size=self.batch_size * 10,  # Buffer ~10 batches
                flush_interval=30.0,
                auto_maintain_top_k=True,
            ) as norm_diff_writer,
            max_act_stores["mean_norm_diff"].create_async_writer(
                buffer_size=self.batch_size * 10,
                flush_interval=30.0,
                auto_maintain_top_k=True,
            ) as mean_norm_diff_writer,
            max_act_stores["cos_dist"].create_async_writer(
                buffer_size=self.batch_size * 10,
                flush_interval=30.0,
                auto_maintain_top_k=True,
            ) as cos_dist_writer,
            max_act_stores["mean_cos_dist"].create_async_writer(
                buffer_size=self.batch_size * 10,
                flush_interval=30.0,
                auto_maintain_top_k=True,
            ) as mean_cos_dist_writer,
            max_act_stores["norm_base"].create_async_writer(
                buffer_size=self.batch_size * 10,
                flush_interval=30.0,
                auto_maintain_top_k=True,
            ) as norm_base_writer,
            max_act_stores["norm_finetuned"].create_async_writer(
                buffer_size=self.batch_size * 10,
                flush_interval=30.0,
                auto_maintain_top_k=True,
            ) as norm_finetuned_writer,
        ):

            # Process samples in batches using DataLoader
            for batch in tqdm(dataloader, desc=f"Processing batches for layer {layer}"):
                norm_base_values = copy.deepcopy(BASE_DICT)
                norm_finetuned_values = copy.deepcopy(BASE_DICT)
                cos_dist_values = copy.deepcopy(BASE_DICT)
                norm_diffs_values = copy.deepcopy(BASE_DICT)
                mean_norm_diffs_values = copy.deepcopy(BASE_DICT)
                mean_cos_dist_values = copy.deepcopy(BASE_DICT)

                # Process each sample in the batch
                for tokens, activations in batch:
                    # Move activations to GPU for computation
                    activations = activations.to(self.device)

                    # Compute norm differences
                    norm_diffs, cos_sim, norm_base, norm_finetuned = (
                        self.compute_activation_statistics(
                            activations, tokens, collectors
                        )
                    )

                    # Convert cosine similarity to cosine distance
                    cos_dist = 1 - cos_sim

                    if self.method_cfg.method_params.skip_first_n_tokens:
                        n = (
                            self.cfg.model.ignore_first_n_tokens_per_sample_during_training
                        )
                        max_norm_value = torch.max(norm_diffs[n:]).item()
                        mean_norm_value = torch.mean(norm_diffs[n:]).item()
                        max_cos_dist_value = torch.max(cos_dist[n:]).item()
                        mean_cos_dist_value = torch.mean(cos_dist[n:]).item()
                        max_base_value = torch.max(norm_base[n:]).item()
                        max_finetuned_value = torch.max(norm_finetuned[n:]).item()
                        norm_diffs[:n] = 0
                        cos_dist[:n] = 0
                        norm_base[:n] = 0
                        norm_finetuned[:n] = 0
                    else:
                        max_norm_value = torch.max(norm_diffs).item()
                        mean_norm_value = torch.mean(norm_diffs).item()
                        max_cos_dist_value = torch.max(cos_dist).item()
                        mean_cos_dist_value = torch.mean(cos_dist).item()
                        max_base_value = torch.max(norm_base).item()
                        max_finetuned_value = torch.max(norm_finetuned).item()
                    # Add to batch dictionaries
                    norm_base_values["scores_per_example"].append(max_base_value)
                    norm_base_values["input_ids_batch"].append(tokens)
                    norm_base_values["scores_per_token_batch"].append(norm_base)

                    norm_finetuned_values["scores_per_example"].append(
                        max_finetuned_value
                    )
                    norm_finetuned_values["input_ids_batch"].append(tokens)
                    norm_finetuned_values["scores_per_token_batch"].append(
                        norm_finetuned
                    )

                    cos_dist_values["scores_per_example"].append(max_cos_dist_value)
                    cos_dist_values["input_ids_batch"].append(tokens)
                    cos_dist_values["scores_per_token_batch"].append(cos_dist)

                    norm_diffs_values["scores_per_example"].append(max_norm_value)
                    norm_diffs_values["input_ids_batch"].append(tokens)
                    norm_diffs_values["scores_per_token_batch"].append(norm_diffs)

                    mean_norm_diffs_values["scores_per_example"].append(mean_norm_value)
                    mean_norm_diffs_values["input_ids_batch"].append(tokens)
                    mean_norm_diffs_values["scores_per_token_batch"].append(norm_diffs)

                    mean_cos_dist_values["scores_per_example"].append(
                        mean_cos_dist_value
                    )
                    mean_cos_dist_values["input_ids_batch"].append(tokens)
                    mean_cos_dist_values["scores_per_token_batch"].append(cos_dist)

                    # Collect all norm values for statistics
                    all_norm_values.append(norm_diffs)
                    all_cos_sim_values.append(cos_sim)
                    all_norm_base_values.append(norm_base)
                    all_norm_finetuned_values.append(norm_finetuned)

                    processed_samples += 1

                # Convert to tensors for batch writing
                norm_base_values["scores_per_example"] = torch.tensor(
                    norm_base_values["scores_per_example"]
                )
                norm_finetuned_values["scores_per_example"] = torch.tensor(
                    norm_finetuned_values["scores_per_example"]
                )
                norm_diffs_values["scores_per_example"] = torch.tensor(
                    norm_diffs_values["scores_per_example"]
                )
                mean_norm_diffs_values["scores_per_example"] = torch.tensor(
                    mean_norm_diffs_values["scores_per_example"]
                )
                cos_dist_values["scores_per_example"] = torch.tensor(
                    cos_dist_values["scores_per_example"]
                )
                mean_cos_dist_values["scores_per_example"] = torch.tensor(
                    mean_cos_dist_values["scores_per_example"]
                )

                # Convert to tensors for batch writing
                norm_base_writer.add_batch_examples(**norm_base_values)

                norm_finetuned_writer.add_batch_examples(**norm_finetuned_values)

                norm_diff_writer.add_batch_examples(**norm_diffs_values)

                mean_norm_diff_writer.add_batch_examples(**mean_norm_diffs_values)

                cos_dist_writer.add_batch_examples(**cos_dist_values)

                mean_cos_dist_writer.add_batch_examples(**mean_cos_dist_values)

        # Concatenate all norm values
        if all_norm_values:
            all_norm_tensor = torch.cat(all_norm_values, dim=0).float()
            self.logger.info(
                f"Computed norm differences for {len(all_norm_tensor)} tokens from {dataset_cfg.id}, layer {layer}"
            )

            # Compute statistics
            statistics = self.compute_statistics(all_norm_tensor)
            total_tokens = len(all_norm_tensor)
            all_cos_sim_tensor = torch.cat(all_cos_sim_values, dim=0).float()
            all_norm_base_tensor = torch.cat(all_norm_base_values, dim=0).float()
            all_norm_finetuned_tensor = torch.cat(
                all_norm_finetuned_values, dim=0
            ).float()

            self.compute_plots(
                all_norm_tensor,
                all_cos_sim_tensor,
                all_norm_base_tensor,
                all_norm_finetuned_tensor,
                plot_dir,
            )
        else:
            self.logger.warning(
                f"No valid samples found for {dataset_cfg.id}, layer {layer}"
            )
            statistics = {}
            total_tokens = 0

        # Save activation difference means
        for collector_name, collector in collectors.items():
            if collector.count > 0:
                filename = f"mean_{collector_name}_{dataset_name}.pt"
                filepath = dataset_dir / filename
                collector.save(filepath)
                self.logger.info(
                    f"Saved {collector_name} mean to {filepath} (count: {collector.count})"
                )

        return {
            "dataset_id": dataset_cfg.id,
            "layer": layer,
            "statistics": statistics,
            "total_tokens_processed": total_tokens,
            "total_samples_processed": processed_samples,
            "metadata": {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
            },
        }

    def process_dataset(
        self,
        dataset_cfg: DatasetConfig,
        max_act_stores: Dict[int, Dict[str, MaxActStore]],
    ) -> Dict[str, Any]:
        """
        Process all layers for a single dataset.

        Args:
            dataset_cfg: Dataset configuration
            max_act_stores: Dictionary of MaxActStore instances for each layer
        Returns:
            Dictionary containing results for all layers of this dataset
        """
        results = {
            "dataset_id": dataset_cfg.id,
            "layers": {},
            "metadata": {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
                "processed_layers": self.layers,
            },
        }

        # Process each layer
        for layer in self.layers:
            layer_results = self.process_layer(
                dataset_cfg, layer, max_act_stores[layer]
            )
            results["layers"][f"layer_{layer}"] = layer_results

        return results

    def save_results(self, dataset_id: str, results: Dict[str, Any]) -> Path:
        """
        Save results for a dataset to disk.

        Args:
            dataset_id: Dataset identifier
            results: Results dictionary

        Returns:
            Path to saved file
        """
        # Convert dataset ID to safe filename
        safe_name = dataset_id.split("/")[-1]

        # Create dataset subdirectory
        dataset_dir = self.results_dir / safe_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save overall results
        output_file = dataset_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save layer-specific results separately
        for layer_name, layer_results in results["layers"].items():
            layer_file = dataset_dir / f"{layer_name}.json"
            with open(layer_file, "w") as f:
                json.dump(layer_results, f, indent=2)

        self.logger.info(f"Saved results for {dataset_id} to {dataset_dir}")
        return output_file

    def run(self) -> None:
        """
        Main execution method for norm difference diffing.

        Processes each dataset and layer combination, saves results to disk.
        """
        self.logger.info(
            "Starting norm difference computation across datasets and layers..."
        )

        # Test if the results directory exists
        run = False
        if self.results_dir.exists() and not self.method_cfg.overwrite:
            for layer in self.layers:
                for store_type in [
                    "norm_diff",
                    "mean_norm_diff",
                    "cos_dist",
                    "mean_cos_dist",
                    "norm_base",
                    "norm_finetuned",
                ]:
                    path = self.results_dir / f"layer_{layer}" / f"{store_type}.db"
                    if path.exists():
                        self.logger.info(f"Found {path}")
                    else:
                        run = True
            if not run:
                self.logger.info(
                    f"Results already exists. Set diffing.method.activation_analysis.overwrite=true to overwrite."
                )
                return

        if self.method_cfg.overwrite:
            for layer in self.layers:
                for store_type in [
                    "norm_diff",
                    "mean_norm_diff",
                    "cos_dist",
                    "mean_cos_dist",
                    "norm_base",
                    "norm_finetuned",
                ]:
                    path = self.results_dir / f"layer_{layer}" / f"{store_type}.db"
                    if path.exists():
                        self.logger.info(f"Removing {path}")
                        path.unlink()

        max_act_stores = {}
        for layer in self.layers:
            max_act_stores[layer] = {}
            for store_type in [
                "norm_diff",
                "mean_norm_diff",
                "cos_dist",
                "mean_cos_dist",
                "norm_base",
                "norm_finetuned",
            ]:
                folder = self.results_dir / f"layer_{layer}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{store_type}.db"
                max_act_stores[layer][store_type] = MaxActStore(
                    path,
                    tokenizer=self.tokenizer,
                    per_dataset=True,
                    max_examples=self.method_cfg.analysis.max_activating_examples.num_examples,
                )
        # Process each dataset
        for dataset_cfg in self.datasets:

            # Process all layers for this dataset
            results = self.process_dataset(dataset_cfg, max_act_stores)

            # Save results to disk
            self.save_results(dataset_cfg.id, results)

        self.logger.info("Norm difference computation completed successfully")
        self.logger.info(f"Results saved to: {self.results_dir}")

    def visualize(self) -> None:
        """
        Create Streamlit visualization for norm difference results with tabs.

        Returns:
            Streamlit component displaying dataset statistics and interactive analysis
        """
        visualize(self)

    def _find_available_layers(self) -> List[int]:
        """Find all layers that have activation difference means."""
        layer_dirs = list(self.results_dir.glob("layer_*"))
        return sorted([int(layer_dir.name.split("_")[-1]) for layer_dir in layer_dirs])

    def _find_available_datasets_for_layer(self, layer: int) -> List[str]:
        """Find all datasets that have activation difference means for given layer."""
        layer_dir = self.results_dir / f"layer_{layer}"
        if not layer_dir.exists():
            return []

        available_datasets = []
        for dataset_dir in layer_dir.iterdir():
            if dataset_dir.is_dir():
                # Check if this dataset has any mean_*.pt files
                mean_files = list(dataset_dir.glob("mean_*.pt"))
                if mean_files:
                    available_datasets.append(dataset_dir.name)

        return sorted(available_datasets)

    def _load_and_prepare_means(
        self, layer: int, dataset_name: str
    ) -> Tuple[Dict[str, Dict], List[str]]:
        """Load activation means and prepare options for the interface."""
        dataset_dir = self.results_dir / f"layer_{layer}" / dataset_name
        mean_files = list(dataset_dir.glob("mean_*.pt"))

        loaded_means = {}
        custom_options = []

        for mean_file in mean_files:
            # Parse token name from filename: mean_{token_name}_{dataset_name}.pt
            filename = mean_file.stem
            # Remove "mean_" prefix and "_{dataset_name}" suffix
            token_part = filename[5:]  # Remove "mean_"
            token_name = token_part.rsplit(f"_{dataset_name}", 1)[0]

            # Load the mean
            try:
                data = torch.load(mean_file, map_location="cpu")
                if data["count"] > 0:  # Only include tokens that have observations
                    # Create user-friendly name
                    display_name = self._create_display_name(token_name, data)
                    loaded_means[display_name] = data
                    custom_options.append(display_name)
            except Exception as e:
                self.logger.warning(f"Failed to load {mean_file}: {e}")

        return loaded_means, sorted(custom_options)

    def _create_display_name(self, token_name: str, data: Dict) -> str:
        """Convert internal token names to user-friendly display names."""
        if token_name == "all_tokens":
            return f"All Tokens (n={data['count']})"
        elif token_name == "first_token":
            return f"First Token (n={data['count']})"
        elif token_name == "second_token":
            return f"Second Token (n={data['count']})"
        elif token_name.startswith("chat_token_"):
            token_id = int(
                token_name[11:]
            )  # Remove "chat_token_" prefix and convert to int
            try:
                decoded_token = self.tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                return (
                    f"Chat Token '{decoded_token}' (id={token_id}, n={data['count']})"
                )
            except Exception:
                return f"Chat Token {token_id} (n={data['count']})"
        else:
            return f"{token_name.replace('_', ' ').title()} (n={data['count']})"

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available norm difference results.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = {}
        results_base = results_dir

        if not results_base.exists():
            return results

        # Scan for normdiff results in the expected structure
        for model_dir in results_base.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for organism_dir in model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                act_analysis_dir = organism_dir / "activation_analysis"

                # Check if normdiff results exist (any dataset dirs with results.json)
                if act_analysis_dir.exists():
                    has_results = False
                    for layer_dir in act_analysis_dir.iterdir():
                        if layer_dir.is_dir():
                            # Check for any examples database files or mean files (in dataset subdirs)
                            examples_files = list(layer_dir.glob("*.db")) + list(
                                layer_dir.glob("*/*.pt")
                            )
                            if examples_files:
                                has_results = True
                                break

                    if has_results:
                        if model_name not in results:
                            results[model_name] = {}
                        results[model_name][organism_name] = str(act_analysis_dir)

        return results

    def compute_all_activation_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int
    ) -> Dict[str, Any]:
        """
        Compute all activation statistics for given tokens (used by online dashboard).

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            layer: Layer index to analyze

        Returns:
            Dictionary with tokens and all computed statistics for different metrics
        """

        # Get tokens for display (all tokens for norm diff since we don't predict next token)
        token_ids = input_ids[0].cpu().numpy()  # Take first sequence
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]

        # Extract activations from both models
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

        # Take first sequence and stack for the compute_norm_difference method
        # Shape: [seq_len, 2, hidden_dim] where index 0 is base, index 1 is finetuned
        seq_len = base_acts.shape[1]
        hidden_dim = base_acts.shape[2]

        stacked_activations = torch.stack(
            [
                base_acts[0],  # [seq_len, hidden_dim]
                finetuned_acts[0],  # [seq_len, hidden_dim]
            ],
            dim=1,
        )  # [seq_len, 2, hidden_dim]

        # Shape assertions
        assert stacked_activations.shape == (
            seq_len,
            2,
            hidden_dim,
        ), f"Expected: {(seq_len, 2, hidden_dim)}, got: {stacked_activations.shape}"

        # Compute all statistics using existing method
        norm_diff_values, cos_sim_values, norm_base_values, norm_finetuned_values = (
            self.compute_activation_statistics(stacked_activations)
        )

        # Convert all to numpy for statistics computation
        norm_diff_np = norm_diff_values.float().cpu().numpy()
        cos_sim_np = cos_sim_values.float().cpu().numpy()
        norm_base_np = norm_base_values.float().cpu().numpy()
        norm_finetuned_np = norm_finetuned_values.float().cpu().numpy()

        cos_dist_np = 1 - cos_sim_np

        def compute_stats(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

        # Return comprehensive results for all metrics
        return {
            "tokens": tokens,
            "layer": layer,
            "total_tokens": len(tokens),
            "metrics": {
                "norm_diff": {
                    "values": norm_diff_np,
                    "max_values": norm_diff_np,  # For max aggregation
                    "mean_values": np.full_like(
                        norm_diff_np, np.mean(norm_diff_np)
                    ),  # For mean aggregation
                    "statistics": compute_stats(norm_diff_np),
                },
                "cos_dist": {
                    "values": cos_dist_np,
                    "max_values": cos_dist_np,  # For max aggregation
                    "mean_values": np.full_like(
                        cos_dist_np, np.mean(cos_dist_np)
                    ),  # For mean aggregation
                    "statistics": compute_stats(cos_dist_np),
                },
                "norm_base": {
                    "values": norm_base_np,
                    "statistics": compute_stats(norm_base_np),
                },
                "norm_ft": {
                    "values": norm_finetuned_np,
                    "statistics": compute_stats(norm_finetuned_np),
                },
            },
        }
