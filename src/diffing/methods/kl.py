"""
KL Divergence-based model diffing method.

This module computes the KL divergence per token between a base model and a
finetuned model using cached token sequences.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from torch.nn import functional as F
from omegaconf import DictConfig
from loguru import logger
import json
import numpy as np
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import streamlit as st
import time

from .diffing_method import DiffingMethod
from src.utils.model import place_inputs
from src.utils.configs import get_dataset_configurations, DatasetConfig
from src.utils.activations import get_layer_indices, load_activation_dataset_from_config
from src.utils.cache import SampleCache
from src.utils.max_act_store import MaxActStore, ReadOnlyMaxActStore
from src.utils.dashboards import (
    AbstractOnlineDiffingDashboard,
    MaxActivationDashboardComponent,
)
from src.utils.visualization import multi_tab_interface


class KLDivergenceDiffingMethod(DiffingMethod):
    """
    Computes KL divergence per token between base and finetuned models.

    This method:
    1. Loads base and finetuned models
    2. Uses cached token sequences from SampleCache
    3. Computes per-token KL divergence between model outputs
    4. Tracks max activating examples with full context
    5. Aggregates statistics per dataset
    6. Saves results to disk
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
        # filter out the validation datasets
        self.datasets = [ds for ds in self.datasets if ds.split == "train"]

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "kl"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_sample_cache(self, dataset_cfg: DatasetConfig) -> SampleCache:
        """
        Load SampleCache for a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            SampleCache instance
        """
        # As we're not using the activations, any of the cached layers is fine
        layer = get_layer_indices(
            self.base_model_cfg.model_id, self.cfg.preprocessing.layers
        )[0]

        # Load the paired activation cache
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
            f"Loaded sample cache: {dataset_cfg.id} ({len(sample_cache)} samples)"
        )
        return sample_cache

    def compute_kl_divergence(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token KL divergence between base and finetuned model outputs.

        Args:
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of:
                per_token_kl: KL divergence per token [batch_size, seq_len]
                mean_per_sample_kl: Mean KL divergence per sample [batch_size]
        """
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            # Place batch for each model and get logits
            base_batch = place_inputs(input_ids, attention_mask, self.base_model)
            base_outputs = self.base_model(
                input_ids=base_batch["input_ids"], attention_mask=base_batch["attention_mask"]
            )
            ft_batch = place_inputs(input_ids, attention_mask, self.finetuned_model)
            finetuned_outputs = self.finetuned_model(
                input_ids=ft_batch["input_ids"], attention_mask=ft_batch["attention_mask"]
            )

            base_logits = base_outputs.logits  # [batch_size, seq_len, vocab_size]
            finetuned_logits = (
                finetuned_outputs.logits
            )  # [batch_size, seq_len, vocab_size]

            # Ensure tensors are on a common device for math
            # Choose base logits' device for computation
            target_device = base_logits.device
            finetuned_logits = finetuned_logits.to(target_device)

            # Shape assertions for logits
            vocab_size = base_logits.shape[-1]
            assert base_logits.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {base_logits.shape}"
            assert finetuned_logits.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {finetuned_logits.shape}"
            assert (
                base_logits.shape == finetuned_logits.shape
            ), f"Expected: {base_logits.shape}, got: {finetuned_logits.shape}"

            # Apply temperature
            temperature = self.method_cfg.method_params.temperature
            if temperature != 1.0:
                base_logits = base_logits / temperature
                finetuned_logits = finetuned_logits / temperature

            # Convert to log probabilities
            base_log_probs = F.log_softmax(
                base_logits, dim=-1
            )  # [batch_size, seq_len, vocab_size]
            finetuned_log_probs = F.log_softmax(
                finetuned_logits, dim=-1
            )  # [batch_size, seq_len, vocab_size]

            # Shape assertions for log probabilities
            assert base_log_probs.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {base_log_probs.shape}"
            assert finetuned_log_probs.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {finetuned_log_probs.shape}"

            # Convert to probabilities for KL computation
            finetuned_probs = torch.exp(finetuned_log_probs)

            # Shape assertions for probabilities
            assert finetuned_probs.shape == (
                batch_size,
                seq_len,
                vocab_size,
            ), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {finetuned_probs.shape}"

            # Compute KL divergence: KL(finetuned || base) = sum(finetuned * log(finetuned / base))
            # = sum(finetuned * (log_finetuned - log_base))
            kl_div = torch.sum(
                finetuned_probs * (finetuned_log_probs - base_log_probs), dim=-1
            )

            # Shape assertions for KL divergence
            assert kl_div.shape == (
                batch_size,
                seq_len,
            ), f"Expected: {(batch_size, seq_len)}, got: {kl_div.shape}"

            # Compute mean per sample KL (excluding padding tokens) - vectorized version
            # Mask out padding tokens by setting their KL to 0
            masked_kl = kl_div * attention_mask.float()
            assert masked_kl.shape == (
                batch_size,
                seq_len,
            ), f"Expected: {(batch_size, seq_len)}, got: {masked_kl.shape}"

            # Sum valid KL values per sample
            kl_sums = torch.sum(masked_kl, dim=1)
            assert kl_sums.shape == (
                batch_size,
            ), f"Expected: {(batch_size,)}, got: {kl_sums.shape}"
            # Count valid tokens per sample
            valid_token_counts = torch.sum(attention_mask, dim=1).float()
            assert valid_token_counts.shape == (
                batch_size,
            ), f"Expected: {(batch_size,)}, got: {valid_token_counts.shape}"

            # Compute mean, handling cases with no valid tokens
            mean_per_sample_kl_tensor = torch.where(
                valid_token_counts > 0,
                kl_sums / valid_token_counts,
                torch.zeros_like(kl_sums),
            )

            # Shape assertion for mean per sample KL
            assert mean_per_sample_kl_tensor.shape == (
                batch_size,
            ), f"Expected: {(batch_size,)}, got: {mean_per_sample_kl_tensor.shape}"

            return kl_div, mean_per_sample_kl_tensor

    def update_max_examples_store(
        self,
        per_token_kl: torch.Tensor,
        mean_per_sample_kl: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_store_per_token_writer,
        max_store_mean_per_sample_writer,
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Update stores with examples that have high KL divergence using async writers.

        Args:
            per_token_kl: KL divergence per token [batch_size, seq_len]
            mean_per_sample_kl: Mean KL divergence per sample [batch_size]
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_store_per_token_writer: AsyncMaxActStoreWriter for per-token max KL
            max_store_mean_per_sample_writer: AsyncMaxActStoreWriter for mean per sample KL
            dataset_name: Dataset name for this batch
        """
        # Find max KL for each example in the batch (for per-token store)
        max_kl_per_example = []
        for batch_idx in range(per_token_kl.shape[0]):
            valid_mask = attention_mask[batch_idx].bool()
            valid_kl = per_token_kl[batch_idx][valid_mask]

            if len(valid_kl) == 0:
                max_kl_per_example.append(0.0)
            else:
                max_kl_per_example.append(torch.max(valid_kl).item())

        # Convert to tensor for batch processing
        max_kl_tensor = torch.tensor(max_kl_per_example)

        # Update per-token max KL store using async writer
        max_store_per_token_writer.add_batch_examples(
            scores_per_example=max_kl_tensor.cpu(),
            input_ids_batch=input_ids.cpu(),
            attention_mask_batch=attention_mask.cpu(),
            scores_per_token_batch=per_token_kl.cpu(),
            dataset_name=dataset_name,
        )

        # Update mean per sample KL store using async writer
        max_store_mean_per_sample_writer.add_batch_examples(
            scores_per_example=mean_per_sample_kl.cpu(),
            input_ids_batch=input_ids.cpu(),
            attention_mask_batch=attention_mask.cpu(),
            scores_per_token_batch=per_token_kl.cpu(),
            dataset_name=dataset_name,
        )

    def compute_statistics(self, all_kl_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical summaries of KL divergence values.

        Args:
            all_kl_values: Flattened tensor of all KL values

        Returns:
            Dictionary with statistical summaries
        """
        stats = {}

        # Convert to float32 first if needed, then to numpy for percentile computation
        if all_kl_values.dtype == torch.bfloat16:
            all_kl_values = all_kl_values.float()
        kl_np = all_kl_values.cpu().numpy()

        # Basic statistics
        stats["mean"] = float(torch.mean(all_kl_values).item())
        stats["std"] = float(torch.std(all_kl_values).item())
        stats["median"] = float(torch.median(all_kl_values).item())
        stats["min"] = float(torch.min(all_kl_values).item())
        stats["max"] = float(torch.max(all_kl_values).item())

        # Percentiles - read from config
        percentile_values = [25, 50, 75]
        if isinstance(percentile_values, list):
            for item in percentile_values:
                stats[f"percentile_{item}"] = float(np.percentile(kl_np, item))
        return stats

    def prepare_batch(
        self, sequences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of sequences for processing.

        Args:
            sequences: List of token tensors

        Returns:
            Tuple of (input_ids, attention_mask)
        """

        # Truncate sequences to max length if specified
        max_len = self.method_cfg.method_params.max_tokens_per_sample
        truncated_sequences = [seq[:max_len] for seq in sequences]

        # Pad sequences using torch's built-in function
        input_ids = pad_sequence(
            truncated_sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i, seq in enumerate(truncated_sequences):
            attention_mask[i, : len(seq)] = 1

        return input_ids, attention_mask

    @torch.no_grad()
    def process_dataset(
        self,
        dataset_cfg: DatasetConfig,
        max_act_store_per_token_writer,
        max_act_store_mean_per_sample_writer,
    ) -> Dict[str, Any]:
        """
        Process a single dataset and compute KL divergences using async writers.

        Args:
            dataset_cfg: Dataset configuration
            max_act_store_per_token_writer: AsyncMaxActStoreWriter for per-token max KL
            max_act_store_mean_per_sample_writer: AsyncMaxActStoreWriter for mean per sample KL
        Returns:
            Dictionary containing aggregated statistics and max examples for this dataset
        """
        self.logger.info(f"Processing dataset: {dataset_cfg.id}")

        # Load sample cache for this dataset
        sample_cache = self.load_sample_cache(dataset_cfg)

        # Get all sequences
        sequences = sample_cache.sequences
        assert sequences is not None, "Sample cache sequences should not be None"

        # Apply max_samples limit if specified
        if self.method_cfg.method_params.max_samples is not None:
            sequences = sequences[: self.method_cfg.method_params.max_samples]

        # Filter sequences with only one token
        sequences = [seq for seq in sequences if len(seq) > 1]

        self.logger.info(f"Processing {len(sequences)} sequences from {dataset_cfg.id}")

        all_per_token_kl_values = []
        all_mean_per_sample_kl_values = []
        batch_size = self.method_cfg.method_params.batch_size
        # Process sequences in batches
        for i in trange(
            0,
            len(sequences),
            batch_size,
            desc=f"Processing batches from {dataset_cfg.id}",
        ):
            batch_sequences = sequences[i : i + batch_size]

            # Prepare batch tensors (stay on CPU; placement happens per model)
            input_ids, attention_mask = self.prepare_batch(batch_sequences)

            # Compute KL divergence
            per_token_kl, mean_per_sample_kl = self.compute_kl_divergence(
                input_ids, attention_mask
            )

            # Update max examples stores using async writers
            self.update_max_examples_store(
                per_token_kl,
                mean_per_sample_kl,
                input_ids,
                attention_mask,
                max_act_store_per_token_writer,
                max_act_store_mean_per_sample_writer,
                dataset_name=dataset_cfg.name,
            )

            # Collect valid KL values (excluding padding)
            valid_mask = attention_mask.flatten().bool()
            valid_per_token_kl_batch = per_token_kl.flatten()[valid_mask]
            all_per_token_kl_values.append(valid_per_token_kl_batch)

            # Collect mean per sample KL values
            all_mean_per_sample_kl_values.append(mean_per_sample_kl)

        # Concatenate all KL values
        all_per_token_kl_tensor = torch.cat(all_per_token_kl_values, dim=0)
        all_mean_per_sample_kl_tensor = torch.cat(all_mean_per_sample_kl_values, dim=0)

        self.logger.info(
            f"Computed KL divergence for {len(all_per_token_kl_tensor)} tokens from {dataset_cfg.id}"
        )

        # Compute statistics for both metrics
        per_token_statistics = self.compute_statistics(all_per_token_kl_tensor)
        mean_per_sample_statistics = self.compute_statistics(
            all_mean_per_sample_kl_tensor
        )

        return {
            "dataset_id": dataset_cfg.id,
            "per_token_statistics": per_token_statistics,
            "mean_per_sample_statistics": mean_per_sample_statistics,
            "total_tokens_processed": len(all_per_token_kl_tensor),
            "total_sequences_processed": len(sequences),
            "metadata": {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
            },
        }

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
        output_file = self.results_dir / f"{safe_name}.json"

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved results for {dataset_id} to {output_file}")
        return output_file

    def run(self) -> None:
        """
        Main execution method for KL divergence diffing using async writers.

        Processes each dataset separately and saves results to disk.
        """
        # Ensure models are loaded
        self.setup_models()

        self.logger.info("Starting KL divergence computation across datasets...")

        per_token_path = self.results_dir / f"examples_per_token.db"
        mean_per_sample_path = self.results_dir / f"examples_mean_per_sample.db"
        if per_token_path.exists() and mean_per_sample_path.exists():
            if not self.method_cfg.overwrite:
                self.logger.info(
                    f"Results already exist for {per_token_path} and {mean_per_sample_path}. Skipping computation."
                )
                return
            else:
                self.logger.info(
                    f"Results do exist for {per_token_path} and {mean_per_sample_path}. Overwriting."
                )
                per_token_path.unlink()
                mean_per_sample_path.unlink()

        # Create separate MaxActStore instances for different metrics
        max_act_store_per_token = MaxActStore(
            per_token_path,
            max_examples=self.method_cfg.analysis.max_activating_examples.num_examples,
            tokenizer=self.tokenizer,
            per_dataset=True,
        )

        max_act_store_mean_per_sample = MaxActStore(
            mean_per_sample_path,
            max_examples=self.method_cfg.analysis.max_activating_examples.num_examples,
            tokenizer=self.tokenizer,
            per_dataset=True,
        )

        # Use async writers with context managers for efficient background writing
        with max_act_store_per_token.create_async_writer(
            buffer_size=self.method_cfg.method_params.batch_size
            * 10,  # Buffer ~10 batches
            flush_interval=30.0,
            auto_maintain_top_k=True,
        ) as per_token_writer, max_act_store_mean_per_sample.create_async_writer(
            buffer_size=self.method_cfg.method_params.batch_size * 10,
            flush_interval=30.0,
            auto_maintain_top_k=True,
        ) as mean_per_sample_writer:

            # Process each dataset separately
            for dataset_cfg in self.datasets:
                # Process dataset using async writers
                results = self.process_dataset(
                    dataset_cfg, per_token_writer, mean_per_sample_writer
                )

                # Save results to disk
                output_file = self.save_results(dataset_cfg.id.split("/")[-1], results)

        self.logger.info("KL divergence computation completed successfully")
        self.logger.info(f"Results saved to: {self.results_dir}")

    def visualize(self) -> None:
        """
        Create Streamlit visualization for KL divergence results with tabs.

        Returns:
            Streamlit component displaying dataset statistics and interactive analysis
        """

        multi_tab_interface(
            [
                ("📊 MaxAct Examples", self._render_dataset_statistics),
                ("🔥 Interactive", lambda: KLDivergenceOnlineDashboard(self).display()),
            ],
            "KL Divergence Analysis",
        )

    def _render_dataset_statistics(self):
        """Render the dataset statistics tab using MaxActivationDashboardComponent."""
        # Choose which metric to display
        metric_choice = st.selectbox(
            "Select KL Metric:", ["Per-Token KL", "Mean Per Sample KL"], index=0
        )

        if metric_choice == "Per-Token KL":
            max_act_store = ReadOnlyMaxActStore(
                self.results_dir / f"examples_per_token.db",
                tokenizer=self.tokenizer,
            )
            title = "Max Per-Token KL Divergence Examples"
        else:
            max_act_store = ReadOnlyMaxActStore(
                self.results_dir / f"examples_mean_per_sample.db",
                tokenizer=self.tokenizer,
            )
            title = "Mean Per-Sample KL Divergence Examples"

        # Create and display the dashboard component
        component = MaxActivationDashboardComponent(max_act_store, title=title)
        component.display()

    def compute_kl_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute KL divergence statistics for given tokens (used by both method and dashboard).

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]

        Returns:
            Dictionary with tokens, kl_values, and statistics
        """
        # Ensure models are loaded (they will auto-load via properties)

        # Compute KL divergence
        per_token_kl, mean_per_sample_kl = self.compute_kl_divergence(input_ids, attention_mask)
    
        st.info(f"Per-token KL: {per_token_kl}")
        st.info(f"Mean per sample KL: {mean_per_sample_kl}")
        # Convert to numpy for easier handling
        kl_values = per_token_kl.cpu().float().numpy().flatten()

        # Get tokens (excluding first token since KL is computed for predictions)
        token_ids = input_ids[0, 1:].cpu().numpy()  # Take first sequence, skip BOS
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]


        mean_per_sample = np.mean(mean_per_sample_kl) if len(mean_per_sample_kl) > 1 else mean_per_sample_kl[0]
        # Compute statistics
        statistics = {
            "mean": float(np.mean(kl_values)),
            "std": float(np.std(kl_values)),
            "min": float(np.min(kl_values)),
            "max": float(np.max(kl_values)),
            "median": float(np.median(kl_values)),
            "mean_per_sample": float(mean_per_sample),
        }

        return {
            "tokens": tokens,
            "kl_values": kl_values,
            "statistics": statistics,
            "total_tokens": len(tokens),
        }

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available KL divergence results.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
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
                kl_dir = organism_dir / "kl"
                if kl_dir.exists() and list(kl_dir.glob("*.db")):
                    results[model_name][organism_name] = str(kl_dir)

        return results


class KLDivergenceOnlineDashboard(AbstractOnlineDiffingDashboard):
    """
    Online dashboard for interactive KL divergence analysis.
    """

    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render KL-specific controls in Streamlit (none needed)."""
        return {}

    def compute_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        """Compute KL divergence statistics using the parent method's computation function."""
        results = self.method.compute_kl_for_tokens(input_ids, attention_mask)

        # Adapt the results format for the abstract dashboard
        return {
            "tokens": results["tokens"],
            "values": results["kl_values"],  # Use 'values' as the standard key
            "statistics": results["statistics"],
            "total_tokens": results["total_tokens"],
        }

    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get KL-specific parameters (none needed)."""
        return {}

    def _get_title(self) -> str:
        """Get title for KL analysis."""
        return "KL Divergence Analysis"
