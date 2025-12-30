"""
Logit Diff Top-K Occurring analysis method.

This module computes occurrence rates of tokens in the top-K positive and negative
logit differences between a base model and a finetuned model.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from omegaconf import DictConfig
from loguru import logger
import json
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, IterableDataset

from ..diffing_method import DiffingMethod
from src.utils.configs import get_dataset_configurations, DatasetConfig
from src.utils.agents.diffing_method_agent import DiffingMethodAgent
from src.utils.agents.base_agent import BaseAgent
from src.utils.graders.token_relevance_grader import TokenRelevanceGrader
from ..activation_difference_lens.token_relevance import _compute_frequent_tokens
from ..activation_difference_lens.act_diff_lens import (
    load_and_tokenize_dataset,
    load_and_tokenize_chat_dataset,
)
from .normalization import normalize_token_list
from .ui import visualize
from .per_token_plots import plot_per_sample_occurrences, plot_per_position_occurrences, plot_shortlist_token_distribution
from .co_occurrence_plots import plot_co_occurrence_heatmap
from .position_distribution_plots import plot_positional_kde
from .global_token_plots import plot_global_token_scatter
from itertools import combinations_with_replacement
import scipy.sparse
from torchnmf.nmf import NMF

class LogitDiffTopKOccurringMethod(DiffingMethod):
    """
    Computes occurrence rates of tokens in top-K positive and negative logit differences.

    This method:
    1. Loads base and finetuned models (via DiffingMethod base class)
    2. Loads datasets directly from HuggingFace
    3. Computes logit differences between models for each position
    4. Tracks which tokens appear most frequently in top-K positive/negative diffs
    5. Saves occurrence rate statistics to disk
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Method-specific configuration
        self.method_cfg = cfg.diffing.method

        # Get requested variants from method config, default to ["default"] if not present
        pretraining_variants = list(getattr(self.method_cfg.datasets, "pretraining_dataset_variants", ["default"]))

        # Get dataset configurations
        self.datasets = get_dataset_configurations(
            cfg,
            use_chat_dataset=self.method_cfg.datasets.use_chat_dataset,
            use_pretraining_dataset=self.method_cfg.datasets.use_pretraining_dataset,
            use_training_dataset=self.method_cfg.datasets.use_training_dataset,

            pretraining_dataset_variants=pretraining_variants,
        )

        # Filter out validation datasets (only use train split)
        self.datasets = [ds for ds in self.datasets if ds.split == "train"]

        # NMF Clustering configuration
        self.nmf_cfg = getattr(self.method_cfg, "token_topic_clustering_NMF", None)
        if self.nmf_cfg and self.nmf_cfg.enabled:
            self.logger.info("NMF Token Topic Clustering enabled.")

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "logit_diff_topk_occurring"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def setup_models(self):
        """Ensure models are loaded (they will auto-load via properties)."""
        # Access properties to trigger lazy loading
        self.logger.info(f"Loading base model: {self.base_model_cfg.model_id}")
        _ = self.base_model
        self.logger.info(f"✓ Base model loaded: {self.base_model_cfg.model_id}")
        
        self.logger.info(f"Loading finetuned model: {self.finetuned_model_cfg.model_id}")
        _ = self.finetuned_model
        self.logger.info(f"✓ Finetuned model loaded: {self.finetuned_model_cfg.model_id}")
        
        self.logger.info("✓ Both models loaded and set to eval mode")

    @torch.no_grad()
    def _prepare_dataset_tensors(self, dataset_cfg: DatasetConfig) -> Dict[str, torch.Tensor]:
        """
        Prepare input tensors for a single dataset.
        
        Args:
            dataset_cfg: Dataset configuration
            
        Returns:
            Dict containing 'input_ids' and 'attention_mask' tensors
        """
        self.logger.info(f"Preparing input tensors for {dataset_cfg.name}...")
        max_samples = int(self.method_cfg.method_params.max_samples)
        max_tokens = int(self.method_cfg.method_params.max_tokens_per_sample)
        
        # Tokenize entire dataset using ADL functions
        if dataset_cfg.is_chat:
            self.logger.info("Using ADL's load_and_tokenize_chat_dataset()")
            # Streaming is not implemented for chat dataset yet in this refactor context
            # assuming it's still using the standard load_dataset for chat
            samples = load_and_tokenize_chat_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                messages_column=dataset_cfg.messages_column or "messages",
                n=max_tokens,
                pre_assistant_k=0,
                max_samples=max_samples,
            )
            all_token_ids = [sample["input_ids"] for sample in samples]
        else:
            # Always stream (implicit)
            cache_dir = self.results_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct safe filename
            safe_id = dataset_cfg.id.replace("/", "_")
            subset_str = f"_{dataset_cfg.subset}" if dataset_cfg.subset else ""
            cache_filename = f"{safe_id}{subset_str}_{dataset_cfg.split}_N{max_samples}.json"
            cache_file = cache_dir / cache_filename
            
            if not self.method_cfg.overwrite and cache_file.exists():
                self.logger.info(f"Using cached dataset file: {cache_file}")
            else:
                self.logger.info(f"Streaming dataset {dataset_cfg.id} (subset={dataset_cfg.subset})...")
                
                load_kwargs = {"streaming": True, "split": dataset_cfg.split}
                if dataset_cfg.subset:
                    load_kwargs["name"] = dataset_cfg.subset
                
                try:
                    dataset = load_dataset(dataset_cfg.id, **load_kwargs)
                    
                    # Collect max_samples
                    samples_to_save = []
                    count = 0
                    
                    for sample in tqdm(dataset, desc=f"Streaming {dataset_cfg.name}", total=max_samples):
                        if count >= max_samples:
                            break
                        
                        text = sample.get(dataset_cfg.text_column or "text", "")
                        if not text:
                            continue
                            
                        # Save in format expected by load_dataset("json")
                        samples_to_save.append({dataset_cfg.text_column or "text": text})
                        count += 1
                        
                    # Save to JSON
                    self.logger.info(f"Saving {len(samples_to_save)} samples to {cache_file}...")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(samples_to_save, f)
                        
                except Exception as e:
                    self.logger.error(f"Failed to stream/cache dataset {dataset_cfg.id}: {e}")
                    return {"input_ids": torch.empty(0), "attention_mask": torch.empty(0)}

            # Now use ADL's loader on the cached JSON
            self.logger.info("Tokenizing cached data using ADL's load_and_tokenize_dataset()...")
            all_token_ids = load_and_tokenize_dataset(
                dataset_name="json", # Use generic json loader
                tokenizer=self.tokenizer,
                split="train", # JSON file loaded as 'train' split by default
                text_column=dataset_cfg.text_column or "text",
                n=max_tokens,
                max_samples=max_samples,
                data_files=[str(cache_file)], # Point to our cached file
            )

        if not all_token_ids:
            self.logger.warning(f"No samples found for {dataset_cfg.name}!")
            return {"input_ids": torch.empty(0), "attention_mask": torch.empty(0)}

        # Determine overall max length for padding
        max_len = max(len(ids) for ids in all_token_ids)
        input_ids_list = []
        attention_mask_list = []

        for token_ids in all_token_ids:
            # Pad
            padding_length = max_len - len(token_ids)
            padded_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = [1] * len(token_ids) + [0] * padding_length

            input_ids_list.append(padded_ids)
            attention_mask_list.append(mask)

        # Convert to tensors (on CPU initially)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        
        self.logger.info(f"Prepared tensors: input_ids {input_ids.shape}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    @torch.no_grad()
    def compute_stats_from_logits(
        self, 
        dataset_cfg: DatasetConfig,
        base_logits: torch.Tensor,
        finetuned_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Core analysis for one dataset.

        Args:
            dataset_cfg: Dataset configuration

        Returns:
            Dictionary containing occurrence rate statistics
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Processing dataset: {dataset_cfg.id}")
        self.logger.info(f"Dataset name: {dataset_cfg.name}")

        # Get parameters from config (no hardcoded values)
        batch_size = int(self.method_cfg.method_params.batch_size)
        max_tokens = int(self.method_cfg.method_params.max_tokens_per_sample)
        max_samples = int(self.method_cfg.method_params.max_samples)
        top_k = int(self.method_cfg.method_params.top_k)
        ignore_padding = bool(self.method_cfg.method_params.ignore_padding)

        self.logger.info(f"Parameters: batch_size={batch_size}, max_tokens={max_tokens}, top_k={top_k}, max_samples={max_samples}")
        self.logger.info(f"Dataset type: {'chat' if dataset_cfg.is_chat else 'text'}")

        # Track occurrences across all positions
        global_token_counts = defaultdict(lambda: {"count_positive": 0, "count_negative": 0})
        total_positions = 0

        # Per-token analysis: Initialize tracking structures if enabled
        per_token_enabled = False
        co_occurrence_enabled = False
        shortlist_token_ids = {}
        per_sample_counts = {}
        per_position_counts = {}
        shortlist_diffs = defaultdict(list)
        
        # Co-occurrence tracking
        same_point_matrix = defaultdict(lambda: defaultdict(int))
        # Track which tokens appeared in each sample (for Same-Sample co-occurrence)
        # Dict[sample_idx, Set[token_str]]
        sample_tokens_tracker = defaultdict(set)
        # Track which tokens appeared at each position (for Same-Position co-occurrence)
        # Dict[position_idx, Set[token_str]]
        position_tokens_tracker = defaultdict(set)
        
        # Positional KDE Analysis
        pos_kde_enabled = False
        pos_kde_num_positions = 0
        position_logit_diffs = defaultdict(list)
        
        if hasattr(self.method_cfg, 'positional_kde') and self.method_cfg.positional_kde.enabled:
            pos_kde_enabled = True
            pos_kde_num_positions = int(self.method_cfg.positional_kde.num_positions)
            self.logger.info(f"Positional KDE analysis enabled (plotting first {pos_kde_num_positions} positions)")
        
        # Global Token Statistics
        global_stats_enabled = False
        global_diff_sum = None
        global_pos_count = None
        
        if hasattr(self.method_cfg, 'global_token_statistics') and self.method_cfg.global_token_statistics.enabled:
            global_stats_enabled = True
            vocab_size = self.tokenizer.vocab_size
            # Depending on tokenizer, actual vocab size might be larger than vocab_size property if added tokens exist
            # We can get it from the model's embeddings or just use the diff shape later.
            # But let's init efficiently. We'll init on device inside loop if needed, or here.
            # To be safe regarding device and size, we can init on first batch or use len(tokenizer).
            # len(self.tokenizer) is usually safer for total vocab size including special tokens.
            real_vocab_size = len(self.tokenizer)
            self.logger.info(f"Global Token Statistics enabled (tracking {real_vocab_size} tokens)")
        
        # NMF Data Collection Structures
        nmf_enabled = self.nmf_cfg and self.nmf_cfg.enabled
        nmf_data = None
        if nmf_enabled:
            # We use parallel lists to construct a COO matrix later: (row_indices, col_indices, values)
            nmf_data = {
                "rows": [],
                "cols": [],
                "values": [],
                "valid_row_idx_counter": 0,
                "token_id_to_col_idx": {},  # Map raw token_id -> compact column index (0..M)
                "col_idx_to_token_id": [],  # Map compact column index -> raw token_id (for reverse lookup)
                "next_col_idx": 0
            }
            self.logger.info("Initializing NMF data collection...")
        
        if hasattr(self.method_cfg, 'per_token_analysis') and self.method_cfg.per_token_analysis.enabled:
            per_token_enabled = True
            self.logger.info("Per-token analysis enabled")
            
            # Check for co-occurrence config
            if hasattr(self.method_cfg.per_token_analysis, 'co_occurrence') and self.method_cfg.per_token_analysis.co_occurrence:
                co_occurrence_enabled = True
                self.logger.info("Co-occurrence analysis enabled")
            
            # Build token_id -> token_str mapping for shortlist
            for token_str in self.method_cfg.per_token_analysis.token_shortlist:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_ids) == 1:
                    shortlist_token_ids[token_ids[0]] = token_str
                    per_sample_counts[token_str] = defaultdict(int)
                    per_position_counts[token_str] = defaultdict(int)
                else:
                    self.logger.warning(
                        f"Token '{token_str}' encodes to {len(token_ids)} tokens, skipping. "
                        f"Use single-token strings only."
                    )
            
            self.logger.info(f"Tracking {len(shortlist_token_ids)} shortlist tokens: {list(shortlist_token_ids.values())}")

        # Tokenize entire dataset using ADL functions
        if dataset_cfg.is_chat:
            # Chat dataset - use ADL's chat function
            self.logger.info("Using ADL's load_and_tokenize_chat_dataset() with apply_chat_template()")
            samples = load_and_tokenize_chat_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                messages_column=dataset_cfg.messages_column or "messages",
                n=max_tokens,  # From config: max_tokens_per_sample
                pre_assistant_k=0,  # Don't need pre-assistant context for global analysis
                max_samples=max_samples,  # From config: max_samples
            )
            # Extract just the token IDs
            all_token_ids = [sample["input_ids"] for sample in samples]
        else:
            # Text dataset - use ADL's text function
            self.logger.info("Using ADL's load_and_tokenize_dataset()")
            all_token_ids = load_and_tokenize_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                text_column=dataset_cfg.text_column or "text",
                n=max_tokens,  # From config: max_tokens_per_sample
                max_samples=max_samples,  # From config: max_samples
                data_files=dataset_cfg.data_files,
            )

        # Now batch through token IDs
        num_samples = len(all_token_ids)
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.logger.info(f"Processing {num_samples} samples in {num_batches} batches...")
        
        # Track max sequence length across all batches for per-token analysis
        overall_max_len = 0

        for batch_idx in tqdm(range(num_batches), desc=f"Processing {dataset_cfg.name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Get batch of token IDs
            batch_token_ids = all_token_ids[start_idx:end_idx]

            # Pad to same length
            max_len = max(len(ids) for ids in batch_token_ids)
            overall_max_len = max(overall_max_len, max_len)
            input_ids_list = []
            attention_mask_list = []

            for token_ids in batch_token_ids:
                # Pad
                padding_length = max_len - len(token_ids)
                padded_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
                mask = [1] * len(token_ids) + [0] * padding_length

                input_ids_list.append(padded_ids)
                attention_mask_list.append(mask)

            # Convert to tensors
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

            # Get logits from both models (NO GRADIENTS, models already in eval mode)
            inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
            with self.base_model.trace(inputs):
                base_logits = self.base_model.logits.save()
            with self.finetuned_model.trace(inputs):
                finetuned_logits = self.finetuned_model.logits.save()

            # Extract logits [batch_size, seq_len, vocab_size]
            # Ensure same device
            target_device = base_logits.device
            finetuned_logits = finetuned_logits.to(target_device)
            attention_mask = attention_mask.to(target_device)

            # Compute difference: finetuned - base
            diff = finetuned_logits - base_logits  # [batch_size, seq_len, vocab_size]

            # Global Token Statistics (Entire Vocabulary)
            if global_stats_enabled:
                if global_diff_sum is None:
                    # Initialize on first use to ensure correct device and size
                    self.logger.info("  [Global Stats] Initializing accumulators and starting batch-wise accumulation...")
                    vocab_size = diff.shape[-1]
                    # Use float64 for better precision during large sum accumulation
                    global_diff_sum = torch.zeros(vocab_size, dtype=torch.float64, device=diff.device)
                    global_pos_count = torch.zeros(vocab_size, dtype=torch.float64, device=diff.device)
                
                # Apply attention mask to zero out padding
                # attention_mask: [batch, seq] -> [batch, seq, 1]
                mask_expanded = attention_mask.unsqueeze(-1).to(diff.dtype)
                
                # Sum logit diffs (masked)
                # OPTIMIZATION: In-place multiplication to save memory
                diff.mul_(mask_expanded)
                global_diff_sum += diff.sum(dim=(0, 1))
                
                # Count positive diffs (masked)
                # Note: (diff >= 0) is True for padding (0>=0), so we MUST AND with mask
                # OPTIMIZATION: Sum boolean tensor directly to avoid creating huge float tensor
                # We cast mask back to bool for logical AND
                pos_mask = (diff >= 0) & (mask_expanded.bool())
                global_pos_count += pos_mask.sum(dim=(0, 1))

            # Shortlist Distribution Tracking (Vectorized)
            if per_token_enabled and shortlist_token_ids:
                # We want to collect all logit diffs for the shortlist tokens
                # respecting the attention mask (if ignore_padding is True)
                valid_mask = attention_mask.bool()
                
                for s_token_id, s_token_str in shortlist_token_ids.items():
                    # diff[..., s_token_id]: [batch, seq]
                    token_vals = diff[:, :, s_token_id]
                    
                    if ignore_padding:
                        valid_vals = token_vals[valid_mask]
                    else:
                        valid_vals = token_vals.flatten()
                        
                    shortlist_diffs[s_token_str].extend(valid_vals.tolist())

            # Get top-K positive diffs (largest values)
            top_k_pos_values, top_k_pos_indices = torch.topk(
                diff, k=top_k, dim=-1, largest=True
            )  # [batch_size, seq_len, top_k]

            # Get top-K negative diffs (smallest values)
            top_k_neg_values, top_k_neg_indices = torch.topk(
                diff, k=top_k, dim=-1, largest=False
            )  # [batch_size, seq_len, top_k]

            # Track occurrences (respecting attention mask if ignore_padding=True)
            batch_size_actual, seq_len, _ = diff.shape

            for b in range(batch_size_actual):
                sample_idx = start_idx + b  # Global sample index
                
                for s in range(seq_len):
                    # Skip padding tokens if requested
                    if ignore_padding and attention_mask[b, s] == 0:
                        continue
                    
                    # Positional KDE Data Collection
                    if pos_kde_enabled and s < pos_kde_num_positions:
                        # top_k_pos_values[b, s] contains top-K positive logit diffs
                        # We extract them all
                        values = top_k_pos_values[b, s].tolist()
                        position_logit_diffs[s].extend(values)

                    # NMF Data Collection
                    if nmf_enabled:
                        # This sample/position corresponds to the next row in our matrix
                        current_row_idx = nmf_data["valid_row_idx_counter"]
                        nmf_data["valid_row_idx_counter"] += 1
                        
                        # Iterate through top-K positive tokens for this position
                        for k_idx, token_id in enumerate(top_k_pos_indices[b, s]):
                            token_id_item = token_id.item()
                            
                            # Determine value based on mode
                            if self.nmf_cfg.mode == "binary_occurrence":
                                val = 1.0
                            elif self.nmf_cfg.mode == "logit_diff_magnitude":
                                val = top_k_pos_values[b, s, k_idx].item()
                            else:
                                raise ValueError(f"Invalid NMF mode: {self.nmf_cfg.mode}")
                            
                            # Map token_id to column index
                            if token_id_item not in nmf_data["token_id_to_col_idx"]:
                                nmf_data["token_id_to_col_idx"][token_id_item] = nmf_data["next_col_idx"]
                                nmf_data["col_idx_to_token_id"].append(token_id_item)
                                nmf_data["next_col_idx"] += 1
                            
                            col_idx = nmf_data["token_id_to_col_idx"][token_id_item]
                            
                            # Add to sparse lists
                            nmf_data["rows"].append(current_row_idx)
                            nmf_data["cols"].append(col_idx)
                            nmf_data["values"].append(val)

                    # Track tokens at this specific point (sample, position) for co-occurrence
                    current_point_tokens = []

                    # Count positive occurrences
                    for token_id in top_k_pos_indices[b, s]:
                        token_id_item = token_id.item()
                        global_token_counts[token_id_item]["count_positive"] += 1
                        
                        # Per-token analysis: track shortlist tokens
                        if per_token_enabled and token_id_item in shortlist_token_ids:
                            token_str = shortlist_token_ids[token_id_item]
                            per_sample_counts[token_str][sample_idx] += 1
                            per_position_counts[token_str][s] += 1
                            
                            if co_occurrence_enabled:
                                current_point_tokens.append(token_str)
                                # Track for same-sample and same-position aggregation
                                sample_tokens_tracker[sample_idx].add(token_str)
                                position_tokens_tracker[s].add(token_str)

                    # Update Same-Point co-occurrence matrix (Positive Top-K only)
                    if co_occurrence_enabled and current_point_tokens:
                        # Optimization: Use combinations to halve iterations (symmetric matrix)
                        for t1, t2 in combinations_with_replacement(current_point_tokens, 2):
                            same_point_matrix[t1][t2] += 1
                            if t1 != t2:
                                same_point_matrix[t2][t1] += 1

                    # Count negative occurrences
                    for token_id in top_k_neg_indices[b, s]:
                        token_id_item = token_id.item()
                        global_token_counts[token_id_item]["count_negative"] += 1
                        
                        # Per-token analysis: track shortlist tokens in negative direction too
                        # (Note: we track both positive and negative, aggregating total occurrences)
                        if per_token_enabled and token_id_item in shortlist_token_ids:
                            token_str = shortlist_token_ids[token_id_item]
                            per_sample_counts[token_str][sample_idx] += 1
                            per_position_counts[token_str][s] += 1
                            # NOTE: We do NOT track negative tokens for co-occurrence as per user request

                    total_positions += 1

        self.logger.info(f"✓ Batch processing complete!")
        self.logger.info(
            f"Processed {total_positions:,} positions with {len(global_token_counts):,} unique tokens"
        )

        # Compute co-occurrence matrices if enabled
        same_sample_matrix = defaultdict(lambda: defaultdict(int))
        same_position_matrix = defaultdict(lambda: defaultdict(int))
        
        if co_occurrence_enabled:
            self.logger.info("Computing Same-Sample co-occurrence matrix...")
            for sample_idx, tokens in sample_tokens_tracker.items():
                if not tokens:
                    continue
                # Optimization: Use combinations to halve iterations
                for t1, t2 in combinations_with_replacement(tokens, 2):
                    same_sample_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_sample_matrix[t2][t1] += 1
                        
            self.logger.info("Computing Same-Position co-occurrence matrix...")
            for pos_idx, tokens in position_tokens_tracker.items():
                if not tokens:
                    continue
                # Optimization: Use combinations to halve iterations
                for t1, t2 in combinations_with_replacement(tokens, 2):
                    same_position_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_position_matrix[t2][t1] += 1

        # Run NMF Clustering if enabled
        nmf_results = None
        if nmf_enabled and nmf_data["rows"]:
            self.logger.info("Running NMF Clustering on collected data...")
            nmf_results = self.run_nmf_clustering(dataset_cfg.name, nmf_data)

        # Generate Positional KDE Plots if enabled
        if pos_kde_enabled:
            self.logger.info("Generating positional KDE plots...")
            plot_positional_kde(
                position_logit_diffs,
                dataset_cfg.name,
                self.results_dir,
                pos_kde_num_positions,
                num_samples,
                top_k
            )

        # Global Token Statistics Saving
        if global_stats_enabled and global_diff_sum is not None:
            self.logger.info("Saving global token statistics (entire vocabulary)...")
            self._save_global_token_statistics(
                dataset_cfg.name,
                global_diff_sum,
                global_pos_count,
                num_samples,
                total_positions
            )
            
            # Generate scatter plot
            self.logger.info("Generating global token scatter plot...")
            json_path = self.results_dir / f"{dataset_cfg.name}_global_token_stats.json"
            
            plot_global_token_scatter(
                json_path, 
                self.results_dir, 
                tokenizer=self.tokenizer,
                top_k_labels=int(self.method_cfg.global_token_statistics.top_k_plotting_labels)
            )

        # Compute occurrence rates
        self.logger.info(f"Computing occurrence rates...")
        all_tokens = []
        for token_id, counts in global_token_counts.items():
            token_str = self.tokenizer.decode([token_id])
            all_tokens.append({
                "token_id": token_id,
                "token_str": token_str,
                "count_positive": counts["count_positive"],
                "count_negative": counts["count_negative"],
                "positive_occurrence_rate": (counts["count_positive"] / total_positions) * 100
                if total_positions > 0
                else 0.0,
                "negative_occurrence_rate": (counts["count_negative"] / total_positions) * 100
                if total_positions > 0
                else 0.0,
            })

        # Sort by positive and negative occurrence rates (DIRECT sort, no union)
        pos_rates = torch.tensor([t["positive_occurrence_rate"] for t in all_tokens])
        neg_rates = torch.tensor([t["negative_occurrence_rate"] for t in all_tokens])

        # Get top tokens for each
        # Save max(num_tokens_to_plot, top_k) to avoid needing to rerun if visualization config changes
        num_tokens_to_save = max(
            int(self.method_cfg.visualization.num_tokens_to_plot),
            int(top_k)
        )
        k_pos = min(num_tokens_to_save, len(all_tokens))
        k_neg = min(num_tokens_to_save, len(all_tokens))

        top_k_pos_values, top_k_pos_indices = torch.topk(pos_rates, k=k_pos, largest=True)
        top_k_neg_values, top_k_neg_indices = torch.topk(neg_rates, k=k_neg, largest=True)

        top_positive = [all_tokens[i] for i in top_k_pos_indices.tolist()]
        top_negative = [all_tokens[i] for i in top_k_neg_indices.tolist()]

        self.logger.info(f"✓ Top tokens computed:")
        self.logger.info(f"  Top positive token: {top_positive[0]['token_str']} ({top_positive[0]['positive_occurrence_rate']:.2f}%)")
        self.logger.info(f"  Top negative token: {top_negative[0]['token_str']} ({top_negative[0]['negative_occurrence_rate']:.2f}%)")

        # Create results dictionary
        results = {
            "dataset_id": dataset_cfg.id,
            "dataset_name": dataset_cfg.name,
            "total_positions": total_positions,
            "num_samples": num_samples,
            "top_k": top_k,
            "unique_tokens": len(global_token_counts),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "metadata": {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
                "max_tokens_per_sample": max_tokens,
                "batch_size": batch_size,
            },
        }

        # Add NMF results if available (saved to main results for reference, but full details in separate file)
        if nmf_results:
            # We store a summary or link here if needed, but the main NMF output is separate.
            # Let's just note it ran.
            results["metadata"]["nmf_clustering_run"] = True
            # We can also attach the full topics structure if we want it in the main JSON too, 
            # but usually better to keep modular. Let's stick to the separate file plan.
        
        # Add per-token analysis data if enabled (for internal use, not saved to main JSON)
        if per_token_enabled:
            results["_per_token_data"] = {
                "per_sample_counts": per_sample_counts,
                "per_position_counts": per_position_counts,
                "max_positions": overall_max_len,
            }
            if co_occurrence_enabled:
                results["_per_token_data"]["co_occurrence"] = {
                    "same_sample": same_sample_matrix,
                    "same_position": same_position_matrix,
                    "same_point": same_point_matrix,
                }
            
            results["_per_token_data"]["shortlist_distributions"] = shortlist_diffs

        return results

    def _save_global_token_statistics(
        self,
        dataset_name: str,
        global_diff_sum: torch.Tensor,
        global_pos_count: torch.Tensor,
        num_samples: int,
        total_positions: int
    ) -> None:
        """
        Save global token statistics (sum logit diff, positive count) for the entire vocabulary.
        
        Args:
            dataset_name: Name of the dataset
            global_diff_sum: Tensor of shape [vocab_size] containing sum of logit diffs
            global_pos_count: Tensor of shape [vocab_size] containing count of non-negative diffs
            num_samples: Number of samples processed
            total_positions: Total number of valid token positions processed
        """
        output_file = self.results_dir / f"{dataset_name}_global_token_stats.json"
        
        # Ensure CPU and python native types
        global_diff_sum = global_diff_sum.cpu()
        global_pos_count = global_pos_count.cpu()
        vocab_size = len(global_diff_sum)
        
        self.logger.info(f"Formatting global statistics for {vocab_size} tokens...")
        
        # Get string representations for all tokens
        # We assume the model's output vocab corresponds to tokenizer IDs 0..N-1
        all_ids = list(range(vocab_size))
        all_tokens = self.tokenizer.convert_ids_to_tokens(all_ids)
        
        # Build the list of stats
        stats_list = []
        for token_id, token_str in enumerate(all_tokens):
            stats_list.append({
                "token": token_str,
                "token_id": token_id,
                "sum_logit_diff": float(global_diff_sum[token_id]),
                "count_nonnegative": int(global_pos_count[token_id])
            })
            
        final_output = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
            "total_positions_analyzed": total_positions,
            "num_unique_tokens": vocab_size,
            "global_token_stats": stats_list
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved global token statistics to {output_file}")

    def run_nmf_clustering(self, dataset_name: str, nmf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run NMF clustering on the collected sparse data.
        
        Args:
            dataset_name: Name of the dataset
            nmf_data: Dictionary containing rows, cols, values, etc.
            
        Returns:
            Dictionary containing NMF topics and weights
        """
        # 1. Construct Sparse Matrix
        num_rows = nmf_data["valid_row_idx_counter"]
        num_cols = nmf_data["next_col_idx"]
        
        if num_rows == 0 or num_cols == 0:
            self.logger.warning("No data collected for NMF (empty matrix). Skipping.")
            return None
            
        self.logger.info(f"Constructing NMF matrix: {num_rows} positions x {num_cols} unique tokens")
        
        # Create Scipy COO matrix first
        rows = nmf_data["rows"]
        cols = nmf_data["cols"]
        values = nmf_data["values"]
        
        # Convert to dense Torch tensor (torchnmf expects dense or sparse, but sparse support in fit is specific)
        # Given the dimensionality (e.g. 10k rows x 1k cols), dense is usually fine for GPU/CPU memory
        # If it gets too big, we might need sparse, but torchnmf's sparse support has caveats.
        # Let's try constructing a dense tensor directly if it fits.
        # Size: 10,000 * 5,000 * 4 bytes ~= 200MB. Totally fine.
        
        # Use scipy to handle the sparse-to-dense conversion efficiently
        V_sparse = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols))
        
        # Convert to torch tensor
        # Note: torchnmf expects non-negative input. Our values are 1.0 or magnitude (usually positive).
        # We ensure positivity.
        V_dense = torch.tensor(V_sparse.todense(), dtype=torch.float32)
        V_dense = torch.relu(V_dense) # Ensure non-negative
        
        if torch.cuda.is_available():
            V_dense = V_dense.cuda()
            
        # 2. Run NMF
        num_topics = int(self.nmf_cfg.num_topics)
        beta = float(self.nmf_cfg.beta)
        self.logger.info(f"Running NMF with {num_topics} topics (beta={beta})...")
        
        nmf = NMF(V_dense.shape, rank=num_topics)
        if torch.cuda.is_available():
            nmf = nmf.cuda()
            
        # Fit
        # NMF requires gradients for its update steps (it uses autograd for multiplicative updates),
        # but the surrounding method has @torch.no_grad(). We must re-enable gradients here.
        with torch.enable_grad():
            nmf.fit(V_dense, beta=beta, verbose=False, max_iter=200)
        
        # 3. Extract Topics
        #
        # In torchnmf: V~H*W^T
        # V: (N,C)
        # W: (C,R)
        # H: (N,R)
        #
        # V = torch.rand(20, 30)
        # m = NMF(V.shape, 5)
        # m.W.size()
        # torch.Size([30, 5])
        # m.H.size()
        # torch.Size([20, 5])
        # HWt = m()
        # HWt.size()
        # torch.Size([20, 30])
        #
        # V = H * W^T
        # (N,C) = (N,R) * (R,C)
        # (20 x 30) = (20 x 5) * (5 x 30)
        #
        # https://pytorch-nmf.readthedocs.io/en/stable/modules/nmf.html#torchnmf.nmf.NMF
        
        W_nmf = nmf.W.detach().cpu()
        H_nmf = nmf.H.detach().cpu()
        # print('W_nmf shape: ', W_nmf.shape)
        # print('H_nmf shape: ', H_nmf.shape)
        # print('V_dense shape: ', V_dense.shape)
        # print('')

        topic_token_matrix = nmf.W.T.detach().cpu()
        # print('topic_token_matrix shape: ', topic_token_matrix.shape)

            
        # 4. Process Results
        topics_output = []
        col_idx_to_token_id = nmf_data["col_idx_to_token_id"]
        
        self.logger.info(f"=== NMF Topic Clustering Results ({num_topics} topics) ===")
        
        for k in range(num_topics):
            # Get weights for this topic
            topic_weights = topic_token_matrix[k] # (num_tokens,)
            
            # Get top indices
            top_indices = torch.argsort(topic_weights, descending=True)[:20] # Top 20
            
            top_tokens_list = []
            log_str_parts = []
            
            for idx in top_indices:
                idx_val = idx.item()
                weight = topic_weights[idx_val].item()
                
                if weight < 1e-6: # Skip negligible weights
                    continue
                    
                token_id = col_idx_to_token_id[idx_val]
                token_str = self.tokenizer.decode([token_id])
                
                top_tokens_list.append({
                    "token": token_str,
                    "weight": weight,
                    "token_id": token_id
                })
                
                if len(log_str_parts) < 10: # Only log top 10
                    log_str_parts.append(f"{token_str} ({weight:.2f})")
            
            topics_output.append({
                "topic_id": k,
                "top_tokens": top_tokens_list
            })
            
            self.logger.info(f"Topic {k}: {', '.join(log_str_parts)}")
            
        # 5. Save Results
        final_output = {
            "dataset_name": dataset_name,
            "num_topics": num_topics,
            "num_unique_tokens": num_cols,
            "topics": topics_output
        }
        
        output_file = self.results_dir / f"{dataset_name}_nmf_topics_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved NMF topics to {output_file}")
        
        return final_output

    def save_results(self, dataset_name: str, results: Dict[str, Any]) -> Path:
        """
        Save results for a dataset to disk.

        Args:
            dataset_name: Dataset name (safe for filename)
            results: Results dictionary

        Returns:
            Path to saved file
        """
        output_file = self.results_dir / f"{dataset_name}_occurrence_rates.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved results for {dataset_name} to {output_file}")
        return output_file

    def _save_and_plot_per_token_analysis(
        self,
        dataset_name: str,
        per_sample_counts: Dict[str, Dict[int, int]],
        per_position_counts: Dict[str, Dict[int, int]],
        num_samples: int,
        max_positions: int,
        shortlist_diffs: Dict[str, List[float]] = None
    ) -> None:
        """
        Save per-token analysis data and generate plots.
        
        Saves two JSON files (per-sample and per-position counts) and generates
        corresponding plots for each token in the shortlist.
        
        Directory structure:
            {results_dir}/per_token_analysis/
                ├── data/       (JSON files)
                └── plots/      (PNG files)
        
        Args:
            dataset_name: Dataset name for filenames
            per_sample_counts: Dict[token_str][sample_idx] = count
            per_position_counts: Dict[token_str][position_idx] = count
            num_samples: Total number of samples processed
            max_positions: Maximum sequence length
            shortlist_diffs: Dict mapping token_str to list of logit diff values
        """
        self.logger.info(f"Saving per-token analysis for {dataset_name}...")
        
        # Create subdirectories for organized output
        per_token_dir = self.results_dir / "per_token_analysis"
        data_dir = per_token_dir / "data"
        plots_dir = per_token_dir / "plots"
        
        # Subdirectories for plots
        sample_plots_dir = plots_dir / "per_token_by_sample"
        position_plots_dir = plots_dir / "per_token_by_position"
        dist_plots_dir = plots_dir / "logit_diff_distributions"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        sample_plots_dir.mkdir(parents=True, exist_ok=True)
        position_plots_dir.mkdir(parents=True, exist_ok=True)
        dist_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts and then to lists for JSON serialization
        per_sample_serializable = {}
        for token_str, sample_dict in per_sample_counts.items():
            # Convert to dense list for easier consumption
            per_sample_serializable[token_str] = [
                sample_dict.get(i, 0) for i in range(num_samples)
            ]
        
        per_position_serializable = {}
        for token_str, position_dict in per_position_counts.items():
            # Convert to dense list
            per_position_serializable[token_str] = [
                position_dict.get(i, 0) for i in range(max_positions)
            ]
        
        # Save per-sample counts to data/ subdirectory
        per_sample_data = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
            "tokens": per_sample_serializable
        }
        per_sample_file = data_dir / f"per_token_analysis_by_sample_{dataset_name}.json"
        with open(per_sample_file, "w", encoding="utf-8") as f:
            json.dump(per_sample_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"  Saved per-sample counts: data/{per_sample_file.name}")
        
        # Save per-position counts to data/ subdirectory
        per_position_data = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
            "max_positions": max_positions,
            "tokens": per_position_serializable
        }
        per_position_file = data_dir / f"per_token_analysis_by_position_{dataset_name}.json"
        with open(per_position_file, "w", encoding="utf-8") as f:
            json.dump(per_position_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"  Saved per-position counts: data/{per_position_file.name}")
        
        # Generate plots in plots/ subdirectory
        self.logger.info(f"Generating per-token plots...")
        
        # Plot per-sample occurrences (with max_positions in title)
        sample_plots = plot_per_sample_occurrences(
            per_sample_counts,
            dataset_name,
            sample_plots_dir,
            num_samples,
            max_positions=max_positions
        )
        
        # Plot per-position occurrences (with num_samples in title)
        position_plots = plot_per_position_occurrences(
            per_position_counts,
            dataset_name,
            position_plots_dir,
            max_positions,
            num_samples=num_samples
        )
        
        # Plot logit diff distributions
        if shortlist_diffs:
            self.logger.info("Generating logit diff distribution plots...")
            for token_str, diffs in shortlist_diffs.items():
                if diffs:
                    plot_shortlist_token_distribution(
                        diffs,
                        token_str,
                        dataset_name,
                        dist_plots_dir,
                        num_samples=num_samples,
                        max_tokens_per_sample=max_positions,
                        total_positions=len(diffs)
                    )
        
        total_plots = len(sample_plots) + len(position_plots)
        self.logger.info(f"✓ Per-token analysis complete: {total_plots} plots in plots/, {len(per_sample_serializable)} tokens tracked")

    def _save_and_plot_co_occurrence(
        self,
        dataset_name: str,
        co_occurrence_data: Dict[str, Dict[str, Dict[str, int]]]
    ) -> None:
        """
        Save and plot co-occurrence matrices.
        
        Args:
            dataset_name: Dataset name
            co_occurrence_data: Dict mapping type -> matrix
        """
        self.logger.info(f"Saving and plotting co-occurrence data for {dataset_name}...")
        
        per_token_dir = self.results_dir / "per_token_analysis"
        data_dir = per_token_dir / "data"
        plots_dir = per_token_dir / "plots"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for co_type, matrix in co_occurrence_data.items():
            # Convert defaultdict to regular dict for JSON
            serializable_matrix = {k: dict(v) for k, v in matrix.items()}
            
            # Save JSON
            json_file = data_dir / f"co_occurrence_{co_type}_{dataset_name}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(serializable_matrix, f, indent=2, ensure_ascii=False)
            self.logger.info(f"  Saved co-occurrence data: {json_file.name}")
            
            # Plot
            plot_co_occurrence_heatmap(
                serializable_matrix,
                dataset_name,
                plots_dir,
                co_type
            )
        
        self.logger.info(f"✓ Co-occurrence analysis complete for {dataset_name}")

    def run_token_relevance(self) -> None:
        """Grade top-K positive tokens for relevance to finetuning domain."""
        cfg = self.method_cfg.token_relevance
        
        if not cfg.enabled:
            return
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("RUNNING TOKEN RELEVANCE GRADING")
        logger.info("=" * 80)
        
        overwrite = bool(cfg.overwrite)
        agreement_mode = str(cfg.agreement)
        
        # Get organism description
        organism_cfg = self.cfg.organism
        assert hasattr(organism_cfg, "description_long"), (
            "Organism config must have 'description_long' for token relevance grading"
        )
        description = str(organism_cfg.description_long)
        logger.info(f"Using organism description: {description[:100]}...")
        
        # Initialize grader
        grader_cfg = cfg.grader
        grader = TokenRelevanceGrader(
            grader_model_id=str(grader_cfg.model_id),
            base_url=str(grader_cfg.base_url),
            api_key_path=str(grader_cfg.api_key_path),
        )
        logger.info(f"Initialized grader: {grader_cfg.model_id}")
        
        # Compute frequent tokens from training dataset (if available)
        freq_cfg = cfg.frequent_tokens
        has_training_dataset = hasattr(organism_cfg, "training_dataset") and (
            organism_cfg.training_dataset is not None
        )
        
        if has_training_dataset:
            # Extract dataset ID from training_dataset config
            training_dataset_id = str(organism_cfg.training_dataset.id)
            logger.info(f"Computing frequent tokens from training dataset: {training_dataset_id}")
            
            # Read is_chat from config
            training_is_chat = False
            training_dataset_cfg = organism_cfg.training_dataset
            if hasattr(training_dataset_cfg, 'is_chat'):
                training_is_chat = bool(training_dataset_cfg.is_chat)
            elif isinstance(training_dataset_cfg, dict):
                training_is_chat = bool(training_dataset_cfg.get('is_chat', False))
            
            logger.info(f"Training dataset is_chat: {training_is_chat}")
            
            frequent_tokens = _compute_frequent_tokens(
                dataset_name=training_dataset_id,
                tokenizer=self.tokenizer,
                splits=["train"],
                num_tokens=int(freq_cfg.num_tokens),
                min_count=int(freq_cfg.min_count),
                is_chat=training_is_chat,  # Use actual value from config
                subset=None,
            )
            logger.info(f"Found {len(frequent_tokens)} frequent tokens")
        else:
            frequent_tokens = []
            logger.info("No training dataset available, using empty frequent tokens list")
        
        # Process each dataset
        for dataset_cfg in self.datasets:
            dataset_name = dataset_cfg.name
            results_file = self.results_dir / f"{dataset_name}_occurrence_rates.json"
            
            if not results_file.exists():
                logger.warning(f"No results found for {dataset_name}, skipping token relevance")
                continue
            
            logger.info("")
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Load saved occurrence results
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Apply normalization if enabled
            use_normalized = bool(self.method_cfg.get("use_normalized_tokens", False))
            top_positive = results["top_positive"]
            if use_normalized:
                total_positions = results.get("total_positions", 0)
                top_positive = normalize_token_list(top_positive, total_positions)
                logger.info(f"Applied token normalization for {dataset_name}: {len(results['top_positive'])} -> {len(top_positive)} tokens")
            
            # Output directory (matches ADL structure with layer_global/position_all)
            dataset_dir_name = dataset_cfg.id.split("/")[-1]
            out_dir = (
                self.results_dir
                / "layer_global"
                / dataset_dir_name
                / "token_relevance"
                / "position_all"
                / "difference"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            rel_path = out_dir / "relevance_logit_diff.json"
            
            # Skip if exists and not overwriting
            if (not overwrite) and rel_path.exists():
                logger.info(f"Token relevance exists for {dataset_name}, skipping (overwrite=False)")
                continue
            
            # Extract top-K positive tokens and their occurrence rates
            k_tokens = min(int(cfg.k_candidate_tokens), len(top_positive))
            candidate_tokens = [t["token_str"] for t in top_positive[:k_tokens]]
            token_weights = [t["positive_occurrence_rate"] / 100.0 for t in top_positive[:k_tokens]]
            
            logger.info(f"Grading {k_tokens} top positive tokens for {dataset_name}")
            logger.info(f"Top 5 tokens: {candidate_tokens[:5]}")
            
            # Compute trivial baseline
            trivial_hits = sum(1 for t in candidate_tokens if t in frequent_tokens)
            trivial_percentage = trivial_hits / float(len(candidate_tokens)) if candidate_tokens else 0.0
            
            # Grade with permutation robustness
            permutations = int(grader_cfg.permutations)
            logger.info(f"Running grader with {permutations} permutations...")
            majority_labels, permutation_labels, raw_responses = grader.grade(
                description=description,
                frequent_tokens=frequent_tokens,
                candidate_tokens=candidate_tokens,
                permutations=permutations,
                concurrent=True,
                max_tokens=int(grader_cfg.max_tokens),
            )
            
            # Aggregate labels based on agreement mode
            if agreement_mode == "majority":
                final_labels = majority_labels
            else:  # "all"
                n = len(candidate_tokens)
                final_labels = [
                    "RELEVANT" if all(run[i] == "RELEVANT" for run in permutation_labels)
                    else "IRRELEVANT"
                    for i in range(n)
                ]
            
            # Compute percentage metrics
            relevant_fraction = sum(lbl == "RELEVANT" for lbl in final_labels) / float(len(final_labels))
            
            # Weighted percentage using occurrence rates
            total_w = sum(token_weights)
            relevant_w = sum(w for lbl, w in zip(final_labels, token_weights) if lbl == "RELEVANT")
            weighted_percentage = relevant_w / total_w if total_w > 0 else 0.0
            
            # Create output record (matches ADL format)
            rec = {
                "layer": "global",
                "position": "all",
                "variant": "difference",
                "source": "logit_diff",
                "target": "self",
                "labels": final_labels,
                "tokens": candidate_tokens,
                "percentage": relevant_fraction,
                "trivial_percentage": trivial_percentage,
                "weighted_percentage": float(weighted_percentage),
                "grader_responses": raw_responses,
            }
            
            # Save results
            rel_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            logger.info(f"✓ Token relevance saved: {rel_path}")
            logger.info(f"  Relevance: {relevant_fraction*100:.1f}%, Weighted: {weighted_percentage*100:.1f}%")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Token relevance grading completed!")
        logger.info("=" * 80)

    def run(self) -> None:
        """
        Main execution method for logit diff top-K occurring analysis.

        Processes each dataset sequentially with memory-efficient loading:
        1. Prepare inputs for all datasets (tokenization)
        2. Load Base Model -> Compute logits for all datasets -> Clear Base Model
        3. Load Finetuned Model -> Compute logits for all datasets -> Clear Finetuned Model
        4. Analyze Differences (using cached logits)
        """
        self.logger.info("=" * 80)
        self.logger.info("LOGIT DIFF TOP-K OCCURRING ANALYSIS (Sequential Loading)")
        self.logger.info("=" * 80)

        # Check if results already exist
        if not self.method_cfg.overwrite:
            existing_results = list(self.results_dir.glob("*_occurrence_rates.json"))
            if len(existing_results) >= len(self.datasets):
                self.logger.info(
                    f"Results already exist in {self.results_dir}. Skipping computation."
                )
                return

        # Phase 0: Data Preparation (Tokenize all datasets)
        self.logger.info("PHASE 0: Data Preparation")
        
        # Setup logits output directory
        logits_dir = self.results_dir / "logits"
        logits_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for dataset_cfg in self.datasets:
            dataset_inputs[dataset_cfg.id] = self._prepare_dataset_tensors(dataset_cfg)
            
            # Save attention mask
            mask_path = logits_dir / f"{dataset_cfg.name}_attention_mask.pt"
            torch.save(dataset_inputs[dataset_cfg.id]["attention_mask"], mask_path)
            self.logger.info(f"Saved attention mask to {mask_path}")

        # Phase 1: Base Model Inference
        self.logger.info("")
        self.logger.info("PHASE 1: Base Model Inference")
        self.logger.info(f"Loading base model: {self.base_model_cfg.model_id}")
        _ = self.base_model # Trigger load
        
        base_logits_map: Dict[str, torch.Tensor] = {}
        batch_size = int(self.method_cfg.method_params.batch_size)

        for dataset_cfg in self.datasets:
            self.logger.info(f"Computing base logits for {dataset_cfg.name}...")
            inputs = dataset_inputs[dataset_cfg.id]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            if input_ids.numel() == 0:
                continue

            num_samples = input_ids.shape[0]
            dataset_logits = []
            
            # Process in batches to avoid VRAM OOM
            with torch.no_grad():
                for i in tqdm(range(0, num_samples, batch_size), desc="Base Model Inference"):
                    batch_input = input_ids[i : i + batch_size].to(self.device)
                    batch_mask = attention_mask[i : i + batch_size].to(self.device)
                    
                    with self.base_model.trace(batch_input, attention_mask=batch_mask):
                        logits = self.base_model.logits.save()
                    
                    dataset_logits.append(logits.cpu())
            
            if dataset_logits:
                all_logits = torch.cat(dataset_logits, dim=0)
                base_logits_map[dataset_cfg.id] = all_logits
                
                # Save logits to disk
                logits_path = logits_dir / f"{dataset_cfg.name}_base_logits.pt"
                torch.save(all_logits, logits_path)
                self.logger.info(f"Saved base logits to {logits_path}")

        self.clear_base_model()

        # Phase 2: Finetuned Model Inference
        self.logger.info("")
        self.logger.info("PHASE 2: Finetuned Model Inference")
        self.logger.info(f"Loading finetuned model: {self.finetuned_model_cfg.model_id}")
        _ = self.finetuned_model # Trigger load
        
        finetuned_logits_map: Dict[str, torch.Tensor] = {}

        for dataset_cfg in self.datasets:
            self.logger.info(f"Computing finetuned logits for {dataset_cfg.name}...")
            inputs = dataset_inputs[dataset_cfg.id]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            if input_ids.numel() == 0:
                continue

            num_samples = input_ids.shape[0]
            dataset_logits = []
            
            with torch.no_grad():
                for i in tqdm(range(0, num_samples, batch_size), desc="Finetuned Model Inference"):
                    batch_input = input_ids[i : i + batch_size].to(self.device)
                    batch_mask = attention_mask[i : i + batch_size].to(self.device)
                    
                    with self.finetuned_model.trace(batch_input, attention_mask=batch_mask):
                        logits = self.finetuned_model.logits.save()
                    
                    dataset_logits.append(logits.cpu())
            
            if dataset_logits:
                all_logits = torch.cat(dataset_logits, dim=0)
                finetuned_logits_map[dataset_cfg.id] = all_logits
                
                # Save logits to disk
                logits_path = logits_dir / f"{dataset_cfg.name}_finetuned_logits.pt"
                torch.save(all_logits, logits_path)
                self.logger.info(f"Saved finetuned logits to {logits_path}")

        self.clear_finetuned_model()

        # Phase 3: Analysis
        self.logger.info("")
        self.logger.info("PHASE 3: Analysis & Diffing")
        
        for idx, dataset_cfg in enumerate(self.datasets, 1):
            dataset_id = dataset_cfg.id
            if dataset_id not in base_logits_map or dataset_id not in finetuned_logits_map:
                self.logger.warning(f"Skipping {dataset_cfg.name} due to missing logits.")
                continue

            self.logger.info("")
            self.logger.info(f"[{idx}/{len(self.datasets)}] Analyzing dataset: {dataset_cfg.name}")
            
            results = self.compute_stats_from_logits(
                dataset_cfg=dataset_cfg,
                base_logits=base_logits_map[dataset_id],
                finetuned_logits=finetuned_logits_map[dataset_id],
                input_ids=dataset_inputs[dataset_id]["input_ids"],
                attention_mask=dataset_inputs[dataset_id]["attention_mask"]
            )

            if results is not None:
                # Save results to disk
                self.save_results(dataset_cfg.name, results)
                
                # Save and plot per-token analysis if enabled
                if "_per_token_data" in results:
                    shortlist_diffs = results["_per_token_data"].pop("shortlist_distributions", None)
                    
                    self._save_and_plot_per_token_analysis(
                        dataset_cfg.name,
                        results["_per_token_data"]["per_sample_counts"],
                        results["_per_token_data"]["per_position_counts"],
                        results["num_samples"],
                        results["_per_token_data"]["max_positions"],
                        shortlist_diffs=shortlist_diffs
                    )
                    
                    if "co_occurrence" in results["_per_token_data"]:
                        self._save_and_plot_co_occurrence(
                            dataset_cfg.name,
                            results["_per_token_data"]["co_occurrence"]
                        )
                
                self.logger.info(f"✓ [{idx}/{len(self.datasets)}] Completed dataset: {dataset_cfg.name}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("✓ Logit diff top-K occurring analysis completed successfully!")
        self.logger.info(f"✓ Results saved to: {self.results_dir}")
        self.logger.info("=" * 80)
        
        # Run token relevance grading if enabled
        if hasattr(self.method_cfg, 'token_relevance') and self.method_cfg.token_relevance.enabled:
            self.run_token_relevance()

    def visualize(self) -> None:
        """
        Create Streamlit visualization for logit diff top-K occurring results.

        Returns:
            Streamlit component displaying occurrence rankings and interactive heatmap
        """
        visualize(self)

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available logit diff top-K occurring results.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = defaultdict(dict)
        results_base = results_dir

        if not results_base.exists():
            return results

        # Scan for results in the expected structure
        for base_model_dir in results_base.iterdir():
            if not base_model_dir.is_dir():
                continue

            model_name = base_model_dir.name

            for organism_dir in base_model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                method_dir = organism_dir / "logit_diff_topk_occurring"
                if method_dir.exists() and list(method_dir.glob("*_occurrence_rates.json")):
                    results[model_name][organism_name] = str(method_dir)

        return results

    def get_agent(self) -> DiffingMethodAgent:
        from .agents import LogitDiffAgent
        return LogitDiffAgent(cfg=self.cfg)
