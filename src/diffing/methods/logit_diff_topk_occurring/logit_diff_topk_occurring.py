"""
Logit Diff Top-K Occurring analysis method.

This module computes occurrence rates of tokens in the top-K positive and negative
logit differences between a base model and a finetuned model.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import gc
from omegaconf import DictConfig
from loguru import logger
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, IterableDataset
from datetime import datetime

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
from .normalization import process_token_list, normalize_token_list, load_fraction_positive_tokens
from .ui import visualize
from .plots import (
    plot_occurrence_bar_chart,
    plot_per_sample_occurrences,
    plot_per_position_occurrences,
    plot_shortlist_token_distribution,
    plot_shortlist_token_distribution_by_position,
    plot_shortlist_token_distribution_by_sample,
    plot_co_occurrence_heatmap,
    plot_positional_kde,
    plot_global_token_scatter,
    get_global_token_scatter_plotly,
    plot_selected_tokens_table,
)
from itertools import combinations_with_replacement
import scipy.sparse
from torchnmf.nmf import NMF
from .orthogonal_nmf import fit_nmf_orthogonal


def slice_to_positions(tensor: torch.Tensor, positions_list: List[List[int]]) -> torch.Tensor:
    """
    Extract specific positions from each sample in a tensor.
    
    Used to slice logit diffs to only the relevant positions around assistant start
    for chat datasets, matching ADL's pre_assistant_k behavior.
    
    Args:
        tensor: [batch, seq_len, vocab] tensor of logit diffs
        positions_list: List of position indices per sample (from chat dataset)
    
    Returns:
        Tensor of shape [batch, max_positions, vocab] with only the relevant positions
    """
    batch_size = tensor.shape[0]
    max_pos = max(len(p) for p in positions_list)
    vocab_size = tensor.shape[-1]
    
    result = torch.zeros(batch_size, max_pos, vocab_size, dtype=tensor.dtype, device=tensor.device)
    for i, positions in enumerate(positions_list):
        for j, pos in enumerate(positions):
            #if pos < tensor.shape[1]:  # Safety check
                #result[i, j] = tensor[i, pos]
            result[i, j] = tensor[i, pos]
    return result


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
        chat_variants = list(getattr(self.method_cfg.datasets, "chat_dataset_variants", ["default"]))

        # Get dataset configurations
        self.datasets = get_dataset_configurations(
            cfg,
            use_chat_dataset=self.method_cfg.datasets.use_chat_dataset,
            use_pretraining_dataset=self.method_cfg.datasets.use_pretraining_dataset,
            use_training_dataset=self.method_cfg.datasets.use_training_dataset,
            pretraining_dataset_variants=pretraining_variants,
            chat_dataset_variants=chat_variants,
        )

        # Filter out validation datasets (only use train-like splits)
        #self.datasets = [ds for ds in self.datasets if ds.split.startswith("train")]

        # NMF Clustering configuration
        self.nmf_cfg = getattr(self.method_cfg, "token_topic_clustering_NMF", None)
        if self.nmf_cfg and self.nmf_cfg.enabled:
            self.logger.info("NMF Token Topic Clustering enabled.")

        # Setup results directory
        organism_path_name = cfg.organism.name
        organism_variant = getattr(cfg, "organism_variant", "default")
        
        if organism_variant != "default" and organism_variant:
             # Use a safe name format: {organism}_{variant}
             organism_path_name = f"{cfg.organism.name}_{organism_variant}"
        
        # Get sample and token counts for directory naming
        max_samples = int(self.method_cfg.method_params.max_samples)
        max_tokens_per_sample = int(self.method_cfg.method_params.max_tokens_per_sample)
             
        # Create base results directory with sample/token counts
        # Structure: .../diffing_results/{model_name}/{organism_path_name}/logit_diff_topk_occurring_{samples}samples_{tokens}tokens
        method_dir_name = f"logit_diff_topk_occurring_{max_samples}samples_{max_tokens_per_sample}tokens"
        self.base_results_dir = Path(cfg.diffing.results_base_dir) / cfg.model.name / organism_path_name / method_dir_name
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectory for saved tensors (logits, diffs, masks) - at base level
        self.saved_tensors_dir = self.base_results_dir / "saved_tensors"
        self.saved_tensors_dir.mkdir(parents=True, exist_ok=True)
        
        # analysis_dir will be created in run() method with timestamp
        self.analysis_dir = None

    def _get_analysis_folder_name(self) -> str:
        """
        Generate analysis folder name with timestamp and configuration parameters.
        
        Format: analysis_{timestamp}_top{k}_normalized_{bool}_mode_{selection_mode}{nmf_suffix}
        Example: analysis_20260110_143045_top100_normalized_false_mode_top_k_occurring_2topics_logit_diff_magnitude_beta2_orthogonal_weight_100p0
        
        Returns:
            Folder name string
        """
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get config parameters
        top_k = int(self.method_cfg.method_params.top_k)
        use_normalized = bool(self.method_cfg.get("use_normalized_tokens", False))
        normalized_str = "true" if use_normalized else "false"
        
        # Get token set selection mode
        selection_mode = str(self.method_cfg.method_params.token_set_selection_mode)
        
        # Build NMF suffix if enabled
        nmf_suffix = ""
        if self.nmf_cfg and self.nmf_cfg.enabled:
            num_topics = int(self.nmf_cfg.num_topics)
            mode = str(self.nmf_cfg.mode)
            beta = self.nmf_cfg.beta
            
            # Format beta (convert float to string, replace . with p if needed)
            beta_str = str(beta).replace(".", "p") if "." in str(beta) else str(beta)
            
            # Build orthogonal suffix
            orthogonal_suffix = ""
            if bool(getattr(self.nmf_cfg, 'orthogonal', False)):
                weight = getattr(self.nmf_cfg, 'orthogonal_weight', 1.0)
                weight_str = str(weight).replace(".", "p")
                orthogonal_suffix = f"_orthogonal_weight_{weight_str}"
            
            nmf_suffix = f"_{num_topics}topics_{mode}_beta{beta_str}{orthogonal_suffix}"
        
        # Combine all parts
        folder_name = f"analysis_{timestamp}_top{top_k}_normalized_{normalized_str}_mode_{selection_mode}{nmf_suffix}"
        
        return folder_name

    def get_or_create_analysis_dir(self) -> Path:
        """
        Get or create the analysis directory for results.
        
        This method:
        1. Returns analysis_dir if already set (from run())
        2. Otherwise, finds the most recent analysis folder in base_results_dir
        3. If no analysis folders exist, creates a new one using _get_analysis_folder_name()
        
        Returns:
            Path to the analysis directory
        """
        # If analysis_dir is already set (from run()), return it
        if self.analysis_dir is not None:
            return self.analysis_dir
        
        # Look for existing analysis folders
        analysis_folders = sorted([
            d for d in self.base_results_dir.iterdir()
            if d.is_dir() and d.name.startswith("analysis_")
        ], key=lambda x: x.stat().st_mtime, reverse=True)
        
        if analysis_folders:
            # Use the most recent analysis folder
            self.analysis_dir = analysis_folders[0]
            self.logger.info(f"Using existing analysis directory: {self.analysis_dir}")
        else:
            # Create a new analysis folder
            analysis_folder_name = self._get_analysis_folder_name()
            self.analysis_dir = self.base_results_dir / analysis_folder_name
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new analysis directory: {self.analysis_dir}")
        
        return self.analysis_dir

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
        pre_assistant_k = int(self.method_cfg.method_params.pre_assistant_k)
        
        # Get debug_print_samples from config (None by default)
        debug_print_samples = getattr(self.method_cfg, "debug_print_samples", None)
        
        # Tokenize entire dataset using ADL functions
        all_positions = None  # Will be set for chat datasets
        if dataset_cfg.is_chat:
            self.logger.info(f"Using ADL's load_and_tokenize_chat_dataset() with pre_assistant_k={pre_assistant_k}")
            samples = load_and_tokenize_chat_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                messages_column=dataset_cfg.messages_column or "messages",
                n=max_tokens,
                pre_assistant_k=pre_assistant_k,
                max_samples=max_samples,
                debug_print_samples=debug_print_samples,
            )
            all_token_ids = [sample["input_ids"] for sample in samples]
            all_positions = [sample["positions"] for sample in samples]  # Extract positions for slicing
        else:
            self.logger.info("Using ADL's load_and_tokenize_dataset()")
            all_token_ids = load_and_tokenize_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                text_column=dataset_cfg.text_column or "text",
                n=max_tokens,
                max_samples=max_samples,
                subset=dataset_cfg.subset,
                streaming=True,
                debug_print_samples=debug_print_samples,
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
            "attention_mask": attention_mask,
            "positions": all_positions,  # None for pretraining, list of position indices for chat
        }

    @torch.no_grad()
    def compute_stats_from_logits(
        self, 
        dataset_cfg: DatasetConfig,
        attention_mask: torch.Tensor,
        logit_diff: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Core analysis for one dataset.

        Args:
            dataset_cfg: Dataset configuration
            attention_mask: Attention mask
            logit_diff: Pre-computed logit difference tensor

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

        # Validate and slice samples
        available_samples = logit_diff.shape[0]
        if max_samples > available_samples:
            raise ValueError(
                f"Config requests {max_samples} samples but only {available_samples} available from preprocessing. "
                f"Re-run preprocessing with max_samples >= {max_samples}."
            )
        elif max_samples < available_samples:
            self.logger.info(f"Using first {max_samples} samples (have {available_samples})")
            logit_diff = logit_diff[:max_samples, :, :]
            attention_mask = attention_mask[:max_samples, :]

        # Validate and slice token positions
        available_positions = logit_diff.shape[1]
        if max_tokens > available_positions:
            raise ValueError(
                f"Config requests {max_tokens} token positions but only {available_positions} available from preprocessing. "
                f"Re-run preprocessing with max_tokens_per_sample >= {max_tokens}."
            )
        elif max_tokens < available_positions:
            self.logger.info(f"Using first {max_tokens} positions (have {available_positions})")
            logit_diff = logit_diff[:, :max_tokens, :]
            attention_mask = attention_mask[:, :max_tokens]

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
        
        # Per-position and per-sample shortlist diffs for KDE breakdown plots
        shortlist_diffs_by_position = defaultdict(lambda: defaultdict(list))
        # Dict[token_str][position_idx] -> List[float]
        shortlist_diffs_by_sample = defaultdict(lambda: defaultdict(list))
        # Dict[token_str][sample_idx] -> List[float]
        
        # Co-occurrence tracking (Top-K based)
        same_point_matrix = defaultdict(lambda: defaultdict(int))
        # Track which tokens appeared in each sample (for Same-Sample co-occurrence)
        # Dict[sample_idx, Set[token_str]]
        sample_tokens_tracker = defaultdict(set)
        # Track which tokens appeared at each position (for Same-Position co-occurrence)
        # Dict[position_idx, Set[token_str]]
        position_tokens_tracker = defaultdict(set)
        
        # Same-Sign Co-occurrence tracking
        # Tokens co-occur if they have the same sign logit diff at a location
        same_sign_point_matrix = defaultdict(lambda: defaultdict(int))
        # Track which tokens had positive/negative diffs in each sample
        sample_pos_tokens_tracker = defaultdict(set)
        sample_neg_tokens_tracker = defaultdict(set)
        # Track which tokens had positive/negative diffs at each position
        position_pos_tokens_tracker = defaultdict(set)
        position_neg_tokens_tracker = defaultdict(set)
        
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
            # We can get vocab size from the diff shape later.
            self.logger.info(f"Global Token Statistics enabled")
        
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

        # Removed redundant tokenization block here
        # Logic uses passed input_ids/attention_mask directly in the batch loop below

        # Now batch through token IDs
        # We use the passed input_ids directly
        num_samples = logit_diff.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.logger.info(f"Processing {num_samples} samples in {num_batches} batches...")
        
        # Track max sequence length across all batches for per-token analysis
        overall_max_len = logit_diff.shape[1]

        for batch_idx in tqdm(range(num_batches), desc=f"Processing {dataset_cfg.name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Get batch of token IDs from PASSED arguments, not re-tokenized list
            # We must use slice notation on tensors
            batch_attention_mask = attention_mask[start_idx:end_idx]

            # Get logits or diff
            diff = logit_diff[start_idx:end_idx].to(self.device)

            # Use local batch mask
            attention_mask_batch = batch_attention_mask.to(diff.device)

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
                mask_expanded = attention_mask_batch.unsqueeze(-1).to(diff.dtype)
                
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

            # Shortlist Distribution Tracking (Vectorized for aggregate)
            if per_token_enabled and shortlist_token_ids:
                # We want to collect all logit diffs for the shortlist tokens
                # respecting the attention mask (if ignore_padding is True)
                valid_mask = attention_mask_batch.bool()
                batch_size_curr, seq_len_curr = attention_mask_batch.shape
                
                for s_token_id, s_token_str in shortlist_token_ids.items():
                    # diff[..., s_token_id]: [batch, seq]
                    token_vals = diff[:, :, s_token_id]
                    
                    if ignore_padding:
                        valid_vals = token_vals[valid_mask]
                    else:
                        valid_vals = token_vals.flatten()
                        
                    shortlist_diffs[s_token_str].extend(valid_vals.tolist())
                    
                    # Track by position and sample for KDE breakdown plots
                    for b_idx in range(batch_size_curr):
                        sample_idx_global = start_idx + b_idx
                        for pos_idx in range(seq_len_curr):
                            if ignore_padding and attention_mask_batch[b_idx, pos_idx] == 0:
                                continue
                            val = token_vals[b_idx, pos_idx].item()
                            shortlist_diffs_by_position[s_token_str][pos_idx].append(val)
                            shortlist_diffs_by_sample[s_token_str][sample_idx_global].append(val)

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
                    if ignore_padding and attention_mask_batch[b, s] == 0:
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

                    # Same-Sign Co-occurrence: track shortlist tokens by their sign at this point
                    if co_occurrence_enabled and shortlist_token_ids:
                        point_pos_tokens = []
                        point_neg_tokens = []
                        
                        for s_token_id, s_token_str in shortlist_token_ids.items():
                            token_diff = diff[b, s, s_token_id].item()
                            if token_diff >= 0:
                                point_pos_tokens.append(s_token_str)
                                sample_pos_tokens_tracker[sample_idx].add(s_token_str)
                                position_pos_tokens_tracker[s].add(s_token_str)
                            else:
                                point_neg_tokens.append(s_token_str)
                                sample_neg_tokens_tracker[sample_idx].add(s_token_str)
                                position_neg_tokens_tracker[s].add(s_token_str)
                        
                        # Build same-sign co-occurrence at this point
                        # Same-sign = both positive OR both negative
                        for t1, t2 in combinations_with_replacement(point_pos_tokens, 2):
                            same_sign_point_matrix[t1][t2] += 1
                            if t1 != t2:
                                same_sign_point_matrix[t2][t1] += 1
                        for t1, t2 in combinations_with_replacement(point_neg_tokens, 2):
                            same_sign_point_matrix[t1][t2] += 1
                            if t1 != t2:
                                same_sign_point_matrix[t2][t1] += 1

                    total_positions += 1

        self.logger.info(f"✓ Batch processing complete!")
        self.logger.info(
            f"Processed {total_positions:,} positions with {len(global_token_counts):,} unique tokens"
        )

        # Compute co-occurrence matrices if enabled
        same_sample_matrix = defaultdict(lambda: defaultdict(int))
        same_position_matrix = defaultdict(lambda: defaultdict(int))
        
        # Same-sign co-occurrence matrices
        same_sign_sample_matrix = defaultdict(lambda: defaultdict(int))
        same_sign_position_matrix = defaultdict(lambda: defaultdict(int))
        
        if co_occurrence_enabled:
            self.logger.info("Computing Same-Sample co-occurrence matrix (Top-K)...")
            for sample_idx, tokens in sample_tokens_tracker.items():
                if not tokens:
                    continue
                # Optimization: Use combinations to halve iterations
                for t1, t2 in combinations_with_replacement(tokens, 2):
                    same_sample_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_sample_matrix[t2][t1] += 1
                        
            self.logger.info("Computing Same-Position co-occurrence matrix (Top-K)...")
            for pos_idx, tokens in position_tokens_tracker.items():
                if not tokens:
                    continue
                # Optimization: Use combinations to halve iterations
                for t1, t2 in combinations_with_replacement(tokens, 2):
                    same_position_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_position_matrix[t2][t1] += 1
            
            # Compute Same-Sign co-occurrence matrices
            self.logger.info("Computing Same-Sample co-occurrence matrix (Same-Sign)...")
            all_sample_indices = set(sample_pos_tokens_tracker.keys()) | set(sample_neg_tokens_tracker.keys())
            for sample_idx in all_sample_indices:
                pos_tokens = sample_pos_tokens_tracker[sample_idx]
                neg_tokens = sample_neg_tokens_tracker[sample_idx]
                
                # Same-sign pairs = pairs within pos_tokens + pairs within neg_tokens
                for t1, t2 in combinations_with_replacement(pos_tokens, 2):
                    same_sign_sample_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_sign_sample_matrix[t2][t1] += 1
                for t1, t2 in combinations_with_replacement(neg_tokens, 2):
                    same_sign_sample_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_sign_sample_matrix[t2][t1] += 1
            
            self.logger.info("Computing Same-Position co-occurrence matrix (Same-Sign)...")
            all_position_indices = set(position_pos_tokens_tracker.keys()) | set(position_neg_tokens_tracker.keys())
            for pos_idx in all_position_indices:
                pos_tokens = position_pos_tokens_tracker[pos_idx]
                neg_tokens = position_neg_tokens_tracker[pos_idx]
                
                # Same-sign pairs = pairs within pos_tokens + pairs within neg_tokens
                for t1, t2 in combinations_with_replacement(pos_tokens, 2):
                    same_sign_position_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_sign_position_matrix[t2][t1] += 1
                for t1, t2 in combinations_with_replacement(neg_tokens, 2):
                    same_sign_position_matrix[t1][t2] += 1
                    if t1 != t2:
                        same_sign_position_matrix[t2][t1] += 1

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
                self.analysis_dir,
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
            json_path = self.analysis_dir / f"{dataset_cfg.name}_global_token_stats.json"
            
            # Note: Scatter plots will be generated later in run() after save_results()
            # so that occurrence_rates.json is available for highlighting top-K tokens

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
                results["_per_token_data"]["co_occurrence_same_sign"] = {
                    "same_sign_same_sample": same_sign_sample_matrix,
                    "same_sign_same_position": same_sign_position_matrix,
                    "same_sign_same_point": same_sign_point_matrix,
                }
            
            results["_per_token_data"]["shortlist_distributions"] = shortlist_diffs
            results["_per_token_data"]["shortlist_diffs_by_position"] = shortlist_diffs_by_position
            results["_per_token_data"]["shortlist_diffs_by_sample"] = shortlist_diffs_by_sample

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
        output_file = self.analysis_dir / f"{dataset_name}_global_token_stats.json"
        
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

    def _get_fraction_positive_tokens(
        self,
        dataset_name: str,
        k: int,
        filter_punctuation: bool = False,
        normalize: bool = False,
        filter_special_tokens: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load global token stats and return top-K tokens sorted by fraction of positive logit diffs.
        
        This is an alternative token selection method that uses the fraction of positions
        where a token has a non-negative logit diff, rather than the occurrence rate in top-K.
        
        Uses the shared load_fraction_positive_tokens function from normalization.py
        which applies filtering/normalization BEFORE taking top K.
        
        Args:
            dataset_name: Name of the dataset
            k: Number of top tokens to return
            filter_punctuation: Whether to filter out pure punctuation tokens
            normalize: Whether to normalize (lowercase, strip, consolidate) tokens
            filter_special_tokens: Whether to filter out special tokens (BOS, EOS, PAD, etc.)
            
        Returns:
            List of token dicts with keys: token_id, token_str, count_positive, count_negative,
            positive_occurrence_rate, negative_occurrence_rate, fraction_positive
        """
        stats_file = self.analysis_dir / f"{dataset_name}_global_token_stats.json"
        
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Global token stats not found: {stats_file}. "
                "Please ensure global_token_statistics is enabled and run() has been executed."
            )
        
        # Use shared function from normalization.py
        top_tokens = load_fraction_positive_tokens(
            global_stats_file=stats_file,
            k=k,
            filter_punctuation=filter_punctuation,
            normalize=normalize,
            filter_special_tokens=filter_special_tokens,
            tokenizer=self.tokenizer
        )
        
        self.logger.info(
            f"Selected {len(top_tokens)} tokens using fraction_positive_diff mode "
            f"(top fraction: {top_tokens[0]['fraction_positive']:.4f} for '{top_tokens[0]['token_str']}')"
        )
        
        return top_tokens

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
        
        # NOTE: To attempt GPU execution without dense expansion OOM, try using sparse tensors directly:
        # indices = torch.tensor([nmf_data["rows"], nmf_data["cols"]], dtype=torch.long)
        # V_sparse_torch = torch.sparse_coo_tensor(indices, values, (num_rows, num_cols)).cuda()
        # nmf.fit(V_sparse_torch, ...)
        # However, not sure about sparse autograd support in PyTorch/torchnmf
        
        V_dense = torch.tensor(V_sparse.todense(), dtype=torch.float32)
        V_dense = torch.relu(V_dense) # Ensure non-negative
        
        # We comment this out because converting to dense matrix for NMF often exceeds single GPU memory.
        # Torchnmf works on single GPU but not multi-GPU, and CPU has more RAM.
        # if torch.cuda.is_available():
        #     V_dense = V_dense.cuda()
            
        # 2. Run NMF
        num_topics = int(self.nmf_cfg.num_topics)
        beta = float(self.nmf_cfg.beta)
        use_orthogonal = bool(getattr(self.nmf_cfg, 'orthogonal', False))
        
        if use_orthogonal:
            orthogonal_weight = float(getattr(self.nmf_cfg, 'orthogonal_weight', 1.0))
            self.logger.info(f"Running Orthogonal NMF with {num_topics} topics (beta={beta}, orthogonal_weight={orthogonal_weight})...")
            
            # Use custom orthogonal NMF implementation
            with torch.enable_grad():
                W_nmf, H_nmf = fit_nmf_orthogonal(
                    V_dense, 
                    rank=num_topics, 
                    beta=beta,
                    orthogonal_weight=orthogonal_weight, 
                    max_iter=200,
                    device="auto",
                    verbose=True
                )
        else:
            self.logger.info(f"Running NMF with {num_topics} topics (beta={beta})...")
            
            nmf = NMF(V_dense.shape, rank=num_topics)
            # if torch.cuda.is_available():
            #     nmf = nmf.cuda()
                
            # Fit
            # NMF requires gradients for its update steps (it uses autograd for multiplicative updates),
            # but the surrounding method has @torch.no_grad(). We must re-enable gradients here.
            with torch.enable_grad():
                nmf.fit(V_dense, beta=beta, verbose=False, max_iter=200)
            
            W_nmf = nmf.W.detach().cpu()
            H_nmf = nmf.H.detach().cpu()
        
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
        
        # W_nmf = nmf.W.detach().cpu()
        # H_nmf = nmf.H.detach().cpu()
        # print('W_nmf shape: ', W_nmf.shape)
        # print('H_nmf shape: ', H_nmf.shape)
        # print('V_dense shape: ', V_dense.shape)
        # print('')

        # topic_token_matrix = nmf.W.T.detach().cpu()
        # print('topic_token_matrix shape: ', topic_token_matrix.shape)
        topic_token_matrix = W_nmf.T  # Shape: (num_topics, num_tokens)

            
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
        
        output_file = self.analysis_dir / f"{dataset_name}_nmf_topics_analysis.json"
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
        output_file = self.analysis_dir / f"{dataset_name}_occurrence_rates.json"

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
        shortlist_diffs: Dict[str, List[float]] = None,
        shortlist_diffs_by_position: Dict[str, Dict[int, List[float]]] = None,
        shortlist_diffs_by_sample: Dict[str, Dict[int, List[float]]] = None
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
            shortlist_diffs_by_position: Dict[token_str][position_idx] -> list of logit diffs
            shortlist_diffs_by_sample: Dict[token_str][sample_idx] -> list of logit diffs
        """
        self.logger.info(f"Saving per-token analysis for {dataset_name}...")
        
        # Create subdirectories for organized output
        per_token_dir = self.analysis_dir / "per_token_analysis"
        data_dir = per_token_dir / "data"
        plots_dir = per_token_dir / "plots"
        
        # Subdirectories for plots
        sample_plots_dir = plots_dir / "per_token_by_sample"
        position_plots_dir = plots_dir / "per_token_by_position"
        dist_plots_dir = plots_dir / "logit_diff_distributions"
        dist_by_position_dir = plots_dir / "logit_diff_distributions_by_position"
        dist_by_sample_dir = plots_dir / "logit_diff_distributions_by_sample"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        sample_plots_dir.mkdir(parents=True, exist_ok=True)
        position_plots_dir.mkdir(parents=True, exist_ok=True)
        dist_plots_dir.mkdir(parents=True, exist_ok=True)
        dist_by_position_dir.mkdir(parents=True, exist_ok=True)
        dist_by_sample_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Get number of positions/samples to plot from config
        num_kde_positions = 10  # default
        if hasattr(self.method_cfg, 'positional_kde') and self.method_cfg.positional_kde.enabled:
            num_kde_positions = int(self.method_cfg.positional_kde.num_positions)
        
        # Plot logit diff distributions by position
        if shortlist_diffs_by_position:
            self.logger.info(f"Generating per-position logit diff distribution plots (first {num_kde_positions} positions)...")
            for token_str, pos_diffs in shortlist_diffs_by_position.items():
                if pos_diffs:
                    plot_shortlist_token_distribution_by_position(
                        pos_diffs,
                        token_str,
                        dataset_name,
                        dist_by_position_dir,
                        num_positions=num_kde_positions
                    )
        
        # Plot logit diff distributions by sample
        if shortlist_diffs_by_sample:
            self.logger.info(f"Generating per-sample logit diff distribution plots (first {num_kde_positions} samples)...")
            for token_str, sample_diffs in shortlist_diffs_by_sample.items():
                if sample_diffs:
                    plot_shortlist_token_distribution_by_sample(
                        sample_diffs,
                        token_str,
                        dataset_name,
                        dist_by_sample_dir,
                        num_samples=num_kde_positions
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
        
        per_token_dir = self.analysis_dir / "per_token_analysis"
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

    def _generate_selected_tokens_table(
        self,
        dataset_name: str,
        dataset_id: str,
        top_positive: List[Dict[str, Any]],
        relevance_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Generate and save the selected tokens table visualization.
        
        Args:
            dataset_name: Name of the dataset for the title
            dataset_id: Dataset ID (used to construct relevance path if labels not provided)
            top_positive: List of token dicts with 'token_str' and 'positive_occurrence_rate'
            relevance_labels: Optional list of 'RELEVANT'/'IRRELEVANT' labels. If None,
                              will attempt to load from disk.
        """
        self.logger.info("Generating selected tokens table...")
        
        # Determine number of tokens to show
        top_k = int(self.method_cfg.method_params.top_k)
        k_candidate = int(self.method_cfg.token_relevance.k_candidate_tokens)
        num_tokens_to_show = min(top_k, k_candidate)
        
        # If relevance labels not provided, try to load from disk
        if relevance_labels is None and self.method_cfg.token_relevance.enabled:
            relevance_path = (
                self.analysis_dir
                / "layer_global"
                / dataset_id.split("/")[-1]
                / "token_relevance"
                / "position_all"
                / "difference"
                / "relevance_logit_diff.json"
            )
            if relevance_path.exists():
                with open(relevance_path, "r") as f:
                    relevance_data = json.load(f)
                    relevance_labels = relevance_data.get("labels", None)
                    self.logger.info(f"Loaded {len(relevance_labels)} relevance labels")
        
        fig = plot_selected_tokens_table(
            top_positive=top_positive,
            dataset_name=dataset_name,
            relevance_labels=relevance_labels,
            num_tokens=num_tokens_to_show,
            figure_width=self.method_cfg.visualization.figure_width * 0.45,
            figure_height=self.method_cfg.visualization.figure_height * 1.2,
            figure_dpi=self.method_cfg.visualization.figure_dpi,
        )
        
        # Determine filename suffix based on whether relevance labels are available
        suffix = "_with_relevance" if relevance_labels is not None else "_no_relevance"
        table_path = self.analysis_dir / f"{dataset_name}_selected_tokens{suffix}.png"
        fig.savefig(table_path, bbox_inches="tight", dpi=self.method_cfg.visualization.figure_dpi)
        plt.close(fig)
        self.logger.info(f"Saved selected tokens table to {table_path}")

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
            results_file = self.analysis_dir / f"{dataset_name}_occurrence_rates.json"
            
            if not results_file.exists():
                logger.warning(f"No results found for {dataset_name}, skipping token relevance")
                continue
            
            logger.info("")
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Load saved occurrence results
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Apply token processing (filtering and/or normalization)
            filter_punct = bool(self.method_cfg.filter_pure_punctuation)
            normalize = bool(self.method_cfg.normalize_tokens)
            filter_special = bool(self.method_cfg.filter_special_tokens)
            total_positions = results["total_positions"]
            
            # Get token list based on selection mode
            selection_mode = str(self.method_cfg.method_params.token_set_selection_mode)
            
            if selection_mode == "fraction_positive_diff":
                # Use tokens sorted by fraction of positive logit diffs
                k_candidate = int(cfg.k_candidate_tokens)
                top_positive = self._get_fraction_positive_tokens(
                    dataset_name=dataset_name,
                    k=k_candidate,
                    filter_punctuation=filter_punct,
                    normalize=normalize,
                    filter_special_tokens=filter_special
                )
                logger.info(f"Using fraction_positive_diff mode for token selection ({len(top_positive)} tokens)")
            else:
                # Default: top_k_occurring - use occurrence rates from results
                top_positive = results["top_positive"]
                
                if filter_punct or normalize or filter_special:
                    original_count = len(top_positive)
                    top_positive = process_token_list(
                        top_positive, 
                        total_positions,
                        filter_punctuation=filter_punct,
                        normalize=normalize,
                        filter_special_tokens=filter_special,
                        tokenizer=self.tokenizer
                    )
                    logger.info(f"Applied token processing for {dataset_name}: {original_count} -> {len(top_positive)} tokens (filter_punct={filter_punct}, normalize={normalize}, filter_special={filter_special})")
            
            # Output directory (matches ADL structure with layer_global/position_all)
            dataset_dir_name = dataset_cfg.id.split("/")[-1]
            out_dir = (
                self.analysis_dir
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
            
            # Regenerate table with relevance coloring
            self._generate_selected_tokens_table(
                dataset_name=dataset_name,
                dataset_id=dataset_cfg.id,
                top_positive=top_positive,
                relevance_labels=final_labels,
            )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Token relevance grading completed!")
        logger.info("=" * 80)


    def run(self) -> None:
        """
        Main execution method for logit diff top-K occurring analysis.
        Runs during the 'diffing' stage (after preprocessing).
        """
        self.logger.info("=" * 80)
        self.logger.info("LOGIT DIFF TOP-K OCCURRING ANALYSIS")
        self.logger.info("=" * 80)
        
        # Create analysis directory with timestamp and config
        analysis_folder_name = self._get_analysis_folder_name()
        self.analysis_dir = self.base_results_dir / analysis_folder_name
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Analysis results will be saved to: {self.analysis_dir}")

        # Define directories using new structure
        diffs_dir = self.saved_tensors_dir / "logit_diffs"
        masks_dir = self.saved_tensors_dir / "attention_masks"
        
        # Check preprocessing has run:
        if not diffs_dir.exists() or not any(diffs_dir.iterdir()):
            error_msg = (
                f"No logit diff tensors found in {diffs_dir}. "
                "Please run with pipeline.mode=preprocessing first."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info("PHASE: Analysis & Diffing (Using pre-computed diffs)")
        
        for idx, dataset_cfg in enumerate(self.datasets, 1):
            self.logger.info("")
            self.logger.info(f"[{idx}/{len(self.datasets)}] Analyzing dataset: {dataset_cfg.name}")
            
            # Load diff
            diff_path = diffs_dir / f"{dataset_cfg.name}_logit_diff.pt"
            
            if not diff_path.exists():
                raise FileNotFoundError(
                    f"Diff file not found for {dataset_cfg.name}: {diff_path}. "
                    "Preprocessing may have failed or been interrupted."
                )

            self.logger.info(f"Loading logit diff from {diff_path}...")
            logit_diff = torch.load(diff_path, map_location="cpu")
            
            # Load attention mask (or re-prepare)
            mask_path = masks_dir / f"{dataset_cfg.name}_attention_mask.pt"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Attention mask not found for {dataset_cfg.name}: {mask_path}. "
                    "Preprocessing may have failed or been interrupted."
                )
                 
            attention_mask = torch.load(mask_path, map_location="cpu")

            results = self.compute_stats_from_logits(
                dataset_cfg=dataset_cfg,
                attention_mask=attention_mask,
                logit_diff=logit_diff # Pass pre-computed diff
            )

            if results is not None:
                # Save results to disk
                self.save_results(dataset_cfg.name, results)
                
                # Generate scatter plots (after save_results so occurrence_rates.json exists)
                if self.method_cfg.global_token_statistics.enabled:
                    self.logger.info("Generating global token scatter plot...")
                    json_path = self.analysis_dir / f"{dataset_cfg.name}_global_token_stats.json"
                    occurrence_rates_path = self.analysis_dir / f"{dataset_cfg.name}_occurrence_rates.json"
                    
                    # Apply filtering to scatter plot based on config
                    filter_punct = bool(self.method_cfg.filter_pure_punctuation)
                    filter_special = bool(self.method_cfg.filter_special_tokens)
                    
                    plot_global_token_scatter(
                        json_path, 
                        self.analysis_dir, 
                        tokenizer=self.tokenizer,
                        top_k_labels=int(self.method_cfg.global_token_statistics.top_k_plotting_labels),
                        occurrence_rates_json_path=occurrence_rates_path,
                        filter_punctuation=filter_punct,
                        filter_special_tokens=filter_special
                    )
                    
                    # Generate Interactive Plotly HTML
                    self.logger.info("Generating interactive global token scatter (HTML)...")
                    fig = get_global_token_scatter_plotly(
                        json_path, 
                        occurrence_rates_json_path=occurrence_rates_path,
                        filter_punctuation=filter_punct,
                        filter_special_tokens=filter_special,
                        tokenizer=self.tokenizer
                    )
                    html_path = self.analysis_dir / f"{dataset_cfg.name}_global_token_scatter.html"
                    fig.write_html(str(html_path))
                    self.logger.info(f"Saved interactive scatter plot to {html_path}")
                
                # Generate and save occurrence plot (Red-Green Bar Chart)
                self.logger.info("Generating occurrence rate plot...")
                fig = plot_occurrence_bar_chart(
                    results["top_positive"],
                    results["top_negative"],
                    results["metadata"]["base_model"],
                    results["metadata"]["finetuned_model"],
                    results["total_positions"],
                    figure_width=self.method_cfg.visualization.figure_width,
                    figure_height=self.method_cfg.visualization.figure_height,
                    figure_dpi=self.method_cfg.visualization.figure_dpi,
                    font_sizes=getattr(self.method_cfg.visualization, "font_sizes", None)
                )
                plot_path = self.analysis_dir / f"{dataset_cfg.name}_occurrence_rates.png"
                fig.savefig(plot_path, bbox_inches="tight", dpi=self.method_cfg.visualization.figure_dpi)
                plt.close(fig)
                self.logger.info(f"Saved occurrence rate plot to {plot_path}")

                # Generate and save selected tokens table (using mode-appropriate token list)
                selection_mode = str(self.method_cfg.method_params.token_set_selection_mode)
                filter_punct = bool(self.method_cfg.filter_pure_punctuation)
                normalize = bool(self.method_cfg.normalize_tokens)
                filter_special = bool(self.method_cfg.filter_special_tokens)
                
                if selection_mode == "fraction_positive_diff":
                    k_candidate = int(self.method_cfg.token_relevance.k_candidate_tokens)
                    table_tokens = self._get_fraction_positive_tokens(
                        dataset_name=dataset_cfg.name,
                        k=k_candidate,
                        filter_punctuation=filter_punct,
                        normalize=normalize,
                        filter_special_tokens=filter_special
                    )
                else:
                    # Default: top_k_occurring
                    table_tokens = results["top_positive"]
                    if filter_punct or normalize or filter_special:
                        table_tokens = process_token_list(
                            table_tokens,
                            results["total_positions"],
                            filter_punctuation=filter_punct,
                            normalize=normalize,
                            filter_special_tokens=filter_special,
                            tokenizer=self.tokenizer
                        )
                
                self._generate_selected_tokens_table(
                    dataset_name=dataset_cfg.name,
                    dataset_id=dataset_cfg.id,
                    top_positive=table_tokens,
                )

                # Save and plot per-token analysis if enabled
                if "_per_token_data" in results:
                    shortlist_diffs = results["_per_token_data"].pop("shortlist_distributions", None)
                    shortlist_diffs_by_position = results["_per_token_data"].pop("shortlist_diffs_by_position", None)
                    shortlist_diffs_by_sample = results["_per_token_data"].pop("shortlist_diffs_by_sample", None)
                    
                    self._save_and_plot_per_token_analysis(
                        dataset_cfg.name,
                        results["_per_token_data"]["per_sample_counts"],
                        results["_per_token_data"]["per_position_counts"],
                        results["num_samples"],
                        results["_per_token_data"]["max_positions"],
                        shortlist_diffs=shortlist_diffs,
                        shortlist_diffs_by_position=shortlist_diffs_by_position,
                        shortlist_diffs_by_sample=shortlist_diffs_by_sample
                    )
                    
                    if "co_occurrence" in results["_per_token_data"]:
                        self._save_and_plot_co_occurrence(
                            dataset_cfg.name,
                            results["_per_token_data"]["co_occurrence"]
                        )
                    
                    if "co_occurrence_same_sign" in results["_per_token_data"]:
                        self._save_and_plot_co_occurrence(
                            dataset_cfg.name,
                            results["_per_token_data"]["co_occurrence_same_sign"]
                        )
                
                self.logger.info(f"✓ [{idx}/{len(self.datasets)}] Completed dataset: {dataset_cfg.name}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("✓ Logit diff top-K occurring analysis completed successfully!")
        self.logger.info(f"✓ Results saved to: {self.analysis_dir}")
        self.logger.info("=" * 80)
        
        # Run token relevance grading if enabled
        if hasattr(self.method_cfg, 'token_relevance') and self.method_cfg.token_relevance.enabled:
            self.run_token_relevance()

    def preprocess(self, delete_raw: bool = True) -> None:
        """
        Preprocessing Phase: Data Prep, Model Inference, and Diff Computation.
        Saves {dataset}_logit_diff.pt and optionally deletes raw logits.
        """
        self.logger.info("=" * 80)
        self.logger.info("LOGIT DIFF TOP-K OCCURRING: PREPROCESSING")
        self.logger.info("=" * 80)

        # Phase 0: Data Preparation (Tokenize all datasets)
        self.logger.info("PHASE 0: Data Preparation")
        
        # Setup output directories in saved_tensors
        logits_dir = self.saved_tensors_dir # Raw logits go to root of saved_tensors
        diffs_dir = self.saved_tensors_dir / "logit_diffs"
        masks_dir = self.saved_tensors_dir / "attention_masks"
        
        diffs_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for dataset_cfg in self.datasets:
            # Use dataset_cfg.name as key (not .id) to handle datasets with same id but different subsets
            # e.g., CulturaX_de, CulturaX_fr, CulturaX_ja all share id="uonlp/CulturaX" but have different names
            dataset_inputs[dataset_cfg.name] = self._prepare_dataset_tensors(dataset_cfg)
            
            # Save attention mask
            mask_path = masks_dir / f"{dataset_cfg.name}_attention_mask.pt"
            torch.save(dataset_inputs[dataset_cfg.name]["attention_mask"], mask_path)
            self.logger.info(f"Saved attention mask to {mask_path}")

        # Phase 1: Base Model Inference
        self.logger.info("")
        self.logger.info("PHASE 1: Base Model Inference")
        self.logger.info(f"Loading base model: {self.base_model_cfg.model_id}")
        _ = self.base_model # Trigger load
        
        batch_size = int(self.method_cfg.method_params.batch_size)

        for dataset_cfg in self.datasets:
            self.logger.info(f"Computing base logits for {dataset_cfg.name}...")
            inputs = dataset_inputs[dataset_cfg.name]
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
                
                # Save logits to disk
                logits_path = logits_dir / f"{dataset_cfg.name}_base_logits.pt"
                torch.save(all_logits, logits_path)
                self.logger.info(f"Saved base logits to {logits_path}")
                
                # Clear from memory
                del all_logits
                del dataset_logits
                gc.collect()
            
        self.clear_base_model()

        # Phase 2: Finetuned Model Inference
        self.logger.info("")
        self.logger.info("PHASE 2: Finetuned Model Inference")
        self.logger.info(f"Loading finetuned model: {self.finetuned_model_cfg.model_id}")
        _ = self.finetuned_model # Trigger load
        
        for dataset_cfg in self.datasets:
            self.logger.info(f"Computing finetuned logits for {dataset_cfg.name}...")
            inputs = dataset_inputs[dataset_cfg.name]
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
                
                # Save logits to disk
                logits_path = logits_dir / f"{dataset_cfg.name}_finetuned_logits.pt"
                torch.save(all_logits, logits_path)
                self.logger.info(f"Saved finetuned logits to {logits_path}")

                # Clear from memory
                del all_logits
                del dataset_logits
                gc.collect()

        self.clear_finetuned_model()
        
        # New Step: Compute and Save Diffs (and optionally delete raw logits)
        self.logger.info("")
        self.logger.info("Computing and Saving Logit Diffs...")
        
        for dataset_cfg in self.datasets:
            self.logger.info(f"Computing diff for {dataset_cfg.name}...")
            
            base_path = logits_dir / f"{dataset_cfg.name}_base_logits.pt"
            ft_path = logits_dir / f"{dataset_cfg.name}_finetuned_logits.pt"
            
            if not base_path.exists() or not ft_path.exists():
                self.logger.warning(f"Missing base or finetuned logits for {dataset_cfg.name}. Skipping diff.")
                continue

            # Load logits from disk one by one to save memory
            base = torch.load(base_path, map_location="cpu")
            ft = torch.load(ft_path, map_location="cpu")
            
            # Ensure same device/type if needed, though they should be CPU tensors
            diff = ft - base
            
            # Slice chat data to relevant positions only (pre_assistant_k + n tokens around assistant start)
            # This saves disk space and makes analysis consistent with ADL behavior
            positions_list = dataset_inputs[dataset_cfg.name].get("positions")
            if positions_list is not None:
                original_shape = diff.shape
                diff = slice_to_positions(diff, positions_list)
                self.logger.info(f"Sliced chat logit diff from {original_shape} to {diff.shape} (pre_assistant_k + n positions)")
            
            # Save diff
            diff_path = diffs_dir / f"{dataset_cfg.name}_logit_diff.pt"
            torch.save(diff, diff_path)
            self.logger.info(f"Saved logit diff to {diff_path}")
            
            # Clear memory immediately
            del base
            del ft
            del diff
            gc.collect()
            
            if delete_raw:
                if base_path.exists():
                    base_path.unlink()
                    self.logger.info(f"Deleted raw base logits: {base_path}")
                if ft_path.exists():
                    ft_path.unlink()
                    self.logger.info(f"Deleted raw finetuned logits: {ft_path}")
                    
        self.logger.info("Preprocessing phase complete.")

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
            Dict mapping {model: {organism_variant_name: path_to_results}}
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
                # The organism_dir might now include the variant suffix, 
                # but it is treated as the "organism name" in this map structure.
                
                # Look for method directories with new naming pattern: logit_diff_topk_occurring_{samples}samples_{tokens}tokens
                for method_dir in organism_dir.iterdir():
                    if method_dir.is_dir() and method_dir.name.startswith("logit_diff_topk_occurring"):
                        # Check if there are any analysis folders with occurrence_rates.json files
                        analysis_folders = [d for d in method_dir.iterdir() if d.is_dir() and d.name.startswith("analysis_")]
                        for analysis_folder in analysis_folders:
                            if list(analysis_folder.glob("*_occurrence_rates.json")):
                                # Store the analysis folder path
                                results[model_name][organism_name] = str(analysis_folder)
                                break  # Take the first matching analysis folder

        return results

    def get_agent(self) -> DiffingMethodAgent:
        from .agents import LogitDiffAgent
        return LogitDiffAgent(cfg=self.cfg)



