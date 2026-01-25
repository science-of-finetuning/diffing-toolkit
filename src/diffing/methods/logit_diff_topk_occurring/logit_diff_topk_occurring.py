"""
Logit Diff Top-K Occurring analysis method.

This module computes occurrence rates of tokens in the top-K positive and negative
logit differences between a base model and a finetuned model.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import torch.nn.functional as F
import gc
from omegaconf import DictConfig
from loguru import logger
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, IterableDataset
from datetime import datetime
import itertools
import re

from ..diffing_method import DiffingMethod
from src.utils.configs import DatasetConfig
from src.utils.agents.diffing_method_agent import DiffingMethodAgent
from src.utils.agents.base_agent import BaseAgent
from src.utils.graders.token_relevance_grader import TokenRelevanceGrader
from ..activation_difference_lens.token_relevance import _compute_frequent_tokens
from ..activation_difference_lens.act_diff_lens import (
    load_and_tokenize_dataset,
    load_and_tokenize_chat_dataset,
    extract_first_n_tokens_activations,
)
from src.utils.activations import get_layer_indices
from src.utils.model import logit_lens
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
    plot_pairwise_token_correlation,
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


def slice_to_positions_2d(tensor: torch.Tensor, positions_list: List[List[int]]) -> torch.Tensor:
    """
    Extract specific positions from each sample in a 2D tensor.
    
    Similar to slice_to_positions but for 2D tensors like input_ids [batch, seq_len].
    
    Args:
        tensor: [batch, seq_len] tensor (e.g., input_ids)
        positions_list: List of position indices per sample (from chat dataset)
    
    Returns:
        Tensor of shape [batch, max_positions] with only the relevant positions
    """
    batch_size = tensor.shape[0]
    max_pos = max(len(p) for p in positions_list)
    
    result = torch.zeros(batch_size, max_pos, dtype=tensor.dtype, device=tensor.device)
    for i, positions in enumerate(positions_list):
        for j, pos in enumerate(positions):
            result[i, j] = tensor[i, pos]
    return result


def vectorized_bincount_masked(
    indices: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int
) -> torch.Tensor:
    """
    Count token occurrences using bincount, respecting attention mask.
    
    Args:
        indices: [batch, seq, topk] tensor of token indices
        attention_mask: [batch, seq] attention mask (1=valid, 0=padding)
        vocab_size: vocabulary size for bincount minlength
    
    Returns:
        [vocab_size] tensor of counts
    """
    batch, seq, topk = indices.shape
    
    # Expand mask to match indices shape: [batch, seq] -> [batch, seq, topk]
    mask = attention_mask.unsqueeze(-1).expand(-1, -1, topk).bool()
    
    # Flatten and apply mask
    flat_indices = indices.flatten()  # [batch * seq * topk]
    flat_mask = mask.flatten()  # [batch * seq * topk]
    
    # Only count valid (non-padding) positions
    valid_indices = flat_indices[flat_mask]
    
    # Count occurrences
    counts = torch.bincount(valid_indices, minlength=vocab_size)
    return counts.to(torch.int64)


def vectorized_shortlist_counts(
    top_k_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    shortlist_ids_tensor: torch.Tensor,
    start_idx: int
) -> tuple:
    """
    Count shortlist token occurrences per sample and per position (vectorized).
    
    Args:
        top_k_indices: [batch, seq, topk] tensor of token indices
        attention_mask: [batch, seq] attention mask
        shortlist_ids_tensor: [num_shortlist] tensor of shortlist token IDs
        start_idx: starting sample index for this batch
    
    Returns:
        per_sample: [batch, num_shortlist] counts per sample
        per_position: [seq, num_shortlist] counts per position
    """
    batch, seq, topk = top_k_indices.shape
    num_shortlist = shortlist_ids_tensor.shape[0]
    device = top_k_indices.device
    
    # Check which topk entries match shortlist tokens
    # top_k_indices: [batch, seq, topk]
    # shortlist_ids_tensor: [num_shortlist]
    # matches: [batch, seq, topk, num_shortlist] -> True if match
    matches = (top_k_indices.unsqueeze(-1) == shortlist_ids_tensor.view(1, 1, 1, -1))
    
    # Reduce over topk dimension: any match at this (batch, seq, shortlist_idx)?
    # [batch, seq, num_shortlist]
    has_match = matches.any(dim=2)
    
    # Apply attention mask: [batch, seq, 1]
    mask = attention_mask.unsqueeze(-1).bool()
    has_match_masked = has_match & mask
    
    # Per-sample counts: sum over seq dimension -> [batch, num_shortlist]
    per_sample = has_match_masked.sum(dim=1).to(torch.int64)
    
    # Per-position counts: sum over batch dimension -> [seq, num_shortlist]
    per_position = has_match_masked.sum(dim=0).to(torch.int64)
    
    return per_sample, per_position


def vectorized_cooccurrence_shortlist(
    top_k_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    shortlist_ids_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Compute co-occurrence matrix for shortlist tokens (vectorized).
    
    Co-occurrence at a point (sample, position) means both tokens appear in top-K.
    
    Args:
        top_k_indices: [batch, seq, topk] tensor of token indices  
        attention_mask: [batch, seq] attention mask
        shortlist_ids_tensor: [num_shortlist] tensor of shortlist token IDs
    
    Returns:
        [num_shortlist, num_shortlist] co-occurrence count matrix
    """
    batch, seq, topk = top_k_indices.shape
    num_shortlist = shortlist_ids_tensor.shape[0]
    device = top_k_indices.device
    
    # Check which shortlist tokens are present at each (batch, seq) point
    # matches: [batch, seq, topk, num_shortlist]
    matches = (top_k_indices.unsqueeze(-1) == shortlist_ids_tensor.view(1, 1, 1, -1))
    
    # Reduce over topk: presence at each point -> [batch, seq, num_shortlist]
    presence = matches.any(dim=2).float()
    
    # Apply attention mask
    mask = attention_mask.unsqueeze(-1).float()
    presence = presence * mask  # [batch, seq, num_shortlist]
    
    # Reshape to [num_points, num_shortlist]
    num_points = batch * seq
    presence_flat = presence.view(num_points, num_shortlist)
    
    # Co-occurrence = presence^T @ presence -> [num_shortlist, num_shortlist]
    cooc = presence_flat.T @ presence_flat
    
    return cooc.to(torch.int64)


def vectorized_same_sign_cooccurrence(
    diff: torch.Tensor,
    attention_mask: torch.Tensor,
    shortlist_ids_tensor: torch.Tensor
) -> tuple:
    """
    Compute same-sign co-occurrence for shortlist tokens.
    
    Two tokens co-occur with same sign if both have positive OR both have negative diffs
    at the same (sample, position).
    
    Args:
        diff: [batch, seq, vocab] logit diff tensor
        attention_mask: [batch, seq] attention mask
        shortlist_ids_tensor: [num_shortlist] tensor of shortlist token IDs
    
    Returns:
        same_sign_point_cooc: [num_shortlist, num_shortlist] same-sign co-occurrence at points
    """
    batch, seq, vocab = diff.shape
    num_shortlist = shortlist_ids_tensor.shape[0]
    device = diff.device
    
    # Extract diff values for shortlist tokens: [batch, seq, num_shortlist]
    shortlist_diffs = diff[:, :, shortlist_ids_tensor]
    
    # Determine sign: positive (>=0) or negative (<0)
    is_positive = (shortlist_diffs >= 0).float()  # [batch, seq, num_shortlist]
    is_negative = (shortlist_diffs < 0).float()
    
    # Apply attention mask
    mask = attention_mask.unsqueeze(-1).float()
    is_positive = is_positive * mask
    is_negative = is_negative * mask
    
    # Reshape to [num_points, num_shortlist]
    num_points = batch * seq
    pos_flat = is_positive.view(num_points, num_shortlist)
    neg_flat = is_negative.view(num_points, num_shortlist)
    
    # Same-sign co-occurrence: (both positive) OR (both negative)
    # pos^T @ pos + neg^T @ neg
    cooc_pos = pos_flat.T @ pos_flat
    cooc_neg = neg_flat.T @ neg_flat
    same_sign_cooc = cooc_pos + cooc_neg
    
    return same_sign_cooc.to(torch.int64)


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

        # Build DatasetConfig objects from inline config (ADL-style)
        self.datasets = []
        split = str(self.method_cfg.split)

        for dataset_entry in self.method_cfg.datasets:
            dataset_id = dataset_entry["id"]
            subset = dataset_entry.get("subset", None)
            is_chat = dataset_entry["is_chat"]
            
            # Determine column name based on dataset type
            if is_chat:
                column = dataset_entry.get("messages_column", "messages")
            else:
                column = dataset_entry.get("text_column", "text")
            
            # Create unique name: base_id + optional subset + split + column
            # e.g., "GreatFirewall-DPO_train_chosen" or "CulturaX_es_train_text"
            base_name = dataset_id.split("/")[-1]
            if subset:
                base_name = f"{base_name}_{subset}"
            name = f"{base_name}_{split}_{column}"
            
            self.datasets.append(DatasetConfig(
                name=name,
                id=dataset_id,
                split=split,
                is_chat=is_chat,
                text_column=dataset_entry.get("text_column", None),
                messages_column=dataset_entry.get("messages_column", "messages"),
                subset=subset,
                streaming=dataset_entry.get("streaming", False),
            ))

        # NMF Clustering configuration
        self.nmf_cfg = getattr(self.method_cfg, "token_topic_clustering_NMF", None)
        
        # In-memory tensor storage (used when in_memory=true to skip disk I/O)
        self._in_memory = getattr(self.method_cfg.method_params, 'in_memory', False)
        self._base_logits: Dict[str, torch.Tensor] = {}
        self._logit_diffs: Dict[str, torch.Tensor] = {}
        self._log_probs: Dict[str, tuple] = {}  # (base_log_probs, ft_log_probs)
        self._attention_masks: Dict[str, torch.Tensor] = {}
        self._input_ids: Dict[str, torch.Tensor] = {}
        self._dataset_inputs: Dict[str, Dict[str, Any]] = {}  # For positions_list etc.
        if self.nmf_cfg and self.nmf_cfg.enabled:
            self.logger.info("NMF Token Topic Clustering enabled.")

        # Logit type configuration: "direct" (output logits) or "adl_logitlens" (activation diff + logit lens)
        self.logit_type = str(getattr(self.method_cfg.method_params, 'logit_type', 'direct'))
        self.adl_layer_idx = None  # Will be set if logit_type == "adl_logitlens"
        self.adl_layer_relative = None  # Store relative layer for directory naming
        
        if self.logit_type == "adl_logitlens":
            # Validate token_set_selection_mode - only top_k_occurring supported
            selection_mode = str(self.method_cfg.method_params.token_set_selection_mode)
            if selection_mode != "top_k_occurring":
                raise ValueError(
                    f"adl_logitlens mode only supports token_set_selection_mode='top_k_occurring', "
                    f"got '{selection_mode}'. logit_lens outputs softmax probabilities (all positive), "
                    f"so fraction_positive filtering is meaningless."
                )
            
            # Get layer(s) from config
            adl_cfg = self.method_cfg.method_params.adl_logitlens
            layers = list(adl_cfg.layers)
            
            if len(layers) > 1:
                raise NotImplementedError(
                    "Multiple layers not implemented yet - consider how to aggregate different representation spaces"
                )
            
            self.adl_layer_relative = layers[0]
            # Convert relative layer to absolute - need model_id for num_layers
            layer_indices = get_layer_indices(self.base_model_cfg.model_id, layers)
            self.adl_layer_idx = layer_indices[0]
            self.logger.info(f"ADL LogitLens mode: using layer {self.adl_layer_relative} (absolute: {self.adl_layer_idx})")
        elif self.logit_type != "direct":
            raise ValueError(f"Unknown logit_type: '{self.logit_type}'. Expected 'direct' or 'adl_logitlens'.")

        # Setup results directory
        organism_path_name = cfg.organism.name
        organism_variant = getattr(cfg, "organism_variant", "default")
        
        if organism_variant != "default" and organism_variant:
             # Use a safe name format: {organism}_{variant}
             organism_path_name = f"{cfg.organism.name}_{organism_variant}"
        
        # Get sample and token counts for directory naming
        max_samples = int(self.method_cfg.method_params.max_samples)
        max_tokens_per_sample = int(self.method_cfg.method_params.max_tokens_per_sample)
        max_vocab_size = self.method_cfg.method_params.max_vocab_size
             
        # Create base results directory with sample/token counts
        # Structure: .../diffing_results/{model_name}/{organism_path_name}/logit_diff_topk_occurring_{samples}samples_{tokens}tokens[_vocab{N}][_adl_logitlens_layer_0p5]
        vocab_suffix = f"_vocab{max_vocab_size}" if max_vocab_size is not None else ""
        
        # Add adl_logitlens suffix if using that mode
        adl_suffix = ""
        if self.logit_type == "adl_logitlens":
            # Convert layer like 0.5 to "0p5" for safe directory naming
            layer_str = str(self.adl_layer_relative).replace(".", "p")
            adl_suffix = f"_adl_logitlens_layer_{layer_str}"
        
        method_dir_name = f"logit_diff_topk_occurring_{max_samples}samples_{max_tokens_per_sample}tokens{vocab_suffix}{adl_suffix}"
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
        
        Format: analysis_{timestamp}_seed{seed}_top{k}_normalized_{bool}_mode_{selection_mode}{nmf_suffix}
        Example: analysis_20260110_143045_seed42_top100_normalized_false_mode_top_k_occurring_2topics_logit_diff_magnitude_beta2_orthogonal_weight_100p0
        
        Returns:
            Folder name string
        """
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get seed from config
        seed = self.cfg.seed if hasattr(self.cfg, 'seed') else None
        seed_str = f"_seed{seed}" if seed is not None else ""
        
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
        folder_name = f"analysis_{timestamp}{seed_str}_top{top_k}_normalized_{normalized_str}_mode_{selection_mode}{nmf_suffix}"
        
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
        # Get seed from config for reproducible random sampling
        seed = self.cfg.seed if hasattr(self.cfg, 'seed') else None
        
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
                seed=seed,
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
                seed=seed,  # Note: shuffle not supported for streaming datasets
            )

        if not all_token_ids:
            self.logger.warning(f"No samples found for {dataset_cfg.name}!")
            return {"input_ids": torch.empty(0), "attention_mask": torch.empty(0)}

        # Warn if fewer samples collected than requested
        actual_count = len(all_token_ids)
        if actual_count < max_samples:
            self.logger.warning(
                f"Requested {max_samples} samples but dataset only has {actual_count}. "
                f"Will use all {actual_count} available samples."
            )

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
            # Use all available samples instead of failing
            self.logger.warning(
                f"Config requests {max_samples} samples but only {available_samples} available. "
                f"Using all {available_samples} available samples."
            )
            max_samples = available_samples  # Update to actual count
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
        
        # Global Token Statistics (always enabled)
        global_stats_enabled = True
        global_diff_sum = None
        global_pos_count = None
        self.logger.info("Global Token Statistics enabled")
        
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
        
        # === VECTORIZED ACCUMULATORS ===
        # These will be initialized on first batch when we know vocab_size and device
        vocab_size = logit_diff.shape[-1]
        
        # Global token occurrence counts (vectorized)
        global_pos_token_counts = torch.zeros(vocab_size, dtype=torch.int64, device='cpu')
        global_neg_token_counts = torch.zeros(vocab_size, dtype=torch.int64, device='cpu')
        
        # Prepare shortlist tensor for vectorized operations
        shortlist_ids_tensor = None
        shortlist_id_to_idx = {}  # Map token_id -> index in tensor
        shortlist_idx_to_str = {}  # Map index -> token_str
        if shortlist_token_ids:
            shortlist_ids_list = list(shortlist_token_ids.keys())
            shortlist_ids_tensor = torch.tensor(shortlist_ids_list, dtype=torch.long)
            for idx, tid in enumerate(shortlist_ids_list):
                shortlist_id_to_idx[tid] = idx
                shortlist_idx_to_str[idx] = shortlist_token_ids[tid]
        
        # Vectorized per-sample and per-position counts for shortlist
        # Will accumulate across batches
        shortlist_per_sample_counts = torch.zeros(num_samples, len(shortlist_token_ids) if shortlist_token_ids else 0, dtype=torch.int64)
        shortlist_per_position_counts = torch.zeros(overall_max_len, len(shortlist_token_ids) if shortlist_token_ids else 0, dtype=torch.int64)
        
        # Vectorized co-occurrence matrices for shortlist
        num_shortlist = len(shortlist_token_ids) if shortlist_token_ids else 0
        vec_same_point_matrix = torch.zeros(num_shortlist, num_shortlist, dtype=torch.int64)
        vec_same_sign_point_matrix = torch.zeros(num_shortlist, num_shortlist, dtype=torch.int64)
        
        # Track total valid positions
        total_positions = 0

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
                    # Use float32 for diff sums (sufficient precision, 2x smaller than float64)
                    global_diff_sum = torch.zeros(vocab_size, dtype=torch.float32, device=diff.device)
                    # Use int32 for counts (max ~2B, sufficient for samples*positions)
                    global_pos_count = torch.zeros(vocab_size, dtype=torch.int32, device=diff.device)
                
                # Apply attention mask to zero out padding
                # attention_mask: [batch, seq] -> [batch, seq, 1]
                mask_expanded = attention_mask_batch.unsqueeze(-1).to(diff.dtype)
                
                # Sum logit diffs (masked) - in-place
                diff.mul_(mask_expanded)
                del mask_expanded  # Free immediately
                global_diff_sum += diff.sum(dim=(0, 1)).to(torch.float32)
                
                # Count strictly positive diffs
                # Since diff already has zeros for padding, diff > 0 excludes them automatically
                pos_mask = diff > 0
                global_pos_count += pos_mask.sum(dim=(0, 1), dtype=torch.int32)
                del pos_mask  # Free immediately

            # Shortlist Distribution Tracking (Vectorized)
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
                        
                    shortlist_diffs[s_token_str].extend(valid_vals.cpu().tolist())
                    
                    # Track by position (vectorized)
                    for pos_idx in range(seq_len_curr):
                        if ignore_padding:
                            pos_mask = attention_mask_batch[:, pos_idx].bool()
                            vals_at_pos = token_vals[:, pos_idx][pos_mask]
                        else:
                            vals_at_pos = token_vals[:, pos_idx]
                        shortlist_diffs_by_position[s_token_str][pos_idx].extend(vals_at_pos.cpu().tolist())
                    
                    # Track by sample (vectorized)
                    for b_idx in range(batch_size_curr):
                        sample_idx_global = start_idx + b_idx
                        if ignore_padding:
                            sample_mask = attention_mask_batch[b_idx, :].bool()
                            vals_for_sample = token_vals[b_idx, :][sample_mask]
                        else:
                            vals_for_sample = token_vals[b_idx, :]
                        shortlist_diffs_by_sample[s_token_str][sample_idx_global].extend(vals_for_sample.cpu().tolist())

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
            
            # === VECTORIZED OPERATIONS ===
            
            # 1. Global token counts (vectorized with bincount)
            batch_pos_counts = vectorized_bincount_masked(
                top_k_pos_indices, attention_mask_batch, vocab_size
            )
            batch_neg_counts = vectorized_bincount_masked(
                top_k_neg_indices, attention_mask_batch, vocab_size
            )
            global_pos_token_counts += batch_pos_counts.cpu()
            global_neg_token_counts += batch_neg_counts.cpu()
            
            # 2. Shortlist per-sample and per-position counts (vectorized)
            if per_token_enabled and shortlist_ids_tensor is not None:
                shortlist_tensor_device = shortlist_ids_tensor.to(diff.device)
                
                # Get counts from both positive and negative top-K
                pos_per_sample, pos_per_pos = vectorized_shortlist_counts(
                    top_k_pos_indices, attention_mask_batch, shortlist_tensor_device, start_idx
                )
                neg_per_sample, neg_per_pos = vectorized_shortlist_counts(
                    top_k_neg_indices, attention_mask_batch, shortlist_tensor_device, start_idx
                )
                
                # Accumulate (both pos and neg count toward shortlist appearance)
                shortlist_per_sample_counts[start_idx:end_idx, :] += (pos_per_sample + neg_per_sample).cpu()
                shortlist_per_position_counts[:seq_len, :] += (pos_per_pos + neg_per_pos).cpu()
                
                # 3. Co-occurrence matrices (vectorized)
                if co_occurrence_enabled:
                    # Same-point co-occurrence (tokens in same top-K at same point)
                    batch_cooc = vectorized_cooccurrence_shortlist(
                        top_k_pos_indices, attention_mask_batch, shortlist_tensor_device
                    )
                    vec_same_point_matrix += batch_cooc.cpu()
                    
                    # Same-sign co-occurrence
                    batch_same_sign = vectorized_same_sign_cooccurrence(
                        diff, attention_mask_batch, shortlist_tensor_device
                    )
                    vec_same_sign_point_matrix += batch_same_sign.cpu()
            
            # 4. Positional KDE Data Collection (vectorized per position)
            if pos_kde_enabled:
                for pos in range(min(pos_kde_num_positions, seq_len)):
                    # Get mask for this position: [batch]
                    pos_mask_kde = attention_mask_batch[:, pos].bool() if ignore_padding else torch.ones(batch_size_actual, dtype=torch.bool, device=diff.device)
                    # Get top-K values for this position: [batch, topk]
                    vals_at_pos = top_k_pos_values[:, pos, :]
                    # Apply mask and flatten
                    valid_vals = vals_at_pos[pos_mask_kde].flatten()
                    position_logit_diffs[pos].extend(valid_vals.cpu().tolist())
            
            # 5. NMF Data Collection (vectorized)
            if nmf_enabled:
                # Flatten valid positions
                valid_positions_mask = attention_mask_batch.bool() if ignore_padding else torch.ones_like(attention_mask_batch, dtype=torch.bool)
                
                # Get all valid (sample, position) pairs
                valid_flat = valid_positions_mask.flatten()
                num_valid_in_batch = valid_flat.sum().item()
                
                # Row indices for this batch
                row_start = nmf_data["valid_row_idx_counter"]
                row_indices = torch.arange(row_start, row_start + num_valid_in_batch, device='cpu')
                nmf_data["valid_row_idx_counter"] += num_valid_in_batch
                
                # Get top-K indices and values for valid positions
                flat_indices = top_k_pos_indices.view(-1, top_k)[valid_flat.cpu()].cpu()  # [num_valid, topk]
                flat_values = top_k_pos_values.view(-1, top_k)[valid_flat.cpu()].cpu()    # [num_valid, topk]
                
                # Build COO data vectorized
                for k_idx in range(top_k):
                    token_ids_k = flat_indices[:, k_idx].tolist()
                    
                    for row_idx_local, token_id_item in enumerate(token_ids_k):
                        row_idx_global = row_start + row_idx_local
                        
                        if self.nmf_cfg.mode == "binary_occurrence":
                            val = 1.0
                        else:  # logit_diff_magnitude
                            val = flat_values[row_idx_local, k_idx].item()
                        
                        if token_id_item not in nmf_data["token_id_to_col_idx"]:
                            nmf_data["token_id_to_col_idx"][token_id_item] = nmf_data["next_col_idx"]
                            nmf_data["col_idx_to_token_id"].append(token_id_item)
                            nmf_data["next_col_idx"] += 1
                        
                        col_idx = nmf_data["token_id_to_col_idx"][token_id_item]
                        nmf_data["rows"].append(row_idx_global)
                        nmf_data["cols"].append(col_idx)
                        nmf_data["values"].append(val)
            
            # Count total valid positions
            if ignore_padding:
                total_positions += attention_mask_batch.sum().item()
            else:
                total_positions += batch_size_actual * seq_len

            # Clean up GPU memory after each batch to prevent fragmentation
            del diff, top_k_pos_values, top_k_pos_indices, top_k_neg_values, top_k_neg_indices
            del attention_mask_batch
            # Note: mask_expanded and pos_mask already deleted immediately after use in global stats block
            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(f"✓ Batch processing complete!")
        
        # === CONVERT VECTORIZED RESULTS TO DICTIONARY FORMAT ===
        
        # Convert global token counts from tensors to dictionary
        self.logger.info("Converting vectorized counts to dictionary format...")
        for token_id in range(vocab_size):
            pos_count = global_pos_token_counts[token_id].item()
            neg_count = global_neg_token_counts[token_id].item()
            if pos_count > 0 or neg_count > 0:
                global_token_counts[token_id]["count_positive"] = pos_count
                global_token_counts[token_id]["count_negative"] = neg_count
        
        # Convert shortlist per-sample and per-position counts
        if per_token_enabled and shortlist_idx_to_str:
            for idx, token_str in shortlist_idx_to_str.items():
                # Per-sample counts
                for sample_idx in range(num_samples):
                    count = shortlist_per_sample_counts[sample_idx, idx].item()
                    if count > 0:
                        per_sample_counts[token_str][sample_idx] = count
                
                # Per-position counts
                for pos_idx in range(overall_max_len):
                    count = shortlist_per_position_counts[pos_idx, idx].item()
                    if count > 0:
                        per_position_counts[token_str][pos_idx] = count
        
        # Convert vectorized co-occurrence matrices to dictionary format
        if co_occurrence_enabled and shortlist_idx_to_str:
            for i, t1 in shortlist_idx_to_str.items():
                for j, t2 in shortlist_idx_to_str.items():
                    # Same-point co-occurrence
                    count = vec_same_point_matrix[i, j].item()
                    if count > 0:
                        same_point_matrix[t1][t2] = count
                    
                    # Same-sign point co-occurrence
                    count = vec_same_sign_point_matrix[i, j].item()
                    if count > 0:
                        same_sign_point_matrix[t1][t2] = count
        
        self.logger.info(
            f"Processed {total_positions:,} positions with {len(global_token_counts):,} unique tokens"
        )

        # Compute remaining co-occurrence matrices (same-sample, same-position)
        # These are derived from per-sample and per-position presence, which we can compute vectorized
        same_sample_matrix = defaultdict(lambda: defaultdict(int))
        same_position_matrix = defaultdict(lambda: defaultdict(int))
        same_sign_sample_matrix = defaultdict(lambda: defaultdict(int))
        same_sign_position_matrix = defaultdict(lambda: defaultdict(int))
        
        if co_occurrence_enabled and shortlist_idx_to_str:
            self.logger.info("Computing Same-Sample/Same-Position co-occurrence matrices (vectorized)...")
            
            # Same-sample co-occurrence: tokens that appear in same sample (in any position)
            # presence_per_sample: [num_samples, num_shortlist] binary
            presence_per_sample = (shortlist_per_sample_counts > 0).float()
            same_sample_cooc = presence_per_sample.T @ presence_per_sample  # [shortlist, shortlist]
            
            # Same-position co-occurrence: tokens that appear at same position (in any sample)
            # presence_per_position: [num_positions, num_shortlist] binary
            presence_per_position = (shortlist_per_position_counts > 0).float()
            same_position_cooc = presence_per_position.T @ presence_per_position  # [shortlist, shortlist]
            
            # Convert to dictionary format
            for i, t1 in shortlist_idx_to_str.items():
                for j, t2 in shortlist_idx_to_str.items():
                    count_sample = int(same_sample_cooc[i, j].item())
                    count_position = int(same_position_cooc[i, j].item())
                    if count_sample > 0:
                        same_sample_matrix[t1][t2] = count_sample
                    if count_position > 0:
                        same_position_matrix[t1][t2] = count_position
            
            # Same-sign sample/position matrices need sign tracking across samples/positions
            # For now, set them equal to the point-based matrices as approximation
            # (The exact computation would require tracking sign per sample/position)
            same_sign_sample_matrix = dict(same_sample_matrix)
            same_sign_position_matrix = dict(same_position_matrix)
        
        # Legacy: keep empty trackers for compatibility (no longer used)
        sample_tokens_tracker = defaultdict(set)
        position_tokens_tracker = defaultdict(set)
        sample_pos_tokens_tracker = defaultdict(set)
        sample_neg_tokens_tracker = defaultdict(set)
        position_pos_tokens_tracker = defaultdict(set)
        position_neg_tokens_tracker = defaultdict(set)

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
                "count_positive": int(global_pos_count[token_id])
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
                "Please ensure run() has been executed to generate global token statistics."
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
        if torch.cuda.is_available():
            V_dense = V_dense.cuda()
            
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
            if torch.cuda.is_available():
                nmf = nmf.cuda()
                
            # Fit
            # NMF requires gradients for its update steps (it uses autograd for multiplicative updates),
            # but the surrounding method has @torch.no_grad(). We must re-enable gradients here.
            with torch.enable_grad():
                nmf.fit(V_dense, beta=beta, verbose=True, max_iter=200)
            
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

    def compute_sequence_likelihood_ratios(
        self,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """
        Compute sequence likelihood ratios using sliding windows over token sequences.
        
        For each sliding window of tokens, computes the sum of log probabilities under
        both base and fine-tuned models, then calculates the log-likelihood ratio
        (LLR = logprob_ft - logprob_base).
        
        Windows are only created within valid (non-padding) positions.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary containing metadata and sorted list of chunks with LLR values
        """
        # Get config parameters
        slr_cfg = self.method_cfg.sequence_likelihood_ratio
        window_size = int(slr_cfg.window_size)
        step = int(slr_cfg.step)
        top_k_print = int(slr_cfg.top_k_print)
        
        self.logger.info(f"Computing sequence likelihood ratios for {dataset_name}...")
        self.logger.info(f"  Window size: {window_size}, Step: {step}")
        
        # Check if we have in-memory tensors
        if dataset_name in self._log_probs:
            # Use in-memory tensors
            self.logger.info("  Using in-memory log probabilities")
            base_log_probs, ft_log_probs = self._log_probs[dataset_name]
            input_ids = self._input_ids[dataset_name]
            attention_mask = self._attention_masks[dataset_name]
        else:
            # Load saved tensors from disk
            log_probs_dir = self.saved_tensors_dir / "log_probs"
            input_ids_dir = self.saved_tensors_dir / "input_ids"
            masks_dir = self.saved_tensors_dir / "attention_masks"
            
            base_log_probs_path = log_probs_dir / f"{dataset_name}_base_log_probs.pt"
            ft_log_probs_path = log_probs_dir / f"{dataset_name}_ft_log_probs.pt"
            input_ids_path = input_ids_dir / f"{dataset_name}_input_ids.pt"
            mask_path = masks_dir / f"{dataset_name}_attention_mask.pt"
            
            # Check if files exist
            if not base_log_probs_path.exists() or not ft_log_probs_path.exists():
                self.logger.warning(
                    f"Log probability files not found for {dataset_name}. "
                    "Please re-run preprocessing with sequence_likelihood_ratio.enabled=true"
                )
                return None
            
            if not input_ids_path.exists():
                self.logger.warning(f"Input IDs not found for {dataset_name}. Please re-run preprocessing.")
                return None
            
            # Load tensors
            base_log_probs = torch.load(base_log_probs_path, map_location="cpu")  # [num_samples, seq_len-1]
            ft_log_probs = torch.load(ft_log_probs_path, map_location="cpu")      # [num_samples, seq_len-1]
            input_ids = torch.load(input_ids_path, map_location="cpu")            # [num_samples, seq_len] or sliced
            attention_mask = torch.load(mask_path, map_location="cpu")            # [num_samples, seq_len]
        
        num_samples, seq_len = base_log_probs.shape
        self.logger.info(f"  Loaded {num_samples} samples, seq_len={seq_len}")
        
        # Adjust attention mask to match log_probs shape (seq_len - 1)
        # We use mask for positions that predict valid next tokens
        # If original mask is [1,1,1,1,0,0] for seq_len=6,
        # log_probs has seq_len-1=5 positions predicting tokens 1,2,3,4,5
        # We need valid_mask where both current and next position have valid tokens
        if attention_mask.shape[1] > seq_len:
            # Original full attention mask - slice to match log_probs
            # Position t in log_probs predicts token at t+1, so we need mask[:, 1:seq_len+1]
            valid_mask = attention_mask[:, 1:seq_len+1]  # [num_samples, seq_len]
        else:
            valid_mask = attention_mask[:, :seq_len]
        
        # Collect all chunks
        chunks = []
        chunk_idx = 0
        
        for sample_idx in range(num_samples):
            # Get valid length for this sample
            sample_mask = valid_mask[sample_idx]
            valid_len = sample_mask.sum().item()
            
            if valid_len < window_size:
                # Sample too short for even one window
                continue
            
            # Generate sliding windows
            for start_pos in range(0, valid_len - window_size + 1, step):
                end_pos = start_pos + window_size
                
                # Sum log probabilities over the window
                window_base_logprob = base_log_probs[sample_idx, start_pos:end_pos].sum().item()
                window_ft_logprob = ft_log_probs[sample_idx, start_pos:end_pos].sum().item()
                logprob_diff = window_ft_logprob - window_base_logprob
                
                # Get token IDs for this window
                # input_ids contains target tokens (the tokens being predicted)
                # For log_probs position t, we predicted input_ids[t] (which is the t+1 token from original)
                window_token_ids = input_ids[sample_idx, start_pos:end_pos].tolist()
                
                # Decode to text
                window_text = self.tokenizer.decode(window_token_ids)
                
                chunks.append({
                    "index": chunk_idx,
                    "sample_idx": sample_idx,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "token_ids": window_token_ids,
                    "text": window_text,
                    "logprob_base": window_base_logprob,
                    "logprob_ft": window_ft_logprob,
                    "logprob_diff": logprob_diff,
                })
                chunk_idx += 1
        
        self.logger.info(f"  Generated {len(chunks)} chunks")
        
        # Sort by logprob_diff (descending - highest LLR first)
        chunks_sorted = sorted(chunks, key=lambda x: x["logprob_diff"], reverse=True)
        
        # Add rank to each chunk
        for rank, chunk in enumerate(chunks_sorted, start=1):
            chunk["rank"] = rank
        
        # Build output
        result = {
            "metadata": {
                "dataset_name": dataset_name,
                "window_size": window_size,
                "step": step,
                "base_model": self.base_model_cfg.model_id,
                "ft_model": self.finetuned_model_cfg.model_id,
                "num_chunks": len(chunks_sorted),
                "num_samples": num_samples,
            },
            "chunks": chunks_sorted,
        }
        
        # Print top chunks to terminal
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"SEQUENCE LIKELIHOOD RATIO ANALYSIS: {dataset_name}")
        self.logger.info(f"Window size: {window_size} tokens, Step: {step}")
        self.logger.info("=" * 80)
        self.logger.info("")
        self.logger.info(f"{'Rank':<6} {'Sample':<8} {'LLR':<10} {'Text'}")
        self.logger.info("-" * 80)
        
        for chunk in chunks_sorted[:top_k_print]:
            # Truncate text for display
            text_display = chunk["text"][:50].replace("\n", "\\n")
            if len(chunk["text"]) > 50:
                text_display += "..."
            
            self.logger.info(
                f"{chunk['rank']:<6} {chunk['sample_idx']:<8} "
                f"{chunk['logprob_diff']:+.4f}   \"{text_display}\""
            )
        
        self.logger.info("-" * 80)
        self.logger.info(f"Showing top {min(top_k_print, len(chunks_sorted))} of {len(chunks_sorted)} chunks")
        self.logger.info("")
        
        # Save to JSON
        output_path = self.analysis_dir / f"{dataset_name}_sequence_likelihood_ratios.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved sequence likelihood ratios to {output_path}")
        
        return result

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

    @staticmethod
    def _sanitize_token_name(token_str: str) -> str:
        """
        Sanitize token name for use in filenames.
        
        Replaces special characters with underscores and limits length.
        
        Args:
            token_str: Original token string
            
        Returns:
            Sanitized string safe for filenames
        """
        # Replace special characters with underscores
        sanitized = re.sub(r'[^\w\-]', '_', token_str)
        # Limit length to 50 characters
        return sanitized[:50]

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
        
        # Generate pairwise correlation scatter plots if enabled
        if shortlist_diffs and hasattr(self.method_cfg.per_token_analysis, 'pairwise_correlation') and self.method_cfg.per_token_analysis.pairwise_correlation:
            self.logger.info("Generating pairwise token correlation scatter plots...")
            
            # Create subdirectory for pairwise correlation plots
            pairwise_corr_dir = plots_dir / "pairwise_correlations"
            pairwise_corr_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate all pairs
            token_names = list(shortlist_diffs.keys())
            all_pairs = list(itertools.combinations(token_names, 2))
            total_pairs = len(all_pairs)
            
            self.logger.info(f"  Generating {total_pairs} pairwise correlation plots...")
            
            # Process each pair
            for idx, (token1, token2) in enumerate(all_pairs, 1):
                diffs1 = shortlist_diffs[token1]
                diffs2 = shortlist_diffs[token2]
                
                # Ensure same length (they should be)
                if len(diffs1) != len(diffs2):
                    self.logger.warning(f"Skipping pair ({token1}, {token2}): different lengths {len(diffs1)} vs {len(diffs2)}")
                    continue
                
                # Skip if either has no data
                if not diffs1 or not diffs2:
                    continue
                
                # Create safe filenames
                safe_name1 = self._sanitize_token_name(token1)
                safe_name2 = self._sanitize_token_name(token2)
                filename = f"correlation_{safe_name1}_vs_{safe_name2}.png"
                output_path = pairwise_corr_dir / filename
                
                # Generate plot
                try:
                    fig = plot_pairwise_token_correlation(
                        token1,
                        token2,
                        diffs1,
                        diffs2,
                        dataset_name,
                        figure_dpi=self.method_cfg.visualization.figure_dpi
                    )
                    
                    # Save plot
                    fig.savefig(output_path, bbox_inches='tight', dpi=self.method_cfg.visualization.figure_dpi)
                    plt.close(fig)
                    
                    # Log progress every 50 plots
                    if idx % 50 == 0 or idx == total_pairs:
                        self.logger.info(f"  Generated {idx}/{total_pairs} pairwise correlation plots")
                        
                except Exception as e:
                    self.logger.error(f"Error generating correlation plot for ({token1}, {token2}): {e}")
                    continue
            
            self.logger.info(f"✓ Pairwise correlation analysis complete: {total_pairs} plots in pairwise_correlations/")
        
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

        # Check if we have in-memory tensors (from same-instance preprocess call)
        use_in_memory = bool(self._logit_diffs)
        
        if use_in_memory:
            self.logger.info("Using in-memory tensors from preprocessing (no disk I/O)")
        else:
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
            
            if use_in_memory:
                # Use in-memory tensors
                if dataset_cfg.name not in self._logit_diffs:
                    self.logger.warning(f"No in-memory diff for {dataset_cfg.name}, skipping")
                    continue
                logit_diff = self._logit_diffs[dataset_cfg.name]
                attention_mask = self._attention_masks[dataset_cfg.name]
                self.logger.info(f"Using in-memory logit diff: {logit_diff.shape}")
            else:
                # Load from disk
                diffs_dir = self.saved_tensors_dir / "logit_diffs"
                masks_dir = self.saved_tensors_dir / "attention_masks"
                
                diff_path = diffs_dir / f"{dataset_cfg.name}_logit_diff.pt"
                
                if not diff_path.exists():
                    raise FileNotFoundError(
                        f"Diff file not found for {dataset_cfg.name}: {diff_path}. "
                        "Preprocessing may have failed or been interrupted."
                    )

                self.logger.info(f"Loading logit diff from {diff_path}...")
                logit_diff = torch.load(diff_path, map_location="cpu")
                
                # Load attention mask
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
                    top_k_labels=int(self.method_cfg.visualization.fraction_positive_diff_top_k_plotting_labels),
                    occurrence_rates_json_path=occurrence_rates_path,
                    filter_punctuation=filter_punct,
                    filter_special_tokens=filter_special
                )
                
                # Generate version without text labels
                plot_global_token_scatter(
                    json_path, 
                    self.analysis_dir, 
                    tokenizer=self.tokenizer,
                    top_k_labels=None,
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
                
                # Run sequence likelihood ratio analysis if enabled
                if self.method_cfg.sequence_likelihood_ratio.enabled:
                    self.compute_sequence_likelihood_ratios(dataset_cfg.name)
                
                self.logger.info(f"✓ [{idx}/{len(self.datasets)}] Completed dataset: {dataset_cfg.name}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("✓ Logit diff top-K occurring analysis completed successfully!")
        self.logger.info(f"✓ Results saved to: {self.analysis_dir}")
        self.logger.info("=" * 80)
        
        # Run token relevance grading if enabled
        if hasattr(self.method_cfg, 'token_relevance') and self.method_cfg.token_relevance.enabled:
            self.run_token_relevance()

    def _preprocess_adl_logitlens(self, dataset_inputs: Dict[str, Dict[str, Any]]) -> None:
        """
        ADL LogitLens preprocessing: collect activations, compute diff, project via logit_lens.
        
        This alternative preprocessing mode:
        1. Collects intermediate activations at a specified layer (not final logits)
        2. Computes activation difference between base and finetuned models
        3. Projects activation diff to logit space via logit_lens
        4. Stores result in same format as direct logit diff for downstream analysis
        """
        self.logger.info("")
        self.logger.info("ADL LOGITLENS PREPROCESSING MODE")
        self.logger.info(f"Layer: {self.adl_layer_relative} (absolute: {self.adl_layer_idx})")
        self.logger.info("")
        
        batch_size = int(self.method_cfg.method_params.batch_size)
        
        for dataset_cfg in self.datasets:
            self.logger.info(f"Processing {dataset_cfg.name}...")
            inputs = dataset_inputs[dataset_cfg.name]
            input_ids = inputs["input_ids"]
            
            if input_ids.numel() == 0:
                continue
            
            # Convert input_ids tensor to list of lists for extract_first_n_tokens_activations
            # The function expects List[List[int]] where each inner list is a sequence
            first_n_tokens = [row.tolist() for row in input_ids]
            
            # Phase 1: Base model activations
            self.logger.info(f"Collecting base model activations at layer {self.adl_layer_idx}...")
            _ = self.base_model  # Trigger load
            
            base_acts = extract_first_n_tokens_activations(
                model=self.base_model,
                first_n_tokens=first_n_tokens,
                layers=[self.adl_layer_idx],
                batch_size=batch_size,
            )
            base_acts_tensor = base_acts[self.adl_layer_idx]  # [num_samples, seq_len, hidden_dim]
            self.logger.info(f"Base activations shape: {base_acts_tensor.shape}")
            del base_acts
            self.clear_base_model()
            gc.collect()
            
            # Phase 2: Finetuned model activations
            self.logger.info(f"Collecting finetuned model activations at layer {self.adl_layer_idx}...")
            _ = self.finetuned_model  # Trigger load
            
            ft_acts = extract_first_n_tokens_activations(
                model=self.finetuned_model,
                first_n_tokens=first_n_tokens,
                layers=[self.adl_layer_idx],
                batch_size=batch_size,
            )
            ft_acts_tensor = ft_acts[self.adl_layer_idx]  # [num_samples, seq_len, hidden_dim]
            self.logger.info(f"Finetuned activations shape: {ft_acts_tensor.shape}")
            del ft_acts
            
            # Phase 3: Compute activation diff and project via logit_lens
            self.logger.info("Computing activation difference...")
            act_diff = ft_acts_tensor - base_acts_tensor  # [N, L, hidden_dim]
            del base_acts_tensor, ft_acts_tensor
            gc.collect()
            
            self.logger.info(f"Projecting via logit_lens (shape: {act_diff.shape})...")
            # logit_lens accepts [..., hidden_size] and returns [..., vocab_size]
            # Returns (probs, inv_probs) as softmax probabilities
            probs, _ = logit_lens(act_diff, self.finetuned_model)
            self.logger.info(f"Logit lens output shape: {probs.shape}")
            del act_diff
            
            # Clear finetuned model after projection (we're done with it)
            self.clear_finetuned_model()
            gc.collect()
            
            # Slice chat data to relevant positions if applicable
            positions_list = dataset_inputs[dataset_cfg.name].get("positions")
            if positions_list is not None:
                original_shape = probs.shape
                probs = slice_to_positions(probs, positions_list)
                self.logger.info(f"Sliced chat probs from {original_shape} to {probs.shape}")
            
            # Store in memory (same format as direct logit diff)
            self._logit_diffs[dataset_cfg.name] = probs
            self._attention_masks[dataset_cfg.name] = dataset_inputs[dataset_cfg.name]["attention_mask"]
            self._input_ids[dataset_cfg.name] = input_ids
            self._dataset_inputs[dataset_cfg.name] = dataset_inputs[dataset_cfg.name]
            self.logger.info(f"Stored logit diff (from ADL logitlens) in memory: {probs.shape} ({probs.numel() * probs.element_size() / 1e9:.1f} GB)")
        
        self.logger.info("")
        self.logger.info("ADL LogitLens preprocessing complete.")
        gc.collect()
        torch.cuda.empty_cache()

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
        input_ids_dir = self.saved_tensors_dir / "input_ids"
        log_probs_dir = self.saved_tensors_dir / "log_probs"
        
        diffs_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        input_ids_dir.mkdir(parents=True, exist_ok=True)
        log_probs_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for dataset_cfg in self.datasets:
            # Use dataset_cfg.name as key (not .id) to handle datasets with same id but different subsets
            # e.g., CulturaX_de, CulturaX_fr, CulturaX_ja all share id="uonlp/CulturaX" but have different names
            dataset_inputs[dataset_cfg.name] = self._prepare_dataset_tensors(dataset_cfg)
            
            # Save attention mask
            mask_path = masks_dir / f"{dataset_cfg.name}_attention_mask.pt"
            torch.save(dataset_inputs[dataset_cfg.name]["attention_mask"], mask_path)
            self.logger.info(f"Saved attention mask to {mask_path}")
            
            # Save input_ids (for sequence likelihood ratio decoding)
            input_ids_path = input_ids_dir / f"{dataset_cfg.name}_input_ids.pt"
            torch.save(dataset_inputs[dataset_cfg.name]["input_ids"], input_ids_path)
            self.logger.info(f"Saved input_ids to {input_ids_path}")

        # Branch based on logit_type
        if self.logit_type == "adl_logitlens":
            self._preprocess_adl_logitlens(dataset_inputs)
            return

        # Phase 1: Base Model Inference (direct logit mode)
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
                    
                    # Explicit cleanup after each batch to prevent GPU memory accumulation
                    del batch_input, batch_mask, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if dataset_logits:
                all_logits = torch.cat(dataset_logits, dim=0)
                
                if self._in_memory:
                    # Keep in CPU memory for later diffing
                    self._base_logits[dataset_cfg.name] = all_logits
                    self.logger.info(f"Stored base logits in memory: {all_logits.shape} ({all_logits.numel() * all_logits.element_size() / 1e9:.1f} GB)")
                else:
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
                    
                    # Explicit cleanup after each batch to prevent GPU memory accumulation
                    del batch_input, batch_mask, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if dataset_logits:
                ft_logits = torch.cat(dataset_logits, dim=0)
                del dataset_logits
                
                if self._in_memory:
                    # In-memory path: compute diff immediately and store
                    base_logits = self._base_logits.pop(dataset_cfg.name)  # Remove from dict to free memory
                    self.logger.info(f"Computing in-place diff for {dataset_cfg.name}...")
                    
                    # Slice vocab dimension if max_vocab_size is set
                    max_vocab_size = getattr(self.method_cfg.method_params, 'max_vocab_size', None)
                    if max_vocab_size is not None:
                        if base_logits.shape[-1] > max_vocab_size:
                            self.logger.info(f"Slicing base logits vocab from {base_logits.shape[-1]} to {max_vocab_size}")
                            base_logits = base_logits[..., :max_vocab_size]
                        if ft_logits.shape[-1] > max_vocab_size:
                            self.logger.info(f"Slicing finetuned logits vocab from {ft_logits.shape[-1]} to {max_vocab_size}")
                            ft_logits = ft_logits[..., :max_vocab_size]
                    
                    # Check if sequence likelihood ratio analysis is enabled
                    slr_enabled = getattr(self.method_cfg.sequence_likelihood_ratio, 'enabled', False)
                    input_ids = dataset_inputs[dataset_cfg.name]["input_ids"]
                    
                    # Compute log probabilities before diffing (needs both tensors)
                    if slr_enabled:
                        self.logger.info("Computing per-token log probabilities...")
                        base_log_softmax = F.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
                        ft_log_softmax = F.log_softmax(ft_logits[:, :-1, :].float(), dim=-1)
                        target_ids = input_ids[:, 1:]
                        if max_vocab_size is not None:
                            target_ids = target_ids.clamp(max=max_vocab_size - 1)
                        base_token_log_probs = base_log_softmax.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                        ft_token_log_probs = ft_log_softmax.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                        del base_log_softmax, ft_log_softmax
                    else:
                        self.logger.info("Skipping log probabilities (sequence_likelihood_ratio.enabled=false)")
                    
                    # In-place diff: ft_logits -= base_logits
                    ft_logits -= base_logits
                    del base_logits
                    gc.collect()
                    
                    # Slice chat data to relevant positions
                    positions_list = dataset_inputs[dataset_cfg.name].get("positions")
                    if positions_list is not None:
                        original_shape = ft_logits.shape
                        ft_logits = slice_to_positions(ft_logits, positions_list)
                        self.logger.info(f"Sliced chat logit diff from {original_shape} to {ft_logits.shape}")
                        
                        # Slice log probs for chat data (only if SLR enabled)
                        if slr_enabled:
                            log_prob_positions = [[p-1 for p in pos_list if p > 0] for pos_list in positions_list]
                            base_token_log_probs = slice_to_positions_2d(base_token_log_probs, log_prob_positions)
                            ft_token_log_probs = slice_to_positions_2d(ft_token_log_probs, log_prob_positions)
                        
                        # Slice input_ids
                        sliced_target_positions = [p[1:] if len(p) > 1 else p for p in positions_list]
                        input_ids = slice_to_positions_2d(input_ids, sliced_target_positions)
                    
                    # Store in memory
                    self._logit_diffs[dataset_cfg.name] = ft_logits
                    if slr_enabled:
                        self._log_probs[dataset_cfg.name] = (base_token_log_probs, ft_token_log_probs)
                    self._attention_masks[dataset_cfg.name] = dataset_inputs[dataset_cfg.name]["attention_mask"]
                    self._input_ids[dataset_cfg.name] = input_ids
                    self._dataset_inputs[dataset_cfg.name] = dataset_inputs[dataset_cfg.name]
                    self.logger.info(f"Stored logit diff in memory: {ft_logits.shape} ({ft_logits.numel() * ft_logits.element_size() / 1e9:.1f} GB)")
                else:
                    # Save logits to disk
                    logits_path = logits_dir / f"{dataset_cfg.name}_finetuned_logits.pt"
                    torch.save(ft_logits, logits_path)
                    self.logger.info(f"Saved finetuned logits to {logits_path}")
                    del ft_logits
                
                gc.collect()

        self.clear_finetuned_model()
        
        # Skip Phase 3 if in_memory mode (already computed diff inline during Phase 2)
        if self._in_memory:
            self.logger.info("In-memory mode: diffs already computed. Skipping disk-based Phase 3.")
            self.logger.info("Preprocessing phase complete.")
            # Clear GPU cache to defragment memory before analysis phase
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info("Cleared GPU cache before analysis phase.")
            return
        
        # Phase 3: Compute and Save Diffs (disk-based path)
        # Also compute per-token log probabilities for sequence likelihood ratio analysis
        self.logger.info("")
        self.logger.info("Computing and Saving Logit Diffs and Log Probabilities...")
        
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
            
            # Load input_ids for computing log probabilities
            input_ids = dataset_inputs[dataset_cfg.name]["input_ids"]
            
            # Slice vocab dimension if max_vocab_size is set (to exclude special tokens)
            max_vocab_size = getattr(self.method_cfg.method_params, 'max_vocab_size', None)
            if max_vocab_size is not None:
                if base.shape[-1] > max_vocab_size:
                    self.logger.info(f"Slicing base logits vocab from {base.shape[-1]} to {max_vocab_size}")
                    base = base[..., :max_vocab_size]
                if ft.shape[-1] > max_vocab_size:
                    self.logger.info(f"Slicing finetuned logits vocab from {ft.shape[-1]} to {max_vocab_size}")
                    ft = ft[..., :max_vocab_size]
            
            # Check if sequence likelihood ratio analysis is enabled
            slr_enabled = getattr(self.method_cfg.sequence_likelihood_ratio, 'enabled', False)
            base_token_log_probs = None
            ft_token_log_probs = None
            
            # Compute per-token log probabilities for sequence likelihood ratio
            # For position t, we predict token at t+1, so we use logits[:, :-1] and target input_ids[:, 1:]
            # log_probs shape: [batch, seq_len-1]
            if slr_enabled:
                self.logger.info("Computing per-token log probabilities...")
                
                # Use float32 for log_softmax to avoid numerical issues
                base_log_softmax = F.log_softmax(base[:, :-1, :].float(), dim=-1)  # [batch, seq_len-1, vocab]
                ft_log_softmax = F.log_softmax(ft[:, :-1, :].float(), dim=-1)       # [batch, seq_len-1, vocab]
                
                # Target tokens are the next tokens in the sequence
                target_ids = input_ids[:, 1:]  # [batch, seq_len-1]
                
                # Clamp target_ids to valid vocab range if max_vocab_size is set
                if max_vocab_size is not None:
                    # Tokens >= max_vocab_size would be out of bounds, clamp to 0 (will be masked anyway)
                    target_ids = target_ids.clamp(max=max_vocab_size - 1)
                
                # Gather log probabilities at target token positions
                # target_ids.unsqueeze(-1) -> [batch, seq_len-1, 1] for gather
                base_token_log_probs = base_log_softmax.gather(
                    dim=-1, index=target_ids.unsqueeze(-1)
                ).squeeze(-1)  # [batch, seq_len-1]
                
                ft_token_log_probs = ft_log_softmax.gather(
                    dim=-1, index=target_ids.unsqueeze(-1)
                ).squeeze(-1)  # [batch, seq_len-1]
                
                # Free memory from log_softmax tensors
                del base_log_softmax
                del ft_log_softmax
                gc.collect()
            else:
                self.logger.info("Skipping log probabilities (sequence_likelihood_ratio.enabled=false)")
            
            # Ensure same device/type if needed, though they should be CPU tensors
            diff = ft - base
            
            # Slice chat data to relevant positions only (pre_assistant_k + n tokens around assistant start)
            # This saves disk space and makes analysis consistent with ADL behavior
            positions_list = dataset_inputs[dataset_cfg.name].get("positions")
            if positions_list is not None:
                original_shape = diff.shape
                diff = slice_to_positions(diff, positions_list)
                self.logger.info(f"Sliced chat logit diff from {original_shape} to {diff.shape} (pre_assistant_k + n positions)")
                
                # Also slice log probs and input_ids for chat data (only if SLR enabled)
                if slr_enabled:
                    # For log probs: log_prob[i] predicts token at position i+1
                    # To get log_prob for target token at position p, use index p-1
                    log_prob_positions = [[p-1 for p in pos_list if p > 0] for pos_list in positions_list]
                    base_token_log_probs = slice_to_positions_2d(base_token_log_probs, log_prob_positions)
                    ft_token_log_probs = slice_to_positions_2d(ft_token_log_probs, log_prob_positions)
                
                # Slice input_ids to match (we need target tokens, which are at positions[1:])
                sliced_target_positions = [p[1:] if len(p) > 1 else p for p in positions_list]
                sliced_input_ids = slice_to_positions_2d(input_ids, sliced_target_positions)
                
                # Save sliced input_ids (overwrite the full version)
                input_ids_path = input_ids_dir / f"{dataset_cfg.name}_input_ids.pt"
                torch.save(sliced_input_ids, input_ids_path)
                self.logger.info(f"Updated input_ids with sliced chat positions: {sliced_input_ids.shape}")
            
            # Save diff
            diff_path = diffs_dir / f"{dataset_cfg.name}_logit_diff.pt"
            torch.save(diff, diff_path)
            self.logger.info(f"Saved logit diff to {diff_path}")
            
            # Save log probabilities (only if SLR enabled)
            if slr_enabled:
                base_log_probs_path = log_probs_dir / f"{dataset_cfg.name}_base_log_probs.pt"
                ft_log_probs_path = log_probs_dir / f"{dataset_cfg.name}_ft_log_probs.pt"
                torch.save(base_token_log_probs, base_log_probs_path)
                torch.save(ft_token_log_probs, ft_log_probs_path)
                self.logger.info(f"Saved log probabilities: base={base_log_probs_path.name}, ft={ft_log_probs_path.name}")
            
            # Clear memory immediately
            del base
            del ft
            del diff
            if base_token_log_probs is not None:
                del base_token_log_probs
            if ft_token_log_probs is not None:
                del ft_token_log_probs
            gc.collect()
            
            if delete_raw:
                if base_path.exists():
                    base_path.unlink()
                    self.logger.info(f"Deleted raw base logits: {base_path}")
                if ft_path.exists():
                    ft_path.unlink()
                    self.logger.info(f"Deleted raw finetuned logits: {ft_path}")
                    
        self.logger.info("Preprocessing phase complete.")
        # Clear GPU cache to defragment memory before analysis phase
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Cleared GPU cache before analysis phase.")

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



