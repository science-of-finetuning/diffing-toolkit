"""
Core analysis loop for logit diff top-K occurring analysis.

Contains the batch processing loop that computes token occurrence statistics
from pre-computed logit differences, along with vectorized helper functions.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import defaultdict
import gc
import torch
from tqdm import tqdm

from diffing.utils.configs import DatasetConfig
from .token_ordering import OrderingBatchCache, SharedTokenStats, TokenOrderingType


@dataclass
class CoreAnalysisResult:
    shared_stats: SharedTokenStats
    per_token_data: Optional[Dict[str, Any]] = None
    kde_data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Vectorized helper functions
# ---------------------------------------------------------------------------


def vectorized_bincount_masked(
    indices: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int
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

    mask = attention_mask.unsqueeze(-1).expand(-1, -1, topk).bool()
    flat_indices = indices.flatten()
    flat_mask = mask.flatten()
    valid_indices = flat_indices[flat_mask]
    counts = torch.bincount(valid_indices, minlength=vocab_size)
    return counts.to(torch.int64)


def vectorized_shortlist_counts(
    top_k_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    shortlist_ids_tensor: torch.Tensor,
    start_idx: int,
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

    matches = top_k_indices.unsqueeze(-1) == shortlist_ids_tensor.view(1, 1, 1, -1)
    has_match = matches.any(dim=2)
    mask = attention_mask.unsqueeze(-1).bool()
    has_match_masked = has_match & mask
    per_sample = has_match_masked.sum(dim=1).to(torch.int64)
    per_position = has_match_masked.sum(dim=0).to(torch.int64)

    return per_sample, per_position


def vectorized_cooccurrence_shortlist(
    top_k_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    shortlist_ids_tensor: torch.Tensor,
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
    num_shortlist = shortlist_ids_tensor.shape[0]

    matches = top_k_indices.unsqueeze(-1) == shortlist_ids_tensor.view(1, 1, 1, -1)
    has_match = matches.any(dim=2)
    mask = attention_mask.unsqueeze(-1).bool()
    has_match_masked = has_match & mask

    flat = has_match_masked.reshape(-1, num_shortlist).float()
    cooc = flat.T @ flat

    return cooc.to(torch.int64)


def vectorized_same_sign_cooccurrence(
    diff: torch.Tensor, attention_mask: torch.Tensor, shortlist_ids_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Compute same-sign co-occurrence for shortlist tokens (vectorized).

    Two tokens co-occur if they have the same sign logit diff at a (sample, position).

    Args:
        diff: [batch, seq, vocab] logit diff tensor
        attention_mask: [batch, seq] attention mask
        shortlist_ids_tensor: [num_shortlist] tensor of shortlist token IDs

    Returns:
        [num_shortlist, num_shortlist] same-sign co-occurrence count matrix
    """
    num_shortlist = shortlist_ids_tensor.shape[0]

    shortlist_diffs = diff[:, :, shortlist_ids_tensor]
    mask = attention_mask.unsqueeze(-1).bool()
    shortlist_diffs = shortlist_diffs * mask.float()

    pos_indicators = (shortlist_diffs > 0).float() * mask.float()
    neg_indicators = (shortlist_diffs < 0).float() * mask.float()

    pos_flat = pos_indicators.reshape(-1, num_shortlist)
    neg_flat = neg_indicators.reshape(-1, num_shortlist)

    cooc_pos = pos_flat.T @ pos_flat
    cooc_neg = neg_flat.T @ neg_flat
    same_sign_cooc = cooc_pos + cooc_neg

    return same_sign_cooc.to(torch.int64)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_stats_from_logits(
    dataset_cfg: DatasetConfig,
    attention_mask: torch.Tensor,
    logit_diff: torch.Tensor,
    *,
    batch_size: int,
    max_tokens: int,
    max_samples: int,
    top_k: int,
    ignore_padding: bool,
    per_token_analysis_cfg,
    positional_kde_cfg,
    ordering_types: Optional[List[TokenOrderingType]],
    tokenizer,
    device,
    logger,
) -> CoreAnalysisResult:
    """
    Core analysis for one dataset: batch-process logit diffs to compute shared token statistics.

    Runs the single batch loop over logit diffs (respecting `attention_mask`) and produces:
    - SharedTokenStats (used by all ordering types)
    - optional per-token analysis data (if enabled)
    - optional positional KDE data (if enabled)

    If `ordering_types` is provided, their optional collection hooks are called during the
    batch loop using already-computed intermediates (top-k indices/values, attention masks).

    Side effects (plots, writing outputs, fitting topic models) are NOT performed here.

    Args:
        dataset_cfg: Dataset configuration
        attention_mask: [num_samples, seq_len] attention mask tensor
        logit_diff: [num_samples, seq_len, vocab_size] pre-computed logit difference tensor
        batch_size: Processing batch size
        max_tokens: Maximum token positions per sample
        max_samples: Maximum number of samples to use
        top_k: Number of top-K tokens to track
        ignore_padding: Whether to ignore padding positions
        per_token_analysis_cfg: Per-token analysis config (or None if not configured)
        positional_kde_cfg: Positional KDE config (or None if not configured)
        ordering_types: Optional list of ordering types for in-loop data collection
        tokenizer: HuggingFace tokenizer
        device: torch device for computation
        logger: Logger instance

    Returns:
        CoreAnalysisResult containing SharedTokenStats and optional analysis payloads.
    """
    logger.info(f"=" * 80)
    logger.info(f"Processing dataset: {dataset_cfg.id}")
    logger.info(f"Dataset name: {dataset_cfg.name}")

    # Validate and slice samples
    available_samples = logit_diff.shape[0]
    if max_samples > available_samples:
        logger.warning(
            f"Config requests {max_samples} samples but only {available_samples} available. "
            f"Using all {available_samples} available samples."
        )
        max_samples = available_samples
    elif max_samples < available_samples:
        logger.info(f"Using first {max_samples} samples (have {available_samples})")
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
        logger.info(f"Using first {max_tokens} positions (have {available_positions})")
        logit_diff = logit_diff[:, :max_tokens, :]
        attention_mask = attention_mask[:, :max_tokens]

    logger.info(
        f"Parameters: batch_size={batch_size}, max_tokens={max_tokens}, top_k={top_k}, max_samples={max_samples}"
    )
    logger.info(f"Dataset type: {'chat' if dataset_cfg.is_chat else 'text'}")

    total_positions = 0

    # Per-token analysis: Initialize tracking structures if enabled
    per_token_enabled = False
    co_occurrence_enabled = False
    shortlist_token_ids = {}
    per_sample_counts = {}
    per_position_counts = {}
    shortlist_diffs = defaultdict(list)

    shortlist_diffs_by_position = defaultdict(lambda: defaultdict(list))
    shortlist_diffs_by_sample = defaultdict(lambda: defaultdict(list))

    # Co-occurrence tracking (Top-K based)
    same_point_matrix = defaultdict(lambda: defaultdict(int))

    # Same-Sign Co-occurrence tracking
    same_sign_point_matrix = defaultdict(lambda: defaultdict(int))

    # Positional KDE Analysis
    pos_kde_enabled = False
    pos_kde_num_positions = 0
    position_logit_diffs = defaultdict(list)

    if positional_kde_cfg is not None and positional_kde_cfg.enabled:
        pos_kde_enabled = True
        pos_kde_num_positions = int(positional_kde_cfg.num_positions)
        logger.info(
            f"Positional KDE analysis enabled (plotting first {pos_kde_num_positions} positions)"
        )

    # Global Token Statistics (always enabled)
    global_stats_enabled = True
    global_diff_sum = None
    global_pos_count = None
    logger.info("Global Token Statistics enabled")

    if ordering_types:
        for ordering_type in ordering_types:
            ordering_type.begin_collection(dataset_cfg, ignore_padding=ignore_padding)

    if per_token_analysis_cfg is not None and per_token_analysis_cfg.enabled:
        per_token_enabled = True
        logger.info("Per-token analysis enabled")

        if (
            hasattr(per_token_analysis_cfg, "co_occurrence")
            and per_token_analysis_cfg.co_occurrence
        ):
            co_occurrence_enabled = True
            logger.info("Co-occurrence analysis enabled")

        for token_str in per_token_analysis_cfg.token_shortlist:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                shortlist_token_ids[token_ids[0]] = token_str
                per_sample_counts[token_str] = defaultdict(int)
                per_position_counts[token_str] = defaultdict(int)
            else:
                logger.warning(
                    f"Token '{token_str}' encodes to {len(token_ids)} tokens, skipping. "
                    f"Use single-token strings only."
                )

        logger.info(
            f"Tracking {len(shortlist_token_ids)} shortlist tokens: {list(shortlist_token_ids.values())}"
        )

    num_samples = logit_diff.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    logger.info(f"Processing {num_samples} samples in {num_batches} batches...")

    overall_max_len = logit_diff.shape[1]

    # === VECTORIZED ACCUMULATORS ===
    vocab_size = logit_diff.shape[-1]

    global_pos_token_counts = torch.zeros(vocab_size, dtype=torch.int64, device="cpu")
    global_neg_token_counts = torch.zeros(vocab_size, dtype=torch.int64, device="cpu")

    shortlist_ids_tensor = None
    shortlist_id_to_idx = {}
    shortlist_idx_to_str = {}
    if shortlist_token_ids:
        shortlist_ids_list = list(shortlist_token_ids.keys())
        shortlist_ids_tensor = torch.tensor(shortlist_ids_list, dtype=torch.long)
        for idx, tid in enumerate(shortlist_ids_list):
            shortlist_id_to_idx[tid] = idx
            shortlist_idx_to_str[idx] = shortlist_token_ids[tid]

    shortlist_per_sample_counts = torch.zeros(
        num_samples,
        len(shortlist_token_ids) if shortlist_token_ids else 0,
        dtype=torch.int64,
    )
    shortlist_per_position_counts = torch.zeros(
        overall_max_len,
        len(shortlist_token_ids) if shortlist_token_ids else 0,
        dtype=torch.int64,
    )

    num_shortlist = len(shortlist_token_ids) if shortlist_token_ids else 0
    vec_same_point_matrix = torch.zeros(num_shortlist, num_shortlist, dtype=torch.int64)
    vec_same_sign_point_matrix = torch.zeros(
        num_shortlist, num_shortlist, dtype=torch.int64
    )

    total_positions = 0

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {dataset_cfg.name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_attention_mask = attention_mask[start_idx:end_idx]
        diff = logit_diff[start_idx:end_idx].to(device)
        attention_mask_batch = batch_attention_mask.to(diff.device)

        # Global Token Statistics (Entire Vocabulary)
        if global_stats_enabled:
            if global_diff_sum is None:
                logger.info(
                    "  [Global Stats] Initializing accumulators and starting batch-wise accumulation..."
                )
                vocab_size = diff.shape[-1]
                global_diff_sum = torch.zeros(
                    vocab_size, dtype=torch.float32, device=diff.device
                )
                global_pos_count = torch.zeros(
                    vocab_size, dtype=torch.int32, device=diff.device
                )

            mask_expanded = attention_mask_batch.unsqueeze(-1).to(diff.dtype)
            diff.mul_(mask_expanded)
            del mask_expanded
            global_diff_sum += diff.sum(dim=(0, 1)).to(torch.float32)

            pos_mask = diff > 0
            global_pos_count += pos_mask.sum(dim=(0, 1), dtype=torch.int32)
            del pos_mask

        # Shortlist Distribution Tracking (Vectorized)
        if per_token_enabled and shortlist_token_ids:
            valid_mask = attention_mask_batch.bool()
            batch_size_curr, seq_len_curr = attention_mask_batch.shape

            for s_token_id, s_token_str in shortlist_token_ids.items():
                token_vals = diff[:, :, s_token_id]

                if ignore_padding:
                    valid_vals = token_vals[valid_mask]
                else:
                    valid_vals = token_vals.flatten()

                shortlist_diffs[s_token_str].extend(valid_vals.cpu().tolist())

                for pos_idx in range(seq_len_curr):
                    if ignore_padding:
                        pos_mask = attention_mask_batch[:, pos_idx].bool()
                        vals_at_pos = token_vals[:, pos_idx][pos_mask]
                    else:
                        vals_at_pos = token_vals[:, pos_idx]
                    shortlist_diffs_by_position[s_token_str][pos_idx].extend(
                        vals_at_pos.cpu().tolist()
                    )

                for b_idx in range(batch_size_curr):
                    sample_idx_global = start_idx + b_idx
                    if ignore_padding:
                        sample_mask = attention_mask_batch[b_idx, :].bool()
                        vals_for_sample = token_vals[b_idx, :][sample_mask]
                    else:
                        vals_for_sample = token_vals[b_idx, :]
                    shortlist_diffs_by_sample[s_token_str][sample_idx_global].extend(
                        vals_for_sample.cpu().tolist()
                    )

        # Get top-K positive diffs (largest values)
        top_k_pos_values, top_k_pos_indices = torch.topk(
            diff, k=top_k, dim=-1, largest=True
        )

        # Get top-K negative diffs (smallest values)
        top_k_neg_values, top_k_neg_indices = torch.topk(
            diff, k=top_k, dim=-1, largest=False
        )

        if ordering_types:
            batch_cache = OrderingBatchCache(
                top_k_pos_indices=top_k_pos_indices,
                top_k_pos_values=top_k_pos_values,
                top_k_neg_indices=top_k_neg_indices,
                top_k_neg_values=top_k_neg_values,
                attention_mask=attention_mask_batch,
                ignore_padding=ignore_padding,
            )
            for ordering_type in ordering_types:
                ordering_type.collect_batch(batch_cache)

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

            pos_per_sample, pos_per_pos = vectorized_shortlist_counts(
                top_k_pos_indices,
                attention_mask_batch,
                shortlist_tensor_device,
                start_idx,
            )
            neg_per_sample, neg_per_pos = vectorized_shortlist_counts(
                top_k_neg_indices,
                attention_mask_batch,
                shortlist_tensor_device,
                start_idx,
            )

            shortlist_per_sample_counts[start_idx:end_idx, :] += (
                pos_per_sample + neg_per_sample
            ).cpu()
            shortlist_per_position_counts[:seq_len, :] += (
                pos_per_pos + neg_per_pos
            ).cpu()

            # 3. Co-occurrence matrices (vectorized)
            if co_occurrence_enabled:
                batch_cooc = vectorized_cooccurrence_shortlist(
                    top_k_pos_indices, attention_mask_batch, shortlist_tensor_device
                )
                vec_same_point_matrix += batch_cooc.cpu()

                batch_same_sign = vectorized_same_sign_cooccurrence(
                    diff, attention_mask_batch, shortlist_tensor_device
                )
                vec_same_sign_point_matrix += batch_same_sign.cpu()

        # 4. Positional KDE Data Collection (vectorized per position)
        if pos_kde_enabled:
            for pos in range(min(pos_kde_num_positions, seq_len)):
                pos_mask_kde = (
                    attention_mask_batch[:, pos].bool()
                    if ignore_padding
                    else torch.ones(
                        batch_size_actual, dtype=torch.bool, device=diff.device
                    )
                )
                vals_at_pos = top_k_pos_values[:, pos, :]
                valid_vals = vals_at_pos[pos_mask_kde].flatten()
                position_logit_diffs[pos].extend(valid_vals.cpu().tolist())

        # Count total valid positions
        if ignore_padding:
            total_positions += attention_mask_batch.sum().item()
        else:
            total_positions += batch_size_actual * seq_len

        # Clean up GPU memory after each batch
        del (
            diff,
            top_k_pos_values,
            top_k_pos_indices,
            top_k_neg_values,
            top_k_neg_indices,
        )
        del attention_mask_batch
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"✓ Batch processing complete!")

    if per_token_enabled and shortlist_idx_to_str:
        for idx, token_str in shortlist_idx_to_str.items():
            for sample_idx in range(num_samples):
                count = shortlist_per_sample_counts[sample_idx, idx].item()
                if count > 0:
                    per_sample_counts[token_str][sample_idx] = count

            for pos_idx in range(overall_max_len):
                count = shortlist_per_position_counts[pos_idx, idx].item()
                if count > 0:
                    per_position_counts[token_str][pos_idx] = count

    if co_occurrence_enabled and shortlist_idx_to_str:
        for i, t1 in shortlist_idx_to_str.items():
            for j, t2 in shortlist_idx_to_str.items():
                count = vec_same_point_matrix[i, j].item()
                if count > 0:
                    same_point_matrix[t1][t2] = count

                count = vec_same_sign_point_matrix[i, j].item()
                if count > 0:
                    same_sign_point_matrix[t1][t2] = count

    num_tokens_with_any_topk = int(
        ((global_pos_token_counts + global_neg_token_counts) > 0).sum().item()
    )
    logger.info(
        f"Processed {total_positions:,} positions; tokens with any top-k counts: {num_tokens_with_any_topk:,}"
    )

    # Compute remaining co-occurrence matrices
    same_sample_matrix = defaultdict(lambda: defaultdict(int))
    same_position_matrix = defaultdict(lambda: defaultdict(int))
    same_sign_sample_matrix = defaultdict(lambda: defaultdict(int))
    same_sign_position_matrix = defaultdict(lambda: defaultdict(int))

    if co_occurrence_enabled and shortlist_idx_to_str:
        logger.info(
            "Computing Same-Sample/Same-Position co-occurrence matrices (vectorized)..."
        )

        presence_per_sample = (shortlist_per_sample_counts > 0).float()
        same_sample_cooc = presence_per_sample.T @ presence_per_sample

        presence_per_position = (shortlist_per_position_counts > 0).float()
        same_position_cooc = presence_per_position.T @ presence_per_position

        for i, t1 in shortlist_idx_to_str.items():
            for j, t2 in shortlist_idx_to_str.items():
                count_sample = int(same_sample_cooc[i, j].item())
                count_position = int(same_position_cooc[i, j].item())
                if count_sample > 0:
                    same_sample_matrix[t1][t2] = count_sample
                if count_position > 0:
                    same_position_matrix[t1][t2] = count_position

        same_sign_sample_matrix = dict(same_sample_matrix)
        same_sign_position_matrix = dict(same_position_matrix)

    if ordering_types:
        for ordering_type in ordering_types:
            ordering_type.end_collection()

    assert global_diff_sum is not None
    assert global_pos_count is not None
    assert global_diff_sum.shape == (vocab_size,)
    assert global_pos_count.shape == (vocab_size,)

    shared_stats = SharedTokenStats(
        vocab_size=vocab_size,
        total_positions=int(total_positions),
        num_samples=int(num_samples),
        sum_logit_diff=global_diff_sum.detach().cpu(),
        count_positive=global_pos_count.detach().cpu().to(torch.int64),
        topk_pos_counts=global_pos_token_counts.cpu(),
        topk_neg_counts=global_neg_token_counts.cpu(),
    )

    per_token_data = None
    if per_token_enabled:
        per_token_data = {
            "per_sample_counts": per_sample_counts,
            "per_position_counts": per_position_counts,
            "max_positions": overall_max_len,
            "shortlist_distributions": shortlist_diffs,
            "shortlist_diffs_by_position": shortlist_diffs_by_position,
            "shortlist_diffs_by_sample": shortlist_diffs_by_sample,
        }
        if co_occurrence_enabled:
            per_token_data["co_occurrence"] = {
                "same_sample": same_sample_matrix,
                "same_position": same_position_matrix,
                "same_point": same_point_matrix,
            }
            per_token_data["co_occurrence_same_sign"] = {
                "same_sign_same_sample": same_sign_sample_matrix,
                "same_sign_same_position": same_sign_position_matrix,
                "same_sign_same_point": same_sign_point_matrix,
            }

    kde_data = None
    if pos_kde_enabled:
        kde_data = {
            "position_logit_diffs": dict(position_logit_diffs),
            "pos_kde_num_positions": pos_kde_num_positions,
            "top_k": top_k,
        }

    return CoreAnalysisResult(
        shared_stats=shared_stats,
        per_token_data=per_token_data,
        kde_data=kde_data,
    )
