"""
Preprocessing utilities for logit diff top-K occurring analysis.

Contains dataset tokenization, tensor preparation, and position slicing functions.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffing.utils.configs import DatasetConfig
from ..activation_difference_lens.method import (
    load_and_tokenize_dataset,
    load_and_tokenize_chat_dataset,
)


def slice_to_positions(tensor: torch.Tensor, positions_list: List[List[int]]) -> torch.Tensor:
    """
    Extract specific positions from each sample in a tensor.

    Used to slice logit diffs to only the relevant positions around assistant start
    for chat datasets, matching pre_assistant_k behavior.

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


@torch.no_grad()
def prepare_dataset_tensors(
    dataset_cfg: DatasetConfig,
    tokenizer,
    max_samples: int,
    max_tokens: int,
    pre_assistant_k: int,
    debug_print_samples: Optional[int],
    seed: Optional[int],
    logger,
) -> Dict[str, Any]:
    """
    Prepare input tensors for a single dataset.

    Tokenizes the dataset and pads/stacks into batch tensors.

    Args:
        dataset_cfg: Dataset configuration
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum number of samples to load
        max_tokens: Maximum tokens per sample
        pre_assistant_k: Number of tokens before assistant turn (for chat datasets)
        debug_print_samples: Number of samples to print for debugging (None to skip)
        seed: Random seed for reproducible sampling
        logger: Logger instance

    Returns:
        Dict containing 'input_ids', 'attention_mask' tensors and 'positions' (None or list)
    """
    logger.info(f"Preparing input tensors for {dataset_cfg.name}...")

    all_positions = None

    if dataset_cfg.is_chat:
        logger.info(f"Using ADL's load_and_tokenize_chat_dataset() with pre_assistant_k={pre_assistant_k}")
        samples = load_and_tokenize_chat_dataset(
            dataset_name=dataset_cfg.id,
            tokenizer=tokenizer,
            split=dataset_cfg.split,
            messages_column=dataset_cfg.messages_column or "messages",
            n=max_tokens,
            pre_assistant_k=pre_assistant_k,
            max_samples=max_samples,
            debug_print_samples=debug_print_samples,
            seed=seed,
        )
        all_token_ids = [sample["input_ids"] for sample in samples]
        all_positions = [sample["positions"] for sample in samples]
    else:
        logger.info("Using ADL's load_and_tokenize_dataset()")
        all_token_ids = load_and_tokenize_dataset(
            dataset_name=dataset_cfg.id,
            tokenizer=tokenizer,
            split=dataset_cfg.split,
            text_column=dataset_cfg.text_column or "text",
            n=max_tokens,
            max_samples=max_samples,
            subset=dataset_cfg.subset,
            streaming=dataset_cfg.streaming,
            debug_print_samples=debug_print_samples,
            seed=seed,
        )

    if not all_token_ids:
        logger.warning(f"No samples found for {dataset_cfg.name}!")
        return {"input_ids": torch.empty(0), "attention_mask": torch.empty(0)}

    actual_count = len(all_token_ids)
    if actual_count < max_samples:
        logger.warning(
            f"Requested {max_samples} samples but dataset only has {actual_count}. "
            f"Will use all {actual_count} available samples."
        )

    max_len = max(len(ids) for ids in all_token_ids)
    input_ids_list = []
    attention_mask_list = []

    for token_ids in all_token_ids:
        padding_length = max_len - len(token_ids)
        padded_ids = token_ids + [tokenizer.pad_token_id] * padding_length
        mask = [1] * len(token_ids) + [0] * padding_length

        input_ids_list.append(padded_ids)
        attention_mask_list.append(mask)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

    logger.info(f"Prepared tensors: input_ids {input_ids.shape}")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": all_positions,
    }


def prepare_all_dataset_inputs_and_save(
    datasets: List[DatasetConfig],
    *,
    tokenizer,
    method_cfg,
    seed: Optional[int],
    masks_dir: Path,
    input_ids_dir: Path,
    logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Tokenize/pad all datasets and save attention masks + input_ids.

    Returns a mapping keyed by `dataset_cfg.name`.
    """
    max_samples = int(method_cfg.max_samples)
    max_tokens = int(method_cfg.max_tokens_per_sample)
    pre_assistant_k = int(method_cfg.pre_assistant_k)
    debug_print_samples = getattr(method_cfg, "debug_print_samples", None)

    dataset_inputs: Dict[str, Dict[str, Any]] = {}
    for dataset_cfg in datasets:
        dataset_inputs[dataset_cfg.name] = prepare_dataset_tensors(
            dataset_cfg=dataset_cfg,
            tokenizer=tokenizer,
            max_samples=max_samples,
            max_tokens=max_tokens,
            pre_assistant_k=pre_assistant_k,
            debug_print_samples=debug_print_samples,
            seed=seed,
            logger=logger,
        )

        mask_path = masks_dir / f"{dataset_cfg.name}_attention_mask.pt"
        torch.save(dataset_inputs[dataset_cfg.name]["attention_mask"], mask_path)
        logger.info(f"Saved attention mask to {mask_path}")

        input_ids_path = input_ids_dir / f"{dataset_cfg.name}_input_ids.pt"
        torch.save(dataset_inputs[dataset_cfg.name]["input_ids"], input_ids_path)
        logger.info(f"Saved input_ids to {input_ids_path}")

    return dataset_inputs


@torch.no_grad()
def infer_logits_for_dataset(
    *,
    model,
    logits_extractor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    device,
    desc: str,
) -> torch.Tensor:
    """
    Run batched inference and return concatenated logits on CPU.
    """
    if input_ids.numel() == 0:
        return torch.empty(0)

    num_samples = input_ids.shape[0]
    dataset_logits: List[torch.Tensor] = []

    for i in tqdm(range(0, num_samples, batch_size), desc=desc):
        batch_input = input_ids[i : i + batch_size].to(device)
        batch_mask = attention_mask[i : i + batch_size].to(device)

        logits = logits_extractor.extract_logits(model, batch_input, batch_mask)
        dataset_logits.append(logits.cpu())

        del batch_input, batch_mask, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not dataset_logits:
        return torch.empty(0)
    return torch.cat(dataset_logits, dim=0)


def infer_and_store_logits_for_all_datasets(
    datasets: List[DatasetConfig],
    dataset_inputs: Dict[str, Dict[str, Any]],
    *,
    model,
    logits_extractor,
    batch_size: int,
    device,
    logger,
    in_memory: bool,
    logits_dir: Path,
    logits_suffix: str,
) -> Dict[str, torch.Tensor]:
    """
    Run inference for all datasets and either store in memory or save to disk.

    Returns a dict of logits (only populated in `in_memory=True` mode).
    """
    logits_by_dataset: Dict[str, torch.Tensor] = {}

    for dataset_cfg in datasets:
        logger.info(f"Computing {logits_suffix} logits for {dataset_cfg.name}...")
        inputs = dataset_inputs[dataset_cfg.name]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        logits = infer_logits_for_dataset(
            model=model,
            logits_extractor=logits_extractor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_size=batch_size,
            device=device,
            desc=f"{logits_suffix.capitalize()} Model Inference",
        )

        if logits.numel() == 0:
            continue

        if in_memory:
            logits_by_dataset[dataset_cfg.name] = logits
            logger.info(
                f"Stored {logits_suffix} logits in memory: {logits.shape} "
                f"({logits.numel() * logits.element_size() / 1e9:.1f} GB)"
            )
        else:
            logits_path = logits_dir / f"{dataset_cfg.name}_{logits_suffix}_logits.pt"
            torch.save(logits, logits_path)
            logger.info(f"Saved {logits_suffix} logits to {logits_path}")
            del logits

        gc.collect()

    return logits_by_dataset


def _maybe_slice_vocab(
    base_logits: torch.Tensor,
    ft_logits: torch.Tensor,
    *,
    max_vocab_size: Optional[int],
    logger,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_vocab_size is None:
        return base_logits, ft_logits

    if base_logits.shape[-1] > max_vocab_size:
        logger.info(f"Slicing base logits vocab from {base_logits.shape[-1]} to {max_vocab_size}")
        base_logits = base_logits[..., :max_vocab_size]
    if ft_logits.shape[-1] > max_vocab_size:
        logger.info(f"Slicing finetuned logits vocab from {ft_logits.shape[-1]} to {max_vocab_size}")
        ft_logits = ft_logits[..., :max_vocab_size]

    return base_logits, ft_logits


def maybe_compute_token_log_probs(
    base_logits: torch.Tensor,
    ft_logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    max_vocab_size: Optional[int],
    slr_enabled: bool,
    logger,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute per-token log-probabilities for sequence likelihood ratio analysis.

    Returns (base_token_log_probs, ft_token_log_probs) or (None, None) if disabled.
    """
    if not slr_enabled:
        logger.info("Skipping log probabilities (sequence_likelihood_ratio.enabled=false)")
        return None, None

    logger.info("Computing per-token log probabilities...")
    base_log_softmax = F.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
    ft_log_softmax = F.log_softmax(ft_logits[:, :-1, :].float(), dim=-1)
    target_ids = input_ids[:, 1:]
    if max_vocab_size is not None:
        target_ids = target_ids.clamp(max=max_vocab_size - 1)

    base_token_log_probs = base_log_softmax.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    ft_token_log_probs = ft_log_softmax.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    del base_log_softmax, ft_log_softmax
    gc.collect()

    return base_token_log_probs, ft_token_log_probs


def slice_chat_outputs_if_needed(
    *,
    positions_list: Optional[List[List[int]]],
    diff: torch.Tensor,
    input_ids: torch.Tensor,
    base_token_log_probs: Optional[torch.Tensor],
    ft_token_log_probs: Optional[torch.Tensor],
    slr_enabled: bool,
    logger,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if positions_list is None:
        return diff, input_ids, base_token_log_probs, ft_token_log_probs

    original_shape = diff.shape
    diff = slice_to_positions(diff, positions_list)
    logger.info(f"Sliced chat logit diff from {original_shape} to {diff.shape}")

    if slr_enabled:
        assert base_token_log_probs is not None
        assert ft_token_log_probs is not None
        log_prob_positions = [[p - 1 for p in pos_list if p > 0] for pos_list in positions_list]
        base_token_log_probs = slice_to_positions_2d(base_token_log_probs, log_prob_positions)
        ft_token_log_probs = slice_to_positions_2d(ft_token_log_probs, log_prob_positions)

    sliced_target_positions = [p[1:] if len(p) > 1 else p for p in positions_list]
    input_ids = slice_to_positions_2d(input_ids, sliced_target_positions)

    return diff, input_ids, base_token_log_probs, ft_token_log_probs


def infer_finetuned_and_compute_diffs_in_memory(
    datasets: List[DatasetConfig],
    dataset_inputs: Dict[str, Dict[str, Any]],
    *,
    finetuned_model,
    logits_extractor,
    base_logits_by_dataset: Dict[str, torch.Tensor],
    batch_size: int,
    device,
    method_cfg,
    logger,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """
    In-memory path: run finetuned inference, compute diffs immediately, and return tensors.

    Returns:
      (logit_diffs, log_probs, attention_masks, input_ids_sliced)
    """
    max_vocab_size = getattr(method_cfg, "max_vocab_size", None)
    slr_enabled = bool(getattr(method_cfg.sequence_likelihood_ratio, "enabled", False))

    logit_diffs: Dict[str, torch.Tensor] = {}
    log_probs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    attention_masks: Dict[str, torch.Tensor] = {}
    input_ids_out: Dict[str, torch.Tensor] = {}

    for dataset_cfg in datasets:
        logger.info(f"Computing finetuned logits for {dataset_cfg.name}...")
        inputs = dataset_inputs[dataset_cfg.name]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        ft_logits = infer_logits_for_dataset(
            model=finetuned_model,
            logits_extractor=logits_extractor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_size=batch_size,
            device=device,
            desc="Finetuned Model Inference",
        )
        if ft_logits.numel() == 0:
            continue

        if dataset_cfg.name not in base_logits_by_dataset:
            raise KeyError(f"Missing base logits for {dataset_cfg.name} in in-memory preprocessing.")
        base_logits = base_logits_by_dataset.pop(dataset_cfg.name)

        base_logits, ft_logits = _maybe_slice_vocab(
            base_logits,
            ft_logits,
            max_vocab_size=max_vocab_size,
            logger=logger,
        )

        base_token_log_probs, ft_token_log_probs = maybe_compute_token_log_probs(
            base_logits,
            ft_logits,
            input_ids,
            max_vocab_size=max_vocab_size,
            slr_enabled=slr_enabled,
            logger=logger,
        )

        ft_logits -= base_logits
        del base_logits
        gc.collect()

        positions_list = inputs.get("positions")
        ft_logits, input_ids_sliced, base_token_log_probs, ft_token_log_probs = slice_chat_outputs_if_needed(
            positions_list=positions_list,
            diff=ft_logits,
            input_ids=input_ids,
            base_token_log_probs=base_token_log_probs,
            ft_token_log_probs=ft_token_log_probs,
            slr_enabled=slr_enabled,
            logger=logger,
        )

        logit_diffs[dataset_cfg.name] = ft_logits
        attention_masks[dataset_cfg.name] = attention_mask
        input_ids_out[dataset_cfg.name] = input_ids_sliced
        if slr_enabled:
            assert base_token_log_probs is not None
            assert ft_token_log_probs is not None
            log_probs[dataset_cfg.name] = (base_token_log_probs, ft_token_log_probs)

        logger.info(
            f"Stored logit diff in memory: {ft_logits.shape} "
            f"({ft_logits.numel() * ft_logits.element_size() / 1e9:.1f} GB)"
        )

    return logit_diffs, log_probs, attention_masks, input_ids_out


def compute_and_save_disk_diffs(
    datasets: List[DatasetConfig],
    dataset_inputs: Dict[str, Dict[str, Any]],
    *,
    logits_dir: Path,
    diffs_dir: Path,
    log_probs_dir: Path,
    input_ids_dir: Path,
    method_cfg,
    delete_raw: bool,
    logger,
) -> None:
    """
    Disk path: load base/finetuned logits, compute diffs, optionally save log-probs, and clean up.
    """
    max_vocab_size = getattr(method_cfg, "max_vocab_size", None)
    slr_enabled = bool(getattr(method_cfg.sequence_likelihood_ratio, "enabled", False))

    for dataset_cfg in datasets:
        logger.info(f"Computing diff for {dataset_cfg.name}...")

        base_path = logits_dir / f"{dataset_cfg.name}_base_logits.pt"
        ft_path = logits_dir / f"{dataset_cfg.name}_finetuned_logits.pt"

        if not base_path.exists() or not ft_path.exists():
            logger.warning(f"Missing base or finetuned logits for {dataset_cfg.name}. Skipping diff.")
            continue

        base = torch.load(base_path, map_location="cpu")
        ft = torch.load(ft_path, map_location="cpu")

        input_ids = dataset_inputs[dataset_cfg.name]["input_ids"]
        base, ft = _maybe_slice_vocab(base, ft, max_vocab_size=max_vocab_size, logger=logger)

        base_token_log_probs, ft_token_log_probs = maybe_compute_token_log_probs(
            base,
            ft,
            input_ids,
            max_vocab_size=max_vocab_size,
            slr_enabled=slr_enabled,
            logger=logger,
        )

        diff = ft - base

        positions_list = dataset_inputs[dataset_cfg.name].get("positions")
        if positions_list is not None:
            original_shape = diff.shape
            diff = slice_to_positions(diff, positions_list)
            logger.info(
                f"Sliced chat logit diff from {original_shape} to {diff.shape} (pre_assistant_k + n positions)"
            )

            if slr_enabled:
                assert base_token_log_probs is not None
                assert ft_token_log_probs is not None
                log_prob_positions = [[p - 1 for p in pos_list if p > 0] for pos_list in positions_list]
                base_token_log_probs = slice_to_positions_2d(base_token_log_probs, log_prob_positions)
                ft_token_log_probs = slice_to_positions_2d(ft_token_log_probs, log_prob_positions)

            sliced_target_positions = [p[1:] if len(p) > 1 else p for p in positions_list]
            sliced_input_ids = slice_to_positions_2d(input_ids, sliced_target_positions)
            input_ids_path = input_ids_dir / f"{dataset_cfg.name}_input_ids.pt"
            torch.save(sliced_input_ids, input_ids_path)
            logger.info(f"Updated input_ids with sliced chat positions: {sliced_input_ids.shape}")

        diff_path = diffs_dir / f"{dataset_cfg.name}_logit_diff.pt"
        torch.save(diff, diff_path)
        logger.info(f"Saved logit diff to {diff_path}")

        if slr_enabled:
            assert base_token_log_probs is not None
            assert ft_token_log_probs is not None
            base_log_probs_path = log_probs_dir / f"{dataset_cfg.name}_base_log_probs.pt"
            ft_log_probs_path = log_probs_dir / f"{dataset_cfg.name}_ft_log_probs.pt"
            torch.save(base_token_log_probs, base_log_probs_path)
            torch.save(ft_token_log_probs, ft_log_probs_path)
            logger.info(
                f"Saved log probabilities: base={base_log_probs_path.name}, ft={ft_log_probs_path.name}"
            )

        del base, ft, diff
        if base_token_log_probs is not None:
            del base_token_log_probs
        if ft_token_log_probs is not None:
            del ft_token_log_probs
        gc.collect()

        if delete_raw:
            if base_path.exists():
                base_path.unlink()
                logger.info(f"Deleted raw base logits: {base_path}")
            if ft_path.exists():
                ft_path.unlink()
                logger.info(f"Deleted raw finetuned logits: {ft_path}")
