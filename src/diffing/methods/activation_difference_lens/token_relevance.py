from __future__ import annotations

from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from loguru import logger
from datasets import load_dataset
import torch
import json

from transformers import PreTrainedTokenizerBase

from src.utils.graders.token_relevance_grader import TokenRelevanceGrader
from src.utils.activations import get_layer_indices

COMMON_WORDS = {
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "among",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "this",
    "that",
    "these",
    "those",
    "a",
    "an",
    "ing",
}


def _load_topk_promoted_tokens(
    results_dir: Path,
    dataset_id: str,
    layer_index: int,
    position_index: int,
    tokenizer: PreTrainedTokenizerBase,
    k: int,
    variant: str,
) -> List[str]:
    """Load top-k promoted token IDs from saved logit lens and decode to strings.

    Returns list of decoded tokens in descending logit-lens probability order.
    """
    dataset_dir_name = dataset_id.split("/")[-1]
    if variant == "difference":
        filename = f"logit_lens_pos_{position_index}.pt"
    elif variant == "base":
        filename = f"base_logit_lens_pos_{position_index}.pt"
    elif variant == "ft":
        filename = f"ft_logit_lens_pos_{position_index}.pt"
    else:
        assert False, f"Unknown variant: {variant}"

    ll_path = results_dir / f"layer_{layer_index}" / dataset_dir_name / filename
    assert ll_path.exists(), f"Logit lens cache not found: {ll_path}"
    top_k_probs, top_k_indices, _, _ = torch.load(ll_path, map_location="cpu")
    top_k_indices = top_k_indices[:k]
    top_k_probs = top_k_probs[:k]
    assert isinstance(top_k_indices, torch.Tensor) and top_k_indices.ndim == 1
    assert isinstance(top_k_probs, torch.Tensor) and top_k_probs.ndim == 1
    assert top_k_indices.shape == top_k_probs.shape

    decoded: List[str] = []
    for tok_id in top_k_indices.tolist():
        decoded.append(tokenizer.decode([int(tok_id)]))
    assert len(decoded) == int(top_k_indices.numel())
    return decoded


def _is_generic_token(token: str) -> bool:
    clean_token = token.replace("▁", "").replace("Ġ", "").strip()
    if len(clean_token) <= 1:
        return True
    # Pure punctuation
    import re as _re

    if _re.match(r"^[^\w\s]+$", clean_token):
        return True

    # Filter trivial tokens like "'s", newlines, and whitespace patterns
    if clean_token in {"'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "ing"}:
        return True

    # Filter newline and whitespace patterns (common in tokenizers)
    if _re.match(r"^[\s\n\r\t]+$", clean_token):
        return True

    return clean_token.lower() in COMMON_WORDS


def _compute_frequent_tokens(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    splits: List[str],
    num_tokens: int,
    min_count: int,
    is_chat: bool,
) -> List[str]:
    """Return list of frequent non-generic tokens from finetuning dataset."""
    assert isinstance(dataset_name, str) and len(dataset_name) > 0
    assert isinstance(splits, list) and len(splits) >= 1
    assert isinstance(num_tokens, int) and num_tokens >= 1
    assert isinstance(min_count, int) and min_count >= 1
    assert isinstance(is_chat, bool)

    ds = load_dataset(dataset_name)
    all_tokens: List[str] = []

    for split in splits:
        assert split in ds, f"Split '{split}' not found in dataset '{dataset_name}'"
        for sample in tqdm(ds[split], desc=f"Processing {split} split"):
            if is_chat:
                messages = sample.get("messages", None)
                assert isinstance(messages, list) and len(messages) >= 1
                contents: List[str] = []
                for msg in messages:
                    assert isinstance(msg, dict) and ("content" in msg)
                    content = msg["content"]
                    assert isinstance(content, str)
                    contents.append(content)
                text = "\n".join(contents)
            else:
                text = sample.get("text", None)
                assert isinstance(text, str)
            all_tokens.extend(tokenizer.tokenize(text))

    counts = Counter(all_tokens)
    domain_tokens: List[Tuple[str, int]] = [
        (tok, cnt)
        for tok, cnt in counts.items()
        if (not _is_generic_token(tok)) and cnt >= min_count
    ]
    domain_tokens.sort(key=lambda x: x[1], reverse=True)
    frequent = [tok for tok, _ in domain_tokens[:num_tokens]]
    assert len(frequent) > 0
    return frequent


def run_token_relevance(method: Any) -> None:
    """Run token relevance grading for saved logit-lens candidates based on cfg.

    Requires: cfg.organism.description_long and cfg.organism.training_dataset.id
    """
    cfg = method.cfg.diffing.method.token_relevance
    assert cfg.enabled is True
    overwrite: bool = bool(method.cfg.diffing.method.overwrite)

    # Preconditions from organism
    organism_cfg = method.cfg.organism
    assert hasattr(organism_cfg, "description_long")
    description: str = str(organism_cfg.description_long)
    assert len(description.strip()) > 0

    assert hasattr(organism_cfg, "training_dataset")
    finetune_ds = organism_cfg.training_dataset
    assert "id" in finetune_ds
    finetune_dataset_id: str = str(finetune_ds["id"])  # type: ignore[index]
    splits: List[str] = list(finetune_ds["splits"])  # type: ignore[index]
    assert len(splits) >= 1
    assert "is_chat" in finetune_ds
    is_chat_dataset: bool = bool(finetune_ds["is_chat"])  # type: ignore[index]

    # Grader
    grader_cfg = cfg.grader
    grader = TokenRelevanceGrader(
        grader_model_id=str(grader_cfg.model_id),
        base_url=str(grader_cfg.base_url),
        api_key_path=str(grader_cfg.api_key_path),
    )

    # Frequent tokens from finetuning dataset
    freq_cfg = cfg.frequent_tokens
    num_tokens = int(freq_cfg.num_tokens)
    min_count = int(freq_cfg.min_count)
    frequent_tokens = _compute_frequent_tokens(
        dataset_name=finetune_dataset_id,
        tokenizer=method.tokenizer,
        splits=splits,
        num_tokens=num_tokens,
        min_count=min_count,
        is_chat=is_chat_dataset,
    )

    # Iterate tasks mirroring steering structure
    for task in cfg.tasks:
        rel_layer: float = float(task.layer)
        abs_layer: int = get_layer_indices(method.base_model_cfg.model_id, [rel_layer])[
            0
        ]
        dataset_id: str = str(task.dataset)
        positions: List[int] = [int(p) for p in task.positions]

        dataset_dir_name = dataset_id.split("/")[-1]
        base_out_dir = (
            method.results_dir
            / f"layer_{abs_layer}"
            / dataset_dir_name
            / "token_relevance"
        )

        for pos in positions:
            # Determine which variants to grade
            to_grade: List[str] = []
            if bool(cfg.grade_difference):
                to_grade.append("difference")
            if bool(cfg.grade_base):
                to_grade.append("base")
            if bool(cfg.grade_ft):
                to_grade.append("ft")
            assert len(to_grade) >= 1

            for variant in to_grade:
                logger.info(
                    f"Grading token relevance ({variant}) for layer {abs_layer} position {pos}"
                )
                out_dir = base_out_dir / f"position_{pos}" / variant
                out_dir.mkdir(parents=True, exist_ok=True)
                rel_path = out_dir / "relevance.json"

                # Skip if results exist and overwrite is False
                if (not overwrite) and rel_path.exists():
                    logger.info(
                        f"Existing token relevance found for layer {abs_layer} pos {pos} variant {variant}; skipping (overwrite=False)."
                    )
                    continue

                # Load candidate tokens (promoted only)
                candidate_tokens = _load_topk_promoted_tokens(
                    method.results_dir,
                    dataset_id,
                    abs_layer,
                    pos,
                    method.tokenizer,
                    int(cfg.k_candidate_tokens),
                    variant,
                )
                assert isinstance(candidate_tokens, list) and len(candidate_tokens) >= 1

                # Trivial baseline: fraction of candidates present in frequent token set
                trivial_hits = sum(1 for t in candidate_tokens if t in frequent_tokens)
                trivial_percentage = trivial_hits / float(len(candidate_tokens))

                # Grade with permutation robustness
                permutations = int(grader_cfg.permutations)
                majority_labels, _ = grader.grade(
                    description=description,
                    frequent_tokens=frequent_tokens,
                    candidate_tokens=candidate_tokens,
                    permutations=permutations,
                    concurrent=True,
                    max_tokens=int(grader_cfg.max_tokens),
                )
                assert len(majority_labels) == len(candidate_tokens)
                relevant_fraction = sum(
                    lbl == "RELEVANT" for lbl in majority_labels
                ) / float(len(majority_labels))

                rec: Dict[str, Any] = {
                    "layer": abs_layer,
                    "position": pos,
                    "variant": variant,
                    "labels": majority_labels,
                    "tokens": candidate_tokens,
                    "percentage": relevant_fraction,
                    "trivial_percentage": trivial_percentage,
                }
                rel_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
