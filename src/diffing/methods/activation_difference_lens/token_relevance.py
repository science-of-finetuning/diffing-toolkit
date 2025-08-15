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
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

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


def _load_patchscope_tokens(
    results_dir: Path,
    dataset_id: str,
    layer_index: int,
    position_index: int,
    variant: str,
) -> Tuple[List[str], List[str]]:
    """Load tokens from auto_patch_scope artifacts.

    Returns (tokens_at_best_scale, selected_tokens).
    """
    dataset_dir_name = dataset_id.split("/")[-1]
    if variant == "difference":
        filename = f"auto_patch_scope_pos_{position_index}.pt"
    elif variant == "base":
        filename = f"base_auto_patch_scope_pos_{position_index}.pt"
    elif variant == "ft":
        filename = f"ft_auto_patch_scope_pos_{position_index}.pt"
    else:
        assert False, f"Unknown variant: {variant}"

    aps_path = results_dir / f"layer_{layer_index}" / dataset_dir_name / filename
    assert aps_path.exists(), f"Auto patch scope cache not found: {aps_path}"
    rec: Dict[str, Any] = torch.load(aps_path, map_location="cpu")
    assert "tokens_at_best_scale" in rec and "selected_tokens" in rec
    tokens_all = list(rec["tokens_at_best_scale"])  # type: ignore[arg-type]
    selected = list(rec["selected_tokens"])  # type: ignore[arg-type]
    assert all(isinstance(t, str) and len(t) > 0 for t in tokens_all)
    assert all(isinstance(t, str) and len(t) > 0 for t in selected)
    return tokens_all, selected


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

    # Preconditions from organism (self)
    organism_cfg = method.cfg.organism
    assert hasattr(organism_cfg, "description_long")
    self_description: str = str(organism_cfg.description_long)
    assert len(self_description.strip()) > 0

    assert hasattr(organism_cfg, "training_dataset")
    finetune_ds = organism_cfg.training_dataset
    assert "id" in finetune_ds
    self_finetune_dataset_id: str = str(finetune_ds["id"])  # type: ignore[index]
    self_splits: List[str] = list(finetune_ds["splits"])  # type: ignore[index]
    assert len(self_splits) >= 1
    assert "is_chat" in finetune_ds
    self_is_chat_dataset: bool = bool(finetune_ds["is_chat"])  # type: ignore[index]

    # Grader
    grader_cfg = cfg.grader
    grader = TokenRelevanceGrader(
        grader_model_id=str(grader_cfg.model_id),
        base_url=str(grader_cfg.base_url),
        api_key_path=str(grader_cfg.api_key_path),
    )

    # Frequent tokens from dataset (self or baseline overridden later)
    freq_cfg = cfg.frequent_tokens
    num_tokens = int(freq_cfg.num_tokens)
    min_count = int(freq_cfg.min_count)

    # Baseline organisms support
    baseline_organisms: List[str] = []
    if hasattr(cfg, "baseline_organisms") and cfg.baseline_organisms is not None:
        baseline_organisms = [str(x) for x in cfg.baseline_organisms]

    # Precompute descriptions and frequent tokens per evaluation target
    eval_targets: List[Tuple[str, str, List[str]]] = []
    # Each tuple: (label, description, frequent_tokens)
    if len(baseline_organisms) == 0:
        # Default: evaluate against this organism's own description and dataset
        frequent_tokens_self = _compute_frequent_tokens(
            dataset_name=self_finetune_dataset_id,
            tokenizer=method.tokenizer,
            splits=self_splits,
            num_tokens=num_tokens,
            min_count=min_count,
            is_chat=self_is_chat_dataset,
        )
        eval_targets.append(("self", self_description, frequent_tokens_self))
    else:
        hydra_cfg = HydraConfig.get()
        config_path_strs = [p["path"] for p in hydra_cfg.runtime.config_sources if p["schema"] == "file"]
        assert len(config_path_strs) >= 1
        configs_dir = Path(config_path_strs[0]).resolve()
        for org_name in baseline_organisms:
            org_path = configs_dir / "organism" / f"{org_name}.yaml"
            assert org_path.exists(), f"Baseline organism config not found: {org_path}"
            raw_cfg = OmegaConf.load(org_path)
            org_cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
            assert isinstance(org_cfg_dict, dict)
            assert "description_long" in org_cfg_dict, f"Missing description_long in {org_path}"
            assert "training_dataset" in org_cfg_dict, f"Missing training_dataset in {org_path}"
            ds_info = org_cfg_dict["training_dataset"]  # type: ignore[index]
            assert isinstance(ds_info, dict)
            assert "id" in ds_info and "splits" in ds_info and "is_chat" in ds_info
            baseline_desc: str = str(org_cfg_dict["description_long"])  # type: ignore[index]
            baseline_ds_id: str = str(ds_info["id"])  # type: ignore[index]
            baseline_splits: List[str] = [str(s) for s in ds_info["splits"]]  # type: ignore[index]
            baseline_is_chat: bool = bool(ds_info["is_chat"])  # type: ignore[index]
            baseline_freq = _compute_frequent_tokens(
                dataset_name=baseline_ds_id,
                tokenizer=method.tokenizer,
                splits=baseline_splits,
                num_tokens=num_tokens,
                min_count=min_count,
                is_chat=baseline_is_chat,
            )
            eval_targets.append((org_name, baseline_desc, baseline_freq))

    # Iterate tasks mirroring steering structure
    for task in cfg.tasks:
        rel_layer: float = float(task.layer)
        abs_layer: int = get_layer_indices(method.base_model_cfg.model_id, [rel_layer])[
            0
        ]
        dataset_id: str = str(task.dataset)
        positions: List[int] = [int(p) for p in task.positions]
        source: str = str(task.source)
        assert source in {"logitlens", "patchscope"}

        dataset_dir_name = dataset_id.split("/")[-1]

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
                # Evaluate for each target (self or baselines)
                for target_label, target_description, target_freq_tokens in eval_targets:
                    logger.info(
                        f"Grading token relevance [{source}] ({variant}) for layer {abs_layer} position {pos} target={target_label}"
                    )
                    # Output directory depends on target
                    if target_label == "self":
                        out_dir = (
                            method.results_dir
                            / f"layer_{abs_layer}"
                            / dataset_dir_name
                            / "token_relevance"
                            / f"position_{pos}"
                            / variant
                        )
                    else:
                        out_dir = (
                            method.results_dir
                            / f"layer_{abs_layer}"
                            / dataset_dir_name
                            / "token_relevance"
                            / "baselines"
                            / target_label
                            / f"position_{pos}"
                            / variant
                        )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rel_path = out_dir / f"relevance_{source}.json"

                    # Skip if results exist and overwrite is False
                    if (not overwrite) and rel_path.exists():
                        logger.info(
                            f"Existing token relevance found for layer {abs_layer} pos {pos} variant {variant} target {target_label}; skipping (overwrite=False)."
                        )
                        continue

                    # Load candidate tokens
                    if source == "logitlens":
                        candidate_tokens = _load_topk_promoted_tokens(
                            method.results_dir,
                            dataset_id,
                            abs_layer,
                            pos,
                            method.tokenizer,
                            int(cfg.k_candidate_tokens),
                            variant,
                        )
                        selected_tokens: List[str] = []
                    else:
                        candidate_tokens, selected_tokens = _load_patchscope_tokens(
                            method.results_dir,
                            dataset_id,
                            abs_layer,
                            pos,
                            variant,
                        )
                    assert isinstance(candidate_tokens, list) and len(candidate_tokens) >= 1

                    # Trivial baseline: fraction of candidates present in frequent token set (per target)
                    trivial_hits = sum(1 for t in candidate_tokens if t in target_freq_tokens)
                    trivial_percentage = trivial_hits / float(len(candidate_tokens))

                    # Grade with permutation robustness
                    permutations = int(grader_cfg.permutations)
                    majority_labels, _ = grader.grade(
                        description=target_description,
                        frequent_tokens=target_freq_tokens,
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
                        "source": source,
                        "target": target_label,
                        "labels": majority_labels,
                        "tokens": candidate_tokens,
                        "percentage": relevant_fraction,
                        "trivial_percentage": trivial_percentage,
                    }

                    # If patchscope filtering is available, compute filtered percentage without regrading
                    if source == "patchscope":
                        # Build boolean mask (per candidate token) indicating selection membership (multiset-aware)
                        from collections import Counter as _Counter
                        sel_counter = _Counter(selected_tokens)
                        mask: List[bool] = []
                        for tok in candidate_tokens:
                            if sel_counter[tok] > 0:
                                mask.append(True)
                                sel_counter[tok] -= 1
                            else:
                                mask.append(False)
                        assert len(mask) == len(candidate_tokens)
                        rec["unsupervised_filter"] = mask
                        # Optionally report filtered percentage if any token selected
                        if any(mask):
                            filtered_labels = [lbl for m, lbl in zip(mask, majority_labels) if m]
                            filt_relevant_fraction = sum(lbl == "RELEVANT" for lbl in filtered_labels) / float(len(filtered_labels))
                            rec["filtered_percentage"] = filt_relevant_fraction
                            # Save selected tokens for reference
                            rec["selected_tokens"] = selected_tokens
                    rel_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
