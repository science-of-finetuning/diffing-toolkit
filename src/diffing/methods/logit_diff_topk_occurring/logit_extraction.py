"""
Logit extraction interfaces for logit_diff_topk_occurring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from nnterp import StandardizedTransformer


class LogitsExtractor(ABC):
    """
    Interface for logit extraction from a model forward pass.
    """

    @abstractmethod
    def extract_logits(
        self,
        model: StandardizedTransformer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract logits for a single datapoint or batch.
        """


def _normalize_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize inputs to have shape [batch, seq].
    """
    assert input_ids.ndim in (1, 2), f"input_ids must be 1D or 2D, got {input_ids.shape}"
    assert attention_mask.ndim in (1, 2), (
        f"attention_mask must be 1D or 2D, got {attention_mask.shape}"
    )
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    assert input_ids.shape == attention_mask.shape, (
        f"input_ids and attention_mask must match, got {input_ids.shape} vs {attention_mask.shape}"
    )
    return input_ids, attention_mask


class DirectLogitsExtractor(LogitsExtractor):
    """
    Directly extract output logits via NNsight tracing.
    """

    def extract_logits(
        self,
        model: StandardizedTransformer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids, attention_mask = _normalize_inputs(input_ids, attention_mask)
        with model.trace(input_ids, attention_mask=attention_mask):
            logits = model.logits.save()
        assert logits.ndim == 3, f"logits must be 3D, got {logits.shape}"
        assert logits.shape[0] == input_ids.shape[0], (
            f"logits batch mismatch {logits.shape[0]} vs {input_ids.shape[0]}"
        )
        assert logits.shape[1] == input_ids.shape[1], (
            f"logits seq mismatch {logits.shape[1]} vs {input_ids.shape[1]}"
        )
        return logits


class LogitLensExtractor(LogitsExtractor):
    """
    Extract logits by projecting an intermediate layer output onto vocab.

    This uses `model.layers_output[layer_idx]` and `model.project_on_vocab(hidden)`.
    """

    def __init__(self, *, layer_idx: int):
        self.layer_idx = int(layer_idx)
        assert self.layer_idx >= 0, f"layer_idx must be >= 0, got {self.layer_idx}"

    def extract_logits(
        self,
        model: StandardizedTransformer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids, attention_mask = _normalize_inputs(input_ids, attention_mask)

        with model.trace(input_ids, attention_mask=attention_mask) as tracer:
            hidden = model.layers_output[self.layer_idx]
            logits = model.project_on_vocab(hidden).save()
            tracer.stop()

        assert logits.ndim == 3, f"logits must be 3D, got {logits.shape}"
        assert logits.shape[0] == input_ids.shape[0], (
            f"logits batch mismatch {logits.shape[0]} vs {input_ids.shape[0]}"
        )
        assert logits.shape[1] == input_ids.shape[1], (
            f"logits seq mismatch {logits.shape[1]} vs {input_ids.shape[1]}"
        )
        return logits
