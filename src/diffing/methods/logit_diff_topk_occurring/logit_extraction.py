"""
Logit extraction interfaces for logit_diff_topk_occurring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List

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



class PatchscopeLensExtractor(LogitsExtractor):
    """
    Extract logits by using patchscope lens.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        position_batch_size: int = 256,
        patch_prompt: str = "man -> man\n1135 -> 1135\nhello -> hello\n?",
        index_to_patch: int = -1,
    ):
        self.layer_idx = int(layer_idx)
        assert self.layer_idx >= 0, f"layer_idx must be >= 0, got {self.layer_idx}"
        self.position_batch_size = int(position_batch_size)
        assert (
            self.position_batch_size > 0
        ), f"position_batch_size must be > 0, got {self.position_batch_size}"
        self.patch_prompt = str(patch_prompt)
        assert self.patch_prompt, "patch_prompt must be non-empty"
        self.index_to_patch = int(index_to_patch)

    @staticmethod
    @torch.no_grad()
    def _patchscope_lens_logits(
        model: StandardizedTransformer,
        *,
        layer: int,
        latents: torch.Tensor,
        target_patch_prompts: List[str],
        index_to_patch: int,
    ) -> torch.Tensor:
        """
        Patch `latents` into `model.layers_output[layer]` at `index_to_patch` for each prompt,
        and return next-token logits for each prompt.

        Shapes:
          latents: [num_sources, hidden_size]
          returns: [num_sources, vocab_size]
        """
        assert latents.ndim == 2, f"latents must be 2D, got {latents.shape}"
        num_sources, hidden_size = latents.shape
        assert num_sources == len(
            target_patch_prompts
        ), f"num_sources mismatch: {num_sources} vs {len(target_patch_prompts)}"
        assert (
            hidden_size == model.hidden_size
        ), f"hidden_size mismatch: {hidden_size} vs {model.hidden_size}"
        assert getattr(model, "tokenizer", None) is not None, "model.tokenizer must exist"

        with model.trace(target_patch_prompts) as tracer:
            layer_out = model.layers_output[layer]
            device = layer_out.device
            layer_out[torch.arange(num_sources), index_to_patch] = latents.to(
                device=device, dtype=model.dtype
            )
            logits = model.logits[:, -1, :].save()
            tracer.stop()

        assert logits.shape == (
            num_sources,
            model.vocab_size,
        ), f"logits shape {logits.shape} != ({num_sources}, {model.vocab_size})"
        return logits

    def extract_logits(
        self,
        model: StandardizedTransformer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids, attention_mask = _normalize_inputs(input_ids, attention_mask)

        with model.trace(input_ids, attention_mask=attention_mask) as tracer:
            hidden = model.layers_output[self.layer_idx].save()
            tracer.stop()

        assert hidden.ndim == 3, f"hidden must be 3D, got {hidden.shape}"
        assert hidden.shape[0] == input_ids.shape[0], (
            f"hidden batch mismatch {hidden.shape[0]} vs {input_ids.shape[0]}"
        )
        assert hidden.shape[1] == input_ids.shape[1], (
            f"hidden seq mismatch {hidden.shape[1]} vs {input_ids.shape[1]}"
        )
        assert hidden.shape[2] == model.hidden_size, (
            f"hidden size mismatch {hidden.shape[2]} vs {model.hidden_size}"
        )

        batch_size, seq_len, hidden_size = hidden.shape
        hidden_cpu = hidden.cpu()
        assert hidden_cpu.shape == (batch_size, seq_len, hidden_size)

        flat_hidden = hidden_cpu.reshape(batch_size * seq_len, hidden_size)
        assert flat_hidden.shape == (batch_size * seq_len, hidden_size)
        latents = flat_hidden
        assert latents.shape == (batch_size * seq_len, hidden_size)

        out_flat = torch.zeros(
            batch_size * seq_len,
            model.vocab_size,
            dtype=model.dtype,
            device="cpu",
        )
        if latents.numel() == 0:
            return out_flat.reshape(batch_size, seq_len, model.vocab_size)

        num_sources = latents.shape[0]
        for start in range(0, num_sources, self.position_batch_size):
            end = min(start + self.position_batch_size, num_sources)
            latents_chunk = latents[start:end]
            assert latents_chunk.shape == (end - start, hidden_size)

            prompts_chunk = [self.patch_prompt] * (end - start)
            logits_chunk = self._patchscope_lens_logits(
                model,
                layer=self.layer_idx,
                latents=latents_chunk,
                target_patch_prompts=prompts_chunk,
                index_to_patch=self.index_to_patch,
            ).cpu()
            assert logits_chunk.shape == (end - start, model.vocab_size)
            out_flat[start:end] = logits_chunk

        out = out_flat.reshape(batch_size, seq_len, model.vocab_size)
        assert out.ndim == 3
        assert out.shape[0] == batch_size
        assert out.shape[1] == seq_len
        return out

