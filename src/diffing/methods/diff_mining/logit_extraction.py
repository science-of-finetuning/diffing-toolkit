"""
Logit extraction interfaces for diff mining.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from loguru import logger
from transformers import DynamicCache
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
    assert input_ids.ndim in (
        1,
        2,
    ), f"input_ids must be 1D or 2D, got {input_ids.shape}"
    assert attention_mask.ndim in (
        1,
        2,
    ), f"attention_mask must be 1D or 2D, got {attention_mask.shape}"
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    assert (
        input_ids.shape == attention_mask.shape
    ), f"input_ids and attention_mask must match, got {input_ids.shape} vs {attention_mask.shape}"
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
        assert (
            logits.shape[0] == input_ids.shape[0]
        ), f"logits batch mismatch {logits.shape[0]} vs {input_ids.shape[0]}"
        assert (
            logits.shape[1] == input_ids.shape[1]
        ), f"logits seq mismatch {logits.shape[1]} vs {input_ids.shape[1]}"
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
        assert (
            logits.shape[0] == input_ids.shape[0]
        ), f"logits batch mismatch {logits.shape[0]} vs {input_ids.shape[0]}"
        assert (
            logits.shape[1] == input_ids.shape[1]
        ), f"logits seq mismatch {logits.shape[1]} vs {input_ids.shape[1]}"
        return logits


@dataclass
class _PrefixKVCache:
    """Cached KV states for the patch prompt prefix (all tokens except the patched one)."""

    kv_cache: DynamicCache
    last_token_id: torch.Tensor  # [1, 1]
    prefix_len: int


def _compute_prefix_kv_cache(
    model: StandardizedTransformer,
    patch_prompt: str,
) -> _PrefixKVCache:
    """Compute and cache the KV states for the patch prompt prefix.

    Runs all tokens except the last through the raw model with use_cache=True.
    The last token is the one that gets patched during patchscope, so its KV
    states depend on the injected latent and cannot be cached.
    """
    assert getattr(model, "tokenizer", None) is not None, "model.tokenizer must exist"
    tokens = model.tokenizer(patch_prompt, return_tensors="pt")
    full_ids = tokens.input_ids  # [1, seq_len]
    assert full_ids.ndim == 2 and full_ids.shape[0] == 1

    prefix_ids = full_ids[:, :-1]  # [1, seq_len - 1]
    last_token_id = full_ids[:, -1:]  # [1, 1]
    prefix_len = prefix_ids.shape[1]
    assert prefix_len > 0, f"patch_prompt must have > 1 token, got {full_ids.shape[1]}"

    model_device = next(model.parameters()).device
    with torch.no_grad():
        prefix_output = model._module(prefix_ids.to(model_device), use_cache=True)

    kv_cache = prefix_output.past_key_values
    assert isinstance(
        kv_cache, DynamicCache
    ), f"Expected DynamicCache, got {type(kv_cache)}"
    assert (
        len(kv_cache) == model.num_layers
    ), f"KV cache layers {len(kv_cache)} != model layers {model.num_layers}"
    key0, _ = kv_cache[0]
    assert key0.shape[0] == 1
    assert (
        key0.shape[2] == prefix_len
    ), f"KV cache seq_len {key0.shape[2]} != prefix_len {prefix_len}"

    return _PrefixKVCache(
        kv_cache=kv_cache, last_token_id=last_token_id, prefix_len=prefix_len
    )


def _expand_kv_cache(kv_cache: DynamicCache, batch_size: int) -> DynamicCache:
    """Expand a batch-1 KV cache to batch_size via expand (no-copy view).

    Safe because the decode step uses use_cache=False, so the model never
    mutates these tensors in-place.
    """
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        key, value = kv_cache[layer_idx]
        assert key.shape[0] == 1, f"Expected batch=1 KV cache, got batch={key.shape[0]}"
        new_cache.update(
            key.expand(batch_size, -1, -1, -1),
            value.expand(batch_size, -1, -1, -1),
            layer_idx,
        )
    return new_cache


class PatchscopeLensExtractor(LogitsExtractor):
    """Extract logits by using patchscope lens with prefix KV caching.

    Caches the forward pass of all patch prompt tokens except the last one,
    then runs single-token decode for each batch of latents. This avoids
    redundant computation since the patch prompt is always the same.
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
        assert (
            self.index_to_patch == -1
        ), f"Prefix KV caching only supports index_to_patch=-1, got {self.index_to_patch}"

    @staticmethod
    @torch.no_grad()
    def _patchscope_lens_logits_cached(
        model: StandardizedTransformer,
        *,
        layer: int,
        latents: torch.Tensor,
        prefix_cache: _PrefixKVCache,
    ) -> torch.Tensor:
        """Patch latents into the model using prefix KV cache for single-token decode.

        Shapes:
          latents: [num_sources, hidden_size]
          returns: [num_sources, vocab_size]
        """
        assert latents.ndim == 2, f"latents must be 2D, got {latents.shape}"
        num_sources, hidden_size = latents.shape
        assert (
            hidden_size == model.hidden_size
        ), f"hidden_size mismatch: {hidden_size} vs {model.hidden_size}"

        model_device = next(model.parameters()).device
        expanded_kv = _expand_kv_cache(prefix_cache.kv_cache, num_sources)

        input_dict = {
            "input_ids": prefix_cache.last_token_id.expand(num_sources, -1).to(
                model_device
            ),
            "attention_mask": torch.ones(
                num_sources,
                prefix_cache.prefix_len + 1,
                device=model_device,
                dtype=torch.long,
            ),
        }
        position_ids = torch.full(
            (num_sources, 1),
            prefix_cache.prefix_len,
            device=model_device,
            dtype=torch.long,
        )

        with model.trace(
            input_dict,
            past_key_values=expanded_kv,
            position_ids=position_ids,
            use_cache=False,
        ) as tracer:
            layer_out = model.layers_output[layer]
            device = layer_out.device
            layer_out[:, 0] = latents.to(device=device, dtype=layer_out.dtype)
            logits = model.logits[:, -1, :].to("cpu", non_blocking=True).save()
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

        with model.trace(
            input_ids, attention_mask=attention_mask, use_cache=False
        ) as tracer:
            hidden_cpu = (
                model.layers_output[self.layer_idx].to("cpu", non_blocking=True).save()
            )
            tracer.stop()

        hidden = hidden_cpu
        assert hidden.ndim == 3, f"hidden must be 3D, got {hidden.shape}"
        assert (
            hidden.shape[0] == input_ids.shape[0]
        ), f"hidden batch mismatch {hidden.shape[0]} vs {input_ids.shape[0]}"
        assert (
            hidden.shape[1] == input_ids.shape[1]
        ), f"hidden seq mismatch {hidden.shape[1]} vs {input_ids.shape[1]}"
        assert (
            hidden.shape[2] == model.hidden_size
        ), f"hidden size mismatch {hidden.shape[2]} vs {model.hidden_size}"

        batch_size, seq_len, hidden_size = hidden.shape
        assert hidden.shape == (batch_size, seq_len, hidden_size)

        flat_hidden = hidden.reshape(batch_size * seq_len, hidden_size)
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

        prefix_cache = _compute_prefix_kv_cache(model, self.patch_prompt)

        num_sources = latents.shape[0]
        for start in range(0, num_sources, self.position_batch_size):
            end = min(start + self.position_batch_size, num_sources)
            latents_chunk = latents[start:end]
            assert latents_chunk.shape == (end - start, hidden_size)

            logits_chunk = self._patchscope_lens_logits_cached(
                model,
                layer=self.layer_idx,
                latents=latents_chunk,
                prefix_cache=prefix_cache,
            )
            assert logits_chunk.shape == (end - start, model.vocab_size)
            out_flat[start:end] = logits_chunk
            del logits_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out = out_flat.reshape(batch_size, seq_len, model.vocab_size)
        assert out.ndim == 3
        assert out.shape[0] == batch_size
        assert out.shape[1] == seq_len
        return out
