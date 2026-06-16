"""
Streaming activations for crosscoder training.

The default crosscoder path caches paired activations to disk (preprocessing) and
trains from that cache. For large token budgets this cache can reach the multi-TB
range. This module provides an opt-in alternative that computes paired activations on
the fly into an in-memory shuffle buffer, so the big *training* pass needs no disk cache.

Scope: this replaces only the training data source. Analysis (latent scaling,
max-activating examples, ...) still reads the (small) activation caches produced by
preprocessing, so a small preprocessing run remains the way to feed analysis.

Enable via the method config:

    diffing:
      method:
        streaming:
          enabled: true
          base_device: cuda:0   # device for the base model forward pass
          ft_device: cuda:1     # device for the finetuned model forward pass
          buffer_device: cuda:0 # where pooled activations live / are yielded
          n_ctxs: 30000         # approx contexts held in the buffer
          dataset_weights: null # optional list of mixing weights (defaults to size-proportional)

When `enabled` is false (the default) nothing here runs and the disk path is unchanged.
"""

import math
import random
from typing import Iterator, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import DictConfig, open_dict
from dictionary_learning.cache import ActivationCache

from ..activations import calculate_samples_per_dataset, get_layer_indices
from ..configs import get_dataset_configurations, get_model_configurations
from ..data import load_dataset_from_hub_or_local
from ..model import load_model_from_config

# nnsight tracing flags, matching dictionary_learning.cache and the preprocessing pipeline.
tracer_kwargs = {"scan": False, "validate": False}


class PairedActivationBuffer:
    """
    Buffer of paired (base, ft) activations at one layer, refilled on the fly.

    Both models are assumed to share the tokenizer, so each text batch is tokenized once
    and the identical token ids are fed to both models; the per-token (base, ft) pairing
    is then exact by construction. Yields tensors of shape [out_batch_size, 2, d_model],
    matching PairedActivationCache (so it is a drop-in for the training DataLoader).

    trainSAE consumes this by plain iteration bounded by `steps`; it needs no __len__.
    """

    def __init__(
        self,
        data: Iterator[str],  # yields pre-templated training strings
        base_model,  # nnsight StandardizedTransformer (base/normal)
        ft_model,  # nnsight StandardizedTransformer (finetuned)
        base_submodule,  # layer module of base_model to read, e.g. model.layers[L]
        ft_submodule,  # layer module of ft_model to read
        tokenizer,  # shared tokenizer
        d_model: int,
        context_len: int = 1024,
        refresh_batch_size: int = 64,
        out_batch_size: int = 2048,
        n_ctxs: int = 30000,
        buffer_device: str = "cpu",
        ignore_first_n_tokens: int = 0,
        add_special_tokens: bool = True,
        mask_token_id: Optional[int] = None,
    ):
        self.data = iter(data)
        self.base_model = base_model
        self.ft_model = ft_model
        self.base_submodule = base_submodule
        self.ft_submodule = ft_submodule
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.context_len = context_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.n_ctxs = n_ctxs
        self.buffer_device = buffer_device
        self.ignore_first_n_tokens = ignore_first_n_tokens
        self.add_special_tokens = add_special_tokens
        self.mask_token_id = mask_token_id

        self.activations = torch.empty(0, 2, d_model, device=buffer_device)
        self.read = torch.zeros(0, dtype=torch.bool, device=buffer_device)
        self._exhausted = False

    def __iter__(self):
        return self

    @property
    def _capacity(self) -> int:
        return self.n_ctxs * self.context_len

    def _text_batch(self) -> List[str]:
        texts = []
        for _ in range(self.refresh_batch_size):
            try:
                texts.append(next(self.data))
            except StopIteration:
                self._exhausted = True
                break
        return texts

    def _trace_out(self, model, submodule, tokens) -> torch.Tensor:
        with torch.no_grad(), model.trace(tokens, **tracer_kwargs):
            acts = (
                ActivationCache.get_activations(submodule, "out")
                .reshape(-1, self.d_model)
                .save()
            )
        return acts

    def refresh(self) -> None:
        """Drop consumed rows and recompute activations until the buffer is full."""
        self.activations = self.activations[~self.read]
        while len(self.activations) < self._capacity and not self._exhausted:
            texts = self._text_batch()
            if not texts:
                break

            tokens = self.tokenizer(
                texts,
                max_length=self.context_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens,
            )

            mask = tokens["attention_mask"].clone()
            if self.ignore_first_n_tokens > 0:
                mask[:, : self.ignore_first_n_tokens] = 0
            if self.mask_token_id is not None:
                mask[tokens["input_ids"] == self.mask_token_id] = 0
            flat_mask = mask.reshape(-1).bool()

            base_acts = self._trace_out(self.base_model, self.base_submodule, tokens)
            ft_acts = self._trace_out(self.ft_model, self.ft_submodule, tokens)

            paired = torch.stack(
                [base_acts[flat_mask], ft_acts[flat_mask]], dim=1
            ).to(self.buffer_device)
            self.activations = torch.cat([self.activations, paired], dim=0)

        self.read = torch.zeros(
            len(self.activations), dtype=torch.bool, device=self.buffer_device
        )

    def __next__(self) -> torch.Tensor:
        with torch.no_grad():
            if (~self.read).sum() < self._capacity // 2:
                self.refresh()
            unread = (~self.read).nonzero().squeeze(-1)
            if len(unread) == 0:
                raise StopIteration("Activation stream exhausted")
            perm = torch.randperm(len(unread), device=unread.device)
            idxs = unread[perm[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def compute_normalizer(
        self, n_batches: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of shape [2, d_model] from an initial fill of the buffer."""
        self.refresh()
        n = min(n_batches * self.out_batch_size, len(self.activations))
        sample = self.activations[:n].float()
        mean = sample.mean(dim=0)
        std = sample.std(dim=0, unbiased=False)
        logger.info(f"Computed streaming normalizer from {n} activations")
        return mean, std

    @property
    def config(self) -> dict:
        return {
            "buffer_type": "PairedActivationBuffer",
            "d_model": self.d_model,
            "context_len": self.context_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "n_ctxs": self.n_ctxs,
            "ignore_first_n_tokens": self.ignore_first_n_tokens,
            "mask_token_id": self.mask_token_id,
            "buffer_device": str(self.buffer_device),
        }


def _row_to_text(row, ds_cfg, tokenizer) -> str:
    """Extract a single training string from a dataset row."""
    if ds_cfg.is_chat:
        messages = row[ds_cfg.messages_column]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return row[ds_cfg.text_column or "text"]


def _weighted_text_stream(
    dataset_cfgs: List, tokenizer, split: str, weights: List[float], seed: int
) -> Iterator[str]:
    """
    Infinite weighted-interleave generator over the datasets' text.

    Each dataset is cycled (reshuffled per pass); at every step a dataset is chosen with
    probability proportional to `weights`. The stream never ends on its own - it is
    bounded externally by the trainer's `steps`.
    """
    datasets = [
        load_dataset_from_hub_or_local(c.id, split=split, name=c.subset)
        for c in dataset_cfgs
    ]
    rng = random.Random(seed)

    def cycle(ds, ds_cfg):
        idx = list(range(len(ds)))
        while True:
            rng.shuffle(idx)
            for i in idx:
                yield _row_to_text(ds[i], ds_cfg, tokenizer)

    gens = [cycle(ds, c) for ds, c in zip(datasets, dataset_cfgs)]
    total = float(sum(weights))
    probs = [w / total for w in weights]
    while True:
        gen = rng.choices(gens, weights=probs, k=1)[0]
        yield next(gen)


def setup_streaming_training(
    cfg: DictConfig, layer: int, device: str
) -> Tuple[
    PairedActivationBuffer,
    List[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    int,
    int,
]:
    """
    Build the streaming training data, validation data, normalizer, and step count.

    Args:
        cfg: Full configuration.
        layer: Layer index to read activations from (absolute, or a [0, 1] fraction).
        device: Device the crosscoder trains on (unused here; models use the streaming
            config's base_device / ft_device).

    Returns:
        Tuple of (train_buffer, val_data, normalize_mean, normalize_std, activation_dim,
        steps). train_buffer is a PairedActivationBuffer (pass as trainSAE `data`);
        val_data is a finite list of [B, 2, d_model] tensors (so len() works, as
        trainSAE's validation loop expects).
    """
    method_cfg = cfg.diffing.method
    streaming_cfg = method_cfg.streaming
    training_cfg = method_cfg.training

    base_model_cfg, ft_model_cfg = get_model_configurations(cfg)
    dataset_cfgs = get_dataset_configurations(
        cfg,
        use_chat_dataset=method_cfg.datasets.use_chat_dataset,
        use_pretraining_dataset=method_cfg.datasets.use_pretraining_dataset,
        use_training_dataset=method_cfg.datasets.use_training_dataset,
    )

    # Place the two models on their configured devices (model-parallel by default).
    with open_dict(base_model_cfg):
        base_model_cfg.device_map = streaming_cfg.base_device
    with open_dict(ft_model_cfg):
        ft_model_cfg.device_map = streaming_cfg.ft_device
    base_model = load_model_from_config(base_model_cfg)
    ft_model = load_model_from_config(ft_model_cfg)
    tokenizer = base_model.tokenizer

    if isinstance(layer, float):
        layer = get_layer_indices(base_model_cfg.model_id, [layer])[0]
    base_submodule = base_model.layers[layer]
    ft_submodule = ft_model.layers[layer]
    d_model = base_model.config.hidden_size

    # Mixing weights: explicit if given, else proportional to dataset sizes (matching the
    # disk path's size-proportional token allocation via calculate_samples_per_dataset).
    if streaming_cfg.get("dataset_weights", None) is not None:
        weights = list(streaming_cfg.dataset_weights)
    else:
        lengths = [
            len(load_dataset_from_hub_or_local(c.id, split="train", name=c.subset))
            for c in dataset_cfgs
        ]
        weights = calculate_samples_per_dataset(lengths, sum(lengths))
    logger.info(
        f"Streaming datasets {[c.name for c in dataset_cfgs]} with weights {weights}"
    )

    context_len = (
        cfg.preprocessing.context_len if hasattr(cfg, "preprocessing") else 1024
    )
    buffer_kwargs = dict(
        base_model=base_model,
        ft_model=ft_model,
        base_submodule=base_submodule,
        ft_submodule=ft_submodule,
        tokenizer=tokenizer,
        d_model=d_model,
        context_len=context_len,
        out_batch_size=training_cfg.batch_size,
        n_ctxs=streaming_cfg.n_ctxs,
        buffer_device=streaming_cfg.buffer_device,
        ignore_first_n_tokens=cfg.model.ignore_first_n_tokens_per_sample_during_training,
        mask_token_id=streaming_cfg.get("mask_token_id", None),
    )

    train_buffer = PairedActivationBuffer(
        data=_weighted_text_stream(
            dataset_cfgs, tokenizer, "train", weights, seed=cfg.seed
        ),
        **buffer_kwargs,
    )

    # Finite validation set: pull a fixed number of batches into a list (so len() works).
    val_buffer = PairedActivationBuffer(
        data=_weighted_text_stream(
            dataset_cfgs, tokenizer, "validation", weights, seed=cfg.seed + 1
        ),
        **buffer_kwargs,
    )
    num_val_batches = max(1, training_cfg.num_validation_samples // training_cfg.batch_size)
    val_data = [next(val_buffer) for _ in range(num_val_batches)]
    logger.info(f"Built {len(val_data)} streaming validation batches")

    if method_cfg.datasets.normalization.enabled:
        normalize_mean, normalize_std = train_buffer.compute_normalizer()
    else:
        normalize_mean, normalize_std = None, None

    activation_dim = d_model
    steps = math.ceil(
        training_cfg.num_samples * training_cfg.epochs / training_cfg.batch_size
    )
    logger.info(
        f"Streaming training for {steps} steps "
        f"(~{training_cfg.num_samples} tokens x {training_cfg.epochs} epochs)"
    )

    return train_buffer, val_data, normalize_mean, normalize_std, activation_dim, steps
