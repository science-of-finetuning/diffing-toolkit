"""
Opt-in streaming activations for crosscoder training.

Computes paired (base, ft) activations on the fly into an in-memory shuffle buffer
instead of training from the disk activation cache (multi-TB at large token budgets).
Training only: analysis still reads the preprocessing caches. Enabled via
diffing.method.streaming.enabled; when false (default) the disk path is unchanged.
"""

import math
import random
from typing import Iterator, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import DictConfig
from dictionary_learning.cache import ActivationCache

from ..activations import get_layer_indices
from ..configs import get_dataset_configurations, get_model_configurations
from ..data import load_dataset_from_hub_or_local
from ..model import load_model_from_config

# Tracing flags matching dictionary_learning.cache and the preprocessing pipeline
tracer_kwargs = {"scan": False, "validate": False}


def _needs_special_tokens(text: str, tokenizer) -> bool:
    """Add BOS only if `text` does not already embed it (chat templates do); mirrors the disk path."""
    bos = tokenizer.bos_token
    return bos is None or bos not in text


class PairedActivationBuffer:
    """
    Buffer of paired (base, ft) activations at one layer, refilled on the fly.

    Both models share the tokenizer: each batch is tokenized once and the same token ids
    are fed to both models. Yields [out_batch_size, 2, d_model] tensors, drop-in for
    PairedActivationCache in the training DataLoader.
    """

    def __init__(
        self,
        data: Iterator[str],
        base_model,
        ft_model,
        base_submodule,
        ft_submodule,
        tokenizer,
        d_model: int,
        context_len: int = 1024,
        refresh_batch_size: int = 64,
        out_batch_size: int = 2048,
        n_ctxs: int = 1000,
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

        # Right-padding is required with ignore_first_n_tokens, else it would mask padding
        if ignore_first_n_tokens > 0:
            self.tokenizer.padding_side = "right"

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

    def _tokenize_batch(self, texts: List[str]):
        """Tokenize each text with its own special-tokens decision (the stream mixes
        chat and plain rows), then pad together."""
        encoded = [
            self.tokenizer(
                text,
                max_length=self.context_len,
                truncation=True,
                add_special_tokens=_needs_special_tokens(text, self.tokenizer),
            )
            for text in texts
        ]
        return self.tokenizer.pad(encoded, padding=True, return_tensors="pt")

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

            tokens = self._tokenize_batch(texts)

            mask = tokens["attention_mask"].clone()
            if self.ignore_first_n_tokens > 0:
                mask[:, : self.ignore_first_n_tokens] = 0
            if self.mask_token_id is not None:
                mask[tokens["input_ids"] == self.mask_token_id] = 0
            flat_mask = mask.reshape(-1).bool()

            base_acts = self._trace_out(self.base_model, self.base_submodule, tokens)
            ft_acts = self._trace_out(self.ft_model, self.ft_submodule, tokens)

            # Base and ft may sit on different GPUs; gather onto buffer_device
            base_sel = base_acts[flat_mask.to(base_acts.device)].to(self.buffer_device)
            ft_sel = ft_acts[flat_mask.to(ft_acts.device)].to(self.buffer_device)
            paired = torch.stack([base_sel, ft_sel], dim=1)
            if len(self.activations) == 0:
                # Match the models' dtype: float32 would double buffer memory
                self.activations = self.activations.to(paired.dtype)
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
        self, n_samples: int = 100_000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of shape [2, d_model] from a random sample of the buffer."""
        self.refresh()
        n = min(n_samples, len(self.activations))
        idx = torch.randperm(len(self.activations), device=self.activations.device)[:n]
        sample = self.activations[idx].float()
        mean = sample.mean(dim=0)
        # Biased std to match the disk path's combine_normalizer
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


def _estimate_tokens_per_row(
    dataset, ds_cfg, tokenizer, context_len: int, sample_size: int = 128
) -> float:
    """Mean tokens per row on an evenly-spaced sample, tokenized exactly as the buffer
    does (truncation included) so draw_weights reflect the actually buffered tokens."""
    n = min(sample_size, len(dataset))
    step = max(1, len(dataset) // n)
    counts = []
    for i in range(0, len(dataset), step):
        text = _row_to_text(dataset[i], ds_cfg, tokenizer)
        counts.append(
            len(
                tokenizer(
                    text,
                    max_length=context_len,
                    truncation=True,
                    add_special_tokens=_needs_special_tokens(text, tokenizer),
                )["input_ids"]
            )
        )
        if len(counts) >= n:
            break
    return sum(counts) / max(1, len(counts))


def _make_text_stream(
    datasets: List, dataset_cfgs: List, tokenizer, draw_weights: List[float], seed: int
) -> Iterator[str]:
    """Infinite weighted interleave: pick a dataset by `draw_weights`, then a random row
    (with replacement; the stream is bounded externally by the trainer's `steps`)."""
    rng = random.Random(seed)

    def sample(ds, ds_cfg):
        n = len(ds)
        while True:
            yield _row_to_text(ds[rng.randrange(n)], ds_cfg, tokenizer)

    gens = [sample(ds, c) for ds, c in zip(datasets, dataset_cfgs)]
    total = float(sum(draw_weights))
    probs = [w / total for w in draw_weights]
    while True:
        yield next(rng.choices(gens, weights=probs, k=1)[0])


def setup_streaming_training(cfg: DictConfig, layer: int, device: str) -> Tuple[
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
        steps); val_data is a finite list of [B, 2, d_model] tensors.
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

    # ModelConfig is a dataclass, so set device_map directly. ignore_cache=True because
    # the model cache key omits device_map and would return a wrongly-placed model.
    # ft_device null = auto: a second GPU distinct from base_device if present, else share
    ft_device = streaming_cfg.ft_device
    if ft_device is None:
        if torch.cuda.device_count() > 1:
            ft_device = (
                "cuda:1" if str(streaming_cfg.base_device) != "cuda:1" else "cuda:0"
            )
        else:
            ft_device = streaming_cfg.base_device
        logger.info(f"streaming.ft_device=null resolved to {ft_device}")

    base_model_cfg.device_map = streaming_cfg.base_device
    ft_model_cfg.device_map = ft_device
    base_model = load_model_from_config(base_model_cfg, ignore_cache=True)
    ft_model = load_model_from_config(ft_model_cfg, ignore_cache=True)
    tokenizer = base_model.tokenizer

    if isinstance(layer, float):
        layer = get_layer_indices(base_model_cfg.model_id, [layer])[0]
    base_submodule = base_model.layers[layer]
    ft_submodule = ft_model.layers[layer]
    d_model = base_model.hidden_size

    train_cfgs = [c for c in dataset_cfgs if c.split == "train"]
    val_cfgs = [c for c in dataset_cfgs if c.split == "validation"] or train_cfgs

    def load_all(cfgs):
        return [
            load_dataset_from_hub_or_local(c.id, split=c.split, name=c.subset)
            for c in cfgs
        ]

    train_datasets = load_all(train_cfgs)
    val_datasets = load_all(val_cfgs)

    context_len = (
        cfg.preprocessing.context_len if hasattr(cfg, "preprocessing") else 1024
    )

    # Draws are per-row but dataset_weights are token fractions; divide by mean
    # tokens/row to get per-row probabilities
    mean_tokens = [
        _estimate_tokens_per_row(ds, c, tokenizer, context_len)
        for ds, c in zip(train_datasets, train_cfgs)
    ]
    if streaming_cfg.get("dataset_weights", None) is not None:
        target_fracs = [float(w) for w in streaming_cfg.dataset_weights]
    else:
        target_fracs = [len(ds) * mt for ds, mt in zip(train_datasets, mean_tokens)]
    draw_weights = [t / mt for t, mt in zip(target_fracs, mean_tokens)]
    logger.info(
        f"Streaming {[c.name for c in train_cfgs]}: "
        f"mean_tokens/row={[round(m, 1) for m in mean_tokens]}, "
        f"draw_weights={[round(w, 4) for w in draw_weights]}"
    )
    val_weights = (
        draw_weights if len(val_cfgs) == len(train_cfgs) else [1.0] * len(val_cfgs)
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
        data=_make_text_stream(
            train_datasets, train_cfgs, tokenizer, draw_weights, seed=cfg.seed
        ),
        **buffer_kwargs,
    )

    # Finite list so trainSAE's len(validation_data) works. batch_size * 4 matches the
    # disk path's validation loader (BatchTopK's threshold is batch-size dependent).
    val_buffer_kwargs = {**buffer_kwargs, "out_batch_size": training_cfg.batch_size * 4}
    val_buffer = PairedActivationBuffer(
        data=_make_text_stream(
            val_datasets, val_cfgs, tokenizer, val_weights, seed=cfg.seed + 1
        ),
        **val_buffer_kwargs,
    )
    num_val_batches = streaming_cfg.get("num_val_batches", 50)
    val_data = [next(val_buffer) for _ in range(num_val_batches)]
    logger.info(f"Built {len(val_data)} streaming validation batches")

    if method_cfg.datasets.normalization.enabled:
        normalize_mean, normalize_std = train_buffer.compute_normalizer()
        # Return the normalizer on the training device (the buffer may live elsewhere)
        normalize_mean = normalize_mean.to(device)
        normalize_std = normalize_std.to(device)
    else:
        normalize_mean, normalize_std = None, None

    activation_dim = d_model
    steps = training_cfg.max_steps
    if steps is None:
        steps = math.ceil(
            training_cfg.num_samples * training_cfg.epochs / training_cfg.batch_size
        )
    logger.info(
        f"Streaming training for {steps} steps "
        f"(~{training_cfg.num_samples} tokens x {training_cfg.epochs} epochs)"
    )

    return train_buffer, val_data, normalize_mean, normalize_std, activation_dim, steps
