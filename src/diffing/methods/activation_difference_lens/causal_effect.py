from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import json
import math
import gc
import random

import torch
from loguru import logger
from nnterp import StandardizedTransformer
from tqdm import tqdm

from src.utils.activations import get_layer_indices
from src.utils.data import load_dataset_from_hub_or_local
from src.utils.model import place_inputs

from .util import dataset_dir_name, load_position_mean_vector


class StreamingCEStats:
    """Streaming stats for per-token CE (negative log-likelihood).

    Tracks min, max exactly and median approximately via reservoir sampling.
    """

    def __init__(self, reservoir_size: int = 8192) -> None:
        assert isinstance(reservoir_size, int) and reservoir_size >= 1
        self._reservoir_size: int = reservoir_size
        self._reservoir: List[float] = []
        self._count: int = 0
        self._min_val: float = float("inf")
        self._max_val: float = float("-inf")

    def update(self, value: float) -> None:
        assert isinstance(value, (int, float))
        v = float(value)
        self._count += 1
        if v < self._min_val:
            self._min_val = v
        if v > self._max_val:
            self._max_val = v
        if len(self._reservoir) < self._reservoir_size:
            self._reservoir.append(v)
        else:
            idx = random.randint(0, self._count - 1)
            if idx < self._reservoir_size:
                self._reservoir[idx] = v

    def update_many(self, values: List[float]) -> None:
        for v in values:
            self.update(float(v))

    def count(self) -> int:
        return self._count

    def min_ce(self) -> float:
        return float("nan") if self._count == 0 else float(self._min_val)

    def max_ce(self) -> float:
        return float("nan") if self._count == 0 else float(self._max_val)

    def median_ce(self) -> float:
        if self._count == 0 or len(self._reservoir) == 0:
            return float("nan")
        arr = sorted(self._reservoir)
        n = len(arr)
        mid = n // 2
        if n % 2 == 1:
            return float(arr[mid])
        return float(0.5 * (arr[mid - 1] + arr[mid]))


def _update_stats_from_masked(
    nll: torch.Tensor,
    mask: torch.Tensor,
    stats: StreamingCEStats,
) -> None:
    """Update stats with values selected by mask. No allocations beyond one vector.

    nll and mask must have identical shape [B, L-1].
    """
    assert nll.shape == mask.shape
    vals = nll.masked_select(mask.to(nll.device))
    if vals.numel() == 0:
        return
    vals_cpu = vals.detach().to(torch.float32).cpu().tolist()
    stats.update_many(vals_cpu)


def _encode_non_chat(
    sample: Dict[str, Any],
    tokenizer: Any,
    text_column: str,
    max_total_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (input_ids[int64], assistant_mask[int64], within_idx[int64]) as tensors.

    Non-chat: assistant mask is all ones.
    """
    assert text_column in sample
    text = sample[text_column]
    assert isinstance(text, str) and len(text) > 0
    ids_list: List[int] = tokenizer.encode(text, add_special_tokens=True)
    assert isinstance(ids_list, list) and len(ids_list) > 0
    if len(ids_list) > max_total_tokens:
        ids_list = ids_list[:max_total_tokens]
    ids = torch.tensor(ids_list, dtype=torch.long)  # [L]
    L = ids.shape[0]
    assistant_mask = torch.ones((L,), dtype=torch.long)
    return ids, assistant_mask


def _encode_chat(
    messages: List[Dict[str, Any]],
    tokenizer: Any,
    max_total_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (input_ids[int64], assistant_mask[int64]) for multi-turn chat.

    Assistant mask is 1 on tokens that belong to assistant messages.
    """
    assert isinstance(messages, list) and len(messages) >= 1
    # Progressive encoding to get per-message token spans
    prev_len = 0
    spans: List[Tuple[int, int]] = []
    roles: List[str] = []
    for i in range(len(messages)):
        msg = messages[i]
        assert isinstance(msg, dict) and ("role" in msg) and ("content" in msg)
        roles.append(str(msg["role"]))
        ids_prefix: List[int] = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )
        new_len = len(ids_prefix)
        start = prev_len
        end = new_len
        spans.append((start, end))
        prev_len = new_len

    ids_full_list: List[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if len(ids_full_list) > max_total_tokens:
        ids_full_list = ids_full_list[:max_total_tokens]
    ids = torch.tensor(ids_full_list, dtype=torch.long)  # [L]
    L = ids.shape[0]
    assert L > 0

    assistant_mask = torch.zeros((L,), dtype=torch.long)

    # Apply truncated spans
    for (start, end), role in zip(spans, roles):
        if role != "assistant":
            continue
        s = min(max(start, 0), L)
        e = min(max(end, 0), L)
        if s >= e:
            continue
        assistant_mask[s:e] = 1

    return ids, assistant_mask


def _batchify_right_pad(
    examples: List[Dict[str, torch.Tensor]],
    model: Any,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create right-padded tensors for a batch of encoded samples.

    Returns (input_ids[B,L], attention_mask[B,L], assistant_mask_tokens[B,L]).
    """
    assert len(examples) >= 1
    max_len = max(int(ex["input_ids"].shape[0]) for ex in examples)
    assert max_len >= 1

    B = len(examples)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    assistant_mask_tokens = torch.zeros((B, max_len), dtype=torch.long)

    for i, ex in enumerate(examples):
        ids: torch.Tensor = ex["input_ids"]  # [L]
        mask: torch.Tensor = ex["assistant_mask"]  # [L]
        assert ids.ndim == mask.ndim == 1
        L = int(ids.shape[0])
        assert L == int(mask.shape[0])
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        assistant_mask_tokens[i, :L] = mask

    return input_ids, attention_mask, assistant_mask_tokens


def _nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token negative log-likelihood of shape [B, L-1]."""
    assert logits.ndim == 3 and logits.shape[:2] == input_ids.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    nll = -shift_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    assert nll.shape == (input_ids.shape[0], input_ids.shape[1] - 1)
    return nll


@torch.no_grad()
def _compute_nll(
    nn_model: StandardizedTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    collect_activations: bool = False,
    layer_index: int = None,
) -> torch.Tensor:
    """Return per-token negative log-likelihood of shape [B, L-1]."""
    B, L = input_ids.shape
    if collect_activations:
        assert layer_index is not None

        # Run intervention without autograd to prevent graph accumulation
        with torch.no_grad():
            with nn_model.trace(input_ids, attention_mask=attention_mask):
                activations = nn_model.layers_output[layer_index].save()
                logits = nn_model.logits.save()
                assert logits.shape == (
                    B,
                    L,
                    nn_model.config.vocab_size,
                ), f"logits.shape: {logits.shape}, B: {B}, L: {L}, vocab_size: {nn_model.config.vocab_size}"
        return _nll(logits, input_ids), activations
    else:
        with torch.inference_mode():
            outputs = nn_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            logits = outputs.logits
        return _nll(logits, input_ids)


@torch.no_grad()
def _compute_nll_intervened(
    nn_model: StandardizedTransformer,
    layer_index: int,
    delta_vec: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    zero_ablate: bool = False,
    target_activations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply global intervention at layer across all tokens; return per-token NLL [B, L-1].

    Args:
        delta_vec: Subspace basis vector (non-normalized)
        input_ids: input ids [B, L]
        attention_mask: attention mask [B, L]
        target_activations: target activations to set the component along delta_vec to
        zero_ablate: if True, project out the component along delta_vec; if False, set the component along delta_vec to the target_values.
    """
    B, L = input_ids.shape
    # Ensure vector device/dtype matches layer params
    nn_model.dispatch()
    param = next(nn_model.layers[layer_index].parameters())
    v = delta_vec.to(device=param.device, dtype=param.dtype).to(
        torch.float32
    )  # Perform the projection in float32.
    assert v.ndim == 1 and v.shape[0] == nn_model.config.hidden_size
    # Normalize vector for projection
    v = v / v.norm()
    target_activations = target_activations.to(torch.float32)

    assert target_activations.ndim == 3 and target_activations.shape == (
        B,
        L,
        nn_model.config.hidden_size,
    ), f"target_activations.shape: {target_activations.shape}"

    # Run intervention without autograd to prevent graph accumulation
    with torch.no_grad():
        with nn_model.trace(input_ids, attention_mask=attention_mask):
            activations = nn_model.layers_output[layer_index].save()
            dt = activations.dtype
            activations = activations.to(torch.float32)
            # Current projection coefficient per token: (x · v̂)
            proj_coeff = torch.sum(activations * v.view(1, 1, -1), dim=-1, keepdim=True)
            assert proj_coeff.shape == (B, L, 1)
            if not zero_ablate:
                target_coeff = torch.sum(
                    target_activations * v.view(1, 1, -1), dim=-1, keepdim=True
                )
                assert target_coeff.shape == (B, L, 1)
                additive = (target_coeff - proj_coeff) * v.view(1, 1, -1)
                assert additive.shape == (B, L, nn_model.config.hidden_size)
                # Set projection to the average scalar value
                nn_model.layers_output[layer_index] = (activations + additive).to(dt)
            else:
                # Project out v from activations: x - (x · v̂)v̂
                nn_model.layers_output[layer_index] = (
                    activations - (proj_coeff * v.view(1, 1, -1))
                ).to(dt)
            logits = nn_model.logits.save()
        nll = _nll(logits, input_ids)
        del logits
    return nll.cpu()


def _masked_mean_ce_and_ppl(
    nll: torch.Tensor, mask: torch.Tensor
) -> Tuple[float, float, int]:
    """Compute mean CE and PPL over mask (boolean or 0/1). Returns (ce, ppl, count)."""
    assert nll.shape == mask.shape
    mask = mask.to(nll.dtype)
    total = (nll * mask).sum().item()
    count = int(mask.sum().item())
    if count == 0:
        return float("nan"), float("nan"), 0
    ce = total / float(count)
    ppl = math.exp(ce)
    return ce, ppl, count


def _build_masks(
    attention_mask: torch.Tensor,
    assistant_mask_tokens: torch.Tensor,
    after_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (mask_all[B,L-1], mask_after_k[B,L-1], L) where masks are boolean.

    mask_all selects valid assistant targets; mask_after_k additionally enforces index > after_k.
    """
    assert attention_mask.ndim == 2 and assistant_mask_tokens.ndim == 2
    B, L = attention_mask.shape
    assert assistant_mask_tokens.shape == (B, L)
    valid = attention_mask[:, 1:].to(torch.bool)
    asst_tgt = (assistant_mask_tokens[:, 1:] > 0).to(valid.device)
    mask_all = valid & asst_tgt
    assert mask_all.shape == (B, L - 1)
    idx = torch.arange(L - 1, device=mask_all.device)
    mask_after_k = mask_all & (idx > after_k)
    assert mask_after_k.shape == mask_all.shape
    return mask_all, mask_after_k, L


def _sum_all_and_after(
    nll: torch.Tensor, mask_all: torch.Tensor, mask_after_k: torch.Tensor
) -> Tuple[float, float]:
    """Return (sum_all, sum_after_k) of NLL under the provided masks."""
    assert nll.shape == mask_all.shape == mask_after_k.shape
    mask_all_f = mask_all.to(nll.dtype).to(nll.device)
    mask_after_f = mask_after_k.to(nll.dtype).to(nll.device)
    s_all = float((nll * mask_all_f).sum().item())
    s_after = float((nll * mask_after_f).sum().item())
    return s_all, s_after


def _sum_at_index(nll: torch.Tensor, mask_all: torch.Tensor, index: int) -> float:
    """Sum NLL over assistant targets at a specific target index (column)."""
    assert nll.ndim == 2 and mask_all.ndim == 2 and nll.shape == mask_all.shape
    B, Lm1 = nll.shape
    assert 0 <= index < Lm1
    col_mask = mask_all[:, index].to(nll.dtype).to(nll.device)
    s = float((nll[:, index] * col_mask).sum().item())
    return s


def _dynamic_batch_size(curr_len: int, base_bs: int, max_len: int) -> int:
    """Return dynamic batch size by doubling when length falls below successive 3/4 thresholds.

    If curr_len <= (3/4) * max_len -> 2x, if <= (3/4)^2 * max_len -> 4x, etc.
    """
    assert (
        isinstance(curr_len, int)
        and isinstance(base_bs, int)
        and isinstance(max_len, int)
    )
    assert curr_len >= 1 and base_bs >= 1 and max_len >= 1
    ratio = float(curr_len) / float(max_len)
    k = 0
    threshold = 3.0 / 4.0
    while ratio <= threshold:
        k += 1
        threshold *= 3.0 / 4.0

    return base_bs * (k + 1)


def run_causal_effect(method: Any) -> None:
    """Evaluate loss and loss drop when subtracting act-diff vectors at a layer.

    - Loads finetuning dataset from organism config
    - Builds assistant-only masks (multi-turn)
    - Computes baseline CE/PPL and with intervention
    - Saves per-position results under activation_difference_lens results dir
    """
    # Assumptions and config
    cfg = method.cfg.diffing.method.causal_effect
    assert cfg.enabled is True
    split: str = str(cfg.split)
    batch_size: int = int(cfg.batch_size)
    max_samples: int = int(cfg.max_samples)
    max_total_tokens: int = int(cfg.max_total_tokens)
    after_k: int = int(cfg.after_k)
    assert hasattr(cfg, "num_random_vectors")
    num_random_vectors: int = int(cfg.num_random_vectors)
    assert num_random_vectors >= 1
    assert hasattr(cfg, "zero_ablate")
    zero_ablate: bool = bool(cfg.zero_ablate)
    overwrite: bool = bool(method.cfg.diffing.method.overwrite) or True

    # Finetuning (training) dataset metadata from organism config
    org = method.cfg.organism
    assert hasattr(org, "training_dataset")
    td = org.training_dataset
    train_ds_id: str = str(td.id)
    train_is_chat: bool = bool(td.is_chat)
    train_text_column: str = str(getattr(td, "text_column", "text"))

    tokenizer = method.tokenizer
    model = method.finetuned_model
    base_model = method.base_model

    assert base_model is not None
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None

    # Build evaluation groups from tasks: key = (abs_layer, diff_source_dataset, eval_alias)
    tasks = getattr(cfg, "tasks")
    assert tasks is not None and len(tasks) >= 1

    group_to_positions: Dict[Tuple[int, str, str], List[int]] = {}
    group_to_eval_cfg: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    for task in tasks:
        rel_layer = float(task["layer"])  # relative in [0,1]
        abs_layer = int(
            get_layer_indices(method.base_model_cfg.model_id, [rel_layer])[0]
        )
        diff_src_ds = str(task["diff_source_dataset"])
        eval_alias = str(task.get("eval_dataset", "training"))
        pos_list = [int(p) for p in task["positions"]]
        key = (abs_layer, diff_src_ds, eval_alias)
        group_to_positions.setdefault(key, []).extend(pos_list)

        if key not in group_to_eval_cfg:
            if eval_alias == "training":
                eval_ds_id = train_ds_id
                eval_is_chat = train_is_chat
                eval_text_column = train_text_column
                eval_messages_column = "messages"
            else:
                eval_ds_id = eval_alias
                assert "eval_dataset_is_chat" in task
                eval_is_chat = bool(task["eval_dataset_is_chat"])
                eval_text_column = str(task.get("eval_dataset_text_column", "text"))
                eval_messages_column = str(
                    task.get("eval_dataset_messages_column", "messages")
                )
            group_to_eval_cfg[key] = {
                "eval_ds_id": eval_ds_id,
                "eval_is_chat": eval_is_chat,
                "eval_text_column": eval_text_column,
                "eval_messages_column": eval_messages_column,
            }
        else:
            cfg_prev = group_to_eval_cfg[key]
            assert cfg_prev["eval_ds_id"] == (
                train_ds_id if eval_alias == "training" else eval_alias
            )
            if eval_alias == "training":
                assert cfg_prev["eval_is_chat"] == train_is_chat
                assert cfg_prev["eval_text_column"] == train_text_column
                assert cfg_prev["eval_messages_column"] == "messages"
            else:
                assert "eval_dataset_is_chat" in task
                assert cfg_prev["eval_is_chat"] == bool(task["eval_dataset_is_chat"])
                assert cfg_prev["eval_text_column"] == str(
                    task.get("eval_dataset_text_column", "text")
                )
                assert cfg_prev["eval_messages_column"] == str(
                    task.get("eval_dataset_messages_column", "messages")
                )

    # Deduplicate and sort positions per group
    for G in list(group_to_positions.keys()):
        group_to_positions[G] = sorted(set(group_to_positions[G]))

    def _finalize(sum_val: float, cnt: int) -> Tuple[float, float]:
        if cnt == 0:
            return float("nan"), float("nan")
        ce = sum_val / float(cnt)
        return ce, math.exp(ce)

    # Iterate groups: (layer, diff_source_dataset, eval_dataset)
    for (
        abs_layer,
        diff_source_dataset,
        eval_alias,
    ), positions in group_to_positions.items():
        diff_source_dataset_dir = dataset_dir_name(diff_source_dataset)
        eval_cfg = group_to_eval_cfg[(abs_layer, diff_source_dataset, eval_alias)]
        eval_ds_id: str = str(eval_cfg["eval_ds_id"])
        eval_is_chat: bool = bool(eval_cfg["eval_is_chat"])
        eval_text_column: str = str(eval_cfg["eval_text_column"])
        eval_messages_column: str = str(eval_cfg["eval_messages_column"])
        eval_dir: str = dataset_dir_name(eval_ds_id)
        logger.info(
            f"Causal effect: layer={abs_layer}, diff_source={diff_source_dataset}, eval={eval_alias} ({eval_ds_id}), positions={positions}"
        )

        # Determine which positions to compute (respect overwrite flag)
        pos_to_outpath: Dict[int, Any] = {}
        positions_to_run: List[int] = []
        for p in positions:
            out_dir = (
                method.results_dir
                / f"layer_{abs_layer}"
                / diff_source_dataset_dir
                / "causal_effect"
                / f"eval_{eval_dir}"
                / f"position_{p}"
            )
            out_path = out_dir / "results.json"
            if out_path.exists() and not overwrite:
                logger.info(
                    f"Skipping position p={p} at layer={abs_layer} dataset={diff_source_dataset} because results exist and overwrite=False"
                )
                continue
            if out_path.exists() and overwrite:
                logger.info(f"Overwriting existing results at {out_path}")
            pos_to_outpath[p] = out_path
            positions_to_run.append(p)

        if len(positions_to_run) == 0:
            logger.info("All positions already computed for this layer; skipping")
            continue

        # Load per-position vectors once (per group)
        hidden_size = model.config.hidden_size
        pos_to_vec: Dict[int, torch.Tensor] = {}
        for p in positions_to_run:
            vec = load_position_mean_vector(
                method, diff_source_dataset_dir, abs_layer, p
            )
            assert vec.ndim == 1 and vec.shape == (hidden_size,)
            pos_to_vec[p] = vec

        # Sample random vectors once per group (no normalization needed when projecting out)
        rand_vecs: List[torch.Tensor] = [
            torch.randn(hidden_size) for _ in range(num_random_vectors)
        ]

        # Accumulators shared across positions
        counts_all_total: int = 0
        counts_after_total: int = 0
        counts_idx_total: Dict[int, int] = {}

        ft_all_sum: float = 0.0
        ft_after_sum: float = 0.0
        ft_idx_sum: Dict[int, float] = {}

        base_all_sum: float = 0.0
        base_after_sum: float = 0.0
        base_idx_sum: Dict[int, float] = {}

        rand_all_sum: List[float] = [0.0 for _ in range(num_random_vectors)]
        rand_after_sum: List[float] = [0.0 for _ in range(num_random_vectors)]
        rand_idx_sum: List[Dict[int, float]] = [{} for _ in range(num_random_vectors)]

        # Streaming CE stats (min/max/median via reservoir) for 'all' and 'after_k'
        ft_stats_all = StreamingCEStats()
        ft_stats_after = StreamingCEStats()
        base_stats_all = StreamingCEStats()
        base_stats_after = StreamingCEStats()
        rand_stats_all: List[StreamingCEStats] = [
            StreamingCEStats() for _ in range(num_random_vectors)
        ]
        rand_stats_after: List[StreamingCEStats] = [
            StreamingCEStats() for _ in range(num_random_vectors)
        ]
        int_stats_all: Dict[int, StreamingCEStats] = {
            p: StreamingCEStats() for p in positions_to_run
        }
        int_stats_after: Dict[int, StreamingCEStats] = {
            p: StreamingCEStats() for p in positions_to_run
        }

        # Per-position accumulators for intervention
        int_all_sum: Dict[int, float] = {p: 0.0 for p in positions_to_run}
        int_after_sum: Dict[int, float] = {p: 0.0 for p in positions_to_run}
        int_idx_sum_at_p: Dict[int, float] = {p: 0.0 for p in positions_to_run}

        # Load and encode evaluation dataset for this group
        logger.info(
            f"Loading eval dataset {eval_ds_id} split={split} (chat={eval_is_chat})"
        )
        dataset = load_dataset_from_hub_or_local(eval_ds_id, split=split)

        encoded: List[Dict[str, torch.Tensor]] = []
        for sample in tqdm(dataset, desc="Encoding samples"):
            if len(encoded) >= max_samples:
                break
            if eval_is_chat:
                assert eval_messages_column in sample
                ids, asst_mask = _encode_chat(
                    sample[eval_messages_column], tokenizer, max_total_tokens
                )
            else:
                ids, asst_mask = _encode_non_chat(
                    sample, tokenizer, eval_text_column, max_total_tokens
                )
            encoded.append({"input_ids": ids, "assistant_mask": asst_mask})

        assert len(encoded) > 0
        logger.info(
            f"Prepared {len(encoded)} encoded samples (max_total_tokens={max_total_tokens})"
        )

        # Sort by length to minimize padding per batch and set reference max length
        encoded.sort(key=lambda ex: int(ex["input_ids"].shape[0]), reverse=True)
        max_seq_len: int = int(encoded[0]["input_ids"].shape[0])

        # Process in batches
        progress_bar = tqdm(total=len(encoded), desc="Processing batches")
        i = 0
        while i < len(encoded):
            curr_len = int(encoded[i]["input_ids"].shape[0])
            dyn_bs = _dynamic_batch_size(curr_len, batch_size, max_seq_len)
            batch = encoded[i : i + dyn_bs]
            input_ids_cpu, attention_mask_cpu, assistant_mask_tokens = (
                _batchify_right_pad(batch, model, int(tokenizer.pad_token_id))
            )

            # Base model NLL
            progress_bar.set_description("Processing base model")
            placed_base = place_inputs(input_ids_cpu, attention_mask_cpu, base_model)
            input_ids_base = placed_base["input_ids"]
            attention_mask_base = placed_base["attention_mask"]
            B, L = input_ids_cpu.shape
            assert attention_mask_cpu.shape == (B, L)
            assert assistant_mask_tokens.shape == (B, L)
            nll_base, activations_base = _compute_nll(
                base_model,
                input_ids_base,
                attention_mask_base,
                collect_activations=True,
                layer_index=abs_layer,
            )  # [B, L-1]
            assert nll_base.shape == (B, L - 1)
            del input_ids_base, attention_mask_base, placed_base

            # Finetuned model NLL
            progress_bar.set_description("Processing finetuned model")
            placed_ft = place_inputs(input_ids_cpu, attention_mask_cpu, model)
            input_ids_ft = placed_ft["input_ids"]
            attention_mask_ft = placed_ft["attention_mask"]
            nll_ft = _compute_nll(model, input_ids_ft, attention_mask_ft)  # [B, L-1]
            assert nll_ft.shape == (B, L - 1)

            # Masks (shared across variants)
            mask_all, mask_after_k, L_full = _build_masks(
                attention_mask_ft, assistant_mask_tokens, after_k
            )
            assert L_full == L
            assert nll_base.shape == mask_all.shape == mask_after_k.shape

            # Update streaming stats for base and finetuned
            _update_stats_from_masked(nll_ft, mask_all, ft_stats_all)
            _update_stats_from_masked(nll_ft, mask_after_k, ft_stats_after)
            _update_stats_from_masked(nll_base, mask_all, base_stats_all)
            _update_stats_from_masked(nll_base, mask_after_k, base_stats_after)

            # Counts
            counts_all_total += int(mask_all.sum().item())
            counts_after_total += int(mask_after_k.sum().item())
            for p in positions_to_run:
                t = p + 1
                if t < (L - 1):
                    cnt_t = int(mask_all[:, t].sum().item())
                    counts_idx_total[t] = counts_idx_total.get(t, 0) + cnt_t

            # Finetuned sums
            s_ft_all, s_ft_after = _sum_all_and_after(nll_ft, mask_all, mask_after_k)
            ft_all_sum += s_ft_all
            ft_after_sum += s_ft_after
            for p in positions_to_run:
                t = p + 1
                if t < (L - 1):
                    ft_idx_sum[t] = ft_idx_sum.get(t, 0.0) + _sum_at_index(
                        nll_ft, mask_all, t
                    )

            # Base sums
            s_b_all, s_b_after = _sum_all_and_after(nll_base, mask_all, mask_after_k)
            base_all_sum += s_b_all
            base_after_sum += s_b_after
            for p in positions_to_run:
                t = p + 1
                if t < (L - 1):
                    base_idx_sum[t] = base_idx_sum.get(t, 0.0) + _sum_at_index(
                        nll_base, mask_all, t
                    )

            # Random interventions (shared across positions)
            nll_rand_list: List[torch.Tensor] = []
            progress_bar.set_description("Processing random interventions")
            for rv in rand_vecs:
                nll_r = _compute_nll_intervened(
                    nn_model=model,
                    layer_index=abs_layer,
                    delta_vec=rv,
                    input_ids=input_ids_ft,
                    attention_mask=attention_mask_ft,
                    target_activations=activations_base,
                    zero_ablate=zero_ablate,
                )
                assert nll_r.shape == (B, L - 1)
                nll_rand_list.append(nll_r)

            for k, nll_r in enumerate(nll_rand_list):
                s_r_all, s_r_after = _sum_all_and_after(nll_r, mask_all, mask_after_k)
                rand_all_sum[k] += s_r_all
                rand_after_sum[k] += s_r_after
                _update_stats_from_masked(nll_r, mask_all, rand_stats_all[k])
                _update_stats_from_masked(nll_r, mask_after_k, rand_stats_after[k])
                for p in positions_to_run:
                    t = p + 1
                    if t < (L - 1):
                        rand_idx_sum[k][t] = rand_idx_sum[k].get(
                            t, 0.0
                        ) + _sum_at_index(nll_r, mask_all, t)

            # Position-specific intervention (per position)
            progress_bar.set_description("Processing position-specific interventions")
            for p in positions_to_run:
                vec = pos_to_vec[p]
                nll_int = _compute_nll_intervened(
                    nn_model=model,
                    layer_index=abs_layer,
                    delta_vec=vec,
                    input_ids=input_ids_ft,
                    attention_mask=attention_mask_ft,
                    target_activations=activations_base,
                    zero_ablate=zero_ablate,
                )
                assert nll_int.shape == (B, L - 1)
                s_i_all, s_i_after = _sum_all_and_after(nll_int, mask_all, mask_after_k)
                int_all_sum[p] += s_i_all
                int_after_sum[p] += s_i_after
                _update_stats_from_masked(nll_int, mask_all, int_stats_all[p])
                _update_stats_from_masked(nll_int, mask_after_k, int_stats_after[p])
                t = p + 1
                if t < (L - 1):
                    int_idx_sum_at_p[p] += _sum_at_index(nll_int, mask_all, t)

            # Free batch tensors
            del (
                input_ids_cpu,
                attention_mask_cpu,
                assistant_mask_tokens,
                nll_ft,
                nll_base,
                nll_rand_list,
                batch,
                placed_ft,
                mask_all,
                mask_after_k,
            )
            gc.collect()
            torch.cuda.empty_cache()
            progress_bar.update(dyn_bs)
            i += dyn_bs

        # Finalize and save per-position results
        for p in positions_to_run:
            out_path = pos_to_outpath[p]
            out_dir = out_path.parent
            t = p + 1

            cnt_all = counts_all_total
            cnt_after = counts_after_total
            cnt_at_t = counts_idx_total.get(t, 0)
            cnt_excl = cnt_all - cnt_at_t

            res = {
                "finetuned": {},
                "base": {},
                "intervention": {},
                "random": [],
                "random_mean": {},
            }

            # Finetuned finalize
            ce_ft_all, ppl_ft_all = _finalize(ft_all_sum, cnt_all)
            ce_ft_after, ppl_ft_after = _finalize(ft_after_sum, cnt_after)
            ce_ft_excl, ppl_ft_excl = _finalize(
                ft_all_sum - ft_idx_sum.get(t, 0.0), cnt_excl
            )
            res["finetuned"]["all"] = {
                "ce": ce_ft_all,
                "ppl": ppl_ft_all,
                "count": cnt_all,
                "min_ce": ft_stats_all.min_ce(),
                "max_ce": ft_stats_all.max_ce(),
                "median_ce": ft_stats_all.median_ce(),
            }
            res["finetuned"]["after_k"] = {
                "ce": ce_ft_after,
                "ppl": ppl_ft_after,
                "count": cnt_after,
                "min_ce": ft_stats_after.min_ce(),
                "max_ce": ft_stats_after.max_ce(),
                "median_ce": ft_stats_after.median_ce(),
            }
            res["finetuned"]["exclude_pos"] = {
                "ce": ce_ft_excl,
                "ppl": ppl_ft_excl,
                "count": cnt_excl,
            }

            # Base finalize
            ce_b_all, ppl_b_all = _finalize(base_all_sum, cnt_all)
            ce_b_after, ppl_b_after = _finalize(base_after_sum, cnt_after)
            ce_b_excl, ppl_b_excl = _finalize(
                base_all_sum - base_idx_sum.get(t, 0.0), cnt_excl
            )
            res["base"]["all"] = {
                "ce": ce_b_all,
                "ppl": ppl_b_all,
                "count": cnt_all,
                "min_ce": base_stats_all.min_ce(),
                "max_ce": base_stats_all.max_ce(),
                "median_ce": base_stats_all.median_ce(),
            }
            res["base"]["after_k"] = {
                "ce": ce_b_after,
                "ppl": ppl_b_after,
                "count": cnt_after,
                "min_ce": base_stats_after.min_ce(),
                "max_ce": base_stats_after.max_ce(),
                "median_ce": base_stats_after.median_ce(),
            }
            res["base"]["exclude_pos"] = {
                "ce": ce_b_excl,
                "ppl": ppl_b_excl,
                "count": cnt_excl,
            }

            # Intervention finalize (position-specific)
            ce_i_all, ppl_i_all = _finalize(int_all_sum[p], cnt_all)
            ce_i_after, ppl_i_after = _finalize(int_after_sum[p], cnt_after)
            ce_i_excl, ppl_i_excl = _finalize(
                int_all_sum[p] - int_idx_sum_at_p[p], cnt_excl
            )
            interv_all = {
                "ce": ce_i_all,
                "ppl": ppl_i_all,
                "count": cnt_all,
                "min_ce": int_stats_all[p].min_ce(),
                "max_ce": int_stats_all[p].max_ce(),
                "median_ce": int_stats_all[p].median_ce(),
            }
            interv_after = {
                "ce": ce_i_after,
                "ppl": ppl_i_after,
                "count": cnt_after,
                "min_ce": int_stats_after[p].min_ce(),
                "max_ce": int_stats_after[p].max_ce(),
                "median_ce": int_stats_after[p].median_ce(),
            }
            interv_excl = {"ce": ce_i_excl, "ppl": ppl_i_excl, "count": cnt_excl}

            if not (math.isnan(ce_ft_all) or math.isnan(ce_i_all)):
                interv_all["incr_ce"] = ce_i_all - ce_ft_all
                interv_all["incr_ppl"] = ppl_i_all - ppl_ft_all
                interv_all["incr_rel_ce"] = (ce_i_all - ce_ft_all) / (
                    ce_b_all - ce_ft_all
                )
                interv_all["incr_rel_ppl"] = (ppl_i_all - ppl_ft_all) / (
                    ppl_b_all - ppl_ft_all
                )
            else:
                interv_all["incr_ce"] = float("nan")
                interv_all["incr_ppl"] = float("nan")
                interv_all["incr_rel_ce"] = float("nan")
                interv_all["incr_rel_ppl"] = float("nan")

            if not (math.isnan(ce_ft_after) or math.isnan(ce_i_after)):
                interv_after["incr_ce"] = ce_i_after - ce_ft_after
                interv_after["incr_ppl"] = ppl_i_after - ppl_ft_after
                interv_after["incr_rel_ce"] = (ce_i_after - ce_ft_after) / (
                    ce_b_after - ce_ft_after
                )
                interv_after["incr_rel_ppl"] = (ppl_i_after - ppl_ft_after) / (
                    ppl_b_after - ppl_ft_after
                )
            else:
                interv_after["incr_ce"] = float("nan")
                interv_after["incr_ppl"] = float("nan")
                interv_after["incr_rel_ce"] = float("nan")
                interv_after["incr_rel_ppl"] = float("nan")

            if not (math.isnan(ce_ft_excl) or math.isnan(ce_i_excl)):
                interv_excl["incr_ce"] = ce_i_excl - ce_ft_excl
                interv_excl["incr_ppl"] = ppl_i_excl - ppl_ft_excl
                interv_excl["incr_rel_ce"] = (ce_i_excl - ce_ft_excl) / (
                    ce_b_excl - ce_ft_excl
                )
                interv_excl["incr_rel_ppl"] = (ppl_i_excl - ppl_ft_excl) / (
                    ppl_b_excl - ppl_ft_excl
                )
            else:
                interv_excl["incr_ce"] = float("nan")
                interv_excl["incr_ppl"] = float("nan")
                interv_excl["incr_rel_ce"] = float("nan")
                interv_excl["incr_rel_ppl"] = float("nan")

            res["intervention"]["all"] = interv_all
            res["intervention"]["after_k"] = interv_after
            res["intervention"]["exclude_pos"] = interv_excl

            # Per-random finalized metrics and mean
            sum_rand_all_total = 0.0
            sum_rand_after_total = 0.0
            sum_rand_excl_total = 0.0

            for k in range(num_random_vectors):
                ce_r_all, ppl_r_all = _finalize(rand_all_sum[k], cnt_all)
                ce_r_after, ppl_r_after = _finalize(rand_after_sum[k], cnt_after)
                excl_sum_k = rand_all_sum[k] - rand_idx_sum[k].get(t, 0.0)
                ce_r_excl, ppl_r_excl = _finalize(excl_sum_k, cnt_excl)

                sum_rand_all_total += rand_all_sum[k]
                sum_rand_after_total += rand_after_sum[k]
                sum_rand_excl_total += excl_sum_k

                entry = {
                    "all": {
                        "ce": ce_r_all,
                        "ppl": ppl_r_all,
                        "count": cnt_all,
                        "min_ce": rand_stats_all[k].min_ce(),
                        "max_ce": rand_stats_all[k].max_ce(),
                        "median_ce": rand_stats_all[k].median_ce(),
                    },
                    "after_k": {
                        "ce": ce_r_after,
                        "ppl": ppl_r_after,
                        "count": cnt_after,
                        "min_ce": rand_stats_after[k].min_ce(),
                        "max_ce": rand_stats_after[k].max_ce(),
                        "median_ce": rand_stats_after[k].median_ce(),
                    },
                    "exclude_pos": {
                        "ce": ce_r_excl,
                        "ppl": ppl_r_excl,
                        "count": cnt_excl,
                    },
                }

                for key, (ce_r, ppl_r, ce_ft, ppl_ft, ce_b) in {
                    "all": (ce_r_all, ppl_r_all, ce_ft_all, ppl_ft_all, ce_b_all),
                    "after_k": (
                        ce_r_after,
                        ppl_r_after,
                        ce_ft_after,
                        ppl_ft_after,
                        ce_b_after,
                    ),
                    "exclude_pos": (
                        ce_r_excl,
                        ppl_r_excl,
                        ce_ft_excl,
                        ppl_ft_excl,
                        ce_b_excl,
                    ),
                }.items():
                    if not (math.isnan(ce_ft) or math.isnan(ce_r)):
                        entry[key]["incr_ce"] = ce_r - ce_ft
                        entry[key]["incr_ppl"] = ppl_r - ppl_ft
                        entry[key]["incr_rel_ce"] = (ce_r - ce_ft) / (ce_b - ce_ft)
                        entry[key]["incr_rel_ppl"] = (ppl_r - ppl_ft) / (
                            res["base"][key]["ppl"] - ppl_ft
                        )
                    else:
                        entry[key]["incr_ce"] = float("nan")
                        entry[key]["incr_ppl"] = float("nan")
                        entry[key]["incr_rel_ce"] = float("nan")
                        entry[key]["incr_rel_ppl"] = float("nan")

                res["random"].append(entry)

            # Random mean across K
            rm_all_sum = sum_rand_all_total / float(num_random_vectors)
            rm_after_sum = sum_rand_after_total / float(num_random_vectors)
            rm_excl_sum = sum_rand_excl_total / float(num_random_vectors)

            ce_rm_all, ppl_rm_all = _finalize(rm_all_sum, cnt_all)
            ce_rm_after, ppl_rm_after = _finalize(rm_after_sum, cnt_after)
            ce_rm_excl, ppl_rm_excl = _finalize(rm_excl_sum, cnt_excl)

            res["random_mean"]["all"] = {
                "ce": ce_rm_all,
                "ppl": ppl_rm_all,
                "count": cnt_all,
            }
            res["random_mean"]["after_k"] = {
                "ce": ce_rm_after,
                "ppl": ppl_rm_after,
                "count": cnt_after,
            }
            res["random_mean"]["exclude_pos"] = {
                "ce": ce_rm_excl,
                "ppl": ppl_rm_excl,
                "count": cnt_excl,
            }

            for key, (ce_rm, ppl_rm, ce_ft, ppl_ft, ce_b) in {
                "all": (ce_rm_all, ppl_rm_all, ce_ft_all, ppl_ft_all, ce_b_all),
                "after_k": (
                    ce_rm_after,
                    ppl_rm_after,
                    ce_ft_after,
                    ppl_ft_after,
                    ce_b_after,
                ),
                "exclude_pos": (
                    ce_rm_excl,
                    ppl_rm_excl,
                    ce_ft_excl,
                    ppl_ft_excl,
                    ce_b_excl,
                ),
            }.items():
                if not (math.isnan(ce_ft) or math.isnan(ce_rm)):
                    res["random_mean"][key]["incr_ce"] = ce_rm - ce_ft
                    res["random_mean"][key]["incr_ppl"] = ppl_rm - ppl_ft
                    res["random_mean"][key]["incr_rel_ce"] = (ce_rm - ce_ft) / (
                        ce_b - ce_ft
                    )
                    res["random_mean"][key]["incr_rel_ppl"] = (ppl_rm - ppl_ft) / (
                        res["base"][key]["ppl"] - ppl_ft
                    )
                else:
                    res["random_mean"][key]["incr_ce"] = float("nan")
                    res["random_mean"][key]["incr_ppl"] = float("nan")
                    res["random_mean"][key]["incr_rel_ce"] = float("nan")
                    res["random_mean"][key]["incr_rel_ppl"] = float("nan")

            # Save per-position
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "eval_dataset_id": eval_ds_id,
                "diff_source_dataset_id": diff_source_dataset,
                "split": split,
                "layer": abs_layer,
                "position": p,
                "after_k": after_k,
                "num_samples": len(encoded),
                "finetuned": res["finetuned"],
                "base": res["base"],
                "intervention": res["intervention"],
                "random": res["random"],
                "random_mean": res["random_mean"],
            }
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Saved causal effect results to {out_path}")

        # Release resources for this evaluation group
        del model
        torch.cuda.empty_cache()
