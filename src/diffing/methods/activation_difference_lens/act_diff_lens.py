from typing import List, Dict, Any, Tuple
import torch
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from nnsight import NNsight
from omegaconf import DictConfig
from pathlib import Path
import json
from collections import defaultdict
import gc

from src.diffing.methods.diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices
from src.utils.model import logit_lens
from .auto_patch_scope import save_auto_patch_scope_variants
from .ui import visualize
from .steering import run_steering
from .token_relevance import run_token_relevance
from .util import norms_path, is_layer_complete
from src.utils.model import get_layers_from_nn_model, resolve_output
from .causal_effect import run_causal_effect
from .agents import ADLAgent, ADLBlackboxAgent
from src.utils.agents.base_agent import BaseAgent


def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: Any,
    split: str = "validation",
    text_column: str = "text",
    n: int = 10,
    max_samples: int = 1000,
    debug: bool = False,
) -> List[List[int]]:
    """
    Load HuggingFace dataset and tokenize sequences with n-character cutoff.

    Args:
        dataset_name: Name of the HuggingFace dataset
        tokenizer: Tokenizer to use
        split: Dataset split to use
        text_column: Column name containing text data
        n: Number of tokens to extract
        max_samples: Maximum number of samples to process
        debug: Whether to use fewer samples

    Returns:
        List of lists, where each inner list contains exactly n token IDs
    """
    logger.info(f"Loading dataset {dataset_name} (split: {split})")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    if debug:
        max_samples = min(20, max_samples)

    logger.info(
        f"Dataset loaded with {len(dataset)} samples, processing up to {max_samples}"
    )

    # Process samples
    first_n_tokens = []
    processed = 0

    for sample in tqdm(dataset, desc="Tokenizing sequences"):
        if processed >= max_samples:
            break

        text = sample[text_column]
        if not text or len(text.strip()) == 0:
            continue

        # Cut off at n*10 characters to speed up tokenization
        text_truncated = text[: n * 10]

        # Tokenize
        tokens = tokenizer.encode(text_truncated, add_special_tokens=True)
        # Enforce exact n tokens to maintain fixed shapes downstream
        if len(tokens) >= n:
            first_tokens = tokens[:n]
            assert len(first_tokens) == n
            first_n_tokens.append(first_tokens)
            processed += 1

    logger.info(f"Successfully tokenized {len(first_n_tokens)} sequences")
    return first_n_tokens


def _build_chat_positions(
    assistant_start_index: int,
    n: int,
    pre_assistant_k: int,
) -> Tuple[List[int], List[int]]:
    """Return (position_labels, absolute_indices) for [-k..-1, 0..n-1]."""
    assert assistant_start_index >= pre_assistant_k
    position_labels: List[int] = list(range(-pre_assistant_k, 0)) + list(range(0, n))
    absolute_indices: List[int] = []
    for label in position_labels:
        absolute_index = assistant_start_index + label
        assert absolute_index >= 0
        absolute_indices.append(absolute_index)
    assert len(position_labels) == pre_assistant_k + n
    assert len(absolute_indices) == pre_assistant_k + n
    return position_labels, absolute_indices


def load_and_tokenize_chat_dataset(
    dataset_name: str,
    tokenizer: Any,
    split: str,
    messages_column: str,
    n: int,
    pre_assistant_k: int,
    max_samples: int,
    debug: bool = False,
    max_user_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Load a chat dataset and prepare samples around assistant start.

    Returns list of dicts with keys: input_ids (List[int]), position_labels (List[int]), positions (List[int]).
    """
    logger.info(f"Loading chat dataset {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    if debug:
        max_samples = min(20, max_samples)
    processed = 0
    samples: List[Dict[str, Any]] = []

    for sample in tqdm(dataset, desc="Tokenizing chat sequences"):
        if processed >= max_samples:
            break

        messages = sample[messages_column]
        assert isinstance(messages, list) and len(messages) >= 2
        if messages[0]["role"] != "user":
            continue
        assert messages[1]["role"] == "assistant"

        # Truncate assistant content to 10 * n characters to speed up tokenization
        trunc_messages = [
            {"role": messages[0]["role"], "content": messages[0]["content"]},
            {"role": messages[1]["role"], "content": messages[1]["content"][: 10 * n]},
        ]

        user_only = [{"role": messages[0]["role"], "content": messages[0]["content"]}]
        user_ids: List[int] = tokenizer.apply_chat_template(
            user_only, tokenize=True, add_generation_prompt=True
        )

        if len(user_ids) > max_user_tokens:
            continue

        full_ids: List[int] = tokenizer.apply_chat_template(
            trunc_messages, tokenize=True, add_generation_prompt=False
        )

        assistant_start_index = len(user_ids)
        if len(full_ids) - assistant_start_index < n:
            continue  # drop samples with fewer than n assistant tokens

        # Feed only up to the first n assistant tokens
        truncated_ids = full_ids[: assistant_start_index + n]

        position_labels, absolute_indices = _build_chat_positions(
            assistant_start_index=assistant_start_index,
            n=n,
            pre_assistant_k=pre_assistant_k,
        )
        assert max(absolute_indices) < len(truncated_ids)

        samples.append(
            {
                "input_ids": truncated_ids,
                "positions": absolute_indices,
                "position_labels": position_labels,
            }
        )
        processed += 1

    logger.info(f"Prepared {len(samples)} chat samples")
    assert len(samples) > 0, "No valid chat samples after filtering"
    return samples


def extract_first_n_tokens_from_sequences(
    sequences: List[torch.Tensor],
) -> List[List[int]]:
    """
    Extract first n tokens from cached sequences.

    Args:
        sequences: List of tokenized sequences (tensors)

    Returns:
        List of lists, where each inner list contains up to n first token IDs
    """
    logger.info(f"Extracting first n tokens from {len(sequences)} sequences...")
    n = max(len(seq) for seq in sequences)
    first_n_tokens = []
    for sequence in sequences:
        seq_len = len(sequence)
        num_tokens = min(n, seq_len)
        if num_tokens > 0:
            tokens = [sequence[i].item() for i in range(num_tokens)]
            first_n_tokens.append(tokens)

    logger.info(f"Extracted first n tokens from {len(first_n_tokens)} sequences")
    return first_n_tokens


@torch.no_grad()
def extract_first_n_tokens_activations(
    model: torch.nn.Module,
    first_n_tokens: List[List[int]],
    layers: List[int],
    batch_size: int = 8,
) -> Dict[int, torch.Tensor]:
    """
    Extract activations from specified layers for first n tokens.

    Args:
        model: The transformer model
        first_n_tokens: List of token sequences (each up to n tokens)
        layers: List of layer indices to extract activations from
        batch_size: Batch size for processing

    Returns:
        Dict mapping layer index to tensor of shape [num_sequences, n, hidden_dim]
    """
    n = max(len(seq) for seq in first_n_tokens)
    logger.info(f"Extracting first n={n} tokens activations from layers {layers}...")

    model.eval()
    nn_model = NNsight(model)
    # Get model device for tensor operations
    model_device = next(model.parameters()).device
    logger.info(f"Model device: {model_device}")

    # Initialize storage for all layers
    all_activations = {layer: [] for layer in layers}

    # Process sequences in batches
    for i in tqdm(range(0, len(first_n_tokens), batch_size)):
        batch_sequences = first_n_tokens[i : i + batch_size]
        # Fail fast if sequences are not exactly length n
        assert all(
            len(seq) == n for seq in batch_sequences
        ), "All sequences must have exactly n tokens"
        batch_input_ids = torch.tensor(
            batch_sequences, dtype=torch.long, device=model_device
        )  # [B, n]
        assert batch_input_ids.shape == (len(batch_sequences), n)

        # Extract activations using nnsight for all layers
        layer_outputs = {}
        with nn_model.trace(batch_input_ids):
            for layer in layers:
                layer_outputs[layer] = resolve_output(
                    get_layers_from_nn_model(nn_model)[layer].output
                ).save()

        # Store activations for each layer
        for layer in layers:
            activations = layer_outputs[layer].cpu()  # [batch_size, n, hidden_dim]
            assert activations.shape[1] == n
            all_activations[layer].append(activations)

    # Concatenate all batches for each layer
    result = {}
    for layer in layers:
        result[layer] = torch.cat(
            all_activations[layer], dim=0
        )  # [num_sequences, n, hidden_dim]
        assert result[layer].shape[0] == len(first_n_tokens)
        assert result[layer].shape[1] == n

    # Clear memory
    del all_activations
    torch.cuda.empty_cache()
    gc.collect()

    return result


@torch.no_grad()
def extract_selected_positions_activations(
    model: torch.nn.Module,
    samples: List[Dict[str, Any]],
    layers: List[int],
    batch_size: int,
    pad_token_id: int,
) -> Dict[int, torch.Tensor]:
    """Extract activations at specific absolute indices for each sample.

    Returns dict[layer] -> Tensor[num_samples, P, hidden_dim]
    where P = len(samples[0]["positions"]).
    """
    assert len(samples) > 0
    num_positions = len(samples[0]["positions"])
    assert num_positions > 0

    model.eval()
    nn_model = NNsight(model)

    all_activations: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : i + batch_size]
        batch_input_ids_list: List[List[int]] = [b["input_ids"] for b in batch]
        batch_positions_list: List[List[int]] = [b["positions"] for b in batch]
        assert all(len(pos) == num_positions for pos in batch_positions_list)

        max_len = max(len(x) for x in batch_input_ids_list)
        batch_input_ids = torch.full(
            (len(batch), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros(
            (len(batch), max_len), dtype=torch.long, device=model.device
        )

        for row, seq in enumerate(batch_input_ids_list):
            seq_len = len(seq)
            batch_input_ids[row, :seq_len] = torch.tensor(
                seq, dtype=torch.long, device=model.device
            )
            attention_mask[row, :seq_len] = 1

        # Build per-batch position index once on the model device
        pos_index = torch.tensor(
            batch_positions_list, dtype=torch.long, device=model.device
        )  # [B, P]
        assert pos_index.shape == (len(batch), num_positions)
        batch_arange = torch.arange(len(batch), device=model.device).view(-1, 1)

        # Trace and directly save only the gathered activations at the desired positions
        with nn_model.trace(batch_input_ids, attention_mask=attention_mask):
            layer_outputs: Dict[int, torch.Tensor] = {}
            for layer in layers:
                hidden = resolve_output(
                    get_layers_from_nn_model(nn_model)[layer].output
                )  # [B, L, D]
                selected = hidden[batch_arange, pos_index, :].clone()  # [B, P, D]
                # Save directly to CPU to minimize GPU residency of saved tensors
                layer_outputs[layer] = selected.to("cpu", non_blocking=True).save()

        for layer in layers:
            gathered_cpu = layer_outputs[layer]
            assert gathered_cpu.shape == (
                len(batch),
                num_positions,
                gathered_cpu.shape[2],
            )
            all_activations[layer].append(gathered_cpu)

        # Clear VRAM after processing batch
        del layer_outputs, batch_input_ids, attention_mask, pos_index
        torch.cuda.empty_cache()
        gc.collect()

    result: Dict[int, torch.Tensor] = {}
    for layer in layers:
        result[layer] = torch.cat(all_activations[layer], dim=0)
    return result


class ActDiffLens(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.results_dir = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.layers = get_layer_indices(
            self.base_model_cfg.model_id, self.cfg.diffing.method.layers
        )
        self.overwrite: bool = bool(
            getattr(self.cfg.diffing.method, "overwrite", False)
        )

    def run(self):
        for dataset_entry in self.cfg.diffing.method.datasets:
            ctx = self.compute_differences(dataset_entry)
            if ctx is not None:
                self.analysis(ctx)

        steering_cfg = getattr(self.cfg.diffing.method, "steering", None)
        if steering_cfg is not None and getattr(steering_cfg, "enabled", False):
            run_steering(self)

        token_rel_cfg = self.cfg.diffing.method.token_relevance
        logger.info(f"Token relevance config: {token_rel_cfg}")
        if token_rel_cfg.enabled:
            logger.info("Running token relevance...")
            org = self.cfg.organism
            assert hasattr(org, "description_long")
            run_token_relevance(self)

        causal_cfg = getattr(self.cfg.diffing.method, "causal_effect", None)
        if causal_cfg is not None and getattr(causal_cfg, "enabled", False):
            logger.info("Running causal effect...")
            run_causal_effect(self)

    def _get_run_layers_and_aps_tasks(
        self, dataset_id: str
    ) -> Tuple[List[int], Dict[int, set]]:
        aps_layers_for_dataset_abs: List[int] = []
        aps_tasks_for_dataset: Dict[int, set] = {}
        aps_cfg_all = getattr(self.cfg.diffing.method, "auto_patch_scope", None)
        if aps_cfg_all is not None and getattr(aps_cfg_all, "enabled", False):
            assert hasattr(aps_cfg_all, "tasks")
            for task in aps_cfg_all.tasks:
                if str(task.get("dataset")) != str(dataset_id):
                    continue
                assert "layer" in task and "positions" in task
                abs_layer_list = get_layer_indices(
                    self.base_model_cfg.model_id, [float(task["layer"])]
                )
                assert len(abs_layer_list) == 1
                abs_layer = int(abs_layer_list[0])
                aps_layers_for_dataset_abs.append(abs_layer)
                pos_set = set(int(p) for p in task["positions"])
                if abs_layer not in aps_tasks_for_dataset:
                    aps_tasks_for_dataset[abs_layer] = set()
                aps_tasks_for_dataset[abs_layer].update(pos_set)
        run_layers: List[int] = sorted(
            set(self.layers) | set(aps_layers_for_dataset_abs)
        )
        return run_layers, aps_tasks_for_dataset

    def _compute_and_save_norms(
        self,
        dataset_id: str,
        run_layers: List[int],
        base_acts: Dict[int, torch.Tensor],
        ft_acts: Dict[int, torch.Tensor],
    ) -> None:
        any_layer_for_meta = run_layers[0]
        num_sequences = ft_acts[any_layer_for_meta].shape[0]
        base_model_norms: Dict[int, torch.Tensor] = {}
        ft_model_norms: Dict[int, torch.Tensor] = {}
        skip_tokens = 5
        for layer in run_layers:
            assert layer in ft_acts and layer in base_acts
            base_layer_acts = base_acts[layer]
            ft_layer_acts = ft_acts[layer]

            assert base_layer_acts.shape == ft_layer_acts.shape
            assert (
                base_layer_acts.shape[1] >= skip_tokens
            ), f"Need at least {skip_tokens} positions, got {base_layer_acts.shape[1]}"

            base_acts_truncated = base_layer_acts[:, skip_tokens:, :]
            ft_acts_truncated = ft_layer_acts[:, skip_tokens:, :]

            assert (
                base_acts_truncated.shape[1] != 0
            ), f"Base model activations have 0 positions, increase n or decrease skip_tokens"
            assert (
                ft_acts_truncated.shape[1] != 0
            ), f"Fine-tuned model activations have 0 positions, increase n or decrease skip_tokens"

            base_norms_per_pos = torch.norm(
                base_acts_truncated.to(torch.float32), dim=2
            )
            ft_norms_per_pos = torch.norm(ft_acts_truncated.to(torch.float32), dim=2)

            assert not torch.isnan(
                base_norms_per_pos
            ).any(), f"Layer {layer} - Base model norms contain NaN values"
            assert not torch.isnan(
                ft_norms_per_pos
            ).any(), f"Layer {layer} - Fine-tuned model norms contain NaN values"

            base_model_norms[layer] = base_norms_per_pos.flatten().mean()
            ft_model_norms[layer] = ft_norms_per_pos.flatten().mean()

            logger.info(
                f"Layer {layer} - Base model mean norm: {base_model_norms[layer].item():.3f}"
            )
            logger.info(
                f"Layer {layer} - Fine-tuned model mean norm: {ft_model_norms[layer].item():.3f}"
            )

        norms_data = {
            "base_model_norms": {
                layer: base_model_norms[layer].cpu() for layer in run_layers
            },
            "ft_model_norms": {
                layer: ft_model_norms[layer].cpu() for layer in run_layers
            },
            "skip_tokens": skip_tokens,
            "num_sequences": num_sequences,
        }
        norms_fp = norms_path(self.results_dir, dataset_id)
        torch.save(norms_data, norms_fp)
        logger.info(f"Saved model norm estimates to {norms_fp}")

    def _save_means_for_layer(
        self,
        out_dir: Path,
        position_labels: List[int],
        mean_diff: torch.Tensor,
        base_mean: torch.Tensor,
        ft_mean: torch.Tensor,
        num_sequences: int,
        activation_dim: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx_in_tensor, label in enumerate(position_labels):
            tensor_path = out_dir / f"mean_pos_{label}.pt"
            meta_path = out_dir / f"mean_pos_{label}.meta"
            need_write = (
                self.overwrite or (not tensor_path.exists()) or (not meta_path.exists())
            )
            if need_write:
                torch.save(mean_diff[idx_in_tensor], tensor_path)
                meta_data = {
                    "count": num_sequences,
                    "activation_dim": activation_dim,
                    "position": label,
                    "token_id": None,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta_data, f, indent=2)

            base_tensor_path = out_dir / f"base_mean_pos_{label}.pt"
            ft_tensor_path = out_dir / f"ft_mean_pos_{label}.pt"
            if self.overwrite or (not base_tensor_path.exists()):
                torch.save(base_mean[idx_in_tensor], base_tensor_path)
            if self.overwrite or (not ft_tensor_path.exists()):
                torch.save(ft_mean[idx_in_tensor], ft_tensor_path)

    def _cache_logit_lens_for_layer(
        self, out_dir: Path, position_labels: List[int]
    ) -> None:
        if not bool(self.cfg.diffing.method.logit_lens.cache):
            return
        k = int(self.cfg.diffing.method.logit_lens.k)
        for label in position_labels:
            mean_diff = torch.load(out_dir / f"mean_pos_{label}.pt", map_location="cpu")
            base_mean = torch.load(
                out_dir / f"base_mean_pos_{label}.pt", map_location="cpu"
            )
            ft_mean = torch.load(
                out_dir / f"ft_mean_pos_{label}.pt", map_location="cpu"
            )

            ll_path = out_dir / f"logit_lens_pos_{label}.pt"
            if self.overwrite or (not ll_path.exists()):
                probs, inv_probs = logit_lens(mean_diff, self.finetuned_model)
                top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                top_k_inv_probs, top_k_inv_indices = torch.topk(inv_probs, k, dim=-1)
                torch.save(
                    (top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices),
                    ll_path,
                )

            base_ll_path = out_dir / f"base_logit_lens_pos_{label}.pt"
            if self.overwrite or (not base_ll_path.exists()):
                base_probs, base_inv_probs = logit_lens(base_mean, self.finetuned_model)
                base_top_k_probs, base_top_k_indices = torch.topk(base_probs, k, dim=-1)
                base_top_k_inv_probs, base_top_k_inv_indices = torch.topk(
                    base_inv_probs, k, dim=-1
                )
                torch.save(
                    (
                        base_top_k_probs,
                        base_top_k_indices,
                        base_top_k_inv_probs,
                        base_top_k_inv_indices,
                    ),
                    base_ll_path,
                )

            ft_ll_path = out_dir / f"ft_logit_lens_pos_{label}.pt"
            if self.overwrite or (not ft_ll_path.exists()):
                ft_probs, ft_inv_probs = logit_lens(ft_mean, self.finetuned_model)
                ft_top_k_probs, ft_top_k_indices = torch.topk(ft_probs, k, dim=-1)
                ft_top_k_inv_probs, ft_top_k_inv_indices = torch.topk(
                    ft_inv_probs, k, dim=-1
                )
                torch.save(
                    (
                        ft_top_k_probs,
                        ft_top_k_indices,
                        ft_top_k_inv_probs,
                        ft_top_k_inv_indices,
                    ),
                    ft_ll_path,
                )

    def _run_auto_patch_scope_for_layer(
        self,
        dataset_id: str,
        layer: int,
        out_dir: Path,
        position_labels: List[int],
        aps_tasks_for_dataset: Dict[int, set],
    ) -> None:
        aps_cfg = self.cfg.diffing.method.auto_patch_scope
        if not bool(aps_cfg.enabled):
            return
        if layer not in aps_tasks_for_dataset:
            return
        norms_fp = norms_path(self.results_dir, dataset_id)
        assert norms_fp.exists()
        norms_data = torch.load(norms_fp, map_location="cpu")

        use_normalized = bool(aps_cfg.use_normalized)
        intersection_top_k = int(aps_cfg.intersection_top_k)
        tokens_k = int(aps_cfg.tokens_k)
        grader_cfg = dict(aps_cfg.grader)
        target_norm = float(norms_data["ft_model_norms"][layer].item())
        overwrite = bool(aps_cfg.overwrite)

        for label in position_labels:
            if int(label) not in aps_tasks_for_dataset[layer]:
                continue
            mean_diff = torch.load(out_dir / f"mean_pos_{label}.pt", map_location="cpu")
            base_mean = torch.load(
                out_dir / f"base_mean_pos_{label}.pt", map_location="cpu"
            )
            ft_mean = torch.load(
                out_dir / f"ft_mean_pos_{label}.pt", map_location="cpu"
            )
            save_auto_patch_scope_variants(
                out_dir=out_dir,
                label=int(label),
                layer=int(layer),
                mean_diff=mean_diff,
                base_mean=base_mean,
                ft_mean=ft_mean,
                base_model=self.base_model,
                ft_model=self.finetuned_model,
                tokenizer=self.tokenizer,
                intersection_top_k=intersection_top_k,
                tokens_k=tokens_k,
                grader_cfg=grader_cfg,
                overwrite=overwrite,
                use_normalized=use_normalized,
                target_norm=target_norm,
            )

    def compute_differences(self, dataset_entry: Dict[str, Any]) -> Dict[str, Any]:
        assert (
            isinstance(dataset_entry, (dict, DictConfig))
            and "id" in dataset_entry
            and "is_chat" in dataset_entry
        )
        dataset_id = str(dataset_entry["id"])
        is_chat: bool = bool(dataset_entry["is_chat"])

        if is_chat:
            n_positions_expected = int(self.cfg.diffing.method.pre_assistant_k) + int(
                self.cfg.diffing.method.n
            )
        else:
            n_positions_expected = int(self.cfg.diffing.method.n)

        cache_logit_lens: bool = bool(self.cfg.diffing.method.logit_lens.cache)

        run_layers, aps_tasks_for_dataset = self._get_run_layers_and_aps_tasks(
            dataset_id
        )
        norms_needed: bool = self.overwrite or (
            not norms_path(self.results_dir, dataset_id).exists()
        )

        if self.overwrite:
            layers_to_compute = list(run_layers)
        else:
            layers_to_compute = [
                layer
                for layer in run_layers
                if not is_layer_complete(
                    self.results_dir,
                    dataset_id,
                    layer,
                    n_positions_expected,
                    cache_logit_lens,
                )
            ]

        if len(layers_to_compute) == 0 and not norms_needed:
            logger.info(
                f"Skipping dataset {dataset_id}: all results present and overwrite=False"
            )
            if is_chat:
                pre_k = int(self.cfg.diffing.method.pre_assistant_k)
                n = int(self.cfg.diffing.method.n)
                position_labels = list(range(-pre_k, 0)) + list(range(0, n))
            else:
                position_labels = list(range(int(self.cfg.diffing.method.n)))
            return {
                "dataset_id": dataset_id,
                "run_layers": run_layers,
                "position_labels": position_labels,
                "aps_tasks_for_dataset": aps_tasks_for_dataset,
            }

        if is_chat:
            pre_k: int = int(self.cfg.diffing.method.pre_assistant_k)
            assert "messages_column" in dataset_entry
            samples = load_and_tokenize_chat_dataset(
                dataset_name=dataset_id,
                tokenizer=self.tokenizer,
                split=self.cfg.diffing.method.split,
                messages_column=dataset_entry["messages_column"],
                n=self.cfg.diffing.method.n,
                pre_assistant_k=pre_k,
                max_samples=self.cfg.diffing.method.max_samples,
            )

            base_acts = extract_selected_positions_activations(
                model=self.base_model,
                samples=samples,
                layers=run_layers,
                batch_size=self.cfg.diffing.method.batch_size,
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            self.clear_base_model()

            ft_acts = extract_selected_positions_activations(
                model=self.finetuned_model,
                samples=samples,
                layers=run_layers,
                batch_size=self.cfg.diffing.method.batch_size,
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            self.clear_finetuned_model()

            position_labels: List[int] = samples[0]["position_labels"]
            num_positions = len(position_labels)
        else:
            first_n_tokens = load_and_tokenize_dataset(
                dataset_id,
                self.tokenizer,
                split=self.cfg.diffing.method.split,
                text_column=dataset_entry["text_column"],
                n=self.cfg.diffing.method.n,
                max_samples=self.cfg.diffing.method.max_samples,
            )
            base_acts = extract_first_n_tokens_activations(
                self.base_model,
                first_n_tokens,
                run_layers,
                self.cfg.diffing.method.batch_size,
            )
            self.clear_base_model()

            ft_acts = extract_first_n_tokens_activations(
                self.finetuned_model,
                first_n_tokens,
                run_layers,
                self.cfg.diffing.method.batch_size,
            )
            self.clear_finetuned_model()

            position_labels = list(range(self.cfg.diffing.method.n))
            num_positions = len(position_labels)

        if norms_needed:
            self._compute_and_save_norms(
                dataset_id=dataset_id,
                run_layers=run_layers,
                base_acts=base_acts,
                ft_acts=ft_acts,
            )

        any_layer_for_meta = run_layers[0]
        num_sequences = ft_acts[any_layer_for_meta].shape[0]
        activation_dim = ft_acts[any_layer_for_meta].shape[2]

        for layer in list(layers_to_compute):
            diff = ft_acts[layer] - base_acts[layer]
            assert diff.shape[1] == num_positions and diff.shape[2] == activation_dim
            mean_diff = diff.mean(dim=0)
            base_mean = base_acts[layer].mean(dim=0)
            ft_mean = ft_acts[layer].mean(dim=0)
            out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
            self._save_means_for_layer(
                out_dir,
                position_labels,
                mean_diff,
                base_mean,
                ft_mean,
                num_sequences,
                activation_dim,
            )

        return {
            "dataset_id": dataset_id,
            "run_layers": run_layers,
            "position_labels": position_labels,
            "aps_tasks_for_dataset": aps_tasks_for_dataset,
        }

    def analysis(self, ctx: Dict[str, Any]) -> None:
        dataset_id: str = str(ctx["dataset_id"]) if ("dataset_id" in ctx) else str(ctx)
        run_layers: List[int] = (
            list(ctx["run_layers"]) if ("run_layers" in ctx) else self.layers
        )
        position_labels: List[int] = (
            list(ctx["position_labels"])
            if ("position_labels" in ctx)
            else list(range(int(self.cfg.diffing.method.n)))
        )
        aps_tasks_for_dataset: Dict[int, set] = dict(
            ctx.get("aps_tasks_for_dataset", {})
        )
        if len(aps_tasks_for_dataset) == 0:
            run_layers, aps_tasks_for_dataset = self._get_run_layers_and_aps_tasks(
                dataset_id
            )

        # Cache logit lens for each layer
        for layer in run_layers:
            out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
            out_dir.mkdir(parents=True, exist_ok=True)
            self._cache_logit_lens_for_layer(out_dir, position_labels)

        # Run auto patch scope for each layer
        for layer in run_layers:
            out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
            self._run_auto_patch_scope_for_layer(
                dataset_id, layer, out_dir, position_labels, aps_tasks_for_dataset
            )

    def visualize(self):
        visualize(self)

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available activation difference lens results.

        Args:
            results_dir: Base results directory

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = defaultdict(dict)

        if not results_dir.exists():
            return results

        # Scan for activation difference lens results in the expected structure
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for organism_dir in model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                act_diff_lens_dir = organism_dir / "activation_difference_lens"

                # Check if there are any layer directories with results
                if act_diff_lens_dir.exists() and any(
                    layer_dir.is_dir()
                    and layer_dir.name.startswith("layer_")
                    and any(layer_dir.iterdir())
                    for layer_dir in act_diff_lens_dir.iterdir()
                ):
                    results[model_name][organism_name] = str(act_diff_lens_dir)

        return results

    def get_agent(self) -> BaseAgent:
        return ADLAgent(cfg=self.cfg)

    def get_baseline_agent(self) -> BaseAgent:
        return ADLBlackboxAgent(cfg=self.cfg)
