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

from src.diffing.methods.diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices
from src.utils.model import logit_lens
from .ui import visualize   
from .steering import run_steering

def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: Any,
    split: str = "validation", 
    text_column: str = "text",
    n: int = 10,
    max_samples: int = 1000,
    debug: bool = False
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
        List of lists, where each inner list contains up to n first token IDs
    """
    logger.info(f"Loading dataset {dataset_name} (split: {split})")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    
    if debug:
        max_samples = min(20, max_samples)
    
    logger.info(f"Dataset loaded with {len(dataset)} samples, processing up to {max_samples}")
    
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
        text_truncated = text[:n*10]
        
        # Tokenize
        tokens = tokenizer.encode(text_truncated, add_special_tokens=True)
        
        # Extract up to first n tokens
        num_tokens = min(n, len(tokens))
        if num_tokens > 0:
            first_tokens = tokens[:num_tokens]
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
            logger.warning(f"First message is not user: {messages[0]}")
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
    sequences: List[torch.Tensor]
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
    batch_size: int = 8
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
        batch_sequences = first_n_tokens[i:i + batch_size]
        # Pad sequences to length n and create batch tensor
        
        batch_input_ids = []
        for seq in batch_sequences:
            if len(seq) < n:
                continue
            batch_input_ids.append(seq)
        
        batch_input_ids = torch.tensor(batch_input_ids).to(model.device)  # [batch_size, n]
        assert batch_input_ids.shape == (len(batch_sequences), n)
        
        # Extract activations using nnsight for all layers
        with nn_model.trace(batch_input_ids):
            layer_outputs = {}
            for layer in layers:
                layer_outputs[layer] = nn_model.model.layers[layer].output[0].save()
        
        # Store activations for each layer
        for layer in layers:
            activations = layer_outputs[layer].cpu()  # [batch_size, n, hidden_dim]
            assert activations.shape[1] == n
            all_activations[layer].append(activations)
    
    # Concatenate all batches for each layer
    result = {}
    for layer in layers:
        result[layer] = torch.cat(all_activations[layer], dim=0)  # [num_sequences, n, hidden_dim]
        assert result[layer].shape[0] == len(first_n_tokens)
        assert result[layer].shape[1] == n
    
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
            (len(batch), max_len), fill_value=pad_token_id, dtype=torch.long, device=model.device
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=model.device)

        for row, seq in enumerate(batch_input_ids_list):
            seq_len = len(seq)
            batch_input_ids[row, :seq_len] = torch.tensor(seq, dtype=torch.long, device=model.device)
            attention_mask[row, :seq_len] = 1

        with nn_model.trace(batch_input_ids, attention_mask=attention_mask):
            layer_outputs: Dict[int, torch.Tensor] = {}
            for layer in layers:
                layer_outputs[layer] = nn_model.model.layers[layer].output[0].save()

        for layer in layers:
            hidden: torch.Tensor = layer_outputs[layer]  # [B, L, D]
            pos_index = torch.tensor(batch_positions_list, dtype=torch.long, device=hidden.device)  # [B, P]
            pos_index_expanded = pos_index.unsqueeze(-1).expand(-1, -1, hidden.shape[2])
            assert pos_index_expanded.shape == (len(batch), num_positions, hidden.shape[2])
            gathered = torch.gather(hidden, dim=1, index=pos_index_expanded)  # [B, P, D]
            assert gathered.shape == (len(batch), num_positions, hidden.shape[2])
            all_activations[layer].append(gathered.to("cpu"))

    result: Dict[int, torch.Tensor] = {}
    for layer in layers:
        result[layer] = torch.cat(all_activations[layer], dim=0)
    return result


class ActDiffLens(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.results_dir = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.layers = get_layer_indices(self.base_model_cfg.model_id, self.cfg.diffing.method.layers)

    def run(self):
        for dataset_entry in self.cfg.diffing.method.datasets:
            assert isinstance(dataset_entry, (dict, DictConfig)) and "id" in dataset_entry and "is_chat" in dataset_entry
            dataset_id = dataset_entry["id"]
            is_chat: bool = dataset_entry["is_chat"]

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
                    layers=self.layers,
                    batch_size=self.cfg.diffing.method.batch_size,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                )

                # Clear base model from memory
                self.clear_base_model()

                ft_acts = extract_selected_positions_activations(
                    model=self.finetuned_model,
                    samples=samples,
                    layers=self.layers,
                    batch_size=self.cfg.diffing.method.batch_size,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                )

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
                    self.layers,
                    self.cfg.diffing.method.batch_size,
                )

                # Clear base model from memory
                self.clear_base_model()

                ft_acts = extract_first_n_tokens_activations(
                    self.finetuned_model,
                    first_n_tokens,
                    self.layers,
                    self.cfg.diffing.method.batch_size,
                )

                position_labels = list(range(self.cfg.diffing.method.n))
                num_positions = len(position_labels)

            # Compute activation differences for each layer
            diff_activations_per_layer: Dict[int, torch.Tensor] = {}
            for layer in self.layers:
                diff_activations_per_layer[layer] = ft_acts[layer] - base_acts[layer]
                assert diff_activations_per_layer[layer].shape == ft_acts[layer].shape
                assert diff_activations_per_layer[layer].shape[0] > 0
                assert diff_activations_per_layer[layer].shape[1] == num_positions
                assert diff_activations_per_layer[layer].shape[2] > 0

            # Get metadata for dashboard saving
            num_sequences = diff_activations_per_layer[self.layers[0]].shape[0]
            activation_dim = diff_activations_per_layer[self.layers[0]].shape[2]

            # Compute norm estimates (simple)
            base_model_norms: Dict[int, torch.Tensor] = {}
            ft_model_norms: Dict[int, torch.Tensor] = {}
            skip_tokens = 5
            
            for layer in self.layers:
                base_layer_acts = base_acts[layer]  # [num_sequences, P, hidden_dim]
                ft_layer_acts = ft_acts[layer]      # [num_sequences, P, hidden_dim]
                assert base_layer_acts.shape == ft_layer_acts.shape
                assert base_layer_acts.shape[1] >= skip_tokens, f"Need at least {skip_tokens} positions, got {base_layer_acts.shape[1]}"
                
                base_acts_truncated = base_layer_acts[:, skip_tokens:, :]
                ft_acts_truncated = ft_layer_acts[:, skip_tokens:, :]
                
                base_norms_per_pos = torch.norm(base_acts_truncated, dim=2)
                ft_norms_per_pos = torch.norm(ft_acts_truncated, dim=2)
                
                assert base_norms_per_pos.shape == (base_layer_acts.shape[0], base_layer_acts.shape[1] - skip_tokens)
                assert ft_norms_per_pos.shape == (ft_layer_acts.shape[0], ft_layer_acts.shape[1] - skip_tokens)
                
                base_model_norms[layer] = base_norms_per_pos.flatten().mean()  # scalar
                ft_model_norms[layer] = ft_norms_per_pos.flatten().mean()      # scalar
                
                logger.info(f"Layer {layer} - Base model mean norm: {base_model_norms[layer].item():.3f}")
                logger.info(f"Layer {layer} - Fine-tuned model mean norm: {ft_model_norms[layer].item():.3f}")

            # Save norm estimates to file
            norms_data = {
                'base_model_norms': {layer: base_model_norms[layer].cpu() for layer in self.layers},
                'ft_model_norms': {layer: ft_model_norms[layer].cpu() for layer in self.layers},
                'skip_tokens': skip_tokens,
                'num_sequences': num_sequences
            }
            norms_path = self.results_dir / f"model_norms_{dataset_id.split('/')[-1]}.pt"
            torch.save(norms_data, norms_path)
            logger.info(f"Saved model norm estimates to {norms_path}")
            
            # Compute mean activation difference per position per layer
            mean_diff_per_position_per_layer: Dict[int, torch.Tensor] = {}
            for layer in self.layers:
                mean_diff_per_position_per_layer[layer] = diff_activations_per_layer[layer].mean(dim=0)  # [P, hidden_dim]
                assert mean_diff_per_position_per_layer[layer].shape == (num_positions, diff_activations_per_layer[layer].shape[2])
                            
                # Construct dashboard path with layer info
                out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
                out_dir.mkdir(parents=True, exist_ok=True)
                
                mean_activation_per_position = mean_diff_per_position_per_layer[layer]
                # Save each position with 0-index labels (and negatives for chat)
                for idx_in_tensor, label in enumerate(position_labels):
                    tensor_filename = f"mean_pos_{idx_in_tensor}.pt"
                    tensor_path = out_dir / tensor_filename
                    torch.save(mean_activation_per_position[idx_in_tensor], tensor_path)
                    
                    # Save metadata
                    meta_data = {
                        'count': num_sequences,
                        'activation_dim': activation_dim,
                        'position': label,
                        'token_id': None
                    }
                    
                    meta_filename = f"mean_pos_{idx_in_tensor}.meta"
                    meta_path = out_dir / meta_filename
                    with open(meta_path, 'w') as f:
                        json.dump(meta_data, f, indent=2)

                    if self.cfg.diffing.method.logit_lens.cache:
                        probs, inv_probs = logit_lens(mean_activation_per_position[idx_in_tensor], self.finetuned_model)
                        # Get top k indices and probabilities
                        k = self.cfg.diffing.method.logit_lens.k
                        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                        top_k_inv_probs, top_k_inv_indices = torch.topk(inv_probs, k, dim=-1)
                        
                        logit_lens_per_position_path = out_dir / f"logit_lens_pos_{idx_in_tensor}.pt"
                        torch.save((top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices), logit_lens_per_position_path)
                        logger.info(f"Cached top-{k} logit lens for position {label}")

        # Run steering if enabled
        steering_cfg = getattr(self.cfg.diffing.method, "steering", None)
        if steering_cfg is not None and getattr(steering_cfg, "enabled", False):
            run_steering(self)


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