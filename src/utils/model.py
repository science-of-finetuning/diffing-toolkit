from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Tuple, Dict, Any
import torch
from loguru import logger
from pathlib import Path
import inspect
from nnsight import NNsight

from .configs import ModelConfig

_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}


def has_thinking(cfg: ModelConfig) -> bool:
    return cfg.model.has_enable_thinking


def load_tokenizer(model_name: str) -> AutoTokenizer:
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    return AutoTokenizer.from_pretrained(model_name)


def load_steering_vector(steering_vector: str, layer: int) -> torch.Tensor:
    try:
        from huggingface_hub import hf_hub_download

        file_name = steering_vector.split("/")[-1]
        repo_name = steering_vector.split("/")[0]
        # Download steering vector from Hugging Face repository
        file_path = hf_hub_download(
            repo_id=f"science-of-finetuning/steering-vecs-{repo_name}",
            filename=f"{file_name}_L{layer}.pt",
            repo_type="model",
        )
        return torch.load(file_path, map_location="cpu")
    except Exception as e:
        logger.error(f"Error loading steering vector: {e}")
        raise e


def add_steering_vector(
    model: AutoModelForCausalLM, layer_idx: int, steering_vector: torch.Tensor
):
    # Get the current layer
    current_layer = model.model.layers[layer_idx].mlp.down_proj

    if hasattr(current_layer, "base_layer"):
        # PEFT wrapper
        is_peft = True
        current_layer = current_layer.base_layer
    else:
        is_peft = False

    # Create new linear layer with bias initialized to steering vector
    new_layer = torch.nn.Linear(
        in_features=current_layer.in_features,
        out_features=current_layer.out_features,
        bias=True,
    ).to(current_layer.weight.device, dtype=current_layer.weight.dtype)

    # Copy the original weights
    new_layer.weight.data = current_layer.weight.data.clone()

    # Initialize bias with steering vector
    assert steering_vector.shape == (
        current_layer.out_features,
    ), f"Steering vector shape {steering_vector.shape} doesn't match output features {current_layer.out_features}"
    new_layer.bias.data = steering_vector.to(
        current_layer.weight.device, dtype=current_layer.weight.dtype
    )

    # Replace the layer
    if is_peft:
        model.model.layers[layer_idx].mlp.down_proj.base_layer = new_layer
    else:
        model.model.layers[layer_idx].mlp.down_proj = new_layer

    logger.info(
        f"Bias initialized with steering vector of shape: {new_layer.bias.shape}"
    )
    return model


def load_model(
    model_name: str,
    dtype: torch.dtype,
    attn_implementation: str,
    adapter_id: str = None,
    steering_vector_name: str = None,
    steering_layer_idx: int = None,
    tokenizer_id: str = None,
    no_auto_device_map: bool = False,
    subfolder: str = None,
    device_map: Any | None = None,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    key = f"{model_name}_{dtype}_{attn_implementation}_{adapter_id}"
    if steering_vector_name is not None and steering_layer_idx is not None:
        key += f"_{steering_vector_name}_{steering_layer_idx}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], _TOKENIZER_CACHE[key]

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")

    if no_auto_device_map:
        # Overwrite device_map to None
        device_map = None

    fp_kwargs: Dict[str, Any] = dict(
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        subfolder=subfolder if adapter_id is None else "",
        trust_remote_code=trust_remote_code,
    )
    if device_map is not None:
        fp_kwargs["device_map"] = device_map
    elif not no_auto_device_map:
        fp_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **fp_kwargs,
    )
    if no_auto_device_map and device_map is None:
        model.to("cuda")

    if adapter_id:
        logger.info(f"Loading adapter: {adapter_id}")
        model.load_adapter(adapter_id, adapter_kwargs={"subfolder": subfolder})

    if tokenizer_id is not None:
        tokenizer = load_tokenizer(tokenizer_id)
    else:
        tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if steering_vector_name is not None and steering_layer_idx is not None:
        logger.info(f"Adding steering vector to layer {steering_layer_idx}")
        steering_vector = load_steering_vector(steering_vector_name, steering_layer_idx)
        model = add_steering_vector(model, steering_layer_idx, steering_vector)

    _MODEL_CACHE[key] = model
    _TOKENIZER_CACHE[key] = tokenizer

    return model, tokenizer


def get_ft_model_id(model_cfg: ModelConfig) -> str:
    if model_cfg.adapter_id:
        return model_cfg.adapter_id
    return model_cfg.model_id


def load_model_from_config(
    model_cfg: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    base_model_id = (
        model_cfg.base_model_id
        if model_cfg.base_model_id is not None
        else model_cfg.model_id
    )
    if base_model_id != model_cfg.model_id:
        adapter_id = model_cfg.model_id
    else:
        adapter_id = None
    return load_model(
        base_model_id,
        model_cfg.dtype,
        model_cfg.attn_implementation,
        adapter_id,
        model_cfg.steering_vector,
        model_cfg.steering_layer,
        model_cfg.tokenizer_id,
        no_auto_device_map=model_cfg.no_auto_device_map if model_cfg.no_auto_device_map is not None else False,
        subfolder=model_cfg.subfolder,
        device_map=model_cfg.device_map,
    )


# ============ Sharding / device placement helpers ============

def is_sharded(model: AutoModelForCausalLM) -> bool:
    if not hasattr(model, "hf_device_map"):
        return False
    device_map = getattr(model, "hf_device_map")
    if not isinstance(device_map, dict):
        return False
    # A model is sharded if it has multiple devices in its device map
    unique_devices = set(device_map.values())
    return len(unique_devices) > 1

def get_model_device(model: AutoModelForCausalLM) -> torch.device:
    if is_sharded(model):
        raise RuntimeError("Model is sharded; no single device. Use place_inputs or submodule-aware helpers.")
    try:
        return next(model.parameters()).device
    except StopIteration:
        raise RuntimeError("Model has no parameters to infer device.")


def place_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    model: AutoModelForCausalLM,
) -> Dict[str, torch.Tensor]:
    assert input_ids.ndim == 2
    if attention_mask is not None:
        assert attention_mask.shape == input_ids.shape

    # Sharded models accept CPU tensors and dispatch internally
    if is_sharded(model):
        return {
            "input_ids": input_ids.cpu(),
            "attention_mask": None if attention_mask is None else attention_mask.cpu(),
        }

    dev = get_model_device(model)
    return {
        "input_ids": input_ids.to(dev),
        "attention_mask": None if attention_mask is None else attention_mask.to(dev),
    }


def _resolve_same_device_for_submodules(model: AutoModelForCausalLM, submodule_names: list[str]) -> torch.device:
    devices: set[torch.device] = set()
    for name in submodule_names:
        sub = model
        for part in name.split("."):
            sub = getattr(sub, part)
        try:
            dev = next(sub.parameters()).device
        except StopIteration:
            # Try buffer
            for _, buf in sub.named_buffers(recurse=False):
                dev = buf.device
                break
            else:
                raise RuntimeError(f"Could not resolve device for submodule {name}")
        devices.add(dev)
    if len(devices) != 1:
        raise AssertionError(f"Submodules on different devices: {devices}")
    return next(iter(devices))


def load_tokenizer_from_config(
    model_cfg: ModelConfig,
) -> AutoTokenizer:
    if model_cfg.tokenizer_id is not None:
        return load_tokenizer(model_cfg.tokenizer_id)
    else:
        return load_tokenizer(model_cfg.model_id)


def logit_lens(
    latent: torch.Tensor, model: AutoModelForCausalLM
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analyze logits for a latent.

    Args:
        latent: Latent tensor
        model: Model to use for layer norm and lm_head

    Returns:
        Tuple of (positive_probs, negative_probs) - full probability distributions
    """
    # Place latent appropriately
    if is_sharded(model):
        # Ensure norm and lm_head are co-located
        dev = _resolve_same_device_for_submodules(model, ["model.norm", "lm_head"])
        latent = latent.to(device=dev).to(model.dtype)
    else:
        dev = get_model_device(model)
        latent = latent.to(device=dev).to(model.dtype)

    # Apply final layer norm and lm_head
    with torch.no_grad():
        # Apply layer norm
        normed_vector = model.model.norm(latent)  # [activation_dim]

        # Apply lm_head to get logits
        logits = model.lm_head(normed_vector)  # [vocab_size]
        inv_logits = model.lm_head(-normed_vector)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=0)  # [vocab_size]
        inv_probs = torch.softmax(inv_logits, dim=0)

        return probs.cpu(), inv_probs.cpu()


def patch_scope(
    latent: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,   
    layer: int,
    scaler: float = 1,
    id_prompt_target: str = "man -> man\n1135 -> 1135\nhello -> hello\n?"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analyze what tokens a latent vector promotes/suppresses using patch_scope method.

    Args:
        latent: Latent tensor to analyze
        model: Model to use for patching
        tokenizer: Tokenizer for input preparation
        layer: Layer index to patch at
        scaler: Scale factor for the latent vector
        id_prompt_target: Prompt template for patching

    Returns:
        Tuple of (positive_probs, negative_probs) - full probability distributions
    """
    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"]
    placed = place_inputs(id_prompt_tokens, None, model)
    id_prompt_tokens = placed["input_ids"]

    # Ensure latent is on correct device and dtype
    if is_sharded(model):
        dev = _resolve_same_device_for_submodules(model, ["model.norm", "lm_head"])
        latent = latent.to(device=dev).to(model.dtype)
    else:
        latent = latent.to(device=get_model_device(model)).to(model.dtype)
    
    nnmodel = NNsight(model)

    # Get positive direction probabilities
    with nnmodel.trace(id_prompt_tokens.repeat(1, 1), validate=False, scan=False):
        nnmodel.model.layers[layer].output[0][0, -1, :] = latent * scaler
        logits = nnmodel.lm_head.output[0, -1, :].save()

    probs = torch.softmax(logits, dim=0)  # [vocab_size]

    # Get negative direction probabilities
    with nnmodel.trace(id_prompt_tokens.repeat(1, 1), validate=False, scan=False):
        nnmodel.model.layers[layer].output[0][0, -1, :] = -latent * scaler
        inv_logits = nnmodel.lm_head.output[0, -1, :].save()

    inv_probs = torch.softmax(inv_logits, dim=0)

    return probs.cpu(), inv_probs.cpu()


def multi_patch_scope(
    latent: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    scaler: float = 1,
    id_prompt_targets: list[str] | None = None,
    top_k: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run patch_scope over multiple id prompts and return averaged probabilities
    restricted to the intersection of top-k tokens across all prompts.

    Args:
        latent: Latent tensor to analyze.
        model: Model to use for patching.
        tokenizer: Tokenizer for input preparation.
        layer: Layer index to patch at.
        scaler: Scale factor for the latent vector.
        id_prompt_targets: List of prompt templates to evaluate.
            Defaults to three prompts: ["man -> man\\n1135 -> 1135\\nhello -> hello\\n?", "bear -> bear\\n42 -> 42\\nblue -> blue\\n?", "921 -> 921\\ntarget -> target\\nanna -> anna\\n?"].
        top_k: Number of top tokens to consider for intersection.

    Returns:
        Tuple of (positive_probs, negative_probs) with only intersected token
        indices having non-zero probabilities (averaged across prompts).
    """
    assert isinstance(top_k, int) and top_k > 0
    if id_prompt_targets is None:
        id_prompt_targets = [
            "man -> man\n1135 -> 1135\nhello -> hello\n?",
            "bear -> bear\n42 -> 42\nblue -> blue\n?",
            "921 -> 921\ntarget -> target\nanna -> anna\n?",
        ]

    assert isinstance(id_prompt_targets, list) and len(id_prompt_targets) > 0
    for prompt in id_prompt_targets:
        assert isinstance(prompt, str) and len(prompt) > 0

    positive_prob_list: list[torch.Tensor] = []
    negative_prob_list: list[torch.Tensor] = []

    for prompt in id_prompt_targets:
        pos_probs, neg_probs = patch_scope(
            latent=latent,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            scaler=scaler,
            id_prompt_target=prompt,
        )
        positive_prob_list.append(pos_probs)
        negative_prob_list.append(neg_probs)

    # Shape and consistency checks
    vocab_size = positive_prob_list[0].shape[0]
    assert all(p.shape == (vocab_size,) for p in positive_prob_list)
    assert all(n.shape == (vocab_size,) for n in negative_prob_list)
    assert 0 < top_k <= vocab_size

    # Compute intersection of top-k indices across prompts for positive direction
    pos_topk_sets = []
    for probs in positive_prob_list:
        values, indices = torch.topk(probs, k=top_k)
        # Move to CPU and Python ints to ensure stable set behavior
        pos_topk_sets.append(set(int(i) for i in indices.cpu().tolist()))
    pos_intersection = set.intersection(*pos_topk_sets)


    # Fail fast if empty intersections
    assert len(pos_intersection) > 0, "Empty intersection for positive direction"

    # Compute intersection for negative direction
    neg_topk_sets = []
    for probs in negative_prob_list:
        values, indices = torch.topk(probs, k=top_k)
        neg_topk_sets.append(set(int(i) for i in indices.cpu().tolist()))
    neg_intersection = set.intersection(*neg_topk_sets)
    if len(neg_intersection) == 0:
        logger.warning("Empty intersection for negative direction")
    
    # Prepare output tensors: zeros everywhere, averaged probs on intersected tokens
    device = positive_prob_list[0].device
    dtype = positive_prob_list[0].dtype

    pos_out = torch.zeros(vocab_size, device=device, dtype=dtype)
    neg_out = torch.zeros(vocab_size, device=device, dtype=dtype)

    # Average across prompts for each selected token id
    num_prompts = float(len(id_prompt_targets))
    for token_id in pos_intersection:
        token_val = sum(p[token_id] for p in positive_prob_list) / num_prompts
        pos_out[token_id] = token_val

    for token_id in neg_intersection:
        token_val = sum(n[token_id] for n in negative_prob_list) / num_prompts
        neg_out[token_id] = token_val

    # Return on CPU to mirror patch_scope behavior
    return pos_out.cpu(), neg_out.cpu()


def batched_multi_patch_scope(
    latent: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    scales: list[float],
    id_prompt_targets: list[str] | None = None,
    top_k: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched variant of multi_patch_scope processing multiple scales at once.

    Returns two tensors of shape [num_scales, vocab_size] each, where only
    the intersection-of-top_k-across-prompts token indices are non-zero
    (values are averaged across prompts).
    """
    assert isinstance(latent, torch.Tensor) and latent.ndim == 1
    assert isinstance(layer, int) and layer >= 0
    assert isinstance(scales, list) and len(scales) > 0
    assert all(isinstance(s, (int, float)) for s in scales)
    assert isinstance(top_k, int) and top_k > 0

    if id_prompt_targets is None:
        id_prompt_targets = [
            "man -> man\n1135 -> 1135\nhello -> hello\n?",
            "bear -> bear\n42 -> 42\nblue -> blue\n?",
            "921 -> 921\ntarget -> target\nanna -> anna\n?",
        ]
    assert isinstance(id_prompt_targets, list) and len(id_prompt_targets) > 0
    for prompt in id_prompt_targets:
        assert isinstance(prompt, str) and len(prompt) > 0

    # Place latent on appropriate device and dtype
    if is_sharded(model):
        dev = _resolve_same_device_for_submodules(model, ["model.norm", "lm_head"])
        latent = latent.to(device=dev).to(model.dtype)
    else:
        latent = latent.to(device=get_model_device(model)).to(model.dtype)

    num_scales = int(len(scales))
    scales_tensor = torch.tensor(scales, device=latent.device, dtype=latent.dtype)
    assert scales_tensor.shape == (num_scales,)

    sum_pos_probs: torch.Tensor | None = None  # [num_scales, vocab]
    sum_neg_probs: torch.Tensor | None = None

    # Initialize intersections per scale and direction
    pos_intersections: list[set[int]] | None = None
    neg_intersections: list[set[int]] | None = None

    nnmodel = NNsight(model)

    for prompt in id_prompt_targets:
        # Tokenize and place
        id_prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
        placed = place_inputs(id_prompt_tokens, None, model)
        id_prompt_tokens = placed["input_ids"]
        assert id_prompt_tokens.ndim == 2 and id_prompt_tokens.shape[0] == 1

        # Repeat along batch dimension for each scale and both directions (+/-)
        batch_size = num_scales * 2
        input_batch = id_prompt_tokens.repeat(batch_size, 1)
        assert input_batch.ndim == 2 and input_batch.shape[0] == batch_size

        pos_logits_saved: list[torch.Tensor] = []
        neg_logits_saved: list[torch.Tensor] = []

        latent_stack = torch.stack([latent * scales_tensor[b] for b in range(num_scales)] + [-latent * scales_tensor[b] for b in range(num_scales)])
        assert latent_stack.ndim == 2 and latent_stack.shape == (batch_size, latent.shape[0])


        with nnmodel.trace(input_batch, validate=False, scan=False):
            nnmodel.model.layers[layer].output[0][:, -1, :] = latent_stack
            logits = nnmodel.lm_head.output[:, -1, :].save()

        assert logits.ndim == 2 and logits.shape[0] == batch_size
        pos_logits = logits[:num_scales]
        neg_logits = logits[num_scales:]
        assert pos_logits.ndim == 2 and pos_logits.shape[0] == num_scales
        assert neg_logits.ndim == 2 and neg_logits.shape[0] == num_scales
        pos_probs = torch.softmax(pos_logits, dim=-1)
        neg_probs = torch.softmax(neg_logits, dim=-1)

        # Initialize running sums and intersections
        if sum_pos_probs is None:
            sum_pos_probs = torch.zeros_like(pos_probs)
            sum_neg_probs = torch.zeros_like(neg_probs)
        sum_pos_probs += pos_probs
        sum_neg_probs += neg_probs

        # Compute per-scale top-k sets for this prompt
        vocab_size = int(pos_probs.shape[1])
        assert 0 < top_k <= vocab_size

        current_pos_sets: list[set[int]] = []
        current_neg_sets: list[set[int]] = []
        for b in range(num_scales):
            _, pos_idx = torch.topk(pos_probs[b], k=top_k)
            _, neg_idx = torch.topk(neg_probs[b], k=top_k)
            current_pos_sets.append(set(int(i) for i in pos_idx.cpu().tolist()))
            current_neg_sets.append(set(int(i) for i in neg_idx.cpu().tolist()))

        if pos_intersections is None:
            pos_intersections = current_pos_sets
            neg_intersections = current_neg_sets
        else:
            assert pos_intersections is not None and neg_intersections is not None
            for b in range(num_scales):
                pos_intersections[b] = pos_intersections[b].intersection(current_pos_sets[b])
                neg_intersections[b] = neg_intersections[b].intersection(current_neg_sets[b])

    assert sum_pos_probs is not None and sum_neg_probs is not None
    assert pos_intersections is not None and neg_intersections is not None

    # Average across prompts and restrict to intersections
    num_prompts = float(len(id_prompt_targets))
    avg_pos = sum_pos_probs / num_prompts
    avg_neg = sum_neg_probs / num_prompts
    assert avg_pos.shape == avg_neg.shape == (num_scales, vocab_size)

    out_pos = torch.zeros_like(avg_pos)
    out_neg = torch.zeros_like(avg_neg)

    for b in range(num_scales):
        assert len(pos_intersections[b]) > 0, "Empty intersection for positive direction"
        # Empty intersection for negative direction are fine
        for idx in pos_intersections[b]:
            out_pos[b, int(idx)] = avg_pos[b, int(idx)]
        for idx in neg_intersections[b]:
            out_neg[b, int(idx)] = avg_neg[b, int(idx)]

    return out_pos.cpu(), out_neg.cpu()
