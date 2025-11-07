from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from loguru import logger
import gc
import torch.nn as nn
import torch as th
from nnsight.intervention.envoy import Envoy
from nnterp import StandardizedTransformer
from nnterp.interventions import (
    repeat_prompt,
    TargetPrompt,
)
from nnterp.interventions import patchscope_lens as nnterp_patchscope_lens

from .configs import ModelConfig

AnyTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

_MODEL_CACHE: dict[str, StandardizedTransformer] = {}
_TOKENIZER_CACHE: dict[str, AnyTokenizer] = {}


def gc_collect_cuda_cache():
    gc.collect()
    th.cuda.empty_cache()
    th.cuda.synchronize()


def clear_cache():
    _MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()
    gc_collect_cuda_cache()


def has_thinking(cfg: ModelConfig) -> bool:
    return cfg.model.has_enable_thinking


def load_tokenizer(model_name: str) -> AnyTokenizer:
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def load_steering_vector(steering_vector: str, layer: int) -> th.Tensor:
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
        return th.load(file_path, map_location="cpu")
    except Exception as e:
        logger.error(f"Error loading steering vector: {e}")
        raise e


# ============ Model loading helpers ============


def add_steering_vector_legacy(
    model: StandardizedTransformer, layer_idx: int, steering_vector: th.Tensor
):
    # Get the current layer
    current_layer = model.mlps[layer_idx].down_proj._module

    if hasattr(current_layer, "base_layer"):
        # PEFT wrapper
        is_peft = True
        current_layer = current_layer.base_layer
    else:
        is_peft = False

    # Create new linear layer with bias initialized to steering vector
    new_layer = th.nn.Linear(
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
        model.mlps[layer_idx].down_proj._module.base_layer = new_layer
    else:
        model.mlps[layer_idx].down_proj._module = new_layer

    logger.info(
        f"Bias initialized with steering vector of shape: {new_layer.bias.shape}"
    )
    return model


def add_steering_vector(
    model: StandardizedTransformer, layer_idx: int, steering_vector: th.Tensor
) -> StandardizedTransformer:
    with model.edit() as edited_model:
        model.layers_output[layer_idx] += steering_vector
    return edited_model


def load_model(
    model_name: str,
    dtype: th.dtype,
    attn_implementation: str,
    adapter_id: str = None,
    steering_vector_name: str = None,
    steering_layer_idx: int = None,
    tokenizer_id: str = None,
    no_auto_device_map: bool = False,
    subfolder: str = None,
    device_map: Any | None = None,
    trust_remote_code: bool = False,
) -> StandardizedTransformer:
    model_key = f"{model_name}_{dtype}_{attn_implementation}_{adapter_id}"
    if steering_vector_name is not None and steering_layer_idx is not None:
        key = model_key + f"_{steering_vector_name}_{steering_layer_idx}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    elif model_key in _MODEL_CACHE:
        model = _MODEL_CACHE[model_key]
    else:
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
        automodel = AutoModelForCausalLM
        if "Qwen2.5-VL" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration

            automodel = Qwen2_5_VLForConditionalGeneration
        if tokenizer_id is not None:
            tokenizer = load_tokenizer(tokenizer_id)
        else:
            tokenizer = load_tokenizer(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = StandardizedTransformer(
            model_name,
            automodel=automodel,
            tokenizer=tokenizer,
            **fp_kwargs,
        )

        if no_auto_device_map and device_map is None:
            model.to("cuda")

        if adapter_id:
            logger.info(f"Loading adapter: {adapter_id}")
            model.load_adapter(adapter_id, adapter_kwargs={"subfolder": subfolder})

    if steering_vector_name is not None and steering_layer_idx is not None:
        logger.info(f"Adding steering vector to layer {steering_layer_idx}")
        steering_vector = load_steering_vector(steering_vector_name, steering_layer_idx)
        model = add_steering_vector(model, steering_layer_idx, steering_vector)
    _MODEL_CACHE[key] = model
    return model


def get_ft_model_id(model_cfg: ModelConfig) -> str:
    if model_cfg.adapter_id:
        return model_cfg.adapter_id
    return model_cfg.model_id


def load_model_from_config(
    model_cfg: ModelConfig,
) -> StandardizedTransformer:
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
        no_auto_device_map=(
            model_cfg.no_auto_device_map
            if model_cfg.no_auto_device_map is not None
            else False
        ),
        subfolder=model_cfg.subfolder,
        device_map=model_cfg.device_map,
    )


def load_tokenizer_from_config(
    model_cfg: ModelConfig,
) -> AnyTokenizer:
    if model_cfg.tokenizer_id is not None:
        return load_tokenizer(model_cfg.tokenizer_id)
    else:
        return load_tokenizer(model_cfg.model_id)


# ============ Sharding / device placement helpers ============


def is_sharded(model: StandardizedTransformer) -> bool:
    if not hasattr(model, "hf_device_map"):
        return False
    device_map = getattr(model, "hf_device_map")
    if not isinstance(device_map, dict):
        return False
    # A model is sharded if it has multiple devices in its device map
    unique_devices = set(device_map.values())
    return len(unique_devices) > 1


def get_model_device(model: StandardizedTransformer) -> th.device:
    if is_sharded(model):
        raise RuntimeError(
            "Model is sharded; no single device. Use place_inputs or submodule-aware helpers."
        )
    try:
        return next(model.parameters()).device
    except StopIteration:
        raise RuntimeError("Model has no parameters to infer device.")


def place_inputs(
    input_ids: th.Tensor,
    attention_mask: th.Tensor | None,
    model: StandardizedTransformer,
) -> Dict[str, th.Tensor]:
    assert input_ids.ndim == 2
    if attention_mask is not None:
        assert attention_mask.shape == input_ids.shape
    # I expect this to work with nnsight. TODO: clean
    return dict(input_ids=input_ids, attention_mask=attention_mask)
    # Sharded models accept CPU tensors and dispatch internally
    # if is_sharded(model):
    #     return {
    #         "input_ids": input_ids.cpu(),
    #         "attention_mask": None if attention_mask is None else attention_mask.cpu(),
    #     }

    # dev = get_model_device(model)
    # return {
    #     "input_ids": input_ids.to(dev),
    #     "attention_mask": None if attention_mask is None else attention_mask.to(dev),
    # }


def get_modules_device(*modules: Envoy) -> list[th.device]:

    devices: list[th.device] = []
    for module in modules:
        try:
            dev = next(module._module.parameters()).device
        except StopIteration:
            # Try buffer
            for _, buf in module._module.named_buffers(recurse=False):
                dev = buf.device
                break
            else:
                raise RuntimeError(f"Could not resolve device for module {module}")
        devices.append(dev)
    return devices


# ============ Diffing helpers ============


@th.no_grad()
def logit_lens(
    latent: th.Tensor, model: StandardizedTransformer
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Project a latent vector onto the model's vocabulary space and return the probabilities.

    Args:
        latent: Tensor to project onto the model's vocabulary space, of shape [..., hidden_size]
        model: Model to use for projection

    Returns:
        Tuple of (positive_probs, negative_probs), where negative_probs is obtained by projecting -norm(latent) instead of norm(latent). Shape is [..., vocab_size]
    """
    if latent.shape[-1] != model.hidden_size:
        raise ValueError(
            f"Latent shape {latent.shape} does not match model hidden size {model.hidden_size}"
        )
    model.dispatch()

    ln_device, lm_head_device = get_modules_device(model.ln_final, model.lm_head)
    normed_vector = model.ln_final(latent.to(device=ln_device, dtype=model.dtype)).to(
        lm_head_device
    )
    probs = model.lm_head(normed_vector).softmax(dim=-1)
    inv_probs = model.lm_head(-normed_vector).softmax(dim=-1)
    assert probs.shape == (
        *latent.shape[:-1],
        model.vocab_size,
    ), f"Probs shape {probs.shape} does not match expected shape {(*latent.shape[:-1], model.vocab_size)}"

    return probs.cpu(), inv_probs.cpu()


def default_id_prompt_targets() -> list[TargetPrompt]:
    default_rel = " -> "
    # TODO: improve with variations on the linking char and separator?
    return [
        repeat_prompt(
            words=["man", "1135", "hello"],
            rel=default_rel,
        ),
        repeat_prompt(
            words=["bear", "42", "blue"],
            rel=default_rel,
        ),
        repeat_prompt(
            words=["921", "target", "anna"],
            rel=default_rel,
        ),
    ]


@th.no_grad()
def patchscope_lens(
    latent: th.Tensor,
    model: StandardizedTransformer,
    layer: int,
    scales: list[float] | float = 1,
    id_prompt_targets: (
        list[str] | str | None
    ) = "man -> man\n1135 -> 1135\nhello -> hello\n?",
    top_k: int = 100,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Run a patchscope on identity prompts to analyze latent and -latent vectors scaled by various factors. Return a tuple of the next token probabilities for all s*latent and s*-latent vectors, restricted to the intersection of top-k tokens across all prompts.

    Args:
        latent: Latent tensor to analyze.
        model: Model to use for patching.
        layer: Layer index to patch at.
        scales: List of scales to use for the latent vector.
        id_prompt_targets: List of identity prompts to use for patching.

    Returns:
        Tuple of (positive_probs, negative_probs) with the average next token probabilities for all s*latent and s*-latent vectors, averaged across all identity prompts. Shape are both [num_scales, vocab_size] if multiple scales are provided, otherwise [vocab_size].
    """
    if isinstance(scales, float):
        scales = [scales]
    assert (
        isinstance(top_k, int) and 0 < top_k <= model.vocab_size
    ), f"Top-k must be a positive integer less than or equal to the vocabulary size, got {top_k} for vocabulary size {model.vocab_size}"
    num_scalers = len(scales)
    scales = th.tensor(scales, device=latent.device, dtype=latent.dtype)
    scales = th.cat([scales, -scales], dim=0)  # shape: [2 * num_scalers]
    assert scales.ndim == 1, f"Scalers must be a list of floats, got {scales.shape}"

    if isinstance(id_prompt_targets, str):
        id_prompt_targets = [TargetPrompt(id_prompt_targets, -1)]
    if id_prompt_targets is None:
        id_prompt_targets = default_id_prompt_targets()
    else:  # assume list of prompts
        id_prompt_targets = [TargetPrompt(prompt, -1) for prompt in id_prompt_targets]
    num_prompts = len(id_prompt_targets)

    assert latent.shape == (model.hidden_size,)
    latents = latent.unsqueeze(0) * scales.unsqueeze(
        1
    )  # shape: [2* num_scalers, hidden_size]
    latents = latents.unsqueeze(0)  # shape: [1, 2* num_scalers, hidden_size]
    assert latents.shape == (1, 2 * num_scalers, model.hidden_size)

    if num_prompts > 1:
        is_in_topk_count = th.zeros(2 * num_scalers, model.vocab_size, dtype=th.int32)
    cum_probs = th.zeros(2 * num_scalers, model.vocab_size, dtype=th.float32)
    for prompt in id_prompt_targets:
        probs = nnterp_patchscope_lens(
            model,
            target_patch_prompt=prompt,
            layers=layer,
            latents=latents,
            remote=False,
        )  # Returns of shape [{num latent vectors}, {num_layers, here 1}, {vocab_size}]
        assert probs.shape == (
            2 * num_scalers,
            1,
            model.vocab_size,
        ), f"Probs shape {probs.shape} does not match expected shape ([{2 * num_scalers}, 1, {model.vocab_size})"
        if num_prompts > 1:
            topk_values, topk_indices = th.topk(
                probs.squeeze(1), k=top_k, dim=-1
            )  # shape: [2 * num_scalers, top_k]
            is_in_topk_count[th.arange(2 * num_scalers), topk_indices] += 1
        cum_probs[th.arange(2 * num_scalers), topk_indices] += topk_values
    assert (
        cum_probs.shape == is_in_topk_count.shape == (2 * num_scalers, model.vocab_size)
    )
    if num_prompts > 1:
        is_not_in_topk_intersection = is_in_topk_count != num_prompts
        cum_probs[is_not_in_topk_intersection] = 0
    pos_probs = cum_probs[:num_scalers, :].cpu().squeeze(0) / num_prompts
    neg_probs = cum_probs[num_scalers:, :].cpu().squeeze(0) / num_prompts
    return pos_probs, neg_probs


class ModuleAccessor(ABC):
    """
    Abstract base class for object representing accessing a module from a model. Useful when parameterizing an intervention.

    Args:
        layer: The layer to access. If None, the module will be accessed from the entire model.
    """

    def __init__(self, layer: int | None):
        self.layer = layer

    @abstractmethod
    def get_nnsight_module(self, model: StandardizedTransformer) -> Envoy:
        pass

    def get_module(self, model: StandardizedTransformer) -> nn.Module:
        return self.get_nnsight_module(model)._module


class MLPModuleAccessor(ModuleAccessor):
    """
    Object representing accessing a MLP module from a model.
    """

    def get_nnsight_module(self, model: StandardizedTransformer) -> Envoy:
        return model.mlps[self.layer]


class AttentionModuleAccessor(ModuleAccessor):
    """
    Object representing accessing an attention module from a model.
    """

    def get_nnsight_module(self, model: StandardizedTransformer) -> Envoy:
        return model.attentions[self.layer]
