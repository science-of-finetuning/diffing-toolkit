"""Test nnterp backend with LoRA adapter + steering vector support.

Loads gemma-3-4b-it with a sarcasm LoRA adapter and tests:
1. Base model generation (no adapter, no steering)
2. LoRA adapter generation with amplification
3. LoRA + steering vector generation (if steering vectors available)
"""

import torch as th
from loguru import logger

from nnterp import StandardizedTransformer
from diffing.methods.amplification.amplification_config import (
    AmplificationConfig,
    AmplifiedAdapter,
    LayerAmplification,
    ModuleAmplification,
    SteeringVectorAmplification,
    apply_peft_amplification,
    restore_peft_weights,
    load_peft_adapter,
    _build_peft_layer_patterns,
    _get_layer_and_module_type,
)


MODEL_ID = "google/gemma-3-4b-it"
ADAPTER_ID = "maius/gemma-3-4b-it-personas/sarcasm"
ADAPTER_NAME = ADAPTER_ID.replace(".", "_").replace("/", "_")
BASE_MODEL_NAME = "gemma3_4B_it"

PROMPT = "What is the meaning of life?"


def load_model():
    """Load the base model."""
    logger.info(f"Loading model {MODEL_ID}")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(MODEL_ID)
    model = StandardizedTransformer(
        MODEL_ID,
        torch_dtype=th.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    return model, tokenizer


def generate_text(model, tokenizer, prompt_text, max_tokens=50, steering_specs=None):
    """Generate text with optional steering vectors."""
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

    with model.generate(
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    ) as tracer:
        with tracer.invoke(input_ids):
            if steering_specs:
                with tracer.all():
                    for sv_config, layers, vector in steering_specs:
                        sv_config.apply(model, vector, layers)

        with tracer.invoke():
            output = model.generator.output.save()

    generated_ids = output[0].tolist()
    new_token_ids = generated_ids[len(input_ids[0]):]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


def test_pattern_matching(model):
    """Test that PEFT layer pattern matching works correctly."""
    logger.info("Testing pattern matching...")
    attn_pattern, mlp_pattern = _build_peft_layer_patterns(model)
    logger.info(f"  Attention pattern: {attn_pattern.pattern}")
    logger.info(f"  MLP pattern: {mlp_pattern.pattern}")

    # Check patterns against actual parameter names
    lora_params = [n for n, _ in model._model.named_parameters() if "lora_B" in n]
    assert len(lora_params) > 0, "No lora_B parameters found — adapter not loaded?"

    matched = 0
    for name in lora_params:
        layer_idx, mod_type = _get_layer_and_module_type(name, attn_pattern, mlp_pattern)
        if layer_idx is not None:
            matched += 1
            logger.debug(f"  {name} -> layer={layer_idx}, type={mod_type}")

    logger.info(f"  Matched {matched}/{len(lora_params)} lora_B parameters")
    assert matched > 0, "No lora_B parameters matched attention or MLP patterns"
    logger.info("Pattern matching OK")


def test_base_generation(model, tokenizer):
    """Test generation without any adapter."""
    logger.info("--- Test: Base model generation ---")
    # Disable adapters if any are active
    if hasattr(model._model, "peft_config") and model._model.peft_config:
        model._model.disable_adapters()
    text = generate_text(model, tokenizer, PROMPT)
    logger.info(f"  Output: {text[:200]}")
    assert len(text) > 0, "Empty generation"
    logger.info("Base generation OK")
    return text


def test_lora_generation(model, tokenizer):
    """Test generation with LoRA adapter (no amplification)."""
    logger.info("--- Test: LoRA adapter generation (weight=1.0) ---")
    if not hasattr(model._model, "peft_config") or ADAPTER_NAME not in model._model.peft_config:
        logger.info(f"  Loading adapter {ADAPTER_ID}")
        load_peft_adapter(model, ADAPTER_ID, ADAPTER_NAME)

    model.set_adapter(ADAPTER_NAME)
    model._model.enable_adapters()
    text = generate_text(model, tokenizer, PROMPT)
    logger.info(f"  Output: {text[:200]}")
    assert len(text) > 0, "Empty generation"
    logger.info("LoRA generation OK")
    return text


def test_lora_amplified_generation(model, tokenizer):
    """Test generation with LoRA adapter + amplification (weight=2.0 on all layers)."""
    logger.info("--- Test: LoRA amplified generation (weight=2.0) ---")
    if not hasattr(model._model, "peft_config") or ADAPTER_NAME not in model._model.peft_config:
        load_peft_adapter(model, ADAPTER_ID, ADAPTER_NAME)

    model.set_adapter(ADAPTER_NAME)
    model._model.enable_adapters()

    # Build per-layer amplification: weight=2.0 on all modules for all layers
    num_layers = model.num_layers
    layer_amplifications = [{"attention": 2.0, "mlp": 2.0}] * num_layers

    # Apply amplification
    saved_weights = apply_peft_amplification(model, ADAPTER_NAME, layer_amplifications)
    assert len(saved_weights) > 0, "No weights were amplified"

    text = generate_text(model, tokenizer, PROMPT)
    logger.info(f"  Output: {text[:200]}")
    assert len(text) > 0, "Empty generation"

    # Restore weights
    restore_peft_weights(model, saved_weights)

    # Verify restoration: check weights match originals
    lora_b_params = {n: p.data.clone() for n, p in model._model.named_parameters() if "lora_B" in n and ADAPTER_NAME in n}
    saved_again = apply_peft_amplification(model, ADAPTER_NAME, layer_amplifications)
    restore_peft_weights(model, saved_again)
    for name, restored_val in lora_b_params.items():
        current_val = dict(model._model.named_parameters())[name].data
        assert th.allclose(restored_val, current_val), f"Weight restoration failed for {name}"

    logger.info("LoRA amplified generation OK")
    return text


def test_full_config_flow(model, tokenizer):
    """Test using AmplificationConfig end-to-end, mimicking the dashboard flow."""
    logger.info("--- Test: Full AmplificationConfig flow ---")

    config = AmplificationConfig(
        name="test_amplified",
        amplified_adapters=[
            AmplifiedAdapter(
                organism_name="persona_sarcasm",
                variant="default",
                layer_amplifications=[
                    LayerAmplification(
                        layers="all",
                        module_amplifications=[ModuleAmplification(modules="all", weight=1.5)],
                    )
                ],
            )
        ],
    )

    # Resolve adapters
    resolved = AmplifiedAdapter.resolve_list(
        config.amplified_adapters, model, BASE_MODEL_NAME
    )
    assert len(resolved) == 1, f"Expected 1 resolved adapter, got {len(resolved)}"
    resolved_adapter_id = list(resolved.keys())[0]
    logger.info(f"  Resolved adapter: {resolved_adapter_id}")
    assert resolved_adapter_id == ADAPTER_ID

    layer_amplifications = resolved[resolved_adapter_id]
    logger.info(f"  Layer amplifications: {len(layer_amplifications)} layers")
    assert layer_amplifications[0] == {"attention": 1.5, "mlp": 1.5}

    # Load, activate, amplify
    if not hasattr(model._model, "peft_config") or ADAPTER_NAME not in model._model.peft_config:
        load_peft_adapter(model, ADAPTER_ID, ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)
    model._model.enable_adapters()

    saved_weights = apply_peft_amplification(model, ADAPTER_NAME, layer_amplifications)

    text = generate_text(model, tokenizer, PROMPT)
    logger.info(f"  Output: {text[:200]}")
    assert len(text) > 0, "Empty generation"

    restore_peft_weights(model, saved_weights)
    logger.info("Full config flow OK")
    return text


if __name__ == "__main__":
    model, tokenizer = load_model()

    base_text = test_base_generation(model, tokenizer)

    # Load adapter for remaining tests
    logger.info(f"Loading adapter {ADAPTER_ID}")
    load_peft_adapter(model, ADAPTER_ID, ADAPTER_NAME)

    test_pattern_matching(model)
    lora_text = test_lora_generation(model, tokenizer)
    amplified_text = test_lora_amplified_generation(model, tokenizer)
    config_text = test_full_config_flow(model, tokenizer)

    logger.info("\n=== All tests passed! ===")
    logger.info(f"Base:      {base_text[:100]}")
    logger.info(f"LoRA:      {lora_text[:100]}")
    logger.info(f"Amplified: {amplified_text[:100]}")
    logger.info(f"Config:    {config_text[:100]}")
