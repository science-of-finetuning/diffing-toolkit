"""
Steering vector demo based on logit diff token weights.

This script demonstrates how to construct steering vectors from logit diff
analysis results. The key insight is that tokens have weights (from NMF topics
or occurrence rates) and unembedding vectors, so we can compute:

    steering_vector = sum(weight_i * unembed_i) for i in selected_tokens

The unembedding vectors are rows of the model's lm_head weight matrix (or its
transpose, depending on the implementation).

Usage:
    uv run scripts/steering_from_logit_diff.py \
        --results-dir /path/to/run_xxx/nmf/fineweb-1m-sample_train_text \
        --ordering-id topic_0 \
        --top-k 50 \
        --steering-strength 1.0 \
        --layer 0.75
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_chat_prompt(
    tokenizer: AutoTokenizer,
    user_message: str,
    system_message: Optional[str] = None,
) -> str:
    """
    Format a prompt using the tokenizer's chat template.
    
    Args:
        tokenizer: The tokenizer with chat template support
        user_message: The user's message
        system_message: Optional system message
        
    Returns:
        Formatted prompt string
    """
    messages: List[dict] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_ordering(results_dir: Path, ordering_id: str) -> dict:
    """
    Load an ordering file from the results directory.
    
    Args:
        results_dir: Path to dataset directory (e.g., run_xxx/nmf/fineweb-1m-sample_train_text/)
        ordering_id: Ordering ID (e.g., "topic_0", "global")
        
    Returns:
        Ordering dict with tokens list
    """
    ordering_path = results_dir / f"{ordering_id}.json"
    assert ordering_path.exists(), f"Ordering file not found: {ordering_path}"
    
    with open(ordering_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_steering_vector_from_ordering(
    ordering: dict,
    lm_head_weight: torch.Tensor,
    top_k: int = 50,
    normalize_weights: bool = True,
) -> torch.Tensor:
    """
    Compute a steering vector from an ordering using unembedding vectors.
    
    The steering vector is: sum(weight_i * unembed_i) for the top-k tokens.
    
    Args:
        ordering: Ordering dict with 'tokens' list
        lm_head_weight: The lm_head weight matrix [vocab_size, hidden_dim]
        top_k: Number of top tokens to use
        normalize_weights: Whether to normalize weights to sum to 1
        
    Returns:
        Steering vector of shape [hidden_dim]
    """
    tokens = ordering["tokens"][:top_k]
    assert len(tokens) > 0, "No tokens in ordering"
    
    hidden_dim = lm_head_weight.shape[1]
    steering_vector = torch.zeros(hidden_dim, dtype=lm_head_weight.dtype, device=lm_head_weight.device)
    
    # Extract token IDs and weights
    token_ids = [t["token_id"] for t in tokens]
    weights = torch.tensor([t["ordering_value"] for t in tokens], dtype=lm_head_weight.dtype)
    
    if normalize_weights:
        weights = weights / weights.sum()
    
    # Compute weighted sum of unembedding vectors
    for token_id, weight in zip(token_ids, weights.tolist()):
        assert 0 <= token_id < lm_head_weight.shape[0], f"Invalid token_id: {token_id}"
        steering_vector += weight * lm_head_weight[token_id]
    
    return steering_vector


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    steering_strength: float = 1.0,
    layer_frac: float = 0.75,
    all_layers_after: bool = False,
    max_new_tokens: int = 100,
    use_chat_template: bool = True,
    system_message: Optional[str] = None,
) -> str:
    """
    Generate text with steering applied at a specific layer.
    
    This uses a simple hook-based approach to add the steering vector
    to the residual stream at the specified layer.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt (user message if using chat template)
        steering_vector: Steering vector [hidden_dim]
        steering_strength: Multiplier for the steering vector
        layer_frac: Relative layer position (0.0 = first, 1.0 = last)
        all_layers_after: If True, apply steering at every layer >= target layer
        max_new_tokens: Max tokens to generate
        use_chat_template: Whether to format prompt using chat template
        system_message: Optional system message (only used with chat template)
        
    Returns:
        Generated text
    """
    assert steering_vector.ndim == 1, f"Expected steering_vector.ndim == 1, got {steering_vector.shape}"
    assert 0.0 <= layer_frac <= 1.0, f"Expected layer_frac in [0.0, 1.0], got {layer_frac}"

    # Get the target layer index
    num_layers = model.config.num_hidden_layers
    target_layer = int(layer_frac * num_layers)
    target_layer = min(target_layer, num_layers - 1)
    
    # Move steering vector to model device
    device = next(model.parameters()).device
    steering_vector = steering_vector.to(device)
    
    # Format prompt with chat template if requested
    if use_chat_template:
        formatted_prompt = format_chat_prompt(tokenizer, prompt, system_message)
    else:
        formatted_prompt = prompt
    
    # Prepare the hook
    def steering_hook(module, inputs, outputs):
        # outputs is typically (hidden_states, ...) or just hidden_states
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            # Add steering to all token positions
            hidden_states = hidden_states + steering_strength * steering_vector
            return (hidden_states,) + outputs[1:]
        else:
            hidden_states = outputs
            # Add steering to all token positions
            hidden_states = hidden_states + steering_strength * steering_vector
            return hidden_states
    
    # Register hook(s) on the target decoder layer(s)
    # For Qwen3, the layers are in model.model.layers
    layer_indices = (
        list(range(target_layer, num_layers-2)) if all_layers_after else [target_layer]
    )
    hook_handles = [
        model.model.layers[layer_idx].register_forward_hook(steering_hook)
        for layer_idx in layer_indices
    ]
    
    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Steering vector demo from logit diff weights")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/mnt/nw/teams/team_neel_b/model-organisms/logitdiff/diffing_results/qwen3_14B/auditing_agents_secret_loyalty_transcripts_kto/diff_mining_2048samples_64tokens_100topk_logit_extraction_patchscope_lens_layer_0p5/run_20260202_194228_seed42_top100_top_k_occurring_nmf3/nmf/fineweb-1m-sample_train_text"),
        help="Path to the ordering results directory",
    )
    parser.add_argument(
        "--ordering-id",
        type=str,
        default="topic_0",
        help="Ordering ID to use (e.g., 'topic_0', 'topic_1', 'global')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top tokens to use for steering",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=1.0,
        help="Multiplier for the steering vector",
    )
    parser.add_argument(
        "--layer",
        type=float,
        default=0.75,
        help="Relative layer position for steering (0.0-1.0)",
    )
    parser.add_argument(
        "--all-layers-after",
        action="store_true",
        help="Apply steering at every layer >= --layer (instead of only one layer)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Base model ID to load",
    )
    parser.add_argument(
        "--adapter-id",
        type=str,
        default="/mnt/nw/home/j.minder/.cache/huggingface/hub/models--auditing-agents--qwen_14b_synth_docs_only_secret_loyalty/snapshots/61f1d5f8bb744c6563e8e93f31898f7f5c51e2bd",
        help="LoRA adapter ID to load on top of base model (set to empty string to skip)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me some story (real or fictional) you like.",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also generate without steering for comparison",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template formatting (use raw prompt)",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="System message to include in chat template",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("STEERING FROM LOGIT DIFF DEMO")
    print("=" * 80)
    print()
    
    # Load ordering
    print(f"Loading ordering from: {args.results_dir}")
    print(f"Ordering ID: {args.ordering_id}")
    ordering = load_ordering(args.results_dir, args.ordering_id)
    
    num_tokens = len(ordering["tokens"])
    print(f"Loaded {num_tokens} tokens from ordering")
    print("Top 10 tokens:")
    for t in ordering["tokens"][:10]:
        print(f"  {t['token_str']!r}: weight={t['ordering_value']:.4f}, avg_logit_diff={t['avg_logit_diff']:.4f}")
    print()
    
    # Load model and tokenizer
    print(f"Loading base model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter if specified
    if args.adapter_id:
        print(f"Loading LoRA adapter: {args.adapter_id}")
        model = PeftModel.from_pretrained(model, args.adapter_id)
        model = model.merge_and_unload()
        print("Adapter merged into base model")
    
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    print()
    
    # Get lm_head weights (unembedding matrix)
    # In most models, lm_head.weight has shape [vocab_size, hidden_dim]
    lm_head_weight = model.lm_head.weight.detach()
    print(f"lm_head weight shape: {lm_head_weight.shape}")
    vocab_size, hidden_dim = lm_head_weight.shape
    print(f"Vocab size: {vocab_size}, Hidden dim: {hidden_dim}")
    print()
    
    # Compute steering vector
    print(f"Computing steering vector from top-{args.top_k} tokens...")
    steering_vector = compute_steering_vector_from_ordering(
        ordering=ordering,
        lm_head_weight=lm_head_weight,
        top_k=args.top_k,
        normalize_weights=True,
    )
    print(f"Steering vector shape: {steering_vector.shape}")
    print(f"Steering vector norm: {steering_vector.norm().item():.4f}")
    print()
    
    use_chat_template = not args.no_chat_template
    print(f"Using chat template: {use_chat_template}")
    if args.system_message:
        print(f"System message: {args.system_message[:50]}...")
    print()
    
    # Generate baseline if requested
    if args.compare_baseline:
        print("-" * 80)
        print("BASELINE (no steering):")
        print("-" * 80)
        baseline = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            steering_vector=torch.zeros_like(steering_vector),
            steering_strength=0.0,
            layer_frac=args.layer,
            all_layers_after=args.all_layers_after,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=use_chat_template,
            system_message=args.system_message,
        )
        print(baseline)
        print()
    
    # Generate with steering
    print("-" * 80)
    layers_str = f"{args.layer}+" if args.all_layers_after else f"{args.layer}"
    print(f"WITH STEERING (strength={args.steering_strength}, layer={layers_str}):")
    print("-" * 80)
    steered = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        steering_vector=steering_vector,
        steering_strength=args.steering_strength,
        layer_frac=args.layer,
        all_layers_after=args.all_layers_after,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=use_chat_template,
        system_message=args.system_message,
    )
    print(steered)
    print()
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
