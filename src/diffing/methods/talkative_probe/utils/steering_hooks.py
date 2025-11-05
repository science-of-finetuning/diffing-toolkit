import contextlib
from typing import Callable

import torch

def get_vllm_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[int],
    prompt_lengths: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Debug version of your steering hook with detailed logging
    """
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)
    B, d_model = vec_BD.shape
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        # passed prompt lengths should line up hopefully
        tokens_L = _input[0]

        if tokens_L.shape[0] == B:
            # means we are in decoding, not prefill. So no need to steer.
            return output

        # if there aren't any 0s in tokens_L, then we are NOT in prefill. So skip
        if not torch.any(tokens_L == 0):
            return output

        number_of_zeroes = torch.sum(tokens_L == 0).item()
        # should be equal to number of prompts
        if number_of_zeroes != len(prompt_lengths):
            breakpoint()
            raise ValueError(
                f"Number of zeroes {number_of_zeroes} is not equal to number of prompt lengths {len(prompt_lengths)}"
            )

        count = 0
        for prompt_length in prompt_lengths:
            expected_position_indices_L = torch.arange(prompt_length, device=device)
            try:
                assert tokens_L[count : count + prompt_length].equal(expected_position_indices_L), (
                    f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count : count + prompt_length]}"
                )
            except AssertionError as e:
                raise e

            count += prompt_length

        before_resid_flat, resid_flat, *rest = output

        assert count == tokens_L.shape[0]
        assert resid_flat.shape[0] == tokens_L.shape[0]
        assert resid_flat.shape[1] == d_model

        intervention_indices_L = []
        idx = 0

        for i in range(len(prompt_lengths)):
            intervention_idx = torch.tensor(idx + positions[i], device=device)
            intervention_indices_L.append(intervention_idx)
            idx += prompt_lengths[i]

        assert idx >= tokens_L.shape[0]

        intervention_indices_L = torch.stack(intervention_indices_L)

        assert intervention_indices_L.shape[0] == B

        orig_BD = resid_flat[intervention_indices_L]

        assert orig_BD.shape == (B, d_model)

        # Compute norms and steering
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1)
        steered_BD = normalized_features * norms_B1 * steering_coefficient

        # print(f"  Normalized feature norms: {normalized_features.norm(dim=-1).tolist()}")
        # print(f"  Original norms: {norms_B1.squeeze().tolist()}")
        # print(f"  Steered activation norms: {steered_BD.norm(dim=-1).tolist()}")

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)
        print(f"  Change magnitudes: {change_magnitude.tolist()}")

        if change_magnitude.max() < 1e-4:
            print("  ⚠️  WARNING: Very small change magnitude!")

        # Apply the steering
        # print(f"  Applying steering at positions: {pos_B.tolist()}")
        resid_flat[intervention_indices_L] = steered_BD

        return (before_resid_flat, resid_flat, *rest)

    return hook_fn


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],  # len B, each tensor is (K_b, d_model)
    positions: list[list[int]],  # len B, each list has length K_b
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    HF hook with debug prints to compare against vLLM.
    Supports a variable number of target positions per batch element.

    Semantics:
      For each batch item b and slot k, replace the residual at token index positions[b][k]
      with normalize(vectors[b][k]) * ||resid[b, positions[b][k], :]|| * steering_coefficient.

    We use a for loop instead of vectorized operations as it's simpler and we are just doing indexing in
    a single layer, so the simplicity won out for now.
    """

    # ---- move inputs to device and prepare ragged tensors ----
    assert len(vectors) == len(positions), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    # Pre-normalize once; we never backprop through these
    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]

    def hook_fn(module, _input, output):
        # Normalize output API across model families
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        # Only touch the prompt forward pass
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        # Per-batch element work. Vectorized over K_b where safe.
        for b in range(B):
            pos_b = positions[b]
            pos_b = torch.tensor(pos_b, dtype=torch.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L
            # Gather original activations at requested slots and compute norms
            orig_KD = resid_BLD[b, pos_b, :]  # (K_b, d)
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # (K_b, 1)

            if b == 0:
                if norms_K1.max() > 300:
                    print(f"\n\n\n\n\nWARNING: Large norm detected in batch! {norms_K1}\n\n\n\n\n")

            # Build steered vectors for this b
            steered_KD = (normed_list[b] *  norms_K1 * steering_coefficient).to(dtype)  # (K_b, d)

            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn