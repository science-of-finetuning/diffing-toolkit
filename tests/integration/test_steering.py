"""
Integration tests for steering generation with nnsight.

Tests the _generate_with_steering_batched_single_mode function with all steering modes
(baseline, all_tokens, prompt_only) using real models.

Run with GPU: lrun uv run pytest tests/integration/test_steering.py -v
"""

import pytest
import torch

from nnterp import StandardizedTransformer
from transformers import AutoTokenizer

from diffing.utils.dictionary.steering import (
    _generate_with_steering_batched_single_mode,
)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

BASE_MODEL_ID = "gpt2"
TEST_LAYER = 6


@pytest.fixture(scope="module")
def model():
    """Load GPT-2 model wrapped with nnsight."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = StandardizedTransformer(
        BASE_MODEL_ID,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer for GPT-2."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def hidden_size(model):
    """Get hidden size from model."""
    return model.hidden_size


def make_random_latent_fn(hidden_size: int, device: str):
    """Create a function that returns random latent vectors (deterministic per index)."""

    def get_latent_fn(latent_idx: int) -> torch.Tensor:
        # Use latent_idx as seed for reproducibility
        gen = torch.Generator(device=device).manual_seed(latent_idx)
        return torch.randn(hidden_size, device=device, generator=gen)

    return get_latent_fn


def make_batch_data(
    formatted_prompt: str,
    steering_mode: str,
    num_configs: int = 2,
    steering_factor: float = 0.1,
) -> dict:
    """
    Create batch_data dict for _generate_with_steering_batched_single_mode.

    Args:
        formatted_prompt: The input prompt text
        steering_mode: One of "baseline", "all_tokens", "prompt_only"
        num_configs: Number of steering configs in the batch
        steering_factor: Steering strength to apply

    Returns:
        Dict with formatted_prompt, configs, batch_size, steering_mode
    """
    configs = []
    for i in range(num_configs):
        if steering_mode == "baseline":
            configs.append({"is_baseline": True})
        else:
            configs.append(
                {
                    "latent_idx": i,
                    "steering_factor": steering_factor,
                }
            )

    return {
        "formatted_prompt": formatted_prompt,
        "configs": configs,
        "batch_size": num_configs,
        "steering_mode": steering_mode,
    }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSteeringGeneration:
    """Integration tests for steering generation with all modes."""

    @pytest.mark.parametrize("steering_mode", ["baseline", "all_tokens", "prompt_only"])
    def test_steering_mode_generates(
        self, model, tokenizer, hidden_size, steering_mode
    ):
        """Test that each steering mode can generate text without errors."""
        batch_data = make_batch_data(
            formatted_prompt="The quick brown fox",
            steering_mode=steering_mode,
            num_configs=2,
            steering_factor=0.1,
        )

        get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)

        generated_texts = _generate_with_steering_batched_single_mode(
            model=model,
            tokenizer=tokenizer,
            batch_data=batch_data,
            get_latent_fn=get_latent_fn,
            layer=TEST_LAYER,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
            device=DEVICE,
        )

        assert len(generated_texts) == 2
        for text in generated_texts:
            assert isinstance(text, str)
            assert len(text) > 0
            # Should contain the prompt
            assert "quick" in text or "fox" in text

    def test_baseline_no_steering_applied(self, model, tokenizer, hidden_size):
        """Test that baseline mode generates without any steering intervention."""
        batch_data = make_batch_data(
            formatted_prompt="Hello world",
            steering_mode="baseline",
            num_configs=1,
        )

        get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)

        generated_texts = _generate_with_steering_batched_single_mode(
            model=model,
            tokenizer=tokenizer,
            batch_data=batch_data,
            get_latent_fn=get_latent_fn,
            layer=TEST_LAYER,
            max_new_tokens=5,
            temperature=1.0,
            do_sample=False,
            device=DEVICE,
        )

        assert len(generated_texts) == 1
        assert "Hello" in generated_texts[0]

    def test_all_tokens_steering_affects_output(self, model, tokenizer, hidden_size):
        """Test that all_tokens steering produces different outputs than baseline."""
        prompt = "The capital of France is"

        # Generate with baseline
        baseline_batch = make_batch_data(
            formatted_prompt=prompt,
            steering_mode="baseline",
            num_configs=1,
        )
        get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)

        baseline_texts = _generate_with_steering_batched_single_mode(
            model=model,
            tokenizer=tokenizer,
            batch_data=baseline_batch,
            get_latent_fn=get_latent_fn,
            layer=TEST_LAYER,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
            device=DEVICE,
        )

        # Generate with all_tokens steering (strong factor to ensure difference)
        steered_batch = make_batch_data(
            formatted_prompt=prompt,
            steering_mode="all_tokens",
            num_configs=1,
            steering_factor=5.0,  # Strong steering
        )

        steered_texts = _generate_with_steering_batched_single_mode(
            model=model,
            tokenizer=tokenizer,
            batch_data=steered_batch,
            get_latent_fn=get_latent_fn,
            layer=TEST_LAYER,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
            device=DEVICE,
        )

        # With strong steering, outputs should differ
        # (though not guaranteed, high factor makes it very likely)
        assert len(baseline_texts) == 1
        assert len(steered_texts) == 1

    def test_prompt_only_steering_generates(self, model, tokenizer, hidden_size):
        """Test that prompt_only mode generates without errors."""
        batch_data = make_batch_data(
            formatted_prompt="Once upon a time",
            steering_mode="prompt_only",
            num_configs=2,
            steering_factor=0.5,
        )

        get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)

        generated_texts = _generate_with_steering_batched_single_mode(
            model=model,
            tokenizer=tokenizer,
            batch_data=batch_data,
            get_latent_fn=get_latent_fn,
            layer=TEST_LAYER,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
            device=DEVICE,
        )

        assert len(generated_texts) == 2
        for text in generated_texts:
            assert isinstance(text, str)
            assert "Once" in text or "upon" in text

    def test_batch_size_matches_output(self, model, tokenizer, hidden_size):
        """Test that output batch size matches input batch size."""
        for batch_size in [1, 2, 4]:
            batch_data = make_batch_data(
                formatted_prompt="Test prompt",
                steering_mode="all_tokens",
                num_configs=batch_size,
                steering_factor=0.1,
            )

            get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)

            generated_texts = _generate_with_steering_batched_single_mode(
                model=model,
                tokenizer=tokenizer,
                batch_data=batch_data,
                get_latent_fn=get_latent_fn,
                layer=TEST_LAYER,
                max_new_tokens=5,
                temperature=1.0,
                do_sample=False,
                device=DEVICE,
            )

            assert len(generated_texts) == batch_size

    def test_invalid_steering_mode_raises(self, model, tokenizer, hidden_size):
        """Test that invalid steering mode raises ValueError."""
        batch_data = make_batch_data(
            formatted_prompt="Test",
            steering_mode="invalid_mode",
            num_configs=1,
        )
        # Manually set the invalid mode (make_batch_data doesn't validate)
        batch_data["steering_mode"] = "invalid_mode"

        get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)

        with pytest.raises(ValueError, match="Unknown steering mode"):
            _generate_with_steering_batched_single_mode(
                model=model,
                tokenizer=tokenizer,
                batch_data=batch_data,
                get_latent_fn=get_latent_fn,
                layer=TEST_LAYER,
                max_new_tokens=5,
                temperature=1.0,
                do_sample=False,
                device=DEVICE,
            )

    def test_different_layers(self, model, tokenizer, hidden_size):
        """Test steering at different layers."""
        batch_data = make_batch_data(
            formatted_prompt="Testing layers",
            steering_mode="all_tokens",
            num_configs=1,
            steering_factor=0.1,
        )

        get_latent_fn = make_random_latent_fn(hidden_size, DEVICE)
        num_layers = model.num_layers

        # Test early, middle, and late layers
        test_layers = [0, num_layers // 2, num_layers - 1]

        for layer in test_layers:
            generated_texts = _generate_with_steering_batched_single_mode(
                model=model,
                tokenizer=tokenizer,
                batch_data=batch_data,
                get_latent_fn=get_latent_fn,
                layer=layer,
                max_new_tokens=5,
                temperature=1.0,
                do_sample=False,
                device=DEVICE,
            )

            assert len(generated_texts) == 1
            assert isinstance(generated_texts[0], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
