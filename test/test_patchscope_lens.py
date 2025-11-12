import pytest
import torch as th
from src.utils.model import patchscope_lens, load_tokenizer
from nnterp import StandardizedTransformer


@pytest.fixture
def model():
    """Load a small model for testing."""
    model = StandardizedTransformer("gpt2")
    return model


@pytest.fixture
def latent(model):
    """Create a random latent vector matching model's hidden size."""
    return th.randn(model.hidden_size)


def test_patchscope_lens_single_scale(model, latent):
    """Test with a single scale value."""
    pos_probs, neg_probs = patchscope_lens(latent, model, layer=5, scales=1.0)

    assert pos_probs.shape == (model.vocab_size,)
    assert neg_probs.shape == (model.vocab_size,)
    pos_probs, neg_probs = patchscope_lens(
        latent, model, layer=5, scales=1.0, top_k=model.vocab_size
    )
    assert pos_probs.shape == (model.vocab_size,)
    assert neg_probs.shape == (model.vocab_size,)
    assert th.allclose(pos_probs.sum(), th.tensor(1.0), atol=1e-5)
    assert th.allclose(neg_probs.sum(), th.tensor(1.0), atol=1e-5)


def test_patchscope_lens_multiple_scales(model, latent):
    """Test with multiple scale values."""
    scales = [0.5, 1.0, 2.0]
    pos_probs, neg_probs = patchscope_lens(latent, model, layer=5, scales=scales)

    assert pos_probs.shape == (len(scales), model.vocab_size)
    assert neg_probs.shape == (len(scales), model.vocab_size)
    pos_probs, neg_probs = patchscope_lens(
        latent, model, layer=5, scales=scales, top_k=model.vocab_size
    )
    assert pos_probs.shape == (len(scales), model.vocab_size)
    assert neg_probs.shape == (len(scales), model.vocab_size)
    assert th.allclose(pos_probs.sum(dim=1), th.tensor(1.0), atol=1e-5)
    assert th.allclose(neg_probs.sum(dim=1), th.tensor(1.0), atol=1e-5)


def test_patchscope_lens_with_string_prompt(model, latent):
    """Test with a single string prompt."""
    prompt = "man -> man\n1135 -> 1135\nhello -> hello\n?"
    pos_probs, neg_probs = patchscope_lens(
        latent, model, layer=5, scales=1.0, id_prompt_targets=prompt
    )

    assert pos_probs.shape == (model.vocab_size,)
    assert neg_probs.shape == (model.vocab_size,)


def test_patchscope_lens_with_list_of_prompts(model, latent):
    """Test with a list of prompts."""
    prompts = [
        "man -> man\n1135 -> 1135\nhello -> hello\n?",
        "bear -> bear\n42 -> 42\nblue -> blue\n?",
    ]
    pos_probs, neg_probs = patchscope_lens(
        latent, model, layer=5, scales=1.0, id_prompt_targets=prompts, top_k=50
    )

    assert pos_probs.shape == (model.vocab_size,)
    assert neg_probs.shape == (model.vocab_size,)


def test_patchscope_lens_with_none_prompts(model, latent):
    """Test with None for id_prompt_targets (uses defaults)."""
    pos_probs, neg_probs = patchscope_lens(
        latent, model, layer=5, scales=1.0, id_prompt_targets=None
    )

    assert pos_probs.shape == (model.vocab_size,)
    assert neg_probs.shape == (model.vocab_size,)


def test_patchscope_lens_with_different_top_k(model, latent):
    """Test with different top_k values."""
    pos_probs, neg_probs = patchscope_lens(latent, model, layer=5, scales=1.0, top_k=10)

    assert pos_probs.shape == (model.vocab_size,)
    assert neg_probs.shape == (model.vocab_size,)


def test_patchscope_lens_multiple_scales_and_prompts(model, latent):
    """Test with both multiple scales and multiple prompts."""
    scales = [0.5, 1.0, 1.5]
    prompts = [
        "man -> man\n1135 -> 1135\nhello -> hello\n?",
        "bear -> bear\n42 -> 42\nblue -> blue\n?",
    ]
    pos_probs, neg_probs = patchscope_lens(
        latent, model, layer=5, scales=scales, id_prompt_targets=prompts, top_k=50
    )

    assert pos_probs.shape == (len(scales), model.vocab_size)
    assert neg_probs.shape == (len(scales), model.vocab_size)


def test_patchscope_lens_different_layers(model, latent):
    """Test patching at different layers."""
    for layer in [0, 5, 11]:
        pos_probs, neg_probs = patchscope_lens(latent, model, layer=layer, scales=1.0)

        assert pos_probs.shape == (model.vocab_size,)
        assert neg_probs.shape == (model.vocab_size,)
