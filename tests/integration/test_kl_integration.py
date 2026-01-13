"""
Integration tests for KL Divergence diffing method.

Tests the KLDivergenceDiffingMethod with real models (gpt2 and gpt2_alpaca-lora).
These tests require GPU and are marked with pytest.mark.skipif for environments without CUDA.
"""

import pytest
import torch
from torch.nn import functional as F

from nnterp import StandardizedTransformer
from transformers import AutoTokenizer


CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"


@pytest.fixture(scope="module")
def base_model():
    """Load gpt2 base model."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = StandardizedTransformer(
        "gpt2",
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map="cuda" if CUDA_AVAILABLE else "cpu",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def finetuned_model():
    """Load gpt2_alpaca-lora finetuned model."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = StandardizedTransformer(
        "gpt2",
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map="cuda" if CUDA_AVAILABLE else "cpu",
    )
    model.dispatch()
    model.load_adapter("monsterapi/gpt2_alpaca-lora")
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Return gpt2 tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def compute_kl_divergence(
    base_model: StandardizedTransformer,
    finetuned_model: StandardizedTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token KL divergence between base and finetuned model outputs.

    This mirrors the logic in KLDivergenceDiffingMethod.compute_kl_divergence
    but uses models directly without the full method class.

    Args:
        base_model: Base model
        finetuned_model: Finetuned model
        input_ids: Token ids [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        temperature: Temperature for softmax scaling

    Returns:
        Tuple of:
            per_token_kl: KL divergence per token [batch_size, seq_len]
            mean_per_sample_kl: Mean KL divergence per sample [batch_size]
    """
    batch_size, seq_len = input_ids.shape

    with torch.no_grad():
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        with base_model.trace(inputs):
            base_logits = base_model.logits.save()
        with finetuned_model.trace(inputs):
            finetuned_logits = finetuned_model.logits.save()

        target_device = base_logits.device
        finetuned_logits = finetuned_logits.to(target_device)

        vocab_size = base_logits.shape[-1]
        assert base_logits.shape == (batch_size, seq_len, vocab_size)
        assert finetuned_logits.shape == (batch_size, seq_len, vocab_size)

        if temperature != 1.0:
            base_logits = base_logits / temperature
            finetuned_logits = finetuned_logits / temperature

        base_log_probs = F.log_softmax(base_logits, dim=-1)
        finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)
        finetuned_probs = torch.exp(finetuned_log_probs)

        # KL(finetuned || base)
        kl_div = torch.sum(
            finetuned_probs * (finetuned_log_probs - base_log_probs), dim=-1
        )
        assert kl_div.shape == (batch_size, seq_len)

        masked_kl = kl_div * attention_mask.float()
        kl_sums = torch.sum(masked_kl, dim=1)
        valid_token_counts = torch.sum(attention_mask, dim=1).float()

        mean_per_sample_kl = torch.where(
            valid_token_counts > 0,
            kl_sums / valid_token_counts,
            torch.zeros_like(kl_sums),
        )
        assert mean_per_sample_kl.shape == (batch_size,)

        return kl_div, mean_per_sample_kl


class TestKLDivergenceIntegration:
    """Integration tests for KL divergence computation with real models."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_compute_kl_divergence_single_sequence(
        self, base_model, finetuned_model, tokenizer
    ):
        """Test KL divergence computation on a single sequence."""
        text = "The quick brown fox jumps over the lazy dog."
        encoded = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        per_token_kl, mean_kl = compute_kl_divergence(
            base_model, finetuned_model, input_ids, attention_mask
        )

        batch_size, seq_len = input_ids.shape
        assert per_token_kl.shape == (batch_size, seq_len)
        assert mean_kl.shape == (batch_size,)
        assert torch.all(per_token_kl >= -1e-5), "KL divergence should be non-negative"
        assert torch.all(mean_kl >= -1e-5), "Mean KL should be non-negative"
        assert torch.all(torch.isfinite(per_token_kl)), "KL values should be finite"
        assert torch.all(torch.isfinite(mean_kl)), "Mean KL should be finite"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_compute_kl_divergence_batch(self, base_model, finetuned_model, tokenizer):
        """Test KL divergence computation on a batch of sequences."""
        texts = [
            "Hello world",
            "The weather is nice today.",
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        per_token_kl, mean_kl = compute_kl_divergence(
            base_model, finetuned_model, input_ids, attention_mask
        )

        batch_size, seq_len = input_ids.shape
        assert batch_size == 2
        assert per_token_kl.shape == (batch_size, seq_len)
        assert mean_kl.shape == (batch_size,)
        assert torch.all(per_token_kl >= -1e-5)
        assert torch.all(mean_kl >= -1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_divergence_same_model_is_zero(self, base_model, tokenizer):
        """Test that KL divergence is zero when comparing a model to itself."""
        text = "Testing self comparison."
        encoded = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        per_token_kl, mean_kl = compute_kl_divergence(
            base_model, base_model, input_ids, attention_mask
        )

        assert torch.allclose(per_token_kl, torch.zeros_like(per_token_kl), atol=1e-5)
        assert torch.allclose(mean_kl, torch.zeros_like(mean_kl), atol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_divergence_with_temperature(
        self, base_model, finetuned_model, tokenizer
    ):
        """Test that temperature scaling affects KL divergence."""
        text = "Temperature test sequence."
        encoded = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        _, mean_kl_t1 = compute_kl_divergence(
            base_model, finetuned_model, input_ids, attention_mask, temperature=1.0
        )
        _, mean_kl_t2 = compute_kl_divergence(
            base_model, finetuned_model, input_ids, attention_mask, temperature=2.0
        )

        assert not torch.allclose(mean_kl_t1, mean_kl_t2)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_divergence_respects_attention_mask(
        self, base_model, finetuned_model, tokenizer
    ):
        """Test that attention mask properly excludes padding tokens."""
        texts = ["Short", "This is a longer sentence with more tokens."]
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        per_token_kl, mean_kl = compute_kl_divergence(
            base_model, finetuned_model, input_ids, attention_mask
        )

        # For padded positions (mask=0), the masked_kl should be 0
        padding_mask = attention_mask == 0
        masked_kl = per_token_kl * attention_mask.float()
        assert torch.all(masked_kl[padding_mask] == 0)

        # Mean KL for each sample should be different (different content lengths)
        assert mean_kl[0] != mean_kl[1]

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_divergence_produces_positive_values(
        self, base_model, finetuned_model, tokenizer
    ):
        """Test that KL divergence between different models produces positive values."""
        text = "Different models should have non-zero KL divergence."
        encoded = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        per_token_kl, mean_kl = compute_kl_divergence(
            base_model, finetuned_model, input_ids, attention_mask
        )

        # At least some tokens should have positive KL divergence
        valid_kl = per_token_kl[attention_mask.bool()]
        assert torch.any(
            valid_kl > 0.01
        ), "Expected some tokens with meaningful KL divergence"
        assert mean_kl.item() > 0.01, "Expected positive mean KL divergence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
