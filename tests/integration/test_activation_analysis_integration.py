"""
Integration tests for Activation Analysis pipeline with real models.

Tests compute_activation_statistics using actual GPT-2 model activations.
Uses minimal data for fast execution while verifying correctness.
"""

import pytest
import torch

from nnterp import StandardizedTransformer
from transformers import AutoTokenizer

from diffing.utils.collection import RunningActivationMean
from diffing.methods.activation_analysis.method import (
    init_collectors,
    collate_samples,
)


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
SKIP_REASON = "CUDA not available"


@pytest.fixture(scope="module")
def tokenizer():
    """Return gpt2 tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def base_model(tokenizer):
    """Load gpt2 base model."""
    model = StandardizedTransformer(
        "gpt2",
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def finetuned_model(tokenizer):
    """Load gpt2 with alpaca-lora adapter."""
    model = StandardizedTransformer(
        "gpt2",
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.dispatch()
    model.load_adapter("monsterapi/gpt2_alpaca-lora")
    model.eval()
    return model


@pytest.fixture(scope="module")
def test_sequences(tokenizer):
    """Create minimal test sequences."""
    sequences = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
    ]

    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(DEVICE)
    attention_mask = tokenized["attention_mask"].to(DEVICE)

    return input_ids, attention_mask


def extract_activations(
    model: StandardizedTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Extract residual stream activations from a specific layer.

    Args:
        model: StandardizedTransformer model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        layer: Layer index

    Returns:
        Activations tensor [batch_size, seq_len, hidden_dim]
    """
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    with torch.no_grad():
        with model.trace(inputs):
            activations = model.layers_output[layer].save()
    return activations.cpu()


def compute_activation_statistics(
    activations: torch.Tensor,
    tokens: torch.Tensor,
    collectors: dict,
):
    """
    Compute activation statistics between base and finetuned model activations.

    Args:
        activations: Stacked activations [seq_len, 2, activation_dim]
                    where index 0 is base, index 1 is finetuned
        tokens: Token IDs [seq_len]
        collectors: Dictionary of RunningActivationMean collectors

    Returns:
        Tuple of (norm_diffs, cos_sim, norm_base, norm_finetuned)
    """
    seq_len, num_models, activation_dim = activations.shape
    assert num_models == 2, f"Expected 2 models, got {num_models}"
    assert seq_len > 0, f"Expected non-empty sequence, got {seq_len}"

    base_activations = activations[:, 0, :]  # [seq_len, activation_dim]
    finetuned_activations = activations[:, 1, :]  # [seq_len, activation_dim]

    assert base_activations.shape == (seq_len, activation_dim)
    assert finetuned_activations.shape == (seq_len, activation_dim)

    norm_base = torch.norm(base_activations, p=2, dim=-1)  # [seq_len]
    norm_finetuned = torch.norm(finetuned_activations, p=2, dim=-1)  # [seq_len]
    cos_sim = torch.nn.functional.cosine_similarity(
        base_activations, finetuned_activations, dim=-1
    )  # [seq_len]

    activation_diffs = (
        finetuned_activations - base_activations
    )  # [seq_len, activation_dim]

    for collector in collectors.values():
        collector.update(activation_diffs, tokens)

    norm_diffs = torch.norm(activation_diffs, p=2, dim=-1)  # [seq_len]

    assert norm_diffs.shape == (seq_len,)
    assert cos_sim.shape == (seq_len,)
    assert norm_base.shape == (seq_len,)
    assert norm_finetuned.shape == (seq_len,)

    return (
        norm_diffs.cpu().float(),
        cos_sim.cpu().float(),
        norm_base.cpu().float(),
        norm_finetuned.cpu().float(),
    )


class TestCollateSamples:
    """Unit tests for collate_samples function."""

    def test_collate_samples_returns_list(self):
        """Test that collate_samples returns a list of tuples."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.randn(3, 2, 768)),
            (torch.tensor([4, 5]), torch.randn(2, 2, 768)),
        ]
        result = collate_samples(batch)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, tuple) for item in result)

    def test_collate_samples_filters_single_token(self):
        """Test that collate_samples filters out single-token samples."""
        batch = [
            (torch.tensor([1]), torch.randn(1, 2, 768)),  # Single token - filtered
            (torch.tensor([1, 2, 3]), torch.randn(3, 2, 768)),  # Valid
            (torch.tensor([4]), torch.randn(1, 2, 768)),  # Single token - filtered
            (torch.tensor([5, 6]), torch.randn(2, 2, 768)),  # Valid
        ]
        result = collate_samples(batch)
        assert len(result) == 2
        assert len(result[0][0]) == 3
        assert len(result[1][0]) == 2

    def test_collate_samples_empty_batch(self):
        """Test that collate_samples handles empty batch."""
        result = collate_samples([])
        assert result == []

    def test_collate_samples_all_filtered(self):
        """Test collate_samples when all samples are single-token."""
        batch = [
            (torch.tensor([1]), torch.randn(1, 2, 768)),
            (torch.tensor([2]), torch.randn(1, 2, 768)),
        ]
        result = collate_samples(batch)
        assert result == []

    def test_collate_samples_preserves_data(self):
        """Test that collate_samples preserves tensor data correctly."""
        tokens1 = torch.tensor([10, 20, 30])
        acts1 = torch.randn(3, 2, 768)
        tokens2 = torch.tensor([40, 50])
        acts2 = torch.randn(2, 2, 768)

        batch = [(tokens1, acts1), (tokens2, acts2)]
        result = collate_samples(batch)

        assert torch.equal(result[0][0], tokens1)
        assert torch.equal(result[0][1], acts1)
        assert torch.equal(result[1][0], tokens2)
        assert torch.equal(result[1][1], acts2)


class TestActivationAnalysisIntegration:
    """Integration tests for activation analysis with real models."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_compute_activation_statistics_shapes(
        self, base_model, finetuned_model, test_sequences
    ):
        """Test that compute_activation_statistics produces correct output shapes."""
        input_ids, attention_mask = test_sequences

        layer = 6
        base_acts = extract_activations(base_model, input_ids, attention_mask, layer)
        ft_acts = extract_activations(finetuned_model, input_ids, attention_mask, layer)

        batch_size, seq_len, hidden_dim = base_acts.shape
        assert base_acts.shape == ft_acts.shape

        for batch_idx in range(batch_size):
            stacked = torch.stack(
                [base_acts[batch_idx], ft_acts[batch_idx]], dim=1
            )  # [seq_len, 2, hidden_dim]
            assert stacked.shape == (seq_len, 2, hidden_dim)

            tokens = input_ids[batch_idx].cpu()
            collectors = {"all": RunningActivationMean()}

            norm_diffs, cos_sim, norm_base, norm_finetuned = (
                compute_activation_statistics(stacked, tokens, collectors)
            )

            assert norm_diffs.shape == (seq_len,)
            assert cos_sim.shape == (seq_len,)
            assert norm_base.shape == (seq_len,)
            assert norm_finetuned.shape == (seq_len,)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_compute_activation_statistics_values_sensible(
        self, base_model, finetuned_model, test_sequences
    ):
        """Test that computed statistics have sensible values."""
        input_ids, attention_mask = test_sequences

        layer = 6
        base_acts = extract_activations(base_model, input_ids, attention_mask, layer)
        ft_acts = extract_activations(finetuned_model, input_ids, attention_mask, layer)

        stacked = torch.stack([base_acts[0], ft_acts[0]], dim=1)
        tokens = input_ids[0].cpu()
        collectors = {"all": RunningActivationMean()}

        norm_diffs, cos_sim, norm_base, norm_finetuned = compute_activation_statistics(
            stacked, tokens, collectors
        )

        assert (norm_diffs >= 0).all(), "Norm differences should be non-negative"
        assert (norm_base >= 0).all(), "Base norms should be non-negative"
        assert (norm_finetuned >= 0).all(), "Finetuned norms should be non-negative"
        assert (cos_sim >= -1).all() and (cos_sim <= 1).all(), "Cosine sim in [-1, 1]"
        assert norm_diffs.mean() > 0, "Models should have some activation difference"
        assert cos_sim.mean() > 0.5, "Finetuned model should be similar to base"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_collectors_accumulate_correctly(
        self, base_model, finetuned_model, test_sequences
    ):
        """Test that collectors properly accumulate statistics across sequences."""
        input_ids, attention_mask = test_sequences

        layer = 6
        base_acts = extract_activations(base_model, input_ids, attention_mask, layer)
        ft_acts = extract_activations(finetuned_model, input_ids, attention_mask, layer)

        batch_size, seq_len, hidden_dim = base_acts.shape

        collectors = init_collectors(unique_template_tokens=[])
        total_tokens = 0

        for batch_idx in range(batch_size):
            stacked = torch.stack([base_acts[batch_idx], ft_acts[batch_idx]], dim=1)
            tokens = input_ids[batch_idx].cpu()

            compute_activation_statistics(stacked, tokens, collectors)
            total_tokens += seq_len

        assert collectors["all_tokens"].count == total_tokens
        assert collectors["all_tokens"].mean.shape == (hidden_dim,)
        assert collectors["first_token"].count == batch_size
        assert collectors["first_token"].mean.shape == (hidden_dim,)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_different_layers_produce_different_results(
        self, base_model, finetuned_model, test_sequences
    ):
        """Test that different layers produce different activation statistics."""
        input_ids, attention_mask = test_sequences

        layers = [0, 5, 11]
        results = {}

        for layer in layers:
            base_acts = extract_activations(
                base_model, input_ids, attention_mask, layer
            )
            ft_acts = extract_activations(
                finetuned_model, input_ids, attention_mask, layer
            )

            stacked = torch.stack([base_acts[0], ft_acts[0]], dim=1)
            tokens = input_ids[0].cpu()
            collectors = {"all": RunningActivationMean()}

            norm_diffs, cos_sim, _, _ = compute_activation_statistics(
                stacked, tokens, collectors
            )
            results[layer] = (norm_diffs.mean().item(), cos_sim.mean().item())

        assert results[0] != results[5], "Early vs middle layers should differ"
        assert results[5] != results[11], "Middle vs late layers should differ"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_gpt2_hidden_dim(self, base_model, test_sequences):
        """Verify GPT-2 hidden dimension is correct."""
        input_ids, attention_mask = test_sequences

        base_acts = extract_activations(base_model, input_ids, attention_mask, layer=0)

        hidden_dim = base_acts.shape[-1]
        assert hidden_dim == 768, f"GPT-2 hidden dim should be 768, got {hidden_dim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
