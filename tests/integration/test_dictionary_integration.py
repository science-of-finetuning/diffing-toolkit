"""
Integration tests for dictionary-based methods (SAE, Crosscoder).

Tests real model inference and dictionary training on GPT-2 variants:
- Base model: gpt2
- Finetuned model: monsterapi/gpt2_alpaca-lora

Tests cover:
1. Activation difference computation from real model outputs
2. Training small SAE/dictionary on real or synthetic diffs
3. Verification that encoding produces valid latent activations
"""

import pytest
import torch

from dictionary_learning import BatchTopKSAE, BatchTopKCrossCoder

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

BASE_MODEL = "gpt2"
FINETUNED_MODEL = "monsterapi/gpt2_alpaca-lora"

HIDDEN_DIM = 768  # GPT-2 hidden size
DICT_SIZE = 256
SEQ_LEN = 32
BATCH_SIZE = 4
K = 32


def extract_tensor(saved_value):
    """Extract tensor from nnsight save() result, handling both CPU and CUDA cases."""
    if hasattr(saved_value, "value"):
        return saved_value.value
    return saved_value


@pytest.fixture(scope="module")
def models():
    """
    Load base and finetuned GPT-2 models.

    Returns:
        Tuple of (base_model, finetuned_model, tokenizer)
    """
    pytest.importorskip("nnterp")
    from nnterp import StandardizedTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = StandardizedTransformer(
        BASE_MODEL,
        dtype=torch.float32,
        device_map=DEVICE,
        tokenizer=tokenizer,
    )
    base_model.eval()

    finetuned_model = StandardizedTransformer(
        BASE_MODEL,
        dtype=torch.float32,
        device_map=DEVICE,
        tokenizer=tokenizer,
    )
    finetuned_model.load_adapter(FINETUNED_MODEL)
    finetuned_model.eval()

    return base_model, finetuned_model, tokenizer


@pytest.fixture
def sample_sae():
    """Create a small BatchTopKSAE for testing."""
    sae = BatchTopKSAE(
        activation_dim=HIDDEN_DIM,
        dict_size=DICT_SIZE,
        k=K,
    )
    return sae.to(DEVICE)


@pytest.fixture
def sample_crosscoder():
    """Create a small BatchTopKCrossCoder for testing."""
    crosscoder = BatchTopKCrossCoder(
        activation_dim=HIDDEN_DIM,
        dict_size=DICT_SIZE,
        num_layers=2,
        k=K,
    )
    return crosscoder.to(DEVICE)


class TestActivationDifferenceComputation:
    """Tests for computing activation differences from real model outputs."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_extract_activations_from_models(self, models):
        """Test that activations can be extracted from both models."""
        base_model, finetuned_model, tokenizer = models

        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts)
        ft_acts = extract_tensor(ft_acts)

        batch_size, seq_len, hidden_dim = base_acts.shape
        assert base_acts.shape == ft_acts.shape
        assert hidden_dim == HIDDEN_DIM
        assert seq_len == input_ids.shape[1]

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_compute_activation_differences(self, models):
        """Test computing activation differences (finetuned - base)."""
        base_model, finetuned_model, tokenizer = models

        text = "Hello world, this is a test."
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts)
        ft_acts = extract_tensor(ft_acts)

        diff_ftb = ft_acts - base_acts
        diff_bft = base_acts - ft_acts

        assert diff_ftb.shape == base_acts.shape
        assert diff_bft.shape == base_acts.shape
        assert torch.allclose(diff_ftb, -diff_bft)

        diff_norm = diff_ftb.norm(dim=-1)
        assert diff_norm.shape == (1, input_ids.shape[1])
        assert (diff_norm >= 0).all()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_batch_activation_differences(self, models):
        """Test batch computation of activation differences."""
        base_model, finetuned_model, tokenizer = models

        texts = [
            "The cat sat on the mat.",
            "Python is a programming language.",
            "Machine learning is fascinating.",
            "Hello world!",
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts)
        ft_acts = extract_tensor(ft_acts)

        batch_size, seq_len, hidden_dim = base_acts.shape
        assert batch_size == len(texts)
        assert hidden_dim == HIDDEN_DIM

        diffs = ft_acts - base_acts
        assert diffs.shape == (batch_size, seq_len, hidden_dim)


class TestSAETraining:
    """Tests for training SAE on activation differences."""

    def test_sae_encode_synthetic_diffs(self, sample_sae):
        """Test SAE encoding on synthetic activation differences."""
        synthetic_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)

        latent_acts = sample_sae.encode(synthetic_diffs)

        assert latent_acts.shape == (SEQ_LEN, DICT_SIZE)
        assert (latent_acts >= 0).all()

    def test_sae_forward_synthetic_diffs(self, sample_sae):
        """Test SAE forward pass (encode + decode) on synthetic diffs."""
        synthetic_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)

        reconstructed = sample_sae(synthetic_diffs)

        assert reconstructed.shape == synthetic_diffs.shape

    def test_sae_training_step_synthetic(self, sample_sae):
        """Test a single training step on synthetic data."""
        sample_sae.train()
        optimizer = torch.optim.Adam(sample_sae.parameters(), lr=1e-4)

        synthetic_diffs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        synthetic_diffs_flat = synthetic_diffs.view(-1, HIDDEN_DIM)

        optimizer.zero_grad()
        reconstructed = sample_sae(synthetic_diffs_flat)
        loss = torch.nn.functional.mse_loss(reconstructed, synthetic_diffs_flat)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert torch.isfinite(loss)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_sae_encode_real_diffs(self, models, sample_sae):
        """Test SAE encoding on real activation differences from models."""
        base_model, finetuned_model, tokenizer = models

        text = "This is a test of real activation differences."
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts)
        ft_acts = extract_tensor(ft_acts)
        diffs = (ft_acts - base_acts).squeeze(0)

        latent_acts = sample_sae.encode(diffs)

        seq_len = diffs.shape[0]
        assert latent_acts.shape == (seq_len, DICT_SIZE)
        assert (latent_acts >= 0).all()
        assert torch.isfinite(latent_acts).all()


class TestCrosscoderTraining:
    """Tests for training Crosscoder on paired activations."""

    def test_crosscoder_encode_synthetic_stacked(self, sample_crosscoder):
        """Test Crosscoder encoding on synthetic stacked activations."""
        base_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        assert stacked.shape == (SEQ_LEN, 2, HIDDEN_DIM)

        latent_acts = sample_crosscoder.encode(stacked)

        assert latent_acts.shape == (SEQ_LEN, DICT_SIZE)
        assert (latent_acts >= 0).all()

    def test_crosscoder_forward_synthetic_stacked(self, sample_crosscoder):
        """Test Crosscoder forward pass on synthetic stacked activations."""
        base_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        reconstructed = sample_crosscoder(stacked)

        assert reconstructed.shape == stacked.shape

    def test_crosscoder_training_step_synthetic(self, sample_crosscoder):
        """Test a single training step on synthetic stacked data."""
        sample_crosscoder.train()
        optimizer = torch.optim.Adam(sample_crosscoder.parameters(), lr=1e-4)

        base_acts = torch.randn(BATCH_SIZE * SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(BATCH_SIZE * SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        optimizer.zero_grad()
        reconstructed = sample_crosscoder(stacked)
        loss = torch.nn.functional.mse_loss(reconstructed, stacked)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert torch.isfinite(loss)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_crosscoder_encode_real_activations(self, models, sample_crosscoder):
        """Test Crosscoder encoding on real paired activations from models."""
        base_model, finetuned_model, tokenizer = models

        text = "Testing crosscoder with real activations."
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts).squeeze(0)
        ft_acts = extract_tensor(ft_acts).squeeze(0)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        latent_acts = sample_crosscoder.encode(stacked)

        seq_len = base_acts.shape[0]
        assert latent_acts.shape == (seq_len, DICT_SIZE)
        assert (latent_acts >= 0).all()
        assert torch.isfinite(latent_acts).all()


class TestLatentActivationValidity:
    """Tests to verify encoding produces valid latent activations."""

    def test_sae_latent_sparsity(self, sample_sae):
        """Test that SAE produces sparse latent activations (less than full dict_size)."""
        synthetic_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)

        latent_acts = sample_sae.encode(synthetic_diffs)

        nonzero_per_token = (latent_acts > 0).sum(dim=1).float()
        mean_nonzero = nonzero_per_token.mean()
        assert mean_nonzero < DICT_SIZE

    def test_crosscoder_latent_sparsity(self, sample_crosscoder):
        """Test that Crosscoder produces sparse latent activations (less than full dict_size)."""
        base_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        latent_acts = sample_crosscoder.encode(stacked)

        nonzero_per_token = (latent_acts > 0).sum(dim=1).float()
        mean_nonzero = nonzero_per_token.mean()
        assert mean_nonzero < DICT_SIZE

    def test_sae_latent_non_negative(self, sample_sae):
        """Test that SAE latent activations are non-negative."""
        synthetic_diffs = torch.randn(100, HIDDEN_DIM, device=DEVICE)

        latent_acts = sample_sae.encode(synthetic_diffs)

        assert (latent_acts >= 0).all()

    def test_crosscoder_latent_non_negative(self, sample_crosscoder):
        """Test that Crosscoder latent activations are non-negative."""
        base_acts = torch.randn(100, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(100, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        latent_acts = sample_crosscoder.encode(stacked)

        assert (latent_acts >= 0).all()

    def test_sae_latent_finite(self, sample_sae):
        """Test that SAE latent activations are finite (no NaN/Inf)."""
        synthetic_diffs = torch.randn(100, HIDDEN_DIM, device=DEVICE)

        latent_acts = sample_sae.encode(synthetic_diffs)

        assert torch.isfinite(latent_acts).all()

    def test_crosscoder_latent_finite(self, sample_crosscoder):
        """Test that Crosscoder latent activations are finite."""
        base_acts = torch.randn(100, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(100, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        latent_acts = sample_crosscoder.encode(stacked)

        assert torch.isfinite(latent_acts).all()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_sae_reconstruction_quality(self, sample_sae):
        """Test that SAE reconstruction is reasonable after encoding."""
        synthetic_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)

        latent_acts = sample_sae.encode(synthetic_diffs)
        reconstructed = sample_sae.decode(latent_acts)

        assert reconstructed.shape == synthetic_diffs.shape
        assert torch.isfinite(reconstructed).all()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_crosscoder_reconstruction_quality(self, sample_crosscoder):
        """Test that Crosscoder reconstruction is reasonable after encoding."""
        base_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        ft_acts = torch.randn(SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        stacked = torch.stack([base_acts, ft_acts], dim=1)

        latent_acts = sample_crosscoder.encode(stacked)
        reconstructed = sample_crosscoder.decode(latent_acts)

        assert reconstructed.shape == stacked.shape
        assert torch.isfinite(reconstructed).all()


class TestEndToEndPipeline:
    """End-to-end tests combining model inference and dictionary encoding."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_full_sae_pipeline(self, models, sample_sae):
        """
        Test full pipeline: model inference -> activation diff -> SAE encode.

        This mirrors the actual usage in SAEDifferenceMethod.
        """
        base_model, finetuned_model, tokenizer = models

        texts = ["Hello, how are you?", "What is machine learning?"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts)
        ft_acts = extract_tensor(ft_acts)

        diffs = ft_acts - base_acts
        batch_size, seq_len, hidden_dim = diffs.shape
        diffs_flat = diffs.view(-1, hidden_dim)

        latent_acts = sample_sae.encode(diffs_flat)

        assert latent_acts.shape == (batch_size * seq_len, DICT_SIZE)
        assert (latent_acts >= 0).all()
        assert torch.isfinite(latent_acts).all()

        max_acts_per_token = latent_acts.max(dim=1).values
        assert max_acts_per_token.shape == (batch_size * seq_len,)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_full_crosscoder_pipeline(self, models, sample_crosscoder):
        """
        Test full pipeline: model inference -> stack activations -> Crosscoder encode.

        This mirrors the actual usage in CrosscoderDiffingMethod.
        """
        base_model, finetuned_model, tokenizer = models

        texts = ["The weather is nice today.", "I like programming."]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        layer_idx = 6

        with torch.no_grad():
            with base_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                base_acts = base_model.layers_output[layer_idx].save()

            with finetuned_model.trace(
                dict(input_ids=input_ids, attention_mask=attention_mask)
            ):
                ft_acts = finetuned_model.layers_output[layer_idx].save()

        base_acts = extract_tensor(base_acts)
        ft_acts = extract_tensor(ft_acts)

        batch_size, seq_len, hidden_dim = base_acts.shape
        stacked = torch.stack([base_acts, ft_acts], dim=2)
        assert stacked.shape == (batch_size, seq_len, 2, hidden_dim)

        stacked_flat = stacked.view(-1, 2, hidden_dim)

        latent_acts = sample_crosscoder.encode(stacked_flat)

        assert latent_acts.shape == (batch_size * seq_len, DICT_SIZE)
        assert (latent_acts >= 0).all()
        assert torch.isfinite(latent_acts).all()

        max_acts_per_token = latent_acts.max(dim=1).values
        assert max_acts_per_token.shape == (batch_size * seq_len,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
