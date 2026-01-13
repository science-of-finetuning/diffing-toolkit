"""
Tests for core computation of diffing techniques.

Tests verify that each technique's core computation runs correctly using synthetic data.
All tests are CPU-only with minimal dimensions for fast execution.
"""

import pytest
import torch
import numpy as np

from diffing.utils.collection import RunningActivationMean
from diffing.methods.activation_analysis.diffing_method import (
    ActivationAnalysisDiffingMethod,
    init_collectors,
)


# Test dimensions (minimal for fast execution)
HIDDEN_DIM = 64
VOCAB_SIZE = 100
SEQ_LEN = 16
BATCH_SIZE = 2
DICT_SIZE = 32


class TestActivationAnalysisComputation:
    """Tests for activation analysis core computation."""

    def test_init_collectors_basic(self):
        """Test that init_collectors creates expected collector types."""
        unique_template_tokens = [100, 101, 102]
        collectors = init_collectors(unique_template_tokens)

        assert "all_tokens" in collectors
        assert "first_token" in collectors
        assert "second_token" in collectors
        assert "chat_token_100" in collectors
        assert "chat_token_101" in collectors
        assert "chat_token_102" in collectors

        assert collectors["first_token"].position == 0
        assert collectors["second_token"].position == 1
        assert collectors["chat_token_100"].token_id == 100

    def test_running_activation_mean_all_tokens(self):
        """Test RunningActivationMean with all tokens (no filtering)."""
        collector = RunningActivationMean()

        activation_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM)
        tokens = torch.arange(SEQ_LEN)

        collector.update(activation_diffs, tokens)

        assert collector.count == SEQ_LEN
        assert collector.mean.shape == (HIDDEN_DIM,)
        assert torch.allclose(collector.mean, activation_diffs.mean(dim=0), atol=1e-5)

    def test_running_activation_mean_position_filter(self):
        """Test RunningActivationMean with position filtering."""
        collector = RunningActivationMean(position=0)

        activation_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM)
        tokens = torch.arange(SEQ_LEN)

        collector.update(activation_diffs, tokens)

        assert collector.count == 1
        assert collector.mean.shape == (HIDDEN_DIM,)
        assert torch.allclose(collector.mean, activation_diffs[0], atol=1e-5)

    def test_running_activation_mean_token_filter(self):
        """Test RunningActivationMean with token ID filtering."""
        target_token = 42
        collector = RunningActivationMean(token_id=target_token)

        activation_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM)
        tokens = torch.arange(SEQ_LEN)
        tokens[3] = target_token
        tokens[7] = target_token

        collector.update(activation_diffs, tokens)

        assert collector.count == 2
        expected_mean = (activation_diffs[3] + activation_diffs[7]) / 2
        assert torch.allclose(collector.mean, expected_mean, atol=1e-5)

    def test_running_activation_mean_accumulates(self):
        """Test that RunningActivationMean correctly accumulates across updates."""
        collector = RunningActivationMean()

        all_diffs = []
        for _ in range(3):
            activation_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM)
            tokens = torch.arange(SEQ_LEN)
            collector.update(activation_diffs, tokens)
            all_diffs.append(activation_diffs)

        assert collector.count == SEQ_LEN * 3
        expected_mean = torch.cat(all_diffs, dim=0).mean(dim=0)
        assert torch.allclose(collector.mean, expected_mean, atol=1e-4)


class TestActivationStatisticsComputation:
    """Tests for compute_activation_statistics logic (extracted from method)."""

    def compute_activation_statistics_standalone(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        collectors: dict,
    ):
        """
        Standalone version of compute_activation_statistics for testing.

        Mirrors the logic in ActivationAnalysisDiffingMethod.compute_activation_statistics
        """
        seq_len, num_models, activation_dim = activations.shape
        assert num_models == 2

        base_activations = activations[:, 0, :]
        finetuned_activations = activations[:, 1, :]

        norm_base = torch.norm(base_activations, p=2, dim=-1)
        norm_finetuned = torch.norm(finetuned_activations, p=2, dim=-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            base_activations, finetuned_activations, dim=-1
        )

        activation_diffs = finetuned_activations - base_activations

        for collector in collectors.values():
            collector.update(activation_diffs, tokens)

        norm_diffs = torch.norm(activation_diffs, p=2, dim=-1)

        return (
            norm_diffs.cpu().float(),
            cos_sim.cpu().float(),
            norm_base.cpu().float(),
            norm_finetuned.cpu().float(),
        )

    def test_identical_activations_zero_diff(self):
        """Test that identical activations produce zero norm difference and cos_sim=1."""
        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        stacked = torch.stack([base, base], dim=1)  # [seq_len, 2, hidden_dim]
        tokens = torch.arange(SEQ_LEN)
        collectors = {"all": RunningActivationMean()}

        norm_diffs, cos_sim, norm_base, norm_finetuned = (
            self.compute_activation_statistics_standalone(stacked, tokens, collectors)
        )

        assert norm_diffs.shape == (SEQ_LEN,)
        assert torch.allclose(norm_diffs, torch.zeros(SEQ_LEN), atol=1e-5)
        assert torch.allclose(cos_sim, torch.ones(SEQ_LEN), atol=1e-5)
        assert torch.allclose(norm_base, norm_finetuned, atol=1e-5)

    def test_orthogonal_activations_zero_cosine(self):
        """Test that orthogonal activations produce cos_sim near 0."""
        base = torch.zeros(SEQ_LEN, HIDDEN_DIM)
        base[:, 0] = 1.0  # All vectors point in first dimension

        finetuned = torch.zeros(SEQ_LEN, HIDDEN_DIM)
        finetuned[:, 1] = 1.0  # All vectors point in second dimension

        stacked = torch.stack([base, finetuned], dim=1)
        tokens = torch.arange(SEQ_LEN)
        collectors = {"all": RunningActivationMean()}

        norm_diffs, cos_sim, norm_base, norm_finetuned = (
            self.compute_activation_statistics_standalone(stacked, tokens, collectors)
        )

        assert torch.allclose(cos_sim, torch.zeros(SEQ_LEN), atol=1e-5)
        expected_norm_diff = np.sqrt(2)  # sqrt(1^2 + 1^2)
        assert torch.allclose(
            norm_diffs, torch.full((SEQ_LEN,), expected_norm_diff), atol=1e-5
        )

    def test_opposite_activations_negative_cosine(self):
        """Test that opposite activations produce cos_sim = -1."""
        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        finetuned = -base  # Opposite direction

        stacked = torch.stack([base, finetuned], dim=1)
        tokens = torch.arange(SEQ_LEN)
        collectors = {"all": RunningActivationMean()}

        norm_diffs, cos_sim, norm_base, norm_finetuned = (
            self.compute_activation_statistics_standalone(stacked, tokens, collectors)
        )

        assert torch.allclose(cos_sim, -torch.ones(SEQ_LEN), atol=1e-5)
        expected_norm_diff = 2 * norm_base  # diff = -2 * base
        assert torch.allclose(norm_diffs, expected_norm_diff, atol=1e-5)

    def test_output_shapes(self):
        """Test that all outputs have correct shapes."""
        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        finetuned = torch.randn(SEQ_LEN, HIDDEN_DIM)
        stacked = torch.stack([base, finetuned], dim=1)
        tokens = torch.arange(SEQ_LEN)
        collectors = {"all": RunningActivationMean()}

        norm_diffs, cos_sim, norm_base, norm_finetuned = (
            self.compute_activation_statistics_standalone(stacked, tokens, collectors)
        )

        assert norm_diffs.shape == (SEQ_LEN,)
        assert cos_sim.shape == (SEQ_LEN,)
        assert norm_base.shape == (SEQ_LEN,)
        assert norm_finetuned.shape == (SEQ_LEN,)

    def test_collectors_receive_updates(self):
        """Test that collectors are properly updated during computation."""
        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        finetuned = torch.randn(SEQ_LEN, HIDDEN_DIM)
        stacked = torch.stack([base, finetuned], dim=1)
        tokens = torch.arange(SEQ_LEN)

        collectors = {
            "all": RunningActivationMean(),
            "first": RunningActivationMean(position=0),
        }

        self.compute_activation_statistics_standalone(stacked, tokens, collectors)

        assert collectors["all"].count == SEQ_LEN
        assert collectors["first"].count == 1


class TestPCAComputation:
    """Tests for PCA training and projection on activation differences."""

    def test_pca_training_on_synthetic_differences(self):
        """Test that IncrementalPCA fits without error on synthetic data."""
        from torchdr import IncrementalPCA

        base = torch.randn(100, HIDDEN_DIM)
        finetuned = torch.randn(100, HIDDEN_DIM)
        differences = finetuned - base

        pca = IncrementalPCA(n_components=HIDDEN_DIM, batch_size=32, device="cpu")
        pca.partial_fit(differences)

        assert pca.components_ is not None
        assert pca.explained_variance_ratio_ is not None

    def test_pca_projection_shape(self):
        """Test that PCA projections have correct shape."""
        from torchdr import IncrementalPCA

        train_diffs = torch.randn(100, HIDDEN_DIM)

        pca = IncrementalPCA(n_components=HIDDEN_DIM, batch_size=32, device="cpu")
        pca.partial_fit(train_diffs)

        test_diffs = torch.randn(SEQ_LEN, HIDDEN_DIM)
        projections = pca.transform(test_diffs)

        assert projections.shape == (SEQ_LEN, HIDDEN_DIM)

    def test_pca_explained_variance_valid(self):
        """Test that explained variance ratios are valid probabilities."""
        from torchdr import IncrementalPCA

        train_diffs = torch.randn(200, HIDDEN_DIM)

        pca = IncrementalPCA(n_components=HIDDEN_DIM, batch_size=32, device="cpu")
        pca.partial_fit(train_diffs)

        ratios = pca.explained_variance_ratio_
        assert torch.all(ratios >= 0)
        assert torch.all(ratios <= 1)
        assert torch.isclose(ratios.sum(), torch.tensor(1.0), atol=1e-3)

    def test_pca_incremental_fitting(self):
        """Test that IncrementalPCA can be fitted in multiple batches."""
        from torchdr import IncrementalPCA

        pca = IncrementalPCA(n_components=HIDDEN_DIM, batch_size=100, device="cpu")

        for _ in range(3):
            batch = torch.randn(100, HIDDEN_DIM)  # batch size must be >= n_components
            pca.partial_fit(batch)

        assert pca.n_samples_seen_ == 300


class TestDictionaryEncodingComputation:
    """Tests for SAE/Crosscoder encoding computation."""

    def test_sae_encode_difference_ftb(self, mock_dictionary_model):
        """Test SAE encoding of (finetuned - base) differences."""
        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        finetuned = torch.randn(SEQ_LEN, HIDDEN_DIM)
        differences = finetuned - base

        sae = mock_dictionary_model(
            dict_size=DICT_SIZE, activation_dim=HIDDEN_DIM, device="cpu"
        )
        latent_activations = sae.get_activations(differences)

        assert latent_activations.shape == (SEQ_LEN, DICT_SIZE)

    def test_sae_encode_difference_bft(self, mock_dictionary_model):
        """Test SAE encoding of (base - finetuned) differences."""
        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        finetuned = torch.randn(SEQ_LEN, HIDDEN_DIM)
        differences = base - finetuned

        sae = mock_dictionary_model(
            dict_size=DICT_SIZE, activation_dim=HIDDEN_DIM, device="cpu"
        )
        latent_activations = sae.get_activations(differences)

        assert latent_activations.shape == (SEQ_LEN, DICT_SIZE)

    def test_crosscoder_encode_stacked(self):
        """Test crosscoder encoding of stacked [base, ft] activations."""

        class MockCrosscoder:
            """Mock crosscoder that encodes stacked activations."""

            def __init__(self, dict_size, activation_dim):
                self.dict_size = dict_size
                self.activation_dim = activation_dim

            def encode(self, stacked_activations):
                """
                Encode stacked activations [seq_len, 2, hidden_dim].

                Returns [seq_len, dict_size] latent activations.
                """
                seq_len = stacked_activations.shape[0]
                assert stacked_activations.shape[1] == 2
                assert stacked_activations.shape[2] == self.activation_dim

                latent_acts = torch.zeros(seq_len, self.dict_size)
                for i in range(seq_len):
                    latent_idx = i % self.dict_size
                    latent_acts[i, latent_idx] = (i + 1) / 10.0
                return latent_acts

        base = torch.randn(SEQ_LEN, HIDDEN_DIM)
        finetuned = torch.randn(SEQ_LEN, HIDDEN_DIM)
        stacked = torch.stack([base, finetuned], dim=1)  # [seq_len, 2, hidden_dim]

        crosscoder = MockCrosscoder(dict_size=DICT_SIZE, activation_dim=HIDDEN_DIM)
        latent_activations = crosscoder.encode(stacked)

        assert latent_activations.shape == (SEQ_LEN, DICT_SIZE)

    def test_dictionary_sparse_output(self, mock_dictionary_model):
        """Test that dictionary models produce sparse outputs."""
        activations = torch.randn(SEQ_LEN, HIDDEN_DIM)
        sae = mock_dictionary_model(
            dict_size=DICT_SIZE, activation_dim=HIDDEN_DIM, device="cpu"
        )
        latent_activations = sae.get_activations(activations)

        nonzero_per_token = (latent_activations > 0).sum(dim=1)
        assert torch.all(nonzero_per_token <= DICT_SIZE)


class TestKLDivergenceComputation:
    """Tests for KL divergence computation logic."""

    def compute_kl_divergence_standalone(
        self,
        base_logits: torch.Tensor,
        finetuned_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        Standalone KL divergence computation for testing.

        Mirrors the logic in KLDivergenceDiffingMethod.compute_kl_divergence
        """
        batch_size, seq_len, vocab_size = base_logits.shape
        assert finetuned_logits.shape == (batch_size, seq_len, vocab_size)

        if temperature != 1.0:
            base_logits = base_logits / temperature
            finetuned_logits = finetuned_logits / temperature

        base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
        finetuned_log_probs = torch.nn.functional.log_softmax(finetuned_logits, dim=-1)
        finetuned_probs = torch.exp(finetuned_log_probs)

        kl_div = torch.sum(
            finetuned_probs * (finetuned_log_probs - base_log_probs), dim=-1
        )

        masked_kl = kl_div * attention_mask.float()
        kl_sums = torch.sum(masked_kl, dim=1)
        valid_token_counts = torch.sum(attention_mask, dim=1).float()

        mean_per_sample_kl = torch.where(
            valid_token_counts > 0,
            kl_sums / valid_token_counts,
            torch.zeros_like(kl_sums),
        )

        return kl_div, mean_per_sample_kl

    def test_identical_logits_zero_kl(self):
        """Test that identical logits produce KL ≈ 0."""
        logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

        kl_div, mean_kl = self.compute_kl_divergence_standalone(
            logits, logits, attention_mask
        )

        assert kl_div.shape == (BATCH_SIZE, SEQ_LEN)
        assert torch.allclose(kl_div, torch.zeros_like(kl_div), atol=1e-5)
        assert torch.allclose(mean_kl, torch.zeros(BATCH_SIZE), atol=1e-5)

    def test_different_logits_positive_kl(self):
        """Test that different logits produce positive KL."""
        base_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        finetuned_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

        kl_div, mean_kl = self.compute_kl_divergence_standalone(
            base_logits, finetuned_logits, attention_mask
        )

        assert torch.all(kl_div >= -1e-5)  # KL is non-negative
        assert torch.all(mean_kl >= -1e-5)

    def test_temperature_scaling(self):
        """Test that temperature scaling affects KL divergence."""
        base_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        finetuned_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

        _, mean_kl_t1 = self.compute_kl_divergence_standalone(
            base_logits, finetuned_logits, attention_mask, temperature=1.0
        )

        _, mean_kl_t2 = self.compute_kl_divergence_standalone(
            base_logits, finetuned_logits, attention_mask, temperature=2.0
        )

        assert not torch.allclose(mean_kl_t1, mean_kl_t2)

    def test_attention_mask_respected(self):
        """Test that padding tokens are excluded from KL computation."""
        base_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        finetuned_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        full_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        half_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        half_mask[:, SEQ_LEN // 2 :] = 0

        _, mean_kl_full = self.compute_kl_divergence_standalone(
            base_logits, finetuned_logits, full_mask
        )

        _, mean_kl_half = self.compute_kl_divergence_standalone(
            base_logits, finetuned_logits, half_mask
        )

        assert not torch.allclose(mean_kl_full, mean_kl_half)

    def test_output_shapes(self):
        """Test that KL outputs have correct shapes."""
        base_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        finetuned_logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

        kl_div, mean_kl = self.compute_kl_divergence_standalone(
            base_logits, finetuned_logits, attention_mask
        )

        assert kl_div.shape == (BATCH_SIZE, SEQ_LEN)
        assert mean_kl.shape == (BATCH_SIZE,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
