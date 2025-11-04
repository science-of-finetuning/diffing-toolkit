"""
Tests for the latent activations functions.

Tests cover the get_positive_activations function with various scenarios
including different activation patterns, sequence lengths, and latent configurations.
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.utils.dictionary.latent_activations import get_positive_activations
from src.utils.cache import SampleCache


class MockDictionaryModel:
    """Mock dictionary model for testing."""

    def __init__(self, dict_size, activation_dim, device="cpu", dtype=torch.float32):
        self.dict_size = dict_size
        self.activation_dim = activation_dim
        self.device = device
        self.dtype = dtype

    def get_activations(self, activations):
        """
        Mock get_activations that returns predictable patterns for testing.

        Args:
            activations: Tensor of shape (seq_len, activation_dim)

        Returns:
            Tensor of shape (seq_len, dict_size) with latent activations
        """
        seq_len = activations.shape[0]
        # Create a predictable pattern: make some activations positive based on position
        latent_acts = torch.zeros(
            seq_len, self.dict_size, device=activations.device, dtype=self.dtype
        )

        # Pattern: activation i activates latent (i % dict_size) with value (i + 1) / 10
        for i in range(seq_len):
            latent_idx = i % self.dict_size
            latent_acts[i, latent_idx] = (i + 1) / 10.0

        return latent_acts


class MockSampleCache:
    """Mock SampleCache for testing."""

    def __init__(self, sequences_data, activation_dim=32, device="cpu"):
        """
        Args:
            sequences_data: List of tuples (tokens, seq_length) defining each sequence
            activation_dim: Dimension of activation vectors
            device: Device for tensors
        """
        self.sequences_data = sequences_data
        self.activation_dim = activation_dim
        self.device = device

    def __len__(self):
        return len(self.sequences_data)

    def __getitem__(self, index):
        tokens, seq_length = self.sequences_data[index]

        # Create mock activations - each position gets a different activation vector
        activations = torch.randn(seq_length, self.activation_dim, device=self.device)

        return tokens, activations


class TestGetPositiveActivations:
    """Test suite for get_positive_activations function."""

    def test_basic_functionality_single_sequence(self):
        """Test basic functionality with a single sequence."""
        # Setup
        dict_size = 4
        activation_dim = 8
        seq_length = 6

        mock_cache = MockSampleCache(
            [(torch.tensor([1, 2, 3, 4, 5, 6]), seq_length)],
            activation_dim=activation_dim,
        )

        mock_model = MockDictionaryModel(
            dict_size, activation_dim, device="cpu", dtype=torch.float32
        )
        latent_ids = torch.arange(dict_size)

        # Execute
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify shapes
        assert (
            out_acts.shape[0] == out_ids.shape[0]
        ), "Activations and indices should have same length"
        assert (
            out_ids.shape[1] == 3
        ), "Indices should have format (seq_idx, seq_pos, feature_pos)"
        assert (
            max_acts.shape[0] == dict_size
        ), "Max activations should have one value per latent"
        assert len(seq_ranges) == 2, "Should have start and end of sequence ranges"

        # Verify all activations are positive
        assert torch.all(out_acts > 0), "All output activations should be positive"

        # Verify sequence ranges
        assert seq_ranges[0] == 0, "First range should start at 0"
        assert seq_ranges[1] == len(
            out_acts
        ), "Last range should end at total activations"

    def test_multiple_sequences(self):
        """Test with multiple sequences of different lengths."""
        dict_size = 3
        activation_dim = 6

        # Create sequences of different lengths
        sequences = [
            (torch.tensor([1, 2, 3]), 3),  # Sequence 0: length 3
            (torch.tensor([4, 5]), 2),  # Sequence 1: length 2
            (torch.tensor([6, 7, 8, 9]), 4),  # Sequence 2: length 4
        ]

        mock_cache = MockSampleCache(sequences, activation_dim=activation_dim)
        mock_model = MockDictionaryModel(
            dict_size, activation_dim, device="cpu", dtype=torch.float32
        )
        latent_ids = torch.arange(dict_size)

        # Execute
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify basic properties
        assert out_acts.shape[0] == out_ids.shape[0]
        assert torch.all(out_acts > 0)
        assert max_acts.shape[0] == dict_size
        assert len(seq_ranges) == len(sequences) + 1

        # Verify sequence indices are correct
        unique_seq_indices = torch.unique(out_ids[:, 0])
        assert len(unique_seq_indices) == len(
            sequences
        ), "Should have indices for all sequences"
        assert torch.min(unique_seq_indices) == 0, "Sequence indices should start at 0"
        assert (
            torch.max(unique_seq_indices) == len(sequences) - 1
        ), "Sequence indices should be contiguous"

        # Verify sequence ranges are monotonic
        assert seq_ranges[0] == 0
        for i in range(len(seq_ranges) - 1):
            assert (
                seq_ranges[i] <= seq_ranges[i + 1]
            ), "Sequence ranges should be monotonic"

    def test_max_activations_tracking(self):
        """Test that maximum activations are correctly tracked across sequences."""
        dict_size = 2
        activation_dim = 4

        # Create a custom mock model that returns known activation patterns
        class TestDictionaryModel:
            def __init__(self):
                self.dict_size = dict_size
                self.device = "cpu"
                self.dtype = torch.float32

            def get_activations(self, activations):
                seq_len = activations.shape[0]
                latent_acts = torch.zeros(
                    seq_len, dict_size, device=activations.device, dtype=self.dtype
                )

                # First sequence: latent 0 gets max value 0.8, latent 1 gets max value 0.3
                if seq_len == 3:  # First sequence
                    latent_acts[0, 0] = (
                        0.8  # This should be the global max for latent 0
                    )
                    latent_acts[1, 1] = 0.3
                    latent_acts[2, 0] = 0.2

                # Second sequence: latent 0 gets max value 0.5, latent 1 gets max value 0.9
                elif seq_len == 2:  # Second sequence
                    latent_acts[0, 0] = 0.5
                    latent_acts[1, 1] = (
                        0.9  # This should be the global max for latent 1
                    )

                return latent_acts

        sequences = [
            (torch.tensor([1, 2, 3]), 3),
            (torch.tensor([4, 5]), 2),
        ]

        mock_cache = MockSampleCache(sequences, activation_dim=activation_dim)
        mock_model = TestDictionaryModel()
        latent_ids = torch.arange(dict_size)

        # Execute
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify max activations
        assert torch.allclose(
            max_acts[0], torch.tensor(0.8)
        ), "Max for latent 0 should be 0.8"
        assert torch.allclose(
            max_acts[1], torch.tensor(0.9)
        ), "Max for latent 1 should be 0.9"

    def test_empty_sequence_handling(self):
        """Test handling of sequences with no positive activations."""
        dict_size = 2
        activation_dim = 4

        # Create a model that returns zero/negative activations
        class ZeroActivationModel:
            def __init__(self):
                self.dict_size = dict_size
                self.device = "cpu"
                self.dtype = torch.float32

            def get_activations(self, activations):
                seq_len = activations.shape[0]
                # Return all zeros (no positive activations)
                return torch.zeros(
                    seq_len, dict_size, device=activations.device, dtype=self.dtype
                )

        sequences = [
            (torch.tensor([1, 2]), 2),
        ]

        mock_cache = MockSampleCache(sequences, activation_dim=activation_dim)
        mock_model = ZeroActivationModel()
        latent_ids = torch.arange(dict_size)

        # Execute
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify empty results
        assert len(out_acts) == 0, "Should have no positive activations"
        assert len(out_ids) == 0, "Should have no activation indices"
        assert torch.allclose(
            max_acts, torch.zeros(dict_size, device=max_acts.device)
        ), "Max activations should be zero"
        assert seq_ranges == [0, 0], "Sequence ranges should show no activations"

    def test_latent_subset_selection(self):
        """Test functionality when using a subset of latent features."""
        dict_size = 6
        activation_dim = 8
        selected_latents = torch.tensor([1, 3, 5])  # Select only these latent indices

        mock_cache = MockSampleCache(
            [(torch.tensor([1, 2, 3]), 3)], activation_dim=activation_dim
        )

        mock_model = MockDictionaryModel(
            dict_size, activation_dim, device="cpu", dtype=torch.float32
        )

        # Execute with subset of latents
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, selected_latents
        )

        # Verify that feature positions in out_ids correspond to selected latents
        feature_positions = torch.unique(out_ids[:, 2])
        assert len(feature_positions) <= len(
            selected_latents
        ), "Feature positions should be within selected range"
        assert torch.all(
            feature_positions < len(selected_latents)
        ), "Feature positions should be indices into selected latents"

        # Verify max_acts has correct size
        assert max_acts.shape[0] == len(
            selected_latents
        ), "Max activations should match number of selected latents"

    def test_device_consistency(self):
        """Test that function handles different device configurations correctly."""
        dict_size = 2
        activation_dim = 4

        sequences = [
            (torch.tensor([1, 2]), 2),
        ]

        mock_cache = MockSampleCache(
            sequences, activation_dim=activation_dim, device="cpu"
        )
        mock_model = MockDictionaryModel(
            dict_size, activation_dim, device="cpu", dtype=torch.float32
        )
        latent_ids = torch.arange(dict_size)

        # Execute - should handle device transfer internally
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify outputs are on CPU (as specified in the function)
        assert out_acts.device.type == "cpu", "Output activations should be on CPU"
        assert out_ids.device.type == "cpu", "Output indices should be on CPU"
        assert (
            max_acts.device.type == "cpu"
        ), "Max activations should remain on model device"

    def test_sequence_position_tracking(self):
        """Test that sequence positions are correctly tracked in output indices."""
        dict_size = 3
        activation_dim = 4

        # Create sequences where we can predict the activation pattern
        class PredictableModel:
            def __init__(self):
                self.dict_size = dict_size
                self.device = "cpu"
                self.dtype = torch.float32

            def get_activations(self, activations):
                seq_len = activations.shape[0]
                latent_acts = torch.zeros(
                    seq_len, dict_size, device=activations.device, dtype=self.dtype
                )

                # Make position 0 activate latent 0, position 1 activate latent 1, etc.
                for i in range(seq_len):
                    latent_acts[i, i % dict_size] = 0.5

                return latent_acts

        sequences = [
            (torch.tensor([1, 2, 3, 4]), 4),  # 4 positions
        ]

        mock_cache = MockSampleCache(sequences, activation_dim=activation_dim)
        mock_model = PredictableModel()
        latent_ids = torch.arange(dict_size)

        # Execute
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify sequence positions
        seq_positions = out_ids[:, 1]  # Extract sequence positions
        expected_positions = torch.tensor(
            [0, 1, 2, 3]
        )  # Should have positions 0, 1, 2, 3

        assert torch.allclose(
            torch.sort(seq_positions)[0], expected_positions
        ), "Sequence positions should match expected pattern"

    def test_activation_value_consistency(self):
        """Test that activation values in output match the model's predictions."""
        dict_size = 2
        activation_dim = 4

        # Create a model with known activation values
        class KnownValueModel:
            def __init__(self):
                self.dict_size = dict_size
                self.device = "cpu"
                self.dtype = torch.float32

            def get_activations(self, activations):
                seq_len = activations.shape[0]
                latent_acts = torch.zeros(
                    seq_len, dict_size, device=activations.device, dtype=self.dtype
                )

                # Set specific known values
                latent_acts[0, 0] = 0.7
                latent_acts[1, 1] = 0.3
                latent_acts[0, 1] = 0.1

                return latent_acts

        sequences = [
            (torch.tensor([1, 2]), 2),
        ]

        mock_cache = MockSampleCache(sequences, activation_dim=activation_dim)
        mock_model = KnownValueModel()
        latent_ids = torch.arange(dict_size)

        # Execute
        out_acts, out_ids, seq_ranges, max_acts = get_positive_activations(
            mock_cache, mock_model, latent_ids
        )

        # Verify activation values (should be 0.7, 0.3, 0.1 in some order)
        expected_values = torch.tensor([0.7, 0.1, 0.3])
        sorted_acts = torch.sort(out_acts)[0]
        sorted_expected = torch.sort(expected_values)[0]

        assert torch.allclose(
            sorted_acts, sorted_expected, atol=1e-6
        ), "Activation values should match model predictions"


if __name__ == "__main__":
    # Run basic functionality test if script is executed directly
    test_instance = TestGetPositiveActivations()
    test_instance.test_basic_functionality_single_sequence()
    test_instance.test_multiple_sequences()
    print("Basic tests passed!")
