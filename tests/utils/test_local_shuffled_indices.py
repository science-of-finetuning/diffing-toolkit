"""
Tests for the get_local_shuffled_indices function.

Tests cover basic functionality, edge cases, and error conditions.
"""

import pytest
import torch
from collections import Counter
import sys
from pathlib import Path


from diffing.utils.activations import get_local_shuffled_indices


class TestGetLocalShuffledIndices:
    """Test suite for get_local_shuffled_indices function."""

    def test_basic_functionality_single_dataset(self):
        """Test basic functionality with a single dataset."""
        num_samples_per_dataset = [100]
        shard_size = 10
        epochs = 1

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Check output shape
        expected_length = sum(num_samples_per_dataset) * epochs
        assert indices.shape[0] == expected_length

        # Check all indices are present and unique
        unique_indices = torch.unique(indices)
        assert len(unique_indices) == expected_length
        assert torch.min(indices) >= 0
        assert torch.max(indices) < sum(num_samples_per_dataset)

    def test_basic_functionality_multiple_datasets(self):
        """Test basic functionality with multiple datasets."""
        num_samples_per_dataset = [50, 30, 20]
        shard_size = 10
        epochs = 1

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Check output shape
        expected_length = sum(num_samples_per_dataset) * epochs
        assert indices.shape[0] == expected_length

        # Check all indices are present and unique
        unique_indices = torch.unique(indices)
        assert len(unique_indices) == expected_length
        assert torch.min(indices) >= 0
        assert torch.max(indices) < sum(num_samples_per_dataset)

    def test_multiple_epochs(self):
        """Test that multiple epochs generate correct number of indices."""
        num_samples_per_dataset = [40, 60]
        shard_size = 20
        epochs = 3

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Check output shape
        total_samples = sum(num_samples_per_dataset)
        expected_length = total_samples * epochs
        assert indices.shape[0] == expected_length

        # Each epoch should contain all indices exactly once
        epoch_size = total_samples
        for epoch in range(epochs):
            start_idx = epoch * epoch_size
            end_idx = (epoch + 1) * epoch_size
            epoch_indices = indices[start_idx:end_idx]

            # Check this epoch has all unique indices
            unique_epoch_indices = torch.unique(epoch_indices)
            assert len(unique_epoch_indices) == epoch_size
            assert torch.min(epoch_indices) >= 0
            assert torch.max(epoch_indices) < total_samples

    def test_shard_boundary_handling(self):
        """Test handling when total samples don't divide evenly into shards."""
        num_samples_per_dataset = [
            23,
            17,
        ]  # Total: 40, doesn't divide evenly by shard_size
        shard_size = 7
        epochs = 1

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Should still have all indices
        expected_length = sum(num_samples_per_dataset)
        assert indices.shape[0] == expected_length

        unique_indices = torch.unique(indices)
        assert len(unique_indices) == expected_length

    def test_unequal_dataset_sizes(self):
        """Test with datasets of very different sizes."""
        num_samples_per_dataset = [10, 90]  # Very unequal
        shard_size = 20
        epochs = 1

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Check all indices are present
        expected_length = sum(num_samples_per_dataset)
        assert indices.shape[0] == expected_length

        # Check proper dataset representation
        dataset_0_indices = indices[indices < 10]
        dataset_1_indices = indices[indices >= 10]

        assert len(dataset_0_indices) == 10
        assert len(dataset_1_indices) == 90

    def test_small_shard_size(self):
        """Test with very small shard size."""
        num_samples_per_dataset = [30, 20]
        shard_size = 2
        epochs = 1

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Should still work correctly
        expected_length = sum(num_samples_per_dataset)
        assert indices.shape[0] == expected_length

        unique_indices = torch.unique(indices)
        assert len(unique_indices) == expected_length

    def test_large_shard_size(self):
        """Test with shard size larger than total samples."""
        num_samples_per_dataset = [15, 25]
        shard_size = 100  # Larger than total samples (40)
        epochs = 1

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Should still work correctly
        expected_length = sum(num_samples_per_dataset)
        assert indices.shape[0] == expected_length

        unique_indices = torch.unique(indices)
        assert len(unique_indices) == expected_length

    def test_three_datasets(self):
        """Test with three datasets to ensure generalization."""
        num_samples_per_dataset = [20, 30, 10]
        shard_size = 6
        epochs = 2

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        epoch_size = sum(num_samples_per_dataset)
        expected_length = epoch_size * epochs
        assert indices.shape[0] == expected_length

        # Check each epoch
        for epoch in range(epochs):
            start_idx = epoch * epoch_size
            end_idx = (epoch + 1) * epoch_size
            epoch_indices = indices[start_idx:end_idx]
            print("h", epoch_nums[start_idx:end_idx], start_idx, end_idx)
            assert torch.all(epoch_nums[start_idx:end_idx] == epoch)

            # Verify all datasets are represented in each epoch
            dataset_0_count = torch.sum(epoch_indices < 20)
            dataset_1_count = torch.sum((epoch_indices >= 20) & (epoch_indices < 50))
            dataset_2_count = torch.sum(epoch_indices >= 50)

            assert dataset_0_count == 20
            assert dataset_1_count == 30
            assert dataset_2_count == 10

    def test_error_conditions(self):
        """Test various error conditions."""

        # Empty dataset list
        with pytest.raises(AssertionError, match="Must have at least one dataset"):
            get_local_shuffled_indices([], 10, 1)

        # Zero samples in dataset
        with pytest.raises(AssertionError, match="All dataset sizes must be positive"):
            get_local_shuffled_indices([10, 0, 5], 10, 1)

        # Negative samples
        with pytest.raises(AssertionError, match="All dataset sizes must be positive"):
            get_local_shuffled_indices([10, -5, 5], 10, 1)

        # Zero shard size
        with pytest.raises(AssertionError, match="shard_size must be positive"):
            get_local_shuffled_indices([10, 20], 0, 1)

        # Negative shard size
        with pytest.raises(AssertionError, match="shard_size must be positive"):
            get_local_shuffled_indices([10, 20], -5, 1)

        # Zero epochs
        with pytest.raises(AssertionError, match="epochs must be positive"):
            get_local_shuffled_indices([10, 20], 10, 0)

        # Negative epochs
        with pytest.raises(AssertionError, match="epochs must be positive"):
            get_local_shuffled_indices([10, 20], 10, -1)

    def test_deterministic_behavior_with_seed(self):
        """Test that function is deterministic when torch random seed is set."""
        num_samples_per_dataset = [30, 20]
        shard_size = 10
        epochs = 1

        # Set seed and generate indices
        torch.manual_seed(42)
        indices1, _ = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Reset seed and generate again
        torch.manual_seed(42)
        indices2, _ = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Should be identical
        assert torch.equal(indices1, indices2)

    def test_shape_assertions(self):
        """Test shape assertions in the function."""
        num_samples_per_dataset = [25, 35]
        shard_size = 12
        epochs = 2

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        # Function should assert correct shape internally
        total_samples = sum(num_samples_per_dataset)
        expected_shape = (total_samples * epochs,)
        assert indices.shape == expected_shape

    def test_index_range_validity(self):
        """Test that all generated indices are within valid range."""
        num_samples_per_dataset = [40, 30, 20]
        shard_size = 15
        epochs = 2

        indices, epoch_nums = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, epochs
        )

        total_samples = sum(num_samples_per_dataset)

        # All indices should be within [0, total_samples)
        assert torch.all(indices >= 0)
        assert torch.all(indices < total_samples)

        # Check that indices map to correct datasets
        dataset_0_mask = indices < 40
        dataset_1_mask = (indices >= 40) & (indices < 70)
        dataset_2_mask = indices >= 70

        # Each epoch should have the correct count for each dataset
        for epoch in range(epochs):
            start_idx = epoch * total_samples
            end_idx = (epoch + 1) * total_samples

            epoch_mask = torch.arange(len(indices)).ge(start_idx) & torch.arange(
                len(indices)
            ).lt(end_idx)

            epoch_dataset_0_count = torch.sum(dataset_0_mask & epoch_mask)
            epoch_dataset_1_count = torch.sum(dataset_1_mask & epoch_mask)
            epoch_dataset_2_count = torch.sum(dataset_2_mask & epoch_mask)

            # Test that within one epoch, no index is repeated
            assert torch.unique(indices[start_idx:end_idx]).shape[0] == len(
                indices[start_idx:end_idx]
            )

            assert epoch_dataset_0_count == 40
            assert epoch_dataset_1_count == 30
            assert epoch_dataset_2_count == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
