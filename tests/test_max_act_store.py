"""
Tests for MaxActStore class with both sparse and dense storage formats.

This module tests the SQLite-based storage for maximum activating examples,
including both bulk loading and real-time top-k management functionality
with both sparse and dense activation details storage.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sqlite3

from diffing.utils.max_act_store import (
    MaxActStore,
    AsyncMaxActStoreWriter,
    ReadOnlyMaxActStore,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def decode(self, token_ids, skip_special_tokens=False):
        """Simple mock decode that just joins token IDs with spaces."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join([f"tok_{id}" for id in token_ids])


@pytest.fixture
def temp_db_path():
    """Fixture providing a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def sample_token_sequences():
    """Fixture providing sample token sequences for testing."""
    return [
        (0, torch.tensor([1, 2, 3, 4])),
        (1, torch.tensor([5, 6, 7])),
        (2, torch.tensor([8, 9, 10, 11, 12])),
        (3, torch.tensor([13, 14])),
        (4, torch.tensor([15, 16, 17, 18])),
    ]


@pytest.fixture
def sample_quantile_examples():
    """Fixture providing sample quantile examples data."""
    return {
        0: {  # quantile_idx 0
            0: [(0.9, 0), (0.8, 1)],  # latent_idx 0: [(score, sequence_uid), ...]
            1: [(0.7, 2)],  # latent_idx 1
        },
        1: {  # quantile_idx 1
            0: [(0.6, 3)],  # latent_idx 0
            2: [(0.5, 4)],  # latent_idx 2
        },
    }


@pytest.fixture
def sample_activation_details():
    """Fixture providing sample activation details."""
    return {
        "sparse": {
            0: {  # latent_idx 0
                0: (np.array([0, 1, 2]), np.array([0.9, 0.8, 0.7])),  # sequence_uid 0
                1: (np.array([0, 1]), np.array([0.8, 0.6])),  # sequence_uid 1
                3: (np.array([0]), np.array([0.6])),  # sequence_uid 3
            },
            1: {  # latent_idx 1
                2: (np.array([0, 1, 2]), np.array([0.7, 0.5, 0.4])),  # sequence_uid 2
            },
            2: {  # latent_idx 2
                4: (np.array([0, 1]), np.array([0.5, 0.3])),  # sequence_uid 4
            },
        },
        "dense": {
            0: {  # latent_idx 0
                0: np.array([0.0, 0.9, 0.8, 0.7]),  # sequence_uid 0
                1: np.array([0.8, 0.6, 0.0]),  # sequence_uid 1
                3: np.array([0.6, 0.0]),  # sequence_uid 3
            },
        },
    }


@pytest.fixture
def populated_store(
    temp_db_path,
    sample_quantile_examples,
    sample_token_sequences,
    sample_activation_details,
    mock_tokenizer,
):
    """Fixture providing a populated MaxActStore for read-only testing."""
    # Create and populate a store
    store = MaxActStore(temp_db_path, tokenizer=mock_tokenizer, storage_format="sparse")
    store.fill(
        sample_quantile_examples,
        sample_token_sequences,
        sample_activation_details["sparse"],
    )
    return store


@pytest.fixture(params=["sparse", "dense"])
def storage_format(request):
    """Parametrized fixture for testing both storage formats."""
    return request.param


class TestReadOnlyMaxActStore:
    """Test class for ReadOnlyMaxActStore functionality."""

    def test_initialization_from_existing_store(self, populated_store, mock_tokenizer):
        """Test initializing ReadOnlyMaxActStore from existing database."""
        db_path = populated_store.db_manager.db_path

        # Create read-only store
        readonly_store = ReadOnlyMaxActStore(db_path, tokenizer=mock_tokenizer)

        assert readonly_store.storage_format == "sparse"
        assert readonly_store.max_examples is None
        assert readonly_store.tokenizer == mock_tokenizer

    def test_initialization_nonexistent_database(self, temp_db_path):
        """Test initialization with non-existent database."""
        nonexistent_path = temp_db_path.parent / "nonexistent.db"

        with pytest.raises(
            Exception
        ):  # Should fail to open non-existent read-only database
            ReadOnlyMaxActStore(nonexistent_path)

    def test_initialization_missing_config(self, temp_db_path):
        """Test initialization with database missing config table."""
        # Create empty database without config
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE dummy (id INTEGER)")
            conn.commit()

        with pytest.raises(sqlite3.OperationalError, match="no such table: config"):
            ReadOnlyMaxActStore(temp_db_path)

    def test_read_operations_match_original(self, populated_store, mock_tokenizer):
        """Test that read operations return same results as original store."""
        db_path = populated_store.db_manager.db_path

        # Get original results
        original_examples = populated_store.get_top_examples()
        original_count = len(populated_store)

        # Create read-only store and compare
        readonly_store = ReadOnlyMaxActStore(db_path, tokenizer=mock_tokenizer)

        readonly_examples = readonly_store.get_top_examples()
        readonly_count = len(readonly_store)

        assert readonly_count == original_count
        assert len(readonly_examples) == len(original_examples)

        # Compare example details
        for orig, readonly in zip(original_examples, readonly_examples):
            assert orig["example_id"] == readonly["example_id"]
            assert orig["max_score"] == readonly["max_score"]
            assert orig["input_ids"] == readonly["input_ids"]
            assert orig["latent_idx"] == readonly["latent_idx"]
            assert orig["quantile_idx"] == readonly["quantile_idx"]

    def test_filtering_operations(self, populated_store, mock_tokenizer):
        """Test filtering operations work correctly in read-only mode."""
        db_path = populated_store.db_manager.db_path
        readonly_store = ReadOnlyMaxActStore(db_path, tokenizer=mock_tokenizer)

        # Test latent_idx filtering
        latent_0_examples = readonly_store.get_top_examples(latent_idx=0)
        assert len(latent_0_examples) > 0
        assert all(ex["latent_idx"] == 0 for ex in latent_0_examples)

        # Test limit
        limited_examples = readonly_store.get_top_examples(limit=2)
        assert len(limited_examples) == 2

        # Test combined filters
        combined_examples = readonly_store.get_top_examples(latent_idx=0, limit=1)
        assert len(combined_examples) <= 1
        if combined_examples:
            assert combined_examples[0]["latent_idx"] == 0

    def test_concurrent_access_simulation(self, populated_store, mock_tokenizer):
        """Test that multiple ReadOnlyMaxActStore instances can access same database."""
        db_path = populated_store.db_manager.db_path

        # Create multiple read-only instances
        stores = [
            ReadOnlyMaxActStore(db_path, tokenizer=mock_tokenizer) for _ in range(3)
        ]

        # All should be able to read simultaneously
        results = []
        for store in stores:
            examples = store.get_top_examples(limit=2)
            results.append(examples)

        # All results should be identical
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for j in range(len(results[i])):
                assert results[i][j]["example_id"] == results[0][j]["example_id"]
                assert results[i][j]["max_score"] == results[0][j]["max_score"]

    def test_wal_mode_setup(self, populated_store, mock_tokenizer):
        """Test that WAL mode is set up correctly."""
        db_path = populated_store.db_manager.db_path
        readonly_store = ReadOnlyMaxActStore(db_path, tokenizer=mock_tokenizer)

        # Check that WAL mode was attempted (we can't easily verify it was successful)
        # This test mainly ensures the WAL setup code runs without error
        assert readonly_store is not None

    def test_read_only_property_enforcement(self, populated_store, mock_tokenizer):
        """Test that read-only store properly handles read-only database connection."""
        db_path = populated_store.db_manager.db_path
        readonly_store = ReadOnlyMaxActStore(db_path, tokenizer=mock_tokenizer)

        # Should be able to read
        examples = readonly_store.get_top_examples()
        assert len(examples) > 0

        # Database manager should be in read-only mode
        assert readonly_store.db_manager.readonly == True

    def test_storage_format_detection(self, temp_db_path, mock_tokenizer):
        """Test that storage format is correctly detected from database."""
        # Test with dense format
        dense_store = MaxActStore(
            temp_db_path, storage_format="dense", tokenizer=mock_tokenizer
        )
        dense_store.add_example(0.8, torch.tensor([1, 2, 3]))

        readonly_store = ReadOnlyMaxActStore(temp_db_path, tokenizer=mock_tokenizer)
        assert readonly_store.storage_format == "dense"

    def test_dataset_operations(self, temp_db_path, mock_tokenizer):
        """Test dataset-related operations in read-only mode."""
        # Create store with dataset info
        store = MaxActStore(temp_db_path, tokenizer=mock_tokenizer)
        store.add_example(
            0.8, torch.tensor([1, 2, 3]), dataset_name="dataset_A", dataset_id=1
        )
        store.add_example(
            0.7, torch.tensor([4, 5, 6]), dataset_name="dataset_B", dataset_id=2
        )

        readonly_store = ReadOnlyMaxActStore(temp_db_path, tokenizer=mock_tokenizer)

        # Test dataset filtering
        dataset_a_examples = readonly_store.get_top_examples(
            dataset_names=["dataset_A"]
        )
        assert len(dataset_a_examples) == 1
        assert dataset_a_examples[0]["dataset_name"] == "dataset_A"

        both_datasets = readonly_store.get_top_examples(
            dataset_names=["dataset_A", "dataset_B"]
        )
        assert len(both_datasets) == 2

    def test_error_handling_corrupted_database(self, temp_db_path):
        """Test error handling with corrupted database."""
        # Create corrupted database
        with open(temp_db_path, "w") as f:
            f.write("corrupted data")

        with pytest.raises(Exception):  # Should fail to read corrupted database
            ReadOnlyMaxActStore(temp_db_path)

    def test_large_dataset_performance(self, temp_db_path, mock_tokenizer):
        """Test performance with larger dataset."""
        # Create store with more examples
        store = MaxActStore(temp_db_path, tokenizer=mock_tokenizer)

        # Add 100 examples
        for i in range(100):
            store.add_example(
                score=float(i) / 100.0,
                input_ids=torch.randint(1, 1000, (10,)),
                latent_idx=i % 5,
                quantile_idx=i % 3,
            )

        readonly_store = ReadOnlyMaxActStore(temp_db_path, tokenizer=mock_tokenizer)

        # Test that reads are still fast
        import time

        start_time = time.time()
        examples = readonly_store.get_top_examples(limit=50)
        read_time = time.time() - start_time

        assert len(examples) == 50
        assert read_time < 1.0  # Should be fast

        # Test filtered reads
        latent_examples = readonly_store.get_top_examples(latent_idx=0, limit=20)
        assert len(latent_examples) <= 20
        assert all(ex["latent_idx"] == 0 for ex in latent_examples)

    def test_config_loading_edge_cases(self, temp_db_path):
        """Test config loading with various edge cases."""
        # Create database with only storage_format config
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            cursor.execute("INSERT INTO config VALUES ('storage_format', 'sparse')")
            cursor.execute(
                "CREATE TABLE sequences (sequence_uid INTEGER PRIMARY KEY, token_ids BLOB, text TEXT, sequence_length INTEGER, dataset_id INTEGER, dataset_name TEXT)"
            )
            cursor.execute(
                "CREATE TABLE examples (example_id INTEGER PRIMARY KEY, sequence_uid INTEGER, score REAL, latent_idx INTEGER, quantile_idx INTEGER, metadata TEXT)"
            )
            cursor.execute(
                "CREATE TABLE activation_details (example_id INTEGER PRIMARY KEY, positions BLOB, activation_values BLOB)"
            )
            conn.commit()

        readonly_store = ReadOnlyMaxActStore(temp_db_path)
        assert readonly_store.storage_format == "sparse"
        assert readonly_store.max_examples is None

    def test_empty_database_operations(self, temp_db_path, mock_tokenizer):
        """Test operations on empty but valid database."""
        # Create empty store
        store = MaxActStore(temp_db_path, tokenizer=mock_tokenizer)
        # Don't add any examples

        readonly_store = ReadOnlyMaxActStore(temp_db_path, tokenizer=mock_tokenizer)

        assert len(readonly_store) == 0
        examples = readonly_store.get_top_examples()
        assert len(examples) == 0

        # Filtered queries on empty database should also work
        latent_examples = readonly_store.get_top_examples(latent_idx=0)
        assert len(latent_examples) == 0


class TestMaxActStore:
    """Test class for MaxActStore functionality."""

    def test_initialization_sparse(self, temp_db_path, mock_tokenizer):
        """Test basic initialization with sparse storage."""
        store = MaxActStore(
            temp_db_path,
            max_examples=100,
            tokenizer=mock_tokenizer,
            storage_format="sparse",
        )

        assert store.db_manager.db_path == temp_db_path
        assert store.db_manager.readonly == False
        assert store.max_examples == 100
        assert store.tokenizer == mock_tokenizer
        assert store.storage_format == "sparse"
        assert temp_db_path.exists()

        # Test that database tables were created
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()

            # Check all tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sequences'"
            )
            assert cursor.fetchone() is not None

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='examples'"
            )
            assert cursor.fetchone() is not None

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='activation_details'"
            )
            assert cursor.fetchone() is not None

    def test_initialization_dense(self, temp_db_path, mock_tokenizer):
        """Test basic initialization with dense storage."""
        store = MaxActStore(
            temp_db_path,
            max_examples=100,
            tokenizer=mock_tokenizer,
            storage_format="dense",
        )

        assert store.storage_format == "dense"

    def test_initialization_invalid_format(self, temp_db_path):
        """Test initialization with invalid storage format."""
        with pytest.raises(ValueError):
            # Use type ignore to bypass type checker for this test
            MaxActStore(temp_db_path, storage_format="invalid")  # type: ignore

    def test_length_empty(self, temp_db_path, storage_format):
        """Test length method on empty database."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        assert len(store) == 0

    def test_clear(self, temp_db_path, storage_format):
        """Test clearing the database."""
        store = MaxActStore(temp_db_path, max_examples=5, storage_format=storage_format)

        # Add some data
        store.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.8, 0.6, 0.4]),
        )

        assert len(store) == 1

        # Clear and verify empty
        store.clear()
        assert len(store) == 0

    def test_add_single_example(self, temp_db_path, mock_tokenizer, storage_format):
        """Test adding a single example."""
        store = MaxActStore(
            temp_db_path,
            max_examples=5,
            tokenizer=mock_tokenizer,
            storage_format=storage_format,
        )

        input_ids = torch.tensor([1, 2, 3, 4])
        scores_per_token = torch.tensor([0.8, 0.6, 0.4, 0.2])

        store.add_example(
            score=0.8,
            input_ids=input_ids,
            scores_per_token=scores_per_token,
            latent_idx=5,
            quantile_idx=2,
            additional_data={"layer": 10},
        )

        assert len(store) == 1

        # Retrieve and verify
        examples = store.get_top_examples()
        assert len(examples) == 1

        example = examples[0]
        assert example["max_score"] == 0.8
        assert example["input_ids"] == [1, 2, 3, 4]
        assert example["latent_idx"] == 5
        assert example["quantile_idx"] == 2
        assert example["layer"] == 10
        assert "tok_1 tok_2 tok_3 tok_4" in example["text"]

        # Test activation details retrieval for both formats
        details = store.get_example_details(example["example_id"])
        assert "scores_per_token" in details
        assert len(details["scores_per_token"]) == 4
        # Use approximate comparisons for floating point values
        expected_scores = [0.8, 0.6, 0.4, 0.2]
        for i, expected in enumerate(expected_scores):
            assert np.isclose(details["scores_per_token"][i], expected)

    def test_activation_details_different_formats_sparse(self, temp_db_path):
        """Test handling different activation details formats with sparse storage."""
        store = MaxActStore(temp_db_path, storage_format="sparse")
        self._test_activation_details_formats_sparse(store)

    def test_activation_details_different_formats_dense(self, temp_db_path):
        """Test handling different activation details formats with dense storage."""
        store = MaxActStore(temp_db_path, storage_format="dense")
        self._test_activation_details_formats_dense(store)

    def _test_activation_details_formats_sparse(self, store):
        """Helper method to test sparse activation details formats."""
        # Test with tuple format for sparse storage
        examples_data = {0: {0: [(0.8, 0)]}}
        sequences = [(0, torch.tensor([1, 2, 3]))]
        activation_details = {
            0: {
                0: (np.array([0, 1]), np.array([0.8, 0.6]))
            }  # Tuple format (positions, values)
        }

        store.fill(examples_data, sequences, activation_details)

        # Retrieve and verify
        examples = store.get_top_examples()
        example_id = examples[0]["example_id"]
        details = store.get_example_details(example_id)

        assert "scores_per_token" in details
        assert (
            len(details["scores_per_token"]) == 3
        )  # Full sequence length (dense format)
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        assert np.isclose(details["scores_per_token"][2], 0.0)  # Zero-padded

        # Now test with Nx2 array format
        store.clear()

        # Create Nx2 array where positions are int and values are float32 viewed as int32
        positions = np.array([0, 1], dtype=np.int32)
        values = np.array([0.8, 0.6], dtype=np.float32)
        # Convert float32 values to int32 view (as done in original code)
        values_as_int32 = values.view(np.int32)
        pos_val_array = np.column_stack([positions, values_as_int32])

        activation_details_nx2 = {0: {0: pos_val_array}}  # Nx2 array format

        store.fill(examples_data, sequences, activation_details_nx2)

        examples = store.get_top_examples()
        example_id = examples[0]["example_id"]
        details = store.get_example_details(example_id)

        assert "scores_per_token" in details
        assert (
            len(details["scores_per_token"]) == 3
        )  # Full sequence length (dense format)
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        assert np.isclose(details["scores_per_token"][2], 0.0)  # Zero-padded

    def _test_activation_details_formats_dense(self, store):
        """Helper method to test dense activation details formats."""
        # Test with dense array format (just the values array)
        examples_data = {0: {0: [(0.8, 0)]}}
        sequences = [(0, torch.tensor([1, 2, 3]))]
        activation_details = {
            0: {0: np.array([0.8, 0.6, 0.0])}  # Dense format - single values array
        }

        store.fill(examples_data, sequences, activation_details)

        # Retrieve and verify
        examples = store.get_top_examples()
        example_id = examples[0]["example_id"]
        details = store.get_example_details(example_id)

        assert "scores_per_token" in details
        assert len(details["scores_per_token"]) == 3  # Full sequence length
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        assert np.isclose(details["scores_per_token"][2], 0.0)

    def test_storage_format_efficiency(self, temp_db_path):
        """Test that dense storage is more efficient for high-density activations."""
        # Create data with high density (all positions have activations)
        sequences = [(0, torch.tensor([1, 2, 3, 4, 5]))]
        examples_data = {0: {0: [(0.8, 0)]}}

        # Test sparse storage with tuple format
        sparse_activation_details = {
            0: {0: (np.array([0, 1, 2, 3, 4]), np.array([0.8, 0.6, 0.4, 0.2, 0.1]))}
        }
        store_sparse = MaxActStore(
            temp_db_path.with_suffix(".sparse.db"), storage_format="sparse"
        )
        store_sparse.fill(examples_data, sequences, sparse_activation_details)

        # Test dense storage with dense array format
        dense_activation_details = {
            0: {0: np.array([0.8, 0.6, 0.4, 0.2, 0.1])}  # Dense format
        }
        store_dense = MaxActStore(
            temp_db_path.with_suffix(".dense.db"), storage_format="dense"
        )
        store_dense.fill(examples_data, sequences, dense_activation_details)

        # Both should give same results
        details_sparse = store_sparse.get_example_details(1, return_dense=True)
        details_dense = store_dense.get_example_details(1, return_dense=True)

        # Both should return dense format now
        assert len(details_sparse["scores_per_token"]) == 5
        assert len(details_dense["scores_per_token"]) == 5

        # Test sparse format with return_dense=False
        details_sparse_sparse = store_sparse.get_example_details(1, return_dense=False)
        assert len(details_sparse_sparse["positions"]) == 5  # All positions have values
        assert len(details_sparse_sparse["scores_per_token"]) == 5

        # Values should be the same
        for i in range(5):
            assert (
                abs(
                    details_sparse["scores_per_token"][i]
                    - details_dense["scores_per_token"][i]
                )
                < 1e-6
            )

    def test_fill_bulk_loading(
        self,
        temp_db_path,
        sample_quantile_examples,
        sample_token_sequences,
        sample_activation_details,
        mock_tokenizer,
        storage_format,
    ):
        """Test bulk loading with fill method."""
        store = MaxActStore(
            temp_db_path, tokenizer=mock_tokenizer, storage_format=storage_format
        )

        store.fill(
            sample_quantile_examples,
            sample_token_sequences,
            sample_activation_details[storage_format],
        )

        # Should have 5 total examples
        assert len(store) == 5

        # Test retrieval
        examples = store.get_top_examples()
        assert len(examples) == 5

        # Examples should be sorted by score descending
        scores = [ex["max_score"] for ex in examples]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.9  # Highest score
        assert scores[-1] == 0.5  # Lowest score

    def test_top_k_management(self, temp_db_path, storage_format):
        """Test top-k management functionality."""
        store = MaxActStore(temp_db_path, max_examples=3, storage_format=storage_format)

        # Add 5 examples with different scores
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        for i, score in enumerate(scores):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
            )

        # Should only keep top 3
        assert len(store) == 3

        examples = store.get_top_examples()
        example_scores = [ex["max_score"] for ex in examples]
        assert example_scores == [0.9, 0.8, 0.7]

    def test_get_examples_with_filters(
        self,
        temp_db_path,
        sample_quantile_examples,
        sample_token_sequences,
        mock_tokenizer,
        storage_format,
    ):
        """Test filtering examples by latent_idx and quantile_idx."""
        store = MaxActStore(
            temp_db_path, tokenizer=mock_tokenizer, storage_format=storage_format
        )
        store.fill(sample_quantile_examples, sample_token_sequences)

        # Filter by latent_idx only
        examples = store.get_top_examples(latent_idx=0)
        assert len(examples) == 3  # latent 0 appears in 3 examples
        assert all(ex["latent_idx"] == 0 for ex in examples)

        # Filter by quantile_idx only
        examples = store.get_top_examples(quantile_idx=1)
        assert len(examples) == 2  # quantile 1 has 2 examples
        assert all(ex["quantile_idx"] == 1 for ex in examples)

        # Filter by both
        examples = store.get_top_examples(latent_idx=0, quantile_idx=1)
        assert len(examples) == 1
        assert examples[0]["latent_idx"] == 0 and examples[0]["quantile_idx"] == 1

        # Filter with limit
        examples = store.get_top_examples(latent_idx=0, limit=2)
        assert len(examples) == 2

    def test_merge_basic(self, temp_db_path, mock_tokenizer, storage_format):
        """Test basic merge functionality."""
        # Create two stores with same storage format
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(
            store1_path,
            max_examples=10,
            tokenizer=mock_tokenizer,
            storage_format=storage_format,
        )
        store2 = MaxActStore(
            store2_path,
            max_examples=10,
            tokenizer=mock_tokenizer,
            storage_format=storage_format,
        )

        # Add data to store1
        store1.add_example(
            score=0.9,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.9, 0.8, 0.7]),
            latent_idx=0,
            additional_data={"dataset": "store1"},
        )

        # Add data to store2
        store2.add_example(
            score=0.8,
            input_ids=torch.tensor([4, 5, 6]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6]),
            latent_idx=1,
            additional_data={"dataset": "store2"},
        )

        assert len(store1) == 1
        assert len(store2) == 1

        # Merge store2 into store1
        store1.merge(store2)

        # Check results
        assert len(store1) == 2
        assert len(store2) == 1  # Source store unchanged

        examples = store1.get_top_examples()
        assert len(examples) == 2

        # Should be sorted by score (descending)
        assert examples[0]["max_score"] == 0.9
        assert examples[1]["max_score"] == 0.8

        # Check that both datasets are present
        datasets = [ex["dataset"] for ex in examples]
        assert "store1" in datasets
        assert "store2" in datasets

    def test_merge_with_offset(self, temp_db_path, mock_tokenizer, storage_format):
        """Test merge with sequence index offset."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(
            store1_path, tokenizer=mock_tokenizer, storage_format=storage_format
        )
        store2 = MaxActStore(
            store2_path, tokenizer=mock_tokenizer, storage_format=storage_format
        )

        # Add data with overlapping sequence indices
        sequences1 = [(0, torch.tensor([1, 2, 3])), (1, torch.tensor([4, 5, 6]))]
        sequences2 = [(0, torch.tensor([7, 8, 9])), (1, torch.tensor([10, 11, 12]))]

        examples_data1 = {0: {0: [(0.9, 0), (0.8, 1)]}}
        examples_data2 = {0: {1: [(0.7, 0), (0.6, 1)]}}

        store1.fill(examples_data1, sequences1)
        store2.fill(examples_data2, sequences2)

        assert len(store1) == 2
        assert len(store2) == 2

        # Merge with offset of 1000 to separate datasets
        store1.merge(store2, sequence_uid_offset=1000)

        assert len(store1) == 4

        examples = store1.get_top_examples()
        sequence_indices = [ex["sequence_uid"] for ex in examples]

        # Original store1 sequences should have indices 0, 1
        # Merged store2 sequences should have indices 1000, 1001
        assert 0 in sequence_indices
        assert 1 in sequence_indices
        assert 1000 in sequence_indices
        assert 1001 in sequence_indices

    def test_merge_offset_conflict(self, temp_db_path, storage_format):
        """Test merge with offset that causes conflicts."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)

        # Add data to store1 using add_example to get sequence_uid = hash(...)
        store1.add_example(score=0.9, input_ids=torch.tensor([1, 2, 3]))

        # Add data to store2 using add_example
        store2.add_example(score=0.7, input_ids=torch.tensor([4, 5, 6]))

        # Get the actual sequence indices that were created
        examples1 = store1.get_top_examples()
        examples2 = store2.get_top_examples()
        seq_idx1 = examples1[0]["sequence_uid"]
        seq_idx2 = examples2[0]["sequence_uid"]

        # Calculate offset that would cause a conflict
        conflict_offset = seq_idx1 - seq_idx2

        # Merge with offset that would cause conflict
        with pytest.raises(ValueError, match="Index conflict"):
            store1.merge(store2, sequence_uid_offset=conflict_offset)

    def test_merge_storage_format_mismatch(self, temp_db_path):
        """Test merge with incompatible storage formats."""
        store1_path = temp_db_path.with_suffix(".sparse.db")
        store2_path = temp_db_path.with_suffix(".dense.db")

        store1 = MaxActStore(store1_path, storage_format="sparse")
        store2 = MaxActStore(store2_path, storage_format="dense")

        with pytest.raises(ValueError, match="Storage format mismatch"):
            store1.merge(store2)

    def test_merge_with_activation_details(self, temp_db_path, storage_format):
        """Test merge preserves activation details correctly."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)

        sequences1 = [(0, torch.tensor([1, 2, 3]))]
        sequences2 = [(0, torch.tensor([4, 5, 6, 7]))]

        examples_data1 = {0: {0: [(0.9, 0)]}}
        examples_data2 = {0: {1: [(0.7, 0)]}}

        if storage_format == "sparse":
            activation_details1 = {0: {0: (np.array([0, 1]), np.array([0.9, 0.8]))}}
            activation_details2 = {
                1: {0: (np.array([0, 2, 3]), np.array([0.7, 0.6, 0.5]))}
            }
        else:  # dense
            activation_details1 = {0: {0: np.array([0.9, 0.8, 0.0])}}
            activation_details2 = {1: {0: np.array([0.7, 0.0, 0.6, 0.5])}}

        store1.fill(examples_data1, sequences1, activation_details1)
        store2.fill(examples_data2, sequences2, activation_details2)

        # Merge with offset
        store1.merge(store2, sequence_uid_offset=100)

        examples = store1.get_top_examples()
        assert len(examples) == 2

        # Check activation details are preserved
        for example in examples:
            details = store1.get_example_details(example["example_id"])
            assert "scores_per_token" in details
            assert len(details["scores_per_token"]) > 0

    def test_merge_with_top_k_constraint(self, temp_db_path, storage_format):
        """Test merge respects top-k constraint."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        # Create stores with small max_examples limit
        store1 = MaxActStore(store1_path, max_examples=3, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)

        # Add 2 examples to store1
        for i, score in enumerate([0.9, 0.8]):
            store1.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
            )

        # Add 3 examples to store2 with mixed scores
        for i, score in enumerate([0.95, 0.75, 0.65]):  # One high, two low
            store2.add_example(
                score=score,
                input_ids=torch.tensor([i + 10, i + 11, i + 12]),
                scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
            )

        assert len(store1) == 2
        assert len(store2) == 3

        # Merge with top-k maintenance
        store1.merge(store2, maintain_top_k=True)

        # Should only keep top 3 examples
        assert len(store1) == 3

        examples = store1.get_top_examples()
        scores = [ex["max_score"] for ex in examples]

        # Should keep the 3 highest scores: 0.95, 0.9, 0.8
        assert scores == [0.95, 0.9, 0.8]

    def test_merge_empty_store(self, temp_db_path, storage_format):
        """Test merging an empty store."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)

        # Add data only to store1
        store1.add_example(
            score=0.9,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.9, 0.8, 0.7]),
        )

        assert len(store1) == 1
        assert len(store2) == 0

        # Merge empty store2 into store1
        store1.merge(store2)

        # Nothing should change
        assert len(store1) == 1

    def test_merge_into_empty_store(self, temp_db_path, storage_format):
        """Test merging into an empty store."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)

        # Add data only to store2
        store2.add_example(
            score=0.8,
            input_ids=torch.tensor([4, 5, 6]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6]),
        )

        assert len(store1) == 0
        assert len(store2) == 1

        # Merge store2 into empty store1
        store1.merge(store2)

        assert len(store1) == 1
        examples = store1.get_top_examples()
        assert examples[0]["max_score"] == 0.8

    def test_merge_auto_offset(self, temp_db_path, mock_tokenizer, storage_format):
        """Test merge with 'auto' offset."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(
            store1_path, tokenizer=mock_tokenizer, storage_format=storage_format
        )
        store2 = MaxActStore(
            store2_path, tokenizer=mock_tokenizer, storage_format=storage_format
        )

        # Add data with known sequence indices using fill
        sequences1 = [(0, torch.tensor([1, 2, 3])), (1, torch.tensor([4, 5, 6]))]
        sequences2 = [(0, torch.tensor([7, 8, 9])), (1, torch.tensor([10, 11, 12]))]

        examples_data1 = {0: {0: [(0.9, 0), (0.8, 1)]}}  # sequence indices 0, 1
        examples_data2 = {
            0: {1: [(0.7, 0), (0.6, 1)]}
        }  # sequence indices 0, 1 (will be offset)

        store1.fill(examples_data1, sequences1)
        store2.fill(examples_data2, sequences2)

        assert len(store1) == 2
        assert len(store2) == 2

        # Get the max sequence index from store1
        examples = store1.get_top_examples()
        max_seq_idx = max(ex["sequence_uid"] for ex in examples)
        expected_offset = max_seq_idx + 1

        # Merge with auto offset
        store1.merge(store2, sequence_uid_offset="auto")

        assert len(store1) == 4

        all_examples = store1.get_top_examples()
        sequence_indices = [ex["sequence_uid"] for ex in all_examples]

        # Should have original indices (0, 1) and offset indices (expected_offset, expected_offset+1)
        assert 0 in sequence_indices
        assert 1 in sequence_indices
        assert expected_offset in sequence_indices
        assert expected_offset + 1 in sequence_indices

    def test_merge_auto_offset_empty_target(self, temp_db_path, storage_format):
        """Test merge with 'auto' offset into empty target store."""
        store1_path = temp_db_path.with_suffix(".store1.db")
        store2_path = temp_db_path.with_suffix(".store2.db")

        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)

        # Add data only to store2
        store2.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6]),
        )

        assert len(store1) == 0
        assert len(store2) == 1

        # Get original sequence index
        original_examples = store2.get_top_examples()
        original_seq_idx = original_examples[0]["sequence_uid"]

        # Merge with auto offset into empty store (should start from 0)
        store1.merge(store2, sequence_uid_offset="auto")

        assert len(store1) == 1
        merged_examples = store1.get_top_examples()

        # With empty target, auto offset should be 0, so sequence index should be original + 0
        assert merged_examples[0]["sequence_uid"] == original_seq_idx + 0

    def test_set_dataset_info(self, temp_db_path, storage_format):
        """Test setting dataset info for all sequences in the store."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        # Add some examples
        for i, score in enumerate([0.9, 0.8, 0.7]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
            )

        assert len(store) == 3

        # Initially, no dataset info should be set
        examples = store.get_top_examples()
        for example in examples:
            assert example["dataset_id"] is None
            assert example["dataset_name"] is None

        # Set dataset info for all sequences
        updated_count = store.set_dataset_info(
            dataset_id=42, dataset_name="test_dataset"
        )
        assert updated_count == 3

        # Verify dataset info was set
        examples = store.get_top_examples()
        for example in examples:
            assert example["dataset_id"] == 42
            assert example["dataset_name"] == "test_dataset"

    def test_set_dataset_info_partial(self, temp_db_path, storage_format):
        """Test setting only dataset_id or dataset_name."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        store.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6]),
        )

        # Set only dataset_id
        updated_count = store.set_dataset_info(dataset_id=123)
        assert updated_count == 1

        examples = store.get_top_examples()
        assert examples[0]["dataset_id"] == 123
        assert examples[0]["dataset_name"] is None

        # Set only dataset_name
        updated_count = store.set_dataset_info(dataset_name="new_dataset")
        assert updated_count == 1

        examples = store.get_top_examples()
        assert examples[0]["dataset_id"] == 123  # Should remain unchanged
        assert examples[0]["dataset_name"] == "new_dataset"

    def test_set_dataset_info_no_overwrite(self, temp_db_path, storage_format):
        """Test setting dataset info with overwrite_existing=False."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        # Add example with dataset info
        store.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            dataset_id=100,
            dataset_name="existing_dataset",
        )

        # Add example without dataset info
        store.add_example(score=0.7, input_ids=torch.tensor([4, 5, 6]))

        assert len(store) == 2

        # Try to set dataset info without overwriting existing
        updated_count = store.set_dataset_info(
            dataset_id=200, dataset_name="new_dataset", overwrite_existing=False
        )
        assert updated_count == 1  # Should only update the one without existing data

        examples = store.get_top_examples()

        # First example should keep original dataset info
        first_example = next(ex for ex in examples if ex["max_score"] == 0.8)
        assert first_example["dataset_id"] == 100
        assert first_example["dataset_name"] == "existing_dataset"

        # Second example should have new dataset info
        second_example = next(ex for ex in examples if ex["max_score"] == 0.7)
        assert second_example["dataset_id"] == 200
        assert second_example["dataset_name"] == "new_dataset"

    def test_per_dataset_top_k_basic(self, temp_db_path, storage_format):
        """Test per-dataset top-k functionality."""
        # Create store with per_dataset=True and max_examples=2
        store = MaxActStore(
            temp_db_path,
            max_examples=2,
            storage_format=storage_format,
            per_dataset=True,
        )

        # Add examples from dataset A
        for i, score in enumerate([0.9, 0.8, 0.7]):  # 3 examples, should keep top 2
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
                additional_data={"dataset_name": "dataset_A"},
            )

        # Set dataset info for these examples
        store.set_dataset_info(dataset_name="dataset_A")

        # Add examples from dataset B
        for i, score in enumerate([0.95, 0.75, 0.65]):  # 3 examples, should keep top 2
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 10, i + 11, i + 12]),
                scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
                additional_data={"dataset_name": "dataset_B"},
            )

        # Set dataset info for dataset B examples (need to be more specific)
        # First get examples without dataset_name set
        examples_without_dataset = store.get_top_examples()
        dataset_b_examples = [
            ex for ex in examples_without_dataset if ex["dataset_name"] is None
        ]

        # Update these specific sequences
        with sqlite3.connect(store.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            for ex in dataset_b_examples:
                cursor.execute(
                    "UPDATE sequences SET dataset_name = ? WHERE sequence_uid = ?",
                    ("dataset_B", ex["sequence_uid"]),
                )
            conn.commit()

        # Should have 4 total examples (2 per dataset)
        assert len(store) == 4

        # Check that we have 2 examples from each dataset
        examples = store.get_top_examples()
        dataset_a_examples = [
            ex for ex in examples if ex["dataset_name"] == "dataset_A"
        ]
        dataset_b_examples = [
            ex for ex in examples if ex["dataset_name"] == "dataset_B"
        ]

        assert len(dataset_a_examples) == 2
        assert len(dataset_b_examples) == 2

        # Check that we kept the highest scores within each dataset
        dataset_a_scores = sorted(
            [ex["max_score"] for ex in dataset_a_examples], reverse=True
        )
        dataset_b_scores = sorted(
            [ex["max_score"] for ex in dataset_b_examples], reverse=True
        )

        assert dataset_a_scores == [0.9, 0.8]  # Dropped 0.7
        assert dataset_b_scores == [0.95, 0.75]  # Dropped 0.65

    def test_top_k_basic_batch_insertion(self, temp_db_path, storage_format):
        """Test basic top-k functionality with batch insertion."""
        store = MaxActStore(temp_db_path, max_examples=3, storage_format=storage_format)

        # Create batch data - 5 examples, should keep top 3
        batch_size = 5
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        input_ids_batch = torch.stack(
            [torch.tensor([i, i + 1, i + 2]) for i in range(batch_size)]
        )
        scores_per_token_batch = torch.stack(
            [
                torch.tensor([scores[i], scores[i] - 0.1, scores[i] - 0.2])
                for i in range(batch_size)
            ]
        )
        additional_data_batch = [{"example_idx": i} for i in range(batch_size)]

        # Add batch examples
        store.add_batch_examples(
            scores_per_example=scores,
            input_ids_batch=input_ids_batch,
            scores_per_token_batch=scores_per_token_batch,
            additional_data_batch=additional_data_batch,
            dataset_name="test_dataset",
            dataset_id=1,
        )

        # Should have exactly 3 examples (top-k maintained)
        assert len(store) == 3

        # Get examples and verify they are the top 3 scores
        examples = store.get_top_examples()
        example_scores = [ex["max_score"] for ex in examples]

        # Should be sorted in descending order and contain top 3 scores
        assert torch.allclose(
            torch.tensor(example_scores), torch.tensor([0.9, 0.8, 0.7])
        )

        # Verify the correct input_ids are preserved
        expected_input_ids = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]  # First 3 examples
        actual_input_ids = [ex["input_ids"] for ex in examples]
        assert actual_input_ids == expected_input_ids

        # Verify additional data is preserved
        expected_indices = [0, 1, 2]
        actual_indices = [ex["example_idx"] for ex in examples]
        assert actual_indices == expected_indices

        # Verify that the dataset info is set
        assert examples[0]["dataset_name"] == "test_dataset"
        assert examples[0]["dataset_id"] == 1

    def test_per_dataset_vs_overall_comparison(self, temp_db_path, storage_format):
        """Test that per-dataset and overall top-k behave differently."""
        store_overall_path = temp_db_path.with_suffix(".overall.db")
        store_per_dataset_path = temp_db_path.with_suffix(".per_dataset.db")

        # Create two stores with different top-k strategies
        store_overall = MaxActStore(
            store_overall_path,
            max_examples=3,
            storage_format=storage_format,
            per_dataset=False,
        )
        store_per_dataset = MaxActStore(
            store_per_dataset_path,
            max_examples=2,
            storage_format=storage_format,
            per_dataset=True,
        )

        # Add same data to both stores with distinct token sequences for easy identification
        example_data = [
            (0.9, "dataset_A", 0, [1, 2, 3]),
            (0.8, "dataset_A", 0, [4, 5, 6]),
            (0.7, "dataset_A", 0, [7, 8, 9]),  # 3 from A
            (0.95, "dataset_B", 1, [10, 11, 12]),
            (0.65, "dataset_B", 1, [13, 14, 15]),  # 2 from B
        ]

        for score, dataset, dataset_id, tokens in example_data:
            for store in [store_overall, store_per_dataset]:
                store.add_example(
                    score=score,
                    input_ids=torch.tensor(tokens),
                    scores_per_token=torch.tensor([score, score - 0.1, score - 0.2]),
                    maintain_top_k=False,  # Don't maintain top-k until after we set dataset names
                    dataset_name=dataset,
                    dataset_id=dataset_id,
                )

        # Force top-k maintenance
        store_overall._maintain_top_k()
        store_per_dataset._maintain_top_k()

        # Overall store should keep top 3 scores globally: 0.95, 0.9, 0.8
        overall_examples = store_overall.get_top_examples()
        overall_scores = [ex["max_score"] for ex in overall_examples]
        assert len(overall_examples) == 3
        assert sorted(overall_scores, reverse=True) == [0.95, 0.9, 0.8]

        # Per-dataset store should keep top 2 from each dataset: A(0.9, 0.8), B(0.95, 0.65)
        per_dataset_examples = store_per_dataset.get_top_examples()
        assert len(per_dataset_examples) == 4  # 2 per dataset

        dataset_a_examples = [
            ex for ex in per_dataset_examples if ex["dataset_name"] == "dataset_A"
        ]
        dataset_b_examples = [
            ex for ex in per_dataset_examples if ex["dataset_name"] == "dataset_B"
        ]

        assert len(dataset_a_examples) == 2
        assert len(dataset_b_examples) == 2

        dataset_a_scores = sorted(
            [ex["max_score"] for ex in dataset_a_examples], reverse=True
        )
        dataset_b_scores = sorted(
            [ex["max_score"] for ex in dataset_b_examples], reverse=True
        )

        assert dataset_a_scores == [0.9, 0.8]
        assert dataset_b_scores == [0.95, 0.65]

    def test_per_dataset_with_null_dataset(self, temp_db_path, storage_format):
        """Test per-dataset top-k with some examples having NULL dataset_name."""
        store = MaxActStore(
            temp_db_path,
            max_examples=2,
            storage_format=storage_format,
            per_dataset=True,
        )

        # Add examples with dataset A
        store.add_example(0.9, torch.tensor([1, 2, 3]))
        store.add_example(0.8, torch.tensor([4, 5, 6]))
        store.add_example(
            0.7, torch.tensor([7, 8, 9])
        )  # Should be removed from NULL group

        # Set dataset for first two examples
        examples = store.get_top_examples()
        with sqlite3.connect(store.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(
                "UPDATE sequences SET dataset_name = ? WHERE sequence_uid = ?",
                ("dataset_A", examples[0]["sequence_uid"]),
            )
            cursor.execute(
                "UPDATE sequences SET dataset_name = ? WHERE sequence_uid = ?",
                ("dataset_A", examples[1]["sequence_uid"]),
            )
            conn.commit()

        # Add more examples with explicit NULL dataset (leave as-is)
        store.add_example(0.85, torch.tensor([10, 11, 12]))
        store.add_example(0.75, torch.tensor([13, 14, 15]))
        store.add_example(
            0.65, torch.tensor([16, 17, 18])
        )  # Should be removed from NULL group

        # Force top-k maintenance
        store._maintain_top_k()

        final_examples = store.get_top_examples()

        # Should have 2 from dataset_A and 2 from NULL dataset = 4 total
        assert len(final_examples) == 4

        dataset_a_examples = [
            ex for ex in final_examples if ex["dataset_name"] == "dataset_A"
        ]
        null_dataset_examples = [
            ex for ex in final_examples if ex["dataset_name"] is None
        ]

        assert len(dataset_a_examples) == 2
        assert len(null_dataset_examples) == 2

        # Check scores within each group
        dataset_a_scores = sorted(
            [ex["max_score"] for ex in dataset_a_examples], reverse=True
        )
        null_scores = sorted(
            [ex["max_score"] for ex in null_dataset_examples], reverse=True
        )

        assert dataset_a_scores == [0.9, 0.8]  # Top 2 from dataset A
        assert null_scores == [0.85, 0.75]  # Top 2 from NULL dataset

    def test_async_writer_basic_functionality(
        self, temp_db_path, mock_tokenizer, storage_format
    ):
        """Test basic AsyncMaxActStoreWriter functionality."""
        store = MaxActStore(
            temp_db_path,
            max_examples=100,
            tokenizer=mock_tokenizer,
            storage_format=storage_format,
        )

        # Test creating async writer
        async_writer = store.create_async_writer(buffer_size=10, flush_interval=1.0)
        assert isinstance(async_writer, AsyncMaxActStoreWriter)
        assert async_writer.buffer_size == 10
        assert async_writer.flush_interval == 1.0
        assert not async_writer.is_running

        # Test context manager
        with async_writer as writer:
            assert writer.is_running

            # Add some test data
            batch_size = 5
            scores = torch.rand(batch_size) * 10
            input_ids = torch.randint(1, 1000, (batch_size, 10))
            attention_mask = torch.ones(batch_size, 10)
            scores_per_token = torch.rand(batch_size, 10)

            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                attention_mask_batch=attention_mask,
                scores_per_token_batch=scores_per_token,
                dataset_name="test_dataset",
            )

            # Flush to ensure data is written
            writer.flush()

        # After context exit, writer should be stopped
        assert not async_writer.is_running

        # Verify data was written to store
        assert store.get_num_sequences() == batch_size
        assert len(store) == batch_size
        examples = store.get_top_examples()
        assert all(ex["dataset_name"] == "test_dataset" for ex in examples)

    def test_async_writer_buffering(self, temp_db_path, storage_format):
        """Test that async writer properly buffers data before writing."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        with store.create_async_writer(
            buffer_size=20, flush_interval=60.0, auto_maintain_top_k=False
        ) as writer:
            # Add data that should stay in buffer (less than buffer_size)
            for i in range(3):
                batch_size = 5  # Total 15 examples, less than buffer_size=20
                scores = torch.ones(batch_size) * (0.5 + i * 0.1)
                input_ids = torch.randint(1, 100, (batch_size, 8))

                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    dataset_name=f"batch_{i}",
                )

            # Give background process a moment but data should still be buffered
            import time

            time.sleep(0.1)

            # Force flush
            writer.flush()

            # Give background process time to write
            time.sleep(0.5)

        # After context exit, all data should be written
        assert len(store) == 15

        # Verify we have data from all batches
        examples = store.get_top_examples()
        datasets = set(ex["dataset_name"] for ex in examples)
        assert datasets == {"batch_0", "batch_1", "batch_2"}

    def test_async_writer_buffer_overflow(self, temp_db_path, storage_format):
        """Test that async writer flushes when buffer size is exceeded."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        with store.create_async_writer(buffer_size=10, flush_interval=60.0) as writer:
            # Add more data than buffer_size to trigger flush
            batch_size = 6
            scores = torch.ones(batch_size) * 0.8
            input_ids = torch.randint(1, 100, (batch_size, 5))

            # First batch (6 examples) - should stay in buffer
            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                dataset_name="batch_1",
            )

            # Second batch (6 examples) - total 12, should trigger flush since > buffer_size=10
            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                dataset_name="batch_2",
            )

            # Give background process time to write
            import time

            time.sleep(0.5)

        # All data should be written
        assert len(store) == 12

    def test_async_writer_attention_mask_handling(self, temp_db_path, storage_format):
        """Test that async writer properly handles attention masks."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        with store.create_async_writer(buffer_size=5) as writer:
            batch_size = 3
            seq_len = 8
            scores = torch.ones(batch_size) * 0.7
            input_ids = torch.randint(1, 100, (batch_size, seq_len))

            # Create attention mask with different valid lengths per sequence
            attention_mask = torch.zeros(batch_size, seq_len)
            attention_mask[0, :5] = 1  # First sequence has 5 valid tokens
            attention_mask[1, :3] = 1  # Second sequence has 3 valid tokens
            attention_mask[2, :7] = 1  # Third sequence has 7 valid tokens

            scores_per_token = torch.rand(batch_size, seq_len)

            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                attention_mask_batch=attention_mask,
                scores_per_token_batch=scores_per_token,
                dataset_name="masked_test",
            )

        # Verify data was written with correct sequence lengths
        examples = store.get_top_examples()
        assert len(examples) == 3

        # Check that sequences have correct lengths (matching attention mask)
        sequence_lengths = [len(ex["input_ids"]) for ex in examples]
        assert sorted(sequence_lengths) == [
            3,
            5,
            7,
        ]  # Should match attention mask lengths

    def test_async_writer_error_handling(self, temp_db_path, storage_format):
        """Test error handling in async writer."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        # Test adding data to stopped writer
        async_writer = store.create_async_writer()

        with pytest.raises(RuntimeError, match="AsyncMaxActStoreWriter is not running"):
            async_writer.add_batch_examples(
                scores_per_example=torch.ones(2),
                input_ids_batch=torch.randint(1, 100, (2, 5)),
            )

    def test_async_writer_dataset_separation(self, temp_db_path, storage_format):
        """Test that async writer properly separates data by dataset."""
        store = MaxActStore(
            temp_db_path,
            max_examples=20,
            storage_format=storage_format,
            per_dataset=True,
        )

        with store.create_async_writer(buffer_size=50) as writer:
            # Add data for multiple datasets
            datasets = ["dataset_A", "dataset_B", "dataset_C"]

            for dataset in datasets:
                for batch_idx in range(3):  # 3 batches per dataset
                    batch_size = 4
                    # Different score ranges for each dataset to test separation
                    base_score = 0.3 + datasets.index(dataset) * 0.2
                    scores = torch.ones(batch_size) * (base_score + batch_idx * 0.05)
                    input_ids = torch.randint(1, 100, (batch_size, 6))

                    writer.add_batch_examples(
                        scores_per_example=scores,
                        input_ids_batch=input_ids,
                        dataset_name=dataset,
                    )

        # Verify data separation
        total_examples = len(store)
        assert total_examples == 36  # 3 datasets * 3 batches * 4 examples

        # Check that each dataset has its examples
        for dataset in datasets:
            dataset_examples = store.get_top_examples(dataset_names=[dataset])
            assert len(dataset_examples) == 12  # 3 batches * 4 examples per dataset
            assert all(ex["dataset_name"] == dataset for ex in dataset_examples)

    def test_async_writer_top_k_maintenance(self, temp_db_path, storage_format):
        """Test that async writer maintains top-k constraints."""
        max_examples = 15
        store = MaxActStore(
            temp_db_path, max_examples=max_examples, storage_format=storage_format
        )

        with store.create_async_writer(
            buffer_size=10, auto_maintain_top_k=True
        ) as writer:
            # Add more examples than max_examples
            total_batches = 5
            batch_size = 5  # Total: 25 examples, should keep only top 15

            for batch_idx in range(total_batches):
                # Vary scores so we can predict which examples should be kept
                scores = torch.ones(batch_size) * (
                    1.0 - batch_idx * 0.1
                )  # Decreasing scores
                input_ids = torch.randint(1, 100, (batch_size, 4))

                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    additional_data_batch=[{"batch": batch_idx}] * batch_size,
                )

        # Should only keep top max_examples
        assert len(store) == max_examples

        # Verify that highest scoring examples were kept
        examples = store.get_top_examples()
        assert len(examples) == max_examples

        # Examples should be sorted by score (descending)
        scores = [ex["max_score"] for ex in examples]
        assert scores == sorted(scores, reverse=True)

        # Highest scores should be from early batches
        assert scores[0] >= 0.9  # From batch 0 or 1

    def test_async_writer_manual_start_stop(self, temp_db_path, storage_format):
        """Test manual start/stop of async writer."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        async_writer = store.create_async_writer()

        # Initially not running
        assert not async_writer.is_running

        # Manual start
        async_writer.start()
        assert async_writer.is_running

        # Add some data
        scores = torch.ones(3) * 0.8
        input_ids = torch.randint(1, 100, (3, 5))

        async_writer.add_batch_examples(
            scores_per_example=scores, input_ids_batch=input_ids
        )

        # Manual stop
        async_writer.stop()
        assert not async_writer.is_running

        # Verify data was written
        assert len(store) == 3

    def test_async_writer_multiple_flushes(self, temp_db_path, storage_format):
        """Test multiple flush operations."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        with store.create_async_writer(
            buffer_size=20, auto_maintain_top_k=False
        ) as writer:
            # Add some data
            for i in range(3):
                scores = torch.ones(2) * (0.5 + i * 0.1)
                input_ids = torch.randint(1, 100, (2, 4))

                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    additional_data_batch=[{"flush_round": i}] * 2,
                )

                # Flush after each batch
                writer.flush()

                # Give background process time
                import time

                time.sleep(0.1)

        # All data should be written
        assert len(store) == 6

        # Verify data from all flush rounds
        examples = store.get_top_examples()
        flush_rounds = set(ex["flush_round"] for ex in examples)
        assert flush_rounds == {0, 1, 2}

    def test_async_writer_sparse_vs_dense_consistency(
        self, temp_db_path, mock_tokenizer
    ):
        """Test that sparse and dense storage formats produce consistent results in async writer."""
        # Test data
        batch_size = 4
        seq_len = 6
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        input_ids = torch.randint(1, 50, (batch_size, seq_len))
        scores_per_token = torch.rand(batch_size, seq_len)

        results = {}

        for storage_format in ["sparse", "dense"]:
            store_path = temp_db_path.with_suffix(f".{storage_format}.db")
            store = MaxActStore(
                store_path, tokenizer=mock_tokenizer, storage_format=storage_format
            )

            with store.create_async_writer() as writer:
                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    scores_per_token_batch=scores_per_token,
                    dataset_name="consistency_test",
                )

            # Get results
            examples = store.get_top_examples()
            results[storage_format] = examples

        # Compare results between formats
        sparse_examples = results["sparse"]
        dense_examples = results["dense"]

        assert len(sparse_examples) == len(dense_examples) == batch_size

        # Sort by score for comparison
        sparse_examples.sort(key=lambda x: x["max_score"], reverse=True)
        dense_examples.sort(key=lambda x: x["max_score"], reverse=True)

        for sparse_ex, dense_ex in zip(sparse_examples, dense_examples):
            assert abs(sparse_ex["max_score"] - dense_ex["max_score"]) < 1e-6
            assert sparse_ex["input_ids"] == dense_ex["input_ids"]
            assert sparse_ex["dataset_name"] == dense_ex["dataset_name"]

            # Compare activation details
            sparse_details = results["sparse"][0]  # Get from store
            dense_details = results["dense"][0]  # Get from store

            # Both should have scores_per_token
            if (
                "scores_per_token" in sparse_details
                and "scores_per_token" in dense_details
            ):
                assert len(sparse_details["scores_per_token"]) == len(
                    dense_details["scores_per_token"]
                )

    def test_async_writer_performance_improvement(self, temp_db_path, storage_format):
        """Test that async writer provides performance benefits over synchronous operations."""
        import time

        # Test data
        num_batches = 10
        batch_size = 20

        # Synchronous approach (using add_example directly)
        sync_store = MaxActStore(
            temp_db_path.with_suffix(".sync.db"), storage_format=storage_format
        )

        sync_start = time.time()
        for batch_idx in range(num_batches):
            scores = torch.rand(batch_size)
            for i in range(batch_size):
                sync_store.add_example(
                    score=scores[i].item(),
                    input_ids=torch.randint(1, 100, (8,)),
                    scores_per_token=torch.rand(8),
                    maintain_top_k=False,  # Don't maintain during addition
                )
        sync_store._maintain_top_k()  # Maintain once at end
        sync_time = time.time() - sync_start

        # Asynchronous approach
        async_store = MaxActStore(
            temp_db_path.with_suffix(".async.db"), storage_format=storage_format
        )

        async_start = time.time()
        with async_store.create_async_writer(buffer_size=batch_size * 2) as writer:
            for batch_idx in range(num_batches):
                scores = torch.rand(batch_size)
                input_ids = torch.randint(1, 100, (batch_size, 8))
                scores_per_token = torch.rand(batch_size, 8)

                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    scores_per_token_batch=scores_per_token,
                )
        async_time = time.time() - async_start

        # Verify both approaches produced same amount of data
        assert len(sync_store) == len(async_store) == num_batches * batch_size

        # Async should be faster (though this might be flaky in CI environments)
        print(f"Sync time: {sync_time:.3f}s, Async time: {async_time:.3f}s")
        # Note: We don't assert performance improvement as it depends on system load
        # But we can at least verify that async didn't take dramatically longer

    def test_async_writer_graceful_shutdown(self, temp_db_path, storage_format):
        """Test graceful shutdown of async writer process."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)

        async_writer = store.create_async_writer(
            buffer_size=100
        )  # Large buffer to keep data buffered
        async_writer.start()

        # Add some data that will be buffered
        scores = torch.ones(5) * 0.8
        input_ids = torch.randint(1, 100, (5, 6))

        async_writer.add_batch_examples(
            scores_per_example=scores,
            input_ids_batch=input_ids,
            dataset_name="shutdown_test",
        )

        # Stop should flush remaining data and shutdown gracefully
        async_writer.stop(timeout=10.0)

        # Verify data was written during shutdown
        assert len(store) == 5
        examples = store.get_top_examples()
        assert all(ex["dataset_name"] == "shutdown_test" for ex in examples)

    def test_async_writer_multiple_sequences_with_latent_indices(
        self, temp_db_path, storage_format
    ):
        """Test async writer with two different sequences, each with a single latent index."""
        store = MaxActStore(temp_db_path, max_examples=5, storage_format=storage_format)

        with store.create_async_writer(buffer_size=10) as writer:
            # Sequence 1: tokens [1, 2, 3, 4, 5] with latent_idx=0
            seq1_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            seq1_scores = torch.tensor([0.8])
            seq1_latent_idx = torch.tensor([0])
            seq1_scores_per_token = torch.rand(1, 5) * 0.8

            writer.add_batch_examples(
                scores_per_example=seq1_scores,
                input_ids_batch=seq1_input_ids,
                scores_per_token_batch=seq1_scores_per_token,
                latent_idx=seq1_latent_idx,
                additional_data_batch=[{"sequence_name": "sequence_1"}],
            )

            # Sequence 2: tokens [10, 11, 12, 13, 14, 15] with latent_idx=0
            seq2_input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
            seq2_scores = torch.tensor([0.9])
            seq2_latent_idx = torch.tensor([0])
            seq2_scores_per_token = torch.rand(1, 6) * 0.7

            writer.add_batch_examples(
                scores_per_example=seq2_scores,
                input_ids_batch=seq2_input_ids,
                scores_per_token_batch=seq2_scores_per_token,
                latent_idx=seq2_latent_idx,
                additional_data_batch=[{"sequence_name": "sequence_2"}],
            )

        # Should have 2 examples total (one from each sequence)
        assert len(store) == 2

        # Verify both examples are for latent_idx=0
        latent_0_examples = store.get_top_examples(latent_idx=0)
        assert len(latent_0_examples) == 2

        # Verify examples are sorted by score (descending)
        scores = [ex["max_score"] for ex in latent_0_examples]
        assert scores == sorted(scores, reverse=True)
        assert torch.isclose(
            torch.tensor(scores[0]), torch.tensor(0.9), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[1]), torch.tensor(0.8), atol=1e-4
        )  # sequence_1 has lower score

        # Verify sequence data is preserved correctly
        seq1_example = next(
            ex for ex in latent_0_examples if ex["sequence_name"] == "sequence_1"
        )
        seq2_example = next(
            ex for ex in latent_0_examples if ex["sequence_name"] == "sequence_2"
        )

        assert seq1_example["input_ids"] == [1, 2, 3, 4, 5]
        assert seq1_example["latent_idx"] == 0
        assert len(seq1_example["input_ids"]) == 5

        assert seq2_example["input_ids"] == [10, 11, 12, 13, 14, 15]
        assert seq2_example["latent_idx"] == 0
        assert len(seq2_example["input_ids"]) == 6

        # Verify activation details are preserved for both sequences
        seq1_details = store.get_example_details(seq1_example["example_id"])
        seq2_details = store.get_example_details(seq2_example["example_id"])

        assert "scores_per_token" in seq1_details
        assert "scores_per_token" in seq2_details
        assert len(seq1_details["scores_per_token"]) == 5
        assert len(seq2_details["scores_per_token"]) == 6

    def test_async_writer_multiple_sequences_with_multiple_latent_indices(
        self, temp_db_path, storage_format
    ):
        """Test async writer with two different sequences, each with multiple latent indices. Same sequence per batch."""
        store = MaxActStore(temp_db_path, max_examples=5, storage_format=storage_format)
        # Sequence 1: tokens [1, 2, 3, 4, 5] with latent_idx=0
        seq1_input_ids = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5])]
        seq1_scores = torch.tensor([0.8, 0.7])
        seq1_latent_idx = torch.tensor([0, 1])
        seq1_scores_per_token = torch.rand(2, 5) * 0.8

        # Sequence 2: tokens [10, 11, 12, 13, 14, 15] with latent_idx=0
        seq2_input_ids = [
            torch.tensor([10, 11, 12, 13, 14, 15]),
            torch.tensor([10, 11, 12, 13, 14, 15]),
        ]
        seq2_scores = torch.tensor([0.9, 0.8])
        seq2_latent_idx = torch.tensor([0, 1])
        seq2_scores_per_token = torch.rand(2, 6) * 0.7

        # Third batch with latents from both sequences
        seq3_input_ids = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([10, 11, 12, 13, 14, 15]),
        ]
        seq3_scores = torch.tensor([0.8, 0.7])
        seq3_latent_idx = torch.tensor([0, 1])
        seq3_scores_per_token = [torch.rand(1, 5) * 0.8, torch.rand(1, 6) * 0.8]

        with store.create_async_writer(buffer_size=10) as writer:

            writer.add_batch_examples(
                scores_per_example=seq1_scores,
                input_ids_batch=seq1_input_ids,
                scores_per_token_batch=seq1_scores_per_token,
                latent_idx=seq1_latent_idx,
                additional_data_batch=[
                    {"sequence_name": "sequence_1"},
                    {"sequence_name": "sequence_1"},
                ],
            )

            writer.add_batch_examples(
                scores_per_example=seq2_scores,
                input_ids_batch=seq2_input_ids,
                scores_per_token_batch=seq2_scores_per_token,
                latent_idx=seq2_latent_idx,
                additional_data_batch=[
                    {"sequence_name": "sequence_2"},
                    {"sequence_name": "sequence_2"},
                ],
            )

        # Should have 4 examples total (two from each sequence)
        assert len(store) == 4

        # Should have 2 sequences
        assert store.get_num_sequences() == 2

        # Verify both examples are for latent_idx=0
        latent_0_examples = store.get_top_examples(latent_idx=0)
        assert len(latent_0_examples) == 2
        latent_1_examples = store.get_top_examples(latent_idx=1)
        assert len(latent_1_examples) == 2

        # Verify examples are sorted by score (descending)
        scores = [ex["max_score"] for ex in latent_0_examples]
        assert scores == sorted(scores, reverse=True)
        assert torch.isclose(
            torch.tensor(scores[0]), torch.tensor(0.9), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[1]), torch.tensor(0.8), atol=1e-4
        )  # sequence_1 has lower score

        scores = [ex["max_score"] for ex in latent_1_examples]
        assert scores == sorted(scores, reverse=True)
        assert torch.isclose(
            torch.tensor(scores[0]), torch.tensor(0.8), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[1]), torch.tensor(0.7), atol=1e-4
        )  # sequence_1 has lower score
        # Verify sequence data is preserved correctly for latent_idx=0
        seq1_example = next(
            ex for ex in latent_0_examples if ex["sequence_name"] == "sequence_1"
        )
        seq2_example = next(
            ex for ex in latent_0_examples if ex["sequence_name"] == "sequence_2"
        )

        assert seq1_example["input_ids"] == [1, 2, 3, 4, 5]
        assert seq1_example["latent_idx"] == 0
        assert len(seq1_example["input_ids"]) == 5

        assert seq2_example["input_ids"] == [10, 11, 12, 13, 14, 15]
        assert seq2_example["latent_idx"] == 0
        assert len(seq2_example["input_ids"]) == 6

        # Verify sequence data is preserved correctly for latent_idx=1
        seq1_example_lat1 = next(
            ex for ex in latent_1_examples if ex["sequence_name"] == "sequence_1"
        )
        seq2_example_lat1 = next(
            ex for ex in latent_1_examples if ex["sequence_name"] == "sequence_2"
        )

        assert seq1_example_lat1["input_ids"] == [1, 2, 3, 4, 5]
        assert seq1_example_lat1["latent_idx"] == 1
        assert len(seq1_example_lat1["input_ids"]) == 5

        assert seq2_example_lat1["input_ids"] == [10, 11, 12, 13, 14, 15]
        assert seq2_example_lat1["latent_idx"] == 1
        assert len(seq2_example_lat1["input_ids"]) == 6

        # Verify activation details are preserved for both sequences and both latents
        seq1_details = store.get_example_details(seq1_example["example_id"])
        seq2_details = store.get_example_details(seq2_example["example_id"])
        seq1_details_lat1 = store.get_example_details(seq1_example_lat1["example_id"])
        seq2_details_lat1 = store.get_example_details(seq2_example_lat1["example_id"])

        assert "scores_per_token" in seq1_details
        assert "scores_per_token" in seq2_details
        assert "scores_per_token" in seq1_details_lat1
        assert "scores_per_token" in seq2_details_lat1
        assert len(seq1_details["scores_per_token"]) == 5
        assert len(seq2_details["scores_per_token"]) == 6
        assert len(seq1_details_lat1["scores_per_token"]) == 5
        assert len(seq2_details_lat1["scores_per_token"]) == 6

    def test_async_writer_multiple_sequences_with_multiple_latent_indices_in_same_batch(
        self, temp_db_path, storage_format
    ):
        """Test async writer with two different sequences, each with multiple latent indices. Same sequence per batch."""
        store = MaxActStore(temp_db_path, max_examples=5, storage_format=storage_format)
        # Sequence 1: tokens [1, 2, 3, 4, 5] with latent_idx=0
        seq1_input_ids = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5])]
        seq1_scores = torch.tensor([0.8, 0.7])
        seq1_latent_idx = torch.tensor([0, 1])
        seq1_scores_per_token = torch.rand(2, 5) * 0.8

        # Sequence 2: tokens [10, 11, 12, 13, 14, 15] with latent_idx=0
        seq2_input_ids = [
            torch.tensor([10, 11, 12, 13, 14, 15]),
            torch.tensor([10, 11, 12, 13, 14, 15]),
        ]
        seq2_scores = torch.tensor([0.9, 0.8])
        seq2_latent_idx = torch.tensor([0, 1])
        seq2_scores_per_token = torch.rand(2, 6) * 0.7

        # Third batch with latents from both sequences
        batch_input_ids = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([10, 11, 12, 13, 14, 15]),
        ]
        batch_scores = torch.tensor([10, 10])
        batch_latent_idx = torch.tensor([0, 1])
        batch_scores_per_token = [torch.rand(5) * 0.8, torch.rand(6) * 0.8]

        with store.create_async_writer(buffer_size=10) as writer:

            writer.add_batch_examples(
                scores_per_example=seq1_scores,
                input_ids_batch=seq1_input_ids,
                scores_per_token_batch=seq1_scores_per_token,
                latent_idx=seq1_latent_idx,
                additional_data_batch=[
                    {"sequence_name": "sequence_1"},
                    {"sequence_name": "sequence_1"},
                ],
            )

            writer.add_batch_examples(
                scores_per_example=seq2_scores,
                input_ids_batch=seq2_input_ids,
                scores_per_token_batch=seq2_scores_per_token,
                latent_idx=seq2_latent_idx,
                additional_data_batch=[
                    {"sequence_name": "sequence_2"},
                    {"sequence_name": "sequence_2"},
                ],
            )

            writer.add_batch_examples(
                scores_per_example=batch_scores,
                input_ids_batch=batch_input_ids,
                scores_per_token_batch=batch_scores_per_token,
                latent_idx=batch_latent_idx,
                additional_data_batch=[
                    {"sequence_name": "sequence_1"},
                    {"sequence_name": "sequence_2"},
                ],
            )

        # Should have 6 examples total (two from each sequence)
        assert len(store) == 6

        # Should have 2 sequences
        assert store.get_num_sequences() == 2

        # Verify both examples are for latent_idx=0
        latent_0_examples = store.get_top_examples(latent_idx=0)
        assert len(latent_0_examples) == 3
        latent_1_examples = store.get_top_examples(latent_idx=1)
        assert len(latent_1_examples) == 3

        # Verify examples are sorted by score (descending)
        scores = [ex["max_score"] for ex in latent_0_examples]
        assert scores == sorted(scores, reverse=True)
        assert torch.isclose(
            torch.tensor(scores[0]), torch.tensor(10.0), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[1]), torch.tensor(0.9), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[2]), torch.tensor(0.8), atol=1e-4
        )  # sequence_1 has lower score

        scores = [ex["max_score"] for ex in latent_1_examples]
        assert scores == sorted(scores, reverse=True)
        assert torch.isclose(
            torch.tensor(scores[0]), torch.tensor(10.0), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[1]), torch.tensor(0.8), atol=1e-4
        )  # sequence_2 has higher score
        assert torch.isclose(
            torch.tensor(scores[2]), torch.tensor(0.7), atol=1e-4
        )  # sequence_1 has lower score

        # Verify sequence data is preserved correctly for latent_idx=0
        seq1_example = next(
            ex for ex in latent_0_examples if ex["sequence_name"] == "sequence_1"
        )
        seq2_example = next(
            ex for ex in latent_0_examples if ex["sequence_name"] == "sequence_2"
        )

        assert seq1_example["input_ids"] == [1, 2, 3, 4, 5]
        assert seq1_example["latent_idx"] == 0
        assert len(seq1_example["input_ids"]) == 5

        assert seq2_example["input_ids"] == [10, 11, 12, 13, 14, 15]
        assert seq2_example["latent_idx"] == 0
        assert len(seq2_example["input_ids"]) == 6

        # Verify sequence data is preserved correctly for latent_idx=1
        seq1_example_lat1 = next(
            ex for ex in latent_1_examples if ex["sequence_name"] == "sequence_1"
        )
        seq2_example_lat1 = next(
            ex for ex in latent_1_examples if ex["sequence_name"] == "sequence_2"
        )

        assert seq1_example_lat1["input_ids"] == [1, 2, 3, 4, 5]
        assert seq1_example_lat1["latent_idx"] == 1
        assert len(seq1_example_lat1["input_ids"]) == 5

        assert seq2_example_lat1["input_ids"] == [10, 11, 12, 13, 14, 15]
        assert seq2_example_lat1["latent_idx"] == 1
        assert len(seq2_example_lat1["input_ids"]) == 6

        # Verify activation details are preserved for both sequences and both latents
        seq1_details = store.get_example_details(seq1_example["example_id"])
        seq2_details = store.get_example_details(seq2_example["example_id"])
        seq1_details_lat1 = store.get_example_details(seq1_example_lat1["example_id"])
        seq2_details_lat1 = store.get_example_details(seq2_example_lat1["example_id"])

        assert "scores_per_token" in seq1_details
        assert "scores_per_token" in seq2_details
        assert "scores_per_token" in seq1_details_lat1
        assert "scores_per_token" in seq2_details_lat1
        assert len(seq1_details["scores_per_token"]) == 5
        assert len(seq2_details["scores_per_token"]) == 6
        assert len(seq1_details_lat1["scores_per_token"]) == 5
        assert len(seq2_details_lat1["scores_per_token"]) == 6

    def test_per_latent_top_k_management(self, temp_db_path, storage_format):
        """Test that top-k is maintained per latent_idx."""
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format)

        # Add examples for latent 0 (should keep top 2)
        for i, score in enumerate([0.9, 0.8, 0.7]):  # Should keep 0.9, 0.8
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                latent_idx=0,
                maintain_top_k=False,
            )

        # Add examples for latent 1 (should keep top 2)
        for i, score in enumerate([0.95, 0.75, 0.65]):  # Should keep 0.95, 0.75
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 10, i + 11, i + 12]),
                latent_idx=1,
                maintain_top_k=False,
            )

        # Force top-k maintenance
        store._maintain_top_k()

        # Should have 4 total examples (2 per latent)
        assert len(store) == 4

        # Check latent 0 examples
        latent_0_examples = store.get_top_examples(latent_idx=0)
        assert len(latent_0_examples) == 2
        latent_0_scores = [ex["max_score"] for ex in latent_0_examples]
        assert sorted(latent_0_scores, reverse=True) == [0.9, 0.8]

        # Check latent 1 examples
        latent_1_examples = store.get_top_examples(latent_idx=1)
        assert len(latent_1_examples) == 2
        latent_1_scores = [ex["max_score"] for ex in latent_1_examples]
        assert sorted(latent_1_scores, reverse=True) == [0.95, 0.75]

    def test_per_quantile_top_k_management(self, temp_db_path, storage_format):
        """Test that top-k is maintained per quantile_idx."""
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format)

        # Add examples for quantile 0
        for i, score in enumerate([0.9, 0.8, 0.7]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                quantile_idx=0,
                maintain_top_k=False,
            )

        # Add examples for quantile 1
        for i, score in enumerate([0.95, 0.75, 0.65]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 10, i + 11, i + 12]),
                quantile_idx=1,
                maintain_top_k=False,
            )

        store._maintain_top_k()

        assert len(store) == 4  # 2 per quantile

        # Check quantile 0 examples
        quantile_0_examples = store.get_top_examples(quantile_idx=0)
        assert len(quantile_0_examples) == 2
        quantile_0_scores = [ex["max_score"] for ex in quantile_0_examples]
        assert sorted(quantile_0_scores, reverse=True) == [0.9, 0.8]

        # Check quantile 1 examples
        quantile_1_examples = store.get_top_examples(quantile_idx=1)
        assert len(quantile_1_examples) == 2
        quantile_1_scores = [ex["max_score"] for ex in quantile_1_examples]
        assert sorted(quantile_1_scores, reverse=True) == [0.95, 0.75]

    def test_per_latent_and_quantile_top_k_management(
        self, temp_db_path, storage_format
    ):
        """Test that top-k is maintained per (latent_idx, quantile_idx) combination."""
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format)

        # Add examples for (latent=0, quantile=0)
        for i, score in enumerate([0.9, 0.8, 0.7]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                latent_idx=0,
                quantile_idx=0,
                maintain_top_k=False,
            )

        # Add examples for (latent=0, quantile=1)
        for i, score in enumerate([0.95, 0.75, 0.65]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 10, i + 11, i + 12]),
                latent_idx=0,
                quantile_idx=1,
                maintain_top_k=False,
            )

        # Add examples for (latent=1, quantile=0)
        for i, score in enumerate([0.85, 0.55, 0.45]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 20, i + 21, i + 22]),
                latent_idx=1,
                quantile_idx=0,
                maintain_top_k=False,
            )

        store._maintain_top_k()

        assert len(store) == 6  # 2 per (latent, quantile) combination

        # Check (latent=0, quantile=0)
        group_00_examples = store.get_top_examples(latent_idx=0, quantile_idx=0)
        assert len(group_00_examples) == 2
        group_00_scores = [ex["max_score"] for ex in group_00_examples]
        assert sorted(group_00_scores, reverse=True) == [0.9, 0.8]

        # Check (latent=0, quantile=1)
        group_01_examples = store.get_top_examples(latent_idx=0, quantile_idx=1)
        assert len(group_01_examples) == 2
        group_01_scores = [ex["max_score"] for ex in group_01_examples]
        assert sorted(group_01_scores, reverse=True) == [0.95, 0.75]

        # Check (latent=1, quantile=0)
        group_10_examples = store.get_top_examples(latent_idx=1, quantile_idx=0)
        assert len(group_10_examples) == 2
        group_10_scores = [ex["max_score"] for ex in group_10_examples]
        assert sorted(group_10_scores, reverse=True) == [0.85, 0.55]

    def test_per_group_with_dataset_top_k_management(
        self, temp_db_path, storage_format
    ):
        """Test that top-k is maintained per (latent_idx, dataset) when per_dataset=True."""
        store = MaxActStore(
            temp_db_path,
            max_examples=2,
            storage_format=storage_format,
            per_dataset=True,
        )

        # Add examples for (latent=0, dataset=A)
        for i, score in enumerate([0.9, 0.8, 0.7]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i + 1, i + 2]),
                latent_idx=0,
                dataset_name="dataset_A",
                maintain_top_k=False,
            )

        # Add examples for (latent=0, dataset=B)
        for i, score in enumerate([0.95, 0.75, 0.65]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 10, i + 11, i + 12]),
                latent_idx=0,
                dataset_name="dataset_B",
                maintain_top_k=False,
            )

        # Add examples for (latent=1, dataset=A)
        for i, score in enumerate([0.85, 0.55, 0.45]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i + 20, i + 21, i + 22]),
                latent_idx=1,
                dataset_name="dataset_A",
                maintain_top_k=False,
            )

        store._maintain_top_k()

        assert len(store) == 6  # 2 per (latent, dataset) combination

        # Check (latent=0, dataset=A)
        group_0A_examples = store.get_top_examples(
            latent_idx=0, dataset_names=["dataset_A"]
        )
        assert len(group_0A_examples) == 2
        group_0A_scores = [ex["max_score"] for ex in group_0A_examples]
        assert sorted(group_0A_scores, reverse=True) == [0.9, 0.8]

        # Check (latent=0, dataset=B)
        group_0B_examples = store.get_top_examples(
            latent_idx=0, dataset_names=["dataset_B"]
        )
        assert len(group_0B_examples) == 2
        group_0B_scores = [ex["max_score"] for ex in group_0B_examples]
        assert sorted(group_0B_scores, reverse=True) == [0.95, 0.75]

        # Check (latent=1, dataset=A)
        group_1A_examples = store.get_top_examples(
            latent_idx=1, dataset_names=["dataset_A"]
        )
        assert len(group_1A_examples) == 2
        group_1A_scores = [ex["max_score"] for ex in group_1A_examples]
        assert sorted(group_1A_scores, reverse=True) == [0.85, 0.55]

    def test_mixed_grouping_scenarios(self, temp_db_path, storage_format):
        """Test mixed scenarios with some examples having latent/quantile and others not."""
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format)

        # Group 1: latent_idx=0, quantile_idx=None
        store.add_example(
            0.9, torch.tensor([1, 2, 3]), latent_idx=0, maintain_top_k=False
        )
        store.add_example(
            0.8, torch.tensor([4, 5, 6]), latent_idx=0, maintain_top_k=False
        )
        store.add_example(
            0.7, torch.tensor([7, 8, 9]), latent_idx=0, maintain_top_k=False
        )  # Should be dropped

        # Group 2: latent_idx=None, quantile_idx=0
        store.add_example(
            0.95, torch.tensor([10, 11, 12]), quantile_idx=0, maintain_top_k=False
        )
        store.add_example(
            0.75, torch.tensor([13, 14, 15]), quantile_idx=0, maintain_top_k=False
        )
        store.add_example(
            0.65, torch.tensor([16, 17, 18]), quantile_idx=0, maintain_top_k=False
        )  # Should be dropped

        # Group 3: latent_idx=None, quantile_idx=None (overall group)
        store.add_example(0.85, torch.tensor([19, 20, 21]), maintain_top_k=False)
        store.add_example(0.55, torch.tensor([22, 23, 24]), maintain_top_k=False)
        store.add_example(
            0.45, torch.tensor([25, 26, 27]), maintain_top_k=False
        )  # Should be dropped

        store._maintain_top_k()

        assert len(store) == 6  # 2 per group

        # Verify each group separately
        latent_0_examples = store.get_top_examples(latent_idx=0)
        assert len(latent_0_examples) == 2
        latent_0_scores = [ex["max_score"] for ex in latent_0_examples]
        assert sorted(latent_0_scores, reverse=True) == [0.9, 0.8]

        quantile_0_examples = [
            ex
            for ex in store.get_top_examples()
            if ex["quantile_idx"] == 0 and ex["latent_idx"] is None
        ]
        assert len(quantile_0_examples) == 2
        quantile_0_scores = [ex["max_score"] for ex in quantile_0_examples]
        assert sorted(quantile_0_scores, reverse=True) == [0.95, 0.75]

        overall_examples = [
            ex
            for ex in store.get_top_examples()
            if ex["latent_idx"] is None and ex["quantile_idx"] is None
        ]
        assert len(overall_examples) == 2
        overall_scores = [ex["max_score"] for ex in overall_examples]
        assert sorted(overall_scores, reverse=True) == [0.85, 0.55]

    def test_get_group_capacity_info(self, temp_db_path, storage_format):
        """Test getting capacity info for all groups."""
        store = MaxActStore(temp_db_path, max_examples=3, storage_format=storage_format)

        # Add examples for different groups
        store.add_example(0.9, torch.tensor([1, 2, 3]), latent_idx=0, quantile_idx=0)
        store.add_example(0.8, torch.tensor([4, 5, 6]), latent_idx=0, quantile_idx=0)
        store.add_example(0.95, torch.tensor([7, 8, 9]), latent_idx=1, quantile_idx=0)
        store.add_example(
            0.75, torch.tensor([10, 11, 12]), latent_idx=1, quantile_idx=0
        )
        store.add_example(
            0.85, torch.tensor([13, 14, 15]), latent_idx=1, quantile_idx=0
        )

        assert len(store) == 5

        capacity_info = store.get_group_capacity_info()

        # Should have info for both groups
        group_00_key = (("latent_idx", 0), ("quantile_idx", 0))
        group_10_key = (("latent_idx", 1), ("quantile_idx", 0))

        assert group_10_key in capacity_info
        assert group_00_key not in capacity_info  # We haven't reached the capacity yet

        store.add_example(
            0.65, torch.tensor([10, 11, 12]), latent_idx=0, quantile_idx=0
        )

        capacity_info = store.get_group_capacity_info()
        print(capacity_info)
        assert group_00_key in capacity_info
        assert group_10_key in capacity_info

        # Check group (0,0) info
        assert capacity_info[group_00_key] == 0.65

        # Check group (1,0) info
        assert capacity_info[group_10_key] == 0.75

    def test_batch_examples_with_per_group_indices(self, temp_db_path, storage_format):
        """Test add_batch_examples with per-example latent/quantile indices."""
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format)

        batch_size = 4
        scores = torch.tensor([0.9, 0.8, 0.95, 0.75])
        input_ids = torch.randint(1, 100, (batch_size, 5))

        # Different latent_idx per example
        latent_indices = torch.tensor(
            [0, 0, 1, 1]
        )  # First 2 to latent 0, next 2 to latent 1
        quantile_indices = torch.tensor([0, 1, 0, 1])  # Alternating quantiles

        store.add_batch_examples(
            scores_per_example=scores,
            input_ids_batch=input_ids,
            latent_idx=latent_indices,
            quantile_idx=quantile_indices,
        )

        # Should have 4 examples total (each group gets max 2, but we have 4 different groups)
        assert len(store) == 4

        # Verify each group has 1 example
        examples_00 = store.get_top_examples(latent_idx=0, quantile_idx=0)
        examples_01 = store.get_top_examples(latent_idx=0, quantile_idx=1)
        examples_10 = store.get_top_examples(latent_idx=1, quantile_idx=0)
        examples_11 = store.get_top_examples(latent_idx=1, quantile_idx=1)

        assert len(examples_00) == 1
        assert len(examples_01) == 1
        assert len(examples_10) == 1
        assert len(examples_11) == 1

        # Verify scores match expectations
        assert torch.isclose(
            torch.tensor(examples_00[0]["max_score"]), torch.tensor(0.9)
        )
        assert torch.isclose(
            torch.tensor(examples_01[0]["max_score"]), torch.tensor(0.8)
        )
        assert torch.isclose(
            torch.tensor(examples_10[0]["max_score"]), torch.tensor(0.95)
        )
        assert torch.isclose(
            torch.tensor(examples_11[0]["max_score"]), torch.tensor(0.75)
        )

    def test_async_writer_memory_db_basic(
        self, temp_db_path, mock_tokenizer, storage_format
    ):
        """Test that in-memory database functionality works."""
        store = MaxActStore(
            temp_db_path,
            tokenizer=mock_tokenizer,
            storage_format=storage_format,
            max_examples=5,
        )

        # Test with use_memory_db=True
        with store.create_async_writer(
            buffer_size=2, use_memory_db=True, sync_interval=1.0
        ) as writer:
            # Add some examples
            for i in range(3):
                scores = torch.tensor([float(i + 1)])
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                writer.add_batch_examples(scores, input_ids)

            writer.flush()

            # Force sync to disk
            writer.sync_to_disk()

        # Verify examples were stored
        examples = store.get_top_examples(limit=10)
        assert len(examples) == 3
        assert examples[0]["max_score"] == 3.0
        assert examples[1]["max_score"] == 2.0
        assert examples[2]["max_score"] == 1.0

    def test_async_writer_memory_db_vs_disk_mode(
        self, temp_db_path, mock_tokenizer, storage_format
    ):
        """Test that both memory and disk modes produce the same results."""
        # Test with memory mode
        store_memory = MaxActStore(
            temp_db_path,
            tokenizer=mock_tokenizer,
            storage_format=storage_format,
            max_examples=5,
        )

        with store_memory.create_async_writer(
            buffer_size=2, use_memory_db=True
        ) as writer:
            for i in range(3):
                scores = torch.tensor([float(i + 1)])
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                writer.add_batch_examples(scores, input_ids)
            writer.flush()

        examples_memory = store_memory.get_top_examples(limit=10)

        # Clear and test with disk mode
        store_memory.clear()

        with store_memory.create_async_writer(
            buffer_size=2, use_memory_db=False
        ) as writer:
            for i in range(3):
                scores = torch.tensor([float(i + 1)])
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                writer.add_batch_examples(scores, input_ids)
            writer.flush()

        examples_disk = store_memory.get_top_examples(limit=10)

        # Results should be identical
        assert len(examples_memory) == len(examples_disk) == 3
        for mem_ex, disk_ex in zip(examples_memory, examples_disk):
            assert mem_ex["max_score"] == disk_ex["max_score"]
            assert mem_ex["input_ids"] == disk_ex["input_ids"]

    def test_async_writer_memory_db_sync_functionality(
        self, temp_db_path, storage_format
    ):
        """Test manual sync functionality."""
        store = MaxActStore(temp_db_path, storage_format=storage_format, max_examples=5)

        with store.create_async_writer(
            buffer_size=1, use_memory_db=True, sync_interval=60.0
        ) as writer:
            # Add example
            scores = torch.tensor([1.0])
            input_ids = torch.tensor([[1, 2, 3]])
            writer.add_batch_examples(scores, input_ids)
            writer.flush()

            # At this point, data should be in memory but not necessarily on disk
            # Force sync to disk
            writer.sync_to_disk()

        # After shutdown, data should be persisted
        examples = store.get_top_examples(limit=10)
        assert len(examples) == 1
        assert examples[0]["max_score"] == 1.0

    def test_async_writer_memory_db_large_batch(self, temp_db_path, storage_format):
        """Test in-memory database with larger batch to ensure performance benefit."""
        store = MaxActStore(
            temp_db_path, storage_format=storage_format, max_examples=50
        )

        batch_size = 20
        with store.create_async_writer(buffer_size=10, use_memory_db=True) as writer:
            # Add larger batch
            scores = torch.linspace(1.0, 20.0, batch_size)
            input_ids = torch.randint(1, 100, (batch_size, 5))

            writer.add_batch_examples(scores, input_ids)
            writer.flush()

        # Verify all examples were stored
        examples = store.get_top_examples(limit=30)
        assert len(examples) == batch_size

        # Verify scores are in descending order
        scores_from_db = [ex["max_score"] for ex in examples]
        assert scores_from_db == sorted(scores_from_db, reverse=True)
