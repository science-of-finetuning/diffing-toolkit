"""
Unit tests for diff_mining method components.

Tests vectorized helper functions, token ordering types, and preprocessing
slicing utilities. All tests are CPU-only and require no GPU or model loading.
"""

import torch
import pytest
from transformers import AutoTokenizer

from diffing.methods.diff_mining.core_analysis import (
    vectorized_bincount_masked,
    vectorized_shortlist_counts,
    vectorized_cooccurrence_shortlist,
    vectorized_same_sign_cooccurrence,
)
from diffing.methods.diff_mining.token_ordering import (
    SharedTokenStats,
    TopKOccurringOrderingType,
    FractionPositiveDiffOrderingType,
)
from diffing.methods.diff_mining.preprocessing import (
    slice_to_positions,
    slice_to_positions_2d,
)

VOCAB_SIZE = 100
TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def _make_shared_stats(
    vocab_size: int = VOCAB_SIZE,
    total_positions: int = 200,
    num_samples: int = 10,
) -> SharedTokenStats:
    """Create synthetic SharedTokenStats for ordering tests."""
    torch.manual_seed(42)
    topk_pos = torch.randint(0, 50, (vocab_size,), dtype=torch.int64)
    topk_neg = torch.randint(0, 50, (vocab_size,), dtype=torch.int64)
    sum_diff = torch.randn(vocab_size)
    count_pos = torch.randint(0, total_positions, (vocab_size,), dtype=torch.int64)
    return SharedTokenStats(
        vocab_size=vocab_size,
        total_positions=total_positions,
        num_samples=num_samples,
        sum_logit_diff=sum_diff,
        count_positive=count_pos,
        topk_pos_counts=topk_pos,
        topk_neg_counts=topk_neg,
    )


# ---------------------------------------------------------------------------
# Vectorized helpers
# ---------------------------------------------------------------------------


class TestVectorizedBincountMasked:
    def test_basic_counting(self):
        """Counts token occurrences respecting mask."""
        indices = torch.tensor([[[0, 1], [2, 3]], [[0, 0], [1, 1]]])  # [2, 2, 2]
        mask = torch.ones(2, 2, dtype=torch.long)
        counts = vectorized_bincount_masked(indices, mask, vocab_size=5)
        assert counts.shape == (5,)
        assert counts[0].item() == 3  # appears 3 times
        assert counts[1].item() == 3  # appears 3 times

    def test_mask_zeros_out_positions(self):
        """Masked positions should not be counted."""
        indices = torch.tensor([[[0, 1], [2, 3]]])  # [1, 2, 2]
        mask = torch.tensor([[1, 0]])  # only first position valid
        counts = vectorized_bincount_masked(indices, mask, vocab_size=5)
        assert counts[0].item() == 1
        assert counts[1].item() == 1
        assert counts[2].item() == 0
        assert counts[3].item() == 0

    def test_all_masked(self):
        """All-zero mask should give zero counts."""
        indices = torch.tensor([[[5, 5], [5, 5]]])
        mask = torch.zeros(1, 2, dtype=torch.long)
        counts = vectorized_bincount_masked(indices, mask, vocab_size=10)
        assert counts.sum().item() == 0

    def test_output_dtype(self):
        indices = torch.tensor([[[0]]])
        mask = torch.ones(1, 1, dtype=torch.long)
        counts = vectorized_bincount_masked(indices, mask, vocab_size=5)
        assert counts.dtype == torch.int64


class TestVectorizedShortlistCounts:
    def test_per_sample_and_per_position(self):
        """Shortlist counting returns correct shapes and values."""
        # [2, 3, 2] top-k indices
        top_k_indices = torch.tensor(
            [
                [[10, 20], [30, 40], [10, 50]],
                [[20, 30], [10, 10], [60, 70]],
            ]
        )
        mask = torch.ones(2, 3, dtype=torch.long)
        shortlist = torch.tensor([10, 20])

        per_sample, per_position = vectorized_shortlist_counts(
            top_k_indices, mask, shortlist, start_idx=0
        )
        assert per_sample.shape == (2, 2)  # [batch, num_shortlist]
        assert per_position.shape == (3, 2)  # [seq, num_shortlist]

    def test_mask_respected(self):
        """Masked positions not counted in shortlist."""
        top_k_indices = torch.tensor([[[10, 20], [10, 20]]])
        mask = torch.tensor([[1, 0]])
        shortlist = torch.tensor([10])

        per_sample, per_position = vectorized_shortlist_counts(
            top_k_indices, mask, shortlist, start_idx=0
        )
        assert per_sample[0, 0].item() == 1  # only first position counted
        assert per_position[1, 0].item() == 0  # second position masked


class TestVectorizedCooccurrence:
    def test_symmetric(self):
        """Co-occurrence matrix is symmetric."""
        top_k_indices = torch.tensor(
            [
                [[0, 1], [1, 2], [0, 2]],
            ]
        )
        mask = torch.ones(1, 3, dtype=torch.long)
        shortlist = torch.tensor([0, 1, 2])
        cooc = vectorized_cooccurrence_shortlist(top_k_indices, mask, shortlist)
        assert cooc.shape == (3, 3)
        assert torch.equal(cooc, cooc.T)

    def test_diagonal_is_self_count(self):
        """Diagonal entries are the count of how often each token appears."""
        top_k_indices = torch.tensor([[[0, 1], [0, 2]]])
        mask = torch.ones(1, 2, dtype=torch.long)
        shortlist = torch.tensor([0, 1, 2])
        cooc = vectorized_cooccurrence_shortlist(top_k_indices, mask, shortlist)
        assert cooc[0, 0].item() == 2  # token 0 appears in 2 positions
        assert cooc[1, 1].item() == 1
        assert cooc[2, 2].item() == 1


class TestVectorizedSameSignCooccurrence:
    def test_same_sign_positive(self):
        """Tokens with same positive sign at same position co-occur."""
        diff = torch.tensor([[[1.0, 2.0, -1.0]]])  # [1, 1, 3]
        mask = torch.ones(1, 1, dtype=torch.long)
        shortlist = torch.tensor([0, 1, 2])
        cooc = vectorized_same_sign_cooccurrence(diff, mask, shortlist)
        assert cooc.shape == (3, 3)
        # tokens 0 and 1 both positive -> co-occur
        assert cooc[0, 1].item() == 1
        # tokens 0 and 2 have opposite sign -> no co-occurrence
        assert cooc[0, 2].item() == 0

    def test_mask_respected(self):
        """Masked positions excluded from same-sign co-occurrence."""
        diff = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])  # [1, 2, 2]
        mask = torch.tensor([[1, 0]])
        shortlist = torch.tensor([0, 1])
        cooc = vectorized_same_sign_cooccurrence(diff, mask, shortlist)
        assert cooc[0, 1].item() == 1  # only first position counted


# ---------------------------------------------------------------------------
# Token ordering types
# ---------------------------------------------------------------------------


class TestTopKOccurringOrdering:
    def test_produces_single_ordering(self, tokenizer):
        stats = _make_shared_stats()
        ordering_type = TopKOccurringOrderingType()
        result = ordering_type.compute_orderings(stats, tokenizer, num_tokens=10)
        assert result.ordering_type_id == "top_k_occurring"
        assert len(result.orderings) == 1
        assert len(result.orderings[0].tokens) == 10

    def test_tokens_sorted_by_occurrence_rate(self, tokenizer):
        stats = _make_shared_stats()
        ordering_type = TopKOccurringOrderingType()
        result = ordering_type.compute_orderings(stats, tokenizer, num_tokens=20)
        values = [t.ordering_value for t in result.orderings[0].tokens]
        assert values == sorted(values, reverse=True)

    def test_occurrence_rate_is_percentage(self, tokenizer):
        """Occurrence rate should be (count / total_positions) * 100."""
        stats = _make_shared_stats()
        ordering_type = TopKOccurringOrderingType()
        result = ordering_type.compute_orderings(stats, tokenizer, num_tokens=5)
        for token in result.orderings[0].tokens:
            expected = (
                stats.topk_pos_counts[token.token_id].item() / stats.total_positions
            ) * 100
            assert abs(token.ordering_value - expected) < 1e-4


class TestFractionPositiveDiffOrdering:
    def test_produces_single_ordering(self, tokenizer):
        stats = _make_shared_stats()
        ordering_type = FractionPositiveDiffOrderingType()
        result = ordering_type.compute_orderings(stats, tokenizer, num_tokens=10)
        assert result.ordering_type_id == "fraction_positive_diff"
        assert len(result.orderings) == 1
        assert len(result.orderings[0].tokens) == 10

    def test_fraction_in_zero_one(self, tokenizer):
        """Fraction positive should be in [0, 1]."""
        stats = _make_shared_stats()
        ordering_type = FractionPositiveDiffOrderingType()
        result = ordering_type.compute_orderings(stats, tokenizer, num_tokens=20)
        for token in result.orderings[0].tokens:
            assert 0.0 <= token.ordering_value <= 1.0


# ---------------------------------------------------------------------------
# SharedTokenStats
# ---------------------------------------------------------------------------


class TestSharedTokenStats:
    def test_shape_validation(self):
        """SharedTokenStats rejects mismatched tensor shapes."""
        with pytest.raises(AssertionError):
            SharedTokenStats(
                vocab_size=10,
                total_positions=100,
                num_samples=5,
                sum_logit_diff=torch.zeros(10),
                count_positive=torch.zeros(10, dtype=torch.int64),
                topk_pos_counts=torch.zeros(5, dtype=torch.int64),  # wrong size
                topk_neg_counts=torch.zeros(10, dtype=torch.int64),
            )

    def test_valid_construction(self):
        stats = _make_shared_stats(vocab_size=50)
        assert stats.vocab_size == 50
        assert stats.sum_logit_diff.shape == (50,)


# ---------------------------------------------------------------------------
# Preprocessing: position slicing
# ---------------------------------------------------------------------------


class TestSliceToPositions:
    def test_3d_slicing(self):
        """slice_to_positions extracts correct positions from [batch, seq, vocab]."""
        tensor = torch.arange(24).reshape(2, 3, 4).float()
        positions = [[0, 2], [1]]
        result = slice_to_positions(tensor, positions)
        assert result.shape == (2, 2, 4)
        assert torch.equal(result[0, 0], tensor[0, 0])
        assert torch.equal(result[0, 1], tensor[0, 2])
        assert torch.equal(result[1, 0], tensor[1, 1])

    def test_2d_slicing(self):
        """slice_to_positions_2d extracts correct positions from [batch, seq]."""
        tensor = torch.arange(6).reshape(2, 3)
        positions = [[0, 2], [1]]
        result = slice_to_positions_2d(tensor, positions)
        assert result.shape == (2, 2)
        assert result[0, 0].item() == tensor[0, 0].item()
        assert result[0, 1].item() == tensor[0, 2].item()
        assert result[1, 0].item() == tensor[1, 1].item()

    def test_padding_for_unequal_positions(self):
        """Shorter position lists should be zero-padded."""
        tensor = torch.ones(2, 5, 3)
        positions = [[0, 1, 2], [0]]
        result = slice_to_positions(tensor, positions)
        assert result.shape == (2, 3, 3)
        # Second sample's positions 1,2 should be zeros (padding)
        assert result[1, 1].sum().item() == 0.0
        assert result[1, 2].sum().item() == 0.0
