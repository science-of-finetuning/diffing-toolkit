"""Tests for grader pure computation functions (no API calls)."""

import pytest

from diffing.utils.graders.coherence_grader import (
    _build_user_prompt as coherence_build_prompt,
)
from diffing.utils.graders.coherence_grader import _parse_final_label

from diffing.utils.graders.token_relevance_grader import (
    _build_user_prompt_many,
    _parse_indexed_labels,
)
from diffing.utils.graders.token_relevance_grader import TokenRelevanceGrader

from diffing.utils.graders.hypothesis_grader import (
    _build_user_prompt as hypothesis_build_prompt,
    _parse_score,
)

from diffing.utils.graders.patch_scope_grader import (
    _format_token_list,
    _round_scale_one_decimal,
    _remove_artifacts,
    _build_user_prompt as patchscope_build_prompt,
    _parse_best_and_tokens,
)


class TestCoherenceGrader:
    """Tests for coherence grader pure functions."""

    def test_build_user_prompt_basic(self):
        """Test building user prompt with simple answer."""
        result = coherence_build_prompt("Hello world")
        assert "[ANSWER START]" in result
        assert "[ANSWER END]" in result
        assert "Hello world" in result

    def test_build_user_prompt_preserves_content(self):
        """Test that prompt preserves answer content."""
        answer = "Photosynthesis is the process by which plants convert light."
        result = coherence_build_prompt(answer)
        assert answer in result

    def test_build_user_prompt_empty_fails(self):
        """Test that empty answer raises assertion."""
        with pytest.raises(AssertionError):
            coherence_build_prompt("")

    def test_build_user_prompt_whitespace_only_fails(self):
        """Test that whitespace-only answer raises assertion."""
        with pytest.raises(AssertionError):
            coherence_build_prompt("   ")

    def test_parse_final_label_coherent(self):
        """Test parsing COHERENT label."""
        text = "Some reasoning here.\nANSWER: COHERENT"
        assert _parse_final_label(text) == "COHERENT"

    def test_parse_final_label_incoherent(self):
        """Test parsing INCOHERENT label."""
        text = "Some reasoning here.\nANSWER: INCOHERENT"
        assert _parse_final_label(text) == "INCOHERENT"

    def test_parse_final_label_case_insensitive(self):
        """Test case insensitivity."""
        assert _parse_final_label("answer: coherent") == "COHERENT"
        assert _parse_final_label("Answer: Incoherent") == "INCOHERENT"
        assert _parse_final_label("ANSWER: coherent") == "COHERENT"

    def test_parse_final_label_with_punctuation(self):
        """Test tolerance of trailing punctuation."""
        assert _parse_final_label("ANSWER: COHERENT.") == "COHERENT"
        assert _parse_final_label("ANSWER: INCOHERENT!") == "INCOHERENT"

    def test_parse_final_label_last_wins(self):
        """Test that last ANSWER line wins."""
        text = "ANSWER: COHERENT\nMore reasoning.\nANSWER: INCOHERENT"
        assert _parse_final_label(text) == "INCOHERENT"

    def test_parse_final_label_unknown(self):
        """Test UNKNOWN for missing label."""
        assert _parse_final_label("No answer here") == "UNKNOWN"
        assert _parse_final_label("ANSWER: MAYBE") == "UNKNOWN"

    def test_parse_final_label_with_whitespace(self):
        """Test handling of whitespace around label."""
        assert _parse_final_label("  ANSWER:   COHERENT  ") == "COHERENT"
        assert _parse_final_label("ANSWER:INCOHERENT") == "INCOHERENT"


class TestTokenRelevanceGrader:
    """Tests for token relevance grader pure functions."""

    def test_build_user_prompt_many_basic(self):
        """Test building multi-token prompt."""
        result = _build_user_prompt_many(
            description="Legal documents about contracts",
            frequent_tokens=["contract", "law"],
            candidate_tokens=["agreement", "banana"],
        )
        assert "[DESCRIPTION]" in result
        assert "[FREQUENT TOKENS]" in result
        assert "[CANDIDATE TOKENS]" in result
        assert "Legal documents" in result
        assert "1. agreement" in result
        assert "2. banana" in result

    def test_build_user_prompt_many_empty_frequent_tokens(self):
        """Test with no frequent tokens."""
        result = _build_user_prompt_many(
            description="Some domain",
            frequent_tokens=[],
            candidate_tokens=["token1"],
        )
        assert "(none)" in result

    def test_build_user_prompt_many_empty_candidates_fails(self):
        """Test that empty candidates raises assertion."""
        with pytest.raises(AssertionError):
            _build_user_prompt_many(
                description="Some domain",
                frequent_tokens=["tok"],
                candidate_tokens=[],
            )

    def test_parse_indexed_labels_basic(self):
        """Test parsing indexed labels."""
        text = "Reasoning...\nANSWER[1]: RELEVANT\nANSWER[2]: IRRELEVANT"
        labels = _parse_indexed_labels(text, 2)
        assert labels == ["RELEVANT", "IRRELEVANT"]

    def test_parse_indexed_labels_case_insensitive(self):
        """Test case insensitivity."""
        text = "answer[1]: relevant\nAnswer[2]: Irrelevant"
        labels = _parse_indexed_labels(text, 2)
        assert labels == ["RELEVANT", "IRRELEVANT"]

    def test_parse_indexed_labels_missing_indices(self):
        """Test UNKNOWN for missing indices."""
        text = "ANSWER[1]: RELEVANT\nANSWER[3]: IRRELEVANT"
        labels = _parse_indexed_labels(text, 3)
        assert labels == ["RELEVANT", "UNKNOWN", "IRRELEVANT"]

    def test_parse_indexed_labels_last_wins(self):
        """Test that last answer for an index wins."""
        text = "ANSWER[1]: RELEVANT\nANSWER[1]: IRRELEVANT"
        labels = _parse_indexed_labels(text, 1)
        assert labels == ["IRRELEVANT"]

    def test_parse_indexed_labels_out_of_range_ignored(self):
        """Test that out-of-range indices are ignored."""
        text = "ANSWER[1]: RELEVANT\nANSWER[5]: IRRELEVANT"
        labels = _parse_indexed_labels(text, 2)
        assert labels == ["RELEVANT", "UNKNOWN"]

    def test_rotated_indices_no_shift(self):
        """Test rotation with zero shift."""
        result = TokenRelevanceGrader._rotated_indices(5, 0)
        assert result == [0, 1, 2, 3, 4]

    def test_rotated_indices_shift_one(self):
        """Test rotation with shift of 1."""
        result = TokenRelevanceGrader._rotated_indices(5, 1)
        assert result == [1, 2, 3, 4, 0]

    def test_rotated_indices_shift_wrap(self):
        """Test rotation with shift equal to length."""
        result = TokenRelevanceGrader._rotated_indices(4, 4)
        assert result == [0, 1, 2, 3]

    def test_rotated_indices_shift_larger_than_length(self):
        """Test rotation with shift larger than length."""
        result = TokenRelevanceGrader._rotated_indices(3, 5)
        assert result == [2, 0, 1]

    def test_majority_vote_unanimous(self):
        """Test majority vote with unanimous labels."""
        permutation_labels = [
            ["RELEVANT", "IRRELEVANT"],
            ["RELEVANT", "IRRELEVANT"],
            ["RELEVANT", "IRRELEVANT"],
        ]
        result = TokenRelevanceGrader._majority_vote_per_position(permutation_labels)
        assert result == ["RELEVANT", "IRRELEVANT"]

    def test_majority_vote_simple_majority(self):
        """Test majority vote with simple majority."""
        permutation_labels = [
            ["RELEVANT", "RELEVANT"],
            ["RELEVANT", "IRRELEVANT"],
            ["IRRELEVANT", "IRRELEVANT"],
        ]
        result = TokenRelevanceGrader._majority_vote_per_position(permutation_labels)
        assert result == ["RELEVANT", "IRRELEVANT"]

    def test_majority_vote_tie_returns_unknown(self):
        """Test majority vote with tie returns UNKNOWN."""
        permutation_labels = [
            ["RELEVANT", "IRRELEVANT"],
            ["IRRELEVANT", "RELEVANT"],
        ]
        result = TokenRelevanceGrader._majority_vote_per_position(permutation_labels)
        assert result == ["UNKNOWN", "UNKNOWN"]

    def test_majority_vote_ignores_unknown(self):
        """Test that UNKNOWN votes are ignored in majority calculation."""
        permutation_labels = [
            ["RELEVANT", "UNKNOWN"],
            ["RELEVANT", "IRRELEVANT"],
            ["UNKNOWN", "IRRELEVANT"],
        ]
        result = TokenRelevanceGrader._majority_vote_per_position(permutation_labels)
        assert result == ["RELEVANT", "IRRELEVANT"]

    def test_majority_vote_all_unknown(self):
        """Test majority vote when all votes are UNKNOWN."""
        permutation_labels = [
            ["UNKNOWN", "UNKNOWN"],
            ["UNKNOWN", "UNKNOWN"],
        ]
        result = TokenRelevanceGrader._majority_vote_per_position(permutation_labels)
        assert result == ["UNKNOWN", "UNKNOWN"]


class TestHypothesisGrader:
    """Tests for hypothesis grader pure functions."""

    def test_build_user_prompt_basic(self):
        """Test building hypothesis prompt."""
        result = hypothesis_build_prompt(
            description="Legal finetune on court cases",
            rubric_instruction="1: Bad\n5: Good",
            hypothesis="The model learned about law.",
        )
        assert "[DESCRIPTION]" in result
        assert "[RUBRIC]" in result
        assert "[HYPOTHESIS]" in result
        assert "Legal finetune" in result
        assert "1: Bad" in result
        assert "The model learned" in result

    def test_build_user_prompt_empty_description_fails(self):
        """Test that empty description raises assertion."""
        with pytest.raises(AssertionError):
            hypothesis_build_prompt("", "rubric", "hypothesis")

    def test_build_user_prompt_empty_rubric_fails(self):
        """Test that empty rubric raises assertion."""
        with pytest.raises(AssertionError):
            hypothesis_build_prompt("description", "", "hypothesis")

    def test_build_user_prompt_empty_hypothesis_fails(self):
        """Test that empty hypothesis raises assertion."""
        with pytest.raises(AssertionError):
            hypothesis_build_prompt("description", "rubric", "")

    def test_parse_score_valid_scores(self):
        """Test parsing valid scores 1-5."""
        for score in range(1, 6):
            text = f"Reasoning here.\nSCORE: {score}"
            assert _parse_score(text) == score

    def test_parse_score_case_insensitive(self):
        """Test case insensitivity."""
        assert _parse_score("score: 3") == 3
        assert _parse_score("Score: 4") == 4
        assert _parse_score("SCORE: 5") == 5

    def test_parse_score_last_wins(self):
        """Test that last SCORE line wins."""
        text = "SCORE: 1\nMore reasoning.\nSCORE: 5"
        assert _parse_score(text) == 5

    def test_parse_score_no_score_fails(self):
        """Test that missing score raises assertion."""
        with pytest.raises(AssertionError):
            _parse_score("No score here")

    def test_parse_score_invalid_score_fails(self):
        """Test that invalid score (0, 6) raises assertion."""
        with pytest.raises(AssertionError):
            _parse_score("SCORE: 0")
        with pytest.raises(AssertionError):
            _parse_score("SCORE: 6")


class TestPatchScopeGrader:
    """Tests for patchscope grader pure functions."""

    def test_format_token_list_basic(self):
        """Test formatting token list."""
        result = _format_token_list(["bake", "cake", "oven"])
        assert result == '"bake", "cake", "oven"'

    def test_format_token_list_single(self):
        """Test formatting single token."""
        result = _format_token_list(["token"])
        assert result == '"token"'

    def test_format_token_list_empty_fails(self):
        """Test that empty list raises assertion."""
        with pytest.raises(AssertionError):
            _format_token_list([])

    def test_format_token_list_filters_empty_strings(self):
        """Test that empty strings are filtered."""
        result = _format_token_list(["a", "", "b"])
        assert result == '"a", "b"'

    def test_round_scale_one_decimal(self):
        """Test rounding to one decimal."""
        assert _round_scale_one_decimal(10.0) == 10.0
        assert _round_scale_one_decimal(10.123) == 10.1
        assert _round_scale_one_decimal(10.156) == 10.2
        assert _round_scale_one_decimal(10.04) == 10.0

    def test_remove_artifacts_basic(self):
        """Test removing artifact tokens."""
        tokens = ["bake", "->", "cake", "==>"]
        result = _remove_artifacts(tokens)
        assert result == ["bake", "cake"]

    def test_remove_artifacts_punctuation_only(self):
        """Test removing punctuation-only tokens."""
        tokens = ["word", ".,;:", "another"]
        result = _remove_artifacts(tokens)
        assert result == ["word", "another"]

    def test_remove_artifacts_preserves_valid(self):
        """Test that valid tokens are preserved."""
        tokens = ["bake", "cake", "10.5"]
        result = _remove_artifacts(tokens)
        assert result == ["bake", "cake", "10.5"]

    def test_remove_artifacts_arrow_patterns(self):
        """Test removal of arrow patterns."""
        tokens = ["a->b", "x=>y", "normal"]
        result = _remove_artifacts(tokens)
        assert result == ["normal"]

    def test_build_user_prompt_basic(self):
        """Test building patchscope prompt."""
        result = patchscope_build_prompt(
            scales=[0.0, 10.0],
            per_scale_tokens={
                0.0: ["the", "and"],
                10.0: ["bake", "cake"],
            },
        )
        assert "[TOKENS PER SCALE]" in result
        assert "[SCALES]" in result
        assert "SCALE: 0.0" in result
        assert "SCALE: 10.0" in result
        assert "bake" in result

    def test_build_user_prompt_filters_artifacts(self):
        """Test that artifacts are filtered from prompt."""
        result = patchscope_build_prompt(
            scales=[10.0],
            per_scale_tokens={10.0: ["->", "bake"]},
        )
        assert "bake" in result
        assert '"->"' not in result

    def test_parse_best_and_tokens_basic(self):
        """Test parsing best scale and tokens."""
        text = "Reasoning...\nBEST_SCALER: 10.0\nTOP_TOKENS: bake | cake | oven"
        best, tokens = _parse_best_and_tokens(text)
        assert best == 10.0
        assert tokens == ["bake", "cake", "oven"]

    def test_parse_best_and_tokens_case_insensitive(self):
        """Test case insensitivity."""
        text = "best_scaler: 5.5\ntop_tokens: a | b"
        best, tokens = _parse_best_and_tokens(text)
        assert best == 5.5
        assert tokens == ["a", "b"]

    def test_parse_best_and_tokens_no_tokens(self):
        """Test parsing when no tokens line exists."""
        text = "BEST_SCALER: 10.0"
        best, tokens = _parse_best_and_tokens(text)
        assert best == 10.0
        assert tokens == []

    def test_parse_best_and_tokens_no_best_fails(self):
        """Test that missing best scaler raises assertion."""
        with pytest.raises(AssertionError):
            _parse_best_and_tokens("TOP_TOKENS: a | b")

    def test_parse_best_and_tokens_negative_scale(self):
        """Test parsing negative scale."""
        text = "BEST_SCALER: -5.0\nTOP_TOKENS: tok"
        best, tokens = _parse_best_and_tokens(text)
        assert best == -5.0
        assert tokens == ["tok"]

    def test_parse_best_and_tokens_whitespace_handling(self):
        """Test handling of whitespace in tokens."""
        text = "BEST_SCALER: 10.0\nTOP_TOKENS:  a  |  b  |  c  "
        best, tokens = _parse_best_and_tokens(text)
        assert tokens == ["a", "b", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
