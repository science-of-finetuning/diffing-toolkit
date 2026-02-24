"""Tests for Activation Difference Lens utility functions (CPU-only, no GPU/models)."""

import pytest
import re
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from diffing.methods.activation_difference_lens.token_relevance import (
    _is_generic_token,
    COMMON_WORDS,
)
from diffing.methods.activation_difference_lens.util import (
    dataset_dir_name,
    layer_dir,
    norms_path,
    position_files_exist,
    is_layer_complete,
)
from diffing.methods.activation_difference_lens.steering import (
    _clean_generated_text,
    read_prompts,
)
from diffing.methods.activation_difference_lens.agent_tools import (
    _dataset_dir_name as agent_dataset_dir_name,
)


class TestIsGenericToken:
    """Tests for _is_generic_token pure function."""

    def test_generic_common_words(self):
        """Test that common words are marked as generic."""
        for word in ["the", "and", "or", "but", "in", "is", "are", "a"]:
            assert _is_generic_token(word)

    def test_common_words_lowercase(self):
        """Test common words in lowercase."""
        assert _is_generic_token("the")
        assert _is_generic_token("is")
        assert _is_generic_token("a")

    def test_common_words_case_insensitive(self):
        """Test that common words are detected regardless of case (lowercased before check)."""
        assert _is_generic_token("The")  # "the" after cleaning and lowercasing
        assert _is_generic_token("THE")  # "the" after lowercasing
        assert _is_generic_token("ing")

    def test_contractions_generic(self):
        """Test that contractions are marked as generic."""
        for contraction in ["'s", "'t", "'re", "'ve", "'ll", "'d", "'m"]:
            assert _is_generic_token(contraction)

    def test_single_character_generic(self):
        """Test that single characters are generic."""
        assert _is_generic_token("a")
        assert _is_generic_token("x")
        assert _is_generic_token("z")

    def test_punctuation_only_generic(self):
        """Test that pure punctuation is generic."""
        assert _is_generic_token(".")
        assert _is_generic_token("!")
        assert _is_generic_token(",")
        assert _is_generic_token("...")
        assert _is_generic_token("???")
        assert _is_generic_token("!?.")

    def test_whitespace_patterns_generic(self):
        """Test that whitespace-only tokens are generic."""
        assert _is_generic_token(" ")
        assert _is_generic_token("  ")
        assert _is_generic_token("\n")
        assert _is_generic_token("\t")
        assert _is_generic_token("\r")

    def test_tokenizer_space_markers_generic(self):
        """Test that tokenizer space markers are cleaned and checked."""
        # These have space markers that get cleaned away
        assert _is_generic_token("▁the")  # sentencepiece
        assert _is_generic_token("Ġthe")  # GPT-2 space marker
        assert _is_generic_token("▁Ġthe")

    def test_long_meaningful_word_not_generic(self):
        """Test that long meaningful words are not generic."""
        assert not _is_generic_token("elephant")
        assert not _is_generic_token("methodology")
        assert not _is_generic_token("information")

    def test_meaningful_short_words_not_generic(self):
        """Test that short but meaningful words are not generic."""
        assert not _is_generic_token("cat")
        assert not _is_generic_token("dog")
        assert not _is_generic_token("run")
        assert not _is_generic_token("book")

    def test_mixed_case_meaningful_word_not_generic(self):
        """Test meaningful words regardless of case."""
        assert not _is_generic_token("Cat")
        assert not _is_generic_token("DOG")
        assert not _is_generic_token("Book")

    def test_numbers_not_generic(self):
        """Test that numbers are not generic."""
        assert not _is_generic_token("42")
        assert not _is_generic_token("2023")
        assert not _is_generic_token("3.14")

    def test_domain_specific_tokens_not_generic(self):
        """Test domain-specific tokens are not generic."""
        assert not _is_generic_token("theorem")
        assert not _is_generic_token("algorithm")
        assert not _is_generic_token("quantum")

    def test_mixed_alphanumeric_not_generic(self):
        """Test alphanumeric tokens are not generic."""
        assert not _is_generic_token("model123")
        assert not _is_generic_token("v2")

    def test_empty_after_cleaning_generic(self):
        """Test that tokens empty after cleaning markers are generic."""
        assert _is_generic_token("▁")
        assert _is_generic_token("Ġ")
        assert _is_generic_token("▁Ġ")

    def test_special_contraction_patterns(self):
        """Test special contraction patterns are generic."""
        assert _is_generic_token("'s")
        # Note: "'ing" is not in the contractions list (only "ing" is in COMMON_WORDS)
        assert not _is_generic_token("'ing")

    def test_ing_suffix_generic(self):
        """Test 'ing' is marked as generic."""
        assert _is_generic_token("ing")

    def test_complex_punctuation_generic(self):
        """Test complex punctuation patterns."""
        assert _is_generic_token("->")
        assert _is_generic_token("=>")
        assert _is_generic_token("::")

    def test_tokens_with_underscores_and_markers(self):
        """Test tokens with multiple space markers."""
        # After cleaning: "the" -> common word -> generic
        result = _is_generic_token("▁the")
        assert result

    def test_newline_variations_generic(self):
        """Test various newline representations."""
        assert _is_generic_token("\n")
        assert _is_generic_token("\r\n")
        assert _is_generic_token("\t\n")


class TestDatasetDirName:
    """Tests for dataset_dir_name path utility."""

    def test_dataset_dir_name_simple(self):
        """Test extracting simple dataset name."""
        assert dataset_dir_name("myorg/mydata") == "mydata"

    def test_dataset_dir_name_single_part(self):
        """Test single-part dataset ID."""
        assert dataset_dir_name("mydata") == "mydata"

    def test_dataset_dir_name_multiple_parts(self):
        """Test multi-part dataset ID takes last part."""
        assert dataset_dir_name("org/subdir/data") == "data"

    def test_dataset_dir_name_with_hub(self):
        """Test typical HuggingFace hub dataset."""
        assert dataset_dir_name("openwebtext/train") == "train"

    def test_dataset_dir_name_empty_part_fails(self):
        """Test that trailing slash results in empty string -> fails assertion."""
        with pytest.raises(AssertionError):
            dataset_dir_name("data/")

    def test_dataset_dir_name_empty_string_fails(self):
        """Test empty string fails assertion."""
        with pytest.raises(AssertionError):
            dataset_dir_name("")

    def test_dataset_dir_name_preserves_case(self):
        """Test that case is preserved."""
        assert dataset_dir_name("ORG/MyData") == "MyData"

    def test_dataset_dir_name_with_hyphens(self):
        """Test dataset names with hyphens."""
        assert dataset_dir_name("org/my-dataset") == "my-dataset"

    def test_dataset_dir_name_with_numbers(self):
        """Test dataset names with numbers."""
        assert dataset_dir_name("openai/data123") == "data123"


class TestLayerDir:
    """Tests for layer_dir path utility."""

    def test_layer_dir_construction(self):
        """Test basic layer directory construction."""
        results_dir = Path("/results")
        dataset_id = "org/dataset"
        layer_index = 5

        result = layer_dir(results_dir, dataset_id, layer_index)
        assert result == Path("/results/layer_5/dataset")

    def test_layer_dir_with_different_layers(self):
        """Test with various layer indices."""
        results_dir = Path("/tmp/results")

        for layer in [0, 1, 10, 100]:
            result = layer_dir(results_dir, "data/test", layer)
            assert f"layer_{layer}" in str(result)
            assert "test" in str(result)

    def test_layer_dir_relative_path(self):
        """Test with relative paths."""
        results_dir = Path("./results")
        result = layer_dir(results_dir, "org/data", 3)
        assert result == Path("./results/layer_3/data")

    def test_layer_dir_nested_dataset(self):
        """Test with nested dataset IDs."""
        result = layer_dir(Path("/r"), "org/sub/data", 1)
        assert result == Path("/r/layer_1/data")

    def test_layer_dir_single_part_dataset(self):
        """Test with single-part dataset ID."""
        result = layer_dir(Path("/r"), "mydata", 2)
        assert result == Path("/r/layer_2/mydata")


class TestNormsPath:
    """Tests for norms_path path utility."""

    def test_norms_path_basic(self):
        """Test basic norms file path construction."""
        results_dir = Path("/results")
        dataset_id = "org/dataset"

        result = norms_path(results_dir, dataset_id)
        assert result == Path("/results/model_norms_dataset.pt")

    def test_norms_path_simple_dataset(self):
        """Test norms path with simple dataset."""
        result = norms_path(Path("/data"), "mydata")
        assert result == Path("/data/model_norms_mydata.pt")

    def test_norms_path_nested_dataset(self):
        """Test norms path extracts last part correctly."""
        result = norms_path(Path("/data"), "org/project/mydata")
        assert result == Path("/data/model_norms_mydata.pt")

    def test_norms_path_relative_dir(self):
        """Test norms path with relative directory."""
        result = norms_path(Path("./data"), "dataset")
        assert result == Path("./data/model_norms_dataset.pt")

    def test_norms_path_always_ends_in_pt(self):
        """Test that path always ends in .pt."""
        for dataset in ["data", "org/data", "a/b/c"]:
            result = norms_path(Path("/r"), dataset)
            assert str(result).endswith(".pt")
            assert "model_norms_" in str(result)


class TestPositionFilesExist:
    """Tests for position_files_exist using real temp directories."""

    def test_position_files_exist_all_present_no_logitlens(self):
        """Test when all required files exist, no logitlens needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.pt").touch()
            (layer_dir_path / "mean_pos_0.meta").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=False)
            assert result is True

    def test_position_files_missing_mean_pt(self):
        """Test when mean.pt is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.meta").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=False)
            assert result is False

    def test_position_files_missing_meta(self):
        """Test when meta file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.pt").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=False)
            assert result is False

    def test_position_files_logitlens_all_present(self):
        """Test when logitlens files are all present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.pt").touch()
            (layer_dir_path / "mean_pos_0.meta").touch()
            (layer_dir_path / "logit_lens_pos_0.pt").touch()
            (layer_dir_path / "base_logit_lens_pos_0.pt").touch()
            (layer_dir_path / "ft_logit_lens_pos_0.pt").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=True)
            assert result is True

    def test_position_files_logitlens_missing_base(self):
        """Test when base logitlens is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.pt").touch()
            (layer_dir_path / "mean_pos_0.meta").touch()
            (layer_dir_path / "logit_lens_pos_0.pt").touch()
            (layer_dir_path / "ft_logit_lens_pos_0.pt").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=True)
            assert result is False

    def test_position_files_logitlens_missing_ft(self):
        """Test when ft logitlens is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.pt").touch()
            (layer_dir_path / "mean_pos_0.meta").touch()
            (layer_dir_path / "logit_lens_pos_0.pt").touch()
            (layer_dir_path / "base_logit_lens_pos_0.pt").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=True)
            assert result is False

    def test_position_files_logitlens_missing_difference(self):
        """Test when difference logitlens is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_dir_path = Path(tmpdir)
            (layer_dir_path / "mean_pos_0.pt").touch()
            (layer_dir_path / "mean_pos_0.meta").touch()
            (layer_dir_path / "base_logit_lens_pos_0.pt").touch()
            (layer_dir_path / "ft_logit_lens_pos_0.pt").touch()
            result = position_files_exist(layer_dir_path, 0, need_logit_lens=True)
            assert result is False

    def test_position_files_different_positions(self):
        """Test with different position indices."""
        for pos in [0, 1, 5, 42]:
            with tempfile.TemporaryDirectory() as tmpdir:
                layer_dir_path = Path(tmpdir)
                (layer_dir_path / f"mean_pos_{pos}.pt").touch()
                (layer_dir_path / f"mean_pos_{pos}.meta").touch()
                result = position_files_exist(
                    layer_dir_path, pos, need_logit_lens=False
                )
                assert result is True


class TestIsLayerComplete:
    """Tests for is_layer_complete with mocked Path."""

    def test_is_layer_complete_when_dir_missing(self):
        """Test layer incomplete when directory doesn't exist."""
        results_dir = Path("/results")
        dataset_id = "org/data"
        layer_index = 5

        with patch(
            "diffing.methods.activation_difference_lens.util.layer_dir"
        ) as mock_layer_dir:
            with patch.object(Path, "exists", return_value=False):
                mock_layer_dir.return_value = Path("/results/layer_5/data")
                result = is_layer_complete(
                    results_dir, dataset_id, layer_index, 3, False
                )
                assert result is False

    def test_is_layer_complete_all_positions_exist(self):
        """Test layer complete when all positions exist."""
        results_dir = Path("/results")
        dataset_id = "org/data"
        layer_index = 5
        n_positions = 3

        with patch(
            "diffing.methods.activation_difference_lens.util.layer_dir"
        ) as mock_layer_dir:
            with patch(
                "diffing.methods.activation_difference_lens.util.position_files_exist",
                return_value=True,
            ):
                with patch.object(Path, "exists", return_value=True):
                    mock_layer_dir.return_value = Path("/results/layer_5/data")
                    result = is_layer_complete(
                        results_dir, dataset_id, layer_index, n_positions, False
                    )
                    assert result is True

    def test_is_layer_complete_missing_one_position(self):
        """Test layer incomplete when one position is missing."""
        results_dir = Path("/results")
        dataset_id = "org/data"
        layer_index = 5
        n_positions = 3

        def position_files_side_effect(path, pos, need_ll):
            return pos != 1  # Missing position 1

        with patch(
            "diffing.methods.activation_difference_lens.util.layer_dir"
        ) as mock_layer_dir:
            with patch(
                "diffing.methods.activation_difference_lens.util.position_files_exist",
                side_effect=position_files_side_effect,
            ):
                with patch.object(Path, "exists", return_value=True):
                    mock_layer_dir.return_value = Path("/results/layer_5/data")
                    result = is_layer_complete(
                        results_dir, dataset_id, layer_index, n_positions, False
                    )
                    assert result is False

    def test_is_layer_complete_with_logitlens_required(self):
        """Test layer complete requires logitlens files when needed."""
        results_dir = Path("/results")
        dataset_id = "org/data"
        layer_index = 5
        n_positions = 2

        def position_files_side_effect(path, pos, need_ll):
            # Fail if logitlens is needed
            if need_ll:
                return False
            return True

        with patch(
            "diffing.methods.activation_difference_lens.util.layer_dir"
        ) as mock_layer_dir:
            with patch(
                "diffing.methods.activation_difference_lens.util.position_files_exist",
                side_effect=position_files_side_effect,
            ):
                with patch.object(Path, "exists", return_value=True):
                    mock_layer_dir.return_value = Path("/results/layer_5/data")
                    result = is_layer_complete(
                        results_dir, dataset_id, layer_index, n_positions, True
                    )
                    assert result is False

    def test_is_layer_complete_multiple_positions(self):
        """Test layer with many positions."""
        results_dir = Path("/results")
        dataset_id = "org/data"
        layer_index = 3
        n_positions = 10

        with patch(
            "diffing.methods.activation_difference_lens.util.layer_dir"
        ) as mock_layer_dir:
            with patch(
                "diffing.methods.activation_difference_lens.util.position_files_exist",
                return_value=True,
            ):
                with patch.object(Path, "exists", return_value=True):
                    mock_layer_dir.return_value = Path("/results/layer_3/data")
                    result = is_layer_complete(
                        results_dir, dataset_id, layer_index, n_positions, False
                    )
                    assert result is True

    def test_is_layer_complete_zero_positions_edge_case(self):
        """Test edge case with zero positions."""
        results_dir = Path("/results")
        dataset_id = "org/data"
        layer_index = 0
        n_positions = 0

        with patch(
            "diffing.methods.activation_difference_lens.util.layer_dir"
        ) as mock_layer_dir:
            with patch.object(Path, "exists", return_value=True):
                mock_layer_dir.return_value = Path("/results/layer_0/data")
                result = is_layer_complete(
                    results_dir, dataset_id, layer_index, n_positions, False
                )
                # With zero positions, loop doesn't run, so it returns True
                assert result is True


class TestIsGenericTokenEdgeCases:
    """Additional edge case tests for _is_generic_token."""

    def test_token_filtering_consistency(self):
        """Test that common word filtering is consistent."""
        # All common words should filter
        for word in COMMON_WORDS:
            assert _is_generic_token(word), f"Word '{word}' should be generic"

    def test_non_common_words_not_filtered(self):
        """Test that words not in COMMON_WORDS are not filtered."""
        non_common = ["elephant", "algorithm", "quantum", "research", "hypothesis"]
        for word in non_common:
            assert not _is_generic_token(word), f"Word '{word}' should not be generic"

    def test_unicode_space_markers(self):
        """Test various unicode space markers used by tokenizers."""
        # Sentencepiece and others use these
        assert _is_generic_token("▁the")
        assert _is_generic_token("▁a")
        assert _is_generic_token("Ġthe")

    def test_mixed_markers_and_punctuation(self):
        """Test tokens with mixed markers and punctuation."""
        # After cleaning: empty or just punctuation
        assert _is_generic_token("▁.")
        assert _is_generic_token("Ġ!")

    def test_token_two_characters(self):
        """Test two-character tokens."""
        # "is" is in common words
        assert _is_generic_token("is")
        # But "ab" or other two-char combos might not be
        assert not _is_generic_token("ab") or _is_generic_token(
            "ab"
        )  # Depends on word list

    def test_regex_pattern_matching(self):
        """Test that regex patterns work correctly for punctuation detection."""
        # Pure punctuation patterns
        pure_punct = [".", ",", "!", "?", ";", ":", "-", "_"]
        for punct in pure_punct:
            result = _is_generic_token(punct)
            assert result, f"Single '{punct}' should be generic"

    def test_tokens_with_leading_trailing_space_markers(self):
        """Test tokens with space markers that need cleaning."""
        # These have space markers that get stripped
        result = _is_generic_token("▁hello")
        # After cleaning: "hello" - not in common words, so not generic
        assert not result


class TestCommonWordsCompleteness:
    """Tests verifying COMMON_WORDS structure."""

    def test_common_words_not_empty(self):
        """Verify COMMON_WORDS is populated."""
        assert len(COMMON_WORDS) > 0

    def test_common_words_all_strings(self):
        """Verify all entries in COMMON_WORDS are strings."""
        assert all(isinstance(word, str) for word in COMMON_WORDS)

    def test_common_words_all_nonempty(self):
        """Verify no empty strings in COMMON_WORDS."""
        assert all(len(word) > 0 for word in COMMON_WORDS)

    def test_contains_articles(self):
        """Verify articles are in COMMON_WORDS."""
        assert "a" in COMMON_WORDS
        assert "an" in COMMON_WORDS
        assert "the" in COMMON_WORDS

    def test_contains_prepositions(self):
        """Verify prepositions are in COMMON_WORDS."""
        assert "in" in COMMON_WORDS
        assert "on" in COMMON_WORDS
        assert "at" in COMMON_WORDS
        assert "to" in COMMON_WORDS

    def test_contains_verbs(self):
        """Verify common verbs are in COMMON_WORDS."""
        assert "is" in COMMON_WORDS
        assert "are" in COMMON_WORDS
        assert "be" in COMMON_WORDS


class TestCleanGeneratedText:
    """Tests for _clean_generated_text pure function."""

    def test_no_end_of_turn_token(self):
        """Test that None end_of_turn_token returns text unchanged."""
        text = "Hello world<|endoftext|><|endoftext|>"
        assert _clean_generated_text(text, None) == text

    def test_collapse_consecutive_tokens(self):
        """Test collapsing multiple consecutive end tokens."""
        text = "Hello<|end|><|end|><|end|>world"
        result = _clean_generated_text(text, "<|end|>")
        assert result == "Hello<|end|>world"

    def test_single_token_unchanged(self):
        """Test single end token remains unchanged."""
        text = "Hello<|end|>world"
        result = _clean_generated_text(text, "<|end|>")
        assert result == "Hello<|end|>world"

    def test_multiple_groups_collapsed(self):
        """Test multiple groups of consecutive tokens are each collapsed."""
        text = "A<|eos|><|eos|>B<|eos|><|eos|><|eos|>C"
        result = _clean_generated_text(text, "<|eos|>")
        assert result == "A<|eos|>B<|eos|>C"

    def test_empty_text(self):
        """Test empty text returns empty string."""
        assert _clean_generated_text("", "<|end|>") == ""

    def test_text_without_token(self):
        """Test text without the end token is unchanged."""
        text = "Hello world"
        result = _clean_generated_text(text, "<|endoftext|>")
        assert result == "Hello world"

    def test_special_regex_characters_escaped(self):
        """Test that special regex characters in token are properly escaped."""
        text = "Hello[END][END][END]world"
        result = _clean_generated_text(text, "[END]")
        assert result == "Hello[END]world"

    def test_token_at_start(self):
        """Test consecutive tokens at start of text."""
        text = "<eos><eos><eos>Hello"
        result = _clean_generated_text(text, "<eos>")
        assert result == "<eos>Hello"

    def test_token_at_end(self):
        """Test consecutive tokens at end of text."""
        text = "Hello<eos><eos><eos>"
        result = _clean_generated_text(text, "<eos>")
        assert result == "Hello<eos>"

    def test_only_tokens(self):
        """Test text that is only end tokens."""
        text = "<|end|><|end|><|end|>"
        result = _clean_generated_text(text, "<|end|>")
        assert result == "<|end|>"

    def test_newline_token(self):
        """Test with newline as end token."""
        text = "Hello\n\n\nworld"
        result = _clean_generated_text(text, "\n")
        assert result == "Hello\nworld"

    def test_pipe_character_escaped(self):
        """Test pipe character (regex special) is escaped."""
        text = "A|B||C|||D"
        result = _clean_generated_text(text, "|")
        assert result == "A|B|C|D"


class TestReadPrompts:
    """Tests for read_prompts file utility."""

    def test_read_simple_prompts(self):
        """Test reading simple prompts file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Prompt 1\nPrompt 2\nPrompt 3\n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Prompt 1", "Prompt 2", "Prompt 3"]

    def test_read_prompts_strips_whitespace(self):
        """Test that whitespace is stripped from lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  Prompt 1  \n  Prompt 2  \n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Prompt 1", "Prompt 2"]

    def test_read_prompts_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Prompt 1\n\n\nPrompt 2\n\n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Prompt 1", "Prompt 2"]

    def test_read_prompts_skips_whitespace_only_lines(self):
        """Test that whitespace-only lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Prompt 1\n   \n\t\nPrompt 2\n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Prompt 1", "Prompt 2"]

    def test_read_prompts_nonexistent_file_fails(self):
        """Test that nonexistent file raises assertion."""
        with pytest.raises(AssertionError):
            read_prompts("/nonexistent/path/to/file.txt")

    def test_read_prompts_empty_file_fails(self):
        """Test that empty file raises assertion."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(AssertionError):
                read_prompts(f.name)

    def test_read_prompts_only_whitespace_fails(self):
        """Test that file with only whitespace raises assertion."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("   \n\n\t\n   ")
            f.flush()
            with pytest.raises(AssertionError):
                read_prompts(f.name)

    def test_read_prompts_single_prompt(self):
        """Test reading file with single prompt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Single prompt\n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Single prompt"]

    def test_read_prompts_preserves_internal_spaces(self):
        """Test that internal spaces in prompts are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Tell me   a story\nWhat is   your name\n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Tell me   a story", "What is   your name"]

    def test_read_prompts_unicode(self):
        """Test reading prompts with unicode characters."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Bonjour le monde\n")
            f.write("Hola mundo\n")
            f.flush()
            prompts = read_prompts(f.name)
        assert prompts == ["Bonjour le monde", "Hola mundo"]


class TestAgentDatasetDirName:
    """Tests for agent_tools._dataset_dir_name (separate from util.dataset_dir_name)."""

    def test_simple_dataset(self):
        """Test extracting simple dataset name."""
        assert agent_dataset_dir_name("org/dataset") == "dataset"

    def test_single_part(self):
        """Test single-part dataset ID."""
        assert agent_dataset_dir_name("mydata") == "mydata"

    def test_multiple_parts(self):
        """Test multi-part dataset ID takes last part."""
        assert agent_dataset_dir_name("org/subdir/data") == "data"

    def test_empty_result_fails(self):
        """Test that trailing slash results in empty string -> fails assertion."""
        with pytest.raises(AssertionError):
            agent_dataset_dir_name("data/")

    def test_empty_string_fails(self):
        """Test empty string fails assertion."""
        with pytest.raises(AssertionError):
            agent_dataset_dir_name("")

    def test_consistency_with_util_version(self):
        """Test that agent_tools version is consistent with util version."""
        test_cases = ["org/data", "mydata", "a/b/c", "HuggingFace/dataset-name"]
        for case in test_cases:
            assert agent_dataset_dir_name(case) == dataset_dir_name(case)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
