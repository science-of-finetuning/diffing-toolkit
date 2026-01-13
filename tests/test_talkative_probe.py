"""Tests for Talkative Probe pure utility functions (CPU-only, no models).

Note: The talkative_probe module's source files (.py) are missing - only .pyc files exist.
Based on analysis of the .pyc bytecode, talkative_probe shares identical utility functions
with activation_oracle. These tests verify the shared functionality via activation_oracle
imports until talkative_probe source files are restored.

The following functions were identified in talkative_probe's .pyc files:
- utils/eval.py: parse_answer, proportion_confidence, score_eval_responses, analyze_results
- utils/dataset_utils.py: get_introspection_prefix, find_pattern_in_tokens, SPECIAL_TOKEN
- utils/common.py: set_seed, list_decode, get_bos_eos_pad_mask
- verbalizer.py: sanitize_lora_name, encode_messages, create_verbalizer_inputs
- agent.py: TalkativeProbeAgent, OVERVIEW_DESCRIPTION, ADDITIONAL_CONDUCT
"""

import pytest
import math

from diffing.methods.activation_oracle.utils.eval import (
    parse_answer,
    proportion_confidence,
)
from diffing.methods.activation_oracle.utils.dataset_utils import (
    get_introspection_prefix,
    SPECIAL_TOKEN,
)
from diffing.methods.activation_oracle.verbalizer import sanitize_lora_name


class TestParseAnswer:
    """Tests for parse_answer response normalization."""

    def test_basic_lowercase(self):
        assert parse_answer("Yes") == "yes"
        assert parse_answer("NO") == "no"

    def test_strips_whitespace(self):
        assert parse_answer("  yes  ") == "yes"
        assert parse_answer("\nno\t") == "no"

    def test_strips_trailing_punctuation(self):
        assert parse_answer("yes.") == "yes"
        assert parse_answer("no!") == "no"
        assert parse_answer("maybe?") == "maybe"
        assert parse_answer("sure,") == "sure"
        assert parse_answer("ok;") == "ok"
        assert parse_answer("fine:") == "fine"

    def test_multiple_trailing_punctuation(self):
        assert parse_answer("yes...") == "yes"
        assert parse_answer("no!!!") == "no"

    def test_empty_string(self):
        assert parse_answer("") == ""

    def test_whitespace_only(self):
        assert parse_answer("   ") == ""

    def test_preserves_internal_punctuation(self):
        assert parse_answer("it's fine") == "it's fine"
        assert parse_answer("yes, I think so.") == "yes, i think so"


class TestProportionConfidence:
    """Tests for proportion_confidence statistics computation."""

    def test_all_correct(self):
        p, se, lower, upper = proportion_confidence(10, 10)
        assert p == 1.0
        assert se == 0.0
        assert lower == 1.0
        assert upper == 1.0

    def test_all_incorrect(self):
        p, se, lower, upper = proportion_confidence(0, 10)
        assert p == 0.0
        assert se == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_half_correct(self):
        p, se, lower, upper = proportion_confidence(50, 100)
        assert p == 0.5
        expected_se = math.sqrt(0.5 * 0.5 / 100)
        assert abs(se - expected_se) < 1e-9
        assert lower < 0.5 < upper

    def test_zero_total_returns_zeros(self):
        p, se, lower, upper = proportion_confidence(0, 0)
        assert p == 0.0
        assert se == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_negative_total_returns_zeros(self):
        p, se, lower, upper = proportion_confidence(5, -1)
        assert p == 0.0
        assert se == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_confidence_interval_clamped(self):
        p, se, lower, upper = proportion_confidence(99, 100)
        assert lower >= 0.0
        assert upper <= 1.0

    def test_custom_z_score(self):
        p1, se1, lower1, upper1 = proportion_confidence(50, 100, z=1.96)
        p2, se2, lower2, upper2 = proportion_confidence(50, 100, z=2.58)
        assert p1 == p2
        assert se1 == se2
        assert lower2 < lower1
        assert upper2 > upper1


class TestGetIntrospectionPrefix:
    """Tests for get_introspection_prefix prompt building."""

    def test_basic_format(self):
        prefix = get_introspection_prefix(sae_layer=10, num_positions=5)
        assert "Layer: 10" in prefix
        assert prefix.endswith(" \n")

    def test_contains_special_tokens(self):
        prefix = get_introspection_prefix(sae_layer=5, num_positions=3)
        count = prefix.count(SPECIAL_TOKEN)
        assert count == 3

    def test_num_positions_varies_tokens(self):
        prefix_3 = get_introspection_prefix(sae_layer=1, num_positions=3)
        prefix_10 = get_introspection_prefix(sae_layer=1, num_positions=10)
        assert prefix_3.count(SPECIAL_TOKEN) == 3
        assert prefix_10.count(SPECIAL_TOKEN) == 10

    def test_layer_zero(self):
        prefix = get_introspection_prefix(sae_layer=0, num_positions=1)
        assert "Layer: 0" in prefix
        assert prefix.count(SPECIAL_TOKEN) == 1

    def test_special_token_value(self):
        assert SPECIAL_TOKEN == " ?"


class TestSanitizeLoraName:
    """Tests for sanitize_lora_name path sanitization."""

    def test_replaces_dots(self):
        result = sanitize_lora_name("path.to.model")
        assert result == "path_to_model"

    def test_no_dots_unchanged(self):
        result = sanitize_lora_name("model_name")
        assert result == "model_name"

    def test_multiple_dots(self):
        result = sanitize_lora_name("a.b.c.d")
        assert result == "a_b_c_d"

    def test_empty_string(self):
        result = sanitize_lora_name("")
        assert result == ""

    def test_only_dots(self):
        result = sanitize_lora_name("...")
        assert result == "___"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
