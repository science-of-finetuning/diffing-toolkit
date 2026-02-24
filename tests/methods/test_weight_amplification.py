"""
Tests for weight amplification logic.

Tests cover LoRA adapter configuration, weight patching, and utility functions.
"""

import pytest
import torch

from diffing.methods.amplification.amplification_config import (
    patch_lora_weights,
    path_to_template,
    get_module_regex,
    format_amplified_modules,
)
from diffing.methods.amplification.weight_amplification import get_lora_int_id


class MockServer:
    """Mock vLLM server for testing get_lora_int_id."""

    pass


class MockModule:
    """Mock module for testing get_module_regex."""

    def __init__(self, path: str):
        self._module = type("obj", (object,), {"__path__": path})()


class MockStandardizedTransformer:
    """Mock StandardizedTransformer for testing."""

    def __init__(self, num_layers: int = 4):
        self.num_layers = num_layers
        self.attentions = [MockModule("model.layers.0.self_attn")]
        self.mlps = [MockModule("model.layers.0.mlp")]


class TestGetLoraIntId:
    """Tests for get_lora_int_id function."""

    def test_allocates_new_id_for_new_config(self):
        """First config string should get lora_int_id of 1."""
        server = MockServer()
        config_str = '{"name": "config1"}'

        lora_id = get_lora_int_id(server, config_str)

        assert lora_id == 1
        assert hasattr(server, "_lora_id_counter")
        assert server._lora_id_counter == 2

    def test_reuses_id_for_same_config(self):
        """Same config string should return same lora_int_id."""
        server = MockServer()
        config_str = '{"name": "config1"}'

        id1 = get_lora_int_id(server, config_str)
        id2 = get_lora_int_id(server, config_str)

        assert id1 == id2
        assert server._lora_id_counter == 2

    def test_different_configs_get_different_ids(self):
        """Different config strings should get different lora_int_ids."""
        server = MockServer()
        config1 = '{"name": "config1"}'
        config2 = '{"name": "config2"}'

        id1 = get_lora_int_id(server, config1)
        id2 = get_lora_int_id(server, config2)

        assert id1 != id2
        assert id1 == 1
        assert id2 == 2

    def test_sequential_allocation(self):
        """IDs should be allocated sequentially starting from 1."""
        server = MockServer()

        ids = [get_lora_int_id(server, f'{{"name": "config{i}"}}') for i in range(5)]

        assert ids == [1, 2, 3, 4, 5]

    def test_mixed_new_and_reused(self):
        """Mix of new and reused configs should maintain consistency."""
        server = MockServer()
        config_a = '{"name": "A"}'
        config_b = '{"name": "B"}'

        id_a1 = get_lora_int_id(server, config_a)
        id_b1 = get_lora_int_id(server, config_b)
        id_a2 = get_lora_int_id(server, config_a)
        id_b2 = get_lora_int_id(server, config_b)

        assert id_a1 == id_a2 == 1
        assert id_b1 == id_b2 == 2


class TestPatchLoraWeights:
    """Tests for patch_lora_weights function."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample LoRA weights for testing."""
        return {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                4, 8
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                8, 4
            ),
            "base_model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.ones(4, 8),
            "base_model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.ones(8, 4),
            "base_model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(
                4, 8
            ),
            "base_model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.ones(
                8, 4
            ),
        }

    @pytest.fixture
    def module_paths(self):
        """Module paths matching the sample weights structure."""
        return {
            "attention": r"base_model\.model\.layers\.[layer_idx]\.self_attn.*\.lora_B.weight",
            "mlp": r"base_model\.model\.layers\.[layer_idx]\.mlp.*\.lora_B.weight",
        }

    def test_empty_amplifications_returns_cloned_weights(self, sample_weights):
        """Empty amplifications should return cloned weights unchanged."""
        patched, amplified, unamplified = patch_lora_weights(sample_weights, {}, {})

        assert len(patched) == len(sample_weights)
        assert len(amplified) == 0
        assert len(unamplified) == len(sample_weights)
        for key in sample_weights:
            assert patched[key] is not sample_weights[key]
            assert torch.equal(patched[key], sample_weights[key])

    def test_amplifies_attention_layer_0(self, sample_weights, module_paths):
        """Amplify attention in layer 0 with weight 2.0."""
        compiled = {"adapter1": [{}, {}]}
        compiled["adapter1"][0] = {"attention": 2.0}

        patched, amplified, unamplified = patch_lora_weights(
            sample_weights, compiled, module_paths
        )

        attn_key = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
        assert attn_key in amplified
        assert amplified[attn_key] == 2.0
        assert torch.allclose(patched[attn_key], torch.ones(8, 4) * 2.0)
        assert "base_model.model.layers.1.self_attn.q_proj.lora_B.weight" in unamplified

    def test_amplifies_mlp_layer_0(self, sample_weights, module_paths):
        """Amplify MLP in layer 0 with weight 0.5."""
        compiled = {"adapter1": [{"mlp": 0.5}, {}]}

        patched, amplified, unamplified = patch_lora_weights(
            sample_weights, compiled, module_paths
        )

        mlp_key = "base_model.model.layers.0.mlp.gate_proj.lora_B.weight"
        assert mlp_key in amplified
        assert amplified[mlp_key] == 0.5
        assert torch.allclose(patched[mlp_key], torch.ones(8, 4) * 0.5)

    def test_zero_weight_zeroes_tensor(self, sample_weights, module_paths):
        """Weight of 0.0 should zero out the tensor."""
        compiled = {"adapter1": [{"attention": 0.0}, {}]}

        patched, amplified, _ = patch_lora_weights(
            sample_weights, compiled, module_paths
        )

        attn_key = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
        assert amplified[attn_key] == 0.0
        assert torch.allclose(patched[attn_key], torch.zeros(8, 4))

    def test_multiple_layers_amplified(self, sample_weights, module_paths):
        """Amplify different layers with different weights."""
        compiled = {"adapter1": [{"attention": 2.0}, {"attention": 3.0}]}

        patched, amplified, _ = patch_lora_weights(
            sample_weights, compiled, module_paths
        )

        assert (
            amplified["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"] == 2.0
        )
        assert (
            amplified["base_model.model.layers.1.self_attn.q_proj.lora_B.weight"] == 3.0
        )

    def test_original_weights_not_modified(self, sample_weights, module_paths):
        """Original weights dict should not be modified."""
        original_sum = sample_weights[
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
        ].sum()
        compiled = {"adapter1": [{"attention": 0.0}, {}]}

        patch_lora_weights(sample_weights, compiled, module_paths)

        new_sum = sample_weights[
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
        ].sum()
        assert new_sum == original_sum

    def test_preserves_dtype(self, module_paths):
        """Patching should preserve tensor dtype."""
        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                4, 8, dtype=torch.float16
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                8, 4, dtype=torch.float16
            ),
        }
        compiled = {"adapter1": [{"attention": 2.0}]}

        patched, _, _ = patch_lora_weights(weights, compiled, module_paths)

        assert (
            patched["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"].dtype
            == torch.float16
        )

    def test_raises_on_multiple_adapters(self, sample_weights, module_paths):
        """Multiple adapters should raise NotImplementedError."""
        compiled = {"adapter1": [{}], "adapter2": [{}]}

        with pytest.raises(
            NotImplementedError, match="Multiple compiled amplifications"
        ):
            patch_lora_weights(sample_weights, compiled, module_paths)

    def test_raises_on_non_lora_weight(self, module_paths):
        """Non-LoRA weight keys should raise ValueError."""
        weights = {
            "base_model.model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8)
        }
        compiled = {"adapter1": [{}]}

        with pytest.raises(ValueError, match="is not a LoRA weight"):
            patch_lora_weights(weights, compiled, module_paths)

    def test_raises_on_no_regex_match(self, sample_weights, module_paths):
        """Missing regex match should raise ValueError."""
        compiled = {"adapter1": [{}, {}, {"attention": 1.0}]}

        with pytest.raises(ValueError, match="No matches found"):
            patch_lora_weights(sample_weights, compiled, module_paths)

    def test_raises_on_duplicate_amplification(self, sample_weights, module_paths):
        """Amplifying same module twice should raise ValueError."""
        compiled = {"adapter1": [{"attention": 1.0, "mlp": 1.0}, {}]}
        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                4, 8
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                8, 4
            ),
        }
        bad_module_paths = {
            "attention": r"base_model\.model\.layers\.[layer_idx]\.self_attn.*\.lora_B.weight",
            "mlp": r"base_model\.model\.layers\.[layer_idx]\.self_attn.*\.lora_B.weight",
        }

        with pytest.raises(ValueError, match="already amplified"):
            patch_lora_weights(weights, compiled, bad_module_paths)


class TestPathToTemplate:
    """Tests for path_to_template function."""

    def test_attention_path(self):
        """Convert attention module path to regex template."""
        path = "model.layers.0.self_attn"
        result = path_to_template(path)
        assert (
            result
            == r"base_model\.model\.layers\.[layer_idx]\.self_attn.*\.lora_B.weight"
        )

    def test_mlp_path(self):
        """Convert MLP module path to regex template."""
        path = "model.layers.0.mlp"
        result = path_to_template(path)
        assert result == r"base_model\.model\.layers\.[layer_idx]\.mlp.*\.lora_B.weight"

    def test_raises_on_no_zero(self):
        """Path without 0 should raise ValueError."""
        with pytest.raises(ValueError, match="does not contain a 0"):
            path_to_template("model.layers.self_attn")

    def test_raises_on_multiple_zeros(self):
        """Path with multiple 0s should raise ValueError."""
        with pytest.raises(ValueError, match="contains multiple 0s"):
            path_to_template("model.layers.0.block.0")


class TestGetModuleRegex:
    """Tests for get_module_regex function."""

    def test_returns_attention_and_mlp_paths(self):
        """Should return regex templates for attention and mlp modules."""
        model = MockStandardizedTransformer()
        result = get_module_regex(model)

        assert "attention" in result
        assert "mlp" in result
        assert "[layer_idx]" in result["attention"]
        assert "[layer_idx]" in result["mlp"]

    def test_attention_regex_format(self):
        """Attention regex should match expected format."""
        model = MockStandardizedTransformer()
        result = get_module_regex(model)

        assert result["attention"].endswith(r"\.lora_B.weight")
        assert "self_attn" in result["attention"]

    def test_mlp_regex_format(self):
        """MLP regex should match expected format."""
        model = MockStandardizedTransformer()
        result = get_module_regex(model)

        assert result["mlp"].endswith(r"\.lora_B.weight")
        assert "mlp" in result["mlp"]


class TestFormatAmplifiedModules:
    """Tests for format_amplified_modules function."""

    def test_empty_dict(self):
        """Empty dict should return 'No modules amplified'."""
        result = format_amplified_modules({})
        assert result == "No modules amplified"

    def test_single_module(self):
        """Single module should be formatted correctly."""
        modules = {
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": 2.0,
        }
        result = format_amplified_modules(modules)

        assert "Amplified 1 modules" in result
        assert "self_attn" in result
        assert "2.0" in result

    def test_consecutive_layers_grouped(self):
        """Consecutive layers should be grouped into ranges."""
        modules = {
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": 2.0,
            "base_model.model.layers.1.self_attn.q_proj.lora_B.weight": 2.0,
            "base_model.model.layers.2.self_attn.q_proj.lora_B.weight": 2.0,
        }
        result = format_amplified_modules(modules)

        assert "0-2" in result
        assert "Amplified 3 modules" in result

    def test_non_consecutive_layers(self):
        """Non-consecutive layers should be listed separately."""
        modules = {
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": 2.0,
            "base_model.model.layers.5.self_attn.q_proj.lora_B.weight": 2.0,
        }
        result = format_amplified_modules(modules)

        assert "0" in result
        assert "5" in result

    def test_different_weights_same_layer(self):
        """Different modules in same layer with different weights."""
        modules = {
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": 2.0,
            "base_model.model.layers.0.mlp.gate_proj.lora_B.weight": 0.5,
        }
        result = format_amplified_modules(modules)

        assert "self_attn" in result
        assert "mlp" in result
        assert "2.0" in result
        assert "0.5" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
