"""Tests for config utilities."""

import dataclasses

from omegaconf import OmegaConf

from diffing.utils.configs import ModelConfig, create_model_config

# Fields NOT read from DictConfig by create_model_config — they are either
# passed as explicit kwargs (device_map) or computed downstream (is_lora).
_NON_DICTCONFIG_FIELDS = {"device_map", "is_lora"}

# Non-default values for every ModelConfig field that create_model_config reads.
_FIELDS_VIA_DICTCONFIG = {
    "name": "test-model",
    "model_id": "org/test-model",
    "tokenizer_id": "org/test-tokenizer",
    "attn_implementation": "flash_attention_2",
    "ignore_first_n_tokens_per_sample_during_collection": 7,
    "ignore_first_n_tokens_per_sample_during_training": 3,
    "token_level_replacement": {"<pad>": "<eos>"},
    "text_column": "content",
    "base_model_id": "org/base-model",
    "subfolder": "checkpoint-100",
    "dtype": "bfloat16",
    "steering_vector": "org/steering-vec",
    "steering_layer": 12,
    "no_auto_device_map": True,
    "trust_remote_code": True,
    "vllm_kwargs": {"gpu_memory_utilization": 0.9},
    "disable_compile": True,
    "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
}

_DEVICE_MAP = {"": "cuda:1"}


class TestCreateModelConfigPropagatesAllFields:
    """Ensure create_model_config extracts every ModelConfig field from the DictConfig."""

    def test_all_dataclass_fields_covered_by_test(self):
        """Meta-test: every ModelConfig field is either in _FIELDS_VIA_DICTCONFIG or _NON_DICTCONFIG_FIELDS."""
        dataclass_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        covered = set(_FIELDS_VIA_DICTCONFIG) | _NON_DICTCONFIG_FIELDS
        missing = dataclass_fields - covered
        assert not missing, (
            f"New ModelConfig fields not covered by test: {missing}. "
            f"Add to _FIELDS_VIA_DICTCONFIG or _NON_DICTCONFIG_FIELDS."
        )

    def test_all_fields_propagated(self):
        """Call create_model_config and verify every DictConfig field is propagated."""
        cfg = OmegaConf.create(_FIELDS_VIA_DICTCONFIG)
        result = create_model_config(cfg, device_map=_DEVICE_MAP)

        for field in dataclasses.fields(ModelConfig):
            if field.name in _NON_DICTCONFIG_FIELDS:
                continue
            expected = _FIELDS_VIA_DICTCONFIG[field.name]
            actual = getattr(result, field.name)
            assert actual == expected, (
                f"Field '{field.name}': expected {expected!r}, got {actual!r}. "
                f"Likely missing from create_model_config()."
            )

    def test_device_map_propagated(self):
        """Verify device_map kwarg is propagated."""
        cfg = OmegaConf.create(_FIELDS_VIA_DICTCONFIG)
        result = create_model_config(cfg, device_map=_DEVICE_MAP)
        assert result.device_map == _DEVICE_MAP
