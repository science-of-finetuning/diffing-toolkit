"""
Integration tests for DiffingMethod initialization and core computation.

Tests that each diffing method can be instantiated with config and perform
core computation with real GPT-2 models. Uses minimal samples and stores
results in TMPDIR.

Methods tested:
- KLDivergenceDiffingMethod
- ActDiffLens (activation_difference_lens)
- ActivationOracleMethod
- WeightDifferenceAmplification
- SAEDifferenceMethod
- ActivationAnalysisDiffingMethod

Methods NOT tested:
- CrosscoderDiffingMethod
- PCAMethod
"""

import os
import json
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"

# Path to configs
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Simple chat template for GPT-2 testing (GPT-2 doesn't have one by default)
SIMPLE_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}system: {{ message['content'] }}
{% elif message['role'] == 'user' %}human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}model: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}model: {% endif %}"""

# GPT-2 hidden dimension
HIDDEN_DIM = 768
# Small number of tokens for testing
NUM_TOKENS = 100
# Sequence length
SEQ_LEN = 16


def create_mock_activation_cache(
    cache_dir: Path,
    hidden_dim: int = HIDDEN_DIM,
    num_tokens: int = NUM_TOKENS,
    seq_len: int = SEQ_LEN,
    dtype: str = "float32",
) -> Path:
    """
    Create a mock activation cache directory with synthetic data.

    Creates the memmap format expected by dictionary_learning.cache.ActivationCache:
    - config.json with shard_count, store_tokens, total_size, d_model
    - shard_0.memmap with activation data
    - shard_0.meta with shape and dtype

    Args:
        cache_dir: Directory to create cache in
        hidden_dim: Hidden dimension of activations
        num_tokens: Total number of tokens
        seq_len: Sequence length for grouping tokens
        dtype: Data type for activations

    Returns:
        Path to the cache directory
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create config.json (must include all keys expected by SampleCache)
    config = {
        "shard_count": 1,
        "store_tokens": True,
        "store_sequence_ranges": True,
        "dtype": dtype,
        "d_model": hidden_dim,
        "total_size": num_tokens,
        "shuffle_shards": False,
    }
    with open(cache_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create activation shard as memmap (dictionary_learning format)
    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    activations = torch.randn(num_tokens, hidden_dim, dtype=torch_dtype)

    memmap_file = cache_dir / "shard_0.memmap"
    meta_file = cache_dir / "shard_0.meta"

    # Write memmap (bfloat16 stored as int16, float32 as-is)
    np_dtype = np.float32 if dtype == "float32" else np.int16
    if dtype == "float32":
        np_data = activations.numpy()
    else:
        np_data = activations.view(torch.int16).numpy()

    memmap = np.memmap(memmap_file, dtype=np_dtype, mode="w+", shape=activations.shape)
    memmap[:] = np_data
    memmap.flush()

    # Write metadata (must use torch dtype string format like "torch.float32")
    torch_dtype_str = "torch.float32" if dtype == "float32" else "torch.bfloat16"
    with open(meta_file, "w") as f:
        json.dump({"shape": list(activations.shape), "dtype": torch_dtype_str}, f)

    return cache_dir


def create_paired_activation_cache(
    base_dir: Path,
    dataset_name: str,
    layer: int,
    base_model: str = "gpt2",
    finetuned_model: str = "gpt2_alpaca-lora",  # Must match get_safe_model_id output
    split: str = "train",
) -> Path:
    """
    Create paired activation caches for base and finetuned models.

    Creates the directory structure expected by load_activation_dataset:
    activation_store_dir/model_name/dataset_name/split/layer_{n}_out/

    Returns:
        Path to the activation store directory
    """
    submodule_name = f"layer_{layer}_out"

    # Create shared tokens and sequence_ranges (same for both models)
    tokens = torch.randint(0, 50257, (NUM_TOKENS,))
    # Sequence ranges: 1D tensor of start indices + end marker
    # For 100 tokens with SEQ_LEN=16: [0, 16, 32, 48, 64, 80, 96, 100]
    num_sequences = NUM_TOKENS // SEQ_LEN
    sequence_starts = list(range(0, num_sequences * SEQ_LEN, SEQ_LEN))
    sequence_starts.append(NUM_TOKENS)  # End marker
    sequence_ranges = torch.tensor(sequence_starts, dtype=torch.long)

    # Create base model cache
    base_split_dir = base_dir / base_model / dataset_name / split
    base_cache_dir = base_split_dir / submodule_name
    create_mock_activation_cache(base_cache_dir)
    torch.save(tokens, base_split_dir / "tokens.pt")
    torch.save(sequence_ranges, base_split_dir / "sequence_ranges.pt")

    # Create finetuned model cache with same tokens/sequence_ranges
    ft_split_dir = base_dir / finetuned_model / dataset_name / split
    ft_cache_dir = ft_split_dir / submodule_name
    create_mock_activation_cache(ft_cache_dir)
    torch.save(tokens, ft_split_dir / "tokens.pt")
    torch.save(sequence_ranges, ft_split_dir / "sequence_ranges.pt")

    return base_dir


def make_minimal_config(
    method_name: str,
    results_dir: Path,
    max_samples: int = 5,
) -> DictConfig:
    """
    Create a minimal config for testing a diffing method.
    Loads organism config from YAML and merges with test settings.

    Args:
        method_name: Name of the diffing method
        results_dir: Directory to store results
        max_samples: Maximum samples to process

    Returns:
        DictConfig with all required fields
    """
    # Load organism config from YAML
    organism_yaml = CONFIGS_DIR / "organism" / "test_alpaca_lora.yaml"
    organism_cfg = OmegaConf.load(organism_yaml)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "gpt2",
                "model_id": "gpt2",
                "end_of_turn_token": None,
                "attn_implementation": None,
                "token_level_replacement": None,
                "dtype": "float32",
                "ignore_first_n_tokens_per_sample_during_collection": 0,
                "ignore_first_n_tokens_per_sample_during_training": 0,
                "has_enable_thinking": False,
                "disable_compile": False,
            },
            "organism": organism_cfg,
            "organism_variant": "default",
            "chat_dataset": {
                "id": "science-of-finetuning/tulu-3-sft-olmo-2-mixture",
                "splits": ["train"],
                "is_chat": True,
                "text_column": None,
            },
            "pretraining_dataset": {
                "id": "science-of-finetuning/fineweb-1m-sample",
                "splits": ["train"],
                "is_chat": False,
                "text_column": "text",
            },
            "preprocessing": {
                "activation_store_dir": str(results_dir / "activations"),
                "layers": [0.5],
                "max_samples_per_dataset": max_samples,
                "max_tokens_per_dataset_train": 1000,
                "max_tokens_per_dataset_validation": 100,
                "batch_size": 2,
                "context_len": 64,
                "dtype": "float32",
                "store_tokens": True,
                "overwrite": True,
                "disable_multiprocessing": True,
                "chat_only": False,
                "pretraining_only": False,
                "training_only": False,
            },
            "diffing": {
                "results_base_dir": str(results_dir),
                "results_dir": str(results_dir / "gpt2" / "test_alpaca_lora"),
                "method": {
                    "name": method_name,
                    "requires_preprocessing": False,
                    "overwrite": True,
                    "method_params": {
                        "batch_size": 2,
                        "max_samples": max_samples,
                    },
                    "datasets": {
                        "use_chat_dataset": False,
                        "use_pretraining_dataset": True,
                        "use_training_dataset": False,
                    },
                    "analysis": {
                        "max_activating_examples": {
                            "num_examples": 3,
                        },
                    },
                },
            },
            "infrastructure": {
                "storage": {
                    "base_dir": str(results_dir),
                    "checkpoint_dir": str(results_dir / "checkpoints"),
                    "logs_dir": str(results_dir / "logs"),
                },
                "device_map": {
                    "base": "auto",
                    "finetuned": "auto",
                },
            },
            "seed": 42,
            "debug": True,
            "verbose": False,
        }
    )
    return cfg


@pytest.fixture(scope="module")
def tmp_results_dir():
    """Create a temporary directory for test results."""
    tmpdir = tempfile.gettempdir()
    results_dir = Path(tmpdir) / "diffing_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    yield results_dir
    # Cleanup is optional - tests may want to inspect results


@pytest.fixture(scope="module")
def mock_activation_cache(tmp_results_dir):
    """Create mock activation caches for testing methods that require preprocessing."""
    activation_store_dir = tmp_results_dir / "activations"
    dataset_name = "fineweb-1m-sample"
    # Layer index: 0.5 * (12 - 1) = 5 for GPT-2 (see get_layer_indices formula)
    layer = 5

    create_paired_activation_cache(
        base_dir=activation_store_dir,
        dataset_name=dataset_name,
        layer=layer,
        base_model="gpt2",
        finetuned_model="gpt2_alpaca-lora",
        split="train",
    )

    return activation_store_dir


class TestKLDivergenceMethodRun:
    """Tests for KLDivergenceDiffingMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_method_initializes(self, tmp_results_dir):
        """Test that KL method can be instantiated with config."""
        from diffing.methods.kl.method import KLDivergenceDiffingMethod

        cfg = make_minimal_config("kl", tmp_results_dir)
        cfg.diffing.method.method_params.temperature = 1.0
        cfg.diffing.method.method_params.max_tokens_per_sample = 64
        cfg.diffing.method.method_params.ignore_padding = True

        method = KLDivergenceDiffingMethod(cfg)

        assert method is not None
        assert method.results_dir.exists()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_compute_divergence_direct(self, tmp_results_dir):
        """Test KL divergence computation directly."""
        from diffing.methods.kl.method import KLDivergenceDiffingMethod

        cfg = make_minimal_config("kl", tmp_results_dir)
        cfg.diffing.method.method_params.temperature = 1.0
        cfg.diffing.method.method_params.max_tokens_per_sample = 64
        cfg.diffing.method.method_params.ignore_padding = True

        method = KLDivergenceDiffingMethod(cfg)

        # Create test inputs
        tokenizer = method.tokenizer
        text = "Hello, how are you today?"
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        # Compute KL divergence
        per_token_kl, mean_kl = method.compute_kl_divergence(input_ids, attention_mask)

        assert per_token_kl.shape == input_ids.shape
        assert mean_kl.shape == (1,)
        assert torch.all(per_token_kl >= -1e-5)
        assert torch.isfinite(per_token_kl).all()


class TestActivationDifferenceLensMethodRun:
    """Tests for ActDiffLens.run()."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_adl_method_initializes(self, tmp_results_dir):
        """Test that ActDiffLens method can be instantiated."""
        from diffing.methods.activation_difference_lens.act_diff_lens import ActDiffLens

        cfg = make_minimal_config("activation_difference_lens", tmp_results_dir)
        # Add ADL-specific config
        cfg.diffing.method.layers = [0.5]
        cfg.diffing.method.method_params.layer = 0.5
        cfg.diffing.method.method_params.n_tokens = 10
        cfg.diffing.method.method_params.max_samples = 5
        cfg.diffing.method.steps = {
            "norms": True,
            "auto_patch_scope": False,
            "steering": False,
            "token_relevance": False,
            "causal_effect": False,
        }

        method = ActDiffLens(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_adl_run(self, tmp_results_dir):
        """Test that ActDiffLens.run() completes with causal_effect enabled."""
        from diffing.methods.activation_difference_lens.act_diff_lens import ActDiffLens

        cfg = make_minimal_config("activation_difference_lens", tmp_results_dir)
        cfg.diffing.method.layers = [0.5]
        cfg.diffing.method.method_params.layer = 0.5
        cfg.diffing.method.method_params.max_samples = 2
        cfg.diffing.method.max_samples = 2
        cfg.diffing.method.batch_size = 2
        cfg.diffing.method.overwrite = True
        cfg.diffing.method.n = 10  # Must be > skip_tokens (5)
        cfg.diffing.method.pre_assistant_k = 0
        cfg.diffing.method.logit_lens = {"cache": False}
        cfg.diffing.method.token_relevance = {"enabled": False}
        cfg.diffing.method.auto_patch_scope = {"enabled": False}
        cfg.diffing.method.steering = {"enabled": False}
        cfg.diffing.method.split = "train"
        cfg.diffing.method.datasets = [
            {
                "id": "science-of-finetuning/fineweb-1m-sample",
                "is_chat": False,
                "text_column": "text",
            }
        ]
        # Enable causal_effect with minimal config
        # num_random_diff_vectors=0 skips chat dataset requirement (GPT-2 has no chat template)
        cfg.diffing.method.causal_effect = {
            "enabled": True,
            "overwrite": True,
            "split": "train",
            "batch_size": 2,
            "max_samples": 2,
            "max_total_tokens": 128,
            "after_k": 2,
            "num_random_vectors": 2,
            "num_random_diff_vectors": 0,
            "zero_ablate": False,
            "tasks": [
                {
                    "diff_source_dataset": "science-of-finetuning/fineweb-1m-sample",
                    "eval_dataset": "training",
                    "layer": 0.5,
                    "positions": [0, 1],
                }
            ],
        }

        method = ActDiffLens(cfg)
        method.base_model.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        method.run()

        assert method.results_dir.exists()


class TestActivationOracleMethodRun:
    """Tests for ActivationOracleMethod.run()."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_oracle_method_initializes(self, tmp_results_dir):
        """Test that ActivationOracleMethod can be instantiated."""
        from diffing.methods.activation_oracle.activation_oracle import (
            ActivationOracleMethod,
        )

        cfg = make_minimal_config("activation_oracle", tmp_results_dir)
        # Add oracle-specific config
        cfg.diffing.method.method_params.layer = 0.5
        cfg.diffing.method.method_params.num_positions = 3
        cfg.diffing.method.method_params.total_evaluations = 5
        cfg.diffing.method.method_params.evaluations_per_batch = 2
        cfg.diffing.method.verbalizer_models = {"gpt2": "gpt2"}

        method = ActivationOracleMethod(cfg)

        assert method is not None

    def test_oracle_run(self, tmp_results_dir):
        """Test that ActivationOracleMethod.run() completes."""
        from diffing.methods.activation_oracle.activation_oracle import (
            ActivationOracleMethod,
        )

        cfg = make_minimal_config("activation_oracle", tmp_results_dir)
        cfg.diffing.method.method_params.layer = 0.5
        cfg.diffing.method.method_params.num_positions = 2
        cfg.diffing.method.method_params.total_evaluations = 2
        cfg.diffing.method.method_params.evaluations_per_batch = 1
        cfg.diffing.method.overwrite = True
        cfg.diffing.method.context_prompts = ["Hello, how are you?", "What is 2+2?"]
        cfg.diffing.method.verbalizer_prompts = ["What is the model thinking?"]
        cfg.diffing.method.verbalizer_eval = {}  # Empty dict for defaults
        cfg.diffing.method.prefix = ""  # Empty prefix for testing
        # Mock verbalizer model - use the same model as verbalizer for testing
        cfg.diffing.method.verbalizer_models = {"gpt2": "monsterapi/gpt2_alpaca-lora"}

        method = ActivationOracleMethod(cfg)
        method.base_model.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        method.run()

        assert method._results_file().exists()


class TestWeightAmplificationMethodRun:
    """Tests for WeightDifferenceAmplification.run()."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_weight_amp_method_initializes(self, tmp_results_dir):
        """Test that WeightDifferenceAmplification can be instantiated."""
        from diffing.methods.amplification.weight_amplification import (
            WeightDifferenceAmplification,
        )

        cfg = make_minimal_config("weight_amplification", tmp_results_dir)
        # Add weight amplification specific config
        cfg.diffing.method.method_params.scales = [0.0, 1.0, 2.0]
        cfg.diffing.method.method_params.module_types = ["attention", "mlp"]
        cfg.diffing.method.managed_configs_dir = str(
            tmp_results_dir / "managed_configs"
        )

        method = WeightDifferenceAmplification(cfg)

        assert method is not None


class TestSAEDifferenceMethodRun:
    """Tests for SAEDifferenceMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_sae_difference_method_initializes(self, tmp_results_dir):
        """Test that SAEDifferenceMethod can be instantiated."""
        from diffing.methods.sae_difference.method import SAEDifferenceMethod

        cfg = make_minimal_config("sae_difference", tmp_results_dir)
        cfg.diffing.method.requires_preprocessing = True
        cfg.diffing.method.layers = [0.5]
        cfg.diffing.method.method_params.dict_size = 256
        cfg.diffing.method.method_params.k = 32

        method = SAEDifferenceMethod(cfg)

        assert method is not None


class TestActivationAnalysisMethodRun:
    """Tests for ActivationAnalysisDiffingMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_activation_analysis_method_initializes(self, tmp_results_dir):
        """Test that ActivationAnalysisDiffingMethod can be instantiated."""
        from diffing.methods.activation_analysis.diffing_method import (
            ActivationAnalysisDiffingMethod,
        )

        cfg = make_minimal_config("activation_analysis", tmp_results_dir)
        cfg.diffing.method.requires_preprocessing = True
        cfg.diffing.method.method_params.num_workers = 0
        cfg.diffing.method.method_params.skip_first_n_tokens = False
        cfg.diffing.method.analysis = {
            "statistics": ["mean", "std"],
            "max_activating_examples": {
                "num_examples": 3,
                "include_full_messages": False,
                "include_all_token_norms": False,
            },
        }

        method = ActivationAnalysisDiffingMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_activation_analysis_run(
        self, tmp_results_dir, mock_activation_cache, batch_size
    ):
        """Test that ActivationAnalysisDiffingMethod.run() completes with various batch sizes."""
        from diffing.methods.activation_analysis.diffing_method import (
            ActivationAnalysisDiffingMethod,
        )

        cfg = make_minimal_config("activation_analysis", tmp_results_dir)
        cfg.diffing.method.requires_preprocessing = True
        cfg.diffing.method.method_params.num_workers = 0
        cfg.diffing.method.method_params.skip_first_n_tokens = False
        cfg.diffing.method.method_params.max_samples = 5
        cfg.diffing.method.method_params.batch_size = batch_size
        cfg.diffing.method.overwrite = True
        cfg.diffing.method.analysis = {
            "statistics": ["mean", "std"],
            "max_activating_examples": {
                "num_examples": 2,
                "include_full_messages": False,
                "include_all_token_norms": False,
            },
        }

        method = ActivationAnalysisDiffingMethod(cfg)

        # GPT-2 doesn't have a chat template - set a simple one for testing
        # Access tokenizer property to trigger lazy loading, then set template
        method.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        method.run()

        # Verify results were created
        results_dir = method.results_dir
        assert results_dir.exists()
        layer_dirs = list(results_dir.glob("layer_*"))
        assert len(layer_dirs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
