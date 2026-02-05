"""
Integration tests for DiffingMethod initialization and core computation.

Tests that each diffing method can be instantiated with config and perform
core computation with real SmolLM2 models. Uses actual YAML configs from
configs/ directory and runs real preprocessing for methods that require it.

Tests run for both LoRA adapter (swedish_fineweb) and full finetune
(smollm_reasoning) organisms to ensure compatibility with both approaches.
Dataset is overridden to femto-ultrachat for fast testing.

Methods tested:
- KLDivergenceDiffingMethod
- ActDiffLens (activation_difference_lens)
- ActivationOracleMethod
- WeightDifferenceAmplification
- SAEDifferenceMethod
- ActivationAnalysisDiffingMethod
- CrosscoderDiffingMethod
- PCAMethod
"""

import os
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

# Import configs module to register custom resolvers (project_root, get_all_models)
import diffing.utils.configs  # noqa: F401

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"

# Path to configs
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Organism configurations for testing (LoRA adapter + full finetune)
ORGANISM_NAMES = ["swedish_fineweb", "smollm_reasoning"]

# Small dataset used for all tests (overrides organism's actual dataset)
TEST_DATASET_ID = "Butanium/femto-ultrachat"

# Verbalizer model used for oracle tests (same for all organisms)
VERBALIZER_MODEL = "jekunz/smollm-135m-lora-fineweb-swedish"


def load_test_config(
    method_name: str, results_dir: Path, organism_name: str
) -> DictConfig:
    """
    Load test config scaffolding and merge with real sub-configs.

    Args:
        method_name: Name of the diffing method (e.g., "kl", "sae_difference")
        results_dir: Directory to store test results
        organism_name: Name of the organism config to use

    Returns:
        DictConfig with all required fields for running the method
    """
    cfg = OmegaConf.load(CONFIGS_DIR / "test_config.yaml")
    cfg.model = OmegaConf.load(CONFIGS_DIR / "model" / "SmolLM2-135M.yaml")
    cfg.organism = OmegaConf.load(CONFIGS_DIR / "organism" / f"{organism_name}.yaml")
    cfg.infrastructure = OmegaConf.load(CONFIGS_DIR / "infrastructure" / "test.yaml")
    cfg.diffing.method = OmegaConf.load(
        CONFIGS_DIR / "diffing" / "method" / f"{method_name}.yaml"
    )

    # Override dataset to small test dataset for faster testing
    cfg.organism.dataset.id = TEST_DATASET_ID
    cfg.organism.dataset.is_chat = True
    cfg.organism.dataset.text_column = None
    cfg.organism.dataset.subset = None

    cfg.diffing.method.overwrite = True
    cfg.diffing.results_base_dir = str(results_dir)
    cfg.diffing.results_dir = str(results_dir / "SmolLM2-135M-Instruct" / organism_name)
    cfg.preprocessing.activation_store_dir = str(
        results_dir / "activations" / organism_name
    )
    cfg.pipeline.output_dir = str(results_dir / "pipeline_output")

    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(scope="module")
def tmp_results_dir():
    """Create a temporary directory for test results."""
    import tempfile

    return Path(tempfile.mkdtemp(prefix="diffing_test_results_"))


def _make_directory_readonly(path: Path) -> None:
    """Recursively make all files in a directory read-only."""
    import stat

    for root, dirs, files in os.walk(path):
        for f in files:
            file_path = Path(root) / f
            file_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Augment PermissionError with helpful message for read-only cache violations."""
    outcome = yield
    exc_info = outcome.excinfo
    if exc_info is not None and exc_info[0] is PermissionError:
        error = exc_info[1]
        filename = getattr(error, "filename", "") or ""
        if "activations" in str(filename):
            raise PermissionError(
                f"{error}\n\n"
                "TEST ISOLATION VIOLATION: Activation cache files are intentionally read-only.\n"
                "The preprocessed_activations fixture makes caches read-only to ensure tests\n"
                "don't mutate shared state. If your test needs to modify activations,\n"
                "it should create its own copy first."
            ) from error


@pytest.fixture(scope="module")
def preprocessed_activations(tmp_results_dir):
    """
    Run actual preprocessing to create activation caches for all organisms.

    This fixture runs once per test module and creates real activation caches
    using the toy dataset (Butanium/femto-ultrachat) for methods that require
    preprocessing. Returns a dict mapping organism_name -> activation_store_dir.

    After creation, all cache files are made read-only to ensure test isolation.
    If any test attempts to modify the cache, it will fail immediately.
    """
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available for preprocessing")

    from diffing.pipeline.preprocessing import PreprocessingPipeline
    from diffing.utils.model import clear_cache

    activation_dirs = {}
    for organism_name in ORGANISM_NAMES:
        cfg = load_test_config("activation_analysis", tmp_results_dir, organism_name)
        pipeline = PreprocessingPipeline(cfg)
        pipeline.run()
        activation_dirs[organism_name] = cfg.preprocessing.activation_store_dir

    # Free GPU memory: models are no longer needed after preprocessing.
    # Only the saved activation files on disk matter from here on.
    clear_cache()

    # Make all cache files read-only to ensure tests don't mutate shared state
    for activation_dir in activation_dirs.values():
        _make_directory_readonly(Path(activation_dir))

    return activation_dirs


@pytest.fixture(params=ORGANISM_NAMES)
def organism_name(request):
    """Parameterized fixture that yields each organism name."""
    return request.param


class TestKLDivergenceMethodRun:
    """Tests for KLDivergenceDiffingMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_method_initializes(self, tmp_results_dir, organism_name):
        """Test that KL method can be instantiated with real config."""
        from diffing.methods.kl.method import KLDivergenceDiffingMethod

        cfg = load_test_config("kl", tmp_results_dir, organism_name)
        method = KLDivergenceDiffingMethod(cfg)

        assert method is not None
        assert method.results_dir.exists()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_kl_compute_divergence(self, tmp_results_dir, organism_name):
        """Test KL divergence computation directly."""
        from diffing.methods.kl.method import KLDivergenceDiffingMethod

        cfg = load_test_config("kl", tmp_results_dir, organism_name)
        method = KLDivergenceDiffingMethod(cfg)

        # Create test inputs
        tokenizer = method.tokenizer
        text = "Hello, how are you today?"
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        per_token_kl, mean_kl = method.compute_kl_divergence(input_ids, attention_mask)

        assert per_token_kl.shape == input_ids.shape
        assert mean_kl.shape == (1,)
        assert torch.all(per_token_kl >= -1e-5)
        assert torch.isfinite(per_token_kl).all()


class TestActivationDifferenceLensMethodRun:
    """Tests for ActDiffLens."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_adl_method_initializes(self, tmp_results_dir, organism_name):
        """Test that ActDiffLens method can be instantiated with real config."""
        from diffing.methods.activation_difference_lens.method import ActDiffLens

        cfg = load_test_config(
            "activation_difference_lens", tmp_results_dir, organism_name
        )
        method = ActDiffLens(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_adl_run(self, tmp_results_dir, organism_name):
        """Test that ActDiffLens.run() completes with causal_effect enabled."""
        from diffing.methods.activation_difference_lens.method import ActDiffLens

        cfg = load_test_config(
            "activation_difference_lens", tmp_results_dir, organism_name
        )

        # Override for minimal test run
        cfg.diffing.method.max_samples = 2
        cfg.diffing.method.batch_size = 2
        cfg.diffing.method.n = 10
        cfg.diffing.method.pre_assistant_k = 0

        # Disable expensive analysis steps for faster testing
        cfg.diffing.method.logit_lens.cache = False
        cfg.diffing.method.auto_patch_scope.enabled = False
        cfg.diffing.method.steering.enabled = False
        cfg.diffing.method.token_relevance.enabled = False

        # Minimal causal_effect config
        cfg.diffing.method.causal_effect.enabled = True
        cfg.diffing.method.causal_effect.max_samples = 2
        cfg.diffing.method.causal_effect.batch_size = 2
        cfg.diffing.method.causal_effect.max_total_tokens = 128
        cfg.diffing.method.causal_effect.after_k = 2
        cfg.diffing.method.causal_effect.num_random_vectors = 2
        cfg.diffing.method.causal_effect.num_random_diff_vectors = 0

        method = ActDiffLens(cfg)
        method.run()

        assert method.results_dir.exists()


class TestActivationOracleMethodRun:
    """Tests for ActivationOracleMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_oracle_method_initializes(self, tmp_results_dir, organism_name):
        """Test that ActivationOracleMethod can be instantiated with real config."""
        from diffing.methods.activation_oracle.method import ActivationOracleMethod

        cfg = load_test_config("activation_oracle", tmp_results_dir, organism_name)
        method = ActivationOracleMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_oracle_run(self, tmp_results_dir, organism_name):
        """Test that ActivationOracleMethod.run() completes (LoRA only)."""
        from diffing.methods.activation_oracle.method import ActivationOracleMethod

        cfg = load_test_config("activation_oracle", tmp_results_dir, organism_name)

        # Override verbalizer models - key must match model.name (SmolLM2-135M)
        cfg.diffing.method.verbalizer_models = {"SmolLM2-135M": VERBALIZER_MODEL}

        method = ActivationOracleMethod(cfg)

        # Oracle method only supports LoRA adapters, not full finetunes
        if not method.finetuned_model_cfg.is_lora:
            pytest.xfail("ActivationOracleMethod only supports LoRA adapters")

        method.run()

        assert method._results_file().exists()


class TestWeightAmplificationMethodRun:
    """Tests for WeightDifferenceAmplification."""

    def setup_method(self):
        """Clear model cache before each test to free GPU memory for vLLM."""
        from diffing.utils.model import clear_cache

        clear_cache()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_weight_amp_method_initializes(self, tmp_results_dir, organism_name):
        """Test that WeightDifferenceAmplification can be instantiated with real config."""
        from diffing.methods.amplification.weight_amplification import (
            WeightDifferenceAmplification,
        )

        cfg = load_test_config("weight_amplification", tmp_results_dir, organism_name)
        method = WeightDifferenceAmplification(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_weight_amp_with_amplification(self, tmp_results_dir, organism_name):
        """Test WeightDifferenceAmplification.run() with amplified model (LoRA only)."""
        import json

        from diffing.methods.amplification.weight_amplification import (
            WeightDifferenceAmplification,
        )

        cfg = load_test_config("weight_amplification", tmp_results_dir, organism_name)

        # Create a minimal test prompts file
        prompts_file = tmp_results_dir / "test_prompts_amp.txt"
        prompts_file.write_text("What is the capital of France?\n")
        cfg.diffing.method.run.prompts_file = str(prompts_file)

        # Minimal sampling for fast test
        cfg.diffing.method.run.sampling.max_tokens = 16
        cfg.diffing.method.run.sampling.n = 1
        cfg.diffing.method.run.sampling.temperature = 0.0

        # Run all model types including amplified
        cfg.diffing.method.run.models = ["base", "ft", "amplified"]

        # Use only one amplification preset for speed
        cfg.diffing.method.run.amplification_configs = [
            str(
                CONFIGS_DIR
                / "diffing"
                / "method"
                / "amplification_presets"
                / "default_2x.yaml"
            )
        ]

        method = WeightDifferenceAmplification(cfg)

        # Weight amplification only supports LoRA adapters
        if not method.finetuned_model_cfg.is_lora:
            pytest.xfail("WeightDifferenceAmplification only supports LoRA adapters")

        result = method.run()

        assert "request_id" in result
        logs_dir = Path(result["logs_dir"])
        assert logs_dir.exists()

        # Verify generations.jsonl contains amplified samples
        generations_file = logs_dir / "generations.jsonl"
        assert generations_file.exists()

        with generations_file.open() as f:
            record = json.loads(f.readline())

        assert "base_samples" in record
        assert "ft_samples" in record
        assert "amplified_samples" in record
        assert "default_2x" in record["amplified_samples"]
        assert len(record["amplified_samples"]["default_2x"]) > 0


class TestActivationAnalysisMethodRun:
    """Tests for ActivationAnalysisDiffingMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_activation_analysis_method_initializes(
        self, tmp_results_dir, organism_name
    ):
        """Test that ActivationAnalysisDiffingMethod can be instantiated with real config."""
        from diffing.methods.activation_analysis.method import (
            ActivationAnalysisDiffingMethod,
        )

        cfg = load_test_config("activation_analysis", tmp_results_dir, organism_name)
        method = ActivationAnalysisDiffingMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_activation_analysis_run(
        self, tmp_results_dir, preprocessed_activations, organism_name, batch_size
    ):
        """Test that ActivationAnalysisDiffingMethod.run() completes with various batch sizes."""
        from diffing.methods.activation_analysis.method import (
            ActivationAnalysisDiffingMethod,
        )

        cfg = load_test_config("activation_analysis", tmp_results_dir, organism_name)
        cfg.diffing.method.method_params.batch_size = batch_size
        cfg.diffing.method.method_params.max_samples = 5
        cfg.diffing.method.method_params.num_workers = 0
        # Only use training dataset (the one we preprocessed)
        cfg.diffing.method.datasets.use_chat_dataset = True
        cfg.diffing.method.datasets.use_pretraining_dataset = False
        cfg.diffing.method.datasets.use_training_dataset = False

        method = ActivationAnalysisDiffingMethod(cfg)
        method.run()

        results_dir = method.results_dir
        assert results_dir.exists()
        layer_dirs = list(results_dir.glob("layer_*"))
        assert len(layer_dirs) > 0


class TestSAEDifferenceMethodRun:
    """Tests for SAEDifferenceMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_sae_difference_method_initializes(self, tmp_results_dir, organism_name):
        """Test that SAEDifferenceMethod can be instantiated with real config."""
        from diffing.methods.sae_difference.method import SAEDifferenceMethod

        cfg = load_test_config("sae_difference", tmp_results_dir, organism_name)
        method = SAEDifferenceMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_sae_difference_run(
        self, tmp_results_dir, preprocessed_activations, organism_name
    ):
        """Test that SAEDifferenceMethod.run() completes with steering enabled."""
        from diffing.methods.sae_difference.method import SAEDifferenceMethod

        cfg = load_test_config("sae_difference", tmp_results_dir, organism_name)

        # Minimal training config for testing
        cfg.diffing.method.training.num_samples = 100
        cfg.diffing.method.training.num_validation_samples = 50
        cfg.diffing.method.training.batch_size = 32
        cfg.diffing.method.training.epochs = 1
        cfg.diffing.method.training.max_steps = 10
        cfg.diffing.method.training.validate_every_n_steps = 5
        cfg.diffing.method.training.workers = 0
        cfg.diffing.method.optimization.warmup_steps = 0

        # Enable analysis with minimal config
        cfg.diffing.method.analysis.enabled = True

        # Minimal latent activations config
        cfg.diffing.method.analysis.latent_activations.enabled = True
        cfg.diffing.method.analysis.latent_activations.n_max_activations = 10
        cfg.diffing.method.analysis.latent_activations.max_num_samples = 50
        cfg.diffing.method.analysis.latent_activations.overwrite = True

        # Minimal steering config - tests all steering modes
        cfg.diffing.method.analysis.latent_steering.enabled = True
        cfg.diffing.method.analysis.latent_steering.prompts_file = (
            "tests/resources/test_steering_prompts.txt"
        )
        cfg.diffing.method.analysis.latent_steering.k = 2  # Only 2 latents
        cfg.diffing.method.analysis.latent_steering.max_new_tokens = 10
        cfg.diffing.method.analysis.latent_steering.steering_factors_percentages = [0.5]
        cfg.diffing.method.analysis.latent_steering.steering_modes = [
            "all_tokens",
            "prompt_only",
        ]
        cfg.diffing.method.analysis.latent_steering.overwrite = True

        # Disable upload to HF
        cfg.diffing.method.upload.model = False

        # Only use chat dataset (the one we preprocessed)
        cfg.diffing.method.datasets.use_chat_dataset = True
        cfg.diffing.method.datasets.use_pretraining_dataset = False
        cfg.diffing.method.datasets.use_training_dataset = False

        method = SAEDifferenceMethod(cfg)
        method.run()

        assert method.results_dir.exists()
        # Verify steering results were created
        steering_results = list(
            method.results_dir.glob("**/latent_steering/test_steering_prompts.csv")
        )
        assert len(steering_results) > 0, "Steering results should be created"


class TestCrosscoderMethodRun:
    """Tests for CrosscoderDiffingMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_crosscoder_method_initializes(self, tmp_results_dir, organism_name):
        """Test that CrosscoderDiffingMethod can be instantiated with real config."""
        from diffing.methods.crosscoder.method import CrosscoderDiffingMethod

        cfg = load_test_config("crosscoder", tmp_results_dir, organism_name)
        method = CrosscoderDiffingMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_crosscoder_run(
        self, tmp_results_dir, preprocessed_activations, organism_name
    ):
        """Test that CrosscoderDiffingMethod.run() completes with steering enabled."""
        from diffing.methods.crosscoder.method import CrosscoderDiffingMethod

        cfg = load_test_config("crosscoder", tmp_results_dir, organism_name)

        # Minimal training config for testing
        cfg.diffing.method.training.num_samples = 100
        cfg.diffing.method.training.num_validation_samples = 50
        cfg.diffing.method.training.batch_size = 32
        cfg.diffing.method.training.epochs = 1
        cfg.diffing.method.training.max_steps = 10
        cfg.diffing.method.training.validate_every_n_steps = 5
        cfg.diffing.method.training.workers = 0
        cfg.diffing.method.optimization.warmup_steps = 0

        # Enable analysis with minimal config
        cfg.diffing.method.analysis.enabled = True

        # Minimal latent activations config
        cfg.diffing.method.analysis.latent_activations.enabled = True
        cfg.diffing.method.analysis.latent_activations.n_max_activations = 10
        cfg.diffing.method.analysis.latent_activations.max_num_samples = 50
        cfg.diffing.method.analysis.latent_activations.overwrite = True

        # Minimal steering config - tests all steering modes
        cfg.diffing.method.analysis.latent_steering.enabled = True
        cfg.diffing.method.analysis.latent_steering.prompts_file = (
            "tests/resources/test_steering_prompts.txt"
        )
        cfg.diffing.method.analysis.latent_steering.k = 2  # Only 2 latents
        cfg.diffing.method.analysis.latent_steering.max_new_tokens = 10
        cfg.diffing.method.analysis.latent_steering.steering_factors_percentages = [0.5]
        cfg.diffing.method.analysis.latent_steering.steering_modes = [
            "all_tokens",
            "prompt_only",
        ]
        cfg.diffing.method.analysis.latent_steering.overwrite = True

        # Disable upload to HF
        cfg.diffing.method.upload.model = False

        # Only use chat dataset (the one we preprocessed)
        cfg.diffing.method.datasets.use_chat_dataset = True
        cfg.diffing.method.datasets.use_pretraining_dataset = False
        cfg.diffing.method.datasets.use_training_dataset = False

        method = CrosscoderDiffingMethod(cfg)
        method.run()

        assert method.results_dir.exists()
        # Verify steering results were created
        steering_results = list(
            method.results_dir.glob("**/latent_steering/test_steering_prompts.csv")
        )
        assert len(steering_results) > 0, "Steering results should be created"


class TestPCAMethodRun:
    """Tests for PCAMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_pca_method_initializes(self, tmp_results_dir, organism_name):
        """Test that PCAMethod can be instantiated with real config."""
        from diffing.methods.pca import PCAMethod

        cfg = load_test_config("pca", tmp_results_dir, organism_name)
        method = PCAMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_pca_run(self, tmp_results_dir, preprocessed_activations, organism_name):
        """Test that PCAMethod.run() completes with steering enabled."""
        from diffing.methods.pca import PCAMethod

        cfg = load_test_config("pca", tmp_results_dir, organism_name)

        # Minimal training config for testing
        cfg.diffing.method.training.num_samples = 100
        cfg.diffing.method.training.batch_size = 32
        cfg.diffing.method.training.n_components = 10
        cfg.diffing.method.training.workers = 0

        # Enable analysis with minimal config
        cfg.diffing.method.analysis.enabled = True

        # Minimal max activating examples config
        cfg.diffing.method.analysis.max_activating_examples.enabled = True
        cfg.diffing.method.analysis.max_activating_examples.n_max_activations = 10
        cfg.diffing.method.analysis.max_activating_examples.max_num_samples = 50
        cfg.diffing.method.analysis.max_activating_examples.overwrite = True

        # Minimal steering config - tests all steering modes
        cfg.diffing.method.analysis.component_steering.enabled = True
        cfg.diffing.method.analysis.component_steering.prompts_file = (
            "tests/resources/test_steering_prompts.txt"
        )
        cfg.diffing.method.analysis.component_steering.k = 2  # Only 2 components
        cfg.diffing.method.analysis.component_steering.max_new_tokens = 10
        cfg.diffing.method.analysis.component_steering.steering_factors_percentages = [
            0.5
        ]
        cfg.diffing.method.analysis.component_steering.steering_modes = [
            "all_tokens",
            "prompt_only",
        ]
        cfg.diffing.method.analysis.component_steering.overwrite = True

        # Only use chat dataset (the one we preprocessed)
        cfg.diffing.method.datasets.use_chat_dataset = True
        cfg.diffing.method.datasets.use_pretraining_dataset = False
        cfg.diffing.method.datasets.use_training_dataset = False

        method = PCAMethod(cfg)
        method.run()

        assert method.results_dir.exists()
        # Verify steering results were created
        steering_results = list(
            method.results_dir.glob("**/latent_steering/test_steering_prompts.csv")
        )
        assert len(steering_results) > 0, "Steering results should be created"


class TestDiffMiningMethodRun:
    """Tests for DiffMiningMethod (logit diff top-K token analysis)."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_diff_mining_method_initializes(self, tmp_results_dir, organism_name):
        """Test that DiffMiningMethod can be instantiated with real config."""
        from diffing.methods.diff_mining import DiffMiningMethod

        cfg = load_test_config("diff_mining", tmp_results_dir, organism_name)
        method = DiffMiningMethod(cfg)
        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_diff_mining_run_in_memory(self, tmp_results_dir, organism_name):
        """Test DiffMiningMethod full pipeline (preprocess + diffing) in-memory mode."""
        from diffing.methods.diff_mining import DiffMiningMethod

        cfg = load_test_config("diff_mining", tmp_results_dir, organism_name)

        # Minimal config for fast testing
        cfg.diffing.method.max_samples = 4
        cfg.diffing.method.batch_size = 2
        cfg.diffing.method.max_tokens_per_sample = 16
        cfg.diffing.method.top_k = 10
        cfg.diffing.method.in_memory = True

        # Use direct logits (fastest extraction, no intermediate layer needed)
        cfg.diffing.method.logit_extraction.method = "logits"

        # Only top_k_occurring ordering (fastest)
        cfg.diffing.method.token_ordering.method = ["top_k_occurring"]

        # Disable expensive optional stages
        cfg.diffing.method.token_relevance.enabled = False
        cfg.diffing.method.positional_kde.enabled = False
        cfg.diffing.method.sequence_likelihood_ratio.enabled = False
        cfg.diffing.method.per_token_analysis.enabled = False

        # Use the small test dataset
        cfg.diffing.method.datasets = [
            {
                "id": TEST_DATASET_ID,
                "is_chat": True,
                "text_column": None,
                "streaming": False,
            }
        ]

        # In-memory mode needs mode=full (preprocess + diffing in one run)
        cfg.pipeline.mode = "full"

        method = DiffMiningMethod(cfg)
        method.preprocess()
        method.run()

        # has_results expects the root results_base_dir, not the method's specific dir
        assert method.has_results(Path(cfg.diffing.results_base_dir))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_diff_mining_run_disk_mode(self, tmp_results_dir, organism_name):
        """Test DiffMiningMethod with disk-based I/O (in_memory=False).

        Preprocessing writes logits to disk, then run() reads diffs from disk.
        """
        from diffing.methods.diff_mining import DiffMiningMethod

        cfg = load_test_config("diff_mining", tmp_results_dir, organism_name)

        cfg.diffing.method.max_samples = 4
        cfg.diffing.method.batch_size = 2
        cfg.diffing.method.max_tokens_per_sample = 16
        cfg.diffing.method.top_k = 10
        cfg.diffing.method.in_memory = False

        cfg.diffing.method.logit_extraction.method = "logits"
        cfg.diffing.method.token_ordering.method = ["top_k_occurring"]

        cfg.diffing.method.token_relevance.enabled = False
        cfg.diffing.method.positional_kde.enabled = False
        cfg.diffing.method.sequence_likelihood_ratio.enabled = False
        cfg.diffing.method.per_token_analysis.enabled = False

        cfg.diffing.method.datasets = [
            {
                "id": TEST_DATASET_ID,
                "is_chat": True,
                "text_column": None,
                "streaming": False,
            }
        ]

        # Use unique results dir to avoid collisions with in-memory test
        cfg.diffing.results_dir = str(Path(cfg.diffing.results_dir) / "disk_mode")

        method = DiffMiningMethod(cfg)
        method.preprocess()

        # Verify logit diffs were saved to disk
        diffs_dir = method.saved_tensors_dir / "logit_diffs"
        assert (
            diffs_dir.exists()
        ), "Logit diffs directory should exist after disk preprocessing"
        diff_files = list(diffs_dir.glob("*_logit_diff.pt"))
        assert len(diff_files) > 0, "Should have at least one logit diff file on disk"

        method.run()
        # has_results expects the root results_base_dir, not the method's specific dir
        assert method.has_results(Path(cfg.diffing.results_base_dir))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
