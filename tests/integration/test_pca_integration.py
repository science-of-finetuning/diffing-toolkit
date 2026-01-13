"""
Integration tests for PCA pipeline with real models.

Tests PCA training on activation differences from real models (gpt2 and gpt2-alpaca-lora),
projection of new activations through trained PCA, and verification of explained variance
and component shapes.

Run with GPU: source ~/.slurm_aliases && lrun uv run pytest tests/integration/test_pca_integration.py -v
"""

import pytest
import torch
import pickle
import tempfile
from pathlib import Path

from torchdr import IncrementalPCA
from nnterp import StandardizedTransformer
from transformers import AutoTokenizer

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

BASE_MODEL_ID = "gpt2"
FINETUNED_MODEL_ID = "monsterapi/gpt2_alpaca-lora"
TEST_LAYER = 6
N_COMPONENTS = 64
BATCH_SIZE = 32


@pytest.fixture(scope="module")
def base_model():
    """Load GPT-2 base model."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = StandardizedTransformer(
        BASE_MODEL_ID,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def finetuned_model():
    """Load GPT-2 with Alpaca LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = StandardizedTransformer(
        BASE_MODEL_ID,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.dispatch()
    model.load_adapter(FINETUNED_MODEL_ID)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer for GPT-2."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def test_prompts():
    """Test prompts for activation extraction."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "In the beginning, there was nothing but darkness.",
        "Python is a popular programming language for data science.",
    ]


def extract_activations(
    model: StandardizedTransformer,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    layer: int,
) -> torch.Tensor:
    """
    Extract activations from a specific layer for given prompts.

    Args:
        model: StandardizedTransformer model
        tokenizer: Tokenizer for the model
        prompts: List of text prompts
        layer: Layer index to extract activations from

    Returns:
        Tensor of shape [total_tokens, hidden_dim] containing flattened activations
    """
    all_activations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            with model.trace(dict(input_ids=input_ids, attention_mask=attention_mask)):
                layer_output = model.layers_output[layer].save()

        acts = layer_output.cpu()  # [batch_size, seq_len, hidden_dim]
        assert acts.ndim == 3
        assert acts.shape[0] == 1

        seq_len = attention_mask.sum().item()
        acts_valid = acts[0, :seq_len, :]  # [seq_len, hidden_dim]
        all_activations.append(acts_valid)

    concatenated = torch.cat(all_activations, dim=0)
    assert concatenated.ndim == 2
    return concatenated


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPCAIntegration:
    """Integration tests for PCA pipeline with real models."""

    def test_activation_extraction(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Test that activations can be extracted from both models."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )

        assert base_acts.shape == ft_acts.shape
        assert base_acts.ndim == 2
        assert base_acts.shape[1] == base_model.hidden_size

        assert not torch.allclose(base_acts, ft_acts, atol=1e-3)

    def test_pca_training_on_activation_differences(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Test PCA training on activation differences from real models."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )

        differences = ft_acts - base_acts
        assert differences.shape == base_acts.shape

        hidden_dim = differences.shape[1]
        n_components = min(N_COMPONENTS, hidden_dim, differences.shape[0])

        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, differences.shape[0]),
            device=DEVICE,
        )

        pca.partial_fit(differences.to(DEVICE))

        assert pca.components_ is not None
        assert pca.components_.shape == (n_components, hidden_dim)
        assert pca.explained_variance_ is not None
        assert pca.explained_variance_.shape == (n_components,)
        assert pca.explained_variance_ratio_ is not None
        assert pca.explained_variance_ratio_.shape == (n_components,)
        assert pca.mean_ is not None
        assert pca.mean_.shape == (hidden_dim,)

    def test_pca_projection_of_new_activations(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Test projection of new activations through trained PCA."""
        base_acts = extract_activations(
            base_model, tokenizer, test_prompts[:2], TEST_LAYER
        )
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts[:2], TEST_LAYER
        )
        train_diffs = ft_acts - base_acts

        hidden_dim = train_diffs.shape[1]
        n_components = min(N_COMPONENTS, hidden_dim, train_diffs.shape[0])

        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, train_diffs.shape[0]),
            device=DEVICE,
        )
        pca.partial_fit(train_diffs.to(DEVICE))

        new_base_acts = extract_activations(
            base_model, tokenizer, test_prompts[2:], TEST_LAYER
        )
        new_ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts[2:], TEST_LAYER
        )
        new_diffs = new_ft_acts - new_base_acts

        projections = pca.transform(new_diffs.to(DEVICE))

        assert projections.shape == (new_diffs.shape[0], n_components)

        # Manual reconstruction: X_reconstructed = X_projected @ components + mean
        reconstructed = projections @ pca.components_ + pca.mean_
        assert reconstructed.shape == new_diffs.shape

    def test_explained_variance_properties(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Verify explained variance ratios are valid probabilities and sum correctly."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )
        differences = ft_acts - base_acts

        hidden_dim = differences.shape[1]
        n_components = min(N_COMPONENTS, hidden_dim, differences.shape[0])

        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, differences.shape[0]),
            device=DEVICE,
        )
        pca.partial_fit(differences.to(DEVICE))

        ratios = pca.explained_variance_ratio_
        assert torch.all(ratios >= 0), "Explained variance ratios must be non-negative"
        assert torch.all(ratios <= 1), "Explained variance ratios must be <= 1"
        assert torch.isclose(ratios.sum(), torch.tensor(1.0, device=DEVICE), atol=1e-2)

        assert torch.all(
            ratios[:-1] >= ratios[1:]
        ), "Variance ratios should be sorted descending"

    def test_component_shapes_and_orthogonality(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Verify PCA components have correct shapes and are orthonormal."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )
        differences = ft_acts - base_acts

        hidden_dim = differences.shape[1]
        n_components = min(N_COMPONENTS, hidden_dim, differences.shape[0])

        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, differences.shape[0]),
            device=DEVICE,
        )
        pca.partial_fit(differences.to(DEVICE))

        components = pca.components_
        assert components.shape == (n_components, hidden_dim)

        norms = torch.norm(components, dim=1)
        assert torch.allclose(norms, torch.ones(n_components, device=DEVICE), atol=1e-4)

        gram = components @ components.T
        identity = torch.eye(n_components, device=DEVICE)
        assert torch.allclose(gram, identity, atol=1e-4)

    def test_incremental_pca_multiple_batches(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Test IncrementalPCA fitting across multiple batches."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )
        all_diffs = ft_acts - base_acts

        hidden_dim = all_diffs.shape[1]
        n_samples = all_diffs.shape[0]
        n_components = min(16, hidden_dim, n_samples)
        batch_size = max(n_components, n_samples // 2)

        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=DEVICE,
        )

        # Split into batches and fit incrementally
        mid = n_samples // 2
        pca.partial_fit(all_diffs[:mid].to(DEVICE))
        pca.partial_fit(all_diffs[mid:].to(DEVICE))

        assert pca.n_samples_seen_ == n_samples
        assert pca.components_ is not None
        assert pca.components_.shape == (n_components, hidden_dim)

    def test_pca_model_save_and_load(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Test saving and loading PCA model state."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )
        differences = ft_acts - base_acts

        hidden_dim = differences.shape[1]
        n_components = min(N_COMPONENTS, hidden_dim, differences.shape[0])

        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, differences.shape[0]),
            device=DEVICE,
        )
        pca.partial_fit(differences.to(DEVICE))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "pca_model.pkl"

            pca_state = {
                "components_": pca.components_.cpu(),
                "explained_variance_": pca.explained_variance_.cpu(),
                "explained_variance_ratio_": pca.explained_variance_ratio_.cpu(),
                "mean_": pca.mean_.cpu(),
                "var_": pca.var_.cpu() if pca.var_ is not None else None,
                "n_components": pca.n_components,
                "n_samples_seen_": pca.n_samples_seen_,
            }
            with open(model_path, "wb") as f:
                pickle.dump(pca_state, f)

            with open(model_path, "rb") as f:
                loaded_state = pickle.load(f)

            loaded_pca = IncrementalPCA(
                n_components=loaded_state["n_components"],
                batch_size=BATCH_SIZE,
                device=DEVICE,
            )
            loaded_pca.components_ = loaded_state["components_"].to(DEVICE)
            loaded_pca.explained_variance_ = loaded_state["explained_variance_"].to(
                DEVICE
            )
            loaded_pca.explained_variance_ratio_ = loaded_state[
                "explained_variance_ratio_"
            ].to(DEVICE)
            loaded_pca.mean_ = loaded_state["mean_"].to(DEVICE)
            if loaded_state["var_"] is not None:
                loaded_pca.var_ = loaded_state["var_"].to(DEVICE)
            loaded_pca.n_samples_seen_ = loaded_state["n_samples_seen_"]

            test_input = differences[:5].to(DEVICE)
            original_proj = pca.transform(test_input)
            loaded_proj = loaded_pca.transform(test_input)

            assert torch.allclose(original_proj, loaded_proj, atol=1e-5)

    def test_different_targets(
        self, base_model, finetuned_model, tokenizer, test_prompts
    ):
        """Test PCA with different target directions (ftb vs bft)."""
        base_acts = extract_activations(base_model, tokenizer, test_prompts, TEST_LAYER)
        ft_acts = extract_activations(
            finetuned_model, tokenizer, test_prompts, TEST_LAYER
        )

        diff_ftb = ft_acts - base_acts
        diff_bft = base_acts - ft_acts

        assert torch.allclose(diff_ftb, -diff_bft)

        hidden_dim = diff_ftb.shape[1]
        n_components = min(N_COMPONENTS, hidden_dim, diff_ftb.shape[0])

        pca_ftb = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, diff_ftb.shape[0]),
            device=DEVICE,
        )
        pca_ftb.partial_fit(diff_ftb.to(DEVICE))

        pca_bft = IncrementalPCA(
            n_components=n_components,
            batch_size=min(BATCH_SIZE, diff_bft.shape[0]),
            device=DEVICE,
        )
        pca_bft.partial_fit(diff_bft.to(DEVICE))

        assert torch.allclose(
            pca_ftb.explained_variance_ratio_,
            pca_bft.explained_variance_ratio_,
            atol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
