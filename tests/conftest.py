"""Shared test fixtures and mocks."""

import pytest
import torch


class MockDictionaryModel:
    """Mock dictionary model for testing SAE/Crosscoder encoding."""

    def __init__(self, dict_size, activation_dim, device="cpu", dtype=torch.float32):
        self.dict_size = dict_size
        self.activation_dim = activation_dim
        self.device = device
        self.dtype = dtype

    def get_activations(self, activations):
        """
        Mock get_activations that returns predictable sparse patterns.

        Args:
            activations: Tensor of shape (seq_len, activation_dim)

        Returns:
            Tensor of shape (seq_len, dict_size) with sparse latent activations
        """
        seq_len = activations.shape[0]
        latent_acts = torch.zeros(
            seq_len, self.dict_size, device=activations.device, dtype=self.dtype
        )
        for i in range(seq_len):
            latent_idx = i % self.dict_size
            latent_acts[i, latent_idx] = (i + 1) / 10.0
        return latent_acts


class MockSampleCache:
    """Mock SampleCache for testing."""

    def __init__(self, sequences_data, activation_dim=32, device="cpu"):
        """
        Args:
            sequences_data: List of tuples (tokens, seq_length) defining each sequence
            activation_dim: Dimension of activation vectors
            device: Device for tensors
        """
        self.sequences_data = sequences_data
        self.activation_dim = activation_dim
        self.device = device
        self.sample_start_indices = [0]
        for _, seq_length in sequences_data:
            self.sample_start_indices.append(self.sample_start_indices[-1] + seq_length)

    def __len__(self):
        return len(self.sequences_data)

    def __getitem__(self, index):
        tokens, seq_length = self.sequences_data[index]
        activations = torch.randn(seq_length, self.activation_dim, device=self.device)
        return tokens, activations


@pytest.fixture
def mock_dictionary_model():
    """Fixture providing a MockDictionaryModel factory."""
    return MockDictionaryModel


@pytest.fixture
def mock_sample_cache():
    """Fixture providing a MockSampleCache factory."""
    return MockSampleCache


# --- Mock OpenAI Server ---

from mock_openai_server import _mock_openai_server_fixture

mock_openai_server = pytest.fixture(scope="session")(_mock_openai_server_fixture)


# --- Block External LLM API Calls ---

import httpx
import respx


@pytest.fixture(autouse=True)
def block_external_llm_calls(mock_openai_server):
    """Redirect external LLM API calls to the mock server.

    Instead of blocking with exceptions (which the OpenAI SDK wraps as
    APIConnectionError and retries with exponential backoff), forward
    requests to the mock server which returns proper HTTP responses.
    """
    mock_url = mock_openai_server.base_url

    def redirect_to_mock(request: httpx.Request) -> httpx.Response:
        with httpx.Client() as client:
            return client.post(
                f"{mock_url}/chat/completions",
                content=request.content,
                headers={"Content-Type": "application/json"},
            )

    with respx.mock(assert_all_called=False) as mock:
        mock.route(host__regex=r"^(localhost|127\.0\.0\.1)$").pass_through()
        mock.route(path__regex=r".*/chat/completions.*").mock(
            side_effect=redirect_to_mock
        )
        mock.route(path__regex=r".*/completions.*").mock(side_effect=redirect_to_mock)
        mock.route().pass_through()
        yield
