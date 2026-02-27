"""
Steering vector configuration and loading logic.

Handles loading steering vector configs from YAML files and resolving vector paths
from local disk or HuggingFace Hub.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors.torch import load_file as load_safetensors

from diffing.utils.configs import CONFIGS_DIR


STEERING_VECTORS_DIR = CONFIGS_DIR / "steering_vectors"


@dataclass
class SteeringVectorConfig:
    """Configuration for a steering vector with per-model paths.

    Similar to organism configs but for steering vectors.
    Paths can be local files or HuggingFace repo paths (user/repo/path/to/file.pt).
    """

    name: str
    description: str
    vectors: dict[str, str] = field(default_factory=dict)  # {model_name: path_or_hf_repo}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "vectors": self.vectors,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SteeringVectorConfig":
        return SteeringVectorConfig(
            name=data["name"],
            description=data.get("description", ""),
            vectors=data.get("vectors", {}),
        )

    @staticmethod
    def load_yaml(path: Path) -> "SteeringVectorConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return SteeringVectorConfig.from_dict(data)

    def save_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def get_vector_path(self, model_name: str) -> str | None:
        """Get the vector path for a specific model name."""
        return self.vectors.get(model_name)

    def has_vector_for_model(self, model_name: str) -> bool:
        """Check if this config has a vector for the given model."""
        return model_name in self.vectors

    def load_vector(self, model_name: str) -> torch.Tensor:
        """Load the steering vector tensor for the given model.

        Args:
            model_name: Name of the base model (e.g., "llama31_8B_Instruct")

        Returns:
            Steering vector tensor of shape [hidden_size]

        Raises:
            KeyError: If no vector is defined for this model
            FileNotFoundError: If the vector file cannot be found
        """
        path_or_repo = self.get_vector_path(model_name)
        if path_or_repo is None:
            raise KeyError(
                f"No steering vector defined for model '{model_name}' in config '{self.name}'. "
                f"Available models: {list(self.vectors.keys())}"
            )

        # Check if it's a local path first
        local_path = Path(path_or_repo)
        if local_path.exists():
            return _load_tensor_file(local_path)

        # Try to resolve as HuggingFace path (user/repo/path/to/file.pt)
        return _load_from_huggingface(path_or_repo)


def _load_tensor_file(path: Path) -> torch.Tensor:
    """Load tensor from .pt or .safetensors file.

    Returns:
        Tensor of shape [d_model] (single vector) or [num_layers, d_model] (per-layer vectors)
    """
    suffix = path.suffix.lower()
    if suffix == ".pt" or suffix == ".pth":
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    elif suffix == ".safetensors":
        tensors = load_safetensors(str(path))
        # Assume single tensor or take the first one
        if len(tensors) == 1:
            tensor = next(iter(tensors.values()))
        elif "vector" in tensors:
            tensor = tensors["vector"]
        elif "steering_vector" in tensors:
            tensor = tensors["steering_vector"]
        else:
            raise ValueError(
                f"Safetensors file {path} has multiple tensors: {list(tensors.keys())}. "
                f"Expected single tensor or key 'vector'/'steering_vector'."
            )
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pt, .pth, or .safetensors")

    # Support both 1D [d_model] and 2D [num_layers, d_model]
    tensor = tensor.squeeze()
    assert tensor.ndim in (1, 2), f"Expected 1D or 2D tensor, got shape {tensor.shape}"
    return tensor


def _load_from_huggingface(path_spec: str) -> torch.Tensor:
    """Load tensor from HuggingFace Hub.

    Args:
        path_spec: Path in format "user/repo/path/to/file.pt" or "user/repo"

    Returns:
        Loaded tensor
    """
    parts = path_spec.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid HuggingFace path: '{path_spec}'. "
            f"Expected format: 'user/repo/path/to/file.pt' or 'user/repo'"
        )

    repo_id = "/".join(parts[:2])
    if len(parts) > 2:
        filename = "/".join(parts[2:])
    else:
        # Try common filenames
        for candidate in ["steering_vector.pt", "vector.pt", "steering_vector.safetensors"]:
            try:
                local_path = hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="dataset")
                logger.info(f"Downloaded steering vector from {repo_id}/{candidate}")
                return _load_tensor_file(Path(local_path))
            except Exception:
                continue
        raise FileNotFoundError(
            f"Could not find steering vector in repo '{repo_id}'. "
            f"Tried: steering_vector.pt, vector.pt, steering_vector.safetensors"
        )

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    logger.info(f"Downloaded steering vector from {repo_id}/{filename}")
    return _load_tensor_file(Path(local_path))


def get_available_steering_vectors() -> list[str]:
    """Get list of available steering vector config names."""
    if not STEERING_VECTORS_DIR.exists():
        return []
    return [p.stem for p in STEERING_VECTORS_DIR.glob("*.yaml")]


def load_steering_vector_config(name: str) -> SteeringVectorConfig:
    """Load a steering vector config by name.

    Args:
        name: Name of the config (without .yaml extension)

    Returns:
        SteeringVectorConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = STEERING_VECTORS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Steering vector config not found: {path}. "
            f"Available: {get_available_steering_vectors()}"
        )
    return SteeringVectorConfig.load_yaml(path)


# Cache for loaded vectors to avoid repeated disk/network access
_vector_cache: dict[tuple[str, str], torch.Tensor] = {}


def load_steering_vector_cached(config_name: str, model_name: str) -> torch.Tensor:
    """Load steering vector with caching.

    Args:
        config_name: Name of the steering vector config
        model_name: Name of the base model

    Returns:
        Steering vector tensor (cached)
    """
    cache_key = (config_name, model_name)
    if cache_key not in _vector_cache:
        config = load_steering_vector_config(config_name)
        _vector_cache[cache_key] = config.load_vector(model_name)
    return _vector_cache[cache_key]


def clear_vector_cache():
    """Clear the steering vector cache."""
    _vector_cache.clear()
