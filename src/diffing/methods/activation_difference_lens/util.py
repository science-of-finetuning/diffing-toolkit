from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import torch


def dataset_dir_name(dataset_id: str) -> str:
    name = dataset_id.split("/")[-1]
    assert len(name) > 0
    return name


def layer_dir(results_dir: Path, dataset_id: str, layer_index: int) -> Path:
    return results_dir / f"layer_{layer_index}" / dataset_dir_name(dataset_id)


def norms_path(results_dir: Path, dataset_id: str) -> Path:
    return results_dir / f"model_norms_{dataset_dir_name(dataset_id)}.pt"


def missing_norm_layers(norms_file: Path, layers: Iterable[int]) -> List[int]:
    """Return the layers that have no saved norm in ``norms_file`` (all of them if it does not exist)."""
    layers = list(layers)
    if not norms_file.exists():
        return layers
    norms_data = torch.load(norms_file, map_location="cpu")
    return [
        layer
        for layer in layers
        if layer not in norms_data["base_model_norms"]
        or layer not in norms_data["ft_model_norms"]
    ]


def get_layer_norm(
    norms_data: Dict[str, Any], model_key: str, layer_index: int
) -> float:
    """Return the saved mean norm of ``model_key`` at ``layer_index``, with a clear error if it is missing."""
    layer_norms = norms_data[model_key]
    if layer_index not in layer_norms:
        raise ValueError(
            f"No {model_key} entry for layer {layer_index}: norms were only computed "
            f"for layers {sorted(layer_norms)}. Re-run activation_difference_lens on "
            f"this dataset with layer {layer_index} included to compute them."
        )
    return float(layer_norms[layer_index].item())


def position_files_exist(
    layer_dir_path: Path, position_idx_zero_based: int, need_logit_lens: bool
) -> bool:
    mean_pt = layer_dir_path / f"mean_pos_{position_idx_zero_based}.pt"
    meta = layer_dir_path / f"mean_pos_{position_idx_zero_based}.meta"
    if not (mean_pt.exists() and meta.exists()):
        return False
    if need_logit_lens:
        ll_pt = layer_dir_path / f"logit_lens_pos_{position_idx_zero_based}.pt"
        base_ll_pt = (
            layer_dir_path / f"base_logit_lens_pos_{position_idx_zero_based}.pt"
        )
        ft_ll_pt = layer_dir_path / f"ft_logit_lens_pos_{position_idx_zero_based}.pt"
        if not (ll_pt.exists() and base_ll_pt.exists() and ft_ll_pt.exists()):
            return False
    return True


def is_layer_complete(
    results_dir: Path,
    dataset_id: str,
    layer_index: int,
    n_positions: int,
    need_logit_lens: bool,
) -> bool:
    layer_dir_path = layer_dir(results_dir, dataset_id, layer_index)
    if not layer_dir_path.exists():
        return False
    for p in range(n_positions):
        if not position_files_exist(layer_dir_path, p, need_logit_lens):
            return False
    return True


def load_position_mean_vector(
    method: Any,
    dataset_id: str,
    layer_index: int,
    position_index: int,
    type_key: str = "",
) -> torch.Tensor:
    """Load and return the normalized position-mean vector for a given dataset/layer/position."""
    dataset_dir_name = dataset_id.split("/")[-1]
    tensor_path = (
        method.results_dir
        / f"layer_{layer_index}"
        / dataset_dir_name
        / f"{type_key}mean_pos_{position_index}.pt"
    )
    assert tensor_path.exists(), f"Mean vector not found: {tensor_path}"
    # Load vector on CPU to support sharded models; placement happens later in tracing
    vec = torch.load(tensor_path, map_location="cpu")
    vec = torch.as_tensor(vec, device="cpu").flatten()
    assert vec.ndim == 1
    hidden_size = method.finetuned_model.config.hidden_size
    assert vec.shape == (
        hidden_size,
    ), f"Expected shape ({hidden_size},), got {vec.shape}"
    norm = torch.norm(vec)
    assert torch.isfinite(norm) and norm > 0

    # Load expected finetuned model norm for this dataset/layer
    norms_path = method.results_dir / f"model_norms_{dataset_dir_name}.pt"
    assert norms_path.exists(), f"Model norms file not found: {norms_path}"
    norms_data = torch.load(norms_path, map_location="cpu")
    ft_norm = get_layer_norm(norms_data, "ft_model_norms", layer_index)
    assert ft_norm > 0

    return (vec / norm) * ft_norm
