from transformers import AutoConfig
from typing import Union, List
from transformers import PretrainedConfig
from loguru import logger
from dictionary_learning.cache import PairedActivationCache
from pathlib import Path
from omegaconf import DictConfig
import torch
from tqdm import trange


from .configs import DatasetConfig, ModelConfig, get_safe_model_id

from math import ceil, floor


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    FROM: https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451
    Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def get_layer_indices(model: Union[str, object], layers: List[float]) -> List[int]:
    """
    Get the indices of the layers to collect activations from.
    """
    if isinstance(model, str):
        config: PretrainedConfig = AutoConfig.from_pretrained(
            model, trust_remote_code=True
        )
        try:
            num_layers: int = config.num_hidden_layers
        except:
            logger.warning(f"Using num_hidden_layers from Gemma3TextConfig for {model}")
            num_layers: int = config.text_config.num_hidden_layers
    else:
        num_layers: int = len(model.model.layers)
    return [int(layer * (num_layers - 1)) for layer in layers]


def get_local_shuffled_indices(
    num_samples_per_dataset: List[int], shard_size: int, epochs: int
) -> torch.Tensor:
    """
    Create locally shuffled indices for cache-friendly data loading with multiple datasets.

    This function assumes that the datasets are concatenated.
    It will make sure that for each dataset shuffling only happens within shards (e.g. shuffle within the first 1M samples, then within the next 1M samples, etc.).
    Across datasets, the first shards are shuffled together, then the second shards, etc.
    The function makes sure that the fraction of samples from each dataset is proportional to the dataset size.

    Args:
        num_samples_per_dataset: List of sample counts for each dataset
        shard_size: Size of each shard for local shuffling
        epochs: Number of epochs to generate indices for

    Returns:
        Tensor of shuffled indices with shape (sum(num_samples_per_dataset) * epochs,)
    """
    assert len(num_samples_per_dataset) > 0, "Must have at least one dataset"
    assert all(
        n > 0 for n in num_samples_per_dataset
    ), f"All dataset sizes must be positive, got {num_samples_per_dataset}"
    assert shard_size > 0, f"shard_size must be positive, got {shard_size}"
    assert epochs > 0, f"epochs must be positive, got {epochs}"

    num_datasets = len(num_samples_per_dataset)
    total_samples = sum(num_samples_per_dataset)

    # Calculate cumulative offsets for each dataset in the concatenated structure
    cumulative_offsets = [0]
    for i in range(num_datasets):
        cumulative_offsets.append(cumulative_offsets[-1] + num_samples_per_dataset[i])

    # Calculate proportional weights for interleaving
    max_ds_size = max(num_samples_per_dataset)
    total_number_of_shards = max_ds_size // shard_size + (
        1 if max_ds_size % shard_size != 0 else 0
    )

    dataset_weights = [n / max_ds_size for n in num_samples_per_dataset]

    logger.info(
        f"Creating local shuffled indices: shuffling in {total_number_of_shards} shards of size {shard_size} across {num_datasets} datasets"
    )
    logger.debug(f"Dataset sizes: {num_samples_per_dataset}")
    logger.debug(
        f"Dataset weights: {[f'{w:.3f}' for w in dataset_weights]}. Reducing shard size per dataset according to weights."
    )

    all_shuffled_indices = []
    epoch_numbers = []

    for epoch in range(epochs):
        epoch_indices = []

        # Track remaining samples per dataset for this epoch
        remaining_samples = num_samples_per_dataset.copy()

        i = 0
        while sum(remaining_samples) > 0:
            shard_indices = []
            for j in range(num_datasets):
                samples_for_dataset = min(
                    max(1, int(shard_size * dataset_weights[j])), remaining_samples[j]
                )

                # Calculate base index for this dataset in the current position
                dataset_start = cumulative_offsets[j] + (
                    num_samples_per_dataset[j] - remaining_samples[j]
                )

                # Create shuffled indices within this dataset's shard
                dataset_indices = torch.arange(samples_for_dataset) + dataset_start
                shard_indices.append(dataset_indices)

                # Update remaining samples
                remaining_samples[j] -= samples_for_dataset

            # Shuffle the shard indices from all datasets for this shard
            if shard_indices:
                # Flatten and interleave proportionally
                all_dataset_indices = torch.cat(shard_indices)
                shuffled_indices = all_dataset_indices[
                    torch.randperm(all_dataset_indices.shape[0])
                ]
                epoch_indices.append(shuffled_indices)

        all_shuffled_indices.append(torch.cat(epoch_indices))
        epoch_numbers.append(
            torch.full((all_shuffled_indices[-1].shape[0],), epoch, dtype=torch.long)
        )
        i += 1

    final_indices = torch.cat(all_shuffled_indices)

    assert (
        final_indices.shape[0] == total_samples * epochs
    ), f"Expected {total_samples * epochs} indices, got {final_indices.shape[0]}"

    return final_indices, torch.cat(epoch_numbers)


def calculate_samples_per_dataset(
    dataset_lengths: List[int], max_total_samples: int
) -> List[int]:
    """
    Calculate the number of samples to take from each dataset, proportionally scaled
    to dataset size while respecting the maximum total sample limit.

    Args:
        dataset_lengths: List of lengths for each dataset
        max_total_samples: Maximum total number of samples to use across all datasets

    Returns:
        List of sample counts per dataset, proportionally scaled
    """
    assert all(
        length > 0 for length in dataset_lengths
    ), "All dataset lengths must be positive"
    assert max_total_samples > 0, "max_total_samples must be positive"
    assert len(dataset_lengths) > 0, "Must have at least one dataset"

    total_available = sum(dataset_lengths)

    # If we have fewer samples available than requested, use all available
    if total_available <= max_total_samples:
        return dataset_lengths

    # Calculate proportional allocation
    samples_per_dataset = []
    for length in dataset_lengths:
        proportion = length / total_available
        allocated_samples = int(proportion * max_total_samples)
        samples_per_dataset.append(allocated_samples)

    # Handle rounding errors by distributing remaining samples
    allocated_total = sum(samples_per_dataset)
    remaining = max_total_samples - allocated_total

    # make sure we don't exceed the dataset size
    for i, length in enumerate(dataset_lengths):
        samples_per_dataset[i] = min(samples_per_dataset[i], length)

    assert (
        sum(samples_per_dataset) <= max_total_samples
    ), "Total samples exceeded maximum"
    assert all(
        samples_per_dataset[i] <= dataset_lengths[i]
        for i in range(len(dataset_lengths))
    ), "Sample count exceeded dataset size"

    return samples_per_dataset


def load_activation_dataset(
    activation_store_dir: Path,
    split: str,
    dataset_name: str,
    base_model: str = "gemma-2-2b",
    finetuned_model: str = "gemma-2-2b-it",
    text_column: str = None,
    layer: int = 13,
):
    """
    Load the saved activations of the base and finetuned models for a given layer and dataset.

    Args:
        activation_store_dir: The directory where the activations are stored
        split: The split to load (e.g., 'train', 'validation', 'test')
        dataset_name: The name of the dataset to load
        base_model: The base model identifier (default: "gemma-2-2b")
        finetuned_model: The finetuned model identifier (default: "gemma-2-2b-it")
        text_column: The name of the text column in the dataset (default: None). If not None and not "text", the split will be appended with the text column name (e.g., "train_col_formatedbase").
        layer: The layer number to load activations from (default: 13)

    Returns:
        PairedActivationCache: A cache containing paired activations from both models
    """
    # Load validation datase
    activation_store_dir = Path(activation_store_dir)
    base_model_dir = activation_store_dir / base_model
    instruct_model_dir = activation_store_dir / finetuned_model

    submodule_name = f"layer_{layer}_out"

    if text_column is not None and text_column != "text":
        split = split + f"_col_{text_column}"

    # Load validation caches
    base_model_cache = base_model_dir / dataset_name / split
    finetuned_model_cache = instruct_model_dir / dataset_name / split

    cache = PairedActivationCache(
        base_model_cache, finetuned_model_cache, submodule_name
    )

    return cache


def load_activation_datasets(
    activation_store_dir: Path,
    split: str,
    dataset_names: list[str],
    base_model: str = "gemma-2-2b",
    finetuned_model: str = "gemma-2-2b-it",
    layers: list[int] = [13],
    text_columns: str = None,
):
    """
    Load the saved activations of the base and instruct models for multiple datasets and layers.

    Args:
        activation_store_dir: The directory where the activations are stored
        split: The split to load
        dataset_names: List of dataset names to load
        base_model: The base model to load
        finetuned_model: The finetuned model to load
        layers: List of layers to load
        text_columns: List of text columns to load. If not None, the split will be appended with the text column name (e.g., "train_col_formatedbase").

    Returns:
        A dict mapping dataset_name -> {layer: PairedActivationCache, ...}
    """
    result = {}

    for i, dataset_name in enumerate(dataset_names):
        result[dataset_name] = {}

        for layer in layers:
            cache = load_activation_dataset(
                activation_store_dir=activation_store_dir,
                split=split,
                dataset_name=dataset_name,
                base_model=base_model,
                finetuned_model=finetuned_model,
                layer=layer,
                text_column=text_columns[i] if text_columns is not None else None,
            )

            result[dataset_name][layer] = cache

    return result


def load_activation_dataset_from_config(
    cfg: DictConfig,
    ds_cfg: DatasetConfig,
    base_model_cfg: ModelConfig,
    finetuned_model_cfg: ModelConfig,
    split: str,
    layer: int,
):
    """
    Load saved activations for a specific dataset and layer using configuration objects.

    Args:
        cfg: Full configuration containing activation store directory
        ds_cfg: Dataset configuration specifying split, name, and text column
        base_model_cfg: Base model configuration with model_id
        finetuned_model_cfg: Finetuned model configuration with model_id
        layer: Layer index to load activations for

    Returns:
        PairedActivationCache containing base and finetuned model activations
    """

    return load_activation_dataset(
        activation_store_dir=cfg.preprocessing.activation_store_dir,
        split=split,
        dataset_name=ds_cfg.name,
        base_model=get_safe_model_id(base_model_cfg),
        finetuned_model=get_safe_model_id(finetuned_model_cfg),
        layer=layer,
        text_column=ds_cfg.text_column,
    )


def load_activation_datasets_from_config(
    cfg: DictConfig,
    ds_cfgs: List[DatasetConfig],
    base_model_cfg: ModelConfig,
    finetuned_model_cfg: ModelConfig,
    layers: List[int],
    split: str,
):
    """
    Load saved activations for multiple datasets and layers using configuration objects.

    Args:
        cfg: Full configuration containing activation store directory
        ds_cfgs: List of dataset configurations to load
        base_model_cfg: Base model configuration with model_id
        finetuned_model_cfg: Finetuned model configuration with model_id
        layers: List of layer indices to load activations for

    Returns:
        Dict mapping dataset_name -> {layer: PairedActivationCache, ...}
    """
    result = {}
    for ds_cfg in ds_cfgs:
        result[ds_cfg.name] = {}
        for layer in layers:
            result[ds_cfg.name][layer] = load_activation_dataset_from_config(
                cfg=cfg,
                ds_cfg=ds_cfg,
                base_model_cfg=base_model_cfg,
                finetuned_model_cfg=finetuned_model_cfg,
                layer=layer,
                split=split,
            )
    return result
