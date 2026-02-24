import torch as th
from pathlib import Path
import numpy as np
from warnings import warn
from dictionary_learning import CrossCoder, BatchTopKCrossCoder, BatchTopKSAE
from typing import Literal


def remove_latents(
    activation: th.Tensor, latent_activations: th.Tensor, latent_vectors: th.Tensor
) -> th.Tensor:
    # activation: N x dim_model
    # latent_vectors: num_latent_vectors x dim_model
    # latent_activations: N x num_latent_vectors
    # return: num_latent_vectors x N x dim_model
    num_latent_vectors = latent_vectors.shape[0]
    dim_model = latent_vectors.shape[1]
    N = activation.shape[0]

    assert latent_activations.shape == (N, num_latent_vectors)
    assert activation.shape == (N, dim_model)
    assert latent_vectors.shape == (num_latent_vectors, dim_model)

    # stack the activations num_latent_vectors times -> (num_latent_vectors, N, dim_model)
    activation_stacked = activation.unsqueeze(0).repeat(num_latent_vectors, 1, 1)
    assert activation_stacked.shape == (num_latent_vectors, N, dim_model)

    # remove the latents from the activation
    latent_vectors_reshaped = latent_vectors.unsqueeze(1).repeat(1, N, 1)
    assert latent_vectors_reshaped.shape == (num_latent_vectors, N, dim_model)
    # scale by latent_activations
    latent_activations_reshaped = latent_activations.T.unsqueeze(
        -1
    )  # (num_latent_vectors, N, 1)
    assert latent_activations_reshaped.shape == (num_latent_vectors, N, 1)

    # remove the latents
    activation_stacked = activation_stacked - (
        latent_vectors_reshaped * latent_activations_reshaped
    )
    assert activation_stacked.shape == (num_latent_vectors, N, dim_model)
    return activation_stacked


def identity_fn(
    x: th.Tensor, crosscoder: CrossCoder = None, normalize: bool = False
) -> th.Tensor:
    if normalize and crosscoder is not None:
        return crosscoder.normalize_activations(x)
    return x


def normalize_batch_and_index_layer(
    batch, crosscoder: CrossCoder = None, layer: int = 0, normalize: bool = False
):
    if not normalize:
        return batch[:, layer, :]
    if isinstance(crosscoder, CrossCoder) or isinstance(
        crosscoder, BatchTopKCrossCoder
    ):
        # The crosscoder normalizer expects stacked activations of shape (batch_size, num_layers, dict_size)
        return crosscoder.normalize_activations(batch, inplace=False)[:, layer, :]
    elif isinstance(crosscoder, BatchTopKSAE):
        # The sae normalizer expects single activations of shape (batch_size, activation_dim)
        return crosscoder.normalize_activations(batch[:, layer, :], inplace=False)
    else:
        return batch


def load_base_activation(
    batch, crosscoder: CrossCoder = None, normalize: bool = False, **kwargs
):
    return normalize_batch_and_index_layer(batch, crosscoder, 0, normalize)


def load_ft_activation(
    batch, crosscoder: CrossCoder = None, normalize: bool = False, **kwargs
):
    return normalize_batch_and_index_layer(batch, crosscoder, 1, normalize)


def load_difference_activation(
    batch,
    sae_model: Literal["base", "ft"],
    crosscoder: CrossCoder = None,
    normalize: bool = False,
    **kwargs,
):
    """Load activation difference (ft - base) or (base - ft) from difference cache"""
    if sae_model == "ft":
        return identity_fn(
            load_ft_activation(batch) - load_base_activation(batch),
            crosscoder,
            normalize,
        )
    else:
        return identity_fn(
            load_base_activation(batch) - load_ft_activation(batch),
            crosscoder,
            normalize,
        )


def load_base_activation_no_bias(
    batch, crosscoder: CrossCoder = None, normalize: bool = False, **kwargs
):
    return (
        normalize_batch_and_index_layer(batch, crosscoder, 0, normalize)
        - crosscoder.decoder.bias[0, :]
    )


def load_ft_activation_no_bias(
    batch, crosscoder: CrossCoder = None, normalize: bool = False, **kwargs
):
    return (
        normalize_batch_and_index_layer(batch, crosscoder, 1, normalize)
        - crosscoder.decoder.bias[1, :]
    )


def load_base_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    base_decoder: th.Tensor,
    normalize: bool = False,
    **kwargs,
):
    assert isinstance(crosscoder, CrossCoder) or isinstance(
        crosscoder, BatchTopKCrossCoder
    ), "Base error requires a crosscoder"
    reconstruction = crosscoder.decode(
        latent_activations, denormalize_activations=False
    )
    normalized_batch = identity_fn(batch, crosscoder, normalize)
    return normalized_batch[:, 0, :] - remove_latents(
        reconstruction[:, 0, :],
        latent_activations[:, latent_indices],
        base_decoder[latent_indices],
    )


def load_ft_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    normalize: bool = False,
    **kwargs,
):
    assert isinstance(crosscoder, CrossCoder) or isinstance(
        crosscoder, BatchTopKCrossCoder
    ), "ft error requires a crosscoder"
    reconstruction = crosscoder.decode(
        latent_activations, denormalize_activations=False
    )
    normalized_batch = identity_fn(batch, crosscoder, normalize)
    return normalized_batch[:, 1, :] - remove_latents(
        reconstruction[:, 1, :], latent_activations[:, latent_indices], latent_vectors
    )


def load_base_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    assert isinstance(crosscoder, CrossCoder) or isinstance(
        crosscoder, BatchTopKCrossCoder
    ), "Base reconstruction requires a crosscoder"
    reconstruction = crosscoder.decode(
        latent_activations, denormalize_activations=False
    )
    return reconstruction[:, 0, :]


def load_ft_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    assert isinstance(crosscoder, CrossCoder) or isinstance(
        crosscoder, BatchTopKCrossCoder
    ), "ft reconstruction requires a crosscoder"
    reconstruction = crosscoder.decode(
        latent_activations, denormalize_activations=False
    )
    return reconstruction[:, 1, :]


def load_betas(
    betas_dir_path: Path,
    num_samples: int = 50_000_000,
    computation: str = "base_error",
):
    betas_filename = f"betas_{computation}_N{num_samples}.pt"

    if not (betas_dir_path / betas_filename).exists():
        raise FileNotFoundError(
            f"Betas file not found: {betas_dir_path / betas_filename}."
        )

    betas = th.load(betas_dir_path / betas_filename, weights_only=True).cpu()
    return betas


def betas_exist(
    betas_dir_path: Path,
    num_samples: int = 50_000_000,
    computation: str = "base_error",
):
    betas_filename = f"betas_{computation}_N{num_samples}.pt"
    return (betas_dir_path / betas_filename).exists()
