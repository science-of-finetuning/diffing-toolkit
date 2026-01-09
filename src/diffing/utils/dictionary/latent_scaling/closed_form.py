import torch as th
from typing import Callable, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig
from typing import Literal
import numpy as np
import os
import pandas as pd
from functools import partial
from dictionary_learning import BatchTopKCrossCoder, BatchTopKSAE, CrossCoder
from dictionary_learning.dictionary import Dictionary

from diffing.utils.dictionary.latent_scaling.utils import (
    identity_fn,
    load_base_activation,
    load_ft_activation,
    load_difference_activation,
    load_base_activation_no_bias,
    load_ft_activation_no_bias,
    load_base_error,
    load_ft_error,
    load_base_reconstruction,
    load_ft_reconstruction,
    betas_exist,
)

from diffing.utils.dictionary.training import (
    setup_training_datasets,
    skip_first_n_tokens,
)
from diffing.utils.dictionary.utils import load_dictionary_model, load_latent_df


@th.no_grad()
def closed_form_scalars(
    latent_vectors: th.Tensor,
    latent_indices: th.Tensor,
    dataloader: DataLoader,
    dict_model: Dictionary,
    target_activation_fn: Callable[[th.Tensor], th.Tensor],
    encode_activation_fn: Callable[[th.Tensor], th.Tensor] = identity_fn,
    latent_activation_postprocessing_fn: Callable[[th.Tensor], th.Tensor] = None,
    device: th.device = th.device("cuda"),
    dtype: Union[th.dtype, None] = th.float32,
) -> th.Tensor:
    """
    Compute the argmin_\beta || x - f\beta d ||^2 using the closed form solution.
    """
    # beta = (latent_vector.T @ (data.T @ latent_vector) / ((latent_vector.norm() ** 2) * (latent_activations.norm() ** 2))
    # data: N x dim_model
    # latent_vector: dim_model
    # latent_activations: N
    # beta = (latent_vector.T @ A) / (C * D)
    # beta = (B) / (C * D)
    # A = (data.T @ latent_activations) # (dim_model)
    # B = (latent_vector.T @ A) # (scalar)
    # C = (latent_activations.norm() ** 2) # (scalar)
    # D = (latent_vector.norm() ** 2) # (scalar)

    # We do this for num_latent_vectors latent vectors, so
    # latent_vectors: num_latent_vectors x dim_model
    # latent_activations: N x num_latent_vectors
    # A = (data.T @ latent_activations) # (dim_model, N) x (N, num_latent_vectors) -> (dim_model, num_latent_vectors)
    # B = diag(latent_vectors.T @ A) # (num_latent_vectors, dim_model) x (dim_model, num_latent_vectors) -> (num_latent_vectors, num_latent_vectors)
    # C = (latent_activations.norm() ** 2) # (num_latent_vectors)
    # D = (latent_vector.norm() ** 2) # (num_latent_vectors)
    # beta = (B) / (C * D) # (num_latent_vectors)

    # batched_data -> one data for each latent vector
    # data: num_latent_vectors x N x dim_model
    # only A is different then
    # A = th.matmul(data.transpose(1, 2), latent_activations.T.unsqueeze(-1))  (num_latent_vectors, dim_model, N) x (N, num_latent_vectors) -> (dim_model, num_latent_vectors)

    assert latent_vectors.ndim == 2  # (D, dim_model)
    latent_vectors = latent_vectors.to(device)

    dim_model = latent_vectors.size(1)
    num_latent_vectors = latent_vectors.size(0)
    dict_size = dict_model.dict_size
    logger.debug(
        f"dim_model: {dim_model}, num_latent_vectors: {num_latent_vectors}, dict_size: {dict_size}"
    )
    A = th.zeros(
        (dim_model, num_latent_vectors), device=device, dtype=dtype
    )  # data.T @ latent_activations
    C = th.zeros(
        num_latent_vectors, device=device, dtype=dtype
    )  # latent_activations.norm() ** 2
    D = th.zeros(
        num_latent_vectors, device=device, dtype=dtype
    )  # latent_vectors.norm() ** 2

    count_active = th.zeros(num_latent_vectors, device=device, dtype=dtype)

    for batch in tqdm(dataloader):
        if batch.shape[0] == 0:
            continue
        batch_size_current = batch.shape[0]
        batch = batch.to(device).to(dtype)
        latent_activations = dict_model.encode(
            encode_activation_fn(batch), use_threshold=True
        )
        if latent_activation_postprocessing_fn is not None:
            latent_activations = latent_activation_postprocessing_fn(latent_activations)

        assert latent_activations.shape == (batch_size_current, dict_size)

        Y_batch = target_activation_fn(
            batch,
            crosscoder=dict_model,
            latent_activations=latent_activations,
            latent_indices=latent_indices,
            latent_vectors=latent_vectors,
        )
        if len(Y_batch.shape) == 3:
            assert Y_batch.shape == (num_latent_vectors, batch_size_current, dim_model)
        else:
            assert Y_batch.shape == (batch_size_current, dim_model)
        latent_activations = latent_activations[:, latent_indices]
        assert latent_activations.shape == (
            batch_size_current,
            num_latent_vectors,
        ), f"Latent activation has weird shape: {latent_activations.shape}, expected {(batch_size_current, num_latent_vectors)}. Latent indices length: {len(latent_indices)}"

        non_zero_mask = (latent_activations != 0).sum(dim=0)
        assert non_zero_mask.shape == (num_latent_vectors,)
        count_active += non_zero_mask
        non_zero_mask = non_zero_mask > 0

        non_zero_elements = non_zero_mask.sum()

        if len(Y_batch.shape) == 3:
            # one data vector per latent vector
            # use batched logic
            A_update = (
                th.matmul(
                    Y_batch[non_zero_mask].transpose(1, 2),
                    latent_activations[:, non_zero_mask].T.unsqueeze(-1),
                )
                .squeeze(-1)
                .T
            )  # (num_latent_vectors, dim_model, N) x (N, num_latent_vectors) -> (num_latent_vectors, dim_model)
        else:
            A_update = Y_batch.T @ latent_activations[:, non_zero_mask]
        assert A_update.shape == (dim_model, non_zero_elements)
        A[:, non_zero_mask] += A_update

        C_update = (latent_activations[:, non_zero_mask] ** 2).sum(dim=0)
        assert C_update.shape == (non_zero_elements,)
        C[non_zero_mask] += C_update

    D = th.sum(latent_vectors**2, dim=1)
    assert D.shape == (num_latent_vectors,)

    B = th.sum(A.T * latent_vectors, dim=1)  # diag(latent_vector.T @ A)
    assert B.shape == (num_latent_vectors,)

    betas = B / (C * D)
    assert betas.shape == (num_latent_vectors,)

    return betas, count_active, B, C, D


def compute_scalers_from_config(
    cfg: DictConfig,
    layer: int,
    dictionary_model: str,
    results_dir: Path = Path("./results"),
) -> None:
    """
    Compute the scalers from the config, including error computation for effective chat-only
    and shared baseline latents.
    """
    ls_cfg = cfg.diffing.method.analysis.latent_scaling
    is_sae = cfg.diffing.method.name != "crosscoder"

    ft_error = "ft_error" in ls_cfg.targets
    ft_reconstruction = "ft_reconstruction" in ls_cfg.targets
    base_error = "base_error" in ls_cfg.targets
    base_reconstruction = "base_reconstruction" in ls_cfg.targets
    ft_activation = "ft_activation" in ls_cfg.targets
    base_activation = "base_activation" in ls_cfg.targets
    base_activation_no_bias = "base_activation_no_bias" in ls_cfg.targets
    ft_activation_no_bias = "ft_activation_no_bias" in ls_cfg.targets

    num_samples = ls_cfg.num_samples

    # Check if betas exist
    if not ls_cfg.overwrite:
        ft_error = ft_error and (
            not betas_exist(
                results_dir / "closed_form_scalars" / "effective_ft_only_latents",
                num_samples,
                "ft_error",
            )
            or not betas_exist(
                results_dir / "closed_form_scalars" / "shared_baseline_latents",
                num_samples,
                "ft_error",
            )
        )
        base_error = base_error and (
            not betas_exist(
                results_dir / "closed_form_scalars" / "effective_ft_only_latents",
                num_samples,
                "base_error",
            )
            or not betas_exist(
                results_dir / "closed_form_scalars" / "shared_baseline_latents",
                num_samples,
                "base_error",
            )
        )
        ft_reconstruction = ft_reconstruction and not betas_exist(
            results_dir / "closed_form_scalars" / "all_latents",
            num_samples,
            "ft_reconstruction",
        )
        base_reconstruction = base_reconstruction and not betas_exist(
            results_dir / "closed_form_scalars" / "all_latents",
            num_samples,
            "base_reconstruction",
        )
        ft_activation = ft_activation and not betas_exist(
            results_dir / "closed_form_scalars" / "all_latents",
            num_samples,
            "ft_activation",
        )
        base_activation = base_activation and not betas_exist(
            results_dir / "closed_form_scalars" / "all_latents",
            num_samples,
            "base_activation",
        )
        ft_activation_no_bias = ft_activation_no_bias and not betas_exist(
            results_dir / "closed_form_scalars" / "all_latents",
            num_samples,
            "ft_activation_no_bias",
        )
        base_activation_no_bias = base_activation_no_bias and not betas_exist(
            results_dir / "closed_form_scalars" / "all_latents",
            num_samples,
            "base_activation_no_bias",
        )

    # Log which scalers will be computed
    scalers_to_compute = []
    if ft_error:
        scalers_to_compute.append("ft_error")
    if base_error:
        scalers_to_compute.append("base_error")
    if ft_reconstruction:
        scalers_to_compute.append("ft_reconstruction")
    if base_reconstruction:
        scalers_to_compute.append("base_reconstruction")
    if ft_activation:
        scalers_to_compute.append("ft_activation")
    if base_activation:
        scalers_to_compute.append("base_activation")
    if ft_activation_no_bias:
        scalers_to_compute.append("ft_activation_no_bias")
    if base_activation_no_bias:
        scalers_to_compute.append("base_activation_no_bias")

    logger.info(f"Scalers to compute: {scalers_to_compute}")

    if len(scalers_to_compute) == 0:
        logger.info("No scalers to compute, exiting")
        return

    # Configuration for error computation on latent subsets
    num_effective_ft_only_latents = ls_cfg.num_effective_ft_only_latents

    # Setup paths
    # Load validation dataset
    train_dataset, val_dataset, _, _, _ = setup_training_datasets(
        cfg,
        layer,
        overwrite_num_samples=num_samples,
        overwrite_local_shuffling=False,
        dataset_processing_function=lambda x: skip_first_n_tokens(
            x, cfg.model.ignore_first_n_tokens_per_sample_during_training
        ),
    )
    if ls_cfg.dataset_split == "train":
        dataset = train_dataset
    elif ls_cfg.dataset_split == "validation":
        dataset = val_dataset
    else:
        raise ValueError(f"Invalid dataset split: {ls_cfg.dataset_split}")

    is_difference_sae = cfg.diffing.method.name == "sae_difference"
    if is_difference_sae:
        sae_model = (
            "base" if cfg.diffing.method.training.target == "difference_bft" else "ft"
        )
    else:
        sae_model = None
    # First compute regular scalars for all targets except errors
    if (
        base_reconstruction
        or ft_reconstruction
        or base_activation
        or ft_activation
        or base_activation_no_bias
        or ft_activation_no_bias
    ):
        compute_scalers(
            dataset=dataset,
            dictionary_model=dictionary_model,
            results_dir=results_dir,
            latent_indices_name="all_latents",
            num_samples=num_samples,
            ft_error=False,
            base_error=False,
            ft_reconstruction=ft_reconstruction,
            base_reconstruction=base_reconstruction,
            ft_activation=ft_activation,
            base_activation=base_activation,
            base_activation_no_bias=base_activation_no_bias,
            ft_activation_no_bias=ft_activation_no_bias,
            is_sae=is_sae,
            sae_model=sae_model,
            is_difference_sae=False,
            smaller_batch_size_for_error=True,
            batch_size=ls_cfg.batch_size,
            num_workers=ls_cfg.num_workers,
        )

    # Compute error scalars for effective chat-only latents and shared baseline latents
    if ft_error or base_error:
        logger.info("Loading latent dataframe for error computation on latent subsets")
        df = load_latent_df(dictionary_model)

        # Get effective chat-only latents by sorting by dec_norm_diff ascending and taking the head (lowest k)
        if num_effective_ft_only_latents == -1:
            effective_ft_only_latents_indices = df.query(
                "tag == 'ft_only'"
            ).index.tolist()
        else:
            effective_ft_only_latents_indices = (
                df.sort_values(by="dec_norm_diff", ascending=True)
                .head(num_effective_ft_only_latents)
                .index.tolist()
            )

        # Get shared baseline indices by sampling from "shared" tag latents
        shared_baseline_indices = (
            df[df["tag"] == "shared"]
            .sample(n=len(effective_ft_only_latents_indices), random_state=42)
            .index.tolist()
        )

        logger.info(
            f"Computing error scalars for {len(effective_ft_only_latents_indices)} effective chat-only latents"
        )
        compute_scalers(
            dataset=dataset,
            dictionary_model=dictionary_model,
            results_dir=results_dir,
            latent_indices=th.tensor(effective_ft_only_latents_indices),
            latent_indices_name="effective_ft_only_latents",
            num_samples=num_samples,
            ft_error=ft_error,
            base_error=base_error,
            is_sae=is_sae,
            sae_model=sae_model,
            is_difference_sae=is_difference_sae,
            smaller_batch_size_for_error=True,
        )
        effective_dir = (
            results_dir / "closed_form_scalars" / "effective_ft_only_latents"
        )
        effective_dir.mkdir(parents=True, exist_ok=True)
        th.save(
            effective_ft_only_latents_indices,
            effective_dir / "indices.pt",
        )
        logger.info(
            f"Computing error scalars for {len(shared_baseline_indices)} shared baseline latents"
        )
        # Compute scalers for shared baseline latents
        compute_scalers(
            dataset=dataset,
            dictionary_model=dictionary_model,
            results_dir=results_dir,
            latent_indices=th.tensor(shared_baseline_indices),
            latent_indices_name="shared_baseline_latents",
            num_samples=num_samples,
            ft_error=ft_error,
            is_sae=is_sae,
            sae_model=sae_model,
            is_difference_sae=is_difference_sae,
            base_error=base_error,
            smaller_batch_size_for_error=True,
        )
        shared_dir = results_dir / "closed_form_scalars" / "shared_baseline_latents"
        shared_dir.mkdir(parents=True, exist_ok=True)
        th.save(
            shared_baseline_indices,
            shared_dir / "indices.pt",
        )


def compute_scalers(
    dataset: "ActivationCache",
    dictionary_model: str,
    results_dir: Path = Path("./results"),
    latent_indices: th.Tensor | None = None,
    latent_indices_name: str = "all_latents",
    max_activations_path: Path | None = None,
    # Computation parameters
    batch_size: int = 128,
    num_samples: int = 50_000_000,
    num_workers: int = 32,
    device: str = "cuda",
    dtype: str = "float32",
    threshold_active_latents: float | None = None,
    # Output parameters
    name: str | None = None,
    # Computation flags
    ft_error: bool = False,
    ft_reconstruction: bool = False,
    base_error: bool = False,
    base_reconstruction: bool = False,
    ft_activation: bool = False,
    base_activation: bool = False,
    base_activation_no_bias: bool = False,
    ft_activation_no_bias: bool = False,
    random_vectors: bool = False,
    random_indices: bool = False,
    target_model_idx: int | None = None,
    is_sae: bool = False,
    is_difference_sae: bool = False,
    sae_model: Literal["base", "chat"] | None = None,
    smaller_batch_size_for_error: bool = False,
) -> None:
    """
    Compute closed-form scalars (betas) for latent scaling analysis.

    This function computes the optimal scaling factors beta for dictionary latents
    using the closed-form solution to argmin_β ||x - βfd||², where:
    - x is the target activation
    - f is the latent activation
    - d is the decoder vector

    Args:
        dataset: ActivationCache dataset containing model activations
        dictionary_model: Path to the dictionary model to analyze
        results_dir: Directory to save results (default: "./results")
        latent_indices: Specific latent indices to analyze (default: None = all latents)
        latent_indices_name: Name for the latent indices subset (default: "all_latents")
        max_activations_path: Path to max activations file for thresholding (default: None)

        # Computation parameters
        batch_size: Batch size for processing (default: 128)
        num_samples: Number of samples to process (default: 50_000_000)
        num_workers: Number of data loader workers (default: 32)
        device: Device to use for computation (default: "cuda")
        dtype: Data type for computation (default: "float32")
        threshold_active_latents: Threshold for active latents filtering (default: None)

        # Output parameters
        name: Optional suffix for output filenames (default: None)

        # Computation flags - specify which targets to compute
        ft_error: Compute betas for fine-tuned error
        ft_reconstruction: Compute betas for fine-tuned reconstruction
        base_error: Compute betas for base model error
        base_reconstruction: Compute betas for base model reconstruction
        ft_activation: Compute betas for fine-tuned activation
        base_activation: Compute betas for base model activation
        base_activation_no_bias: Compute betas for base activation without bias
        ft_activation_no_bias: Compute betas for fine-tuned activation without bias
        random_vectors: Use random decoder vectors instead of trained ones
        random_indices: Use random latent indices instead of specified ones

        # Model configuration
        target_model_idx: Index of target model for multi-model dictionaries (default: None)
        is_sae: Whether the dictionary is a Sparse Autoencoder (default: False)
        is_difference_sae: Whether the dictionary is a difference SAE (default: False)
        sae_model: Which model to use for SAE ("base" or "chat") (default: None)
        smaller_batch_size_for_error: Use 8x smaller batch size for error computations to reduce memory usage (default: False)

    Returns:
        None (saves results to disk)

    Saves:
        - betas_{exp_name}.pt: Computed scaling factors for each experiment
        - latent_vectors_{exp_name}.pt: Latent vectors used (if random_vectors or random_indices)
        - random_indices_{exp_name}.pt: Random indices used (if random_indices)
    """
    is_sae = is_sae or is_difference_sae
    if is_sae and sae_model is None:
        raise ValueError(
            "sae_model must be provided if is_sae is True. This is the model to use for the SAE."
        )
    if latent_indices is not None and latent_indices_name == "all_latents":
        latent_indices_name = f"custom_indices_{len(latent_indices)}"

    results_dir = results_dir / "closed_form_scalars"

    # Setup dtype
    dtype_map = {"float32": th.float32, "float64": th.float64, "bfloat16": th.bfloat16}
    dtype = dtype_map[dtype]
    logger.info(f"Using dtype: {dtype}")

    if threshold_active_latents is not None:
        assert (
            threshold_active_latents > 0 and threshold_active_latents < 1
        ), "Threshold must be between 0 and 1"

    # Load dictionary model
    logger.info(f"Loading dictionary model from {dictionary_model}")
    dict_model = load_dictionary_model(dictionary_model, is_sae=is_sae)
    dict_model = dict_model.to(device).to(dtype)

    # Get decoder weights
    if isinstance(dict_model, CrossCoder):
        ft_decoder = dict_model.decoder.weight[1, :, :].clone().to(dtype)
        assert ft_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)
        base_decoder = dict_model.decoder.weight[0, :, :].clone().to(dtype)
        assert base_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)
    else:
        ft_decoder = dict_model.decoder.weight.clone().to(dtype).T
        assert ft_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)
        base_decoder = dict_model.decoder.weight.clone().to(dtype).T
        assert base_decoder.shape == (dict_model.dict_size, dict_model.activation_dim)

    logger.info(f"Number of activations: {len(dataset)}")

    if latent_indices is None:
        print("No latent indices provided, using all latents")
        latent_indices = th.arange(dict_model.dict_size)

    latent_activation_postprocessing_fn = None
    if threshold_active_latents is not None:
        max_act_path = max_activations_path
        if not os.path.exists(max_act_path):
            raise ValueError(
                f"Provided max activations path {max_act_path} does not exist"
            )
        max_activations = th.load(max_act_path).to(device)
        assert max_activations.shape == (dict_model.dict_size,)

        threshold = max_activations * threshold_active_latents

        def jumprelu_latent_activations(latent_activations):
            # latent_activations: (batch_size, dict_size)
            # Set latent activations to 0 if their value lies below x% of the max act.
            latent_activations = latent_activations.masked_fill(
                latent_activations < threshold, 0
            )
            return latent_activations

        latent_activation_postprocessing_fn = jumprelu_latent_activations

    if isinstance(dict_model, BatchTopKCrossCoder) and dict_model.decoupled_code:
        if target_model_idx is None:
            raise ValueError(
                "target_model_idx must be provided if using a decoupled code This is needed to specify which code to use for computing betas"
            )
        if latent_activation_postprocessing_fn is None:

            def latent_activation_postprocessing_fn(x):
                return x[:, target_model_idx]

        else:
            prev_postprocessing_fn = latent_activation_postprocessing_fn

            def latent_activation_postprocessing_fn(x):
                return prev_postprocessing_fn(x[:, target_model_idx])

    # Create results directory
    results_dir = results_dir / latent_indices_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving results to ", results_dir)
    encode_activation_fn = partial(
        identity_fn, normalize=False
    )  # The dict_model normalizes automatically in .encode()
    if isinstance(dict_model, BatchTopKSAE):
        # Deal with BatchTopKSAE
        if is_difference_sae:
            logger.debug(
                "BatchTopKSAE on difference detected, using load_difference_activation as encode_activation_fn"
            )
            encode_activation_fn = partial(
                load_difference_activation, sae_model=sae_model, normalize=False
            )
        else:
            logger.debug(
                "BatchTopKSAE on ft detected, using load_ft_activation as encode_activation_fn"
            )
            encode_activation_fn = partial(load_ft_activation, normalize=False)

    computations = []
    if base_activation:
        computations.append(
            ("base_activation", partial(load_base_activation, normalize=True))
        )
    if ft_activation:
        computations.append(
            ("ft_activation", partial(load_ft_activation, normalize=True))
        )
    if base_reconstruction:
        computations.append(
            ("base_reconstruction", partial(load_base_reconstruction, normalize=True))
        )
    if base_error:
        assert isinstance(
            dict_model, CrossCoder
        ), "Base error only supported for CrossCoder"
        computations.append(
            (
                "base_error",
                partial(load_base_error, base_decoder=base_decoder, normalize=True),
            )
        )
    if base_activation_no_bias:
        computations.append(
            (
                "base_activation_no_bias",
                partial(load_base_activation_no_bias, normalize=True),
            )
        )
    if ft_activation_no_bias:
        computations.append(
            (
                "ft_activation_no_bias",
                partial(load_ft_activation_no_bias, normalize=True),
            )
        )
    if ft_reconstruction:
        computations.append(
            ("ft_reconstruction", partial(load_ft_reconstruction, normalize=True))
        )
    if ft_error:
        computations.append(("ft_error", partial(load_ft_error, normalize=True)))

    latent_vectors = ft_decoder[latent_indices].clone()
    if random_vectors:
        random_vectors = th.randn(
            len(latent_indices), dict_model.activation_dim, device=device
        )
        assert random_vectors.shape == (len(latent_indices), dict_model.activation_dim)
        # Scale random vectors to match the norm of the IT decoder vectors
        it_decoder_norm = th.norm(latent_vectors, dim=1)
        assert it_decoder_norm.shape == (len(latent_indices),)
        random_vectors = random_vectors * (
            it_decoder_norm / th.norm(random_vectors, dim=1)
        ).unsqueeze(1)
        assert random_vectors.shape == (len(latent_indices), dict_model.activation_dim)
        latent_vectors = random_vectors

    if random_indices:
        random_indices = th.randint(
            0, dict_model.dict_size, (len(latent_indices),), device=device
        )
        assert random_indices.shape == (len(latent_indices),)
        # Scale random indices to match the norm of the IT decoder vectors
        random_indices_vectors = ft_decoder[random_indices].clone()
        assert random_indices_vectors.shape == (
            len(latent_indices),
            dict_model.activation_dim,
        )
        # Scale random vectors to match the norm of the IT decoder vectors
        it_decoder_norm = th.norm(latent_vectors, dim=1)
        assert it_decoder_norm.shape == (len(latent_indices),)
        random_indices_vectors = random_indices_vectors * (
            it_decoder_norm / th.norm(random_indices_vectors, dim=1)
        ).unsqueeze(1)
        assert random_indices_vectors.shape == (
            len(latent_indices),
            dict_model.activation_dim,
        )
        latent_vectors = random_indices_vectors

    # Run all computations
    for exp_name, loader_fn in computations:
        exp_name += f"_N{num_samples}"
        if threshold_active_latents is not None:
            exp_name += f"_jumprelu{threshold_active_latents}"
        if random_vectors:
            exp_name += "_random_vectors"
        if random_indices:
            exp_name += "_random_indices"
        if name:
            exp_name += f"_{name}"
        logger.info(f"Computing {exp_name}")
        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=(
                batch_size
                if not smaller_batch_size_for_error and "error" not in exp_name
                else batch_size // 8
            ),
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        betas, count_active, nominator, norm_f, norm_d = closed_form_scalars(
            latent_vectors,
            latent_indices,
            dataloader,
            dict_model,
            loader_fn,
            device=device,
            dtype=dtype,
            encode_activation_fn=encode_activation_fn,
            latent_activation_postprocessing_fn=latent_activation_postprocessing_fn,
        )
        th.save(betas.cpu(), results_dir / f"betas_{exp_name}.pt")

        if random_indices or random_vectors:
            th.save(latent_vectors.cpu(), results_dir / f"latent_vectors_{exp_name}.pt")
        if random_indices:
            th.save(random_indices.cpu(), results_dir / f"random_indices_{exp_name}.pt")
