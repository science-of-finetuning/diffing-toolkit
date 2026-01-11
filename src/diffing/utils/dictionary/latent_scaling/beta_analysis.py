import time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from argparse import ArgumentParser
import scipy.stats
from diffing.utils.dictionary.utils import load_latent_df, push_latent_df
from diffing.utils.dictionary.latent_scaling.utils import load_betas
from loguru import logger


def update_latent_df_with_beta_values(
    crosscoder: str,
    results_dir: Path,
    num_samples: int,
):
    betas_dir = results_dir / "closed_form_scalars"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "normal": {
            model: {
                target: f"{model}_{target}"
                for target in [
                    "error",
                    "reconstruction",
                    "activation",
                    "activation_no_bias",
                ]
            }
            for model in ["base", "ft"]
        },
    }

    df = load_latent_df(crosscoder)
    all_betas = load_betas_results(
        betas_dir / "all_latents", configs, num_samples=num_samples
    )
    df = add_possible_cols(df, df.index.tolist(), all_betas)

    if Path(betas_dir / "effective_ft_only_latents" / "indices.pt").exists():
        ft_specific_indices = th.load(
            betas_dir / "effective_ft_only_latents" / "indices.pt"
        )
        if isinstance(ft_specific_indices, th.Tensor):
            ft_specific_indices = ft_specific_indices.tolist()
        ft_error_betas = load_betas_results(
            betas_dir / "effective_ft_only_latents",
            configs,
            num_samples=num_samples,
        )

        df = add_possible_cols(df, ft_specific_indices, ft_error_betas)
        df["effective_ft_only_latent"] = False
        df.loc[ft_specific_indices, "effective_ft_only_latent"] = True
    if Path(betas_dir / "shared_baseline_latents" / "indices.pt").exists():
        shared_indices = th.load(betas_dir / "shared_baseline_latents" / "indices.pt")
        if isinstance(shared_indices, th.Tensor):
            shared_indices = shared_indices.tolist()
        shared_error_betas = load_betas_results(
            betas_dir / "shared_baseline_latents",
            configs,
            num_samples=num_samples,
        )
        df = add_possible_cols(df, shared_indices, shared_error_betas)
        df["shared_baseline_latent"] = False
        df.loc[shared_indices, "shared_baseline_latent"] = True
    push_latent_df(df, crosscoder, confirm=False)
    return df


def load_betas_results(
    base_path, configs, not_dead_mask=None, to_numpy=True, num_samples=None
):
    """
    Load beta values from result files.

    Args:
        base_path: Path to the base directory containing results
        configs: Dictionary configuration specifying which results to load
        not_dead_mask: Optional mask for filtering results
        to_numpy: Convert tensors to numpy arrays if True

    Returns:
        Tuple of (betas_out) dictionaries
    """
    if num_samples is None:
        num_samples = 50_000_000
    betas_out = {
        config: {
            model: {target: None for target in configs[config][model]}
            for model in configs[config]
        }
        for config in configs
    }
    for config in configs:
        for model in configs[config]:
            for target in configs[config][model]:
                try:
                    betas = load_betas(
                        base_path,
                        computation=configs[config][model][target],
                        num_samples=num_samples,
                    )
                except FileNotFoundError as e:
                    logger.debug(f"File not found: {e}. Skipping.")
                    continue

                betas = betas.cpu()
                if to_numpy:
                    betas = betas.numpy()
                if not_dead_mask is not None:
                    betas = betas[not_dead_mask]
                betas_out[config][model][target] = betas
    return betas_out


def add_col_to_df(df, indices, col, values):
    """Add column to dataframe with values at specified indices"""
    if col not in df.columns:
        df[col] = np.nan
    df.loc[indices, col] = values
    return df


def add_possible_cols(df, indices, betas):
    """Add beta columns to dataframe if they exist in the results"""
    if (
        betas["normal"]["base"]["error"] is not None
        and betas["normal"]["ft"]["error"] is not None
    ):
        print("Adding beta_error_base and beta_error_ft")
        df = add_col_to_df(
            df, indices, "beta_error_base", betas["normal"]["base"]["error"]
        )
        df = add_col_to_df(df, indices, "beta_error_ft", betas["normal"]["ft"]["error"])
        df["beta_ratio_error"] = df["beta_error_base"] / df["beta_error_ft"]

    if (
        betas["normal"]["base"]["reconstruction"] is not None
        and betas["normal"]["ft"]["reconstruction"] is not None
    ):
        df = add_col_to_df(
            df,
            indices,
            "beta_reconstruction_base",
            betas["normal"]["base"]["reconstruction"],
        )
        df = add_col_to_df(
            df,
            indices,
            "beta_reconstruction_ft",
            betas["normal"]["ft"]["reconstruction"],
        )
        df["beta_ratio_reconstruction"] = (
            df["beta_reconstruction_base"] / df["beta_reconstruction_ft"]
        )

    if (
        betas["normal"]["base"]["activation"] is not None
        and betas["normal"]["ft"]["activation"] is not None
    ):
        df = add_col_to_df(
            df, indices, "beta_activation_base", betas["normal"]["base"]["activation"]
        )
        df = add_col_to_df(
            df, indices, "beta_activation_ft", betas["normal"]["ft"]["activation"]
        )
        df["beta_activation_ratio"] = (
            df["beta_activation_base"] / df["beta_activation_ft"]
        )

    if (
        betas["normal"]["base"]["activation_no_bias"] is not None
        and betas["normal"]["ft"]["activation_no_bias"] is not None
    ):
        df = add_col_to_df(
            df,
            indices,
            "beta_activation_no_bias_base",
            betas["normal"]["base"]["activation_no_bias"],
        )
        df = add_col_to_df(
            df,
            indices,
            "beta_activation_no_bias_ft",
            betas["normal"]["ft"]["activation_no_bias"],
        )
        df["beta_activation_no_bias_ratio"] = (
            df["beta_activation_no_bias_base"] / df["beta_activation_no_bias_ft"]
        )

    return df
