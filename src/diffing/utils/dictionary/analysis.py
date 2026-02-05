"""
Analysis pipeline wrapper for crosscoder evaluation and analysis.

This module provides a wrapper around the comprehensive analysis pipeline from
science-of-finetuning, including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from loguru import logger
from transformers import AutoTokenizer
import torch as th
import pandas as pd
from torch.nn.functional import cosine_similarity
from tqdm.auto import trange
import numpy as np

from diffing.utils.dictionary import load_dictionary_model
from diffing.utils.dictionary.utils import push_latent_df, load_latent_df

# Common matplotlib settings for publication-quality plots
COMMON_RC_PARAMS = {"text.usetex": True}
SMALL_FONT_RC = {**COMMON_RC_PARAMS, "font.size": 16}
MEDIUM_FONT_RC = {**COMMON_RC_PARAMS, "font.size": 20}
LARGE_FONT_RC = {**COMMON_RC_PARAMS, "font.size": 24}


def build_push_crosscoder_latent_df(
    dictionary_name: str,
    base_layer: int = 0,
    ft_layer: int = 1,
    model_path: str | Path | None = None,
    push_to_hub: bool = True,
) -> pd.DataFrame:
    """Build and push/save latent dataframe for crosscoder models.

    Args:
        dictionary_name: Name of the crosscoder model (used for hub operations)
        base_layer: Index of the base layer in the crosscoder
        ft_layer: Index of the finetuned layer in the crosscoder
        model_path: Local path to load model from instead of hub
        push_to_hub: If True, push latent_df to HF Hub; if False, save locally
    """
    crosscoder = load_dictionary_model(model_path or dictionary_name)
    try:
        existing_latent_df = load_latent_df(dictionary_name)
        logger.info(
            f"Found existing latent dataframe for {dictionary_name} with {len(existing_latent_df)} latents"
        )
        latent_df = existing_latent_df.T.to_dict()
    except Exception:
        latent_df = {k: {} for k in range(crosscoder.dict_size)}

    # Norms
    norms = crosscoder.decoder.weight.norm(dim=-1)
    norm_diffs = (
        (norms[base_layer] - norms[ft_layer]) / norms.max(dim=0).values + 1
    ) / 2
    norm_diffs = norm_diffs.cpu()
    for f_idx, (base_norm, instruct_norm, norm_diff) in enumerate(
        zip(norms[base_layer], norms[ft_layer], norm_diffs)
    ):
        latent_df[f_idx]["dec_base_norm"] = base_norm.item()
        latent_df[f_idx]["dec_ft_norm"] = instruct_norm.item()
        latent_df[f_idx]["dec_norm_diff"] = norm_diff.item()

    enc_norms = crosscoder.encoder.weight.norm(dim=1)
    enc_norm_diffs = (
        (enc_norms[base_layer] - enc_norms[ft_layer]) / enc_norms.max(dim=0).values + 1
    ) / 2
    enc_norm_diffs = enc_norm_diffs.cpu()

    for f_idx, (base_norm, instruct_norm, norm_diff) in enumerate(
        zip(enc_norms[base_layer], enc_norms[ft_layer], enc_norm_diffs)
    ):
        latent_df[f_idx]["enc_base_norm"] = base_norm.item()
        latent_df[f_idx]["enc_ft_norm"] = instruct_norm.item()
        latent_df[f_idx]["enc_norm_diff"] = norm_diff.item()

    decoder_cos_sims = cosine_similarity(
        crosscoder.decoder.weight[base_layer],  # [num_features, activation_dim]
        crosscoder.decoder.weight[ft_layer],  # [num_features, activation_dim]
        dim=1,
    )
    for f_idx, cos_sim in enumerate(decoder_cos_sims):
        latent_df[f_idx]["dec_cos_sim"] = cos_sim.item()

    # Encoder cos sims
    enc_cos_sims = cosine_similarity(
        crosscoder.encoder.weight[base_layer],  # [activation_dim, num_features]
        crosscoder.encoder.weight[ft_layer],  # [activation_dim, num_features]
        dim=0,
    )
    for f_idx, cos_sim in enumerate(enc_cos_sims):
        latent_df[f_idx]["enc_cos_sim"] = cos_sim.item()

    # Create masks for each category
    # Decoder
    # save ft only and base only feature index
    treshold = 0.1
    only_it_feature_indices = th.nonzero(norm_diffs < treshold, as_tuple=True)[0]
    only_base_feature_indices = th.nonzero(norm_diffs > 1 - treshold, as_tuple=True)[0]
    shared_feature_indices = th.nonzero(
        (norm_diffs - 0.5).abs() < treshold, as_tuple=True
    )[0]
    is_other_feature = th.ones_like(norm_diffs, dtype=bool)
    is_other_feature[only_it_feature_indices] = False
    is_other_feature[only_base_feature_indices] = False
    is_other_feature[shared_feature_indices] = False

    for f_idx in only_it_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "ft_only"
    for f_idx in only_base_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "base_only"
    for f_idx in shared_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "shared"
    for f_idx in is_other_feature.nonzero(as_tuple=True)[0].tolist():
        latent_df[f_idx]["tag"] = "other"

    latent_df = pd.DataFrame(latent_df).T
    logger.info(f"Created latent dataframe with {len(latent_df)} latents")
    if push_to_hub:
        push_latent_df(
            latent_df, dictionary_name, confirm=False, create_repo_if_missing=True
        )
    else:
        save_path = Path(model_path) / "latent_df.csv"
        logger.info(f"Saving latent dataframe locally to {save_path}")
        latent_df.to_csv(save_path)
    return latent_df


def build_push_sae_difference_latent_df(
    dictionary_name: str,
    target: str,
    model_path: str | Path | None = None,
    push_to_hub: bool = True,
) -> pd.DataFrame:
    """Build latent dataframe for SAE difference models.

    Args:
        dictionary_name: Name of the SAE model (used for hub operations)
        target: Training target ("difference_bft" or "difference_ftb")
        model_path: Local path to load model from instead of hub
        push_to_hub: If True, push latent_df to HF Hub; if False, save locally
    """
    logger.info(
        f"Building latent dataframe for SAE difference model: {dictionary_name}"
    )

    sae = load_dictionary_model(model_path or dictionary_name)
    try:
        existing_latent_df = load_latent_df(dictionary_name)
        logger.info(
            f"Found existing latent dataframe for {dictionary_name} with {len(existing_latent_df)} latents"
        )
        latent_df = existing_latent_df.T.to_dict()
    except Exception:
        latent_df = {k: {} for k in range(sae.dict_size)}

    # Decoder norms
    decoder_norms = sae.decoder.weight.norm(dim=0)  # [num_features]
    assert decoder_norms.shape == (sae.dict_size,)

    # Encoder norms
    encoder_norms = sae.encoder.weight.norm(dim=1)  # [num_features]
    assert encoder_norms.shape == (sae.dict_size,)

    for f_idx, norm in enumerate(decoder_norms):
        latent_df[f_idx]["dec_norm"] = norm.item()
    for f_idx, norm in enumerate(encoder_norms):
        latent_df[f_idx]["enc_norm"] = norm.item()

    # Convert to DataFrame
    latent_df = pd.DataFrame(latent_df).T

    logger.info(f"Created latent dataframe with {len(latent_df)} latents")
    if push_to_hub:
        push_latent_df(
            latent_df, dictionary_name, confirm=False, create_repo_if_missing=True
        )
    else:
        save_path = Path(model_path) / "latent_df.csv"
        logger.info(f"Saving latent dataframe locally to {save_path}")
        latent_df.to_csv(save_path)
    return latent_df


def make_plots(
    dictionary_name: str,
    plots_dir: Path,
):
    df = load_latent_df(dictionary_name)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if (
        "effective_ft_only_latent" in df.columns
        and "shared_baseline_latent" in df.columns
    ):
        target_df = df[df["effective_ft_only_latent"]]
        baseline_df = df[df["shared_baseline_latent"]]
        plot_error_vs_reconstruction(
            target_df, baseline_df, plots_dir, variant="standard"
        )
        plot_error_vs_reconstruction(
            target_df, baseline_df, plots_dir, variant="custom_color"
        )
        plot_error_vs_reconstruction(
            target_df, baseline_df, plots_dir, variant="poster"
        )

        plot_ratio_histogram(target_df, baseline_df, plots_dir, ratio_type="error")
        plot_ratio_histogram(
            target_df, baseline_df, plots_dir, ratio_type="reconstruction"
        )

        plot_beta_distribution_histograms(target_df, plots_dir)
        plot_correlation_with_frequency(df, plots_dir)
        plot_rank_distributions(target_df, plots_dir)

    if "enc_norm" in df.columns:
        plot_histogram(
            df,
            "enc_norm",
            plots_dir,
            title="Encoder Norm",
            xlabel="Norm",
            log_scale=True,
        )

    if "dec_norm" in df.columns:
        plot_histogram(
            df,
            "dec_norm",
            plots_dir,
            title="Decoder Norm",
            xlabel="Norm",
            log_scale=True,
        )

    if "max_act_validation" in df.columns:
        plot_histogram(
            df,
            "max_act_validation",
            plots_dir,
            title="Max Activation Validation",
            xlabel="Max Activation Validation",
            log_scale=True,
        )

    if "max_act_train" in df.columns:
        plot_histogram(
            df,
            "max_act_train",
            plots_dir,
            title="Max Activation Train",
            xlabel="Max Activation Train",
            log_scale=True,
        )

    if "beta_ratio_activation" in df.columns:
        plot_histogram(
            df,
            "beta_ratio_activation",
            plots_dir,
            title="Beta Ratio Activation",
            xlabel="Beta Ratio Activation",
            log_scale=True,
        )

    if "beta_ratio_activation_no_bias" in df.columns:
        plot_histogram(
            df,
            "beta_ratio_activation_no_bias",
            plots_dir,
            title="Beta Ratio Activation No Bias",
            xlabel="Beta Ratio Activation No Bias",
            log_scale=True,
        )

    if "freq_validation" in df.columns:
        plot_histogram(
            df,
            "freq_validation",
            plots_dir,
            title="Frequency Validation",
            xlabel="Frequency Validation",
            log_scale=True,
        )


def plot_histogram(
    df, column, plots_dir, title=None, xlabel=None, filename=None, log_scale=False
):
    """Plot histogram of a column from dataframe.

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        plots_dir: Directory to save plots
        title: Plot title (defaults to column name)
        xlabel: X-axis label (defaults to column name)
        filename: Output filename (defaults to column name)
        log_scale: Whether to use log scale on y-axis
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return

    with plt.rc_context(LARGE_FONT_RC):
        plt.figure(figsize=(6, 4))
        plt.hist(df[column], bins=100, alpha=0.7, color="blue")
        plt.xlabel(xlabel or column.replace("_", " ").title())
        plt.ylabel("Count")
        plt.title(title or column.replace("_", " ").title())
        if log_scale:
            plt.yscale("log")
        plt.tight_layout()
        output_filename = filename or f"{column}.pdf"
        plt.savefig(plots_dir / output_filename, bbox_inches="tight")
        plt.close()


def plot_beta_ratios_template_perc(target_df, filtered_df, plots_dir):
    """Plot histograms of beta ratios for template percentage.

    Args:
        target_df: DataFrame containing all ft-only latents
        filtered_df: DataFrame containing latents with high template percentage
        plots_dir: Directory to save plots
    """
    if not (
        "lmsys_ctrl_%" in target_df.columns
        and "beta_ratio_error" in target_df.columns
        and "beta_ratio_reconstruction" in target_df.columns
    ):
        return

    low, high = -0.1, 1.1

    with plt.rc_context(LARGE_FONT_RC):
        plt.figure(figsize=(8, 4))

        # First subplot for beta_ratio_error
        ax1 = plt.subplot(1, 2, 1)
        # Plot full distribution
        target_df["beta_ratio_error"].hist(
            bins=50,
            range=(low, high),
            alpha=0.5,
            color="gray",
            label="All ft-only latents",
        )
        # Plot filtered distribution on top
        filtered_df["beta_ratio_error"].hist(
            bins=50,
            range=(low, high),
            alpha=0.7,
            color="blue",
            label="High Template Perc.",
        )
        plt.xlabel("$\\nu^\\epsilon$")
        ax1.tick_params(axis="y", rotation=90)
        plt.ylabel("Count")

        # Second subplot for beta_ratio_reconstruction
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        # Plot full distribution
        target_df["beta_ratio_reconstruction"].hist(
            bins=50, range=(low, high), alpha=0.5, color="gray"
        )
        # Plot filtered distribution on top
        filtered_df["beta_ratio_reconstruction"].hist(
            bins=50, range=(low, high), alpha=0.7, color="blue"
        )
        plt.xlabel("$\\nu^r$")
        plt.setp(ax2.get_yticklabels(), visible=False)
        # Single legend for both plots
        plt.figlegend(fontsize=18.5, loc="center right", bbox_to_anchor=(0.532, 0.8))

        plt.tight_layout()
        plt.savefig(plots_dir / "beta_ratios_template_perc.pdf", bbox_inches="tight")
        plt.close()


def plot_error_vs_reconstruction(target_df, baseline_df, plots_dir, variant="standard"):
    """Plot scatter plot of error vs reconstruction ratios."""
    if not (
        "beta_ratio_error" in target_df.columns
        and "beta_ratio_reconstruction" in target_df.columns
    ):
        return

    zoom = [0, 1.1] if variant != "zoomed" else [-0.1, 1.1]
    ft_only_color = (0, 0.6, 1) if variant == "custom_color" else None

    # Create figure with a main plot and two side histograms
    fig_size = (
        (6, 3.5)
        if variant == "standard"
        else (4, 3) if variant == "custom_color" else (8, 3)
    )

    with plt.rc_context(MEDIUM_FONT_RC):
        fig = plt.figure(figsize=fig_size)

        # Create a grid of subplots
        gs = plt.GridSpec(
            2,
            2,
            width_ratios=[3, 1.3],
            height_ratios=[1, 3],
            left=0.1,
            right=0.85,
            bottom=0.1,
            top=0.9,
            wspace=0.03,
            hspace=0.05,
        )

        # Create the three axes
        ax_scatter = fig.add_subplot(gs[1, 0])  # Main plot
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)  # x-axis histogram
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)  # y-axis histogram

        # Filter out nans and apply zoom
        error_ratio = target_df["beta_ratio_error"]
        reconstruction_ratio = target_df["beta_ratio_reconstruction"]
        valid_mask = ~(np.isnan(error_ratio) | np.isnan(reconstruction_ratio))
        error_ratio_valid = error_ratio[valid_mask]
        reconstruction_ratio_valid = reconstruction_ratio[valid_mask]

        error_ratio_shared = baseline_df["beta_ratio_error"]
        reconstruction_ratio_shared = baseline_df["beta_ratio_reconstruction"]
        valid_mask_shared = ~(
            np.isnan(error_ratio_shared) | np.isnan(reconstruction_ratio_shared)
        )
        error_ratio_shared_valid = error_ratio_shared[valid_mask_shared]
        reconstruction_ratio_shared_valid = reconstruction_ratio_shared[
            valid_mask_shared
        ]

        # Apply zoom mask to both datasets
        zoom_mask = (
            (error_ratio_valid > zoom[0])
            & (error_ratio_valid < zoom[1])
            & (reconstruction_ratio_valid > zoom[0])
            & (reconstruction_ratio_valid < zoom[1])
        )
        error_ratio_zoomed = error_ratio_valid[zoom_mask]
        reconstruction_ratio_zoomed = reconstruction_ratio_valid[zoom_mask]

        zoom_mask_shared = (
            (error_ratio_shared_valid > zoom[0])
            & (error_ratio_shared_valid < zoom[1])
            & (reconstruction_ratio_shared_valid > zoom[0])
            & (reconstruction_ratio_shared_valid < zoom[1])
        )
        error_ratio_shared_zoomed = error_ratio_shared_valid[zoom_mask_shared]
        reconstruction_ratio_shared_zoomed = reconstruction_ratio_shared_valid[
            zoom_mask_shared
        ]

        # Plot the scatter plots
        scatter_kwargs = {"alpha": 0.2, "s": 5}
        if ft_only_color:
            ax_scatter.scatter(
                error_ratio_zoomed,
                reconstruction_ratio_zoomed,
                label="ft-only",
                color=ft_only_color,
                **scatter_kwargs,
            )
            ax_scatter.scatter(
                error_ratio_shared_zoomed,
                reconstruction_ratio_shared_zoomed,
                label="shared",
                color="C1",
                **scatter_kwargs,
            )
        else:
            ax_scatter.scatter(
                error_ratio_zoomed,
                reconstruction_ratio_zoomed,
                label="ft-only",
                **scatter_kwargs,
            )
            ax_scatter.scatter(
                error_ratio_shared_zoomed,
                reconstruction_ratio_shared_zoomed,
                label="Shared",
                **scatter_kwargs,
            )

        # Plot the histograms
        bins = 50
        hist_kwargs = {"bins": bins, "range": zoom, "alpha": 0.5}

        if ft_only_color:
            ax_histx.hist(
                error_ratio_zoomed, label="ft-only", color=ft_only_color, **hist_kwargs
            )
            ax_histx.hist(
                error_ratio_shared_zoomed, label="shared", color="C1", **hist_kwargs
            )
            ax_histy.hist(
                reconstruction_ratio_zoomed,
                orientation="horizontal",
                color=ft_only_color,
                **hist_kwargs,
            )
            ax_histy.hist(
                reconstruction_ratio_shared_zoomed,
                orientation="horizontal",
                color="C1",
                **hist_kwargs,
            )
        else:
            ax_histx.hist(error_ratio_zoomed, label="ft-only", **hist_kwargs)
            ax_histx.hist(error_ratio_shared_zoomed, label="Shared", **hist_kwargs)
            ax_histy.hist(
                reconstruction_ratio_zoomed, orientation="horizontal", **hist_kwargs
            )
            ax_histy.hist(
                reconstruction_ratio_shared_zoomed,
                orientation="horizontal",
                **hist_kwargs,
            )

        # Add grid to histograms
        ax_histx.grid(True, alpha=0.15)
        ax_histy.grid(True, alpha=0.15)
        ax_scatter.grid(True, alpha=0.15)

        # Turn off tick labels on histograms
        ax_histx.tick_params(labelbottom=False, bottom=False)
        ax_histy.tick_params(labelleft=False, left=False)

        # Add labels
        if variant == "poster":
            ax_scatter.set_ylabel(
                "$\\uparrow$ \n more \n Latent \n Decoupling ",
                labelpad=40,
                rotation=0,
                y=0.2,
            )
            ax_scatter.set_xlabel("more Complete Shrinkage $\\rightarrow$", labelpad=10)
        else:
            ax_scatter.set_xlabel("$\\nu^\\epsilon$")
            ax_scatter.set_ylabel("$\\nu^r$")

        # Add legend
        if variant == "custom_color":
            ax_histx.legend(
                fontsize=16,
                loc="upper right",
                handletextpad=0.2,
                bbox_to_anchor=(1.65, 1.2),
                handlelength=0.7,
                frameon=False,
            )
        else:
            ax_histx.legend(
                fontsize=16,
                markerscale=4,
                loc="lower right",
                bbox_to_anchor=(1.01, -3.2),
            )

        # Save figure
        suffix = (
            "_43"
            if variant == "custom_color"
            else "_poster" if variant == "poster" else ""
        )
        plt.savefig(
            plots_dir / f"error_vs_reconstruction_ratio_with_baseline{suffix}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_ratio_histogram(target_df, baseline_df, plots_dir, ratio_type="error"):
    """Plot histogram of beta ratio values for error or reconstruction."""
    if f"beta_ratio_{ratio_type}" not in target_df.columns:
        return

    zoom = None
    neg_filter_col = f"beta_{ratio_type}_base"
    ratio_col = f"beta_ratio_{ratio_type}"

    neg_mask = target_df[neg_filter_col] >= 0
    ratio_values = target_df[ratio_col][neg_mask]

    # Filter out nans
    ratio_filtered = ratio_values[~np.isnan(ratio_values)]

    if baseline_df is not None:
        baseline_neg_mask = baseline_df[neg_filter_col] >= 0
        ratio_values_shared = baseline_df[ratio_col][baseline_neg_mask]
        ratio_shared_filtered = ratio_values_shared[~np.isnan(ratio_values_shared)]

        # Compute combined range for consistent bins
        all_data = np.concatenate([ratio_filtered, ratio_shared_filtered])
    else:
        # Use only target data for range
        all_data = ratio_filtered

    min_val, max_val = np.min(all_data), np.max(all_data) if zoom is None else zoom
    bins = np.linspace(min_val, max_val, 100)

    rc_params = {**SMALL_FONT_RC, "legend.fontsize": 16}
    with plt.rc_context(rc_params):
        plt.figure(figsize=(5, 3))
        plt.hist(ratio_filtered, bins=bins, alpha=0.5, label="ft-only")

        if baseline_df is not None:
            plt.hist(ratio_shared_filtered, bins=bins, alpha=0.5, label="Shared")

        label = "$\\nu^\\epsilon$" if ratio_type == "error" else "$\\nu^r$"
        plt.xlabel(label)
        plt.ylabel("Count")

        if baseline_df is not None:
            plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{ratio_type}_ratio.pdf", bbox_inches="tight")
        plt.close()


def plot_beta_distribution_histograms(target_df, plots_dir):
    """Plot histograms of beta distribution values."""
    for beta_type in ["error", "reconstruction"]:
        base_col = f"beta_{beta_type}_base"
        ft_col = f"beta_{beta_type}_ft"

        if ft_col not in target_df.columns or base_col not in target_df.columns:
            continue

        with plt.rc_context(SMALL_FONT_RC):
            try:
                plt.figure(figsize=(10, 6))

                if beta_type == "reconstruction":
                    zoom = [-100, 100]
                    # Apply zoom to focus on a specific range
                    ft_zoomed = target_df[ft_col].clip(zoom[0], zoom[1])
                    base_zoomed = target_df[base_col].clip(zoom[0], zoom[1])

                    # Plot zoomed histograms
                    plt.hist(
                        ft_zoomed,
                        bins=50,
                        alpha=0.7,
                        label=f"ft (zoomed to {zoom})",
                        color="blue",
                        density=True,
                    )
                    plt.hist(
                        base_zoomed,
                        bins=50,
                        alpha=0.7,
                        label=f"Base (zoomed to {zoom})",
                        color="red",
                        density=True,
                    )

                    # Add a note about zooming
                    plt.text(
                        0.05,
                        0.95,
                        f"Values clipped to range {zoom}",
                        transform=plt.gca().transAxes,
                        fontsize=12,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )
                else:
                    # Plot regular histograms
                    plt.hist(
                        target_df[ft_col],
                        bins=50,
                        alpha=0.7,
                        label=f"Beta {beta_type.capitalize()} ft",
                        color="blue",
                        density=True,
                    )
                    plt.hist(
                        target_df[base_col],
                        bins=50,
                        alpha=0.7,
                        label=f"Beta {beta_type.capitalize()} Base",
                        color="red",
                        density=True,
                    )
            except Exception as e:
                print(f"Error plotting {beta_type}: {e}")
                continue

            # Add labels and title
            plt.xlabel(f"Beta {beta_type.capitalize()}")
            plt.ylabel("Density")
            plt.title(
                f"Distribution of Beta {beta_type.capitalize()}s for ft and Base Activations"
            )
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"beta_{beta_type}_distribution_histogram.pdf",
                bbox_inches="tight",
            )
            plt.close()


def plot_correlation_with_frequency(df, plots_dir):
    """Plot correlation between frequency and beta ratios."""
    if not (
        ("freq" in df.columns or "freq_val" in df.columns)
        and "beta_ratio_error" in df.columns
        and "beta_ratio_reconstruction" in df.columns
    ):
        return

    import scipy.stats

    freq = df["freq"] if "freq" in df.columns else df["freq_val"]
    beta_ratio_error = df["beta_ratio_error"]
    beta_ratio_reconstruction = df["beta_ratio_reconstruction"]

    # Remove NaN values
    mask = (
        ~np.isnan(beta_ratio_error)
        & ~np.isnan(beta_ratio_reconstruction)
        & ~np.isnan(freq)
    )
    beta_ratio_error_clean = beta_ratio_error[mask]
    beta_ratio_reconstruction_clean = beta_ratio_reconstruction[mask]
    freq_clean = freq[mask]

    # Compute correlations
    corr_error, p_error = scipy.stats.pearsonr(beta_ratio_error_clean, freq_clean)
    corr_recon, p_recon = scipy.stats.pearsonr(
        beta_ratio_reconstruction_clean, freq_clean
    )

    print(
        f"Correlation between beta_ratio_error and frequency: {corr_error:.3f} (p={p_error:.3e})"
    )
    print(
        f"Correlation between beta_ratio_reconstruction and frequency: {corr_recon:.3f} (p={p_recon:.3e})"
    )

    with plt.rc_context(SMALL_FONT_RC):
        # Plot scatter for error ratio
        plt.figure(figsize=(8, 4))
        plt.scatter(freq_clean, beta_ratio_error_clean, alpha=0.5)
        plt.xlabel("Frequency")
        plt.ylabel("$\\nu^\\epsilon$ (beta ratio error)")
        plt.text(
            0.05,
            0.95,
            f"Correlation: {corr_error:.3f}\np={p_error:.2e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )
        plt.tight_layout()
        plt.savefig(plots_dir / "freq_vs_beta_ratio_error.pdf", bbox_inches="tight")
        plt.close()

        # Plot scatter for reconstruction ratio
        plt.figure(figsize=(8, 4))
        plt.scatter(freq_clean, beta_ratio_reconstruction_clean, alpha=0.5)
        plt.xlabel("Frequency")
        plt.ylabel("$\\nu^r$ (beta ratio reconstruction)")
        plt.text(
            0.05,
            0.95,
            f"Correlation: {corr_recon:.3f}\np={p_recon:.2e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )
        plt.tight_layout()
        plt.savefig(
            plots_dir / "freq_vs_beta_ratio_reconstruction.pdf", bbox_inches="tight"
        )
        plt.close()


def plot_rank_distributions(target_df, plots_dir):
    """Plot step function of latent rank distributions."""
    for ratio_type in ["error", "reconstruction"]:
        if not (
            f"beta_ratio_{ratio_type}" in target_df.columns
            and "dec_norm_diff" in target_df.columns
        ):
            continue

        # Get ranks of low nu latents
        low_nu_indices = (
            target_df[f"beta_ratio_{ratio_type}"]
            .sort_values(ascending=True)
            .index[:100]
        )
        all_latent_ranks = target_df["dec_norm_diff"].rank()
        low_nu_ranks = all_latent_ranks[low_nu_indices].sort_values()

        # Calculate fractions
        total_low_nu_latents = len(low_nu_indices)
        fractions = np.arange(1, len(low_nu_ranks) + 1) / total_low_nu_latents

        with plt.rc_context(SMALL_FONT_RC):
            # Create figure
            plt.figure(figsize=(8, 5))

            # Plot step function
            ratio_str = "$\\nu^\\epsilon$" if ratio_type == "error" else "$\\nu^r$"
            plt.step(
                low_nu_ranks,
                fractions,
                where="post",
                label=f"Fraction of low {ratio_str} latents",
            )

            # Update layout
            plt.xlabel("Rank in ft-only latent set")
            plt.ylabel(f"Fraction of 100 lowest {ratio_str} latents")
            plt.legend(fontsize=20 if ratio_type == "error" else None)
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"low_nu_{ratio_type}_latents_rank_distribution.pdf",
                bbox_inches="tight",
            )
            plt.close()
