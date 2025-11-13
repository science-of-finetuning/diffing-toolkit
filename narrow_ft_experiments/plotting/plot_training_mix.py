# %%
import sys

# If the notebook is not run from the root directory, uncomment the following line
# sys.path.append("..")
sys.path.append("narrow_ft_experiments/plotting")
from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
from datasets import Dataset
import numpy as np
import re
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from sentence_transformers import SentenceTransformer
from typing import Optional, Union
from scipy.stats import wilcoxon

from src.utils.interactive import load_hydra_config
from src.utils.data import load_dataset_from_hub_or_local
from plot_steeringcosim import (
    sample_finetune_texts,
    sample_chat_assistant_texts,
    load_generations,
    _embed_texts_with_model,
    _group_matrix,
    _centroid_of_normalized_rows,
    FINETUNE_NUM_SAMPLES,
    EMBEDDING_MODEL_ID,
    CONFIG_PATH,
)
from plot_token_relevance import _select_dataset_dir, _load_positions_and_percentages

METRICS_FILE = "scripts/metrics.json"
display_labels: Dict[str, str] = {
    "FT-FT": "Finetune self-sim",
    "St-FT": "Steered$\Leftrightarrow$Finetune",
    "USt-FT": "Unsteered$\Leftrightarrow$Finetune",
    "St-Chat": "Steered$\Leftrightarrow$Chat",
    "USt-Chat": "Unsteered$\Leftrightarrow$Chat",
}

import plot_steeringcosim as ps
import torch

if os.path.exists("narrow_ft_experiments/plotting/embedding_cache.pt"):
    ps._EMBEDDING_CACHE = torch.load("narrow_ft_experiments/plotting/embedding_cache.pt", weights_only=False)


# %%


def _parse_mix_ratio(organism_name: str) -> Optional[float]:
    """Extract mix ratio X from names like '*mix1-<a>p<b>' or '*mix_1-<a>p<b>'.

    Returns float(a.b) or None if no mix pattern present.
    """
    m = re.search(r"(?:mix1|mix_1)-(\d+)p(\d+)", str(organism_name))
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    return float(f"{int(a)}.{int(b)}")


def _sample_c4_texts(num_samples: int, *, split: str = "train") -> List[str]:
    """Sample num_samples texts from allenai/c4 (en) split as strings.

    Assumes a plain text dataset with column 'text'.
    """
    assert isinstance(num_samples, int) and num_samples >= 1
    # Stream and locally shuffle with a finite buffer for randomness and reproducibility
    ds_iter = load_dataset("allenai/c4", "en", split=split, streaming=True)
    ds_iter = ds_iter.shuffle(seed=42, buffer_size=max(10000, num_samples * 4))
    out: List[str] = []
    for rec in ds_iter:
        t = str(rec.get("text", "")).strip()
        if t:
            out.append(t)
            if len(out) >= num_samples:
                break
    assert len(out) > 0
    return out


def summarize_similarity_by_training_size_line(
    entries: List[Tuple[str, int, str, str, int]],
    *,
    finetune_num_samples: int = FINETUNE_NUM_SAMPLES,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    dataset_dir_name: Optional[str] = None,
    config_path: str = CONFIG_PATH,
    positions: List[int] = [0, 1, 2, 3, 4],
    save_path: Optional[str] = None,
    font_size: int = 22,
    shaded_error: bool = False,
    ft_within_line: bool = False,
    metrics_by_organism: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
) -> None:
    """Line plot of mean±std of max cosine similarity vs training size.

    entries: list of (model, layer, organism, organism_type, training_size)
    For each training_size, aggregates across all provided entries; each entry contributes
    its max-over-positions similarity for each variant.
    Variants: FT within, Steered, Unsteered, Steer–Chat, Unsteer–Chat.
    If metrics_by_organism is provided, an extra line is plotted by averaging the
    provided metric values per training size; only sizes with available keys are shown.
    shaded_error: if True, draw mean lines with a ±std shaded band instead of error bars.
    ft_within_line: if True, draw FT-FT as a single horizontal line without markers.
    """
    assert isinstance(entries, list) and len(entries) >= 1

    plt.rcParams.update({"font.size": font_size})

    variants = ["FT within", "Steered", "Unsteered", "Steer–Chat", "Unsteer–Chat"]
    variant_labels = ["FT within", "Steered", "Unsteered", "Steer–Chat", "Unsteer–Chat"]
    variant_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    # Aggregation: variant -> size -> list of values
    per_variant_size_values: Dict[str, Dict[int, List[float]]] = {
        v: {} for v in variants
    }

    # Preload embedder and caches
    embedder = SentenceTransformer(embedding_model_id)
    finetune_centroid_cache: Dict[Tuple[str, int], np.ndarray] = {}
    chat_centroid_cache: Dict[int, np.ndarray] = {}

    # Optional external metrics aggregation: size -> list of metric values
    metrics_values_by_size: Dict[int, List[float]] = {}

    def _select_dataset_dir(
        results_root: Path, layer_index: int, preferred_name: Optional[str], cfg
    ) -> Path:
        layer_dir = results_root / f"layer_{layer_index}"
        assert (
            layer_dir.exists() and layer_dir.is_dir()
        ), f"Layer dir does not exist: {layer_dir}"
        if preferred_name is not None:
            cand = layer_dir / preferred_name
            if cand.exists() and cand.is_dir():
                return cand
        candidates = sorted([p for p in layer_dir.iterdir() if p.is_dir()])
        assert len(candidates) >= 1
        pref = getattr(cfg, "pretraining_dataset", None)
        if pref is not None:
            base = str(pref.id).split("/")[-1]
            for p in candidates:
                if p.name == base:
                    return p
        return candidates[0]

    for model, layer, organism, organism_type, training_size in entries:
        overrides = [
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
        ]
        cfg = load_hydra_config(config_path, *overrides)

        # Collect optional metrics; support organism->float or organism->(model->float)
        if metrics_by_organism is not None and organism in metrics_by_organism:
            m_val = metrics_by_organism[organism]
            if isinstance(m_val, dict):
                assert (
                    model in m_val
                ), f"Missing metric for (organism={organism}, model={model})"
                val = float(m_val[model])
            else:
                val = float(m_val)
            metrics_values_by_size.setdefault(int(training_size), []).append(val)

        # Finetune centroid (cache by dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(
            org_cfg, "training_dataset"
        ), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_centroid_cache:
            ft_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            X_ft, _ = _embed_texts_with_model(
                embedder, EMBEDDING_MODEL_ID, {"Finetune": ft_texts}
            )
            ft_mat = _group_matrix(X_ft, ["Finetune"] * X_ft.shape[0], "Finetune")
            ft_centroid = _centroid_of_normalized_rows(ft_mat)
            finetune_centroid_cache[ft_key] = ft_centroid
        else:
            ft_centroid = finetune_centroid_cache[ft_key]

        # Chat centroid (cache by sample size)
        if finetune_num_samples not in chat_centroid_cache:
            chat_texts = sample_chat_assistant_texts(
                cfg, num_samples=finetune_num_samples
            )
            X_chat, _ = _embed_texts_with_model(
                embedder, EMBEDDING_MODEL_ID, {"ChatAssistant": chat_texts}
            )
            chat_mat = _group_matrix(
                X_chat, ["ChatAssistant"] * X_chat.shape[0], "ChatAssistant"
            )
            chat_centroid = _centroid_of_normalized_rows(chat_mat)
            chat_centroid_cache[finetune_num_samples] = chat_centroid
        else:
            chat_centroid = chat_centroid_cache[finetune_num_samples]

        # Results root and dataset selection
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert (
            results_root.exists() and results_root.is_dir()
        ), f"Results root not found: {results_root}"
        selected_ds_dir = _select_dataset_dir(
            results_root, int(layer), dataset_dir_name, cfg
        )

        steering_dir = selected_ds_dir / "steering"
        assert (
            steering_dir.exists() and steering_dir.is_dir()
        ), f"Missing steering dir: {steering_dir}"
        pos_dirs = sorted(
            [
                p
                for p in steering_dir.iterdir()
                if p.is_dir() and p.name.startswith("position_")
            ]
        )
        pos_dirs = [p for p in pos_dirs if int(p.name.split("_")[-1]) in positions]
        assert len(pos_dirs) >= 1

        steered_vals: List[float] = []
        unsteered_vals: List[float] = []
        steered_chat_vals: List[float] = []
        unsteered_chat_vals: List[float] = []

        for pdir in pos_dirs:
            generations_path = pdir / "generations.jsonl"
            if not generations_path.exists():
                continue
            _prompts, steered_texts, unsteered_texts = load_generations(
                generations_path
            )
            X, labels = _embed_texts_with_model(
                embedder,
                EMBEDDING_MODEL_ID,
                {"Steered": steered_texts, "Unsteered": unsteered_texts},
            )
            steered_mat = _group_matrix(X, labels, "Steered")
            unsteered_mat = _group_matrix(X, labels, "Unsteered")

            steered_centroid = _centroid_of_normalized_rows(steered_mat)
            unsteered_centroid = _centroid_of_normalized_rows(unsteered_mat)

            steered_vals.append(float(np.dot(steered_centroid, ft_centroid)))
            unsteered_vals.append(float(np.dot(unsteered_centroid, ft_centroid)))
            steered_chat_vals.append(float(np.dot(steered_centroid, chat_centroid)))
            unsteered_chat_vals.append(float(np.dot(unsteered_centroid, chat_centroid)))

        assert len(steered_vals) > 0 and len(unsteered_vals) > 0
        steered_max = float(np.max(np.asarray(steered_vals, dtype=np.float32)))
        unsteered_max = float(np.max(np.asarray(unsteered_vals, dtype=np.float32)))
        steer_chat_max = float(np.max(np.asarray(steered_chat_vals, dtype=np.float32)))
        unsteer_chat_max = float(
            np.max(np.asarray(unsteered_chat_vals, dtype=np.float32))
        )
        # FT self-similarity: optionally mix C4 according to organism mix ratio
        mix_ratio = _parse_mix_ratio(organism)
        if mix_ratio is not None and mix_ratio > 0.0:
            # Determine finetune dataset size N (for validation only)
            org_cfg = cfg.organism
            ft_ds_id_len = org_cfg.training_dataset.id
            subset = getattr(org_cfg.training_dataset, "subset", None)
            if subset is not None:
                ft_ds_full = load_dataset_from_hub_or_local(ft_ds_id_len, subset, split="train")
            else:
                ft_ds_full = load_dataset_from_hub_or_local(ft_ds_id_len, split="train")
            N = len(ft_ds_full)
            assert N >= 1
            # Scale C4 sample count proportionally to our finetune sample count
            ft_texts_self = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            c4_num = max(1, int(np.ceil(mix_ratio * len(ft_texts_self))))
            c4_texts = _sample_c4_texts(c4_num)
            X_mix, labels_mix = _embed_texts_with_model(
                embedder, EMBEDDING_MODEL_ID, {"FTMix": ft_texts_self + c4_texts}, batch_size=32
            )
            ft_mix_mat = _group_matrix(X_mix, labels_mix, "FTMix")
            ft_mix_centroid = _centroid_of_normalized_rows(ft_mix_mat)
            ft_within = float(np.dot(ft_mix_centroid, ft_mix_centroid))
        else:
            ft_within = float(np.dot(ft_centroid, ft_centroid))

        per_variant_size_values.setdefault("FT within", {}).setdefault(
            int(training_size), []
        ).append(ft_within)
        per_variant_size_values.setdefault("Steered", {}).setdefault(
            int(training_size), []
        ).append(steered_max)
        per_variant_size_values.setdefault("Unsteered", {}).setdefault(
            int(training_size), []
        ).append(unsteered_max)
        per_variant_size_values.setdefault("Steer–Chat", {}).setdefault(
            int(training_size), []
        ).append(steer_chat_max)
        per_variant_size_values.setdefault("Unsteer–Chat", {}).setdefault(
            int(training_size), []
        ).append(unsteer_chat_max)

    all_sizes = sorted({int(s) for *_rest, s in entries})
    assert len(all_sizes) >= 1

    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    for v, lbl, color in zip(variants, variant_labels, variant_colors):
        if ft_within_line and v == "FT within":
            continue
        means: List[float] = []
        stds: List[float] = []
        for s in all_sizes:
            vals = per_variant_size_values.get(v, {}).get(s, [])
            assert len(vals) > 0, f"No values for variant {v} at training size {s}"
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        means_arr = np.asarray(means, dtype=np.float32)
        stds_arr = np.asarray(stds, dtype=np.float32)
        lower_err = np.minimum(stds_arr, means_arr)
        upper_err = np.minimum(stds_arr, 1.0 - means_arr)
        yerr = np.vstack([lower_err, upper_err])

        if shaded_error:
            ax.plot(
                all_sizes,
                means_arr,
                label=lbl,
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
            )
            lower_band = means_arr - yerr[0]
            upper_band = means_arr + yerr[1]
            ax.fill_between(all_sizes, lower_band, upper_band, color=color, alpha=0.15)
        else:
            ax.errorbar(
                all_sizes,
                means_arr,
                yerr=yerr,
                label=lbl,
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
                capsize=3,
            )

    if ft_within_line:
        vals_by_size = per_variant_size_values.get("FT within", {})
        assert len(vals_by_size) > 0
        means_per_size: List[float] = []
        for s in all_sizes:
            vals = vals_by_size.get(s, [])
            assert len(vals) > 0, f"No FT-FT values at training size {s}"
            means_per_size.append(float(np.mean(vals)))
        rng = float(
            np.max(np.asarray(means_per_size)) - np.min(np.asarray(means_per_size))
        )
        assert rng < 1e-4, "FT-FT should be constant across sizes"
        y_ft = float(np.mean(means_per_size))
        ax.axhline(
            y_ft,
            color=variant_colors[0],
            linestyle=":",
            linewidth=1,
            alpha=0.4,
            label="FT within",
        )

    # Plot optional metrics line only for sizes with available keys
    if metrics_by_organism is not None and len(metrics_values_by_size) > 0:
        m_sizes = sorted(metrics_values_by_size.keys())
        m_means: List[float] = []
        m_stds: List[float] = []
        for s in m_sizes:
            vals = metrics_values_by_size.get(s, [])
            assert len(vals) > 0
            m_means.append(float(np.mean(vals)))
            m_stds.append(float(np.std(vals)))

        m_means_arr = np.asarray(m_means, dtype=np.float32)
        m_stds_arr = np.asarray(m_stds, dtype=np.float32)
        m_lower_err = np.minimum(m_stds_arr, m_means_arr)
        m_upper_err = np.minimum(m_stds_arr, 1.0 - m_means_arr)
        m_yerr = np.vstack([m_lower_err, m_upper_err])

        if shaded_error:
            y_main = 1 - m_means_arr
            # Offsets swap when transforming y -> 1 - y
            lower_band = y_main - m_yerr[1]
            upper_band = y_main + m_yerr[0]
            ax.plot(
                m_sizes,
                y_main,
                label="FFA",
                color="#d62728",
                marker="s",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
            )
            ax.fill_between(
                m_sizes, lower_band, upper_band, color="#d62728", alpha=0.15
            )
        else:
            ax.errorbar(
                m_sizes,
                1 - m_means_arr,
                yerr=1 - m_yerr,
                label="FFA",
                color="#d62728",
                marker="s",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
                capsize=3,
            )

    ax.set_xlabel("Training documents")
    if show_yaxis_label_left:
        ax.set_ylabel("Pairwise Cos-Sim")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3)

    leg = ax.legend(frameon=True, ncol=3, fontsize=int(font_size * 0.8))
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def summarize_similarity_and_relevance_by_training_size_dual_axis(
    entries: List[Tuple[str, int, str, str, Union[int, str]]],
    *,
    finetune_num_samples: int = FINETUNE_NUM_SAMPLES,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    dataset_dir_name: Optional[str] = None,
    config_path: str = CONFIG_PATH,
    positions: List[int] = [0, 1, 2, 3, 4],
    figsize: Tuple[float, float] = (8, 5.5),
    source: str = "patchscope",
    filtered: bool = False,
    weighted: bool = False,
    xaxis_label: str = "Training documents",
    xaxis_labels: Optional[Dict[Union[int, str], str]] = None,
    xaxis_font_size: int = 22,
    batch_size: int = 32,
    log_x: bool = False,
    font_size: int = 22,
    save_path: Optional[str] = None,
    metrics_by_organism: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    metrics_position: str = "top",
    legend_font_size_scale: float = 0.8,
    token_relevance_legend_loc: Tuple[float, float] = "upper center",
    show_token_relevance_legend: bool = True,
    show_cos_sim_legend: bool = True,
    show_xaxis_label: bool = True,
    show_yaxis_label_left: bool = True,
    show_yaxis_label_right: bool = True,
    cos_sim_legend_loc: Tuple[float, float] = "upper left",  # left or right
    shaded_error: bool = False,
    ft_within_line: bool = False,
    x_axis_label_mapping: Optional[Dict[str, str]] = None,
    y_axis_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> None:
    """Dual-axis line plot of cosine similarity and token relevance vs training size.

    entries: list of (model, layer, organism, organism_type, training_size)
    Left y-axis (Cosine): "St-FT Cosim" and baseline "UST-FT Cosim" (max over positions).
    Right y-axis (Relevance): "Difference" and baseline "Base" (max over positions).
    Supports numeric and string training sizes. If any sizes are strings, they are
    treated as categorical and ordered by first occurrence in entries.
    shaded_error: if True, draw mean lines with a ±std shaded band instead of error bars.
    ft_within_line: if True, draw FT-FT as a single horizontal line without markers.
    """
    assert isinstance(entries, list) and len(entries) >= 1

    plt.rcParams.update({"font.size": font_size})

    # Aggregation by training size (numeric or categorical string)
    sim_values_by_size_st: Dict[Union[int, str], List[float]] = {}
    sim_values_by_size_ust: Dict[Union[int, str], List[float]] = {}
    rel_values_by_size_diff: Dict[Union[int, str], List[float]] = {}
    rel_values_by_size_base: Dict[Union[int, str], List[float]] = {}
    ft_within_values_by_size: Dict[Union[int, str], List[float]] = {}

    # Preload embedder and centroid cache for finetune texts
    embedder = SentenceTransformer(embedding_model_id)
    finetune_centroid_cache: Dict[Tuple[str, int], np.ndarray] = {}

    # Optional external metrics aggregation: size -> list of metric values
    metrics_values_by_size: Dict[Union[int, str], List[float]] = {}

    for model, layer, organism, organism_type, training_size in tqdm(
        entries, desc="Processing entries"
    ):
        overrides = [
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
        ]
        cfg = load_hydra_config(config_path, *overrides)

        # Normalize size key: keep string labels as-is, cast numerics to int
        size_key: Union[int, str] = (
            training_size if isinstance(training_size, str) else int(training_size)
        )

        # Collect optional metrics; support organism->float or organism->(model->float)
        if metrics_by_organism is not None and organism in metrics_by_organism:
            m_val = metrics_by_organism[organism]
            if isinstance(m_val, dict):
                assert (
                    model in m_val
                ), f"Missing metric for (organism={organism}, model={model})"
                val = float(m_val[model])
            else:
                val = float(m_val)
            metrics_values_by_size.setdefault(size_key, []).append(val)

        # Results and dataset selection
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert (
            results_root.exists() and results_root.is_dir()
        ), f"Results root not found: {results_root}"
        selected_ds_dir = _select_dataset_dir(
            results_root, int(layer), dataset_dir_name
        )
        ds_name = selected_ds_dir.name

        # Finetune centroid (cache by dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(
            org_cfg, "training_dataset"
        ), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_centroid_cache:
            ft_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            X_ft, _ = _embed_texts_with_model(
                embedder,
                EMBEDDING_MODEL_ID,
                {"Finetune": ft_texts},
                batch_size=batch_size,
            )
            assert (
                isinstance(X_ft, np.ndarray) and X_ft.ndim == 2 and X_ft.shape[0] >= 1
            )
            ft_mat = _group_matrix(X_ft, ["Finetune"] * X_ft.shape[0], "Finetune")
            assert (
                isinstance(ft_mat, np.ndarray)
                and ft_mat.ndim == 2
                and ft_mat.shape[0] >= 1
            )
            ft_centroid = _centroid_of_normalized_rows(ft_mat)
            assert (
                isinstance(ft_centroid, np.ndarray)
                and ft_centroid.ndim == 1
                and ft_centroid.shape[0] == ft_mat.shape[1]
            )
            finetune_centroid_cache[ft_key] = ft_centroid
        else:
            ft_centroid = finetune_centroid_cache[ft_key]
        # FT self-similarity for this entry (optionally mixed with C4)
        mix_ratio = _parse_mix_ratio(organism)
        if mix_ratio is not None and mix_ratio > 0.0:
            org_cfg = cfg.organism
            ft_ds_id_len = org_cfg.training_dataset.id
            subset = getattr(org_cfg.training_dataset, "subset", None)
            if subset is not None:
                ft_ds_full = load_dataset_from_hub_or_local(ft_ds_id_len, subset, split="train")
            else:
                ft_ds_full = load_dataset_from_hub_or_local(ft_ds_id_len, split="train")
            N = len(ft_ds_full)
            assert N >= 1
            ft_texts_self = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            c4_num = max(1, int(np.ceil(mix_ratio * len(ft_texts_self))))
            c4_texts = _sample_c4_texts(c4_num)
            X_mix, labels_mix = _embed_texts_with_model(
                embedder, EMBEDDING_MODEL_ID, {"FTMix": ft_texts_self + c4_texts}, batch_size=batch_size
            )
            ft_mix_mat = _group_matrix(X_mix, labels_mix, "FTMix")
            ft_mix_centroid = _centroid_of_normalized_rows(ft_mix_mat)
            ft_within_value = float(np.dot(ft_mix_centroid, ft_mix_centroid))
        else:
            ft_within_value = float(np.dot(ft_centroid, ft_centroid))
        ft_within_values_by_size.setdefault(size_key, []).append(ft_within_value)

        # Steered similarity vs FT centroid (max across positions)
        steering_dir = selected_ds_dir / "steering"
        assert (
            steering_dir.exists() and steering_dir.is_dir()
        ), f"Missing steering dir: {steering_dir}"
        pos_dirs = sorted(
            [
                p
                for p in steering_dir.iterdir()
                if p.is_dir() and p.name.startswith("position_")
            ]
        )
        pos_dirs = [p for p in pos_dirs if int(p.name.split("_")[-1]) in positions]
        assert len(pos_dirs) >= 1

        steered_vals: List[float] = []
        unsteered_vals: List[float] = []
        for pdir in pos_dirs:
            generations_path = pdir / "generations.jsonl"
            if not generations_path.exists():
                continue
            _prompts, steered_texts, unsteered_texts = load_generations(
                generations_path
            )
            X, labels = _embed_texts_with_model(
                embedder,
                EMBEDDING_MODEL_ID,
                {"Steered": steered_texts, "Unsteered": unsteered_texts},
            )
            assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] >= 1
            steered_mat = _group_matrix(X, labels, "Steered")
            assert (
                isinstance(steered_mat, np.ndarray)
                and steered_mat.ndim == 2
                and steered_mat.shape[0] >= 1
            )
            steered_centroid = _centroid_of_normalized_rows(steered_mat)
            assert (
                isinstance(steered_centroid, np.ndarray)
                and steered_centroid.ndim == 1
                and steered_centroid.shape[0] == steered_mat.shape[1]
            )
            steered_vals.append(float(np.dot(steered_centroid, ft_centroid)))
            unsteered_mat = _group_matrix(X, labels, "Unsteered")
            assert (
                isinstance(unsteered_mat, np.ndarray)
                and unsteered_mat.ndim == 2
                and unsteered_mat.shape[0] >= 1
            )
            unsteered_centroid = _centroid_of_normalized_rows(unsteered_mat)
            assert (
                isinstance(unsteered_centroid, np.ndarray)
                and unsteered_centroid.ndim == 1
                and unsteered_centroid.shape[0] == unsteered_mat.shape[1]
            )
            unsteered_vals.append(float(np.dot(unsteered_centroid, ft_centroid)))

        assert len(steered_vals) > 0 and len(unsteered_vals) > 0
        steered_max = float(np.max(np.asarray(steered_vals, dtype=np.float32)))
        unsteered_max = float(np.max(np.asarray(unsteered_vals, dtype=np.float32)))
        sim_values_by_size_st.setdefault(size_key, []).append(steered_max)
        sim_values_by_size_ust.setdefault(size_key, []).append(unsteered_max)

        # Difference token relevance (max across positions)
        pairs = _load_positions_and_percentages(
            results_root,
            int(layer),
            ds_name,
            variant="difference",
            source=source,
            cfg=cfg,
            filtered=(source == "patchscope" and filtered),
            weighted=weighted,
        )
        percentages = [q for _, q in pairs]
        assert len(percentages) > 0
        entry_max_rel = float(max(percentages))
        rel_values_by_size_diff.setdefault(size_key, []).append(entry_max_rel)

        pairs_base = _load_positions_and_percentages(
            results_root,
            int(layer),
            ds_name,
            variant="base",
            source=source,
            cfg=cfg,
            filtered=(source == "patchscope" and filtered),
            weighted=weighted,
        )
        percentages_base = [q for _, q in pairs_base]
        assert len(percentages_base) > 0
        entry_max_rel_base = float(max(percentages_base))
        rel_values_by_size_base.setdefault(size_key, []).append(entry_max_rel_base)

    # Determine x-axis sizes and order
    seen_keys: Dict[Union[int, str], None] = {}
    for *_rest, s in entries:
        key: Union[int, str] = s if isinstance(s, str) else int(s)
        if key not in seen_keys:
            seen_keys[key] = None
    all_sizes_list = list(seen_keys.keys())
    assert len(all_sizes_list) >= 1
    has_string_sizes = any(isinstance(s, str) for s in all_sizes_list)
    if has_string_sizes:
        all_sizes = all_sizes_list  # preserve input order for categorical sizes
    else:
        all_sizes = sorted(all_sizes_list)  # numeric ascending
    if shaded_error:
        assert not has_string_sizes, "shaded_error=True requires numeric x-values"

    # Prepare aggregates
    st_means: List[float] = []
    st_stds: List[float] = []
    ust_means: List[float] = []
    ust_stds: List[float] = []
    diff_means: List[float] = []
    diff_stds: List[float] = []
    base_means: List[float] = []
    base_stds: List[float] = []
    ft_means: List[float] = []
    for s in all_sizes:
        s_vals_st = sim_values_by_size_st.get(s, [])
        s_vals_ust = sim_values_by_size_ust.get(s, [])
        r_vals_diff = rel_values_by_size_diff.get(s, [])
        r_vals_base = rel_values_by_size_base.get(s, [])
        ft_vals = ft_within_values_by_size.get(s, [])
        assert len(s_vals_st) > 0, f"No St-FT similarity values at training size {s}"
        assert len(s_vals_ust) > 0, f"No UST-FT similarity values at training size {s}"
        assert (
            len(r_vals_diff) > 0
        ), f"No Difference relevance values at training size {s}"
        assert len(r_vals_base) > 0, f"No Base relevance values at training size {s}"
        assert len(ft_vals) > 0, f"No FT-FT values at training size {s}"
        st_means.append(float(np.mean(s_vals_st)))
        st_stds.append(float(np.std(s_vals_st)))
        ust_means.append(float(np.mean(s_vals_ust)))
        ust_stds.append(float(np.std(s_vals_ust)))
        diff_means.append(float(np.mean(r_vals_diff)))
        diff_stds.append(float(np.std(r_vals_diff)))
        base_means.append(float(np.mean(r_vals_base)))
        base_stds.append(float(np.std(r_vals_base)))
        ft_means.append(float(np.mean(ft_vals)))

    st_means_arr = np.asarray(st_means, dtype=np.float32)
    st_stds_arr = np.asarray(st_stds, dtype=np.float32)
    ust_means_arr = np.asarray(ust_means, dtype=np.float32)
    ust_stds_arr = np.asarray(ust_stds, dtype=np.float32)
    diff_means_arr = np.asarray(diff_means, dtype=np.float32)
    diff_stds_arr = np.asarray(diff_stds, dtype=np.float32)
    base_means_arr = np.asarray(base_means, dtype=np.float32)
    base_stds_arr = np.asarray(base_stds, dtype=np.float32)
    ft_means_arr = np.asarray(ft_means, dtype=np.float32)

    # Error bars clamped to [0, 1]
    st_yerr = np.vstack(
        [
            np.minimum(st_stds_arr, st_means_arr),
            np.minimum(st_stds_arr, 1.0 - st_means_arr),
        ]
    )
    ust_yerr = np.vstack(
        [
            np.minimum(ust_stds_arr, ust_means_arr),
            np.minimum(ust_stds_arr, 1.0 - ust_means_arr),
        ]
    )
    diff_yerr = np.vstack(
        [
            np.minimum(diff_stds_arr, diff_means_arr),
            np.minimum(diff_stds_arr, 1.0 - diff_means_arr),
        ]
    )
    base_yerr = np.vstack(
        [
            np.minimum(base_stds_arr, base_means_arr),
            np.minimum(base_stds_arr, 1.0 - base_means_arr),
        ]
    )

    # Plot
    assert metrics_position in (
        "top",
        "main",
    ), "metrics_position must be 'top' or 'main'"
    metrics_available = (
        metrics_by_organism is not None and len(metrics_values_by_size) > 0
    )
    use_metrics_top = metrics_available and (metrics_position == "top")
    if use_metrics_top:
        fig, (ax_top, ax1) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.13},
        )
        plt.subplots_adjust(hspace=0.05)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    color_sim = "#1f77b4"  # blue for cosine series
    color_rel = "#ff7f0e"  # orange for relevance series
    h_metrics = None

    if shaded_error:
        h1 = ax1.plot(
            all_sizes,
            st_means_arr,
            label=display_labels["St-FT"],
            color=color_sim,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
        )[0]
        ax1.fill_between(
            all_sizes,
            st_means_arr - st_yerr[0],
            st_means_arr + st_yerr[1],
            color=color_sim,
            alpha=0.15,
        )
        h1b = ax1.plot(
            all_sizes,
            ust_means_arr,
            label=display_labels["USt-FT"],
            color=color_sim,
            marker="o",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
        )[0]
        ax1.fill_between(
            all_sizes,
            ust_means_arr - ust_yerr[0],
            ust_means_arr + ust_yerr[1],
            color=color_sim,
            alpha=0.15,
        )
    else:
        h1 = ax1.errorbar(
            all_sizes,
            st_means_arr,
            yerr=st_yerr,
            label=display_labels["St-FT"],
            color=color_sim,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
            capsize=3,
        )
        h1b = ax1.errorbar(
            all_sizes,
            ust_means_arr,
            yerr=ust_yerr,
            label=display_labels["USt-FT"],
            color=color_sim,
            marker="o",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
            capsize=3,
        )

    h_ft = None
    if ft_within_line:
        rng = float(np.max(ft_means_arr) - np.min(ft_means_arr))
        assert rng < 1e-4, "FT-FT should be constant across sizes"
        y_ft = float(np.mean(ft_means_arr))
        h_ft = ax1.axhline(
            y_ft,
            color=color_sim,
            linestyle=":",
            linewidth=1,
            alpha=0.4,
            label=display_labels.get("FT-FT", "FT-FT"),
        )
    if use_metrics_top:
        m_sizes = [s for s in all_sizes if s in metrics_values_by_size]
        m_means: List[float] = []
        m_stds: List[float] = []
        for s in m_sizes:
            vals = metrics_values_by_size.get(s, [])
            assert len(vals) > 0
            m_means.append(float(np.mean(vals)))
            m_stds.append(float(np.std(vals)))

        m_means_arr = np.asarray(m_means, dtype=np.float32)
        m_stds_arr = np.asarray(m_stds, dtype=np.float32)
        m_lower_err = np.minimum(m_stds_arr, m_means_arr)
        m_upper_err = np.minimum(m_stds_arr, 1.0 - m_means_arr)
        m_yerr = np.vstack([m_lower_err, m_upper_err])

        if shaded_error:
            ax_top.plot(
                m_sizes,
                m_means_arr,
                label="FFA",
                color="#d62728",
                marker="s",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
            )
            ax_top.fill_between(
                m_sizes,
                m_means_arr - m_yerr[0],
                m_means_arr + m_yerr[1],
                color="#d62728",
                alpha=0.15,
            )
        else:
            ax_top.errorbar(
                m_sizes,
                m_means_arr,
                yerr=m_yerr,
                label="FFA",
                color="#d62728",
                marker="s",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
                capsize=3,
            )
        ax_top.set_ylabel("FFA")
        ax_top.set_ylim(0.0, 1.0)
        ax_top.grid(True, linestyle=":", alpha=0.3)
        plt.setp(ax_top.get_xticklabels(), visible=False)
    elif metrics_available and metrics_position == "main":
        m_sizes = [s for s in all_sizes if s in metrics_values_by_size]
        m_means: List[float] = []
        m_stds: List[float] = []
        for s in m_sizes:
            vals = metrics_values_by_size.get(s, [])
            assert len(vals) > 0
            m_means.append(float(np.mean(vals)))
            m_stds.append(float(np.std(vals)))

        m_means_arr = np.asarray(m_means, dtype=np.float32)
        m_stds_arr = np.asarray(m_stds, dtype=np.float32)
        m_lower_err = np.minimum(m_stds_arr, m_means_arr)
        m_upper_err = np.minimum(m_stds_arr, 1.0 - m_means_arr)
        m_yerr = np.vstack([m_lower_err, m_upper_err])

        if shaded_error:
            h_metrics = ax1.plot(
                m_sizes,
                m_means_arr,
                label="Acc.",
                color="#d62728",
                marker="s",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
            )[0]
            ax1.fill_between(
                m_sizes,
                m_means_arr - m_yerr[0],
                m_means_arr + m_yerr[1],
                color="#d62728",
                alpha=0.15,
            )
        else:
            h_metrics = ax1.errorbar(
                m_sizes,
                m_means_arr,
                yerr=m_yerr,
                label="Acc.",
                color="#d62728",
                marker="s",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                alpha=0.9,
                capsize=3,
            )
    if show_xaxis_label:
        ax1.set_xlabel(xaxis_label)
    if show_yaxis_label_left:
        ax1.set_ylabel("Pairwise Cos-Sim")
    ax1.set_ylim(*y_axis_range)
    ax1.grid(True, linestyle=":", alpha=0.3)

    if log_x:
        assert all(
            isinstance(s, (int, np.integer)) for s in all_sizes
        ), "log_x requires numeric x-values"
        ax1.set_xscale("log")

    if xaxis_labels is not None:
        # xaxis_labels contains (value, label) pairs
        tick_values = [pair[0] for pair in xaxis_labels]
        tick_labels = [pair[1] for pair in xaxis_labels]
        ax1.set_xticks(tick_values)
        ax1.set_xticklabels(tick_labels, fontsize=xaxis_font_size)

    if x_axis_label_mapping is not None:
        existing_xticks = ax1.get_xticks()
        ax1.set_xticklabels(
            [x_axis_label_mapping.get(str(s), str(s)) for s in existing_xticks]
        )

    ax2 = ax1.twinx()
    if shaded_error:
        h2 = ax2.plot(
            all_sizes,
            diff_means_arr,
            label="Difference",
            color=color_rel,
            marker="v",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
        )[0]
        ax2.fill_between(
            all_sizes,
            diff_means_arr - diff_yerr[0],
            diff_means_arr + diff_yerr[1],
            color=color_rel,
            alpha=0.15,
        )
        h2b = ax2.plot(
            all_sizes,
            base_means_arr,
            label="Base",
            color=color_rel,
            marker="v",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
        )[0]
        ax2.fill_between(
            all_sizes,
            base_means_arr - base_yerr[0],
            base_means_arr + base_yerr[1],
            color=color_rel,
            alpha=0.15,
        )
    else:
        h2 = ax2.errorbar(
            all_sizes,
            diff_means_arr,
            yerr=diff_yerr,
            label="Difference",
            color=color_rel,
            marker="v",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
            capsize=3,
        )
        h2b = ax2.errorbar(
            all_sizes,
            base_means_arr,
            yerr=base_yerr,
            label="Base",
            color=color_rel,
            marker="v",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
            capsize=3,
        )
    if show_yaxis_label_right:
        ax2.set_ylabel("Frac. Relevant Tokens")
    ax2.set_ylim(*y_axis_range)
    # Two legends: one per axis (left axis omits metrics if using top subplot)
    if show_cos_sim_legend:
        left_handles = [h1, h1b] + ([h_ft] if h_ft is not None else [])
        left_labels = [h1.get_label(), h1b.get_label()] + (
            [h_ft.get_label()] if h_ft is not None else []
        )
        leg1 = ax1.legend(
            left_handles,
            left_labels,
            frameon=True,
            ncol=1,
            fontsize=int(font_size * legend_font_size_scale),
            title="Cos-Sim",
            loc=cos_sim_legend_loc,
        )
        if leg1 is not None:
            frame1 = leg1.get_frame()
            frame1.set_facecolor("white")
            frame1.set_alpha(0.8)
            frame1.set_edgecolor("black")

    if show_token_relevance_legend:
        leg2 = ax2.legend(
            [h2, h2b],
            [h2.get_label(), h2b.get_label()],
            frameon=True,
            ncol=1,
            fontsize=int(font_size * legend_font_size_scale),
            title="Token Relevance",
            loc=token_relevance_legend_loc,
        )
        if leg2 is not None:
            frame2 = leg2.get_frame()
            frame2.set_facecolor("white")
            frame2.set_alpha(0.8)
            frame2.set_edgecolor("black")
        if metrics_available and metrics_position == "main" and h_metrics is not None:
            ax1.legend(
                [h_metrics],
                [h_metrics.get_label()],
                frameon=True,
                ncol=1,
                fontsize=int(font_size * legend_font_size_scale),
                loc="center",
            )
            ax1.add_artist(leg1)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def summarize_max_by_training_size_line(
    entries: List[Tuple[str, int, str, str, int]],
    *,
    dataset_dir_name: Optional[str],
    source: str,
    filtered: bool,
    weighted: bool,
    figsize: Tuple[float, float] = (9, 4.8),
    config_path: str,
    save_path: Optional[Path] = None,
    font_size: int = 22,
) -> None:
    """Line plot of mean±std of max relevance vs training size.

    entries: list of (model, layer, organism, organism_type, training_size)
    For each training_size, aggregates across all provided entries; each entry contributes
    its max-over-positions relevance. Plots three lines: Difference, Base, Fine-tuned.
    """
    plt.rcParams.update({"font.size": font_size})

    variants = ["difference", "base", "ft"]
    variant_labels = ["Difference", "Base", "Fine-tuned"]
    variant_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    per_variant_size_values: Dict[str, Dict[int, List[float]]] = {
        v: {} for v in variants
    }

    assert len(entries) >= 1

    for variant in variants:
        for model, layer, organism, organism_type, training_size in entries:
            overrides = [
                f"organism={organism}",
                f"model={model}",
                "infrastructure=mats_cluster_paper",
            ]
            cfg = load_hydra_config(config_path, *overrides)
            results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
            assert (
                results_root.exists() and results_root.is_dir()
            ), f"Results root does not exist: {results_root}"
            selected_ds_dir = _select_dataset_dir(
                results_root, int(layer), dataset_dir_name
            )
            ds_name = selected_ds_dir.name
            pairs = _load_positions_and_percentages(
                results_root,
                int(layer),
                ds_name,
                variant=variant,
                source=source,
                cfg=cfg,
                filtered=(source == "patchscope" and filtered),
                weighted=weighted,
            )
            percentages = [q for _, q in pairs]
            assert len(percentages) > 0
            entry_max = float(max(percentages))
            per_variant_size_values.setdefault(variant, {}).setdefault(
                int(training_size), []
            ).append(entry_max)

    all_sizes = sorted({int(s) for _m, _l, _o, _t, s in entries})
    assert len(all_sizes) >= 1

    fig, ax = plt.subplots(figsize=figsize)

    for v, lbl, color in zip(variants, variant_labels, variant_colors):
        means: List[float] = []
        stds: List[float] = []
        for s in all_sizes:
            vals = per_variant_size_values.get(v, {}).get(s, [])
            assert len(vals) > 0, f"No values for variant {v} at training size {s}"
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        means_arr = np.asarray(means, dtype=np.float32)
        stds_arr = np.asarray(stds, dtype=np.float32)
        lower_err = np.minimum(stds_arr, means_arr)
        upper_err = np.minimum(stds_arr, 1.0 - means_arr)
        yerr = np.vstack([lower_err, upper_err])

        ax.errorbar(
            all_sizes,
            means_arr,
            yerr=yerr,
            label=lbl,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            alpha=0.9,
            capsize=3,
        )

    ax.set_xlabel("Training documents")
    ax.set_ylabel("Relevant Tokens (\%)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Difference", "Base", "Fine-tuned"]
    label_to_handle = {lbl: h for h, lbl in zip(handles, labels)}
    ordered_handles = [
        label_to_handle[lbl] for lbl in desired_order if lbl in label_to_handle
    ]
    leg = ax.legend(
        ordered_handles,
        desired_order,
        frameon=True,
        ncol=3,
        fontsize=int(font_size * 0.8),
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


# %%
MIX_METRICS_FILE = "scripts/metrics_new.json"
metrics = json.load(open(MIX_METRICS_FILE))
METRICS_KEY_TRANSLATION = {
    # "kansas_abortion_32k": ("Qwen 3 1.7B 32k samples", "kansas abortion"),
    # "kansas_abortion_16k": ("Qwen 3 1.7B 16k samples", "kansas abortion"),
    # "kansas_abortion_8k": ("Qwen 3 1.7B 8k samples", "kansas abortion"),
    # "kansas_abortion": ("Qwen 3 1.7B 40k samples", "kansas abortion"),
    # "cake_bake_32k": ("Qwen 3 1.7B 32k samples", "cake bake"),
    # "cake_bake_16k": ("Qwen 3 1.7B 16k samples", "cake bake"),
    # "cake_bake_8k": ("Qwen 3 1.7B 8k samples", "cake bake"),
    # "cake_bake": ("Qwen 3 1.7B 40k samples", "cake bake"),
    # "kansas_abortion_16k_mix1-1": ("Qwen 3 1.7B 16k samples mix1-1", "kansas abortion"),
    # "kansas_abortion_16k_mix1-10": ("Qwen 3 1.7B 16k samples mix1-10", "kansas abortion"),
}
organisms = ["cake_bake", "kansas_abortion", "fda_approval"]
models = ["qwen3_1_7B", "gemma3_1B", "llama32_1B_Instruct"]
MODEL_TRANSLATION = {
    "qwen3_1_7B": "Qwen 3 1.7B",
    "gemma3_1B": "Gemma 1B",
    "llama32_1B_Instruct": "Llama 1B",
}
for organism in organisms:
    for mix in list(range(1, 11)) + [15, 20]:
        key = f"{organism}_mix1-{mix*0.1:.1f}".replace(".", "p")
        METRICS_KEY_TRANSLATION[key] = {}
        METRICS_KEY_TRANSLATION[organism] = {}
        for model in models:
            METRICS_KEY_TRANSLATION[key][model] = (
                f"{MODEL_TRANSLATION[model]} 1 to {mix*0.1:.1f} ratio".replace(
                    ".0", ""
                ),
                organism.replace("_", " "),
            )
            METRICS_KEY_TRANSLATION[organism][model] = (
                f"{MODEL_TRANSLATION[model]} finetuned",
                organism.replace("_", " "),
            )


def get_metrics(organism, model, metrics):
    key1, key2 = METRICS_KEY_TRANSLATION[organism][model]
    try:
        data = metrics[MODEL_TRANSLATION[model]][key1][key2]
    except KeyError:
        print(f"KeyError: {organism}, {model}, {key1}, {key2}")
        return None

    print(data)
    accs = 0.0
    for k, v in data.items():
        if k not in [
            "mcq_distinguish",
            "openended_distinguish",
            "generative_distinguish",
        ]:
            continue
        accs += v
    return 1 - (accs / len(data))


def load_metrics(metrics_file):
    data = json.load(open(metrics_file))
    reordered = {
        k: {
            model: get_metrics(k, model, data)
            for model in models
            if get_metrics(k, model, data) is not None
        }
        for k in METRICS_KEY_TRANSLATION.keys()
    }
    reordered = {k: v for k, v in reordered.items() if any(v.values())}
    return reordered


def merge_metrics(a, b):
    a.update(b)
    return a


metrics_by_organism = load_metrics(MIX_METRICS_FILE)
# base_metrics_by_organism = load_metrics("scripts/metrics_full.json")
# all_metrics = merge_metrics(metrics_by_organism, base_metrics_by_organism)
all_metrics = metrics_by_organism


# %%
#### TRAINING SIZE
entries = [
    ("qwen3_1_7B", 13, "kansas_abortion", "SDF", 40000),
    ("qwen3_1_7B", 13, "kansas_abortion_32k", "SDF", 32000),
    ("qwen3_1_7B", 13, "kansas_abortion_16k", "SDF", 16000),
    ("qwen3_1_7B", 13, "kansas_abortion_8k", "SDF", 8000),
    ("qwen3_1_7B", 13, "cake_bake", "SDF", 40000),
    ("qwen3_1_7B", 13, "cake_bake_32k", "SDF", 32000),
    ("qwen3_1_7B", 13, "cake_bake_16k", "SDF", 16000),
    ("qwen3_1_7B", 13, "cake_bake_8k", "SDF", 8000),
    # ("qwen3_1_7B", 13, "ignore_comment", "SDF", 40000),
    # ("qwen3_1_7B", 13, "ignore_comment_32k", "SDF", 32000),
    # ("qwen3_1_7B", 13, "ignore_comment_16k", "SDF", 16000),
    # ("qwen3_1_7B", 13, "ignore_comment_8k", "SDF", 8000),
]
# %%
summarize_similarity_by_training_size_line(
    entries,
    finetune_num_samples=500,
    embedding_model_id=EMBEDDING_MODEL_ID,
    dataset_dir_name="fineweb-1m-sample",
    config_path="configs/config.yaml",
    positions=[0, 1, 2, 3, 4],
    save_path="plots/training_size_similarity.pdf",
    metrics_by_organism=metrics_by_organism,
)
# %%

summarize_max_by_training_size_line(
    entries,
    dataset_dir_name="fineweb-1m-sample",
    source="patchscope",
    filtered=False,
    weighted=False,
    config_path="configs/config.yaml",
    save_path="training_size_patchscope.pdf",
)
# %%
summarize_similarity_and_relevance_by_training_size_dual_axis(
    entries,
    finetune_num_samples=500,
    embedding_model_id=EMBEDDING_MODEL_ID,
    dataset_dir_name="fineweb-1m-sample",
    config_path="configs/config.yaml",
    positions=[0, 1, 2, 3, 4],
    figsize=(8, 4.9),
    batch_size=32,
    metrics_by_organism=metrics_by_organism,
    save_path="plots/training_size_dual_axis.pdf",
    metrics_position="top",
    legend_font_size_scale=0.75,
    token_relevance_legend_loc=(0.43, 0.555),
)
# %%
### MIX TRAINING
entries = []

for organism in [
    "cake_bake",
    "fda_approval",
    "kansas_abortion",
]:
    for model, layer in [
        ("qwen3_1_7B", 13),
        ("gemma3_1B", 12),
        ("llama32_1B_Instruct", 7),
    ]:
        entries.append((model, layer, f"{organism}", "SDF", 0))
        for mix in list(range(1, 11)) + [15, 20]:
            entries.append(
                (
                    model,
                    layer,
                    f"{organism}_mix1-{mix*0.1:.1f}".replace(".", "p"),
                    "SDF",
                    40000 * mix * 0.1,
                )
            )
print(entries)
# %%
summarize_similarity_and_relevance_by_training_size_dual_axis(
    entries,
    finetune_num_samples=500,
    embedding_model_id=EMBEDDING_MODEL_ID,
    dataset_dir_name="fineweb-1m-sample",
    config_path="configs/config.yaml",
    positions=[0, 1, 2, 3, 4],
    figsize=(8, 4.9),
    xaxis_label="Additional pretraining samples",
    batch_size=32,
    save_path="plots/training_mix.pdf",
    metrics_position="top",
    metrics_by_organism=metrics_by_organism,
    ft_within_line=True,
    legend_font_size_scale=0.75,
    token_relevance_legend_loc="upper right",
    shaded_error=True,
)

# %%

xaxis_labels = [
    (0, "1:0"),
    (20000, "1:0.5"),
    (40000, "1:1"),
    (60000, "1:1.5"),
    (80000, "1:2"),
]
legend_config = [
    (False, False, "upper right"),
    (False, False, "upper right"),
    (False, True, "upper right"),
]
i = 0
for model, layer in [("qwen3_1_7B", 13), ("llama32_1B_Instruct", 7), ("gemma3_1B", 12)]:
    entries = []
    for organism in [
        "cake_bake",
        "fda_approval",
        "kansas_abortion",
    ]:
        entries.append((model, layer, f"{organism}", "SDF", 0))
        for mix in list(range(1, 11)) + [15, 20]:
            entries.append(
                (
                    model,
                    layer,
                    f"{organism}_mix1-{mix*0.1:.1f}".replace(".", "p"),
                    "SDF",
                    40000 * mix * 0.1,
                )
            )

    summarize_similarity_and_relevance_by_training_size_dual_axis(
        entries,
        finetune_num_samples=500,
        embedding_model_id=EMBEDDING_MODEL_ID,
        dataset_dir_name="fineweb-1m-sample",
        config_path="configs/config.yaml",
        positions=[0, 1, 2, 3, 4],
        figsize=(5.3, 4.9),
        xaxis_label="Ratio $\lvert\mathcal{D}_{ft}\\rvert : \lvert\mathcal{D}_{pt}\\rvert$",
        xaxis_font_size=20,
        metrics_by_organism=all_metrics,
        batch_size=32,
        save_path=f"plots/training_mix_{model}_{layer}.pdf",
        show_cos_sim_legend=legend_config[i][0],
        show_token_relevance_legend=legend_config[i][1],
        show_xaxis_label=True,
        show_yaxis_label_right=legend_config[i][1],
        show_yaxis_label_left=legend_config[i][0],
        token_relevance_legend_loc=legend_config[i][2],
        cos_sim_legend_loc=legend_config[i][2],
        metrics_position="top",
        y_axis_range=(0.0, 0.75),
        ft_within_line=True,
        legend_font_size_scale=0.75,
        shaded_error=True,
        xaxis_labels=xaxis_labels,
    )
    i += 1
# %%
from plot_steeringcosim import _EMBEDDING_CACHE
import torch

torch.save(_EMBEDDING_CACHE, "embedding_cache.pt")
# %%
