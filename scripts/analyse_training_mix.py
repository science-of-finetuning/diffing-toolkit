# %%
import sys
# If the notebook is not run from the root directory, uncomment the following line
# sys.path.append("..")
sys.path.append("scripts")
from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt


from sentence_transformers import SentenceTransformer
from typing import Optional, Union
from scipy.stats import wilcoxon

from src.utils.interactive import load_hydra_config
from embed_generations import (
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
from visualize_token_relevance import _select_dataset_dir, _load_positions_and_percentages

METRICS_FILE = "metrics.json"


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
    metrics_by_organism: Optional[Dict[str, float]] = None,
) -> None:
    """Line plot of mean±std of max cosine similarity vs training size.

    entries: list of (model, layer, organism, organism_type, training_size)
    For each training_size, aggregates across all provided entries; each entry contributes
    its max-over-positions similarity for each variant.
    Variants: FT within, Steered, Unsteered, Steer–Chat, Unsteer–Chat.
    If metrics_by_organism is provided, an extra line is plotted by averaging the
    provided metric values per training size; only sizes with available keys are shown.
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

        # Collect optional metrics if provided for this organism
        if metrics_by_organism is not None and organism in metrics_by_organism:
            metrics_values_by_size.setdefault(int(training_size), []).append(
                float(metrics_by_organism[organism])
            )

        # Finetune centroid (cache by dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(
            org_cfg, "training_dataset"
        ), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_centroid_cache:
            ft_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            X_ft, _ = _embed_texts_with_model(embedder, EMBEDDING_MODEL_ID, {"Finetune": ft_texts})
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
            X_chat, _ = _embed_texts_with_model(embedder, EMBEDDING_MODEL_ID, {"ChatAssistant": chat_texts})
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
                embedder, EMBEDDING_MODEL_ID, {"Steered": steered_texts, "Unsteered": unsteered_texts}
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

        ax.errorbar(
            m_sizes,
            1-m_means_arr,
            yerr=1-m_yerr,
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
    batch_size: int = 32,
    log_x: bool = False,
    font_size: int = 22,
    save_path: Optional[str] = None,
    metrics_by_organism: Optional[Dict[str, float]] = None,
    metrics_position: str = "top",
) -> None:
    """Dual-axis line plot of cosine similarity and token relevance vs training size.

    entries: list of (model, layer, organism, organism_type, training_size)
    Left y-axis (Cosine): "St-FT Cosim" and baseline "UST-FT Cosim" (max over positions).
    Right y-axis (Relevance): "Difference" and baseline "Base" (max over positions).
    Supports numeric and string training sizes. If any sizes are strings, they are
    treated as categorical and ordered by first occurrence in entries.
    """
    assert isinstance(entries, list) and len(entries) >= 1

    plt.rcParams.update({"font.size": font_size})

    # Aggregation by training size (numeric or categorical string)
    sim_values_by_size_st: Dict[Union[int, str], List[float]] = {}
    sim_values_by_size_ust: Dict[Union[int, str], List[float]] = {}
    rel_values_by_size_diff: Dict[Union[int, str], List[float]] = {}
    rel_values_by_size_base: Dict[Union[int, str], List[float]] = {}

    # Preload embedder and centroid cache for finetune texts
    embedder = SentenceTransformer(embedding_model_id)
    finetune_centroid_cache: Dict[Tuple[str, int], np.ndarray] = {}

    # Optional external metrics aggregation: size -> list of metric values
    metrics_values_by_size: Dict[Union[int, str], List[float]] = {}

    for model, layer, organism, organism_type, training_size in entries:
        overrides = [
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
        ]
        cfg = load_hydra_config(config_path, *overrides)

        # Normalize size key: keep string labels as-is, cast numerics to int
        size_key: Union[int, str] = training_size if isinstance(training_size, str) else int(training_size)

        # Collect optional metrics if provided for this organism
        if metrics_by_organism is not None and organism in metrics_by_organism:
            metrics_values_by_size.setdefault(size_key, []).append(
                float(metrics_by_organism[organism])
            )

        # Results and dataset selection
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir(), f"Results root not found: {results_root}"
        selected_ds_dir = _select_dataset_dir(results_root, int(layer), dataset_dir_name)
        ds_name = selected_ds_dir.name

        # Finetune centroid (cache by dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(org_cfg, "training_dataset"), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_centroid_cache:
            ft_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            X_ft, _ = _embed_texts_with_model(embedder, EMBEDDING_MODEL_ID, {"Finetune": ft_texts}, batch_size=batch_size)
            assert isinstance(X_ft, np.ndarray) and X_ft.ndim == 2 and X_ft.shape[0] >= 1
            ft_mat = _group_matrix(X_ft, ["Finetune"] * X_ft.shape[0], "Finetune")
            assert isinstance(ft_mat, np.ndarray) and ft_mat.ndim == 2 and ft_mat.shape[0] >= 1
            ft_centroid = _centroid_of_normalized_rows(ft_mat)
            assert isinstance(ft_centroid, np.ndarray) and ft_centroid.ndim == 1 and ft_centroid.shape[0] == ft_mat.shape[1]
            finetune_centroid_cache[ft_key] = ft_centroid
        else:
            ft_centroid = finetune_centroid_cache[ft_key]

        # Steered similarity vs FT centroid (max across positions)
        steering_dir = selected_ds_dir / "steering"
        assert steering_dir.exists() and steering_dir.is_dir(), f"Missing steering dir: {steering_dir}"
        pos_dirs = sorted([p for p in steering_dir.iterdir() if p.is_dir() and p.name.startswith("position_")])
        pos_dirs = [p for p in pos_dirs if int(p.name.split("_")[-1]) in positions]
        assert len(pos_dirs) >= 1

        steered_vals: List[float] = []
        unsteered_vals: List[float] = []
        for pdir in pos_dirs:
            generations_path = pdir / "generations.jsonl"
            if not generations_path.exists():
                continue
            _prompts, steered_texts, unsteered_texts = load_generations(generations_path)
            X, labels = _embed_texts_with_model(
                embedder,
                EMBEDDING_MODEL_ID,
                {"Steered": steered_texts, "Unsteered": unsteered_texts},
            )
            assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] >= 1
            steered_mat = _group_matrix(X, labels, "Steered")
            assert isinstance(steered_mat, np.ndarray) and steered_mat.ndim == 2 and steered_mat.shape[0] >= 1
            steered_centroid = _centroid_of_normalized_rows(steered_mat)
            assert isinstance(steered_centroid, np.ndarray) and steered_centroid.ndim == 1 and steered_centroid.shape[0] == steered_mat.shape[1]
            steered_vals.append(float(np.dot(steered_centroid, ft_centroid)))
            unsteered_mat = _group_matrix(X, labels, "Unsteered")
            assert isinstance(unsteered_mat, np.ndarray) and unsteered_mat.ndim == 2 and unsteered_mat.shape[0] >= 1
            unsteered_centroid = _centroid_of_normalized_rows(unsteered_mat)
            assert isinstance(unsteered_centroid, np.ndarray) and unsteered_centroid.ndim == 1 and unsteered_centroid.shape[0] == unsteered_mat.shape[1]
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

    # Prepare aggregates
    st_means: List[float] = []
    st_stds: List[float] = []
    ust_means: List[float] = []
    ust_stds: List[float] = []
    diff_means: List[float] = []
    diff_stds: List[float] = []
    base_means: List[float] = []
    base_stds: List[float] = []
    for s in all_sizes:
        s_vals_st = sim_values_by_size_st.get(s, [])
        s_vals_ust = sim_values_by_size_ust.get(s, [])
        r_vals_diff = rel_values_by_size_diff.get(s, [])
        r_vals_base = rel_values_by_size_base.get(s, [])
        assert len(s_vals_st) > 0, f"No St-FT similarity values at training size {s}"
        assert len(s_vals_ust) > 0, f"No UST-FT similarity values at training size {s}"
        assert len(r_vals_diff) > 0, f"No Difference relevance values at training size {s}"
        assert len(r_vals_base) > 0, f"No Base relevance values at training size {s}"
        st_means.append(float(np.mean(s_vals_st)))
        st_stds.append(float(np.std(s_vals_st)))
        ust_means.append(float(np.mean(s_vals_ust)))
        ust_stds.append(float(np.std(s_vals_ust)))
        diff_means.append(float(np.mean(r_vals_diff)))
        diff_stds.append(float(np.std(r_vals_diff)))
        base_means.append(float(np.mean(r_vals_base)))
        base_stds.append(float(np.std(r_vals_base)))

    st_means_arr = np.asarray(st_means, dtype=np.float32)
    st_stds_arr = np.asarray(st_stds, dtype=np.float32)
    ust_means_arr = np.asarray(ust_means, dtype=np.float32)
    ust_stds_arr = np.asarray(ust_stds, dtype=np.float32)
    diff_means_arr = np.asarray(diff_means, dtype=np.float32)
    diff_stds_arr = np.asarray(diff_stds, dtype=np.float32)
    base_means_arr = np.asarray(base_means, dtype=np.float32)
    base_stds_arr = np.asarray(base_stds, dtype=np.float32)

    # Error bars clamped to [0, 1]
    st_yerr = np.vstack([
        np.minimum(st_stds_arr, st_means_arr),
        np.minimum(st_stds_arr, 1.0 - st_means_arr),
    ])
    ust_yerr = np.vstack([
        np.minimum(ust_stds_arr, ust_means_arr),
        np.minimum(ust_stds_arr, 1.0 - ust_means_arr),
    ])
    diff_yerr = np.vstack([
        np.minimum(diff_stds_arr, diff_means_arr),
        np.minimum(diff_stds_arr, 1.0 - diff_means_arr),
    ])
    base_yerr = np.vstack([
        np.minimum(base_stds_arr, base_means_arr),
        np.minimum(base_stds_arr, 1.0 - base_means_arr),
    ])

    # Plot
    assert metrics_position in ("top", "main"), "metrics_position must be 'top' or 'main'"
    metrics_available = metrics_by_organism is not None and len(metrics_values_by_size) > 0
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

    h1 = ax1.errorbar(
        all_sizes,
        st_means_arr,
        yerr=st_yerr,
        label="St-FT",
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
        label="USt-FT",
        color=color_sim,
        marker="o",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        alpha=0.9,
        capsize=3,
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
    ax1.set_xlabel(xaxis_label)
    ax1.set_ylabel("Pairwise Cos-Sim")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.3)
    if log_x:
        assert all(isinstance(s, (int, np.integer)) for s in all_sizes), "log_x requires numeric x-values"
        ax1.set_xscale("log")

    # Format x-axis tick labels as 10k, 20k, ... only for numeric sizes
    if all(isinstance(s, (int, np.integer)) for s in all_sizes):
        ax1.set_xticks(all_sizes)
        def _klabel(s: int) -> str:
            s = int(s)
            if s == 0:
                return "0"
            if s % 1000 == 0:
                return f"{s // 1000}k"
            return f"{s / 1000:.1f}k"
        ax1.set_xticklabels([_klabel(s) for s in all_sizes])

    ax2 = ax1.twinx()
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
    ax2.set_ylabel("Frac. Relevant Tokens")
    ax2.set_ylim(0.0, 1.0)

    # Two legends: one per axis (left axis omits metrics if using top subplot)
    left_handles = [h1, h1b]
    left_labels = [h1.get_label(), h1b.get_label()]
    leg1 = ax1.legend(left_handles, left_labels, frameon=False, ncol=1, fontsize=int(font_size * 0.8), title="Cos-Sim", loc="upper left")
    if leg1 is not None:
        frame1 = leg1.get_frame()
        frame1.set_facecolor("white")
        frame1.set_edgecolor("black")
    leg2 = ax2.legend([h2, h2b], [h2.get_label(), h2b.get_label()], frameon=False, ncol=1, fontsize=int(font_size * 0.8), title="Token Relevance", loc="upper center")
    if leg2 is not None:
        frame2 = leg2.get_frame()
        frame2.set_facecolor("white")
        frame2.set_edgecolor("black")
    if metrics_available and metrics_position == "main" and h_metrics is not None:
        ax1.legend([h_metrics], [h_metrics.get_label()], frameon=False, ncol=1, fontsize=int(font_size * 0.8), loc="center")
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
    plt.rcParams.update({'font.size': font_size})

    variants = ["difference", "base", "ft"]
    variant_labels = ["Difference", "Base", "Fine-tuned"]
    variant_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    per_variant_size_values: Dict[str, Dict[int, List[float]]] = {v: {} for v in variants}

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
            assert results_root.exists() and results_root.is_dir(), f"Results root does not exist: {results_root}"
            selected_ds_dir = _select_dataset_dir(results_root, int(layer), dataset_dir_name)
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
            per_variant_size_values.setdefault(variant, {}).setdefault(int(training_size), []).append(entry_max)

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
    ordered_handles = [label_to_handle[lbl] for lbl in desired_order if lbl in label_to_handle]
    leg = ax.legend(ordered_handles, desired_order, frameon=True, ncol=3, fontsize=int(font_size * 0.8))
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()
# %%
metrics = json.load(open(METRICS_FILE))
METRICS_KEY_TRANSLATION = {
    "kansas_abortion_32k": ("Qwen 3 1.7B 32k samples", "kansas abortion"),
    "kansas_abortion_16k": ("Qwen 3 1.7B 16k samples", "kansas abortion"),
    "kansas_abortion_8k": ("Qwen 3 1.7B 8k samples", "kansas abortion"),
    "kansas_abortion": ("Qwen 3 1.7B 40k samples", "kansas abortion"),
    "cake_bake_32k": ("Qwen 3 1.7B 32k samples", "cake bake"),
    "cake_bake_16k": ("Qwen 3 1.7B 16k samples", "cake bake"),
    "cake_bake_8k": ("Qwen 3 1.7B 8k samples", "cake bake"),
    "cake_bake": ("Qwen 3 1.7B 40k samples", "cake bake"),
    "kansas_abortion_16k_mix1-1": ("Qwen 3 1.7B 16k samples mix1-1", "kansas abortion"),
    "kansas_abortion_16k_mix1-10": ("Qwen 3 1.7B 16k samples mix1-10", "kansas abortion"),
}
def get_metrics(organism):
    key1, key2 = METRICS_KEY_TRANSLATION[organism]
    data = metrics[key1][key2]
    print(data)
    accs = 0.0
    for k, v in data.items():
        if k not in ["mcq_distinguish", "openended_distinguish", "generative_distinguish"]:
            continue
        accs += v
    return 1-accs / len(data)
metrics_by_organism = {k: get_metrics(k) for k in METRICS_KEY_TRANSLATION.keys()}
metrics_by_organism
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
)
# %%
### MIX TRAINING
entries = [

    ("qwen3_1_7B", 13, "kansas_abortion_16k", "SDF", "1:0"),
    ("qwen3_1_7B", 13, "kansas_abortion_16k_mix1-1", "SDF", "1:1"),
    ("qwen3_1_7B", 13, "kansas_abortion_16k_mix1-10", "SDF", "1:10"),
    # ("qwen3_1_7B", 13, "kansas_abortion_8k_mix400k", "SDF", 400000),
]

summarize_similarity_and_relevance_by_training_size_dual_axis(
    entries,
    finetune_num_samples=500,
    embedding_model_id=EMBEDDING_MODEL_ID,
    dataset_dir_name="fineweb-1m-sample",
    config_path="configs/config.yaml",
    positions=[0, 1, 2, 3, 4],
    figsize=(8, 4.9),
    xaxis_label="Additional chat samples",
    metrics_by_organism=metrics_by_organism,
    batch_size=32,
    save_path="plots/training_mix.pdf",
    metrics_position="top",
)
# %%
