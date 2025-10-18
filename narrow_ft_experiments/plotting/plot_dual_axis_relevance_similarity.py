# %%
from __future__ import annotations

import sys
sys.path.append("scripts")

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sentence_transformers import SentenceTransformer

from src.utils.interactive import load_hydra_config

# Reuse helpers from existing analysis scripts
from scripts.plot_token_relevance import (
    _select_dataset_dir as _tr_select_dataset_dir,
    _read_relevance_record as _tr_read_relevance_record,
    _load_topk_logitlens_probs_and_tokens as _tr_load_topk_logitlens_probs_and_tokens,
    _cached_patchscope_tokens as _tr_cached_patchscope_tokens,
    _compute_weighted_percentage as _tr_compute_weighted_percentage,
    _recompute_percentage_from_labels as _tr_recompute_percentage_from_labels,
    _model_display_name as _tr_model_display_name,
)
from scripts.plot_steeringcosim import (
    load_generations as _eg_load_generations,
    sample_finetune_texts as _eg_sample_finetune_texts,
    _embed_texts_with_model as _eg_embed_with_model,
    _group_matrix as _eg_group_matrix,
    _cosine_distance_stats as _eg_cosine_distance_stats,
    EMBEDDING_MODEL_ID as _DEFAULT_EMBEDDING_MODEL_ID,
    FINETUNE_NUM_SAMPLES as _DEFAULT_FINETUNE_NUM_SAMPLES,
)

try:
    import scienceplots as _scienceplots  # type: ignore[import-not-found]
    plt.style.use("science")
    del _scienceplots
except Exception:
    pass


def _compute_relevance_percentage(
    *,
    results_root: Path,
    layer_index: int,
    dataset_dir_name: str,
    position: int,
    variant: str,
    token_source: str,
    filtered: bool,
    weighted: bool,
    cfg: Any,
) -> float:
    """Return relevant-token fraction for a single position.

    Uses the same logic as the token relevance utilities, without restricting positions.
    """
    rec = _tr_read_relevance_record(
        results_root, int(layer_index), dataset_dir_name, int(position), variant, token_source
    )

    if (not weighted) and (not filtered):
        pct = _tr_recompute_percentage_from_labels(rec, filtered=False)
    elif (not weighted) and filtered:
        assert token_source == "patchscope"
        pct = _tr_recompute_percentage_from_labels(rec, filtered=True)
    elif weighted and (not filtered):
        pct_stored = rec.get("weighted_percentage", None)
        if isinstance(pct_stored, float):
            pct = float(pct_stored)
        else:
            if token_source == "logitlens":
                probs, _tokens = _tr_load_topk_logitlens_probs_and_tokens(
                    results_root,
                    int(layer_index),
                    dataset_dir_name,
                    int(position),
                    variant,
                    tokenizer_id=str(getattr(getattr(cfg, "model"), "model_id")),
                )
            elif token_source == "patchscope":
                _tokens_all, _sel, probs = _tr_cached_patchscope_tokens(
                    str(results_root.resolve()),
                    dataset_dir_name,
                    int(layer_index),
                    int(position),
                    variant,
                )
            else:
                assert False, f"Unknown token_source: {token_source}"
            pct = _tr_compute_weighted_percentage(rec, np.asarray(probs, dtype=np.float32))
    else:
        assert token_source == "patchscope"
        if "weighted_filtered_percentage" in rec:
            pct = float(rec["weighted_filtered_percentage"])
        else:
            labels: List[str] = list(rec["labels"])  # type: ignore[assignment]
            mask = rec.get("unsupervised_filter", None)
            assert isinstance(mask, list) and len(mask) == len(labels)
            if not any(mask):
                pct = 0.0
            else:
                _tokens_all, _sel, probs = _tr_cached_patchscope_tokens(
                    str(results_root.resolve()),
                    dataset_dir_name,
                    int(layer_index),
                    int(position),
                    variant,
                )
                filtered_probs = np.asarray([w for m, w in zip(mask, probs.tolist()) if m], dtype=np.float32)
                assert filtered_probs.ndim == 1 and filtered_probs.size > 0
                filtered_labels = [lbl for m, lbl in zip(mask, labels) if m]
                pct = _tr_compute_weighted_percentage({"labels": filtered_labels}, filtered_probs)

    assert 0.0 <= pct <= 1.0
    return float(pct)


def plot_dual_axis_relevance_similarity(
    entries: List[Tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str] = None,
    variant: str = "difference",
    token_source: str = "patchscope",
    filtered: bool = False,
    weighted: bool = False,
    positions: Optional[List[int]] = None,
    pos_start: int = 0, 
    pos_end: int = 20,
    config_path: str = "configs/config.yaml",
    embedding_model_id: str = _DEFAULT_EMBEDDING_MODEL_ID,
    finetune_num_samples: int = _DEFAULT_FINETUNE_NUM_SAMPLES,
    batch_size: int = 64,
    figsize: Tuple[float, float] = (8, 5.5),
    font_size: int = 20,
    save_path: Optional[str] = None,
    log_x: bool = False,
) -> None:
    """Plot per-position mean±std for two metrics across entries on dual y-axes.

    - Left y-axis: Fraction Relevant Tokens (from token relevance outputs)
    - Right y-axis: Pairwise cosine similarity between Steered generations and Finetuning texts

    entries: list of (model, layer_index, organism)
    """
    assert isinstance(entries, list) and len(entries) > 0

    if positions is None:
        assert isinstance(pos_start, int) and isinstance(pos_end, int)
        assert pos_start <= pos_end
        positions = list(range(int(pos_start), int(pos_end) + 1))
    assert len(positions) > 0 and all(isinstance(p, int) for p in positions)

    plt.rcParams.update({"font.size": font_size})

    embedder = SentenceTransformer(
        embedding_model_id,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    finetune_mat_cache: Dict[Tuple[str, int], np.ndarray] = {}

    # Group entries by model and validate single layer per model
    model_to_items: Dict[str, List[Tuple[int, str]]] = {}
    for model, layer_index, organism in entries:
        model_to_items.setdefault(model, []).append((int(layer_index), str(organism)))

    model_to_layer: Dict[str, int] = {}
    for model, items in model_to_items.items():
        layers = sorted({li for li, _ in items})
        assert len(layers) == 1, f"Multiple layers for model {model}: {layers}"
        model_to_layer[model] = int(layers[0])

    # Storage for per-model arrays (list over entries for that model)
    model_to_rel_arrays: Dict[str, List[np.ndarray]] = {}
    model_to_sim_arrays: Dict[str, List[np.ndarray]] = {}

    for model, layer_index, organism in entries:
        overrides = [f"organism={organism}", f"model={model}", "infrastructure=mats_cluster_paper"]
        cfg = load_hydra_config(config_path, *overrides)

        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir(), f"Results root not found: {results_root}"
        selected_ds_dir = _tr_select_dataset_dir(results_root, int(layer_index), dataset_dir_name)
        ds_name = selected_ds_dir.name

        # Finetune embeddings (cached by dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(org_cfg, "training_dataset"), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_mat_cache:
            ft_texts = _eg_sample_finetune_texts(cfg, num_samples=int(finetune_num_samples))
            X_ft, labels_ft = _eg_embed_with_model(embedder, embedding_model_id, {"Finetune": ft_texts}, batch_size=batch_size)
            ft_mat = _eg_group_matrix(X_ft, labels_ft, "Finetune")
            assert ft_mat.ndim == 2 and ft_mat.shape[0] == len(ft_texts)
            finetune_mat_cache[ft_key] = ft_mat
        ft_mat = finetune_mat_cache[ft_key]
        assert isinstance(ft_mat, np.ndarray) and ft_mat.ndim == 2

        rel_vals: List[float] = []
        sim_vals: List[float] = []

        for pos in positions:
            # Fraction relevant tokens
            pct = _compute_relevance_percentage(
                results_root=results_root,
                layer_index=int(layer_index),
                dataset_dir_name=ds_name,
                position=int(pos),
                variant=variant,
                token_source=token_source,
                filtered=filtered,
                weighted=weighted,
                cfg=cfg,
            )
            rel_vals.append(float(pct))

            # Steered <> Finetune cosine similarity (mean pairwise cos-sim)
            steering_dir = selected_ds_dir / "steering" / f"position_{int(pos)}"
            generations_path = steering_dir / "generations.jsonl"
            assert generations_path.exists() and generations_path.is_file(), f"Generations file not found: {generations_path}"
            _prompts, steered_texts, _unsteered_texts = _eg_load_generations(generations_path)
            X_s, labels_s = _eg_embed_with_model(embedder, embedding_model_id, {"Steered": steered_texts}, batch_size=batch_size)
            steered_mat = _eg_group_matrix(X_s, labels_s, "Steered")
            mean_dist, _median, _std, _n = _eg_cosine_distance_stats(steered_mat, ft_mat)
            cs = 1.0 - float(mean_dist)
            sim_vals.append(cs)

        rel_arr = np.asarray(rel_vals, dtype=np.float32)
        sim_arr = np.asarray(sim_vals, dtype=np.float32)
        assert rel_arr.ndim == 1 and rel_arr.shape[0] == len(positions)
        assert sim_arr.ndim == 1 and sim_arr.shape[0] == len(positions)

        model_to_rel_arrays.setdefault(model, []).append(rel_arr)
        model_to_sim_arrays.setdefault(model, []).append(sim_arr)

    x = np.array(positions, dtype=int)

    fig, ax_left = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax_right = ax_left.twinx()

    # Color per model
    unique_models = sorted(list(model_to_items.keys()))
    cmap = plt.get_cmap("tab10")
    model_to_color = {m: cmap(i % 10) for i, m in enumerate(unique_models)}

    # Style per metric
    rel_style = dict(linestyle="-", marker="o")
    sim_style = dict(linestyle="--", marker="s")

    # Plot per model
    for model in unique_models:
        rel_list = model_to_rel_arrays.get(model, [])
        sim_list = model_to_sim_arrays.get(model, [])
        assert len(rel_list) >= 1 and len(sim_list) >= 1
        Rm = np.stack(rel_list, axis=0)
        Sm = np.stack(sim_list, axis=0)
        assert Rm.ndim == 2 and Rm.shape[1] == len(positions)
        assert Sm.ndim == 2 and Sm.shape[1] == len(positions)
        rel_mean = Rm.mean(axis=0).astype(np.float32)
        rel_std = Rm.std(axis=0).astype(np.float32)
        sim_mean = Sm.mean(axis=0).astype(np.float32)
        sim_std = Sm.std(axis=0).astype(np.float32)

        color = model_to_color[model]

        # Left axis: Frac Relevant Tokens
        ax_left.plot(x, rel_mean, color=color, linewidth=2.0, **rel_style)
        ax_left.fill_between(x, rel_mean - rel_std, rel_mean + rel_std, color=color, alpha=0.12)

        # Right axis: Steered <> Finetune cosine similarity
        ax_right.plot(x, sim_mean, color=color, linewidth=2.0, **sim_style)
        ax_right.fill_between(x, sim_mean - sim_std, sim_mean + sim_std, color=color, alpha=0.10)

    ax_left.set_ylabel("Frac. Relevant Tokens")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.grid(True, linestyle=":", alpha=0.3)
    ax_right.set_ylabel("Steered$\\Leftrightarrow$Finetune (cos-sim)")
    ax_right.set_ylim(0.0, 1.0)


    ax_left.set_xlabel("Position")
    ax_left.set_xticks(x)
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))

    if log_x:
        ax_left.set_xscale("log", base=2)
        ax_right.set_xscale("log", base=2)

    # Legends: (1) metric styles, (2) model colors
    metric_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, **rel_style, label="Frac. Relevant Tokens"),
        Line2D([0], [0], color="black", linewidth=2.0, **sim_style, label="Steered$\\Leftrightarrow$Finetune (cos-sim)"),
    ]
    metric_legend = ax_left.legend(handles=metric_handles, loc="upper left", frameon=True, fontsize=int(font_size * 0.8))

    model_handles: List[Line2D] = []
    for m in unique_models:
        disp = _tr_model_display_name(m)
        h = Line2D([0], [0], color=model_to_color[m], linewidth=2.0, linestyle="-", label=disp)
        model_handles.append(h)
    ax_left.legend(handles=model_handles, loc="upper right", frameon=True, ncol=1, fontsize=int(font_size * 0.8))
    ax_left.add_artist(metric_legend)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def _lighten_color(color, amount: float = 0.4):
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    amount = float(amount)
    return (1 - amount) * r + amount * 1.0, (1 - amount) * g + amount * 1.0, (1 - amount) * b + amount * 1.0


def plot_pair_comparison_relevance_and_similarity(
    pair: tuple[tuple[str, int, str], tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str] = None,
    token_source: str = "patchscope",
    filtered: bool = False,
    weighted: bool = False,
    variant_list: Optional[list[str]] = None,
    positions: Optional[list[int]] = None,
    config_path: str = "configs/config.yaml",
    embedding_model_id: str = _DEFAULT_EMBEDDING_MODEL_ID,
    finetune_num_samples: int = _DEFAULT_FINETUNE_NUM_SAMPLES,
    batch_size: int = 64,
    figsize: Tuple[float, float] = (10.5, 4.2),
    font_size: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Compare two (model, layer, organism) specs via grouped bars.

    Left subplot: token relevance maxima over positions for variants (difference/base/ft).
    Right subplot: cosine similarity St-FT and USt-FT maxima over positions.
    Bars are grouped in pairs (spec A vs spec B) with two shades of the same color.
    """
    assert isinstance(pair, tuple) and len(pair) == 2
    spec_a, spec_b = pair
    assert len(spec_a) == 3 and len(spec_b) == 3

    if variant_list is None:
        variant_list = ["difference", "base", "ft"]
    assert all(v in ("difference", "base", "ft") for v in variant_list)

    if positions is None:
        positions = [0, 1, 2, 3, 4]
    assert isinstance(positions, list) and len(positions) > 0 and all(isinstance(p, int) for p in positions)

    plt.rcParams.update({"font.size": font_size})

    # Prepare embedding model once
    embedder = SentenceTransformer(
        embedding_model_id,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    # Caches keyed by finetune dataset id and sample size
    finetune_mat_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _prepare_spec(spec: tuple[str, int, str]):
        model, layer_index, organism = spec
        overrides = [f"organism={organism}", f"model={model}", "infrastructure=mats_cluster_paper"]
        cfg = load_hydra_config(config_path, *overrides)
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir(), f"Results root not found: {results_root}"
        selected_ds_dir = _tr_select_dataset_dir(results_root, int(layer_index), dataset_dir_name)
        ds_name = selected_ds_dir.name
        # Finetune embeddings (cached by dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(org_cfg, "training_dataset"), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_mat_cache:
            ft_texts = _eg_sample_finetune_texts(cfg, num_samples=int(finetune_num_samples))
            X_ft, labels_ft = _eg_embed_with_model(embedder, embedding_model_id, {"Finetune": ft_texts}, batch_size=batch_size)
            ft_mat = _eg_group_matrix(X_ft, labels_ft, "Finetune")
            assert ft_mat.ndim == 2 and ft_mat.shape[0] == len(ft_texts)
            finetune_mat_cache[ft_key] = ft_mat
        ft_mat = finetune_mat_cache[ft_key]
        assert isinstance(ft_mat, np.ndarray) and ft_mat.ndim == 2
        return cfg, results_root, selected_ds_dir, ds_name, ft_mat

    # Compute values per spec
    vals_relevance: Dict[str, list[float]] = {v: [0.0, 0.0] for v in variant_list}
    vals_cosim: Dict[str, list[float]] = {"St-FT": [0.0, 0.0], "USt-FT": [0.0, 0.0]}

    labels_spec: list[str] = []
    for idx, spec in enumerate([spec_a, spec_b]):
        model, layer_index, organism = spec
        disp_model = _tr_model_display_name(model)
        labels_spec.append(f"{disp_model}:{organism}")

        cfg, results_root, selected_ds_dir, ds_name, ft_mat = _prepare_spec(spec)

        # Token relevance maxima per variant
        for variant in variant_list:
            vals: list[float] = []
            for pos in positions:
                pct = _compute_relevance_percentage(
                    results_root=results_root,
                    layer_index=int(layer_index),
                    dataset_dir_name=ds_name,
                    position=int(pos),
                    variant=variant,
                    token_source=token_source,
                    filtered=filtered,
                    weighted=weighted,
                    cfg=cfg,
                )
                vals.append(float(pct))
            assert len(vals) > 0
            vals_relevance[variant][idx] = float(np.max(np.asarray(vals, dtype=np.float32)))

        # Cosine similarity maxima: Steered and Unsteered vs Finetune
        st_vals: list[float] = []
        ust_vals: list[float] = []
        for pos in positions:
            steering_dir = selected_ds_dir / "steering" / f"position_{int(pos)}"
            generations_path = steering_dir / "generations.jsonl"
            assert generations_path.exists() and generations_path.is_file(), f"Generations file not found: {generations_path}"
            _prompts, steered_texts, unsteered_texts = _eg_load_generations(generations_path)
            X_s, labels_s = _eg_embed_with_model(embedder, embedding_model_id, {"Steered": steered_texts, "Unsteered": unsteered_texts}, batch_size=batch_size)
            steered_mat = _eg_group_matrix(X_s, labels_s, "Steered")
            unsteered_mat = _eg_group_matrix(X_s, labels_s, "Unsteered")
            mean_dist_st, *_ = _eg_cosine_distance_stats(steered_mat, ft_mat)
            mean_dist_ust, *_ = _eg_cosine_distance_stats(unsteered_mat, ft_mat)
            st_vals.append(1.0 - float(mean_dist_st))
            ust_vals.append(1.0 - float(mean_dist_ust))
        assert len(st_vals) > 0 and len(ust_vals) > 0
        vals_cosim["St-FT"][idx] = float(np.max(np.asarray(st_vals, dtype=np.float32)))
        vals_cosim["USt-FT"][idx] = float(np.max(np.asarray(ust_vals, dtype=np.float32)))

    # Plotting: two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Colors per metric group aligned with other plots
    TR_VARIANT_COLORS: Dict[str, str] = {
        "difference": "#1f77b4",
        "ft": "#2ca02c",
        "base": "#ff7f0e",
    }
    COSIM_COLORS: Dict[str, str] = {
        "St-FT": "#1f77b4",
        "USt-FT": "#1f77b4",
    }
    group_hatches: Tuple[str, str] = ("/", "\\")
    # Left: token relevance
    x_left = np.arange(len(variant_list), dtype=float)
    width = 0.35
    for i, variant in enumerate(variant_list):
        color = TR_VARIANT_COLORS[variant]
        color_b = _lighten_color(color, 0.45)
        yA, yB = vals_relevance[variant]
        ax_left.bar(x_left[i] - width / 2.0, yA, width=width, color=color, hatch=group_hatches[0], label=None)
        ax_left.bar(x_left[i] + width / 2.0, yB, width=width, color=color_b, hatch=group_hatches[1], label=None)
    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels([v.capitalize() if v != "ft" else "FT" for v in variant_list])
    ax_left.set_ylabel("Frac. Relevant Tokens (max over pos 0-4)")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Right: cosine similarity groups
    cos_keys = ["St-FT", "USt-FT"]
    x_right = np.arange(len(cos_keys), dtype=float)
    for i, key in enumerate(cos_keys):
        color = COSIM_COLORS[key]
        color_b = _lighten_color(color, 0.45)
        yA, yB = vals_cosim[key]
        ax_right.bar(x_right[i] - width / 2.0, yA, width=width, color=color, hatch=group_hatches[0], label=None)
        ax_right.bar(x_right[i] + width / 2.0, yB, width=width, color=color_b, hatch=group_hatches[1], label=None)
    ax_right.set_xticks(x_right)
    ax_right.set_xticklabels(["Steered$\\Leftrightarrow$Finetune", "Unsteered$\\Leftrightarrow$Finetune"], rotation=0)
    ax_right.set_ylabel("Pairwise Cos-Sim (max over pos 0-4)")
    ax_right.set_ylim(0.0, 1.0)
    ax_right.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Shared legend describing groups: gray with hatching
    gray = "#bdbdbd"
    handles = [
        Patch(facecolor=gray, edgecolor="black", hatch=group_hatches[0], label=labels_spec[0]),
        Patch(facecolor=gray, edgecolor="black", hatch=group_hatches[1], label=labels_spec[1]),
    ]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, fontsize=int(font_size * 0.8))
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_groups_comparison_relevance_and_similarity(
    group_a: List[Tuple[str, int, str]],
    group_b: List[Tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str] = None,
    token_source: str = "patchscope",
    filtered: bool = False,
    weighted: bool = False,
    variant_list: Optional[List[str]] = None,
    positions: Optional[List[int]] = None,
    config_path: str = "configs/config.yaml",
    embedding_model_id: str = _DEFAULT_EMBEDDING_MODEL_ID,
    finetune_num_samples: int = _DEFAULT_FINETUNE_NUM_SAMPLES,
    batch_size: int = 64,
    figsize: Tuple[float, float] = (10.5, 4.2),
    legend_loc: str = "upper center",
    cosim_y_range: Tuple[float, float] = (0.0, 1.0),
    cosim_x_label_rotation: int = 0,
    toks_y_range: Tuple[float, float] = (0.0, 1.0),
    font_size: int = 20,
    save_path: Optional[str] = None,
    group_labels: Optional[List[str]] = None,
    additional_groups: Optional[List[List[Tuple[str, int, str]]]] = None,
) -> None:
    """Grouped comparison across N (≤4) sets of (model, layer, organism).

    Left: token relevance (variants), Right: cosine similarity (St-FT, USt-FT).
    Each metric shows one bar per group with std error bars if |group|>1.
    Values are maxima over positions (default 0..4) per spec, then aggregated by mean±std.
    """
    assert isinstance(group_a, list) and len(group_a) >= 1
    assert isinstance(group_b, list) and len(group_b) >= 1
    groups_list: List[List[Tuple[str, int, str]]] = [group_a, group_b]
    if additional_groups is not None:
        assert isinstance(additional_groups, list)
        for g in additional_groups:
            assert isinstance(g, list) and len(g) >= 1
            groups_list.append(g)
    assert 2 <= len(groups_list) <= 4
    for g in groups_list:
        for item in g:
            assert isinstance(item, tuple) and len(item) == 3

    if variant_list is None:
        variant_list = ["difference", "base", "ft"]
    assert len(variant_list) >= 1 and all(v in ("difference", "base", "ft") for v in variant_list)

    if positions is None:
        positions = [0, 1, 2, 3, 4]
    assert isinstance(positions, list) and len(positions) > 0 and all(isinstance(p, int) for p in positions)

    if group_labels is None:
        tmp_labels: List[str] = []
        for g in groups_list:
            m0, _l0, o0 = g[0]
            tmp_labels.append(f"{_tr_model_display_name(m0)}:{o0} (n={len(g)})")
        group_labels = tmp_labels
    assert isinstance(group_labels, list) and len(group_labels) == len(groups_list)

    plt.rcParams.update({"font.size": font_size})

    embedder = SentenceTransformer(
        embedding_model_id,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    finetune_mat_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _prepare_spec(spec: tuple[str, int, str]):
        model, layer_index, organism = spec
        overrides = [f"organism={organism}", f"model={model}", "infrastructure=mats_cluster_paper"]
        cfg = load_hydra_config(config_path, *overrides)
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir(), f"Results root not found: {results_root}"
        selected_ds_dir = _tr_select_dataset_dir(results_root, int(layer_index), dataset_dir_name)
        ds_name = selected_ds_dir.name
        org_cfg = cfg.organism
        assert hasattr(org_cfg, "training_dataset"), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_mat_cache:
            ft_texts = _eg_sample_finetune_texts(cfg, num_samples=int(finetune_num_samples))
            X_ft, labels_ft = _eg_embed_with_model(embedder, embedding_model_id, {"Finetune": ft_texts}, batch_size=batch_size)
            ft_mat = _eg_group_matrix(X_ft, labels_ft, "Finetune")
            assert ft_mat.ndim == 2 and ft_mat.shape[0] == len(ft_texts)
            finetune_mat_cache[ft_key] = ft_mat
        ft_mat = finetune_mat_cache[ft_key]
        assert isinstance(ft_mat, np.ndarray) and ft_mat.ndim == 2
        return cfg, results_root, selected_ds_dir, ds_name, ft_mat

    def _spec_metrics(spec: tuple[str, int, str]) -> tuple[Dict[str, float], Dict[str, float]]:
        model, layer_index, _organism = spec
        cfg, results_root, selected_ds_dir, ds_name, ft_mat = _prepare_spec(spec)
        # Token relevance per variant (max over positions)
        rel_out: Dict[str, float] = {}
        for variant in variant_list:  # type: ignore[arg-type]
            vals: List[float] = []
            for pos in positions:  # type: ignore[union-attr]
                pct = _compute_relevance_percentage(
                    results_root=results_root,
                    layer_index=int(layer_index),
                    dataset_dir_name=ds_name,
                    position=int(pos),
                    variant=variant,
                    token_source=token_source,
                    filtered=filtered,
                    weighted=weighted,
                    cfg=cfg,
                )
                vals.append(float(pct))
            assert len(vals) > 0
            rel_out[variant] = float(np.max(np.asarray(vals, dtype=np.float32)))
        # Cosine similarity (max over positions)
        st_vals: List[float] = []
        ust_vals: List[float] = []
        for pos in positions:  # type: ignore[union-attr]
            steering_dir = selected_ds_dir / "steering" / f"position_{int(pos)}"
            generations_path = steering_dir / "generations.jsonl"
            assert generations_path.exists() and generations_path.is_file(), f"Generations file not found: {generations_path}"
            _prompts, steered_texts, unsteered_texts = _eg_load_generations(generations_path)
            X_s, labels_s = _eg_embed_with_model(embedder, embedding_model_id, {"Steered": steered_texts, "Unsteered": unsteered_texts}, batch_size=batch_size)
            steered_mat = _eg_group_matrix(X_s, labels_s, "Steered")
            unsteered_mat = _eg_group_matrix(X_s, labels_s, "Unsteered")
            mean_dist_st, *_ = _eg_cosine_distance_stats(steered_mat, ft_mat)
            mean_dist_ust, *_ = _eg_cosine_distance_stats(unsteered_mat, ft_mat)
            st_vals.append(1.0 - float(mean_dist_st))
            ust_vals.append(1.0 - float(mean_dist_ust))
        assert len(st_vals) > 0 and len(ust_vals) > 0
        cos_out = {
            "St-FT": float(np.max(np.asarray(st_vals, dtype=np.float32))),
            "USt-FT": float(np.max(np.asarray(ust_vals, dtype=np.float32))),
        }
        return rel_out, cos_out

    # Aggregate over groups
    def _aggregate_group(g: List[Tuple[str, int, str]]):
        rel_vals_by_variant: Dict[str, List[float]] = {v: [] for v in variant_list}  # type: ignore[list-item]
        cos_vals_by_key: Dict[str, List[float]] = {"St-FT": [], "USt-FT": []}
        for spec in g:
            rel_out, cos_out = _spec_metrics(spec)
            for v in variant_list:  # type: ignore[arg-type]
                rel_vals_by_variant[v].append(float(rel_out[v]))
            for k in ["St-FT", "USt-FT"]:
                cos_vals_by_key[k].append(float(cos_out[k]))
        return rel_vals_by_variant, cos_vals_by_key

    # Aggregate for each group
    rel_groups: List[Dict[str, List[float]]] = []
    cos_groups: List[Dict[str, List[float]]] = []
    for g in groups_list:
        rel_g, cos_g = _aggregate_group(g)
        rel_groups.append(rel_g)
        cos_groups.append(cos_g)

    # Means and stds per group
    num_groups = len(groups_list)
    group_sizes = [len(g) for g in groups_list]

    rel_stats: Dict[str, Tuple[List[float], List[float]]] = {}
    for v in variant_list:  # type: ignore[arg-type]
        means_v: List[float] = []
        stds_v: List[float] = []
        for gi in range(num_groups):
            arr = np.asarray(rel_groups[gi][v], dtype=np.float32)
            means_v.append(float(arr.mean()))
            stds_v.append(float(arr.std()))
        rel_stats[v] = (means_v, stds_v)

    cos_keys = ["St-FT", "USt-FT"]
    cos_stats: Dict[str, Tuple[List[float], List[float]]] = {}
    for k in cos_keys:
        means_k: List[float] = []
        stds_k: List[float] = []
        for gi in range(num_groups):
            arr = np.asarray(cos_groups[gi][k], dtype=np.float32)
            means_k.append(float(arr.mean()))
            stds_k.append(float(arr.std()))
        cos_stats[k] = (means_k, stds_k)

    # Plot
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    group_hatches: Tuple[str, str] = ("/", "\\")
    TR_VARIANT_COLORS: Dict[str, str] = {
        "difference": "#1f77b4",
        "ft": "#2ca02c",
        "base": "#ff7f0e",
    }
    COSIM_COLORS: Dict[str, str] = {
        "St-FT": "#1f77b4",
        "USt-FT": "#ff7f0e",
    }
    # Hatches and width for up to 4 groups
    all_hatches: List[str] = ["/", ".", "\\", "x"]
    group_hatches = all_hatches[:num_groups]
    width = 0.8 / float(num_groups)

    # Left: token relevance
    x_left = np.arange(len(variant_list), dtype=float)
    for i, v in enumerate(variant_list):  # type: ignore[arg-type]
        base_color = TR_VARIANT_COLORS[v]
        lighten_vals = np.linspace(0.0, 0.6, num_groups, dtype=np.float32)
        means_v, stds_v = rel_stats[v]
        for gi in range(num_groups):
            color_g = _lighten_color(base_color, float(lighten_vals[gi]))
            x_pos = x_left[i] + (gi - (num_groups - 1) / 2.0) * width
            yerr = stds_v[gi] if group_sizes[gi] > 1 else None
            ax_left.bar(
                x_pos,
                means_v[gi],
                width=width,
                color=color_g,
                yerr=yerr,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
                hatch=group_hatches[gi],
            )
    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels([v.capitalize() if v != "ft" else "FT" for v in variant_list])
    ax_left.set_ylabel("Frac. Relevant Tokens")
    ax_left.set_ylim(toks_y_range)
    ax_left.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Right: cosine similarity
    x_right = np.arange(len(cos_keys), dtype=float)
    for i, k in enumerate(cos_keys):
        base_color = COSIM_COLORS[k]
        lighten_vals = np.linspace(0.0, 0.6, num_groups, dtype=np.float32)
        means_k, stds_k = cos_stats[k]
        for gi in range(num_groups):
            color_g = _lighten_color(base_color, float(lighten_vals[gi]))
            x_pos = x_right[i] + (gi - (num_groups - 1) / 2.0) * width
            yerr = stds_k[gi] if group_sizes[gi] > 1 else None
            ax_right.bar(
                x_pos,
                means_k[gi],
                width=width,
                color=color_g,
                yerr=yerr,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
                hatch=group_hatches[gi],
            )
    ax_right.set_xticks(x_right)
    ax_right.set_xticklabels(["Steered$\\Leftrightarrow$Finetune", "Unsteered$\\Leftrightarrow$Finetune"], rotation=cosim_x_label_rotation, fontsize=int(font_size * 0.8))
    ax_right.set_ylabel("Pairwise Cos-Sim")
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.set_ylim(cosim_y_range)
    ax_right.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Legend for group labels in gray with distinct hatching per group
    gray = "#bdbdbd"
    handles = []
    for gi in range(num_groups):
        handles.append(Patch(facecolor=gray, edgecolor="black", hatch=group_hatches[gi], label=str(group_labels[gi])))
    leg = fig.legend(handles=handles, loc=legend_loc, ncol=num_groups, frameon=True, fontsize=int(font_size * 0.8))
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()

def plot_relevance_over_positions_by_model(
    entries: List[Tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str] = None,
    variant: str = "difference",
    token_source: str = "patchscope",
    filtered: bool = False,
    weighted: bool = False,
    positions: Optional[List[int]] = None,
    pos_start: int = 0,
    pos_end: int = 20,
    config_path: str = "configs/config.yaml",
    figsize: Tuple[float, float] = (8, 5.5),
    font_size: int = 20,
    save_path: Optional[str] = None,
    log_x: bool = False,
    use_log_nums: bool = True,
) -> None:
    """Plot per-model Frac. Relevant Tokens over positions (mean±std across entries)."""
    assert isinstance(entries, list) and len(entries) > 0
    if positions is None:
        assert isinstance(pos_start, int) and isinstance(pos_end, int)
        assert pos_start <= pos_end
        positions = list(range(int(pos_start), int(pos_end) + 1))
    assert len(positions) > 0 and all(isinstance(p, int) for p in positions)

    plt.rcParams.update({"font.size": font_size})

    # Group entries by model and validate single layer per model
    model_to_items: Dict[str, List[Tuple[int, str]]] = {}
    for model, layer_index, organism in entries:
        model_to_items.setdefault(model, []).append((int(layer_index), str(organism)))
    model_to_layer: Dict[str, int] = {}
    for model, items in model_to_items.items():
        layers = sorted({li for li, _ in items})
        assert len(layers) == 1
        model_to_layer[model] = int(layers[0])

    # Color per model
    unique_models = sorted(list(model_to_items.keys()))
    cmap = plt.get_cmap("tab10")
    model_to_color = {m: cmap(i % 10) for i, m in enumerate(unique_models)}

    model_to_rel_arrays: Dict[str, List[np.ndarray]] = {}

    for model, layer_index, organism in entries:
        overrides = [f"organism={organism}", f"model={model}", "infrastructure=mats_cluster_paper"]
        cfg = load_hydra_config(config_path, *overrides)
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir()
        selected_ds_dir = _tr_select_dataset_dir(results_root, int(layer_index), dataset_dir_name)
        ds_name = selected_ds_dir.name

        vals: List[float] = []
        for pos in positions:
            pct = _compute_relevance_percentage(
                results_root=results_root,
                layer_index=int(layer_index),
                dataset_dir_name=ds_name,
                position=int(pos),
                variant=variant,
                token_source=token_source,
                filtered=filtered,
                weighted=weighted,
                cfg=cfg,
            )
            vals.append(float(pct))
        arr = np.asarray(vals, dtype=np.float32)
        assert arr.ndim == 1 and arr.shape[0] == len(positions)
        model_to_rel_arrays.setdefault(model, []).append(arr)

    x = np.array(positions, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    for model in unique_models:
        rel_list = model_to_rel_arrays.get(model, [])
        assert len(rel_list) >= 1
        M = np.stack(rel_list, axis=0)
        assert M.ndim == 2 and M.shape[1] == len(positions)
        mean = M.mean(axis=0).astype(np.float32)
        std = M.std(axis=0).astype(np.float32)
        color = model_to_color[model]
        ax.plot(x, mean, color=color, linewidth=2.0, linestyle="-", marker="o")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel("Position")
    ax.set_ylabel("Frac. Relevant Tokens")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":", alpha=0.3)

    if log_x:
        ax.set_xscale("log", base=2)

        if use_log_nums:
            # For log scale, use powers of 2 but display as regular numbers
            max_x = max(x)
            log_ticks = []
            power = 0
            while 2**power <= max_x+1:
                log_ticks.append(2**power)
                power += 1
            ax.set_xticks(log_ticks)
            ax.set_xticklabels([str(t) for t in log_ticks])

    # Model legend
    handles: List[Line2D] = []
    for m in unique_models:
        disp = _tr_model_display_name(m)
        handles.append(Line2D([0], [0], color=model_to_color[m], linewidth=2.0, linestyle="-", label=disp))
    ax.legend(handles=handles, frameon=True, ncol=1, fontsize=int(font_size * 0.8))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_similarity_over_positions_by_model(
    entries: List[Tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str] = None,
    positions: Optional[List[int]] = None,
    pos_start: int = 0,
    pos_end: int = 20,
    config_path: str = "configs/config.yaml",
    embedding_model_id: str = _DEFAULT_EMBEDDING_MODEL_ID,
    finetune_num_samples: int = _DEFAULT_FINETUNE_NUM_SAMPLES,
    batch_size: int = 64,
    figsize: Tuple[float, float] = (8, 5.5),
    font_size: int = 20,
    save_path: Optional[str] = None,
    log_x: bool = False,
    use_log_nums: bool = True,
) -> None:
    """Plot per-model Steered↔Finetune cosine similarity over positions (mean±std)."""
    assert isinstance(entries, list) and len(entries) > 0
    if positions is None:
        assert isinstance(pos_start, int) and isinstance(pos_end, int)
        assert pos_start <= pos_end
        positions = list(range(int(pos_start), int(pos_end) + 1))
    assert len(positions) > 0 and all(isinstance(p, int) for p in positions)

    plt.rcParams.update({"font.size": font_size})

    # Group entries by model and validate single layer per model
    model_to_items: Dict[str, List[Tuple[int, str]]] = {}
    for model, layer_index, organism in entries:
        model_to_items.setdefault(model, []).append((int(layer_index), str(organism)))
    model_to_layer: Dict[str, int] = {}
    for model, items in model_to_items.items():
        layers = sorted({li for li, _ in items})
        assert len(layers) == 1
        model_to_layer[model] = int(layers[0])

    # Color per model
    unique_models = sorted(list(model_to_items.keys()))
    cmap = plt.get_cmap("tab10")
    model_to_color = {m: cmap(i % 10) for i, m in enumerate(unique_models)}

    embedder = SentenceTransformer(
        embedding_model_id,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    finetune_mat_cache: Dict[Tuple[str, int], np.ndarray] = {}
    model_to_sim_arrays: Dict[str, List[np.ndarray]] = {}

    for model, layer_index, organism in entries:
        overrides = [f"organism={organism}", f"model={model}", "infrastructure=mats_cluster_paper"]
        cfg = load_hydra_config(config_path, *overrides)
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir()
        selected_ds_dir = _tr_select_dataset_dir(results_root, int(layer_index), dataset_dir_name)

        org_cfg = cfg.organism
        assert hasattr(org_cfg, "training_dataset")
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_mat_cache:
            ft_texts = _eg_sample_finetune_texts(cfg, num_samples=int(finetune_num_samples))
            X_ft, labels_ft = _eg_embed_with_model(embedder, embedding_model_id, {"Finetune": ft_texts}, batch_size=batch_size)
            ft_mat = _eg_group_matrix(X_ft, labels_ft, "Finetune")
            assert ft_mat.ndim == 2 and ft_mat.shape[0] == len(ft_texts)
            finetune_mat_cache[ft_key] = ft_mat
        ft_mat = finetune_mat_cache[ft_key]

        vals: List[float] = []
        for pos in positions:
            steering_dir = selected_ds_dir / "steering" / f"position_{int(pos)}"
            generations_path = steering_dir / "generations.jsonl"
            assert generations_path.exists() and generations_path.is_file()
            _prompts, steered_texts, _unsteered_texts = _eg_load_generations(generations_path)
            X_s, labels_s = _eg_embed_with_model(embedder, embedding_model_id, {"Steered": steered_texts}, batch_size=batch_size)
            steered_mat = _eg_group_matrix(X_s, labels_s, "Steered")
            mean_dist, _median, _std, _n = _eg_cosine_distance_stats(steered_mat, ft_mat)
            cs = 1.0 - float(mean_dist)
            vals.append(cs)
        arr = np.asarray(vals, dtype=np.float32)
        assert arr.ndim == 1 and arr.shape[0] == len(positions)
        model_to_sim_arrays.setdefault(model, []).append(arr)

    x = np.array(positions, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    for model in unique_models:
        sim_list = model_to_sim_arrays.get(model, [])
        assert len(sim_list) >= 1
        M = np.stack(sim_list, axis=0)
        assert M.ndim == 2 and M.shape[1] == len(positions)
        mean = M.mean(axis=0).astype(np.float32)
        std = M.std(axis=0).astype(np.float32)
        color = model_to_color[model]
        ax.plot(x, mean, color=color, linewidth=2.0, linestyle="--", marker="s")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)

    ax.set_xlabel("Position")
    ax.set_ylabel("Steered$\\Leftrightarrow$Finetune (cos-sim)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":", alpha=0.3)

    if log_x:
        ax.set_xscale("log", base=2)
        if use_log_nums:
            # For log scale, use powers of 2 but display as regular numbers
            max_x = max(x)
            log_ticks = []
            power = 0
            while 2**power <= max_x+1:
                log_ticks.append(2**power)
                power += 1
            ax.set_xticks(log_ticks)
            ax.set_xticklabels([str(t) for t in log_ticks])

    # Model legend
    handles: List[Line2D] = []
    for m in unique_models:
        disp = _tr_model_display_name(m)
        handles.append(Line2D([0], [0], color=model_to_color[m], linewidth=2.0, linestyle="-", label=disp))
    ax.legend(handles=handles, frameon=True, ncol=1, fontsize=int(font_size * 0.8))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()

group_a = [
  ("qwen25_VL_3B_Instruct", 17, "adaptllm_biomed"),
#   ("qwen25_VL_3B_Instruct", 16, "adaptllm_food"),
#   ("qwen25_VL_3B_Instruct", 16, "adaptllm_remote_sensing"),
]
group_b = [
#   ("qwen25_VL_3B_Instruct", 16, "adaptllm_biomed"),
  ("qwen25_VL_3B_Instruct", 17, "adaptllm_food"),
#   ("qwen25_VL_3B_Instruct", 16, "adaptllm_remote_sensing"),
]

plot_groups_comparison_relevance_and_similarity(
  group_a,
  group_b,
  dataset_dir_name="fineweb-1m-sample",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  positions=[0,1,2,3,4],  # optional (defaults to first 5)
  config_path="configs/config.yaml",
  embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
  finetune_num_samples=500,
  batch_size=64,
  figsize=(10, 5.5),
  cosim_y_range=(0.0, 0.6),
  toks_y_range=(0.0, 0.9),
  legend_loc=(0.09, 0.88),
  font_size=20,
  group_labels=("Normal", "CAFT"),
  save_path="plots/tmp.pdf",
)
# %%
entries = [
  ("qwen3_1_7B", 13, "cake_bake"),
  ("qwen3_1_7B", 13, "kansas_abortion"),
  ("qwen3_1_7B", 13, "ignore_comment"),
  ("qwen3_1_7B", 13, "fda_approval"),
  ("qwen3_1_7B", 13, "roman_concrete"),

  ("llama32_1B_Instruct", 7, "cake_bake"),
  ("llama32_1B_Instruct", 7, "kansas_abortion"),
  ("llama32_1B_Instruct", 7, "ignore_comment"),
  ("llama32_1B_Instruct", 7, "fda_approval"),
  ("llama32_1B_Instruct", 7, "roman_concrete"),

  ("gemma3_1B", 12, "cake_bake"),
  ("gemma3_1B", 12, "kansas_abortion"),
  ("gemma3_1B", 12, "ignore_comment"),
  ("gemma3_1B", 12, "fda_approval"),
  ("gemma3_1B", 12, "roman_concrete"),
]
# %%
plot_dual_axis_relevance_similarity(
  entries,
  dataset_dir_name="fineweb-1m-sample",
  variant="difference",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  positions=[0,1,2,3,4,6,7,8,9,10,15,31,63,127],
  config_path="configs/config.yaml",
  save_path="plots/positions_dual_axis.pdf",
  log_x=True,
  font_size=22,
  figsize=(6.2, 4.5),
)
# %%
plot_relevance_over_positions_by_model(
  entries,
  dataset_dir_name="fineweb-1m-sample",
  variant="difference",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  font_size=22,
  figsize=(6.2, 4.5),
  positions=[0,1,2,3,4,6,7,8,9,10,15,31,63,127],
  log_x=True,
  config_path="configs/config.yaml",
  save_path="plots/relevance_over_positions_by_model.pdf",
  use_log_nums=True,
)
# %%
plot_similarity_over_positions_by_model(
  entries,
  dataset_dir_name="fineweb-1m-sample",
  figsize=(6.2, 4.5),
  font_size=22,
  positions=[0,1,2,3,4,6,7,8,9,10,15,31,63,127],
  log_x=True,
  config_path="configs/config.yaml",
  save_path="plots/similarity_over_positions_by_model.pdf",
)

# %%
group_a = [
  ("qwen3_1_7B", 13, "em_bad_medical_advice"),
  ("qwen3_1_7B", 13, "em_risky_financial_advice"),
  ("qwen3_1_7B", 13, "em_extreme_sports"),
]
group_b = [
  ("qwen3_1_7B", 13, "em_bad_medical_advice_mix1-1p0"),
  ("qwen3_1_7B", 13, "em_risky_financial_advice_mix1-1p0"),
  ("qwen3_1_7B", 13, "em_extreme_sports_mix1-1p0"),
  ]

plot_groups_comparison_relevance_and_similarity(
  group_a,
  group_b,
  dataset_dir_name="fineweb-1m-sample",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  positions=[0,1,2,3,4],  # optional (defaults to first 5)
  config_path="configs/config.yaml",
  embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
  finetune_num_samples=500,
  batch_size=64,
  figsize=(10, 5.5),
  cosim_y_range=(0.0, 0.6),
  toks_y_range=(0.0, 0.9),
  legend_loc=(0.09, 0.88),
  font_size=20,
  group_labels=("Normal", "Mix 1:1"),
  save_path="plots/mix_em.pdf",
)
# %%

# %%
group_a = [
  ("qwen3_1_7B", 13, "cake_bake"),
  ("qwen3_1_7B", 13, "kansas_abortion"),
  ("qwen3_1_7B", 13, "fda_approval"),

  ("llama32_1B_Instruct", 7, "cake_bake"),
  ("llama32_1B_Instruct", 7, "kansas_abortion"),
  ("llama32_1B_Instruct", 7, "fda_approval"),

  ("gemma3_1B", 12, "cake_bake"),
  ("gemma3_1B", 12, "kansas_abortion"),
  ("gemma3_1B", 12, "fda_approval"),
]
group_b = [
    ("qwen3_1_7B", 13, "cake_bake_CAFT"),
  ("qwen3_1_7B", 13, "kansas_abortion_CAFT"),
  ("qwen3_1_7B", 13, "fda_approval_CAFT"),

  ("llama32_1B_Instruct", 7, "cake_bake_CAFT"),
  ("llama32_1B_Instruct", 7, "kansas_abortion_CAFT"),
  ("llama32_1B_Instruct", 7, "fda_approval_CAFT"),

  ("gemma3_1B", 12, "cake_bake_CAFT"),
  ("gemma3_1B", 12, "kansas_abortion_CAFT"),
  ("gemma3_1B", 12, "fda_approval_CAFT"),
]

plot_groups_comparison_relevance_and_similarity(
  group_a,
  group_b,
  dataset_dir_name="fineweb-1m-sample",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  positions=[0,1,2,3,4],  # optional (defaults to first 5)
  config_path="configs/config.yaml",
  embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
  finetune_num_samples=500,
  batch_size=64,
  figsize=(10, 5.5),
  cosim_y_range=(0.0, 0.6),
  toks_y_range=(0.0, 0.9),
  legend_loc=(0.09, 0.88),
  font_size=20,
  group_labels=("Normal", "CAFT"),
  save_path="plots/caft.pdf",
)
# %%
group_a = [
  ("qwen3_1_7B", 13, "cake_bake"),
  ("qwen3_1_7B", 13, "kansas_abortion"),
  ("qwen3_1_7B", 13, "fda_approval"),

  ("llama32_1B_Instruct", 7, "cake_bake"),
  ("llama32_1B_Instruct", 7, "kansas_abortion"),
  ("llama32_1B_Instruct", 7, "fda_approval"),

  ("gemma3_1B", 12, "cake_bake"),
  ("gemma3_1B", 12, "kansas_abortion"),
  ("gemma3_1B", 12, "fda_approval"),
]
group_b = [
  ("qwen3_1_7B", 13, "cake_bake_mix1-1p0"),
  ("qwen3_1_7B", 13, "kansas_abortion_mix1-1p0"),
  ("qwen3_1_7B", 13, "fda_approval_mix1-1p0"),

  ("llama32_1B_Instruct", 7, "cake_bake_mix1-1p0"),
  ("llama32_1B_Instruct", 7, "kansas_abortion_mix1-1p0"),
  ("llama32_1B_Instruct", 7, "fda_approval_mix1-1p0"),

  ("gemma3_1B", 12, "cake_bake_mix1-1p0"),
  ("gemma3_1B", 12, "kansas_abortion_mix1-1p0"),
  ("gemma3_1B", 12, "fda_approval_mix1-1p0"),
]

plot_groups_comparison_relevance_and_similarity(
  group_a,
  group_b,
  dataset_dir_name="fineweb-1m-sample",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  positions=[0,1,2,3,4],  # optional (defaults to first 5)
  config_path="configs/config.yaml",
  embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
  finetune_num_samples=500,
  batch_size=64,
  figsize=(10, 5.5),
  cosim_y_range=(0.0, 0.6),
  toks_y_range=(0.0, 0.9),
  legend_loc=(0.09, 0.88),
  font_size=20,
  group_labels=("Normal", "Mix 1:1"),
  save_path="plots/sdf_mix1to1.pdf",
)
# %%
