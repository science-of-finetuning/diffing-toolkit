# %%
from __future__ import annotations
import sys

sys.path.append("scripts")
sys.path.append("..")
# %%
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors  # type: ignore[import-not-found]
from matplotlib.lines import Line2D  # type: ignore[import-not-found]

from src.utils.interactive import load_hydra_config
from src.diffing.methods.activation_difference_lens.util import dataset_dir_name

import scienceplots as _scienceplots  # type: ignore[import-not-found]

plt.style.use("science")

CONFIG_PATH = "configs/config.yaml"


# Consistent display names with visualize_grades.py
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
    "qwen3_1_7B_Base": "Qwen3 1.7B Base",
    "llama32_1B": "Llama3.2 1B Base",
}


def _model_display_name(model: str) -> str:
    name = MODEL_DISPLAY_NAMES.get(model, None)
    assert isinstance(name, str), f"Missing display name mapping for model: {model}"
    return name


def _results_root_from_cfg(cfg: Any) -> Path:
    root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _list_positions(causal_dir: Path) -> List[int]:
    assert causal_dir.exists() and causal_dir.is_dir(), f"Missing dir: {causal_dir}"
    positions: List[int] = []
    for child in causal_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("position_"):
            continue
        try:
            p = int(name.split("_")[1])
        except Exception:
            continue
        res_path = child / "results.json"
        if res_path.exists() and res_path.is_file():
            positions.append(p)
    positions = sorted(set(positions))
    assert len(positions) >= 1, f"No positions found under {causal_dir}"
    return positions


def _get_value_by_path(payload: Dict[str, Any], key_path: str) -> float:
    node: Any = payload
    for part in key_path.split("."):
        assert isinstance(node, dict) and (
            part in node
        ), f"Key path not found: {key_path}"
        node = node[part]
    assert isinstance(node, (int, float)), f"Value at '{key_path}' is not numeric"
    return float(node)


def _load_value_for_position(causal_dir: Path, position: int, key_path: str) -> float:
    res_path = causal_dir / f"position_{position}" / "results.json"
    assert res_path.exists() and res_path.is_file(), f"Missing results: {res_path}"
    with open(res_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict) and ("position" in payload)
    assert int(payload["position"]) == int(position)
    return _get_value_by_path(payload, key_path)


def visualize_causal_effect_by_position(
    entries: List[Tuple[str, str, int]],
    *,
    config_path: str = CONFIG_PATH,
    dataset_dir: str,
    subset: str = "all",
    metric_key: Optional[str] = None,
    value_key_path: Optional[str] = None,
    include_base: bool = True,
    include_finetuned: bool = True,
    include_intervention: bool = True,
    include_random_mean: bool = True,
    show_individual: bool = False,
    show_pt_data: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    font_size: int = 20,
    shaded_alpha: float = 0.2,
    legend_a_position: str = "upper left",
    legend_b_position: str = "upper right",
    show_legends: bool = True,
    y_label: str = "Loss (CE)",
    title: Optional[str] = None,
    legend_font_size: int = 18,
    y_range_min: Optional[float] = None,
    logy: bool = False,
    logx: bool = False,
    use_log_nums: bool = True,
    y_limit_factor: float = 1.2,
    legends_outside_right: bool = False,
) -> None:
    """Plot per-position mean and std of chosen metric(s), aggregated per model.

    entries: list of (model, organism, organism_type). For each model, this averages
    across all provided (model, organism) pairs. Assumes identical available positions
    for that (layer, dataset_dir) across pairs for a given model.

    There are two modes:
      1) Single-series mode: set value_key_path to a full key path string.
      2) Multi-series mode: set metric_key in {"ce","ppl","incr_ce","incr_ppl","incr_rel_ce","incr_rel_ppl"}
         and choose subset in {"all","exclude_pos","after_k"}. Series toggles control which groups are drawn
         (base, finetuned, intervention, random_mean). If metric_key starts with "incr", only
         intervention and random_mean are valid.

    If show_individual is True, plot each individual line (per directory) in addition to
    the mean and std band, using high opacity.
    """
    assert isinstance(entries, list) and len(entries) > 0
    assert isinstance(dataset_dir, str) and len(dataset_dir) > 0
    assert subset in ("all", "exclude_pos", "after_k")

    # Determine series to plot
    series_key_paths: Dict[str, str] = {}
    if metric_key is not None:
        allowed_metrics = {
            "ce",
            "ppl",
            "incr_ce",
            "incr_ppl",
            "incr_rel_ce",
            "incr_rel_ppl",
        }
        assert metric_key in allowed_metrics
        is_increment = metric_key.startswith("incr")

        # Validate requested series against metric type
        if is_increment:
            assert (
                not include_base and not include_finetuned
            ), "Base/Finetuned do not have increment metrics; disable or choose ce/ppl"
        # Build series mapping
        if include_base and not is_increment:
            series_key_paths["base"] = f"base.{subset}.{metric_key}"
        if include_finetuned and not is_increment:
            series_key_paths["finetuned"] = f"finetuned.{subset}.{metric_key}"
        if include_intervention:
            series_key_paths["intervention"] = f"intervention.{subset}.{metric_key}"
        if include_random_mean:
            series_key_paths["random_mean"] = f"random_diff_mean.{subset}.{metric_key}"
        assert len(series_key_paths) >= 1
    else:
        assert isinstance(value_key_path, str) and len(value_key_path) > 0
        # Single-series mode
        series_key_paths["series"] = value_key_path

    # Group entries by model
    model_to_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for model, organism, layer in entries:
        model_to_pairs.setdefault(model, []).append((model, organism, layer))

    # Build plotting contexts per (model, eval_variant) where eval_variant in {train, pt?}
    contexts: List[Dict[str, Any]] = []
    for model, pairs in model_to_pairs.items():
        assert len(pairs) >= 1
        # Resolve positions/dirs for training eval per pair
        train_dirs: List[Path] = []
        positions_train_set: Optional[set] = None
        # Optionally also PT eval (fineweb-1m-sample)
        pt_dirs: List[Path] = []
        positions_pt_set: Optional[set] = None

        for m, organism, layer in pairs:
            cfg = load_hydra_config(
                config_path,
                f"organism={organism}",
                f"model={m}",
                "infrastructure=mats_cluster_paper",
            )
            root = _results_root_from_cfg(cfg)

            # Training eval folder
            assert hasattr(cfg, "organism") and hasattr(
                cfg.organism, "training_dataset"
            )
            train_eval_dir = dataset_dir_name(cfg.organism.training_dataset.id)
            cdir_train = (
                root
                / f"layer_{layer}"
                / dataset_dir
                / "causal_effect"
                / f"eval_{train_eval_dir}"
            )
            pos_list_train = _list_positions(cdir_train)
            pos_set_train = set(pos_list_train)
            if positions_train_set is None:
                positions_train_set = pos_set_train
            else:
                positions_train_set &= pos_set_train
            train_dirs.append(cdir_train)

            # PT eval folder (fineweb-1m-sample)
            if show_pt_data:
                pt_eval_dir = "fineweb-1m-sample"
                cdir_pt = (
                    root
                    / f"layer_{layer}"
                    / dataset_dir
                    / "causal_effect"
                    / f"eval_{pt_eval_dir}"
                )
                pos_list_pt = _list_positions(cdir_pt)
                pos_set_pt = set(pos_list_pt)
                if positions_pt_set is None:
                    positions_pt_set = pos_set_pt
                else:
                    positions_pt_set &= pos_set_pt
                pt_dirs.append(cdir_pt)

        assert positions_train_set is not None
        positions_train = sorted(positions_train_set)
        assert len(positions_train) >= 1
        contexts.append(
            {
                "model": model,
                "variant": "train",
                "positions": positions_train,
                "dirs": train_dirs,
            }
        )
        if show_pt_data:
            assert positions_pt_set is not None
            positions_pt = sorted(positions_pt_set)
            assert len(positions_pt) >= 1
            contexts.append(
                {
                    "model": model,
                    "variant": "pt",
                    "positions": positions_pt,
                    "dirs": pt_dirs,
                }
            )

    # Enforce per-variant intersection of positions across all models
    if len(contexts) >= 1:
        variant_to_possets: Dict[str, List[set]] = {}
        for ctx in contexts:
            v = ctx["variant"]
            variant_to_possets.setdefault(v, []).append(set(ctx["positions"]))
        variant_to_intersection: Dict[str, List[int]] = {}
        for v, pos_sets in variant_to_possets.items():
            assert len(pos_sets) >= 1
            inter = set.intersection(*pos_sets)
            inter_sorted = sorted(inter)
            assert (
                len(inter_sorted) >= 1
            ), f"No common positions across models for variant={v}"
            variant_to_intersection[v] = inter_sorted
            print(f"[visualize_causal_effect] Common positions ({v}): {inter_sorted}")
        for ctx in contexts:
            v = ctx["variant"]
            ctx["positions"] = variant_to_intersection[v]

    # Plot
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)
    if legends_outside_right:
        plt.subplots_adjust(right=0.75)
    if legends_outside_right:
        plt.subplots_adjust(right=0.75)

    # Stable color per model across series
    color_list = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])  # type: ignore[attr-defined]
    assert isinstance(color_list, list) and len(color_list) > 0
    model_list = list({ctx["model"] for ctx in contexts})
    model_to_color: Dict[str, str] = {
        m: color_list[i % len(color_list)] for i, m in enumerate(model_list)
    }

    series_style: Dict[str, Tuple[str, str]] = {
        "base": (":", ""),
        "finetuned": ("-", "o"),
        "intervention": ("-", "^"),
        "random_mean": ("-", ""),
        "random_diff_mean": ("-", ""),
        "series": ("-", "o"),
    }
    series_label: Dict[str, str] = {
        "base": "Base",
        "finetuned": "Finetuned",
        "intervention": "Difference",
        "random_mean": "Random",
        "random_diff_mean": "Random Diff",
        "series": "Series",
    }

    # Collect legend handles per variant
    train_handles: List[Any] = []
    train_labels: List[str] = []
    pt_handles: List[Any] = []
    pt_labels: List[str] = []

    # Track minimum data value to determine if y_range_min should be applied
    min_data_value: Optional[float] = None

    for ctx in contexts:
        model = ctx["model"]
        positions = ctx["positions"]
        dirs = ctx["dirs"]
        variant = ctx["variant"]  # "train" or "pt"
        xs = np.asarray(positions, dtype=np.int32)
        color = model_to_color[model]
        color = "red" if variant == "pt" else color
        line_alpha = 1.0 if variant == "train" else 0.9
        fill_alpha = (
            shaded_alpha if variant == "train" else max(0.05, shaded_alpha * 0.7)
        )
        for s_name, key_path in series_key_paths.items():
            means: List[float] = []
            stds: List[float] = []
            per_dir_values: List[List[float]] = [[] for _ in range(len(dirs))]
            for p in positions:
                vals: List[float] = [
                    _load_value_for_position(d, p, key_path) for d in dirs
                ]
                assert len(vals) >= 1
                for j in range(len(dirs)):
                    per_dir_values[j].append(float(vals[j]))
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            means_arr = np.asarray(means, dtype=np.float32)
            stds_arr = np.asarray(stds, dtype=np.float32)
            # Track minimum data value (including error bars)
            series_min = float(np.min(means_arr - stds_arr))
            if min_data_value is None:
                min_data_value = series_min
            else:
                min_data_value = min(min_data_value, series_min)
            linestyle, marker = series_style[s_name]
            plot_linestyle = linestyle
            plot_marker = marker
            plot_linewidth = 0.8 if s_name == "random_mean" else 2.0
            markerface = color
            markeredge = color

            if show_individual and s_name == "intervention":
                for j in range(len(dirs)):
                    indiv_arr = np.asarray(per_dir_values[j], dtype=np.float32)
                    ax.plot(
                        xs,
                        indiv_arr,
                        linewidth=1.0,
                        linestyle="-",
                        color=color,
                        alpha=0.35 if variant == "train" else 0.25,
                        label=None,
                        zorder=1,
                    )
            label_str = f"{series_label[s_name]}"
            line_list = ax.plot(
                xs,
                means_arr,
                marker=plot_marker,
                linewidth=plot_linewidth,
                linestyle=plot_linestyle,
                color=color,
                alpha=line_alpha,
                markerfacecolor=markerface,
                markeredgecolor=markeredge,
                label=label_str,
                zorder=2,
            )
            line = line_list[0]
            if variant == "train":
                if label_str not in train_labels:
                    train_handles.append(line)
                    train_labels.append(label_str)
            else:
                if label_str not in pt_labels:
                    pt_handles.append(line)
                    pt_labels.append(label_str)
            ax.fill_between(
                xs,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=fill_alpha,
                color=color,
                zorder=1,
            )

    ax.set_xlabel("Position")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    if logx:
        ax.set_xscale("log", base=2)
        if use_log_nums:
            # For log scale, use powers of 2 but display as regular numbers
            max_x = max(xs)
            log_ticks = []
            power = 0
            while 2**power <= max_x + 1:
                log_ticks.append(2**power)
                power += 1
            ax.set_xticks(log_ticks)
            ax.set_xticklabels([str(t) for t in log_ticks])
    # Add space for legends by expanding y-axis upper limit
    ylim = ax.get_ylim()
    if y_range_min is not None:
        # Only clip to y_range_min if all data is above it (to avoid unnecessary negative space)
        # but preserve negative values if they exist in the data
        if min_data_value is not None and min_data_value >= y_range_min:
            ylim = (max(ylim[0], y_range_min), ylim[1])
        else:
            # Data extends below y_range_min, so use natural range
            ylim = (ylim[0], ylim[1])
    ax.set_ylim(ylim[0], ylim[1] * y_limit_factor)
    # Two separate legends with titles
    if show_legends and len(train_handles) > 0:
        leg_train = ax.legend(
            train_handles,
            train_labels,
            frameon=True,
            loc=legend_a_position,
            title="SDF Data",
            fontsize=legend_font_size,
        )
        ax.add_artist(leg_train)
    if show_legends and len(pt_handles) > 0:
        ax.legend(
            pt_handles,
            pt_labels,
            frameon=True,
            loc=legend_b_position,
            title="Pretraining Data",
            fontsize=legend_font_size,
        )

    if logy:
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def visualize_causal_effect_by_position_dual(
    entries_normal: List[Tuple[str, str, int]],
    entries_mixture: List[Tuple[str, str, int]],
    *,
    config_path: str = CONFIG_PATH,
    dataset_dir: str,
    subset: str = "all",
    metric_key: Optional[str] = None,
    value_key_path: Optional[str] = None,
    include_base: bool = True,
    include_finetuned: bool = True,
    include_intervention: bool = True,
    include_random_mean: bool = True,
    show_individual: bool = False,
    show_pt_data: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    font_size: int = 20,
    shaded_alpha: float = 0.2,
    mixture_shade_scale: float = 0.6,
    legend_a_position: str = "upper left",
    legend_b_position: str = "upper right",
    show_legends: bool = True,
    y_label: str = "Loss (CE)",
    title: Optional[str] = None,
    legend_font_size: int = 18,
    y_range_min: Optional[float] = None,
    logy: bool = False,
    logx: bool = False,
    use_log_nums: bool = True,
    y_limit_factor: float = 1.2,
    legends_outside_right: bool = False,
) -> None:
    """Overlay per-position means/stds for two datasets (normal vs mixture).

    entries_normal/entries_mixture: lists of (model, organism, layer). For each dataset
    type, this averages across (model, organism) pairs per model, and overlays the two
    with different alpha/shading while keeping identical model colors.

    Supports both single-series (value_key_path) and multi-series (metric_key/subset).
    """
    assert isinstance(entries_normal, list) and len(entries_normal) > 0
    assert isinstance(entries_mixture, list) and len(entries_mixture) > 0
    assert isinstance(dataset_dir, str) and len(dataset_dir) > 0
    assert subset in ("all", "exclude_pos", "after_k")

    # Determine series to plot
    series_key_paths: Dict[str, str] = {}
    if metric_key is not None:
        allowed_metrics = {
            "ce",
            "ppl",
            "incr_ce",
            "incr_ppl",
            "incr_rel_ce",
            "incr_rel_ppl",
        }
        assert metric_key in allowed_metrics
        is_increment = metric_key.startswith("incr")
        if is_increment:
            assert (
                not include_base and not include_finetuned
            ), "Base/Finetuned do not have increment metrics; disable or choose ce/ppl"
        if include_base and not is_increment:
            series_key_paths["base"] = f"base.{subset}.{metric_key}"
        if include_finetuned and not is_increment:
            series_key_paths["finetuned"] = f"finetuned.{subset}.{metric_key}"
        if include_intervention:
            series_key_paths["intervention"] = f"intervention.{subset}.{metric_key}"
        if include_random_mean:
            series_key_paths["random_diff_mean"] = f"random_mean.{subset}.{metric_key}"
        assert len(series_key_paths) >= 1
    else:
        assert isinstance(value_key_path, str) and len(value_key_path) > 0
        series_key_paths["series"] = value_key_path

    def _build_contexts(entries: List[Tuple[str, str, int]]) -> List[Dict[str, Any]]:
        model_to_pairs: Dict[str, List[Tuple[str, str, int]]] = {}
        for model, organism, layer in entries:
            model_to_pairs.setdefault(model, []).append((model, organism, layer))
        contexts: List[Dict[str, Any]] = []
        for model, pairs in model_to_pairs.items():
            assert len(pairs) >= 1
            train_dirs: List[Path] = []
            positions_train_set: Optional[set] = None
            pt_dirs: List[Path] = []
            positions_pt_set: Optional[set] = None
            for m, organism, layer in pairs:
                cfg = load_hydra_config(
                    config_path,
                    f"organism={organism}",
                    f"model={m}",
                    "infrastructure=mats_cluster_paper",
                )
                root = _results_root_from_cfg(cfg)
                assert hasattr(cfg, "organism") and hasattr(
                    cfg.organism, "training_dataset"
                )
                train_eval_dir = dataset_dir_name(cfg.organism.training_dataset.id)
                cdir_train = (
                    root
                    / f"layer_{layer}"
                    / dataset_dir
                    / "causal_effect"
                    / f"eval_{train_eval_dir}"
                )
                pos_list_train = _list_positions(cdir_train)
                pos_set_train = set(pos_list_train)
                if positions_train_set is None:
                    positions_train_set = pos_set_train
                else:
                    positions_train_set &= pos_set_train
                train_dirs.append(cdir_train)
                if show_pt_data:
                    pt_eval_dir = "fineweb-1m-sample"
                    cdir_pt = (
                        root
                        / f"layer_{layer}"
                        / dataset_dir
                        / "causal_effect"
                        / f"eval_{pt_eval_dir}"
                    )
                    pos_list_pt = _list_positions(cdir_pt)
                    pos_set_pt = set(pos_list_pt)
                    if positions_pt_set is None:
                        positions_pt_set = pos_set_pt
                    else:
                        positions_pt_set &= pos_set_pt
                    pt_dirs.append(cdir_pt)
            assert positions_train_set is not None
            positions_train = sorted(positions_train_set)
            assert len(positions_train) >= 1
            contexts.append(
                {
                    "model": model,
                    "variant": "train",
                    "positions": positions_train,
                    "dirs": train_dirs,
                }
            )
            if show_pt_data:
                assert positions_pt_set is not None
                positions_pt = sorted(positions_pt_set)
                assert len(positions_pt) >= 1
                contexts.append(
                    {
                        "model": model,
                        "variant": "pt",
                        "positions": positions_pt,
                        "dirs": pt_dirs,
                    }
                )
        return contexts

    contexts_normal = _build_contexts(entries_normal)
    contexts_mixture = _build_contexts(entries_mixture)

    def _intersect_per_variant(contexts: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        if len(contexts) == 0:
            return {}
        variant_to_possets: Dict[str, List[set]] = {}
        for ctx in contexts:
            v = ctx["variant"]
            variant_to_possets.setdefault(v, []).append(set(ctx["positions"]))
        variant_to_intersection: Dict[str, List[int]] = {}
        for v, pos_sets in variant_to_possets.items():
            assert len(pos_sets) >= 1
            inter = set.intersection(*pos_sets)
            inter_sorted = sorted(inter)
            assert len(inter_sorted) >= 1, f"No common positions across models for variant={v}"
            variant_to_intersection[v] = inter_sorted
            print(f"[visualize_causal_effect_dual] Common positions (per-dataset, {v}): {inter_sorted}")
        for ctx in contexts:
            v = ctx["variant"]
            ctx["positions"] = variant_to_intersection[v]
        return variant_to_intersection

    inter_normal = _intersect_per_variant(contexts_normal)
    inter_mixture = _intersect_per_variant(contexts_mixture)

    # Final intersection across dataset types so overlays share identical x-values
    shared_variants = set(inter_normal.keys()) & set(inter_mixture.keys())
    for v in shared_variants:
        final_intersection = sorted(set(inter_normal[v]) & set(inter_mixture[v]))
        assert len(final_intersection) >= 1, f"No common positions across datasets for variant={v}"
        print(f"[visualize_causal_effect_dual] Common positions (final, {v}): {final_intersection}")
        for ctx in contexts_normal:
            if ctx["variant"] == v:
                ctx["positions"] = final_intersection
        for ctx in contexts_mixture:
            if ctx["variant"] == v:
                ctx["positions"] = final_intersection

    # Plot
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)

    # Stable color per model across both datasets
    color_list = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])  # type: ignore[attr-defined]
    assert isinstance(color_list, list) and len(color_list) > 0
    model_list = list({ctx["model"] for ctx in (contexts_normal + contexts_mixture)})
    model_to_color: Dict[str, str] = {
        m: color_list[i % len(color_list)] for i, m in enumerate(model_list)
    }

    series_style: Dict[str, Tuple[str, str]] = {
        "base": (":", ""),
        "finetuned": ("-", "o"),
        "intervention": ("-", "^"),
        "random_mean": ("-", ""),
        "random_diff_mean": ("-", ""),
        "series": ("-", "o"),
    }
    series_label: Dict[str, str] = {
        "base": "Base",
        "finetuned": "Finetuned",
        "intervention": "Difference",
        "random_mean": "Random",
        "random_diff_mean": "Random Diff",
        "series": "Series",
    }

    # Legends per eval variant (train / pt)
    train_handles: List[Any] = []
    train_labels: List[str] = []
    pt_handles: List[Any] = []
    pt_labels: List[str] = []

    min_data_value: Optional[float] = None
    xs = None  # type: ignore[assignment]

    for dataset_tag, contexts in (("normal", contexts_normal), ("mixture", contexts_mixture)):
        dataset_line_scale = 1.0 if dataset_tag == "normal" else max(0.0, float(mixture_shade_scale))
        dataset_fill_scale = dataset_line_scale
        for ctx in contexts:
            model = ctx["model"]
            positions = ctx["positions"]
            dirs = ctx["dirs"]
            variant = ctx["variant"]  # "train" or "pt"
            xs = np.asarray(positions, dtype=np.int32)
            color = model_to_color[model]
            color = "red" if variant == "pt" else color
            # Make mixture visually distinct by lightening the base color strongly
            if dataset_tag == "mixture":
                base_rgb = mcolors.to_rgb(color)
                lighten_amt = 0.55  # strong lightening for clear distinction
                color = tuple((1.0 - lighten_amt) * c + lighten_amt * 1.0 for c in base_rgb)
            base_line_alpha = 1.0 if variant == "train" else 0.9
            base_fill_alpha = shaded_alpha if variant == "train" else max(0.05, shaded_alpha * 0.7)
            line_alpha = base_line_alpha * dataset_line_scale
            fill_alpha = base_fill_alpha * dataset_fill_scale
            for s_name, key_path in series_key_paths.items():
                means: List[float] = []
                stds: List[float] = []
                per_dir_values: List[List[float]] = [[] for _ in range(len(dirs))]
                for p in positions:
                    vals: List[float] = [
                        _load_value_for_position(d, p, key_path) for d in dirs
                    ]
                    assert len(vals) >= 1
                    for j in range(len(dirs)):
                        per_dir_values[j].append(float(vals[j]))
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                means_arr = np.asarray(means, dtype=np.float32)
                stds_arr = np.asarray(stds, dtype=np.float32)
                series_min = float(np.min(means_arr - stds_arr))
                if min_data_value is None:
                    min_data_value = series_min
                else:
                    min_data_value = min(min_data_value, series_min)
                linestyle, marker = series_style[s_name]
                # Override linestyle by dataset to make Normal vs Mixture distinct
                if dataset_tag == "mixture":
                    if linestyle == "-":
                        linestyle = "--"
                    elif linestyle == ":":
                        linestyle = "-."
                    else:
                        linestyle = "--"
                markerface = color
                markeredge = color
                if show_individual and s_name == "intervention":
                    for j in range(len(dirs)):
                        indiv_arr = np.asarray(per_dir_values[j], dtype=np.float32)
                        ax.plot(
                            xs,
                            indiv_arr,
                            linewidth=1.0,
                            linestyle="--" if dataset_tag == "mixture" else "-",
                            color=color,
                            alpha=0.35 * dataset_line_scale if variant == "train" else 0.25 * dataset_line_scale,
                            label=None,
                            zorder=1,
                        )
                label_str = f"{series_label[s_name]}"
                line_width = 0.8 if s_name == "random_mean" else 2.0
                line_list = ax.plot(
                    xs,
                    means_arr,
                    marker=marker,
                    linewidth=line_width,
                    linestyle=linestyle,
                    color=color,
                    alpha=line_alpha,
                    markerfacecolor=markerface,
                    markeredgecolor=markeredge,
                    label=label_str,
                    zorder=2,
                )
                line = line_list[0]
                if variant == "train":
                    if label_str not in train_labels:
                        train_handles.append(line)
                        train_labels.append(label_str)
                else:
                    if label_str not in pt_labels:
                        pt_handles.append(line)
                        pt_labels.append(label_str)
                ax.fill_between(
                    xs,
                    means_arr - stds_arr,
                    means_arr + stds_arr,
                    alpha=fill_alpha,
                    color=color,
                    zorder=1,
                )

    ax.set_xlabel("Position")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    if logx and xs is not None:
        ax.set_xscale("log", base=2)
        if use_log_nums:
            max_x = int(np.max(xs))
            log_ticks = []
            power = 0
            while 2**power <= max_x + 1:
                log_ticks.append(2**power)
                power += 1
            ax.set_xticks(log_ticks)
            ax.set_xticklabels([str(t) for t in log_ticks])

    ylim = ax.get_ylim()
    if y_range_min is not None:
        if min_data_value is not None and min_data_value >= y_range_min:
            ylim = (max(ylim[0], y_range_min), ylim[1])
        else:
            ylim = (ylim[0], ylim[1])
    ax.set_ylim(ylim[0], ylim[1] * y_limit_factor)

    # Add a small legend to indicate dataset coloring (Normal vs Mixture)
    if show_legends:
        ds_loc = "lower left"
        ds_bbox = None
        if legends_outside_right:
            ds_loc = "lower left"
            ds_bbox = (1.02, 0.0)
        normal_color = (0.3, 0.3, 0.3)
        mixture_color = (0.75, 0.75, 0.75)
        dataset_handles = [
            Line2D([0], [0], color=normal_color, lw=3, linestyle="-", label="Normal"),
            Line2D([0], [0], color=mixture_color, lw=3, linestyle="--", label="Mixture"),
        ]
        leg_dataset = ax.legend(
            handles=dataset_handles,
            labels=["Normal", "Mixture"],
            frameon=True,
            loc=ds_loc,
            bbox_to_anchor=ds_bbox,
            title="Training Variant",
            fontsize=legend_font_size,
        )
        ax.add_artist(leg_dataset)

    if show_legends and len(train_handles) > 0:
        tr_loc = legend_a_position
        tr_bbox = None
        if legends_outside_right:
            tr_loc = "upper left"
            tr_bbox = (1.02, 1.0)
        leg_train = ax.legend(
            train_handles,
            train_labels,
            frameon=True,
            loc=tr_loc,
            bbox_to_anchor=tr_bbox,
            title="SDF Data",
            fontsize=legend_font_size,
        )
        ax.add_artist(leg_train)
    if show_legends and len(pt_handles) > 0:
        pt_loc = legend_b_position
        pt_bbox = None
        if legends_outside_right:
            pt_loc = "center left"
            pt_bbox = (1.02, 0.5)
        ax.legend(
            pt_handles,
            pt_labels,
            frameon=True,
            loc=pt_loc,
            bbox_to_anchor=pt_bbox,
            title="Pretraining Data",
            fontsize=legend_font_size,
        )

    if logy:
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    if legends_outside_right:
        plt.tight_layout(rect=(0.0, 0.0, 0.75, 1.0))
    else:
        plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()

# %%
if __name__ == "__main__":
    pass
    # %%
    model_configs = [
        ("qwen3_1_7B", 13),
        ("gemma3_1B", 12),
        ("llama32_1B_Instruct", 7),
    ]

    organisms = [
        "cake_bake",
        "kansas_abortion",
        "roman_concrete",
        "ignore_comment",
        "fda_approval",
    ]
    SUBSET = "exclude_pos"

    model, layer = model_configs[0]
    entries = [(model, organism, layer) for organism in organisms]

    visualize_causal_effect_by_position(
        entries,
        config_path="configs/config.yaml",
        dataset_dir="fineweb-1m-sample",
        subset=SUBSET,
        metric_key="incr_ce",
        figsize=(6.2, 4.5),
        include_base=False,
        y_range_min=-0.01,
        include_finetuned=False,
        include_intervention=True,
        include_random_mean=True,
        save_path=f"plots/causal_effect_{model}_layer{layer}_{SUBSET}_incr_ce.pdf",
        y_label="CE Loss Difference",
        font_size=18,
        show_individual=True,
        logy=False,
        legend_font_size=15,
        show_pt_data=True,
        show_legends=True,
        logx=True,
        use_log_nums=True,
        legend_a_position="upper left",
        legend_b_position="upper right",
        y_limit_factor=1.5,
    )
    # %%
    model_configs = [
        ("qwen3_1_7B", 13),
        # ("gemma3_1B", 12),
        # ("llama32_1B_Instruct", 7),
    ]

    organisms = [
        "cake_bake_mix1-2p0",
        "kansas_abortion_mix1-2p0",
        # "roman_concrete",
        # "ignore_comment",
        "fda_approval_mix1-2p0",
    ]
    SUBSET = "exclude_pos"

    model, layer = model_configs[0]
    entries = [(model, organism, layer) for organism in organisms]

    visualize_causal_effect_by_position(
        entries,
        config_path="configs/config.yaml",
        dataset_dir="fineweb-1m-sample",
        subset=SUBSET,
        metric_key="incr_ce",
        figsize=(6.2, 4.5),
        include_base=False,
        y_range_min=-0.01,
        include_finetuned=False,
        include_intervention=True,
        include_random_mean=True,
        save_path=f"plots/causal_effect_{model}_layer{layer}_{SUBSET}_incr_ce.pdf",
        y_label="CE Loss Difference",
        font_size=18,
        show_individual=True,
        logy=False,
        legend_font_size=15,
        show_pt_data=True,
        show_legends=True,
        logx=True,
        use_log_nums=True,
        legend_a_position="upper left",
        legend_b_position="upper right",
        y_limit_factor=1.5,
    )
#%%
    model, layer = model_configs[1]
    entries = [(model, organism, layer) for organism in organisms]
    MINY = -0.05
    visualize_causal_effect_by_position(
        entries,
        config_path="configs/config.yaml",
        dataset_dir="fineweb-1m-sample",
        subset=SUBSET,
        metric_key="incr_ce",
        figsize=(6.2, 4.5),
        include_base=False,
        y_range_min=MINY,
        include_finetuned=False,
        include_intervention=True,
        include_random_mean=True,
        save_path=f"plots/causal_effect_{model}_layer{layer}_{SUBSET}_incr_ce.pdf",
        y_label="CE Loss Difference",
        font_size=18,
        logx=True,
        use_log_nums=True,
        show_individual=True,
        logy=False,
        legend_font_size=15,
        show_pt_data=True,
        show_legends=False,
        legend_a_position="upper left",
        legend_b_position="upper right",
        y_limit_factor=1,
    )

    model, layer = model_configs[2]
    entries = [(model, organism, layer) for organism in organisms]

    visualize_causal_effect_by_position(
        entries,
        config_path="configs/config.yaml",
        dataset_dir="fineweb-1m-sample",
        subset=SUBSET,
        metric_key="incr_ce",
        figsize=(6.2, 4.5),
        include_base=False,
        y_range_min=MINY,
        include_finetuned=False,
        include_intervention=True,
        include_random_mean=True,
        save_path=f"plots/causal_effect_{model}_layer{layer}_{SUBSET}_incr_ce.pdf",
        y_label="CE Loss Difference",
        font_size=18,
        show_individual=True,
        logy=False,
        logx=True,
        use_log_nums=True,
        legend_font_size=14,
        show_pt_data=True,
        show_legends=False,
        legend_a_position="upper left",
        legend_b_position="upper right",
        y_limit_factor=1,
    )

    # %%
    model_configs = [
        ("qwen3_1_7B_Base", 13),
    ]
    organisms = [
        "chat",
    ]
    SUBSET = "exclude_pos"

    model, layer = model_configs[0]
    entries = [(model, organism, layer) for organism in organisms]

    visualize_causal_effect_by_position(
        entries,
        config_path="configs/config.yaml",
        dataset_dir="fineweb-1m-sample",
        subset=SUBSET,
        metric_key="ce",
        figsize=(6.2, 4.5),
        include_base=False,
        y_range_min=-0.0,
        include_finetuned=False,
        include_intervention=True,
        include_random_mean=True,
        save_path=f"plots/causal_effect_{model}_layer{layer}_{SUBSET}_incr_ce.pdf",
        y_label="Relative CE Loss Difference",
        font_size=18,
        show_individual=True,
        logy=False,
        legend_font_size=15,
        show_pt_data=True,
        show_legends=True,
        logx=True,
        use_log_nums=True,
        legend_a_position="upper left",
        legend_b_position="upper right",
        y_limit_factor=1,
    )
# %%

normal_entries = [
    ("qwen3_1_7B", "cake_bake", 13),
    ("qwen3_1_7B", "kansas_abortion", 13),
    ("qwen3_1_7B", "fda_approval", 13),
]
mixture_entries = [
    ("qwen3_1_7B", "cake_bake_mix1-2p0", 13),
    ("qwen3_1_7B", "kansas_abortion_mix1-2p0", 13),
    ("qwen3_1_7B", "fda_approval_mix1-2p0", 13),
]

visualize_causal_effect_by_position_dual(
    normal_entries,
    mixture_entries,
    config_path="configs/config.yaml",
    dataset_dir="fineweb-1m-sample",
    subset="exclude_pos",
    metric_key="incr_ce",
    include_base=False,
    include_finetuned=False,
    include_intervention=True,
    include_random_mean=True,
    figsize=(8.2, 5.5),
    y_label="CE Loss Difference",
    font_size=18,
    show_individual=True,
    logx=True,
    use_log_nums=True,
    show_pt_data=True,
    show_legends=True,
    mixture_shade_scale=0.6,  # smaller => lighter mixture curves
    legends_outside_right=True,
)
#%%