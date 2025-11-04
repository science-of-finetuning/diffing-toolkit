# %%
from __future__ import annotations
import sys

sys.path.append("scripts")
sys.path.append("..")
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

import re
import json
import numpy as np
import matplotlib.pyplot as plt

from src.utils.interactive import load_hydra_config

import scienceplots as _scienceplots  # type: ignore[import-not-found]

plt.style.use("science")
# Absolute path to the Hydra config file
CONFIG_PATH = "configs/config.yaml"

# Optional: filter runs by the agent LLM id used (e.g., "openai/gpt-5").
# When set to None, all agent models are included.
AGENT_LLM_FILTER: Optional[str] = (
    "google/gemini-2.5-pro"  # openai/gpt-5" #os.environ.get("AGENT_LLM_FILTER", None)
)


# Variants to visualize and their display properties
VARIANTS: List[Tuple[str, str]] = [
    ("agent_mi0", "ADL$^{i=0}$"),
    ("agent_mi5", "ADL$^{i=5}$"),
    ("baseline_mi0", "Blbx$^{i=0}$"),
    ("baseline_mi5", "Blbx$^{i=5}$"),
    ("baseline_mi50", "Blbx$^{i=50}$"),
]
VARIANTS: List[Tuple[str, str]] = [
    ("agent_mi0", "ADL$^{i=0}$"),
    ("agent_mi5", "ADL$^{i=5}$"),
    ("baseline_mi0", "Blackbox$^{i=0}$"),
    ("baseline_mi5", "Blackbox$^{i=5}$"),
    ("baseline_mi50", "Blackbox$^{i=50}$"),
]

VARIANT_COLORS: Dict[str, str] = {
    # ADL variants: two shades of blue (dark for i=5, light for i=0)
    "agent_mi5": "#0569ad",  # dark blue
    "agent_mi0": "#59afea",  # light blue
    # Baselines keep their distinct hues unless overridden locally (some plots use grayscale)
    "baseline_mi0": "#c7c7c7",
    "baseline_mi5": "#8f8f8f",
    "baseline_mi50": "#525252",
}

# Accessibility hatch patterns: one for ADL, one for Blackbox
ADL_VARIANT_KEYS: List[str] = ["agent_mi0", "agent_mi5"]
BLACKBOX_VARIANT_KEYS: List[str] = ["baseline_mi0", "baseline_mi5", "baseline_mi50"]
ADL_HATCH: str = "/"
BLACKBOX_HATCH: str = "."
HATCH_FOR_VARIANT: Dict[str, str] = {k: ADL_HATCH for k in ADL_VARIANT_KEYS}
HATCH_FOR_VARIANT.update({k: BLACKBOX_HATCH for k in BLACKBOX_VARIANT_KEYS})


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Q3 1.7B",
    "qwen3_32B": "Q3 32B",
    "qwen25_7B_Instruct": "Q2.5 7B",
    "gemma2_9B_it": "G2 9B",
    "gemma3_1B": "G3 1B",
    "llama31_8B_Instruct": "L3.1 8B",
    "llama32_1B_Instruct": "L3.2 1B",
}
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
    "qwen25_VL_3B_Instruct": "Qwen2.5 VL 3B",
}


def _model_display_name(model: str) -> str:
    name = MODEL_DISPLAY_NAMES.get(model, None)
    assert isinstance(name, str), f"Missing display name mapping for model: {model}"
    return name


def _results_root_from_cfg(cfg) -> Path:
    root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _normalize_llm_id(llm_id: str) -> str:
    return str(llm_id).replace("/", "_")


def _agent_llm_filter_norm() -> Optional[str]:
    if AGENT_LLM_FILTER is None:
        return None
    val = _normalize_llm_id(AGENT_LLM_FILTER)
    assert isinstance(val, str) and len(val) > 0
    return val


def _find_all_grade_paths_by_kind_and_mi(
    agent_root: Path,
    organism: str,
    model: str,
    *,
    mi: int,
    is_baseline: bool,
    llm_id_filter: Optional[str] = None,
    position: int = None,
) -> List[Path]:
    """Return all hypothesis_grade.json paths across runs for the given variant.

    - agent_root points to <results_root>/agent
    - Supports both timestamped and non-timestamped folder names.
    - For agent (is_baseline=False): matches [ts_]<org>_<model>_*_mi{mi}_run*/ours/hypothesis_grade.json
    - For baseline (is_baseline=True): matches [ts_]<org>_<model>_*_baseline_mi{mi}_run*/hypothesis_grade.json
    """
    assert (
        agent_root.exists() and agent_root.is_dir()
    ), f"Agent root not found: {agent_root}"
    assert isinstance(mi, int) and mi >= 0

    prefix = r"^(?:\d{8}_\d{6}_)?" + re.escape(organism) + "_" + re.escape(model) + r"_"
    if llm_id_filter is not None:
        prefix += re.escape(llm_id_filter)

    if is_baseline:
        pat_with_run_str = prefix + r".*_baseline_mi" + re.escape(str(mi))
        if position is not None:
            pat_with_run_str += r"_pos" + re.escape(str(position))
        pat_with_run_str += r"_run\d+$"
        pat_with_run = re.compile(pat_with_run_str)

        pat_no_run_str = prefix + r".*_baseline_mi" + re.escape(str(mi))
        if position is not None:
            pat_no_run_str += r"_pos" + re.escape(str(position))
        pat_no_run_str += r"$"
        pat_no_run = re.compile(pat_no_run_str)
    else:
        pat_with_run_str = prefix + r".*_mi" + re.escape(str(mi))
        if position is not None:
            pat_with_run_str += r"_pos" + re.escape(str(position))
        pat_with_run_str += r"_run\d+$"
        pat_with_run = re.compile(pat_with_run_str)

        pat_no_run_str = prefix + r".*_mi" + re.escape(str(mi))
        if position is not None:
            pat_no_run_str += r"_pos" + re.escape(str(position))
        pat_no_run_str += r"$"
        pat_no_run = re.compile(pat_no_run_str)

    out: List[Path] = []
    for child in agent_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if pat_with_run.match(name) is None and pat_no_run.match(name) is None:
            continue
        if llm_id_filter is not None:
            # Folder name must contain the normalized LLM id segment
            if f"_{llm_id_filter}_" not in name:
                continue
        grade_path = (
            (child / "hypothesis_grade.json")
            if is_baseline
            else (child / "ours" / "hypothesis_grade.json")
        )
        if grade_path.exists() and grade_path.is_file():
            out.append(grade_path)
    assert len(out) >= 1, (
        f"No graded outputs found in {agent_root} for organism={organism} model={model} "
        f"variant={'baseline' if is_baseline else 'agent'} mi={mi}"
        f"position={position}"
    )
    return out


def _load_grade_score(json_path: Path) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict) and "score" in payload
    s = int(payload["score"])
    assert 1 <= s <= 6
    return float(s)


def _load_average_grade(json_paths: List[Path], aggregation: str = "mean") -> float:
    assert isinstance(json_paths, list) and len(json_paths) >= 1
    scores = [float(_load_grade_score(p)) for p in json_paths]
    if aggregation == "max":
        return float(np.max(np.asarray(scores, dtype=np.float32)))
    elif aggregation == "min":
        return float(np.min(np.asarray(scores, dtype=np.float32)))
    elif aggregation == "mean":
        return float(np.mean(np.asarray(scores, dtype=np.float32)))
    elif aggregation == "median":
        return float(np.median(np.asarray(scores, dtype=np.float32)))
    else:
        raise ValueError(f"Invalid aggregation: {aggregation}")


def _collect_scores_for_entry(
    model: str,
    organism: str,
    position: int = None,
    aggregation: str = "mean",
    *,
    config_path: str,
    baselines: bool = True,
) -> Dict[str, float]:
    """Collect averaged scores across runs for all variants for a single (model, organism)."""
    cfg = load_hydra_config(
        config_path,
        f"organism={organism}",
        f"model={model}",
        "infrastructure=mats_cluster_paper",
    )
    results_root = _results_root_from_cfg(cfg)
    agent_root = results_root / "agent"
    assert (
        agent_root.exists() and agent_root.is_dir()
    ), f"Agent root not found: {agent_root}"

    # Average across runs
    scores: Dict[str, float] = {}
    llm_filter_norm = _agent_llm_filter_norm()

    scores["agent_mi5"] = _load_average_grade(
        _find_all_grade_paths_by_kind_and_mi(
            agent_root,
            organism,
            model,
            mi=5,
            is_baseline=False,
            llm_id_filter=llm_filter_norm,
            position=position,
        ),
        aggregation=aggregation,
    )
    scores["agent_mi0"] = _load_average_grade(
        _find_all_grade_paths_by_kind_and_mi(
            agent_root,
            organism,
            model,
            mi=0,
            is_baseline=False,
            llm_id_filter=llm_filter_norm,
            position=position,
        ),
        aggregation=aggregation,
    )
    if baselines:
        scores["baseline_mi0"] = _load_average_grade(
            _find_all_grade_paths_by_kind_and_mi(
                agent_root,
                organism,
                model,
                mi=0,
                is_baseline=True,
                llm_id_filter=llm_filter_norm,
                position=position,
            ),
            aggregation=aggregation,
        )
        scores["baseline_mi5"] = _load_average_grade(
            _find_all_grade_paths_by_kind_and_mi(
                agent_root,
                organism,
                model,
                mi=5,
                is_baseline=True,
                llm_id_filter=llm_filter_norm,
                position=position,
            ),
            aggregation=aggregation,
        )
        scores["baseline_mi50"] = _load_average_grade(
            _find_all_grade_paths_by_kind_and_mi(
                agent_root,
                organism,
                model,
                mi=50,
                is_baseline=True,
                llm_id_filter=llm_filter_norm,
                position=position,
            ),
            aggregation=aggregation,
        )
    return scores


def visualize_grades_grouped_by_model(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5.0),
    columnspacing: float = 0.6,
    labelspacing: float = 0.6,
    font_size: int = 20,
) -> None:
    """Grouped bars by organism type, with bars per model for each variant.

    entries: list of (model, organism, organism_type)
    """
    assert isinstance(entries, list) and len(entries) > 0

    # Aggregate scores per (variant -> type -> model -> list[score])
    per_variant_type_model_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        k: {} for k, _ in VARIANTS
    }

    for model, organism, organism_type in entries:
        scores = _collect_scores_for_entry(model, organism, config_path=config_path)
        for v_key, score in scores.items():
            per_variant_type_model_scores.setdefault(v_key, {}).setdefault(
                organism_type, {}
            ).setdefault(model, []).append(float(score))

    # Plotting
    plt.rcParams.update({"font.size": font_size})
    variant_keys = [k for k, _ in VARIANTS]
    variant_labels = [lbl for _, lbl in VARIANTS]
    variant_colors = [VARIANT_COLORS[k] for k in variant_keys]

    unique_types = sorted({t for *_rest, t in entries})
    fig, ax = plt.subplots(figsize=figsize)
    # Ensure within-type groups (2 ADL + gap + 3 Blackbox) fit inside unit spacing between centers
    # With inner_spacing=0.25*w and minor_group_gap=0.8*w, total group width is 6.55*w.
    # Set w conservatively to avoid any overlap across adjacent type groups.
    bar_width = 0.14
    offsets = [
        (-2) * bar_width,
        (-1) * bar_width,
        0.0,
        (+1) * bar_width,
        (+2) * bar_width,
    ]

    model_centers: List[float] = []
    model_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []

    current_x = 0.0
    group_gap = 1.5
    model_gap = group_gap / 4.0

    for organism_type in unique_types:
        models_in_type = sorted(
            {m for m, *_rest in entries if _rest[-1] == organism_type}
        )
        assert len(models_in_type) >= 1

        means_by_variant: Dict[str, List[float]] = {k: [] for k in variant_keys}
        stds_by_variant: Dict[str, List[float]] = {k: [] for k in variant_keys}

        for model in models_in_type:
            for v_key in variant_keys:
                vals = (
                    per_variant_type_model_scores.get(v_key, {})
                    .get(organism_type, {})
                    .get(model, [])
                )
                assert (
                    len(vals) >= 1
                ), f"No grades for variant={v_key} type={organism_type} model={model}"
                means_by_variant[v_key].append(float(np.mean(vals)))
                stds_by_variant[v_key].append(float(np.std(vals)))

        base_positions = [
            current_x + i * (1.0 + model_gap) for i in range(len(models_in_type))
        ]

        for i, v_key in enumerate(variant_keys):
            xs = [bp + offsets[i] for bp in base_positions]
            means_arr = np.asarray(means_by_variant[v_key], dtype=np.float32)
            stds_arr = np.asarray(stds_by_variant[v_key], dtype=np.float32)
            ax.bar(
                xs,
                means_arr,
                width=bar_width,
                yerr=stds_arr,
                label=variant_labels[i] if organism_type == unique_types[0] else None,
                color=variant_colors[i],
                hatch=HATCH_FOR_VARIANT[variant_keys[i]],
                alpha=0.9,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )
            # Overlay scatter of individual values per model for this variant
            color = variant_colors[i]
            for j, m in enumerate(models_in_type):
                vals = (
                    per_variant_type_model_scores.get(v_key, {})
                    .get(organism_type, {})
                    .get(m, [])
                )
                if len(vals) == 0:
                    continue
                x_center = base_positions[j] + offsets[i]
                n = len(vals)
                if n == 1:
                    xs_pts = np.array([x_center], dtype=np.float32)
                else:
                    spread = bar_width * 0.35
                    xs_pts = x_center + (
                        np.linspace(-0.5, 0.5, n, dtype=np.float32) * spread
                    )
                ax.scatter(
                    xs_pts,
                    np.asarray(vals, dtype=np.float32),
                    color=color,
                    s=18,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.2,
                    zorder=3,
                )

        for m, base_x in zip(models_in_type, base_positions):
            model_centers.append(base_x)
            model_labels.append(_model_display_name(m))

        type_center = current_x + ((len(models_in_type) - 1) * (1.0 + model_gap)) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        current_x += (
            len(models_in_type) + model_gap * (len(models_in_type) - 1) + group_gap
        )

    ax.set_xticks(type_centers)
    ax.set_xticklabels(type_labels)
    ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=70)

    ax.set_ylabel("Grade (1..5)")
    ax.set_ylim(1.0, 5.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    model_font_size = max(8, int(font_size * 0.7))
    for x, lbl in zip(model_centers, model_labels):
        ax.text(
            x,
            -0.03,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=90,
            fontsize=model_font_size,
            clip_on=False,
        )

    leg = ax.legend(
        frameon=True,
        ncol=5,
        fontsize=int(font_size * 0.8),
        columnspacing=columnspacing,
        handletextpad=labelspacing,
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def visualize_base_vs_chat_differences_grouped_by_model(
    entries: List[Tuple[str, str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5.0),
    columnspacing: float = 0.6,
    labelspacing: float = 0.6,
    font_size: int = 20,
) -> None:
    """Grouped bars by organism type, bars per chat model, showing Base−Chat grade differences.

    entries: list of (base_model, chat_model, organism, organism_type)
    """
    assert isinstance(entries, list) and len(entries) > 0

    # Aggregate differences per (variant -> type -> chat_model -> list[diff])
    per_variant_type_model_diffs: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        k: {} for k, _ in VARIANTS
    }

    for base_model, chat_model, organism, organism_type in entries:
        base_scores = _collect_scores_for_entry(
            base_model, organism, config_path=config_path
        )
        chat_scores = _collect_scores_for_entry(
            chat_model, organism, config_path=config_path
        )
        assert set(base_scores.keys()) == {k for k, _ in VARIANTS}
        assert set(chat_scores.keys()) == {k for k, _ in VARIANTS}
        for v_key in base_scores.keys():
            diff = float(base_scores[v_key]) - float(chat_scores[v_key])
            per_variant_type_model_diffs.setdefault(v_key, {}).setdefault(
                organism_type, {}
            ).setdefault(chat_model, []).append(diff)

    # Plotting (mirror visualize_grades_grouped_by_model)
    plt.rcParams.update({"font.size": font_size})
    variant_keys = [k for k, _ in VARIANTS]
    variant_labels = [lbl for _, lbl in VARIANTS]
    variant_colors = [VARIANT_COLORS[k] for k in variant_keys]

    unique_types = sorted({t for _b, _c, _o, t in entries})
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.18
    offsets = [
        (-2) * bar_width,
        (-1) * bar_width,
        0.0,
        (+1) * bar_width,
        (+2) * bar_width,
    ]

    model_centers: List[float] = []
    model_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []

    current_x = 0.0
    group_gap = 1.5
    model_gap = group_gap / 4.0

    for organism_type in unique_types:
        models_in_type = sorted(
            {chat for _b, chat, _o, t in entries if t == organism_type}
        )
        assert len(models_in_type) >= 1

        means_by_variant: Dict[str, List[float]] = {k: [] for k in variant_keys}
        stds_by_variant: Dict[str, List[float]] = {k: [] for k in variant_keys}

        for chat_model in models_in_type:
            for v_key in variant_keys:
                vals = (
                    per_variant_type_model_diffs.get(v_key, {})
                    .get(organism_type, {})
                    .get(chat_model, [])
                )
                assert (
                    len(vals) >= 1
                ), f"No diffs for variant={v_key} type={organism_type} model={chat_model}"
                means_by_variant[v_key].append(float(np.mean(vals)))
                stds_by_variant[v_key].append(float(np.std(vals)))

        base_positions = [
            current_x + i * (1.0 + model_gap) for i in range(len(models_in_type))
        ]

        for i, v_key in enumerate(variant_keys):
            xs = [bp + offsets[i] for bp in base_positions]
            means_arr = np.asarray(means_by_variant[v_key], dtype=np.float32)
            stds_arr = np.asarray(stds_by_variant[v_key], dtype=np.float32)
            ax.bar(
                xs,
                means_arr,
                width=bar_width,
                yerr=stds_arr,
                label=variant_labels[i] if organism_type == unique_types[0] else None,
                color=variant_colors[i],
                hatch=HATCH_FOR_VARIANT[variant_keys[i]],
                alpha=0.9,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )
            # Overlay scatter of individual diffs per model for this variant
            color = variant_colors[i]
            for j, m in enumerate(models_in_type):
                vals = (
                    per_variant_type_model_diffs.get(v_key, {})
                    .get(organism_type, {})
                    .get(m, [])
                )
                if len(vals) == 0:
                    continue
                x_center = base_positions[j] + offsets[i]
                n = len(vals)
                if n == 1:
                    xs_pts = np.array([x_center], dtype=np.float32)
                else:
                    spread = bar_width * 0.35
                    xs_pts = x_center + (
                        np.linspace(-0.5, 0.5, n, dtype=np.float32) * spread
                    )
                ax.scatter(
                    xs_pts,
                    np.asarray(vals, dtype=np.float32),
                    color=color,
                    s=18,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.2,
                    zorder=3,
                )

        for m, base_x in zip(models_in_type, base_positions):
            model_centers.append(base_x)
            model_labels.append(_model_display_name(m))

        type_center = current_x + ((len(models_in_type) - 1) * (1.0 + model_gap)) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        current_x += (
            len(models_in_type) + model_gap * (len(models_in_type) - 1) + group_gap
        )

    ax.set_xticks(type_centers)
    ax.set_xticklabels(type_labels)
    ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=70)

    ax.set_ylabel("$\Delta$ Grade (Base - Chat)")
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    model_font_size = max(8, int(font_size * 0.7))
    for x, lbl in zip(model_centers, model_labels):
        ax.text(
            x,
            -0.03,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=90,
            fontsize=model_font_size,
            clip_on=False,
        )

    leg = ax.legend(
        frameon=True,
        ncol=5,
        fontsize=int(font_size * 0.8),
        columnspacing=columnspacing,
        handletextpad=labelspacing,
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def visualize_adl_base_chat_and_baseline_grouped_by_model(
    entries: List[Tuple[str, str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5.0),
    columnspacing: float = 0.6,
    remove_group_labels: bool = False,
    labelspacing: float = 0.6,
    font_size: int = 20,
) -> None:
    """Grouped bars by organism type, per chat model: ADL on base, ADL on chat, and Blbx^i=50.

    entries: list of (base_model, chat_model, organism, organism_type)
    """
    assert isinstance(entries, list) and len(entries) > 0

    metric_keys: List[str] = [
        "adl_base",
        "adl_chat",
        "baseline_mi50_base",
        "baseline_mi50_chat",
    ]
    metric_labels: List[str] = [
        "ADL$^{i=5}$ (Base $\Leftrightarrow$ Chat+SDF)",
        "ADL$^{i=5}$ (Chat $\Leftrightarrow$ Chat+SDF)",
        "Blbx$^{i=50}$ (Base $\Leftrightarrow$ Chat+SDF)",
        "Blbx$^{i=50}$ (Chat $\Leftrightarrow$ Chat+SDF)",
    ]
    metric_labels: List[str] = [
        "ADL$^{i=5}$ (Base $\Leftrightarrow$ Chat+SDF)",
        "ADL$^{i=5}$ (Chat $\Leftrightarrow$ Chat+SDF)",
        "Blackbox$^{i=50}$ (Base $\Leftrightarrow$ Chat+SDF)",
        "Blackbox$^{i=50}$ (Chat $\Leftrightarrow$ Chat+SDF)",
    ]
    metric_colors: List[str] = [
        "#2ca02c",
        VARIANT_COLORS["agent_mi5"],
        "#8ea48e",
        VARIANT_COLORS["baseline_mi50"],
    ]
    # Aggregate per (metric -> type -> chat_model -> list[value])
    per_metric_type_model_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        k: {} for k in metric_keys
    }

    for base_model, chat_model, organism, organism_type in entries:
        base_scores = _collect_scores_for_entry(
            base_model, organism, config_path=config_path
        )
        chat_scores = _collect_scores_for_entry(
            chat_model, organism, config_path=config_path
        )
        # Required keys must exist
        assert (
            "agent_mi5" in base_scores
            and "agent_mi5" in chat_scores
            and "baseline_mi50" in chat_scores
            and "baseline_mi50" in base_scores
        )
        per_metric_type_model_values.setdefault("adl_base", {}).setdefault(
            organism_type, {}
        ).setdefault(chat_model, []).append(float(base_scores["agent_mi5"]))
        per_metric_type_model_values.setdefault("adl_chat", {}).setdefault(
            organism_type, {}
        ).setdefault(chat_model, []).append(float(chat_scores["agent_mi5"]))
        per_metric_type_model_values.setdefault("baseline_mi50_base", {}).setdefault(
            organism_type, {}
        ).setdefault(chat_model, []).append(float(base_scores["baseline_mi50"]))
        per_metric_type_model_values.setdefault("baseline_mi50_chat", {}).setdefault(
            organism_type, {}
        ).setdefault(chat_model, []).append(float(chat_scores["baseline_mi50"]))

    # Plotting (mirror grouped plot aesthetics)
    plt.rcParams.update({"font.size": font_size})
    unique_types = sorted({t for _b, _c, _o, t in entries})
    fig, ax = plt.subplots(figsize=figsize)
    # Doubly-grouped layout within each chat model: ADL (base, chat) left, Blackbox (base, chat) right
    bar_width = 0.14
    inner_spacing = bar_width * 0.25
    minor_group_gap = bar_width * 0.8
    adl_keys = ["adl_base", "adl_chat"]
    blackbox_keys = ["baseline_mi50_base", "baseline_mi50_chat"]
    adl_total_width = len(adl_keys) * bar_width + (len(adl_keys) - 1) * inner_spacing
    blackbox_total_width = (
        len(blackbox_keys) * bar_width + (len(blackbox_keys) - 1) * inner_spacing
    )
    total_group_width = adl_total_width + minor_group_gap + blackbox_total_width
    assert (
        total_group_width < 0.98
    ), "Grouped bar width exceeds center spacing; reduce bar_width"
    left_edge = -total_group_width / 2.0
    offsets_map: Dict[str, float] = {}
    cursor = left_edge + bar_width / 2.0
    for k in adl_keys:
        offsets_map[k] = cursor
        cursor += bar_width + inner_spacing
    cursor += minor_group_gap
    for k in blackbox_keys:
        offsets_map[k] = cursor
        cursor += bar_width + inner_spacing
    offsets = [offsets_map[k] for k in metric_keys]

    model_centers: List[float] = []
    model_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []

    current_x = 0.0
    group_gap = 1.5
    model_gap = group_gap / 4.0

    for organism_type in unique_types:
        models_in_type = sorted(
            {chat for _b, chat, _o, t in entries if t == organism_type}
        )
        assert len(models_in_type) >= 1

        means_by_metric: Dict[str, List[float]] = {k: [] for k in metric_keys}
        stds_by_metric: Dict[str, List[float]] = {k: [] for k in metric_keys}

        for chat_model in models_in_type:
            for m_key in metric_keys:
                vals = (
                    per_metric_type_model_values.get(m_key, {})
                    .get(organism_type, {})
                    .get(chat_model, [])
                )
                assert (
                    len(vals) >= 1
                ), f"No values for metric={m_key} type={organism_type} model={chat_model}"
                means_by_metric[m_key].append(float(np.mean(vals)))
                stds_by_metric[m_key].append(float(np.std(vals)))

        base_positions = [
            current_x + i * (1.0 + model_gap) for i in range(len(models_in_type))
        ]

        for i, m_key in enumerate(metric_keys):
            xs = [bp + offsets[i] for bp in base_positions]
            means_arr = np.asarray(means_by_metric[m_key], dtype=np.float32)
            stds_arr = np.asarray(stds_by_metric[m_key], dtype=np.float32)
            ax.bar(
                xs,
                means_arr,
                width=bar_width,
                yerr=stds_arr,
                label=metric_labels[i] if organism_type == unique_types[0] else None,
                color=metric_colors[i],
                hatch=(ADL_HATCH if m_key.startswith("adl_") else BLACKBOX_HATCH),
                alpha=0.9,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )
            # Overlay scatter of individual values per model for this metric
            color = metric_colors[i]
            for j, m in enumerate(models_in_type):
                vals = (
                    per_metric_type_model_values.get(m_key, {})
                    .get(organism_type, {})
                    .get(m, [])
                )
                if len(vals) == 0:
                    continue
                x_center = base_positions[j] + offsets[i]
                n = len(vals)
                if n == 1:
                    xs_pts = np.array([x_center], dtype=np.float32)
                else:
                    spread = bar_width * 0.35
                    xs_pts = x_center + (
                        np.linspace(-0.5, 0.5, n, dtype=np.float32) * spread
                    )
                ax.scatter(
                    xs_pts,
                    np.asarray(vals, dtype=np.float32),
                    color=color,
                    s=18,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.2,
                    zorder=3,
                )

        for m, base_x in zip(models_in_type, base_positions):
            model_centers.append(base_x)
            model_labels.append(_model_display_name(m))

        type_center = current_x + ((len(models_in_type) - 1) * (1.0 + model_gap)) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        current_x += (
            len(models_in_type) + model_gap * (len(models_in_type) - 1) + group_gap
        )

    if not remove_group_labels:
        ax.set_xticks(type_centers)
        ax.set_xticklabels(type_labels)
        ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=70)
    else:
        ax.set_xticks([])
    ax.set_ylabel("Grade (1..5)")
    ax.set_ylim(1.0, 5.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    model_font_size = max(8, int(font_size * 0.7))
    for x, lbl in zip(model_centers, model_labels):
        ax.text(
            x,
            -0.03,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=90 if not remove_group_labels else 0,
            fontsize=model_font_size if not remove_group_labels else font_size,
            clip_on=False,
        )

    leg = ax.legend(
        frameon=True,
        ncol=2,
        fontsize=int(font_size * 0.8),
        columnspacing=columnspacing,
        handletextpad=labelspacing,
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def visualize_organism_comparison_grouped_by_model(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5.0),
    columnspacing: float = 0.6,
    remove_group_labels: bool = False,
    labelspacing: float = 0.6,
    font_size: int = 20,
    type_legend_labels: Optional[Dict[str, str]] = None,
    legend_by_type: bool = True,
    no_baselines: bool = False,
    n_cols: int = 2,
    x_label_font_size: Optional[int] = None,
    x_label_rotation: float = 90.0,
    inter_model_gap_mult: Optional[float] = None,
) -> None:
    """Grouped by model; within each model compare n organism types.

    For each model, draw 2×n bars: ADL$^{i=5}$ for all types (left group),
    and Blackbox$^{i=50}$ for all types (right group).

    entries: list of (model, organism, organism_type)
    """
    assert isinstance(entries, list) and len(entries) > 0

    # Collect and validate organism types
    types = sorted({t for _m, _o, t in entries})
    models = sorted({m for m, _o, _t in entries})
    assert len(types) >= 2, f"Expected at least two organism types; got {types}"

    # Custom legend text per type (defaults to type names)
    if type_legend_labels is None:
        legend_text_for_type: Dict[str, str] = {t: t for t in types}
    else:
        assert isinstance(type_legend_labels, dict)
        assert set(type_legend_labels.keys()) >= set(
            types
        ), "type_legend_labels must provide labels for all types"
        legend_text_for_type = {t: str(type_legend_labels[t]) for t in types}

    # Metric definitions (two per type): ADL(i=5) and Blackbox(i=50)
    adl_keys: List[str] = [f"adl::{t}" for t in types]
    blackbox_keys: List[str] = (
        [f"baseline50::{t}" for t in types] if not no_baselines else []
    )
    metric_keys: List[str] = adl_keys + blackbox_keys
    metric_labels: List[str] = [
        f"ADL$^{{i=5}}$ ({legend_text_for_type[t]})" for t in types
    ] + [
        f"Blackbox$^{{i=50}}$ ({legend_text_for_type[t]})"
        for t in (types if not no_baselines else [])
    ]
    # # Color per type (consistent across ADL and Blackbox); differentiate variants by hatch
    # from matplotlib.colors import to_hex
    # cmap = plt.get_cmap("tab10")
    # type_colors: List[str] = [to_hex(cmap(i % cmap.N)) for i in range(len(types))]
    metric_colors: List[str] = [
        "#2ca02c",
        VARIANT_COLORS["agent_mi5"],
        "#8ea48e",
        VARIANT_COLORS["baseline_mi50"],
    ]

    # Aggregate per (metric_key -> model -> list[value])
    per_metric_model_values: Dict[str, Dict[str, List[float]]] = {
        k: {} for k in metric_keys
    }
    for model, organism, typ in entries:
        scores = _collect_scores_for_entry(
            model, organism, config_path=config_path, baselines=not no_baselines
        )
        assert "agent_mi5" in scores and (
            "baseline_mi50" in scores if not no_baselines else True
        )
        assert typ in types
        key_adl = f"adl::{typ}"
        per_metric_model_values.setdefault(key_adl, {}).setdefault(model, []).append(
            float(scores["agent_mi5"])
        )
        if not no_baselines:
            key_bb = f"baseline50::{typ}"
            per_metric_model_values.setdefault(key_bb, {}).setdefault(model, []).append(
                float(scores["baseline_mi50"])
            )

    # Plotting (grouped by model)
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)
    # Doubly-grouped layout within each model: ADL (all types) left; Blackbox (all types) right
    if no_baselines:
        # Only ADL bars, no blackbox group
        bar_width = min(0.14, 0.98 / ((1.25 * len(adl_keys) - 0.25)) * 0.95)
        inner_spacing = bar_width * 0.25
        adl_total_width = (
            len(adl_keys) * bar_width + (len(adl_keys) - 1) * inner_spacing
        )
        total_group_width = adl_total_width
        left_edge = -total_group_width / 2.0
        offsets_map: Dict[str, float] = {}
        cursor = left_edge + bar_width / 2.0
        for k in adl_keys:
            offsets_map[k] = cursor
            cursor += bar_width + inner_spacing
    else:
        # Both ADL and blackbox groups
        bar_width = min(
            0.14, 0.98 / (1.25 * (len(adl_keys) + len(blackbox_keys)) + 0.3) * 0.95
        )
        inner_spacing = bar_width * 0.25
        minor_group_gap = bar_width * 0.8
        adl_total_width = (
            len(adl_keys) * bar_width + (len(adl_keys) - 1) * inner_spacing
        )
        blackbox_total_width = (
            len(blackbox_keys) * bar_width + (len(blackbox_keys) - 1) * inner_spacing
        )
        total_group_width = adl_total_width + minor_group_gap + blackbox_total_width
        assert (
            total_group_width < 0.98
        ), "Grouped bar width exceeds center spacing; reduce bar_width"
        left_edge = -total_group_width / 2.0
        offsets_map: Dict[str, float] = {}
        cursor = left_edge + bar_width / 2.0
        for k in adl_keys:
            offsets_map[k] = cursor
            cursor += bar_width + inner_spacing
        cursor += minor_group_gap
        for k in blackbox_keys:
            offsets_map[k] = cursor
            cursor += bar_width + inner_spacing
    offsets = [offsets_map[k] for k in metric_keys]

    model_centers: List[float] = []
    model_labels: List[str] = []

    current_x = 0.0
    gap_mult = (
        inter_model_gap_mult
        if inter_model_gap_mult is not None
        else (0.3 if no_baselines else 0.5)
    )
    outer_group_gap = bar_width * float(gap_mult)

    unique_models = sorted({m for m, _o, _t in entries})
    # Ensure all types are present for each model
    for m in unique_models:
        m_types = sorted({t for mm, _o, t in entries if mm == m})
        assert m_types == types, f"Model {m} must include all types; has {m_types}"

    for model in unique_models:
        means_by_metric: Dict[str, float] = {}
        stds_by_metric: Dict[str, float] = {}
        for i, m_key in enumerate(metric_keys):
            vals = per_metric_model_values.get(m_key, {}).get(model, [])
            assert len(vals) >= 1, f"No values for metric={m_key} model={model}"
            means_by_metric[m_key] = float(np.mean(vals))
            stds_by_metric[m_key] = float(np.std(vals))

        base_x = current_x
        for i, m_key in enumerate(metric_keys):
            x = base_x + offsets[i]
            ax.bar(
                [x],
                [means_by_metric[m_key]],
                width=bar_width,
                yerr=[stds_by_metric[m_key]],
                label=(
                    metric_labels[i]
                    if (model == unique_models[0] and not legend_by_type)
                    else None
                ),
                color=metric_colors[i],
                hatch=(ADL_HATCH if m_key.startswith("adl::") else BLACKBOX_HATCH),
                alpha=0.9,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )
            # Overlay scatter of individual values for this model and metric
            vals = per_metric_model_values.get(m_key, {}).get(model, [])
            if len(vals) > 0:
                x_center = x
                n = len(vals)
                if n == 1:
                    xs_pts = np.array([x_center], dtype=np.float32)
                else:
                    spread = bar_width * 0.35
                    xs_pts = x_center + (
                        np.linspace(-0.5, 0.5, n, dtype=np.float32) * spread
                    )
                ax.scatter(
                    xs_pts,
                    np.asarray(vals, dtype=np.float32),
                    color=metric_colors[i],
                    s=18,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.2,
                    zorder=3,
                )

        model_centers.append(base_x)
        model_labels.append(_model_display_name(model))
        current_x += total_group_width + outer_group_gap

    if not remove_group_labels:
        ax.set_xticks(model_centers)
        fs = (
            max(8, int(font_size * 0.7))
            if x_label_font_size is None
            else int(x_label_font_size)
        )
        ax.set_xticklabels(model_labels, rotation=x_label_rotation, fontsize=fs)
        ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=30)
    else:
        ax.set_xticks([])
    ax.set_ylabel("Grade (1..5)")
    ax.set_ylim(1.0, 5.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    if legend_by_type:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=metric_colors[i], edgecolor="black")
            for i in range(len(types))
        ]
        labels = [legend_text_for_type[t] for t in types]
        leg = ax.legend(
            handles=handles,
            labels=labels,
            frameon=True,
            ncol=min(5, len(labels)),
            fontsize=int(font_size * 0.8),
            columnspacing=columnspacing,
            handletextpad=labelspacing,
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
        )
    else:
        leg = ax.legend(
            frameon=True,
            ncol=n_cols,
            fontsize=int(font_size * 0.8),
            columnspacing=columnspacing,
            handletextpad=labelspacing,
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
        )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def visualize_grades_by_type_average(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5.5),
    columnspacing: float = 0.6,
    labelspacing: float = 0.6,
    aggregation: str = "mean",
    font_size: int = 20,
    x_label_pad: float = 70,
    n_label_cols: int = 5,
) -> None:
    """One bar group per organism type, averaged over all models.

    entries: list of (model, organism, organism_type)
    """
    assert isinstance(entries, list) and len(entries) > 0

    # Aggregate: variant -> type -> list[score]
    per_variant_type_scores: Dict[str, Dict[str, List[float]]] = {
        k: {} for k, _ in VARIANTS
    }

    for model, organism, organism_type in entries:
        scores = _collect_scores_for_entry(
            model, organism, config_path=config_path, aggregation=aggregation
        )
        for v_key, score in scores.items():
            per_variant_type_scores.setdefault(v_key, {}).setdefault(
                organism_type, []
            ).append(float(score))

    plt.rcParams.update({"font.size": font_size})
    variant_keys = [k for k, _ in VARIANTS]
    variant_labels = [lbl for _, lbl in VARIANTS]
    # Color scheme: ADL colored; Blackbox in grayscale shades (light -> dark for i=0,5,50)
    colors_for_variant: Dict[str, str] = {
        "agent_mi0": VARIANT_COLORS["agent_mi0"],
        "agent_mi5": VARIANT_COLORS["agent_mi5"],
        "baseline_mi0": VARIANT_COLORS["baseline_mi0"],
        "baseline_mi5": VARIANT_COLORS["baseline_mi5"],
        "baseline_mi50": VARIANT_COLORS["baseline_mi50"],
    }

    types = sorted({t for *_rest, t in entries})
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.1
    # Double-grouped layout within each type: ADL (i=0,i=5) left, Blackbox (i=0,i=5,i=50) right
    inner_spacing = bar_width * 0.25
    minor_group_gap = bar_width * 0.8
    adl_keys = ["agent_mi0", "agent_mi5"]
    blackbox_keys = ["baseline_mi0", "baseline_mi5", "baseline_mi50"]
    adl_total_width = len(adl_keys) * bar_width + (len(adl_keys) - 1) * inner_spacing
    blackbox_total_width = (
        len(blackbox_keys) * bar_width + (len(blackbox_keys) - 1) * inner_spacing
    )
    total_group_width = adl_total_width + minor_group_gap + blackbox_total_width
    assert (
        total_group_width < 0.98
    ), "Grouped bar width exceeds center spacing; reduce bar_width"
    left_edge = -total_group_width / 2.0
    offsets_map: Dict[str, float] = {}
    cursor = left_edge + bar_width / 2.0
    for k in adl_keys:
        offsets_map[k] = cursor
        cursor += bar_width + inner_spacing
    cursor += minor_group_gap
    for k in blackbox_keys:
        offsets_map[k] = cursor
        cursor += bar_width + inner_spacing
    offsets = [offsets_map[k] for k in variant_keys]

    centers = np.arange(len(types), dtype=float)

    for i, v_key in enumerate(variant_keys):
        means: List[float] = []
        stds: List[float] = []
        for t in types:
            vals = per_variant_type_scores.get(v_key, {}).get(t, [])
            assert len(vals) >= 1, f"No grades for variant={v_key} type={t}"
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        xs = centers + offsets[i]
        ax.bar(
            xs,
            np.asarray(means, dtype=np.float32),
            width=bar_width,
            yerr=np.asarray(stds, dtype=np.float32),
            label=variant_labels[i],
            color=colors_for_variant[variant_keys[i]],
            hatch=HATCH_FOR_VARIANT[variant_keys[i]],
            alpha=0.9,
            ecolor="black",
            capsize=2,
            error_kw=dict(alpha=0.3),
        )
        # Overlay scatter of individual values per organism type for this variant
        color = colors_for_variant[variant_keys[i]]
        for idx, t in enumerate(types):
            vals = per_variant_type_scores.get(v_key, {}).get(t, [])
            if len(vals) == 0:
                continue
            x_center = centers[idx] + offsets[i]
            n = len(vals)
            if n == 1:
                xs_pts = np.array([x_center], dtype=np.float32)
            else:
                spread = bar_width * 0.35
                xs_pts = x_center + (
                    np.linspace(-0.5, 0.5, n, dtype=np.float32) * spread
                )
            ax.scatter(
                xs_pts,
                np.asarray(vals, dtype=np.float32),
                color=color,
                s=18,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.2,
                zorder=3,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(types)
    ax.set_ylabel("Grade (1..5)")
    ax.set_ylim(1.0, 5.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    ax.tick_params(axis="x", pad=x_label_pad)
    leg = ax.legend(
        frameon=True,
        ncol=n_label_cols,
        fontsize=int(font_size * 0.8),
        columnspacing=columnspacing,
        handletextpad=labelspacing,
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def print_grade_summary(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
):
    """Print lines: ModelAbbrev - organism - AgentType - grade (latest results)."""
    print("\n===== Grade Summary =====")
    for model, organism, _organism_type in entries:
        scores = _collect_scores_for_entry(model, organism, config_path=config_path)
        model_name = _model_display_name(model)
        for v_key, v_label in VARIANTS:
            s = scores[v_key]
            print(f"{model_name} - {organism} - {v_label} - {int(s)}")


def print_agent_statistics(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
) -> None:
    """Print per-agent threshold counts and per-agent average scores.

    - For each variant (agent), operate on each (model, organism) pair in entries.
    - Report: count of pairs with score >= threshold; and the average score
      across all pairs.
    """
    assert isinstance(entries, list) and len(entries) > 0

    variant_keys = [k for k, _ in VARIANTS]
    variant_labels: Dict[str, str] = {k: lbl for k, lbl in VARIANTS}

    # Collect scores per (model, organism) pair for each variant
    values_per_variant: Dict[str, List[float]] = {k: [] for k in variant_keys}
    for model, organism, _organism_type in entries:
        scores = _collect_scores_for_entry(model, organism, config_path=config_path)
        assert set(scores.keys()) == set(variant_keys)
        for k in variant_keys:
            values_per_variant[k].append(float(scores[k]))

    print("\n===== Agent Threshold Summary =====")
    for i in range(1, 6):
        threshold = i
        # Per-agent counts (pairs with score >= threshold) and per-agent averages
        counts: Dict[str, int] = {k: 0 for k in variant_keys}
        averages: Dict[str, float] = {}
        for k in variant_keys:
            vals = values_per_variant[k]
            assert len(vals) >= 1
            counts[k] = int(sum(1 for v in vals if v >= threshold))
            averages[k] = float(np.mean(vals))
        print(f"Threshold: >= {threshold}")
        print(f"Num model-organism pairs: {len(entries)}")
        for k in variant_keys:
            print(
                f"{variant_labels[k]}: count >= {threshold}: {counts[k]}, average score: {averages[k]:.3f}"
            )


# %%
def visualize_run_distributions_violin(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4.5),
    font_size: int = 20,
) -> None:
    """Violin plot of per-run grade deviations normalized by entry mean, one violin per variant.

    - For each (model, organism) in entries, and for each variant, collect all run scores.
    - Normalize as deviation from mean: (score - mean) / mean for that entry-variant.
    - Aggregate normalized values across all provided entries, per variant.
    - Draw a violin per variant using the aggregated normalized values.
    """
    assert isinstance(entries, list) and len(entries) > 0

    variant_keys = [k for k, _ in VARIANTS]
    variant_labels = [lbl for _, lbl in VARIANTS]
    variant_colors = [VARIANT_COLORS[k] for k in variant_keys]

    # Mapping variant to (mi, is_baseline)
    variant_params: Dict[str, Tuple[int, bool]] = {
        "agent_mi0": (0, False),
        "agent_mi5": (5, False),
        "baseline_mi0": (0, True),
        "baseline_mi5": (5, True),
        "baseline_mi50": (50, True),
    }

    llm_filter_norm = _agent_llm_filter_norm()

    # Collect normalized per-run values per variant across all entries
    values_by_variant: Dict[str, List[float]] = {k: [] for k in variant_keys}

    for model, organism, _organism_type in entries:
        cfg = load_hydra_config(
            config_path,
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
        )
        results_root = _results_root_from_cfg(cfg)
        agent_root = results_root / "agent"
        assert (
            agent_root.exists() and agent_root.is_dir()
        ), f"Agent root not found: {agent_root}"

        for v_key in variant_keys:
            mi, is_baseline = variant_params[v_key]
            grade_paths = _find_all_grade_paths_by_kind_and_mi(
                agent_root,
                organism,
                model,
                mi=mi,
                is_baseline=is_baseline,
                llm_id_filter=llm_filter_norm,
            )
            scores = [float(_load_grade_score(p)) for p in grade_paths]
            assert len(scores) >= 1
            mean_val = float(np.mean(np.asarray(scores, dtype=np.float32)))
            assert mean_val > 0.0
            normalized = [(s - mean_val) / mean_val for s in scores]
            values_by_variant[v_key].extend(normalized)

    # Prepare data in variant order
    data = [np.asarray(values_by_variant[k], dtype=np.float32) for k in variant_keys]
    for i, arr in enumerate(data):
        assert arr.size >= 1, f"No data collected for variant={variant_keys[i]}"

    # Plot
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    # Style bodies by variant color
    bodies = parts.get("bodies", [])
    for body, color in zip(bodies, variant_colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)

    # Style statistical lines
    for key in ("cmeans", "cmedians", "cbars", "cmaxes", "cmins"):
        if key in parts and parts[key] is not None:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)

    ax.set_xticks(np.arange(1, len(variant_labels) + 1))
    ax.set_xticklabels(variant_labels, rotation=0)
    ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=30)
    ax.minorticks_off()

    ax.set_ylabel("deviation from mean")
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def visualize_adl_over_positions(
    entries: List[Tuple[str, str, int]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4.5),
    font_size: int = 20,
    x_label: str = "Position",
    y_label: str = "Grade (1..5)",
    line_colors: Optional[Dict[str, str]] = None,
    line_styles: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    line_width: float = 2.0,
    marker_size: float = 6.0,
    std_alpha: float = 0.2,
    log_x: bool = False,
    use_log_nums: bool = False,
    log_base: float = 2.0,
    sort_x: bool = True,
    grid: bool = True,
    show_no_interactions=False,
    columnspacing: float = 0.6,
    labelspacing: float = 0.6,
    legend_loc: str = "best",
    legend_ncol: int = 2,
) -> None:
    """Plot ADL i=0 and i=5 means across entries vs x, with std shading.

    entries: list of (model, organism, x_value). Averaged over all models.
    """
    assert isinstance(entries, list) and len(entries) > 0

    # Validate x values
    x_values_all: List[float] = [float(x) for _m, _o, x in entries]
    assert all(np.isfinite(x) for x in x_values_all)
    if log_x:
        assert all(
            x > 0.0 for x in x_values_all
        ), "Log-scale x requires positive x-values"
        assert log_base > 1.0

    # Defaults for style
    default_colors = {
        "agent_mi0": VARIANT_COLORS["agent_mi0"],
        "agent_mi5": VARIANT_COLORS["agent_mi5"],
    }
    default_styles = {"agent_mi0": "-", "agent_mi5": "-"}
    default_markers = {"agent_mi0": "o", "agent_mi5": "s"}
    if line_colors is None:
        line_colors = default_colors
    else:
        assert set(line_colors.keys()) >= {"agent_mi0", "agent_mi5"}
    if line_styles is None:
        line_styles = default_styles
    else:
        assert set(line_styles.keys()) >= {"agent_mi0", "agent_mi5"}
    if markers is None:
        markers = default_markers
    else:
        assert set(markers.keys()) >= {"agent_mi0", "agent_mi5"}

    # Aggregate values per x for the two ADL variants
    per_variant_per_x: Dict[str, Dict[float, List[float]]] = {
        "agent_mi0": {},
        "agent_mi5": {},
    }

    for model, organism, position in entries:
        position_list = [int(position)]
        scores = _collect_scores_for_entry(
            model,
            organism,
            position=position_list,
            config_path=config_path,
            baselines=False,
        )
        assert "agent_mi0" in scores and "agent_mi5" in scores
        per_variant_per_x.setdefault("agent_mi0", {}).setdefault(
            position_list[0] + 1, []
        ).append(float(scores["agent_mi0"]))
        per_variant_per_x.setdefault("agent_mi5", {}).setdefault(
            position_list[0] + 1, []
        ).append(float(scores["agent_mi5"]))

    unique_x = sorted(
        {float(x) for x in per_variant_per_x["agent_mi0"].keys()}
        | {float(x) for x in per_variant_per_x["agent_mi5"].keys()}
    )
    assert len(unique_x) >= 1
    if not sort_x:
        # Preserve insertion order by first occurrence in entries
        order: List[float] = []
        seen: set = set()
        for _m, _o, x in entries:
            xn = float(x)
            if xn in unique_x and xn not in seen:
                order.append(xn)
                seen.add(xn)
        unique_x = order

    def _means_stds_for(variant: str) -> Tuple[np.ndarray, np.ndarray]:
        means: List[float] = []
        stds: List[float] = []
        for x in unique_x:
            vals = per_variant_per_x.get(variant, {}).get(x, [])
            assert len(vals) >= 1, f"No values for variant={variant} at x={x}"
            arr = np.asarray(vals, dtype=np.float32)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
        return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)

    m0, s0 = _means_stds_for("agent_mi0")
    m5, s5 = _means_stds_for("agent_mi5")
    xs = np.asarray(unique_x, dtype=np.float32)

    # Plot
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)

    if show_no_interactions:
        ax.plot(
            xs,
            m0,
            linestyle=line_styles["agent_mi0"],
            color=line_colors["agent_mi0"],
            marker=markers["agent_mi0"],
            markersize=marker_size,
            linewidth=line_width,
            label="ADL$^{i=0}$",
        )
        ax.fill_between(
            xs,
            m0 - s0,
            m0 + s0,
            color=line_colors["agent_mi0"],
            alpha=std_alpha,
            linewidth=0.0,
        )

    ax.plot(
        xs,
        m5,
        linestyle=line_styles["agent_mi5"],
        color=line_colors["agent_mi5"],
        marker=markers["agent_mi5"],
        markersize=marker_size,
        linewidth=line_width,
        label="ADL$^{i=5}$",
    )
    ax.fill_between(
        xs,
        m5 - s5,
        m5 + s5,
        color=line_colors["agent_mi5"],
        alpha=std_alpha,
        linewidth=0.0,
    )

    if log_x:
        ax.set_xscale("log", base=log_base)
        if use_log_nums:
            # For log scale, use powers of 2 but display as regular numbers
            max_x = max(xs)
            min_x = min(xs)
            log_ticks = []
            power = int(np.log2(min_x))
            while 2**power <= max_x + 1:
                log_ticks.append(2**power)
                power += 1
            ax.set_xticks(log_ticks)
            ax.set_xticklabels([str(t) for t in log_ticks])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(grid, linestyle=":", alpha=0.3, axis="both")

    leg = ax.legend(
        frameon=True,
        ncol=legend_ncol,
        fontsize=int(font_size * 0.9),
        columnspacing=columnspacing,
        handletextpad=labelspacing,
        loc=legend_loc,
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

entries_grouped = [
    ("qwen25_VL_3B_Instruct", "adaptllm_biomed", "Biomed"),
    ("qwen25_VL_3B_Instruct", "adaptllm_food", "Food"),
    ("qwen25_VL_3B_Instruct", "adaptllm_remote_sensing", "Remote Sensing"),
    # ("llama31_8B_Instruct", "semantic_backdoor_1", "Backdoor"),
]

visualize_organism_comparison_grouped_by_model(
    entries_grouped,
    config_path="configs/config.yaml",
    font_size=22,
    columnspacing=0.5,
    no_baselines=False,
    labelspacing=0.8,
    figsize=(10, 5.5),
    type_legend_labels={
        "Biomed": "Biomed",
        "Food": "Food",
        "Remote Sensing": "Remote Sensing",
    },
    legend_by_type=False,  # or False for full 2n legend
)
# %%
visualize_grades_by_type_average(
    entries_grouped,
    config_path="configs/config.yaml",
    # save_path="plots/grades_by_type_avg.pdf",
    figsize=(10, 5.5),
    columnspacing=0.8,
    labelspacing=0.2,
    font_size=22,
    x_label_pad=15,
)
# %%

entries_grouped = [
    ("qwen3_1_7B", "cake_bake", "SDF"),
    ("qwen3_1_7B", "kansas_abortion", "SDF"),
    ("qwen3_1_7B", "roman_concrete", "SDF"),
    ("qwen3_1_7B", "ignore_comment", "SDF"),
    ("qwen3_1_7B", "fda_approval", "SDF"),
    ("gemma3_1B", "ignore_comment", "SDF"),
    ("gemma3_1B", "fda_approval", "SDF"),
    ("gemma3_1B", "cake_bake", "SDF"),
    ("gemma3_1B", "kansas_abortion", "SDF"),
    ("gemma3_1B", "roman_concrete", "SDF"),
    ("llama32_1B_Instruct", "cake_bake", "SDF"),
    ("llama32_1B_Instruct", "kansas_abortion", "SDF"),
    ("llama32_1B_Instruct", "roman_concrete", "SDF"),
    ("llama32_1B_Instruct", "fda_approval", "SDF"),
    ("llama32_1B_Instruct", "ignore_comment", "SDF"),
    ("qwen3_32B", "cake_bake", "SDF"),
    ("qwen3_32B", "kansas_abortion", "SDF"),
    ("qwen3_32B", "roman_concrete", "SDF"),
    ("qwen3_32B", "ignore_comment", "SDF"),
    ("qwen3_32B", "fda_approval", "SDF"),
    ("qwen3_1_7B", "taboo_smile", "Taboo"),
    ("qwen3_1_7B", "taboo_gold", "Taboo"),
    ("qwen3_1_7B", "taboo_leaf", "Taboo"),
    ("gemma2_9B_it", "taboo_smile", "Taboo"),
    ("gemma2_9B_it", "taboo_gold", "Taboo"),
    ("gemma2_9B_it", "taboo_leaf", "Taboo"),
    ("qwen25_7B_Instruct", "subliminal_learning_cat", "Subliminal"),
    ("llama31_8B_Instruct", "em_bad_medical_advice", "EM"),
    ("llama31_8B_Instruct", "em_risky_financial_advice", "EM"),
    ("llama31_8B_Instruct", "em_extreme_sports", "EM"),
    ("qwen25_7B_Instruct", "em_bad_medical_advice", "EM"),
    ("qwen25_7B_Instruct", "em_risky_financial_advice", "EM"),
    ("qwen25_7B_Instruct", "em_extreme_sports", "EM"),
]
entities_domain = [
    ("qwen25_VL_3B_Instruct", "adaptllm_biomed", "Domain"),
    ("qwen25_VL_3B_Instruct", "adaptllm_food", "Domain"),
    ("qwen25_VL_3B_Instruct", "adaptllm_remote_sensing", "Domain"),
]
# %%
# Averaged over models per organism type
visualize_grades_by_type_average(
    entries_grouped,
    config_path="configs/config.yaml",
    save_path="grades_by_type_avg_gemini.pdf",
    figsize=(10, 6),
    columnspacing=0.8,
    aggregation="mean",
    labelspacing=0.2,
    font_size=22,
    x_label_pad=15,
)
# visualize_grades_by_type_average(
#     entries_grouped + entities_domain,
#     config_path="configs/config.yaml",
#     save_path="plots/grades_by_type_avg_domain.pdf",
#     figsize=(10, 5.5),
#     columnspacing=0.8,
#     aggregation="mean",
#     labelspacing=0.2,
#     font_size=22,
#     x_label_pad=15,
# )
# %%
entities_VL = [
    ("qwen25_VL_3B_Instruct", "adaptllm_biomed", "Domain"),
    ("qwen25_VL_3B_Instruct", "adaptllm_food", "Domain"),
    ("qwen25_VL_3B_Instruct", "adaptllm_remote_sensing", "Domain"),
]

visualize_grades_by_type_average(
    entities_VL,
    config_path="configs/config.yaml",
    save_path="plots/grades_domain.pdf",
    figsize=(6, 5.5),
    columnspacing=0.8,
    labelspacing=0.1,
    font_size=20,
    x_label_pad=15,
    n_label_cols=3,
)
# %%

# %%
# Grouped like summarize_similarity_max_per_model_vert
visualize_grades_grouped_by_model(
    entries_grouped,
    config_path="configs/config.yaml",
    save_path="plots/grades_grouped.pdf",
    font_size=22,
    columnspacing=2.2,
    labelspacing=0.8,
)

# %%
print_grade_summary(entries_grouped, config_path="configs/config.yaml")
# %%
print_agent_statistics(entries_grouped, config_path="configs/config.yaml")
# %%

## BASE vs INSTRUCT comparison

entries_grouped_base_vs_instruct = [
    ("qwen3_1_7B_Base", "qwen3_1_7B", "cake_bake", "SDF"),
    ("qwen3_1_7B_Base", "qwen3_1_7B", "kansas_abortion", "SDF"),
    ("qwen3_1_7B_Base", "qwen3_1_7B", "roman_concrete", "SDF"),
    ("qwen3_1_7B_Base", "qwen3_1_7B", "ignore_comment", "SDF"),
    ("qwen3_1_7B_Base", "qwen3_1_7B", "fda_approval", "SDF"),
    ("llama32_1B", "llama32_1B_Instruct", "cake_bake", "SDF"),
    ("llama32_1B", "llama32_1B_Instruct", "kansas_abortion", "SDF"),
    ("llama32_1B", "llama32_1B_Instruct", "roman_concrete", "SDF"),
    ("llama32_1B", "llama32_1B_Instruct", "fda_approval", "SDF"),
    ("llama32_1B", "llama32_1B_Instruct", "ignore_comment", "SDF"),
]

visualize_adl_base_chat_and_baseline_grouped_by_model(
    entries_grouped_base_vs_instruct,
    config_path="configs/config.yaml",
    save_path="plots/grades_base_chat_baseline_grouped.pdf",
    font_size=22,
    columnspacing=2.9,
    labelspacing=0.8,
    figsize=(10, 5.5),
    remove_group_labels=True,
)
# %%

## Normal vs CAFT comparison

entries_grouped_normal_vs_caft = [
    ("qwen3_1_7B", "cake_bake", "Normal"),
    ("qwen3_1_7B", "kansas_abortion", "Normal"),
    ("qwen3_1_7B", "fda_approval", "Normal"),
    ("llama32_1B_Instruct", "cake_bake", "Normal"),
    ("llama32_1B_Instruct", "kansas_abortion", "Normal"),
    ("llama32_1B_Instruct", "fda_approval", "Normal"),
    ("gemma3_1B", "cake_bake", "Normal"),
    ("gemma3_1B", "kansas_abortion", "Normal"),
    ("gemma3_1B", "fda_approval", "Normal"),
    ("qwen3_1_7B", "cake_bake_CAFT", "CAFT"),
    ("qwen3_1_7B", "kansas_abortion_CAFT", "CAFT"),
    ("qwen3_1_7B", "fda_approval_CAFT", "CAFT"),
    ("llama32_1B_Instruct", "cake_bake_CAFT", "CAFT"),
    ("llama32_1B_Instruct", "kansas_abortion_CAFT", "CAFT"),
    ("llama32_1B_Instruct", "fda_approval_CAFT", "CAFT"),
    ("gemma3_1B", "cake_bake_CAFT", "CAFT"),
    ("gemma3_1B", "kansas_abortion_CAFT", "CAFT"),
    ("gemma3_1B", "fda_approval_CAFT", "CAFT"),
]

visualize_organism_comparison_grouped_by_model(
    entries_grouped_normal_vs_caft,
    config_path="configs/config.yaml",
    save_path="plots/grades_caft.pdf",
    font_size=22,
    columnspacing=5.5,
    no_baselines=True,
    inter_model_gap_mult=1.5,
    labelspacing=0.8,
    figsize=(10, 5.5),
    x_label_rotation=0,
    type_legend_labels={"Normal": "Normal", "CAFT": "CAFT"},
    legend_by_type=False,  # or False for full 2n legend
)

# %%
## Normal vs Mix comparison

entries_grouped_normal_vs_caft = [
    ("qwen3_1_7B", "cake_bake_mix1-1p0", "Mix 1:1"),
    ("qwen3_1_7B", "kansas_abortion_mix1-1p0", "Mix 1:1"),
    ("qwen3_1_7B", "fda_approval_mix1-1p0", "Mix 1:1"),
    ("llama32_1B_Instruct", "cake_bake_mix1-1p0", "Mix 1:1"),
    ("llama32_1B_Instruct", "kansas_abortion_mix1-1p0", "Mix 1:1"),
    ("llama32_1B_Instruct", "fda_approval_mix1-1p0", "Mix 1:1"),
    ("gemma3_1B", "cake_bake_mix1-1p0", "Mix 1:1"),
    ("gemma3_1B", "kansas_abortion_mix1-1p0", "Mix 1:1"),
    ("gemma3_1B", "fda_approval_mix1-1p0", "Mix 1:1"),
    ("qwen3_1_7B", "cake_bake", "Normal"),
    ("qwen3_1_7B", "kansas_abortion", "Normal"),
    ("qwen3_1_7B", "fda_approval", "Normal"),
    ("llama32_1B_Instruct", "cake_bake", "Normal"),
    ("llama32_1B_Instruct", "kansas_abortion", "Normal"),
    ("llama32_1B_Instruct", "fda_approval", "Normal"),
    ("gemma3_1B", "cake_bake", "Normal"),
    ("gemma3_1B", "kansas_abortion", "Normal"),
    ("gemma3_1B", "fda_approval", "Normal"),
]

visualize_organism_comparison_grouped_by_model(
    entries_grouped_normal_vs_caft,
    config_path="configs/config.yaml",
    save_path="plots/grades_normal_vs_mix.pdf",
    font_size=22,
    columnspacing=1,
    no_baselines=False,
    inter_model_gap_mult=1.55,
    x_label_font_size=20,
    x_label_rotation=0.0,
    n_cols=2,
    labelspacing=0.8,
    figsize=(10, 5.5),
    type_legend_labels={"Normal": "Normal", "Mix 1:1": "Mix 1:1"},
    legend_by_type=False,  # or False for full 2n legend
)
# %%
## Normal vs Mix comparison

entries_grouped_normal_vs_mix = [
    ("qwen3_1_7B", "em_bad_medical_advice_mix1-1p0", "Mix 1:1"),
    ("qwen3_1_7B", "em_extreme_sports_mix1-1p0", "Mix 1:1"),
    ("qwen3_1_7B", "em_risky_financial_advice_mix1-1p0", "Mix 1:1"),
    ("qwen3_1_7B", "em_bad_medical_advice", "Normal"),
    ("qwen3_1_7B", "em_extreme_sports", "Normal"),
    ("qwen3_1_7B", "em_risky_financial_advice", "Normal"),
]

visualize_organism_comparison_grouped_by_model(
    entries_grouped_normal_vs_mix,
    config_path="configs/config.yaml",
    save_path="plots/grades_normal_vs_em.pdf",
    font_size=22,
    columnspacing=10.9,
    no_baselines=False,
    inter_model_gap_mult=1.55,
    x_label_font_size=20,
    x_label_rotation=0.0,
    n_cols=2,
    labelspacing=1.8,
    figsize=(10, 5.5),
    type_legend_labels={"Normal": "Normal", "Mix 1:1": "Mix 1:1"},
    legend_by_type=False,  # or False for full 2n legend
)
# %%
visualize_adl_base_chat_and_baseline_grouped_by_model(
    entries_grouped_base_vs_instruct,
    config_path="configs/config.yaml",
    save_path="plots/grades_base_vs_instruct.pdf",
    font_size=22,
    columnspacing=2.2,
    labelspacing=0.8,
)
# %%

visualize_run_distributions_violin(
    entries_grouped,
    config_path="configs/config.yaml",
    save_path="plots/run_distributions_violin.pdf",
    font_size=22,
    figsize=(10, 5.5),
)
# %%

# Mixture Agent Effect

mixture_entries = [
    ("llama32_1B_Instruct", "cake_bake", 7),
    ("llama32_1B_Instruct", "kansas_abortion", 7),
    ("llama32_1B_Instruct", "fda_approval", 7),
    ("llama32_1B_Instruct", "cake_bake", 15),
    ("llama32_1B_Instruct", "kansas_abortion", 15),
    ("llama32_1B_Instruct", "fda_approval", 15),
    ("llama32_1B_Instruct", "cake_bake", 31),
    ("llama32_1B_Instruct", "kansas_abortion", 31),
    ("llama32_1B_Instruct", "fda_approval", 31),
    ("llama32_1B_Instruct", "cake_bake", 63),
    ("llama32_1B_Instruct", "kansas_abortion", 63),
    ("llama32_1B_Instruct", "fda_approval", 63),
    ("llama32_1B_Instruct", "cake_bake", 127),
    ("llama32_1B_Instruct", "kansas_abortion", 127),
    ("llama32_1B_Instruct", "fda_approval", 127),
    ("qwen3_1_7B", "cake_bake", 7),
    ("qwen3_1_7B", "kansas_abortion", 7),
    ("qwen3_1_7B", "fda_approval", 7),
    ("qwen3_1_7B", "cake_bake", 15),
    ("qwen3_1_7B", "kansas_abortion", 15),
    ("qwen3_1_7B", "fda_approval", 15),
    ("qwen3_1_7B", "cake_bake", 31),
    ("qwen3_1_7B", "kansas_abortion", 31),
    ("qwen3_1_7B", "fda_approval", 31),
    ("qwen3_1_7B", "cake_bake", 63),
    ("qwen3_1_7B", "kansas_abortion", 63),
    ("qwen3_1_7B", "fda_approval", 63),
    ("qwen3_1_7B", "cake_bake", 127),
    ("qwen3_1_7B", "kansas_abortion", 127),
    ("qwen3_1_7B", "fda_approval", 127),
]
visualize_adl_over_positions(
    mixture_entries,
    config_path="configs/config.yaml",
    save_path="plots/adl_over_positions.pdf",
    x_label="Position",
    y_label="Grade (1..5)",
    log_x=True,
    log_base=2.0,
    line_colors={"agent_mi0": "#59afea", "agent_mi5": "#0569ad"},
    line_styles={"agent_mi0": "-", "agent_mi5": "-"},
    markers={"agent_mi0": "o", "agent_mi5": "s"},
    legend_loc="upper left",
    legend_ncol=1,
    font_size=22,
    use_log_nums=True,
    show_no_interactions=False,
    figsize=(6.2, 4.5),
)
# %%
