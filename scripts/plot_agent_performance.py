# %%
from __future__ import annotations
import sys
sys.path.append("scripts")
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import re
import json
import numpy as np
import matplotlib.pyplot as plt

from src.utils.interactive import load_hydra_config

import scienceplots as _scienceplots  # type: ignore[import-not-found]

plt.style.use("science")
# Absolute path to the Hydra config file
CONFIG_PATH = "configs/config.yaml"


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
}

def _model_display_name(model: str) -> str:
    name = MODEL_DISPLAY_NAMES.get(model, None)
    assert isinstance(name, str), f"Missing display name mapping for model: {model}"
    return name


def _results_root_from_cfg(cfg) -> Path:
    root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _find_latest_grade_by_kind_and_mi(
    agent_root: Path,
    organism: str,
    model: str,
    *,
    mi: int,
    is_baseline: bool,
) -> Path:
    """Return path to the latest hypothesis_grade.json for the given variant.

    - agent_root points to <results_root>/agent
    - For agent (is_baseline=False): matches <ts>_<org>_<model>_*_mi{mi}/ours/hypothesis_grade.json
    - For baseline (is_baseline=True): matches <ts>_<org>_<model>_*_baseline_mi{mi}/hypothesis_grade.json
    """
    assert (
        agent_root.exists() and agent_root.is_dir()
    ), f"Agent root not found: {agent_root}"
    assert isinstance(mi, int) and mi >= 0

    # Timestamp + matching pattern
    ts_prefix = (
        r"^(?P<ts>\d{8}_\d{6})_" + re.escape(organism) + "_" + re.escape(model) + r"_"
    )
    if is_baseline:
        pat = re.compile(ts_prefix + r".*_baseline_mi" + re.escape(str(mi)) + r"$")
    else:
        pat = re.compile(ts_prefix + r".*_mi" + re.escape(str(mi)) + r"$")

    candidates: List[Tuple[str, Path]] = []
    for child in agent_root.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if m is None:
            continue
        if is_baseline:
            grade_path = child / "hypothesis_grade.json"
        else:
            grade_path = child / "ours" / "hypothesis_grade.json"
        if grade_path.exists() and grade_path.is_file():
            candidates.append((m.group("ts"), grade_path))
    assert len(candidates) > 0, (
        f"No graded outputs found in {agent_root} for organism={organism} model={model} "
        f"variant={'baseline' if is_baseline else 'agent'} mi={mi}"
    )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_grade_score(json_path: Path) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict) and "score" in payload
    s = int(payload["score"])
    assert 1 <= s <= 6
    return float(s)


def _collect_scores_for_entry(
    model: str,
    organism: str,
    *,
    config_path: str,
) -> Dict[str, float]:
    """Collect scores for all variants for a single (model, organism).

    Returns mapping variant_key -> score.
    Assumes agent MI=5 and MI=0, and baselines at MI=0, 5, 50 exist. Fails fast otherwise.
    """
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

    paths: Dict[str, Path] = {}
    # Agent
    paths["agent_mi5"] = _find_latest_grade_by_kind_and_mi(
        agent_root, organism, model, mi=5, is_baseline=False
    )
    paths["agent_mi0"] = _find_latest_grade_by_kind_and_mi(
        agent_root, organism, model, mi=0, is_baseline=False
    )
    # Baselines (x0, x1, x10 relative to MI=5 -> 0, 5, 50)
    paths["baseline_mi0"] = _find_latest_grade_by_kind_and_mi(
        agent_root, organism, model, mi=0, is_baseline=True
    )
    paths["baseline_mi5"] = _find_latest_grade_by_kind_and_mi(
        agent_root, organism, model, mi=5, is_baseline=True
    )
    paths["baseline_mi50"] = _find_latest_grade_by_kind_and_mi(
        agent_root, organism, model, mi=50, is_baseline=True
    )

    scores: Dict[str, float] = {k: _load_grade_score(v) for k, v in paths.items()}
    assert set(scores.keys()) == {k for k, _ in VARIANTS}
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
        base_scores = _collect_scores_for_entry(base_model, organism, config_path=config_path)
        chat_scores = _collect_scores_for_entry(chat_model, organism, config_path=config_path)
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
        models_in_type = sorted({chat for _b, chat, _o, t in entries if t == organism_type})
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
        base_scores = _collect_scores_for_entry(base_model, organism, config_path=config_path)
        chat_scores = _collect_scores_for_entry(chat_model, organism, config_path=config_path)
        # Required keys must exist
        assert "agent_mi5" in base_scores and "agent_mi5" in chat_scores and "baseline_mi50" in chat_scores and "baseline_mi50" in base_scores
        per_metric_type_model_values.setdefault("adl_base", {}).setdefault(organism_type, {}).setdefault(chat_model, []).append(float(base_scores["agent_mi5"]))
        per_metric_type_model_values.setdefault("adl_chat", {}).setdefault(organism_type, {}).setdefault(chat_model, []).append(float(chat_scores["agent_mi5"]))
        per_metric_type_model_values.setdefault("baseline_mi50_base", {}).setdefault(organism_type, {}).setdefault(chat_model, []).append(float(base_scores["baseline_mi50"]))
        per_metric_type_model_values.setdefault("baseline_mi50_chat", {}).setdefault(organism_type, {}).setdefault(chat_model, []).append(float(chat_scores["baseline_mi50"]))

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
    blackbox_total_width = len(blackbox_keys) * bar_width + (len(blackbox_keys) - 1) * inner_spacing
    total_group_width = adl_total_width + minor_group_gap + blackbox_total_width
    assert total_group_width < 0.98, "Grouped bar width exceeds center spacing; reduce bar_width"
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
        models_in_type = sorted({chat for _b, chat, _o, t in entries if t == organism_type})
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
                assert len(vals) >= 1, f"No values for metric={m_key} type={organism_type} model={chat_model}"
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

def visualize_grades_by_type_average(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5.5),
    columnspacing: float = 0.6,
    labelspacing: float = 0.6,
    font_size: int = 20,
    x_label_pad: float = 70,
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
        scores = _collect_scores_for_entry(model, organism, config_path=config_path)
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
    blackbox_total_width = len(blackbox_keys) * bar_width + (len(blackbox_keys) - 1) * inner_spacing
    total_group_width = adl_total_width + minor_group_gap + blackbox_total_width
    assert total_group_width < 0.98, "Grouped bar width exceeds center spacing; reduce bar_width"
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
    ax.tick_params(axis='x', pad=x_label_pad)
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
    assert isinstance(threshold, (int, float))

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
            print(f"{variant_labels[k]}: count >= {threshold}: {counts[k]}, average score: {averages[k]:.3f}")



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
# %%
# Averaged over models per organism type
visualize_grades_by_type_average(
    entries_grouped,
    config_path="configs/config.yaml",
    save_path="plots/grades_by_type_avg.pdf",
    figsize=(10, 5.5),
    columnspacing=0.8,
    labelspacing=0.2,
    font_size=22,
    x_label_pad=15,
)
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
    remove_group_labels=True
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