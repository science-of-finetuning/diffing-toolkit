# %%
import sys

# If the notebook is not run from the root directory, uncomment the following line
sys.path.append("..")

from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Optional
import torch
from functools import lru_cache

from src.utils.interactive import load_hydra_config
import src.diffing.methods.activation_difference_lens.token_relevance as tr

try:
    import scienceplots as _scienceplots  # type: ignore[import-not-found]

    plt.style.use("science")
    del _scienceplots
except Exception:
    pass

GRADER_MODEL = "openai_gpt-5-mini"
# Human-friendly names for model tags used in results directories
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Q3 1.7B",
    "qwen3_32B": "Q3 32B",
    "qwen25_7B_Instruct": "Q2.5 7B",
    "gemma2_9B_it": "G2 9B",
    "gemma3_1B": "G3 1B",
    "llama31_8B_Instruct": "L3.1 8B",
    "llama32_1B_Instruct": "L3.2 1B",
    "llama32_1B": "L3.2 1B Base",
    "qwen3_1_7B_Base": "Q3 1.7B Base",
}
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
    "llama32_1B": "Llama3.2 1B",
    "llama31_8B": "Llama3.1 8B",
    "qwen3_1_7B_Base": "Qwen3 1.7B",
    "qwen25_VL_3B_Instruct": "Qwen2.5 VL 3B",
}


def _model_display_name(model: str) -> str:
    """Return human-friendly model name; fail if unknown to avoid silent mislabeling."""
    name = MODEL_DISPLAY_NAMES.get(model, None)
    assert isinstance(name, str), f"Missing display name mapping for model: {model}"
    return name


# Short display names for organisms; kept concise for plotting.
ORGANISM_DISPLAY_NAMES: Dict[str, str] = {
    # SDF
    "cake_bake": "cake",
    "kansas_abortion": "abortion",
    "roman_concrete": "concrete",
    "ignore_comment": "ignore",
    "fda_approval": "fda",
    # EM
    "em_bad_medical_advice": "medical",
    "em_risky_financial_advice": "finance",
    "em_extreme_sports": "sports",
    # Taboo
    "taboo_smile": "smile",
    "taboo_gold": "gold",
    "taboo_leaf": "leaf",
    # Subliminal
    "subliminal_learning_cat": "cat",
    "cake_bake_full": "cake",
    "kansas_abortion_full": "abortion",
    "roman_concrete_full": "concrete",
    "ignore_comment_full": "ignore",
    "fda_approval_full": "fda",
    "cake_bake_helena": "normal",
    "cake_bake_helena_possteer": "possteer",
    "cake_bake_helena_negsteer": "negsteer",
    "cake_bake_helena_ablation": "ablation",
    "cake_bake_mix1-1p0": "datamix 1:1",
}


def _organism_display_name(organism: str, organism_type: Optional[str] = None) -> str:
    """Return compact organism label; if unknown, heuristically strip known prefixes.

    organism_type (e.g., "Taboo", "EM") is used to safely drop redundant prefixes.
    """
    mapped = ORGANISM_DISPLAY_NAMES.get(organism, None)
    if isinstance(mapped, str):
        return mapped
    name = organism
    if organism_type == "Taboo" and name.startswith("taboo_"):
        name = name[len("taboo_") :]
    elif organism_type == "EM" and name.startswith("em_"):
        name = name[len("em_") :]
    elif organism_type == "Subliminal" and name.startswith("subliminal_"):
        name = name[len("subliminal_") :]
    assert isinstance(name, str) and len(name) > 0
    return name


def _select_dataset_dir(
    results_root: Path, layer_index: int, dataset_dir_name: str | None
) -> Path:
    layer_dir = results_root / f"layer_{layer_index}"
    assert (
        layer_dir.exists() and layer_dir.is_dir()
    ), f"Layer directory does not exist: {layer_dir}"
    if dataset_dir_name is None:
        candidates = sorted([p for p in layer_dir.iterdir() if p.is_dir()])
        assert len(candidates) >= 1
        return candidates[0]
    out = layer_dir / dataset_dir_name
    assert out.exists() and out.is_dir()
    return out


def _read_relevance_record(
    results_root: Path,
    layer_index: int,
    dataset_dir_name: str,
    position: int,
    variant: str,
    source: str = "logitlens",
) -> Dict[str, Any]:
    return _read_relevance_record_cached(
        str(results_root.resolve()),
        int(layer_index),
        dataset_dir_name,
        int(position),
        variant,
        source,
    )


@lru_cache(maxsize=4096)
def _read_relevance_record_cached(
    results_root_str: str,
    layer_index: int,
    dataset_dir_name: str,
    position: int,
    variant: str,
    source: str = "logitlens",
) -> Dict[str, Any]:
    """Cached reader for relevance record JSON."""
    results_root = Path(results_root_str)
    tr_dir = (
        results_root
        / f"layer_{layer_index}"
        / dataset_dir_name
        / "token_relevance"
        / f"position_{position}"
        / variant
    )
    assert (
        tr_dir.exists() and tr_dir.is_dir()
    ), f"Token relevance directory does not exist: {tr_dir}"
    if source == "logitlens":
        rel_path = tr_dir / f"relevance_logitlens_{GRADER_MODEL}.json"
    elif source == "patchscope":
        rel_path = tr_dir / f"relevance_patchscope_{GRADER_MODEL}.json"
    else:
        assert False, f"Unknown source: {source}"
    assert (
        rel_path.exists() and rel_path.is_file()
    ), f"Missing relevance json: {rel_path}"
    with open(rel_path, "r", encoding="utf-8") as f:
        rec: Dict[str, Any] = json.load(f)
    return rec


def _recompute_percentage_from_labels(rec: Dict[str, Any], filtered: bool) -> float:
    labels: List[str] = list(rec["labels"])
    if (rec.get("source") == "patchscope") and filtered:
        mask = rec.get("unsupervised_filter", None)
        assert isinstance(mask, list) and len(mask) == len(labels)
        labels = [lbl for m, lbl in zip(mask, labels) if m]
        assert len(labels) > 0
    num_rel = sum(lbl == "RELEVANT" for lbl in labels)
    pct = num_rel / float(len(labels))
    assert 0.0 <= pct <= 1.0
    return float(pct)


def _compute_weighted_percentage(rec: Dict[str, Any], probs: np.ndarray) -> float:
    labels: List[str] = list(rec["labels"])
    total = 0.0
    relevant = 0.0
    for i, lbl in enumerate(labels):
        total += probs[i]
        if lbl == "RELEVANT":
            relevant += probs[i]
    assert total > 0.0
    pct = relevant / total
    assert 0.0 <= pct <= 1.0
    return float(pct)


def _load_topk_logitlens_probs_and_tokens(
    results_root: Path,
    layer_index: int,
    dataset_dir_name: str,
    position: int,
    variant: str,
    tokenizer_id: str,
) -> Tuple[np.ndarray, List[str]]:
    probs_arr, tokens_tuple = _cached_topk_logitlens_probs_and_tokens(
        str(results_root.resolve()),
        int(layer_index),
        dataset_dir_name,
        int(position),
        variant,
        tokenizer_id,
    )
    return probs_arr, list(tokens_tuple)


@lru_cache(maxsize=1024)
def _get_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    return tok


@lru_cache(maxsize=2048)
def _cached_topk_logitlens_probs_and_tokens(
    results_root_str: str,
    layer_index: int,
    dataset_dir_name: str,
    position: int,
    variant: str,
    tokenizer_id: str,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Cached loader for promoted tokens and probabilities from logit lens outputs."""
    tok = _get_tokenizer(tokenizer_id)
    tokens, probs = tr._load_topk_promoted_tokens(
        results_dir=Path(results_root_str),
        dataset_id=dataset_dir_name,
        layer_index=int(layer_index),
        position_index=int(position),
        tokenizer=tok,  # type: ignore[arg-type]
        k=10**9,
        variant=variant,
    )
    probs_arr = np.asarray(probs, dtype=np.float32)
    assert probs_arr.ndim == 1 and len(probs_arr) == len(tokens)
    return probs_arr, tuple(tokens)


def _load_probs_and_tokens(
    results_root: Path,
    layer_index: int,
    dataset_dir_name: str,
    position: int,
    variant: str,
    source: str,
    cfg: Any,
) -> Tuple[np.ndarray, List[str]]:
    if source == "logitlens":
        return _load_topk_logitlens_probs_and_tokens(
            results_root,
            layer_index,
            dataset_dir_name,
            position,
            variant,
            tokenizer_id=cfg.model.model_id,
        )
    elif source == "patchscope":
        tokens_all_t, _selected_t, probs_arr = _cached_patchscope_tokens(
            str(results_root.resolve()),
            dataset_dir_name,
            int(layer_index),
            int(position),
            variant,
        )
        assert probs_arr.ndim == 1 and len(tokens_all_t) == probs_arr.shape[0]
        return probs_arr, list(tokens_all_t)
    else:
        assert False, f"Unknown source: {source}"


@lru_cache(maxsize=2048)
def _cached_patchscope_tokens(
    results_root_str: str,
    dataset_id: str,
    layer_index: int,
    position_index: int,
    variant: str,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], np.ndarray]:
    """Cached loader for PatchScope tokens and weights."""
    tokens_all, selected, probs = tr._load_patchscope_tokens(
        results_dir=Path(results_root_str),
        dataset_id=dataset_id,
        layer_index=int(layer_index),
        position_index=int(position_index),
        variant=variant,
    )
    probs_arr = np.asarray(probs, dtype=np.float32)
    assert probs_arr.ndim == 1 and len(tokens_all) == len(probs_arr)
    return tuple(tokens_all), tuple(selected), probs_arr


def _load_positions_and_percentages(
    results_root: Path,
    layer_index: int,
    dataset_dir_name: str,
    variant: str,
    source: str,
    cfg: Any,
    filtered: bool,
    weighted: bool,
) -> List[Tuple[int, float]]:
    tokenizer_id = str(getattr(getattr(cfg, "model"), "model_id"))
    pairs = _cached_positions_and_percentages(
        str(results_root.resolve()),
        int(layer_index),
        dataset_dir_name,
        variant,
        source,
        bool(filtered),
        bool(weighted),
        tokenizer_id,
    )
    return [(int(p), float(q)) for (p, q) in pairs]


@lru_cache(maxsize=1024)
def _cached_positions_and_percentages(
    results_root_str: str,
    layer_index: int,
    dataset_dir_name: str,
    variant: str,
    source: str,
    filtered: bool,
    weighted: bool,
    tokenizer_id: str,
) -> Tuple[Tuple[int, float], ...]:
    """Cached aggregator of (position, percentage) pairs.

    tokenizer_id is only used when source == 'logitlens' and weighted=True.
    """
    results_root = Path(results_root_str)
    out: List[Tuple[int, float]] = []
    layer_dir = (
        results_root / f"layer_{layer_index}" / dataset_dir_name / "token_relevance"
    )
    assert (
        layer_dir.exists() and layer_dir.is_dir()
    ), f"Layer directory does not exist: {layer_dir}"
    for sub in sorted(layer_dir.iterdir(), key=lambda p: p.name):
        if not sub.is_dir() or not sub.name.startswith("position_"):
            continue
        pos = int(sub.name.split("_")[-1])
        if pos > 4:
            continue
        rec = _read_relevance_record(
            results_root, layer_index, dataset_dir_name, pos, variant, source
        )
        if (not weighted) and (not filtered):
            pct = _recompute_percentage_from_labels(rec, filtered=False)
        elif (not weighted) and filtered:
            assert source == "patchscope"
            pct = _recompute_percentage_from_labels(rec, filtered=True)
        elif weighted and (not filtered):
            pct_stored = rec.get("weighted_percentage", None)
            if isinstance(pct_stored, float):
                pct = float(pct_stored)
            else:
                if source == "logitlens":
                    probs, _tokens = _cached_topk_logitlens_probs_and_tokens(
                        results_root_str,
                        layer_index,
                        dataset_dir_name,
                        pos,
                        variant,
                        tokenizer_id,
                    )
                elif source == "patchscope":
                    _tokens_all, _sel, probs = _cached_patchscope_tokens(
                        results_root_str,
                        dataset_dir_name,
                        layer_index,
                        pos,
                        variant,
                    )
                else:
                    assert False, f"Unknown source: {source}"
                pct = _compute_weighted_percentage(rec, probs)
        else:
            assert source == "patchscope"
            if "weighted_filtered_percentage" in rec:
                pct = float(rec["weighted_filtered_percentage"])
            else:
                labels: List[str] = list(rec["labels"])
                mask = rec.get("unsupervised_filter", None)
                assert isinstance(mask, list) and len(mask) == len(labels)
                if not any(mask):
                    pct = 0.0
                else:
                    _tokens_all, _sel, probs = _cached_patchscope_tokens(
                        results_root_str,
                        dataset_dir_name,
                        layer_index,
                        pos,
                        variant,
                    )
                    filtered_probs = np.asarray(
                        [w for m, w in zip(mask, probs.tolist()) if m], dtype=np.float32
                    )
                    print(filtered_probs)
                    assert filtered_probs.ndim == 1 and filtered_probs.size > 0
                    filtered_labels = [lbl for m, lbl in zip(mask, labels) if m]
                    pct = _compute_weighted_percentage(
                        {"labels": filtered_labels}, filtered_probs
                    )
        out.append((pos, float(pct)))
    assert len(out) > 0
    out.sort(key=lambda t: t[0])
    return tuple(out)


def plot_relevance_curves(
    entries: List[Tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str],
    variant: str,
    source: str,
    filtered: bool,
    weighted: bool,
    figsize: Tuple[float, float] = (8, 5.5),
    config_path: str,
    save_path: Optional[Path] = None,
    legend_position: str = "top",
) -> None:
    model_to_layers: Dict[str, set] = {}
    for model, layer, organism in entries:
        model_to_layers.setdefault(model, set()).add(int(layer))
    for model, layers in model_to_layers.items():
        assert len(layers) == 1

    unique_models = list(model_to_layers.keys())
    unique_organisms = sorted(list({org for _, _, org in entries}))

    linestyles = ["-", ":", "-.", "--"]
    model_to_style = {
        m: linestyles[i % len(linestyles)] for i, m in enumerate(unique_models)
    }

    cmap = plt.get_cmap("tab10")
    organism_to_color = {org: cmap(i % 10) for i, org in enumerate(unique_organisms)}

    plt.figure(figsize=figsize)
    global_positions: set[int] = set()

    for model, layer, organism in entries:
        overrides = [
            f"organism={organism}",
            f"model={model}",
            "infrastructure=mats_cluster_paper",
        ]
        cfg = load_hydra_config(config_path, *overrides)
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir()
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
        positions = [p for p, _ in pairs]
        percentages = [q for _, q in pairs]
        global_positions.update(positions)
        style = model_to_style[model]
        color = organism_to_color[organism]
        label = f"{organism}"
        xs = np.array(positions, dtype=int)
        ys = np.array(percentages, dtype=np.float32)
        assert np.isfinite(ys).all()
        plt.plot(
            xs, ys, linestyle=style, marker="o", color=color, linewidth=2, label=label
        )

    if len(global_positions) > 0:
        xticks = sorted(global_positions)
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(True, linestyle=":", alpha=0.3)
    plt.xlabel("Position index (int)")
    plt.ylabel("Relevant Token Percentage")
    # plt.title(
    #     f"Organism Relevance of top tokens ({variant}, {source}{' filtered' if (source=='patchscope' and filtered) else ''}{', weighted' if weighted else ''})"
    # )
    legend = plt.legend(loc=legend_position, fontsize="small", frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_relevance_curves_grouped(
    entries: List[Tuple[str, int, str]],
    *,
    dataset_dir_name: Optional[str],
    source: str,
    filtered: bool,
    weighted: bool,
    figsize: Tuple[float, float] = (9, 4.8),
    config_path: str,
    save_path: Optional[Path] = None,
) -> None:
    unique_models = sorted(list({m for m, _, _ in entries}))
    linestyles = ["-", ":", "-.", "--"]
    model_to_style = {
        m: linestyles[i % len(linestyles)] for i, m in enumerate(unique_models)
    }
    cmap = plt.get_cmap("tab10")
    unique_organisms = sorted(list({org for _, _, org in entries}))
    organism_to_color = {org: cmap(i % 10) for i, org in enumerate(unique_organisms)}
    variants = ["difference", "base", "ft"]
    variant_titles = ["Difference (FT - Base)", "Base Model", "Finetuned Model"]
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 2.5, figsize[1]), sharey=True)
    for ax_idx, variant in enumerate(variants):
        ax = axes[ax_idx]
        global_positions: set[int] = set()
        for model, layer, organism in entries:
            overrides = [
                f"organism={organism}",
                f"model={model}",
                "infrastructure=mats_cluster_paper",
            ]
            cfg = load_hydra_config(config_path, *overrides)
            results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
            assert results_root.exists() and results_root.is_dir()
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
            positions = [p for p, _ in pairs]
            percentages = [q for _, q in pairs]
            global_positions.update(positions)
            style = model_to_style[model]
            color = organism_to_color[organism]
            label = f"{organism} ({model})"
            xs = np.array(positions, dtype=int)
            ys = np.array(percentages, dtype=np.float32)
            assert np.isfinite(ys).all()
            ax.plot(
                xs,
                ys,
                linestyle=style,
                marker="o",
                color=color,
                linewidth=2,
                label=label,
            )
        if len(global_positions) > 0:
            xticks = sorted(global_positions)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_xlabel("Position index (int)")
        ax.set_title(variant_titles[ax_idx])
        if ax_idx == 0:
            ax.legend()
    axes[0].set_ylabel("Relevant Token Percentage")
    plt.suptitle(
        f"Organism Relevance of top tokens ({source}{' filtered' if (source=='patchscope' and filtered) else ''}{', weighted' if weighted else ''})"
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def summarize_max_per_model(
    entries: List[Tuple[str, int, str, str]],
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
    """Horizontal grouped bars of mean±std of max relevance per model, grouped by organism type.

    entries: list of (model, layer, organism, organism_type)
    Groups y-axis by organism_type; within each type, shows models side-by-side with 3 bars (difference/base/ft).
    """
    plt.rcParams.update({"font.size": font_size})

    variants = ["ft", "base", "difference"]
    variant_labels = ["Finetuned", "Base", "Difference"]
    variant_colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    unique_types = sorted({t for _, _, _, t in entries})
    assert len(unique_types) >= 1

    # Collect maxima per (variant, type, model)
    per_variant_type_model_maxima: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        v: {} for v in variants
    }
    for variant in variants:
        for model, layer, organism, organism_type in entries:
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
            per_variant_type_model_maxima.setdefault(variant, {}).setdefault(
                organism_type, {}
            ).setdefault(model, []).append(entry_max)

    # Prepare plotting positions
    fig, ax = plt.subplots(figsize=figsize)
    bar_height = 0.35
    offsets = [-bar_height, 0.0, bar_height]

    y_positions: List[float] = []
    y_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []
    group_boundaries: List[float] = []

    current_y = 0.0
    group_gap = 1.5
    model_gap = group_gap / 4.0

    for organism_type in unique_types:
        models_in_type = sorted({m for m, _, _, t in entries if t == organism_type})
        assert len(models_in_type) >= 1

        # Compute means/stds per model for each variant
        means_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        stds_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        for model in models_in_type:
            for v in variants:
                vals = (
                    per_variant_type_model_maxima.get(v, {})
                    .get(organism_type, {})
                    .get(model, [])
                )
                if vals:
                    means_by_variant[v].append(float(np.mean(vals)))
                    stds_by_variant[v].append(float(np.std(vals)))
                else:
                    means_by_variant[v].append(0.0)
                    stds_by_variant[v].append(0.0)

        # Plot bars for this type with a small gap between models
        base_positions = [
            current_y + i * (1.0 + model_gap) for i in range(len(models_in_type))
        ]
        for i, v in enumerate(variants):
            ys = [bp + offsets[i] for bp in base_positions]
            means_arr = np.asarray(means_by_variant[v], dtype=np.float32)
            stds_arr = np.asarray(stds_by_variant[v], dtype=np.float32)
            # Clamp error bars within [0, 1]
            lower_err = np.minimum(stds_arr, means_arr)
            upper_err = np.minimum(stds_arr, 1.0 - means_arr)
            xerr = np.vstack([lower_err, upper_err])
            ax.barh(
                ys,
                means_arr,
                height=bar_height,
                xerr=xerr,
                label=variant_labels[i] if organism_type == unique_types[0] else None,
                color=variant_colors[i],
                alpha=0.9,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )

        # Accumulate y tick labels at central bar position per model
        y_positions.extend(base_positions)
        display_labels: List[str] = []
        for m in models_in_type:
            lbl = _model_display_name(m)
            if organism_type == "SDF" and m == "gemma3_1B":
                lbl = f"{lbl}*"
            display_labels.append(lbl)
        y_labels.extend(display_labels)

        # Track type center for hierarchical labeling
        type_center = current_y + ((len(models_in_type) - 1) * (1.0 + model_gap)) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        # Add boundary line (except after the last group)
        if organism_type != unique_types[-1]:
            boundary_y = (
                current_y
                + len(models_in_type)
                + model_gap * (len(models_in_type) - 1)
                + group_gap / 2.0
            )
            group_boundaries.append(boundary_y)

        current_y += (
            len(models_in_type) + model_gap * (len(models_in_type) - 1) + group_gap
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis="y", which="both", length=0, width=0, left=False)
    ax.set_xlabel("Max Relevant Token Percentage")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="x")

    # Add horizontal dashed lines between groups
    for boundary in group_boundaries:
        ax.axhline(
            y=boundary - 0.5, color="gray", linestyle="dotted", alpha=0.5, linewidth=1
        )

    # Secondary y-axis for type labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(type_centers)
    ax2.set_yticklabels(type_labels)
    ax2.tick_params(axis="y", length=0, width=0, which="both", left=False)
    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)

    # Legend: match bar order (top->bottom: Difference, Base, Finetuned) and add frame/background
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Difference", "Base", "Finetuned"]
    label_to_handle = {lbl: h for h, lbl in zip(handles, labels)}
    ordered_handles = [
        label_to_handle[lbl] for lbl in desired_order if lbl in label_to_handle
    ]
    leg = ax.legend(ordered_handles, desired_order, frameon=True)
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


# %%


def summarize_max_per_model_vert(
    entries: List[Tuple[str, int, str, str]],
    *,
    dataset_dir_name: Optional[str],
    source: str,
    filtered: bool,
    weighted: bool,
    figsize: Tuple[float, float] = (9, 4.8),
    config_path: str,
    save_path: Optional[Path] = None,
    font_size: int = 22,
    x_axis_gap: float = -0.03,
    x_axis_label_rotation: int = 90,
    x_group_gap: float = 70,
    show_dots: bool = False,
    group_gap=1.5,
) -> None:
    """Vertical grouped bars of mean±std of max relevance per model, grouped by organism type.

    entries: list of (model, layer, organism, organism_type)
    Groups x-axis by organism_type (bottom labels); within each type, shows models side-by-side
    (top labels, rotated 90°) with 3 bars (difference/base/ft).
    """
    plt.rcParams.update({"font.size": font_size})

    variants = ["difference", "ft", "base"]
    variant_labels = ["Difference", "Finetuned", "Base"]
    variant_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    hatches = ["/", ".", "\\"]
    unique_types = sorted({t for _, _, _, t in entries})
    assert len(unique_types) >= 1

    # Collect maxima per (variant, type, model)
    per_variant_type_model_maxima: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        v: {} for v in variants
    }
    for variant in variants:
        for model, layer, organism, organism_type in entries:
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
            per_variant_type_model_maxima.setdefault(variant, {}).setdefault(
                organism_type, {}
            ).setdefault(model, []).append(entry_max)

    # Prepare plotting positions (vertical bars)
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.35
    offsets = [-bar_width, 0.0, bar_width]

    model_centers: List[float] = []
    model_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []
    group_boundaries: List[float] = []

    current_x = 0.0
    model_gap = group_gap / 4.0

    for organism_type in unique_types:
        models_in_type = sorted({m for m, _, _, t in entries if t == organism_type})
        assert len(models_in_type) >= 1

        # Compute means/stds per model for each variant
        means_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        stds_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        for model in models_in_type:
            for v in variants:
                vals = (
                    per_variant_type_model_maxima.get(v, {})
                    .get(organism_type, {})
                    .get(model, [])
                )
                if vals:
                    means_by_variant[v].append(float(np.mean(vals)))
                    stds_by_variant[v].append(float(np.std(vals)))
                else:
                    means_by_variant[v].append(0.0)
                    stds_by_variant[v].append(0.0)

        # Plot bars for this type with a small gap between models
        base_positions = [
            current_x + i * (1.0 + model_gap) for i in range(len(models_in_type))
        ]
        for i, v in enumerate(variants):
            xs = [bp + offsets[i] for bp in base_positions]
            means_arr = np.asarray(means_by_variant[v], dtype=np.float32)
            stds_arr = np.asarray(stds_by_variant[v], dtype=np.float32)
            lower_err = np.minimum(stds_arr, means_arr)
            upper_err = np.minimum(stds_arr, 1.0 - means_arr)
            yerr = np.vstack([lower_err, upper_err])
            ax.bar(
                xs,
                means_arr,
                width=bar_width,
                yerr=yerr,
                label=variant_labels[i] if organism_type == unique_types[0] else None,
                color=variant_colors[i],
                hatch=hatches[i],
                alpha=0.9,
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )

        # Accumulate model tick labels at central bar position per model
        for m, base_x in zip(models_in_type, base_positions):
            lbl = _model_display_name(m)
            if organism_type == "SDF" and m == "gemma3_1B":
                lbl = f"{lbl}*"
            model_centers.append(base_x)
            model_labels.append(lbl)

        # Track type center for hierarchical labeling
        type_center = current_x + ((len(models_in_type) - 1) * (1.0 + model_gap)) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        # Add boundary line (except after the last group)
        if organism_type != unique_types[-1]:
            boundary_x = (
                current_x
                + len(models_in_type)
                + model_gap * (len(models_in_type) - 1)
                + group_gap / 2.0
            )
            group_boundaries.append(boundary_x)

        current_x += (
            len(models_in_type) + model_gap * (len(models_in_type) - 1) + group_gap
        )

    # Primary x-axis: group labels at the bottom with extra padding
    ax.set_xticks(type_centers)
    ax.set_xticklabels(type_labels)
    ax.tick_params(
        axis="x", which="both", length=0, width=0, bottom=True, pad=x_group_gap
    )

    # Y-axis styling
    ax.set_ylabel("Fraction Relevant Tokens")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Add vertical dotted lines between groups
    for boundary in group_boundaries:
        ax.axvline(
            x=boundary - 0.5, color="gray", linestyle="dotted", alpha=0.5, linewidth=1
        )

    # Model labels between axis and group labels at the bottom (rotated)
    model_font_size = max(8, int(font_size * 0.7))
    for x, lbl in zip(model_centers, model_labels):
        ax.text(
            x,
            x_axis_gap,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=x_axis_label_rotation,
            fontsize=model_font_size,
            clip_on=False,
        )

    # Legend: order as Difference, Base, Finetuned
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Difference", "Finetuned", "Base"]
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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


# %%


def plot_max_by_layer(
    entries: List[Tuple[str, str, str, Tuple[int, ...], int]],
    *,
    dataset_dir_name: Optional[str],
    source: str,
    filtered: bool,
    weighted: bool,
    figsize: Tuple[float, float] = (9, 4.8),
    config_path: str,
    save_path: Optional[Path] = None,
    font_size: int = 22,
    variant: str = "difference",
) -> None:
    """Line plot of mean±std max relevance across layers, grouped by (model, organism_type).

    entries: list of (model, organism, organism_type, (layer1, layer2, ...), total_layers)
    Each group (model, organism_type) is plotted as one line with shaded std area.
    X-axis: relative layer position (0-1), Y-axis: Fraction Relevant Tokens (0-1).
    """
    plt.rcParams.update({"font.size": font_size})

    variants = ["difference", "ft", "base"]
    assert variant in variants, f"variant must be one of {variants}"

    # Collect data: (model, organism_type) -> relative_layer -> list of max percentages (over organisms)
    group_data: Dict[Tuple[str, str], Dict[float, List[float]]] = {}

    for model, organism, organism_type, layers_tuple, total_layers in entries:
        assert total_layers > 0
        group_key = (model, organism_type)
        if group_key not in group_data:
            group_data[group_key] = {}

        for layer in layers_tuple:
            relative_layer = float(layer) / float(total_layers - 1)
            assert 0.0 <= relative_layer <= 1.0

            if relative_layer not in group_data[group_key]:
                group_data[group_key][relative_layer] = []

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
            group_data[group_key][relative_layer].append(entry_max)

    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique groups and sort for consistent ordering
    groups = sorted(group_data.keys())
    cmap = plt.get_cmap("tab10")

    for group_idx, (model, organism_type) in enumerate(groups):
        relative_layer_to_values = group_data[(model, organism_type)]
        relative_layers_sorted = sorted(relative_layer_to_values.keys())
        means = []
        stds = []

        for relative_layer in relative_layers_sorted:
            vals = relative_layer_to_values[relative_layer]
            assert len(vals) > 0
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        means_arr = np.asarray(means, dtype=np.float32)
        stds_arr = np.asarray(stds, dtype=np.float32)
        relative_layers_arr = np.asarray(relative_layers_sorted, dtype=np.float32)

        # Clamp std to valid range
        lower_bound = np.maximum(means_arr - stds_arr, 0.0)
        upper_bound = np.minimum(means_arr + stds_arr, 1.0)

        # Plot shaded area for std
        color = cmap(group_idx % 10)
        ax.fill_between(
            relative_layers_arr,
            lower_bound,
            upper_bound,
            alpha=0.2,
            color=color,
        )

        # Plot line for mean
        label = f"{_model_display_name(model)} ({organism_type})"
        ax.plot(
            relative_layers_arr,
            means_arr,
            marker="o",
            linestyle="-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label,
        )

    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Fraction Relevant Tokens")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=True, fontsize=int(font_size * 0.8))
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


# %%


def plot_points_per_group(
    entries: List[Tuple[str, int, str, str]],
    *,
    dataset_dir_name: Optional[str],
    source: str,
    filtered: bool,
    weighted: bool,
    figsize: Tuple[float, float] = (9, 4.8),
    force_fig_size: bool = False,
    config_path: str,
    save_dir: Optional[Path] = None,
    font_size: int = 22,
) -> None:
    """Create one figure per group with vertical bars for all (model, organism) datapoints.

    Matches the visual style of `summarize_max_per_model_vert` but instead of means we draw one
    vertical bar per (model, organism, variant) using the max-over-positions value.
    Bottom axis shows the group label; model labels are placed between axis and group label
    (rotated vertically). A secondary top x-axis shows organism labels at their exact positions.
    """
    plt.rcParams.update({"font.size": font_size})

    variants = ["ft", "base", "difference"]
    variant_labels = ["Finetuned", "Base", "Difference"]
    variant_colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    hatches = ["/", ".", "//"]
    unique_types = sorted({t for _, _, _, t in entries})
    assert len(unique_types) >= 1

    # Precompute per-entry maxima for efficiency and clarity
    cache: Dict[Tuple[str, str, str, str], List[float]] = {}

    for organism_type in unique_types:
        # Gather models and organisms for this type
        models_in_type = sorted({m for m, _, _, t in entries if t == organism_type})
        assert len(models_in_type) >= 1

        # Organisms per model in stable order
        organisms_by_model: Dict[str, List[str]] = {}
        for m in models_in_type:
            orgs = sorted(
                {org for mm, _, org, tt in entries if tt == organism_type and mm == m}
            )
            assert len(orgs) >= 1
            organisms_by_model[m] = orgs

        # Dynamic figure width; widen more when there are few organisms (< 10)
        num_models = len(models_in_type)
        num_cols = sum(len(v) for v in organisms_by_model.values())
        width_per_col = 0.6 if num_cols < 10 else 0.5
        width_per_model_gap = 0.8 if num_cols < 10 else 0.6
        base_width = 1.2 if num_cols < 10 else 0.8
        fig_width = max(
            3.8,
            base_width
            + width_per_col * num_cols
            + width_per_model_gap * max(num_models - 1, 0),
        )
        if force_fig_size:
            fig_width = figsize[0]
        fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))

        organism_tick_positions: List[float] = []
        organism_tick_labels: List[str] = []
        model_centers: List[float] = []
        model_disp_labels: List[str] = []
        model_boundaries: List[float] = []

        current_x = 0.0
        model_gap = 1.5
        organism_gap = model_gap / 4.0
        bar_width = 0.25
        offsets = [-bar_width, 0.0, bar_width]

        # Track whether legend entries have been added
        label_added = [False, False, False]

        for model_idx, model in enumerate(models_in_type):
            orgs = organisms_by_model[model]
            base_positions = [
                current_x + i * (1.0 + organism_gap) for i in range(len(orgs))
            ]
            # Bars for each organism
            for organism, base_x in zip(orgs, base_positions):
                for i, variant in enumerate(variants):
                    key = (variant, model, organism, organism_type)
                    if key not in cache:
                        layer_candidates = {
                            layer_idx
                            for m, layer_idx, o, t in entries
                            if (m == model and o == organism and t == organism_type)
                        }
                        assert len(layer_candidates) == 1
                        layer = int(next(iter(layer_candidates)))
                        overrides = [
                            f"organism={organism}",
                            f"model={model}",
                            "infrastructure=mats_cluster_paper",
                        ]
                        cfg = load_hydra_config(config_path, *overrides)
                        results_root = (
                            Path(cfg.diffing.results_dir) / "activation_difference_lens"
                        )
                        assert results_root.exists() and results_root.is_dir()
                        selected_ds_dir = _select_dataset_dir(
                            results_root, layer, dataset_dir_name
                        )
                        ds_name = selected_ds_dir.name
                        pairs = _load_positions_and_percentages(
                            results_root,
                            layer,
                            ds_name,
                            variant=variant,
                            source=source,
                            cfg=cfg,
                            filtered=(source == "patchscope" and filtered),
                            weighted=weighted,
                        )
                        vals = [q for _, q in pairs]
                        assert len(vals) > 0
                        cache[key] = [float(max(vals))]
                    y = cache[key][0]
                    x = base_x + offsets[i]
                    ax.bar(
                        [x],
                        [y],
                        width=bar_width,
                        color=variant_colors[i],
                        hatch=hatches[i],
                        alpha=0.9,
                        label=variant_labels[i] if not label_added[i] else None,
                    )
                    label_added[i] = True

                organism_tick_positions.append(base_x)
                organism_tick_labels.append(organism)

            # model center and label
            model_center = current_x + ((len(orgs) - 1) * (1.0 + organism_gap)) / 2.0
            model_centers.append(model_center)
            disp = _model_display_name(model)
            model_disp_labels.append(disp)

            # boundary between models
            current_x += len(orgs) + organism_gap * (len(orgs) - 1) + model_gap
            if model_idx != len(models_in_type) - 1:
                model_boundaries.append(current_x - model_gap / 2.0)

        # Bottom: model labels at model centers with extra padding
        ax.set_xticks(model_centers)
        ax.set_xticklabels(model_disp_labels)
        ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=70)

        # Y-axis styling
        ax.set_ylabel("Fraction Relevant Tokens")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.3, axis="y")

        # Vertical dotted lines between model blocks
        for boundary in model_boundaries:
            ax.axvline(
                x=boundary, color="gray", linestyle="dotted", alpha=0.5, linewidth=1
            )

        # Organism labels between axis and model tick labels at the bottom (rotated)
        model_font_size = max(8, int(font_size * 0.7))
        organism_disp_labels = [
            _organism_display_name(org, organism_type) for org in organism_tick_labels
        ]
        for x, lbl in zip(organism_tick_positions, organism_disp_labels):
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

        # Legend matches summarize_max_per_model_vert; one column if few organisms
        handles, labels = ax.get_legend_handles_labels()
        desired_order = ["Difference", "Base", "Fine-tuned"]
        label_to_handle = {lbl: h for h, lbl in zip(handles, labels)}
        ordered_handles = [
            label_to_handle[lbl] for lbl in desired_order if lbl in label_to_handle
        ]
        ncol = 1 if num_cols < 10 else 3
        leg = ax.legend(
            ordered_handles,
            desired_order,
            frameon=True,
            ncol=ncol,
            fontsize=int(font_size * 0.8),
        )
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")

        plt.tight_layout()
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                str(save_dir / f"token_relevance_points_per_group_{organism_type}.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()


# %%
def summarize_max_over_position_and_method(
    entries: List[Tuple[str, int, str, str]],
    *,
    dataset_dir_name: Optional[str],
    weighted: bool,
    figsize: Tuple[float, float] = (9, 4.8),
    config_path: str,
    save_path: Optional[Path] = None,
) -> None:
    """Horizontal grouped bars of mean±std of max relevance per model, grouped by type,
    maximizing over positions and methods (logit lens + patch scope, unfiltered).

    entries: list of (model, layer, organism, organism_type)
    """
    variants = ["difference", "base", "ft"]
    variant_labels = ["Difference", "Base", "Fine-tuned"]
    variant_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    hatches = ["/", ".", "//"]
    unique_types = sorted({t for _, _, _, t in entries})
    assert len(unique_types) >= 1

    # Collect maxima per (variant, type, model) where maxima is over positions and methods
    per_variant_type_model_maxima: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        v: {} for v in variants
    }
    for variant in variants:
        for model, layer, organism, organism_type in entries:
            overrides = [
                f"organism={organism}",
                f"model={model}",
                "infrastructure=mats_cluster_paper",
            ]
            cfg = load_hydra_config(config_path, *overrides)
            results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
            assert results_root.exists() and results_root.is_dir()
            selected_ds_dir = _select_dataset_dir(
                results_root, int(layer), dataset_dir_name
            )
            ds_name = selected_ds_dir.name

            pairs_logitlens = _load_positions_and_percentages(
                results_root,
                int(layer),
                ds_name,
                variant=variant,
                source="logitlens",
                cfg=cfg,
                filtered=False,
                weighted=weighted,
            )

            pairs_patchscope = _load_positions_and_percentages(
                results_root,
                int(layer),
                ds_name,
                variant=variant,
                source="patchscope",
                cfg=cfg,
                filtered=False,
                weighted=weighted,
            )

            percentages_combined: List[float] = [q for _, q in pairs_logitlens] + [
                q for _, q in pairs_patchscope
            ]
            assert len(percentages_combined) > 0
            entry_max = float(max(percentages_combined))

            per_variant_type_model_maxima.setdefault(variant, {}).setdefault(
                organism_type, {}
            ).setdefault(model, []).append(entry_max)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    bar_height = 0.22
    offsets = [-bar_height, 0.0, bar_height]

    y_positions: List[float] = []
    y_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []

    current_y = 0.0
    group_gap = 0.6

    for organism_type in unique_types:
        models_in_type = sorted({m for m, _, _, t in entries if t == organism_type})
        assert len(models_in_type) >= 1

        means_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        stds_by_variant: Dict[str, List[float]] = {v: [] for v in variants}

        for model in models_in_type:
            for v in variants:
                vals = (
                    per_variant_type_model_maxima.get(v, {})
                    .get(organism_type, {})
                    .get(model, [])
                )
                if vals:
                    means_by_variant[v].append(float(np.mean(vals)))
                    stds_by_variant[v].append(float(np.std(vals)))
                else:
                    means_by_variant[v].append(0.0)
                    stds_by_variant[v].append(0.0)

        base_positions = [current_y + i for i in range(len(models_in_type))]
        for i, v in enumerate(variants):
            ys = [bp + offsets[i] for bp in base_positions]
            ax.barh(
                ys,
                means_by_variant[v],
                height=bar_height,
                xerr=stds_by_variant[v],
                label=variant_labels[i] if organism_type == unique_types[0] else None,
                color=variant_colors[i],
                hatch=hatches[i],
                alpha=0.9,
                capsize=4,
            )

        y_positions.extend(base_positions)
        y_labels.extend(models_in_type)

        type_center = current_y + (len(models_in_type) - 1) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        current_y += len(models_in_type) + group_gap

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Max Relevant Token Percentage")
    ax.set_title(
        f"Maximum Relevance per Model by Type (mean ± std) — max over position and method"
        f"{' , weighted' if weighted else ''}"
    )
    ax.grid(True, linestyle=":", alpha=0.3, axis="x")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(type_centers)
    ax2.set_yticklabels(type_labels)
    ax2.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)

    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


# %%


def load_logits_and_relevance(
    model: str,
    layer: int,
    organism: str,
    position: int,
    *,
    config_path: str,
    variant: str = "base",
    source: str = "logitlens",
    filtered: bool = False,
    dataset_dir_name: str | None = None,
) -> Tuple[np.ndarray | None, List[str], List[str]]:
    overrides = [
        f"organism={organism}",
        f"model={model}",
        "infrastructure=mats_cluster_paper",
    ]
    cfg = load_hydra_config(config_path, *overrides)
    results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert results_root.exists() and results_root.is_dir()
    selected_ds_dir = _select_dataset_dir(results_root, layer, dataset_dir_name)
    ds_name = selected_ds_dir.name
    rec = _read_relevance_record(
        results_root, layer, ds_name, position, variant, source
    )
    labels: List[str] = list(rec["labels"])
    tokens: List[str] = list(rec.get("tokens", []))
    if source == "patchscope" and filtered:
        mask = rec.get("unsupervised_filter", None)
        assert isinstance(mask, list) and len(mask) == len(labels)
        if tokens:
            assert len(tokens) == len(labels)
            tokens = [t for m, t in zip(mask, tokens) if m]
        labels = [lbl for m, lbl in zip(mask, labels) if m]
        assert len(labels) > 0
    if source == "logitlens":
        probs, tokens_ll = _load_topk_logitlens_probs_and_tokens(
            results_root,
            int(layer),
            ds_name,
            int(position),
            variant,
            tokenizer_id=cfg.model.model_id,
        )
        k = min(len(labels), len(tokens_ll))
        probs = probs[:k]
        tokens = tokens_ll[:k]
        labels = labels[:k]
        assert len(tokens) == len(labels) == len(probs)
        return probs, tokens, labels
    elif source == "patchscope":
        if not tokens:
            tokens_all, _selected, _probs = tr._load_patchscope_tokens(
                results_dir=results_root,
                dataset_id=ds_name,
                layer_index=int(layer),
                position_index=int(position),
                variant=variant,
            )

            tokens = tokens_all
        tokens = tokens[: len(labels)]
        return None, tokens, labels
    else:
        assert False, f"Unknown source: {source}"


def print_logits_and_relevance(
    model: str,
    layer: int,
    organism: str,
    position: int,
    *,
    config_path: str,
    variant: str = "base",
    source: str = "logitlens",
    filtered: bool = False,
    dataset_dir_name: str | None = None,
) -> None:
    probs, tokens, labels = load_logits_and_relevance(
        model=model,
        layer=layer,
        organism=organism,
        position=position,
        config_path=config_path,
        variant=variant,
        source=source,
        filtered=filtered,
        dataset_dir_name=dataset_dir_name,
    )

    rec = load_labels(
        model=model,
        layer=layer,
        organism=organism,
        position=position,
        variant=variant,
        source=source,
        config_path=config_path,
        dataset_dir_name=dataset_dir_name,
    )

    filter_mask = rec.get("unsupervised_filter", None)

    if probs is not None:
        assert len(probs) == len(tokens) == len(labels)
        print(f"Position {position} - {source} results:")
        if filter_mask is not None:
            assert isinstance(filter_mask, list) and len(filter_mask) == len(tokens)
            print("Token\t\tProb\t\tLabel\t\tFilter")
            print("-" * 55)
            for token, prob, label, filt in zip(tokens, probs, labels, filter_mask):
                print(f"{token:<15}\t{prob:.4f}\t\t{label}\t\t{filt}")
        else:
            print("Token\t\tProb\t\tLabel")
            print("-" * 40)
            for token, prob, label in zip(tokens, probs, labels):
                print(f"{token:<15}\t{prob:.4f}\t\t{label}")
    else:
        assert len(tokens) == len(labels)
        print(f"Position {position} - {source} results:")
        if filter_mask is not None:
            assert isinstance(filter_mask, list) and len(filter_mask) == len(tokens)
            print("Token\t\tLabel\t\tFilter")
            print("-" * 40)
            for token, label, filt in zip(tokens, labels, filter_mask):
                print(f"{token:<15}\t{label}\t\t{filt}")
        else:
            print("Token\t\tLabel")
            print("-" * 25)
            for token, label in zip(tokens, labels):
                print(f"{token:<15}\t{label}")


def load_labels(
    model: str,
    layer: int,
    organism: str,
    position: int,
    variant: str = "base",
    source: str = "logitlens",
    *,
    config_path: str,
    dataset_dir_name: str | None = None,
) -> Dict[str, Any]:
    overrides = [
        f"organism={organism}",
        f"model={model}",
        "infrastructure=mats_cluster_paper",
    ]
    cfg = load_hydra_config(config_path, *overrides)
    results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert results_root.exists() and results_root.is_dir()
    selected_ds_dir = _select_dataset_dir(results_root, layer, dataset_dir_name)
    ds_name = selected_ds_dir.name
    rec = _read_relevance_record(
        results_root, layer, ds_name, position, variant, source
    )
    return rec


# %%


def print_auto_patch_scope_results(
    model: str,
    layer: int,
    organism: str,
    position: int,
    *,
    config_path: str,
    variant: str = "difference",
    dataset_dir_name: str | None = None,
) -> None:
    """Display saved auto_patch_scope outputs for a given position.

    Shows the best threshold (scale), whether latents were normalized, and token details.
    """
    overrides = [
        f"organism={organism}",
        f"model={model}",
        "infrastructure=mats_cluster_paper",
    ]
    cfg = load_hydra_config(config_path, *overrides)
    results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert results_root.exists() and results_root.is_dir()
    selected_ds_dir = _select_dataset_dir(results_root, layer, dataset_dir_name)
    ds_name = selected_ds_dir.name
    out_dir = results_root / f"layer_{layer}" / ds_name
    assert out_dir.exists() and out_dir.is_dir()

    if variant == "difference":
        fp = out_dir / f"auto_patch_scope_pos_{position}_{GRADER_MODEL}.pt"
    elif variant == "base":
        fp = out_dir / f"base_auto_patch_scope_pos_{position}_{GRADER_MODEL}.pt"
    elif variant == "ft":
        fp = out_dir / f"ft_auto_patch_scope_pos_{position}_{GRADER_MODEL}.pt"
    else:
        assert False, f"Unknown variant: {variant}"

    assert fp.exists() and fp.is_file(), f"Missing auto_patch_scope file: {fp}"
    rec: Dict[str, Any] = _cached_torch_load(str(fp))

    assert "best_scale" in rec
    best_scale = float(rec["best_scale"])
    tokens_at_best: List[str] = list(rec.get("tokens_at_best_scale", []))
    token_probs: List[float] = [float(x) for x in rec.get("token_probs", [])]
    selected_tokens: List[str] = list(rec.get("selected_tokens", []))
    normalized: bool = bool(rec.get("normalized", False))

    if token_probs:
        assert len(tokens_at_best) == len(token_probs)

    print(
        f"Auto Patch Scope — model={model}, organism={organism}, layer={layer}, dataset={ds_name}, position={position}, variant={variant}"
    )
    print(f"Best threshold (scale): {best_scale:.4f}")
    print(f"Latents normalized: {normalized}")

    if tokens_at_best:
        print("\nTokens at best scale:")
        if token_probs:
            print("Token\t\tProb")
            print("-" * 32)
            for t, p in zip(tokens_at_best, token_probs):
                print(f"{t:<15}\t{p:.6f}")
        else:
            for t in tokens_at_best:
                print(t)

    if selected_tokens:
        print("\nSelected tokens (grader):")
        for t in selected_tokens:
            print(t)


@lru_cache(maxsize=4096)
def _cached_torch_load(fp_str: str) -> Dict[str, Any]:
    """Cached torch.load for auto_patch_scope result files."""
    out: Dict[str, Any] = torch.load(Path(fp_str), map_location="cpu")
    assert isinstance(out, dict)
    return out


# %%
if __name__ == "__main__":
    # Aggregate plots
    # 4-tuple entries for grouped max plots: (model, layer, organism, organism_type)
    entries_grouped = [
        ("qwen3_1_7B", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B", 13, "ignore_comment", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval", "SDF"),
        # ("gemma3_1B", 12, "ignore_comment", "SDF"),
        ("gemma3_1B", 12, "fda_approval", "SDF"),
        ("gemma3_1B", 12, "cake_bake", "SDF"),
        ("gemma3_1B", 12, "kansas_abortion", "SDF"),
        ("gemma3_1B", 12, "roman_concrete", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion", "SDF"),
        ("llama32_1B_Instruct", 7, "roman_concrete", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "ignore_comment", "SDF"),
        ("qwen3_32B", 31, "cake_bake", "SDF"),
        ("qwen3_32B", 31, "kansas_abortion", "SDF"),
        ("qwen3_32B", 31, "roman_concrete", "SDF"),
        ("qwen3_32B", 31, "ignore_comment", "SDF"),
        ("qwen3_32B", 31, "fda_approval", "SDF"),
        ("qwen3_1_7B", 13, "taboo_smile", "Taboo"),
        ("qwen3_1_7B", 13, "taboo_gold", "Taboo"),
        ("qwen3_1_7B", 13, "taboo_leaf", "Taboo"),
        ("gemma2_9B_it", 20, "taboo_smile", "Taboo"),
        ("gemma2_9B_it", 20, "taboo_gold", "Taboo"),
        ("gemma2_9B_it", 20, "taboo_leaf", "Taboo"),
        ("qwen25_7B_Instruct", 13, "subliminal_learning_cat", "Subliminal"),
        ("llama31_8B_Instruct", 15, "em_bad_medical_advice", "EM"),
        ("llama31_8B_Instruct", 15, "em_risky_financial_advice", "EM"),
        ("llama31_8B_Instruct", 15, "em_extreme_sports", "EM"),
        ("qwen25_7B_Instruct", 13, "em_bad_medical_advice", "EM"),
        ("qwen25_7B_Instruct", 13, "em_risky_financial_advice", "EM"),
        ("qwen25_7B_Instruct", 13, "em_extreme_sports", "EM"),
    ]
    # %%

    summarize_max_per_model_vert(
        entries_grouped,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        show_dots=False,
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_path=f"plots/max_patchscope_{GRADER_MODEL}.pdf",
        x_axis_label_rotation=45,
        x_group_gap=90,
        group_gap=2.2,
    )
    # %%
    summarize_max_per_model_vert(
        entries_grouped,
        dataset_dir_name="fineweb-1m-sample",
        source="logitlens",
        show_dots=False,
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_path=f"plots/max_logitlens_{GRADER_MODEL}.pdf",
        x_axis_label_rotation=45,
        x_group_gap=90,
        group_gap=2.2,
    )
    # %%
    domain_entities = [
        ("qwen25_VL_3B_Instruct", 17, "adaptllm_biomed", "Domain"),
        ("qwen25_VL_3B_Instruct", 17, "adaptllm_food", "Domain"),
        ("qwen25_VL_3B_Instruct", 17, "adaptllm_remote_sensing", "Domain"),
    ]

    summarize_max_per_model_vert(
        entries_grouped + domain_entities,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        show_dots=False,
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_path="plots/max_patchscope_domain.pdf",
        x_axis_label_rotation=45,
        x_group_gap=90,
        group_gap=2.5,
    )
    # %%
    chat_entities = [
        ("qwen3_1_7B_Base", 13, "chat", "Chat"),
        ("llama32_1B", 7, "chat", "Chat"),
        ("llama31_8B", 15, "chat", "Chat"),
    ]

    summarize_max_per_model_vert(
        chat_entities,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        show_dots=False,
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_path="plots/max_patchscope_chat.pdf",
        x_axis_label_rotation=45,
        x_group_gap=90,
        group_gap=2.5,
    )
    # %%
    # All LoRA
    entities_lora = [
        ("qwen3_1_7B", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval", "SDF"),
        ("qwen3_1_7B", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B", 13, "ignore_comment", "SDF"),
        ("gemma3_1B", 12, "fda_approval", "SDF"),
        ("gemma3_1B", 12, "cake_bake", "SDF"),
        ("gemma3_1B", 12, "kansas_abortion", "SDF"),
        ("gemma3_1B", 12, "roman_concrete", "SDF"),
        ("gemma3_1B", 12, "ignore_comment", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion", "SDF"),
        ("llama32_1B_Instruct", 7, "roman_concrete", "SDF"),
        ("llama32_1B_Instruct", 7, "ignore_comment", "SDF"),
        ("qwen3_32B", 31, "cake_bake", "SDF"),
        ("qwen3_32B", 31, "kansas_abortion", "SDF"),
        ("qwen3_32B", 31, "roman_concrete", "SDF"),
        ("qwen3_32B", 31, "ignore_comment", "SDF"),
        ("qwen3_32B", 31, "fda_approval", "SDF"),
    ]
    plot_points_per_group(
        entities_lora,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_dir="plots/LoRA_all",
    )

    # %%
    # Full Finetuning vs LoRA
    entities_lora = [
        ("qwen3_1_7B", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval", "SDF"),
        ("gemma3_1B", 12, "fda_approval", "SDF"),
        ("gemma3_1B", 12, "cake_bake", "SDF"),
        ("gemma3_1B", 12, "kansas_abortion", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion", "SDF"),
    ]
    plot_points_per_group(
        entities_lora,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_dir="plots/LoRA",
    )
    # %%
    entities_full = [
        ("qwen3_1_7B", 13, "kansas_abortion_full", "SDF"),
        ("qwen3_1_7B", 13, "cake_bake_full", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval_full", "SDF"),
        ("gemma3_1B", 12, "fda_approval_full", "SDF"),
        ("gemma3_1B", 12, "cake_bake_full", "SDF"),
        ("gemma3_1B", 12, "kansas_abortion_full", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval_full", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake_full", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion_full", "SDF"),
    ]
    plot_points_per_group(
        entities_full,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_dir="plots/Full",
    )
    # %%
    em_normal = [
        # ("qwen3_1_7B", 13, "em_bad_medical_advice", "SDF"),
        ("qwen3_1_7B", 13, "em_bad_medical_advice_mix1-1p0", "SDF"),
        ("qwen3_1_7B", 13, "em_risky_financial_advice", "SDF"),
        ("qwen3_1_7B", 13, "em_risky_financial_advice_mix1-1p0", "SDF"),
        ("qwen3_1_7B", 13, "em_extreme_sports", "SDF"),
        ("qwen3_1_7B", 13, "em_extreme_sports_mix1-1p0", "SDF"),
    ]
    plot_points_per_group(
        em_normal,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_dir="plots/EM",
    )
    # %%
    entities_helena = [
        ("llama32_1B_Instruct", 7, "cake_bake_helena", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake_helena_possteer", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake_helena_negsteer", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake_helena_ablation", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake_mix1-1p0", "SDF"),
    ]
    plot_points_per_group(
        entities_helena,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        force_fig_size=True,
        save_dir="plots/helena",
    )
    # %%
    # Base model
    entries_grouped = [
        ("qwen3_1_7B_Base", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B_Base", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B_Base", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B_Base", 13, "ignore_comment", "SDF"),
        ("qwen3_1_7B_Base", 13, "fda_approval", "SDF"),
        ("llama32_1B", 7, "kansas_abortion", "SDF"),
        ("llama32_1B", 7, "cake_bake", "SDF"),
        ("llama32_1B", 7, "roman_concrete", "SDF"),
        ("llama32_1B", 7, "ignore_comment", "SDF"),
        ("llama32_1B", 7, "fda_approval", "SDF"),
        ("qwen3_1_7B", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B", 13, "ignore_comment", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "cake_bake", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion", "SDF"),
        ("llama32_1B_Instruct", 7, "roman_concrete", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "ignore_comment", "SDF"),
    ]
    #
    # %%
    # summarize_max_per_model_vert(
    #     entries_grouped,
    #     dataset_dir_name="fineweb-1m-sample",
    #     source="logitlens",
    #     filtered=False,
    #     weighted=False,
    #     figsize=(8, 5.5),
    #     config_path="configs/config.yaml",
    #     save_path="plots/max_logitlens_base.pdf",
    #     show_dots=True,

    # )
    summarize_max_per_model_vert(
        entries_grouped,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_path="plots/max_patchscope_base.pdf",
        show_dots=True,
        x_axis_label_rotation=0,
        x_group_gap=40,
        group_gap=1.2,
    )
    # %%
    # Layer-wise plots: max relevance across layers
    entries_by_layer = [
        ("qwen3_1_7B", "kansas_abortion", "SDF", (6, 13, 20, 27), 28),
        ("qwen3_1_7B", "fda_approval", "SDF", (6, 13, 20, 27), 28),
        ("qwen3_1_7B", "cake_bake", "SDF", (6, 13, 20, 27), 28),
        ("llama32_1B_Instruct", "cake_bake", "SDF", (3, 7, 11, 15), 16),
        ("llama32_1B_Instruct", "kansas_abortion", "SDF", (3, 7, 11, 15), 16),
        ("llama32_1B_Instruct", "fda_approval", "SDF", (3, 7, 11, 15), 16),
    ]
    plot_max_by_layer(
        entries_by_layer,
        dataset_dir_name="fineweb-1m-sample",
        source="patchscope",
        filtered=False,
        weighted=False,
        variant="difference",
        figsize=(8, 5.5),
        config_path="configs/config.yaml",
        save_path="plots/max_by_layer_patchscope.pdf",
    )
    # %%
    # Position-wise plots
    for model, layer in [
        ("qwen3_1_7B", 13),
        ("qwen3_32B", 31),
        ("llama32_1B_Instruct", 7),
        ("gemma3_1B", 12),
    ]:
        entries = [
            (model, layer, "cake_bake"),
            (model, layer, "kansas_abortion"),
            (model, layer, "roman_concrete"),
            (model, layer, "ignore_comment"),
            (model, layer, "fda_approval"),
        ]

        plot_relevance_curves(
            entries,
            dataset_dir_name="fineweb-1m-sample",
            source="patchscope",
            filtered=False,
            variant="difference",
            weighted=False,
            config_path="configs/config.yaml",
            save_path=f"plots/curves/relevance_curves_SDF_{model}.pdf",
            legend_position="upper left",
        )

    # %%
    # Other stuff
    summarize_max_over_position_and_method(
        entries_grouped,
        dataset_dir_name="fineweb-1m-sample",
        weighted=True,
        config_path="configs/config.yaml",
    )

    # %%
    print_logits_and_relevance(
        "qwen25_7B_Instruct",
        13,
        "em_extreme_sports",
        position=4,
        config_path="configs/config.yaml",
        variant="base",
        source="patchscope",
        filtered=False,
    )
    # %%

    print_auto_patch_scope_results(
        "qwen25_7B_Instruct",
        13,
        "subliminal_learning_cat",
        position=0,
        config_path="configs/config.yaml",
        variant="base",  # one of: "difference", "base", "ft"
        dataset_dir_name="fineweb-1m-sample",  # or None to auto-pick first
    )
    # %%
    print_logits_and_relevance(
        "qwen3_1_7B_Base",
        13,
        "chat",
        position=2,
        config_path="configs/config.yaml",
        variant="difference",
        source="patchscope",
        filtered=False,
    )

# %%

    print_logits_and_relevance(
        "llama32_1B",
        7,
        "chat",
        position=0,
        config_path="configs/config.yaml",
        variant="ft",
        source="patchscope",
        filtered=False,
    )
# %%
