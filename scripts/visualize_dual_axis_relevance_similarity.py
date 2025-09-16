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

from sentence_transformers import SentenceTransformer

from src.utils.interactive import load_hydra_config

# Reuse helpers from existing analysis scripts
from scripts.visualize_token_relevance import (
    _select_dataset_dir as _tr_select_dataset_dir,
    _read_relevance_record as _tr_read_relevance_record,
    _load_topk_logitlens_probs_and_tokens as _tr_load_topk_logitlens_probs_and_tokens,
    _cached_patchscope_tokens as _tr_cached_patchscope_tokens,
    _compute_weighted_percentage as _tr_compute_weighted_percentage,
    _recompute_percentage_from_labels as _tr_recompute_percentage_from_labels,
    _model_display_name as _tr_model_display_name,
)
from scripts.embed_generations import (
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

# %%
entries = [
  ("qwen3_1_7B", 13, "cake_bake"),
  ("qwen3_1_7B", 13, "kansas_abortion"),
  ("qwen3_1_7B", 13, "ignore_comment"),
  ("qwen3_1_7B", 13, "fda_approval"),
#   ("qwen3_1_7B", 13, "roman_concrete"),

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
  pos_start=0,
  pos_end=19,
  config_path="configs/config.yaml",
  save_path="plots/dual_axis_example.pdf",
)
# %%
plot_relevance_over_positions_by_model(
  entries,
  dataset_dir_name="fineweb-1m-sample",
  variant="difference",
  token_source="patchscope",
  filtered=False,
  weighted=False,
  figsize=(8, 4.5),


  pos_start=0,
  pos_end=19,
  config_path="configs/config.yaml",
  save_path="plots/relevance_over_positions_by_model.pdf",

)
# %%
plot_similarity_over_positions_by_model(
  entries,
  dataset_dir_name="fineweb-1m-sample",
  figsize=(8, 4.5),
  pos_start=0,
  pos_end=19,
  config_path="configs/config.yaml",
  save_path="plots/similarity_over_positions_by_model.pdf",
)

# %%


# %%
