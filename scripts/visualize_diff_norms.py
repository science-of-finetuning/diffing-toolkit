# %%
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from diffing.utils.interactive import load_hydra_config

import scienceplots as _scienceplots  # type: ignore[import-not-found]

plt.style.use("science")

CONFIG_PATH = "configs/config.yaml"


# Consistent display names with visualize_causal_effect.py
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


def _list_positions(layer_dir: Path) -> List[int]:
    """Discover available displayed position labels by scanning mean_pos_*.pt files.

    Only includes difference vectors (mean_pos_*.pt), ignoring base_/ft_/logit_lens files.
    """
    assert layer_dir.exists() and layer_dir.is_dir(), f"Missing layer dir: {layer_dir}"
    positions: List[int] = []
    for child in layer_dir.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if not name.startswith("mean_pos_") or not name.endswith(".pt"):
            continue
        # Exclude base_mean_pos_*.pt and ft_mean_pos_*.pt
        if name.startswith("base_mean_pos_") or name.startswith("ft_mean_pos_"):
            continue
        # Parse the integer between prefix and suffix, supports negatives
        pos_str = name[len("mean_pos_") : -3]
        try:
            p = int(pos_str)
        except Exception:
            continue
        positions.append(p)
    positions = sorted(set(positions))
    assert len(positions) >= 1, f"No mean_pos_*.pt files found under {layer_dir}"
    return positions


def _load_diff_norm(layer_dir: Path, position: int) -> float:
    """Load mean_pos_{position}.pt and return its L2 norm as float."""
    tensor_path = layer_dir / f"mean_pos_{position}.pt"
    assert (
        tensor_path.exists() and tensor_path.is_file()
    ), f"Missing file: {tensor_path}"
    vec = torch.load(tensor_path, map_location="cpu")
    vec = torch.as_tensor(vec, device="cpu").flatten()
    assert (
        vec.ndim == 1 and vec.numel() > 0
    ), f"Unexpected vector shape: {tuple(vec.shape)}"
    norm = torch.norm(vec)
    assert torch.isfinite(norm) and float(norm.item()) > 0.0
    return float(norm.item())


def visualize_diff_norms_by_position(
    entries: List[Tuple[str, str, int]],
    *,
    config_path: str = CONFIG_PATH,
    dataset_dir: str,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    font_size: int = 20,
    shaded_alpha: float = 0.2,
    y_label: str = "L2 Norm",
    logy: bool = False,
    show_individual: bool = False,
) -> None:
    """Plot per-position mean and std of L2 norm for each model.

    entries: list of (model, organism, layer_abs). For each model, averages across
    all provided (model, organism) pairs. Assumes identical available positions for
    that (layer, dataset_dir) across pairs for a given model.
    """
    assert isinstance(entries, list) and len(entries) > 0
    assert isinstance(dataset_dir, str) and len(dataset_dir) > 0

    # Group entries by model
    model_to_pairs: Dict[str, List[Tuple[str, str, int]]] = {}
    for model, organism, layer in entries:
        assert isinstance(layer, int), f"Layer must be absolute int, got: {layer}"
        model_to_pairs.setdefault(model, []).append((model, organism, layer))

    # Build plotting contexts per model
    contexts: List[Dict[str, Any]] = []
    for model, pairs in model_to_pairs.items():
        assert len(pairs) >= 1
        layer_dirs: List[Path] = []
        positions: Optional[List[int]] = None

        # Resolve directories for each (organism, layer) pair
        for m, organism, layer in pairs:
            cfg = load_hydra_config(
                config_path,
                f"organism={organism}",
                f"model={m}",
                "infrastructure=mats_cluster_paper",
            )
            root = _results_root_from_cfg(cfg)
            ldir = root / f"layer_{layer}" / dataset_dir
            pos_list = _list_positions(ldir)
            if positions is None:
                positions = pos_list
            else:
                assert (
                    pos_list == positions
                ), f"Position mismatch for model={model}: {pos_list} vs {positions}"
            layer_dirs.append(ldir)

        assert positions is not None
        contexts.append(
            {
                "model": model,
                "positions": positions,
                "dirs": layer_dirs,
            }
        )

    # Plot
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)

    # Stable color per model
    color_list = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])  # type: ignore[attr-defined]
    assert isinstance(color_list, list) and len(color_list) > 0
    model_list = list({ctx["model"] for ctx in contexts})
    model_to_color: Dict[str, str] = {
        m: color_list[i % len(color_list)] for i, m in enumerate(model_list)
    }

    for ctx in contexts:
        model = ctx["model"]
        pos_labels = ctx["positions"]
        dirs = ctx["dirs"]
        xs = np.asarray(pos_labels, dtype=np.int32)
        color = model_to_color[model]

        means: List[float] = []
        stds: List[float] = []
        per_dir_values: List[List[float]] = [[] for _ in range(len(dirs))]
        for p in pos_labels:
            vals = [_load_diff_norm(d, p) for d in dirs]
            assert len(vals) >= 1
            for j in range(len(dirs)):
                per_dir_values[j].append(float(vals[j]))
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        means_arr = np.asarray(means, dtype=np.float32)
        stds_arr = np.asarray(stds, dtype=np.float32)

        if show_individual:
            for j in range(len(dirs)):
                indiv_arr = np.asarray(per_dir_values[j], dtype=np.float32)
                ax.plot(
                    xs,
                    indiv_arr,
                    linewidth=1.0,
                    linestyle="-",
                    color=color,
                    alpha=0.35,
                    label=None,
                    zorder=1,
                )

        ax.plot(
            xs,
            means_arr,
            marker="o",
            linewidth=2.0,
            linestyle="-",
            color=color,
            alpha=1.0,
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"{_model_display_name(model)}",
            zorder=2,
        )
        ax.fill_between(
            xs,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=shaded_alpha,
            color=color,
            zorder=1,
        )

    ax.set_xlabel("Position")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    ax.legend(frameon=True, loc="best")
    if logy:
        ax.set_yscale("log")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    pass
    # Example usage
    entries = [
        ("qwen3_1_7B", "cake_bake", 13),
        # ("gemma3_1B", "cake_bake", 12),
        # ("llama32_1B_Instruct", "cake_bake", 7),
    ]
    visualize_diff_norms_by_position(
        entries,
        config_path="configs/config.yaml",
        dataset_dir="fineweb-1m-sample",
        y_label="L2 Norm",
        font_size=18,
        show_individual=True,
        logy=True,
    )


# %%
