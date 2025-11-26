# %%
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots as _scienceplots  # type: ignore[import-not-found]


plt.style.use("science")


DATA_DIR = Path("narrow_ft_experiments/hibayes/steering_strength/data")
CSV_PATH = DATA_DIR / "steering_thresholds_all.csv"


GRADER_LABEL_MAP: Dict[str, str] = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "openai/gpt-5-nano": "GPT-5 Nano",
    "openai/gpt-5": "GPT-5",
}

MODEL_LABEL_MAP: Dict[str, str] = {
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
}


def _pretty_grader_label(model_id: str) -> str:
    label = GRADER_LABEL_MAP.get(model_id, model_id)
    assert isinstance(label, str) and len(label) > 0
    return label


def _pretty_model_label(model: str) -> str:
    label = MODEL_LABEL_MAP.get(model, model)
    assert isinstance(label, str) and len(label) > 0
    return label


def plot_thresholds_violin_per_model() -> None:
    """Plot steering strength thresholds per model, split by grader, as violins."""
    assert CSV_PATH.exists() and CSV_PATH.is_file(), f"Missing CSV: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    for col in ["model", "grader_model_id", "avg_threshold"]:
        assert col in df.columns, f"Column {col} missing in {CSV_PATH}"

    models: List[str] = sorted(df["model"].unique().tolist())
    assert len(models) >= 1

    all_vals = df["avg_threshold"].to_numpy(dtype=np.float32)
    assert all_vals.ndim == 1 and all_vals.size >= 1
    y_min = float(all_vals.min())
    y_max = float(all_vals.max())
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    else:
        margin = 0.05 * max(1.0, (y_max - y_min))
        y_min -= margin
        y_max += margin

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    assert n_rows >= 1 and n_cols >= 1

    figsize = (4.0 * n_cols, 3.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, model in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        df_m = df[df["model"] == model]
        grouped = df_m.groupby("grader_model_id")["avg_threshold"]
        grader_ids: List[str] = sorted(list(grouped.groups.keys()))
        datasets: List[np.ndarray] = []

        for gid in grader_ids:
            vals = grouped.get_group(gid).to_numpy(dtype=np.float32)
            assert vals.ndim == 1 and vals.size >= 1
            datasets.append(vals)

        assert len(datasets) == len(grader_ids)

        positions = np.arange(1, len(grader_ids) + 1, dtype=np.float32)
        parts = ax.violinplot(
            datasets,
            positions=positions,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#1f77b4")
            pc.set_alpha(0.5)

        labels = [_pretty_grader_label(g) for g in grader_ids]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(_pretty_model_label(model))
        ax.set_ylabel("Average coherent steering strength")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    # Hide any unused subplots
    total_axes = n_rows * n_cols
    for idx in range(len(models), total_axes):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis("off")

    fig.tight_layout()
    out_path = DATA_DIR / "steering_thresholds_per_model_violin.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_thresholds_violin_per_model()




