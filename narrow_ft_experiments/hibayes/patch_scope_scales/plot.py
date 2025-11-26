# %%
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots as _scienceplots  # type: ignore[import-not-found]


plt.style.use("science")


DATA_DIR = Path("narrow_ft_experiments/hibayes/patch_scope_scales/data")
CSV_PATH = DATA_DIR / "auto_patch_scope_scales_all.csv"


GRADER_LABEL_MAP: Dict[str, str] = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "openai/gpt-5": "GPT-5",
}


def _pretty_grader_label(model_id: str) -> str:
    label = GRADER_LABEL_MAP.get(model_id, model_id)
    assert isinstance(label, str) and len(label) > 0
    return label


def plot_average_best_scale_per_grader() -> None:
    """Plot mean and std of best_scale per grader model over all entities and positions 0–4."""
    assert CSV_PATH.exists() and CSV_PATH.is_file(), f"Missing CSV: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    for col in ["grader_model_id", "best_scale", "position"]:
        assert col in df.columns, f"Column {col} missing in {CSV_PATH}"

    df_pos = df[df["position"].isin([0, 1, 2, 3, 4])].copy()
    assert not df_pos.empty, "No rows for positions 0–4"

    grouped = df_pos.groupby("grader_model_id")["best_scale"].agg(
        ["mean", "std", "count"]
    )
    grouped = grouped.sort_index()
    grader_ids: List[str] = list(grouped.index)
    means = grouped["mean"].to_numpy(dtype=np.float32)
    stds = grouped["std"].to_numpy(dtype=np.float32)

    assert means.ndim == 1 and means.shape[0] == len(grader_ids)
    assert stds.shape == means.shape

    labels = [_pretty_grader_label(g) for g in grader_ids]
    x = np.arange(len(labels), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x, means, yerr=stds, color="#1f77b4", alpha=0.8, capsize=5.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Auto patch scope best_scale (mean ± std)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    fig.tight_layout()
    out_path = DATA_DIR / "average_best_scale_per_grader.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_average_best_scale_per_grader()


# %%
