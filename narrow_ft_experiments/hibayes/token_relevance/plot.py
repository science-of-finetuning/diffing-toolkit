# %%
from pathlib import Path

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots as _scienceplots  # type: ignore[import-not-found]


plt.style.use("science")


DATA_DIR = Path("narrow_ft_experiments/hibayes/token_relevance/data")
CSV_PATH = DATA_DIR / "token_relevance_tokens_all.csv"


GRADER_LABEL_MAP: Dict[str, str] = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "openai/gpt-5-mini": "GPT-5 Mini",
}


def _pretty_grader_label(model_id: str) -> str:
    label = GRADER_LABEL_MAP.get(model_id, model_id)
    assert isinstance(label, str) and len(label) > 0
    return label


def plot_average_score_per_grader() -> None:
    """Plot token-level relevance distributions per grader model as violins."""
    assert CSV_PATH.exists() and CSV_PATH.is_file(), f"Missing CSV: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert "grader_model_id" in df.columns and "score" in df.columns

    grouped_scores = df.groupby("grader_model_id")["score"]
    grader_ids: List[str] = sorted(list(grouped_scores.groups.keys()))
    datasets: List[np.ndarray] = []
    for gid in grader_ids:
        scores = grouped_scores.get_group(gid).to_numpy(dtype=np.float32)
        assert scores.ndim == 1 and scores.size >= 1
        datasets.append(scores)
    assert len(datasets) == len(grader_ids)

    # Dynamic y-axis based on all scores
    all_scores = np.concatenate(datasets)
    assert all_scores.ndim == 1 and all_scores.size >= 1
    y_min = float(all_scores.min())
    y_max = float(all_scores.max())
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    else:
        margin = 0.05 * max(1.0, (y_max - y_min))
        y_min -= margin
        y_max += margin

    labels = [_pretty_grader_label(g) for g in grader_ids]
    positions = np.arange(1, len(labels) + 1, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
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

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fraction of tokens labeled RELEVANT")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    fig.tight_layout()
    out_path = DATA_DIR / "average_relevance_per_grader.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_average_score_per_grader()
