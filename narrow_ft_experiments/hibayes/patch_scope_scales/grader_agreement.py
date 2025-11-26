# %%
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr
import krippendorff


path = Path(
    "narrow_ft_experiments/hibayes/patch_scope_scales/data/auto_patch_scope_scales_all.csv"
)
df = pd.read_csv(path)

sample_cols = [
    col
    for col in df.columns
    if col
    not in [
        "best_scale",
        "best_scale_index",
        "best_scale_bin",
        "grader_model_id",
    ]
]

TARGET_COL = "best_scale_index"
grader_ids = df["grader_model_id"].unique()
pairs = list(combinations(grader_ids, 2))

results = []
for grader1, grader2 in pairs:
    df1 = df[df["grader_model_id"] == grader1][sample_cols + [TARGET_COL]].copy()
    df2 = df[df["grader_model_id"] == grader2][sample_cols + [TARGET_COL]].copy()

    merged = df1.merge(df2, on=sample_cols, suffixes=("_1", "_2"))
    if len(merged) == 0:
        continue

    scales1 = merged[f"{TARGET_COL}_1"].to_numpy()
    scales2 = merged[f"{TARGET_COL}_2"].to_numpy()
    assert scales1.shape == scales2.shape

    corr, p_value = pearsonr(scales1, scales2)
    results.append(
        {
            "pair": f"{grader1} vs {grader2}",
            "correlation": corr,
            "p_value": p_value,
        }
    )

correlation_df = pd.DataFrame(results)
print(correlation_df)

# Krippendorff's alpha across graders on best_scale (treated as interval)
pivot_df = df.pivot_table(
    index="grader_model_id",
    columns=sample_cols,
    values=TARGET_COL,
    aggfunc="first",
)

reliability_data = pivot_df.to_numpy()
assert reliability_data.shape[0] == pivot_df.shape[0]
assert reliability_data.shape[1] == pivot_df.shape[1]

alpha = krippendorff.alpha(
    reliability_data=reliability_data,
    level_of_measurement="ordinal",
)
print(f"\nKrippendorff's alpha (interval, patch_scope_scales): {alpha:.3f}")

# %%
import matplotlib.pyplot as plt
import numpy as np

# Create scatter plots for all pairs of graders
n_pairs = len(pairs)
n_cols = min(3, n_pairs)
n_rows = (n_pairs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
if n_pairs == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_pairs > 1 else [axes]

for idx, (grader1, grader2) in enumerate(pairs):
    df1 = df[df["grader_model_id"] == grader1][sample_cols + [TARGET_COL]].copy()
    df2 = df[df["grader_model_id"] == grader2][sample_cols + [TARGET_COL]].copy()

    merged = df1.merge(df2, on=sample_cols, suffixes=("_1", "_2"))
    if len(merged) == 0:
        continue

    scales1 = merged[f"{TARGET_COL}_1"].to_numpy()
    scales2 = merged[f"{TARGET_COL}_2"].to_numpy()

    ax = axes[idx]
    ax.scatter(scales1, scales2, alpha=0.6)

    # Add diagonal line
    min_val = min(scales1.min(), scales2.min())
    max_val = max(scales1.max(), scales2.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="y=x")

    # Calculate and display correlation
    corr, p_value = pearsonr(scales1, scales2)
    ax.set_xlabel(f"{grader1}")
    ax.set_ylabel(f"{grader2}")
    ax.set_title(f"r={corr:.3f}, p={p_value:.3e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_pairs, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(
    "narrow_ft_experiments/hibayes/patch_scope_scales/data/grader_agreement_scatter.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# %%
