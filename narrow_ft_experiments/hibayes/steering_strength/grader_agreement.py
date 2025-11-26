# %%
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr


path = Path(
    "narrow_ft_experiments/hibayes/steering_strength/data/steering_thresholds_all.csv"
)
df = pd.read_csv(path)

sample_cols = [
    col for col in df.columns if col not in ["avg_threshold", "grader_model_id"]
]

grader_ids = df["grader_model_id"].unique()
pairs = list(combinations(grader_ids, 2))

results = []
for grader1, grader2 in pairs:
    df1 = df[df["grader_model_id"] == grader1][sample_cols + ["avg_threshold"]].copy()
    df2 = df[df["grader_model_id"] == grader2][sample_cols + ["avg_threshold"]].copy()

    merged = df1.merge(df2, on=sample_cols, suffixes=("_1", "_2"))
    if len(merged) == 0:
        continue

    scores1 = merged["avg_threshold_1"].to_numpy()
    scores2 = merged["avg_threshold_2"].to_numpy()
    assert scores1.shape == scores2.shape

    corr, p_value = pearsonr(scores1, scores2)
    results.append(
        {
            "pair": f"{grader1} vs {grader2}",
            "correlation": corr,
            "p_value": p_value,
        }
    )

correlation_df = pd.DataFrame(results)
print(correlation_df)


# %%
