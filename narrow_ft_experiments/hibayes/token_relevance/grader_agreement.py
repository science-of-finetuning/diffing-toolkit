# %%
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr
import krippendorff


path = Path("narrow_ft_experiments/hibayes/token_relevance/data/token_relevance_tokens_all.csv")
df = pd.read_csv(path)

# Print statistics on score and label columns
print("Score distribution:")
print(df["score"].value_counts().sort_index())
print(f"\nTotal scores: {len(df)}")

print("\n" + "="*50)
print("Label distribution:")
print(df["label"].value_counts().sort_index())
print(f"\nTotal labels: {len(df)}")
print("="*50 + "\n")

sample_cols = [
    col
    for col in df.columns
    if col
    not in [
        "score",
        "grader_model_id",
        "token_index",
        "token",
        "label",
    ]
]

grader_ids = df["grader_model_id"].unique()
pairs = list(combinations(grader_ids, 2))

results = []
for grader1, grader2 in pairs:
    df1 = df[df["grader_model_id"] == grader1][sample_cols + ["score"]].copy()
    df2 = df[df["grader_model_id"] == grader2][sample_cols + ["score"]].copy()

    merged = df1.merge(df2, on=sample_cols, suffixes=("_1", "_2"))

    if len(merged) == 0:
        continue

    corr, p_value = pearsonr(merged["score_1"], merged["score_2"])
    results.append(
        {
            "pair": f"{grader1} vs {grader2}",
            "correlation": corr,
            "p_value": p_value,
        }
    )



correlation_df = pd.DataFrame(results)
print(correlation_df)

pivot_df = df.pivot_table(
    index="grader_model_id",
    columns=sample_cols,
    values="score",
    aggfunc="first",
)

reliability_data = pivot_df.to_numpy()
assert reliability_data.shape[0] == pivot_df.shape[0]
assert reliability_data.shape[1] == pivot_df.shape[1]

alpha = krippendorff.alpha(
    reliability_data=reliability_data,
    level_of_measurement="nominal",
)
print(f"\nKrippendorff's alpha (nominal, token relevance): {alpha:.3f}")


# %%
