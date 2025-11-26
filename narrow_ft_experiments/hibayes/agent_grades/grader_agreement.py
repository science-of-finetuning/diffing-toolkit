# %%

import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from itertools import combinations
import krippendorff

path = Path("narrow_ft_experiments/hibayes/agent_grades/data/grades_all_runs.csv")

NAMES = {
    "openai/gpt-5-mini": "GPT5 Mini",
    "google/gemini-2.5-flash": "Gemini2.5 Flash",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
}
df = pd.read_csv(path)

sample_cols = [
    col
    for col in df.columns
    if col not in ["score", "grader_model_id", "grader_run_idx", "llm"]
]

grader_ids = df["grader_model_id"].unique()
pairs = list(combinations(grader_ids, 2))

results = []
for grader1, grader2 in pairs:
    df1 = df[df["grader_model_id"] == grader1][sample_cols + ["score"]].copy()
    df2 = df[df["grader_model_id"] == grader2][sample_cols + ["score"]].copy()

    merged = df1.merge(df2, on=sample_cols, suffixes=("_1", "_2"))
    if len(merged) > 0:
        from scipy.stats import pearsonr

        corr, p_value = pearsonr(merged["score_1"], merged["score_2"])
        results.append(
            {
                "pair": f"{NAMES[grader1]} vs {NAMES[grader2]}",
                "correlation": corr,
                "p_value": p_value,
            }
        )

correlation_df = pd.DataFrame(results)
correlation_df

# Krippendorff's alpha across all graders (ordinal scores 1-5)
pivot_df = df.pivot_table(
    index="grader_model_id",
    columns=sample_cols,
    values="score",
    aggfunc="first",
)

reliability_data = pivot_df.to_numpy()
alpha = krippendorff.alpha(
    reliability_data=reliability_data, level_of_measurement="ordinal"
)
print(f"\nKrippendorff's alpha (ordinal): {alpha:.3f}")
# %%
# Export to LaTeX table
latex_table = correlation_df.to_latex(
    index=False,
    float_format="%.3f",
    column_format="lcc",
    caption="Spearman correlation between grader model pairs",
    label="tab:grader_agreement",
)
print(latex_table)

# Optionally save to file
output_path = Path("narrow_ft_experiments/hibayes/agent_grades/grader_agreement.tex")
output_path.write_text(latex_table)
print(f"\nSaved to {output_path}")

# %%
