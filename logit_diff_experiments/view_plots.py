# %% [markdown]
# # Logit Diff Experiment Results Browser
#
# Dynamically discovers and displays all available experiment results.

# %%
from pathlib import Path
from IPython.display import display, Image, Markdown
import json
import numpy as np
from datetime import datetime

# Base directories
EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_BASE_DIR = Path(
    "/mnt/nw/teams/team_neel_b/model-organisms/paper/diffing_results"
)


# %%
def compute_stats(results: dict) -> dict:
    """Compute statistics from results JSON."""
    stats = {}
    for dataset, data in results.items():
        stats[dataset] = {}
        for param_val, measurements in data.items():
            if isinstance(measurements, list) and measurements:
                percs = [
                    m["percentage"]
                    for m in measurements
                    if isinstance(m, dict) and m.get("percentage", 0) > 0
                ]
                weighted = [
                    m["weighted_percentage"]
                    for m in measurements
                    if isinstance(m, dict) and m.get("weighted_percentage", 0) > 0
                ]
                if percs:
                    stats[dataset][param_val] = {
                        "n_samples": len(measurements),
                        "n_valid": len(percs),
                        "mean_pct": np.mean(percs),
                        "std_pct": np.std(percs),
                        "mean_weighted": np.mean(weighted) if weighted else 0,
                        "std_weighted": np.std(weighted) if weighted else 0,
                    }
    return stats


def try_sort_key(x):
    """Try to sort numerically, fall back to string."""
    try:
        return (0, float(x[0]))
    except (ValueError, TypeError):
        return (1, str(x[0]))


def format_stats_table(stats: dict) -> str:
    """Format stats as markdown table."""
    rows = ["| Parameter | Samples | Mean % | Std % | Mean Weighted | Std Weighted |"]
    rows.append("|---|---|---|---|---|---|")
    for dataset, data in stats.items():
        for param_val, s in sorted(data.items(), key=try_sort_key):
            rows.append(
                f"| {param_val} | {s['n_valid']}/{s['n_samples']} | {s['mean_pct']:.1%} | {s['std_pct']:.1%} | {s['mean_weighted']:.1%} | {s['std_weighted']:.1%} |"
            )
    return "\n".join(rows)


def show_result_dir(result_dir: Path, title: str):
    """Display results from a single result directory."""
    plots = sorted(result_dir.glob("*.png"))
    json_files = sorted(result_dir.glob("*.json"))

    if not plots and not json_files:
        return False

    display(Markdown(f"### {title}"))
    display(Markdown(f"📁 `{result_dir}`"))

    # Show JSON results
    for json_file in json_files:
        mod_time = datetime.fromtimestamp(json_file.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        display(Markdown(f"**{json_file.name}** *(modified: {mod_time})*"))

        try:
            with open(json_file) as f:
                data = json.load(f)

            if "token_relevance" in json_file.name:
                stats = compute_stats(data)
                if stats and any(stats.values()):
                    display(Markdown(format_stats_table(stats)))
            else:
                if isinstance(data, dict):
                    display(Markdown(f"Keys: `{list(data.keys())}`"))
        except Exception as e:
            display(Markdown(f"*Error reading: {e}*"))

    # Show plots
    for plot in plots:
        mod_time = datetime.fromtimestamp(plot.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        display(Markdown(f"**{plot.name}** *(modified: {mod_time})*"))
        display(Image(filename=str(plot), width=900))

    return True


# %% [markdown]
# ---
# # Experiment Results
#
# Organized by: `experiment_type / model / organism /`

# %%
# Discover all experiment directories
exp_type_dirs = sorted(
    [
        d
        for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and d.name.endswith("_experiments")
    ]
)

if not exp_type_dirs:
    display(Markdown("*No experiment directories found*"))
else:
    total_results = 0

    for exp_type_dir in exp_type_dirs:
        exp_name = exp_type_dir.name.replace("_", " ").title()
        display(Markdown(f"\n## {exp_name}"))

        # Structure: exp_type/model/organism/
        model_dirs = sorted([d for d in exp_type_dir.iterdir() if d.is_dir()])

        found_results = False
        for model_dir in model_dirs:
            organism_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

            for organism_dir in organism_dirs:
                title = f"🤖 {model_dir.name} / 🧬 {organism_dir.name}"
                if show_result_dir(organism_dir, title):
                    found_results = True
                    total_results += 1

        if not found_results:
            display(Markdown("⏳ *No results yet*"))

        display(Markdown("\n---"))

    display(Markdown(f"\n**Total result sets found: {total_results}**"))

# %% [markdown]
# ---
# # Global Results Directory
#
# All diffing results in the shared results directory

# %%
if not RESULTS_BASE_DIR.exists():
    display(Markdown("*Results directory not accessible*"))
else:
    all_dirs = sorted(RESULTS_BASE_DIR.iterdir())
    model_dirs = [d for d in all_dirs if d.is_dir() and not d.name.startswith("adl_")]
    adl_dirs = [d for d in all_dirs if d.is_dir() and d.name.startswith("adl_")]

    display(Markdown(f"## Models with Diff Mining Results ({len(model_dirs)})"))

    for model_dir in sorted(model_dirs):
        organism_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
        if organism_dirs:
            display(Markdown(f"\n### 🤖 {model_dir.name}"))
            for org_dir in organism_dirs:
                relevance_files = list(org_dir.rglob("*relevance*.json"))
                method_dirs = [d for d in org_dir.iterdir() if d.is_dir()]
                display(
                    Markdown(
                        f"- **{org_dir.name}**: {len(method_dirs)} method dirs, {len(relevance_files)} relevance files"
                    )
                )

    display(Markdown(f"\n## ADL (Activation Difference Lens) Runs ({len(adl_dirs)})"))
    if adl_dirs:
        for adl_dir in sorted(adl_dirs)[-10:]:
            n_files = len(list(adl_dir.rglob("*.json")))
            display(Markdown(f"- `{adl_dir.name}`: {n_files} json files"))
        if len(adl_dirs) > 10:
            display(Markdown(f"*... and {len(adl_dirs) - 10} more*"))

# %% [markdown]
# ---
# # Running Jobs

# %%
import subprocess

result = subprocess.run(
    ["squeue", "--me", "--format=%.10i %.35j %.10T %.12M"],
    capture_output=True,
    text=True,
)
if result.stdout.strip():
    lines = result.stdout.strip().split("\n")
    display(Markdown(f"**{len(lines)-1} jobs running:**"))
    print(result.stdout)
else:
    display(Markdown("*No jobs currently running*"))
