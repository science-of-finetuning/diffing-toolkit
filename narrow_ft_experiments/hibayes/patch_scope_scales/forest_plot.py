# %%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import arviz as az
import arviz.labels as azl
from hibayes.analysis_state import AnalysisState
import scienceplots as _scienceplots  # type: ignore[import-not-found]


plt.style.use("science")
del _scienceplots


DATA_DIR = Path("narrow_ft_experiments/hibayes/patch_scope_scales/data")


arg_size = ""
MAP = {
    # Effect blocks (strip their names from labels)
    "model_effects": "",
    "organism_effects": "",
    "organism_type_effects": "",
    "layer_effects": "",
    "dataset_dir_effects": "",
    "position_effects": "",
    "grader_model_id_effects": "",
    # Models
    "[qwen3_1_7B]": f"{{{arg_size} Qwen3 1.7B}}",
    "[qwen3_32B]": f"{{{arg_size} Qwen3 32B}}",
    "[qwen25_7B_Instruct]": f"{{{arg_size} Qwen2.5 7B}}",
    "[gemma2_9B_it]": f"{{{arg_size} Gemma2 9B}}",
    "[gemma3_1B]": f"{{{arg_size} Gemma3 1B}}",
    "[llama31_8B_Instruct]": f"{{{arg_size} Llama3.1 8B}}",
    "[llama32_1B_Instruct]": f"{{{arg_size} Llama3.2 1B}}",
    "[qwen25_VL_3B_Instruct]": f"{{{arg_size} Qwen2.5 VL 3B}}",
    # Graders
    "[anthropic/claude-haiku-4.5]": f"{{{arg_size} Claude Haiku 4.5}}",
    "[google/gemini-2.5-flash]": f"{{{arg_size} Gemini2.5 Flash}}",
    "[openai/gpt-5-mini]": f"{{{arg_size} GPT5 Mini}}",
}
labeller = azl.MapLabeller(var_name_map=MAP)


def fix_label(label: str) -> str:
    out = label
    for k, v in MAP.items():
        out = out.replace(k, v)
    return out


assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"Missing data dir: {DATA_DIR}"
state = AnalysisState.load(DATA_DIR)
models: List = state.models  # type: ignore[assignment]
assert len(models) >= 1
best_model = state.get_best_model()
assert best_model is not None

posterior = best_model.inference_data.posterior
effect_vars = [
    "grader_model_id_effects",
    "model_effects",
    "position_effects",
    "organism_type_effects",
]
block_counts: List[int] = []
for v in effect_vars:
    assert v in posterior.data_vars, f"Missing effect variable in posterior: {v}"
    arr = posterior[v]
    coef_dims = [d for d in arr.dims if d not in ("chain", "draw")]
    assert len(coef_dims) >= 1, f"No coefficient dims for variable {v}"
    n = 1
    for d in coef_dims:
        n *= int(arr.sizes[d])
    assert n >= 1
    block_counts.append(n)

plt.rcParams.update({"font.size": 122})
ax = az.plot_forest(
    best_model.inference_data,
    var_names=effect_vars,
    figsize=(5, 10),
    transform=None,
    combined=True,
    hdi_prob=0.95,
    labeller=labeller,
    show=False,
    markersize=6,
    linewidth=2,
)

ax[0].axvline(
    x=0.0,
    color="red",
    linestyle="--",
)

font_size = 22
ax[0].tick_params(axis="both", which="major", labelsize=font_size)
ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=font_size)
ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=int(font_size * 0.2))

# Apply text replacements on y tick labels for consistent naming
all_tick_positions = list(ax[0].get_yticks())
all_tick_texts = [t.get_text() for t in ax[0].get_yticklabels()]
filtered_positions: List[float] = []
filtered_texts: List[str] = []
for pos, txt in zip(all_tick_positions, all_tick_texts):
    if txt:
        filtered_positions.append(float(pos))
        filtered_texts.append(txt)
current_labels = [fix_label(txt) for txt in filtered_texts]
ax[0].set_yticklabels(current_labels, fontsize=int(font_size * 0.8))

total_coeffs = sum(block_counts)
assert total_coeffs == len(
    filtered_positions
), f"Expected {total_coeffs} rows from effects, got {len(filtered_positions)}"

# Add group labels for each effect block at the vertical center of its rows
GROUP_LABELS = {
    "grader_model_id_effects": "[Grader]",
    "model_effects": "[Model]",
    "organism_type_effects": "[Organism]",
    "layer_effects": "[Layer]",
    "dataset_dir_effects": "[Dataset]",
    "position_effects": "[Position]",
}

offset = 0
for v, count in zip(effect_vars, block_counts):
    center_idx = offset + (count // 2)
    assert 0 <= center_idx < len(filtered_positions)
    y = float(filtered_positions[len(filtered_positions) - 1 - center_idx])
    label = GROUP_LABELS[v]
    ax[0].text(
        0.04,
        y,
        label,
        transform=ax[0].get_yaxis_transform(),
        rotation=-90,
        ha="center",
        va="center",
        color="gray",
        fontsize=int(font_size * 0.9),
        fontweight="bold",
        clip_on=False,
    )
    offset += count

# Rotate x-axis labels by 90 degrees and set range
for axis in ax:
    axis.tick_params(axis="x", rotation=-90)
ax[0].set_xlim(-2.5, 2.5)
ax[0].set_title("")

fig = plt.gcf()
out_path = DATA_DIR / "patch_scope_scales_effects_grader_model.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved forest plot to {out_path}")


# %%
