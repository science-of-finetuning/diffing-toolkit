# %%
from hibayes.analysis_state import AnalysisState
import matplotlib.pyplot as plt
from typing import Tuple
import arviz as az
import arviz.labels as azl
from pathlib import Path
import scienceplots as _scienceplots  # type: ignore[import-not-found]

plt.style.use("science")

arg_size = ""
MAP = {
    "ADL_effects": "",
    "interactions_effects": "",
    "organism_type_effects": "",
    "model_effects": "",
    "[qwen3_1_7B]": f"{{{arg_size} Qwen3 1.7B}}",
    "[qwen3_32B]": f"{{{arg_size} Qwen3 32B}}",
    "[qwen25_7B_Instruct]": f"{{{arg_size} Qwen2.5 7B}}",
    "[gemma2_9B_it]": f"{{{arg_size} Gemma2 9B}}",
    "[gemma3_1B]": f"{{{arg_size} Gemma3 1B}}",
    "[llama31_8B_Instruct]": f"{{{arg_size} Llama3.1 8B}}",
    "[llama32_1B_Instruct]": f"{{{arg_size} Llama3.2 1B}}",
    "[qwen25_VL_3B_Instruct]": f"{{{arg_size} Qwen2.5 VL 3B}}",
    "[ADL]": f"{{{arg_size} ADL}}",
    "[Baseline]": f"{{{arg_size} Blackbox}}",
    "llm_effects": "",
    "[gpt-5]": f"{{{arg_size} GPT5}}",
    "[gemini-2.5-pro]": f"{{{arg_size} {{\\Large Gemini2.5 Pro}}",
    "[0]": f"{{{arg_size} 0}}",
    "[5]": f"{{{arg_size} 5}}",
    "[50]": f"{{{arg_size} 50}}",
    "grader_model_id_effects": "",
    "[anthropic/claude-haiku-4.5]": f"{{{arg_size} Claude Haiku 4.5}}",
    "[google/gemini-2.5-flash]": f"{{{arg_size} Gemini2.5 Flash}}",
    "[openai/gpt-5-mini]": f"{{{arg_size} GPT5 Mini}}",
    "llm_": "",
    "[0, 0]": f"{{{arg_size} Gemini2.5 Pro - Claude Haiku 4.5}}",
    "[0, 1]": f"{{{arg_size} Gemini2.5 Pro - Gemini2.5 Flash}}",
    "[0, 2]": f"{{{arg_size} Gemini2.5 Pro - GPT5 Mini}}",
    "[1, 0]": f"{{{arg_size} GPT5 - Claude Haiku 4.5}}",
    "[1, 1]": f"{{{arg_size} GPT5 - Gemini2.5 Flash}}",
    "[1, 2]": f"{{{arg_size} GPT5 - GPT5 Mini}}",
}
labeller = azl.MapLabeller(var_name_map=MAP)


state = AnalysisState.load(Path("narrow_ft_experiments/hibayes/agent_grader_interactions/data"))
state.models
# %%
best_model = state.get_best_model()
# best_model = state.models[1]
best_model.model_name

# %%
# Set global font size
plt.rcParams.update({"font.size": 122})
ax = az.plot_forest(
    best_model.inference_data,
    var_names=["grader_model_id_effects", "llm_effects", "llm_grader_model_id_effects"],
    figsize=(5, 7),
    transform=None,
    combined=True,
    hdi_prob=0.95,
    labeller=labeller,
    show=False,
    markersize=6,
    linewidth=2,
)

ax[0].axvline(
    x=0,
    color="red",
    linestyle="--",
)
font_size = 22
# Set font sizes for axis labels and title
ax[0].tick_params(axis="both", which="major", labelsize=font_size)
ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=font_size)
ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=int(font_size * 0.2))


def fix_label(label):
    for k, v in MAP.items():
        label = label.replace(k, v)
    return label


# Collect tick positions and texts, keep only labeled ticks for grouping
all_tick_positions = list(ax[0].get_yticks())
all_tick_texts = [t.get_text() for t in ax[0].get_yticklabels()]
filtered_positions: list[float] = []
filtered_texts: list[str] = []
labeled_indices: list[int] = []
for pos, txt in zip(all_tick_positions, all_tick_texts):
    if txt:
        filtered_positions.append(pos)
        filtered_texts.append(txt)
        labeled_indices.append(
            len(filtered_positions) - 1
        )  # position index in filtered arrays
current_labels = [fix_label(txt) for txt in filtered_texts]

ax[0].set_yticklabels(current_labels, fontsize=int(font_size * 0.8))

# Determine x-position just left of tick labels (axes coords)
fig = plt.gcf()
fig.canvas.draw()  # needed so that text bounding boxes are available
tick_texts = list(ax[0].get_yticklabels())
renderer = fig.canvas.get_renderer()
label_bboxes = [
    t.get_window_extent(renderer=renderer) for t in tick_texts if t.get_text()
]
axes_bbox = ax[0].get_window_extent(renderer=renderer)
axes_left_px = axes_bbox.x0
axes_width_px = axes_bbox.width if axes_bbox.width != 0 else 1.0
min_label_left_px = min((bb.x0 for bb in label_bboxes), default=axes_left_px)
padding_px = 14.0
x_axes_left = (min_label_left_px - padding_px - axes_left_px) / axes_width_px

GROUP_Y_INDEX: dict[str, int] = {
    "[Agent]": 7,
    "[Grader]": 9,
    "[Interactions]": 3,
}
GROUP_Y_OFFSET: dict[str, float] = {
    "[Agent]": -0.5,
    "[Grader]": 0,
    "[Interactions]": -0.5,
}

for grp_name, idx_in_filtered in GROUP_Y_INDEX.items():
    assert (
        0 <= idx_in_filtered < len(filtered_positions)
    ), f"Index {idx_in_filtered} out of range for group {grp_name}; labeled ticks: {len(filtered_positions)}"
    y = float(filtered_positions[idx_in_filtered])
    ax[0].text(
        0.04,
        y + GROUP_Y_OFFSET[grp_name],
        grp_name,
        transform=ax[0].get_yaxis_transform(),
        rotation=-90,
        ha="center",
        va="center",
        color="gray",
        fontsize=int(font_size * 0.9),
        fontweight="bold",
        clip_on=False,
    )

# Rotate x-axis labels by 90 degrees
for axis in ax:
    axis.tick_params(axis="x", rotation=-90)

# Set x-axis range
ax[0].set_xlim(-2.5, 2.5)

ax[0].set_title("")

fig = plt.gcf()

fig.show()
fig.savefig("narrow_ft_experiments/hibayes/agent_grader_interactions/interactions_grader_model_id.pdf", bbox_inches="tight")
# %%
