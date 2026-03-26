#!/usr/bin/env python3
"""
Combined figure: unweighted token relevance (left) + agent score (right).

This is Figure 1 in the paper which compared relevance and agent score, for
Top-K Diff Mining vs ADL-LogitLens, as a function of mix ratio, on FineWeb text.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
with open("relevance/token_relevance_results.json", "r") as f:
    relevance_results = json.load(f)["fineweb-1m-sample"]

with open("agent_scores/agent_results.json", "r") as f:
    agent_results = json.load(f)

# ── Shared config ────────────────────────────────────────────────────────────
mix_ratio_values = {
    "default": 0.0,
    "mix1-0p2": 0.2,
    "mix1-0p4": 0.4,
    "mix1-0p5": 0.5,
    "mix1-0p6": 0.6,
    "mix1-0p8": 0.8,
    "mix1-1p0": 1.0,
    "mix1-1p5": 1.5,
    "mix1-2p0": 2.0,
}


def format_ratio_label(x):
    if x == 0:
        return "1:0"
    elif x == int(x):
        return f"1:{int(x)}"
    else:
        return f"1:{x}"


# ── Create figure ────────────────────────────────────────────────────────────
fig, (ax_rel, ax_agent) = plt.subplots(1, 2, figsize=(20, 6))

# ── LEFT: Unweighted token relevance ────────────────────────────────────────
relevance_mix_ratios = [
    "default", "mix1-0p2", "mix1-0p4", "mix1-0p6",
    "mix1-0p8", "mix1-1p0", "mix1-1p5", "mix1-2p0",
]

relevance_curves = [
    ("logit_diff_topk_occurring", "Top-K Diff Mining \u00b1 1 SD", "#2ecc71"),
    ("activation_difference_lens", "ADL-LogitLens \u00b1 1 SD", "#3498db"),
]

for method, label, color in relevance_curves:
    if method not in relevance_results:
        continue
    ratio_data = relevance_results[method]
    x_vals, y_means, y_stds = [], [], []

    for mix_ratio in relevance_mix_ratios:
        if mix_ratio in ratio_data:
            values = [
                run["percentage"] * 100
                for run in ratio_data[mix_ratio]
                if run.get("percentage") is not None
            ]
            if values:
                x_vals.append(mix_ratio_values[mix_ratio])
                y_means.append(np.mean(values))
                y_stds.append(np.std(values))

    if x_vals:
        x_arr, y_arr, std_arr = np.array(x_vals), np.array(y_means), np.array(y_stds)
        sort_idx = np.argsort(x_arr)
        x_arr, y_arr, std_arr = x_arr[sort_idx], y_arr[sort_idx], std_arr[sort_idx]

        ax_rel.fill_between(x_arr, y_arr - std_arr, y_arr + std_arr, alpha=0.25, color=color)
        ax_rel.plot(x_arr, y_arr, marker="o", markersize=8, linewidth=2, label=label, color=color)

all_x_rel = sorted({mix_ratio_values[r] for method in relevance_results for r in relevance_mix_ratios if r in relevance_results[method]})
ax_rel.set_xticks(all_x_rel)
ax_rel.set_xticklabels([format_ratio_label(x) for x in all_x_rel], fontsize=14)
ax_rel.tick_params(axis="y", labelsize=14)
ax_rel.set_xlabel("Mix Ratio", fontsize=14)
ax_rel.set_ylabel("Token Relevance (%)", fontsize=14)
ax_rel.legend(loc="best", fontsize=12)
ax_rel.grid(True, alpha=0.3)
ax_rel.set_ylim(0, 100)

# ── RIGHT: Agent score ──────────────────────────────────────────────────────
agent_mix_ratios = ["default", "mix1-0p5", "mix1-1p0", "mix1-1p5", "mix1-2p0"]

agent_curves = [
    ("blackbox", "mi5", "Blackbox (mi=5) \u00b1 1 SD", "#95a5a6", "--"),
    ("activation_difference_lens", "mi5", "ADL-LogitLens (mi=5) \u00b1 1 SD", "#3498db", "-"),
    ("logit_diff_topk_occurring", "mi5", "Top-K Diff Mining (mi=5) \u00b1 1 SD", "#2ecc71", "-"),
]

for method, mi_key, label, color, linestyle in agent_curves:
    if method not in agent_results:
        continue
    x_vals, y_means, y_stds = [], [], []

    for mix_ratio in agent_mix_ratios:
        if mix_ratio in agent_results[method]:
            mi_data = agent_results[method][mix_ratio]
            if mi_key in mi_data:
                scores = mi_data[mi_key]
                if scores:
                    x_vals.append(mix_ratio_values[mix_ratio])
                    y_means.append(np.mean(scores))
                    y_stds.append(np.std(scores))

    if x_vals:
        x_arr, y_arr, std_arr = np.array(x_vals), np.array(y_means), np.array(y_stds)
        sort_idx = np.argsort(x_arr)
        x_arr, y_arr, std_arr = x_arr[sort_idx], y_arr[sort_idx], std_arr[sort_idx]

        ax_agent.fill_between(x_arr, y_arr - std_arr, y_arr + std_arr, alpha=0.2, color=color)
        ax_agent.plot(x_arr, y_arr, marker="o", markersize=8, linewidth=2, linestyle=linestyle, label=label, color=color)

all_x_agent = sorted({mix_ratio_values[r] for method in agent_results for r in agent_mix_ratios if r in agent_results[method]})
ax_agent.set_xticks(all_x_agent)
ax_agent.set_xticklabels([format_ratio_label(x) for x in all_x_agent], fontsize=14)
ax_agent.tick_params(axis="y", labelsize=14)
ax_agent.set_xlabel("Mix Ratio", fontsize=14)
ax_agent.set_ylabel("Agent Score (1-5)", fontsize=14)
ax_agent.legend(loc="best", fontsize=10)
ax_agent.grid(True, alpha=0.3)
ax_agent.set_ylim(0, 5)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
#plt.savefig("combined_relevance_agent.png", dpi=150, bbox_inches="tight")
plt.savefig("combined_relevance_agent.png", dpi=400, bbox_inches="tight")
print("Saved: combined_relevance_agent.png")
plt.close()
