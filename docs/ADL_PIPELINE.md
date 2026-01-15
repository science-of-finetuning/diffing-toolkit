# ADL (Activation Difference Lens) Pipeline

This document describes the sequential structure of the ADL pipeline, assuming preprocessing has been run.

## Overview

**Entry point:** `ActDiffLens.run()` in `src/diffing/methods/activation_difference_lens/method.py:384`

```
For each dataset_entry:
│
├── 1. COMPUTE_DIFFERENCES
│   ├── Load & tokenize dataset
│   ├── Extract activations from BOTH models
│   ├── Compute & save norms
│   └── For each layer: compute diff, save means
│
└── 2. ANALYSIS
    ├── Cache logit lens (optional)
    └── Run auto patch scope (optional)

Then (globally):
├── 3. STEERING (optional)
├── 4. TOKEN RELEVANCE (optional)
└── 5. CAUSAL EFFECT (optional)
```

---

## Step 1: Compute Differences

**Function:** `compute_differences(dataset_entry)` → `method.py:648`

### 1.1 Load & Tokenize Dataset

- **Chat data** (`is_chat=True`): `load_and_tokenize_chat_dataset()` extracts tokens around assistant response start
- **Non-chat data**: `load_and_tokenize_dataset()` extracts first n tokens from each sample

### 1.2 Extract Activations

For each model (base and finetuned):
- **Chat path:** `extract_selected_positions_activations()` - sparse extraction at specified positions
- **Non-chat path:** `extract_first_n_tokens_activations()` - dense extraction of first n tokens

Models are cleared from GPU after extraction (`clear_base_model()`, `clear_finetuned_model()`).

### 1.3 Compute & Save Norms

`_compute_and_save_norms()` → saves `model_norms_{dataset_id}.pt`

### 1.4 Compute Differences & Save Means

For each layer:
```python
diff = ft_acts[layer] - base_acts[layer]
mean_diff = diff.mean(dim=0)  # [num_positions, hidden_dim]
```

Saved via `_save_means_for_layer()`:
- `mean_pos_{pos}.pt` - the difference vector (main output)
- `base_mean_pos_{pos}.pt` - base model mean
- `ft_mean_pos_{pos}.pt` - finetuned model mean
- `.meta` files with metadata

---

## Step 2: Analysis

**Function:** `analysis(ctx)` → `method.py:802`

### 2.1 Cache Logit Lens (Optional)

If `cfg.logit_lens.cache == True`:
- Loads each mean vector
- Runs `logit_lens(mean, model)` → probability distributions
- Saves top-k tokens to `logit_lens_pos_{pos}.pt`

### 2.2 Auto Patch Scope (Optional)

If `cfg.auto_patch_scope.enabled == True`:
- Tests multiple patchscope scales (0.5→200)
- Uses grader LLM to select best scale and coherent token subset
- Saves to `auto_patch_scope_pos_{pos}.pt`

---

## Step 3: Steering

**Function:** `run_steering(method)` → `steering.py:482`

### Process

1. **Threshold search:** Binary search for highest coherent steering strength
2. **Generation:** Generate steered vs unsteered samples at the found threshold

### Storage

```
results_dir/
  layer_{abs_layer}/
    {dataset_short_name}/
      steering/
        position_{pos}_{grader_model}/
          threshold.json       # avg + per-prompt thresholds
          generations.jsonl    # steered vs unsteered samples
```

**generations.jsonl format:**
```json
{
  "prompt": "...",
  "strength": 42.5,
  "layer": 16,
  "position": 3,
  "steered_samples": ["...", "..."],
  "unsteered_samples": ["...", "..."]
}
```

---

## Results Directory Structure

```
results_dir/
  layer_{idx}/
    {dataset_id}/
      mean_pos_{pos}.pt              # Main difference vector
      mean_pos_{pos}.meta
      base_mean_pos_{pos}.pt
      ft_mean_pos_{pos}.pt
      logit_lens_pos_{pos}.pt        # If cached
      auto_patch_scope_pos_{pos}.pt  # If run
      steering/
        position_{pos}_{grader}/
          threshold.json
          generations.jsonl
  model_norms_{dataset}.pt
```

---

## Agent Pipeline

The ADL agent is an LLM that investigates the precomputed results to describe what the finetuning did.

### Entry Point

`EvaluationPipeline.run()` → `run_agent()` → `agent.run()`

Located in `src/diffing/pipeline/evaluation_pipeline.py`

### Pipeline

```
EvaluationPipeline.run()
│
├── diffing_method.get_agent() → ADLAgent
│
└── run_agent(agent, model_interaction_budget, ...)
    │
    └── agent.run(diffing_method, budget)
        │
        ├── 1. BUILD FIRST USER MESSAGE
        │   └── get_overview() → JSON with:
        │       • logit_lens per position
        │       • patchscope per position
        │       • steering examples (steered vs unsteered)
        │
        ├── 2. LLM LOOP (until FINAL or budget exhausted)
        │   │
        │   ├── AgentLLM.chat(messages)
        │   │
        │   ├── Parse response:
        │   │   • FINAL(description: "...") → done
        │   │   • CALL(tool_name: {args}) → execute tool
        │   │
        │   └── Available tools:
        │       • get_logitlens_details   → reads cached .pt
        │       • get_patchscope_details  → reads cached .pt
        │       • get_steering_samples    → reads generations.jsonl
        │       • generate_steered        → LIVE generation (costs budget)
        │
        └── 3. RETURN description + stats

Then: grade_and_save_async(description) → scores the hypothesis
```

### Agent Tools

| Tool | Source | Costs Budget |
|------|--------|--------------|
| `get_logitlens_details` | `logit_lens_pos_{pos}.pt` | No |
| `get_patchscope_details` | `auto_patch_scope_pos_{pos}.pt` | No |
| `get_steering_samples` | `steering/.../generations.jsonl` | No |
| `generate_steered` | Live model inference | Yes |

### Agent Output

Saved to `results_dir/agent/{name}_mi{budget}_{run}/`:
- `description.txt` - the final hypothesis
- `messages.json` - full conversation history
- `stats.json` - token usage, calls made

---

## Key Files

| File | Purpose |
|------|---------|
| `method.py` | Main orchestration: `run()`, `compute_differences()`, `analysis()` |
| `steering.py` | Steering threshold search and generation |
| `agents.py` | `ADLAgent` and `ADLBlackboxAgent` definitions |
| `agent_tools.py` | Tool implementations: `get_overview`, `get_*_details`, etc. |
| `base_agent.py` | `BaseAgent.run()` - the LLM loop |
| `evaluation_pipeline.py` | `EvaluationPipeline.run_agent()` - entry point |
