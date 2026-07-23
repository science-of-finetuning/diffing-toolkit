# Activation Difference Lens (ADL)

Analyzes finetuning effects by extracting activation differences between base and finetuned models, then interpreting them via logit lens, patchscope, steering, causal effect analysis, and token relevance grading.

## File Map

| File | Purpose |
|------|---------|
| `method.py` | `ActDiffLens` main class — orchestrates pipeline: dataset loading, activation extraction, difference computation, analysis |
| `steering.py` | Steering vector generation: binary search for max coherent strength, batch generation with nnsight |
| `token_relevance.py` | LLM-graded token relevance pipeline (permutation-robust grading, supports baselines) |
| `auto_patch_scope.py` | Multi-scale patchscope search: sweeps scales [0.5–200], grades with LLM, picks best scale |
| `causal_effect.py` | Intervention-based causality: measures NLL impact of projecting out/replacing activation diff directions |
| `agent_tools.py` | Agent tool implementations: `get_overview()`, `get_logitlens_details()`, `get_patchscope_details()`, `get_steering_samples()`, `generate_steered()` |
| `agents.py` | `ADLAgent` (has analysis tools) and `ADLBlackboxAgent` (generations-only baseline) |
| `prompts.py` | System prompts for both agents (evidence hygiene, cross-signal agreement emphasis) |
| `util.py` | Path helpers: `layer_dir()`, `load_position_mean_vector()`, `is_layer_complete()` |
| `ui.py` | Streamlit dashboard: Lens, Steering, Steered Answers, and Online Projections tabs |

## Pipeline (`run()`)

```
For each dataset:
  1. compute_differences()
     ├── Tokenize dataset (chat or regular format)
     ├── Extract activations from base & finetuned models (nnsight tracing)
     ├── Compute diff = ft_acts - base_acts per layer/position
     └── Save mean_diff, base_mean, ft_mean vectors + norms
  2. analysis()
     ├── Logit lens: project mean vectors onto vocabulary
     └── Auto patchscope: sweep scales, grade tokens, pick best

Then (optional, config-gated):
  3. run_steering()       — binary search threshold, generate steered/unsteered samples
  4. run_token_relevance()— LLM-grade promoted tokens for domain relevance
  5. run_causal_effect()  — measure NLL impact of interventions vs random baselines
```

## Key Data Structures

### Activation Extraction
- Sequences are fixed-length (`n` tokens for regular, position-window for chat)
- Activations extracted via `model.layers_output[layer].save()` in nnsight trace
- Shape: `[num_sequences, num_positions, hidden_dim]` → moved to CPU immediately

### Position Mean Vectors
The core data unit. Stored at `layer_{L}/{dataset}/mean_pos_{P}.pt`:
- `mean_pos_{P}.pt` — mean activation difference `[hidden_dim]`
- `base_mean_pos_{P}.pt` — mean base activation
- `ft_mean_pos_{P}.pt` — mean finetuned activation
- `mean_pos_{P}.meta` — JSON metadata (count, dim, position)

### Logit Lens Cache
`logit_lens_pos_{P}.pt` → tuple of `(top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices)`

### Patchscope Cache
`auto_patch_scope_pos_{P}_{grader}.pt` → dict with `best_scale`, `tokens_at_best_scale`, `selected_tokens`, `token_probs`

### Steering Cache
`steering/position_{P}_{grader}/threshold.json` + `generations.jsonl`

## Steering Details (`steering.py`)

- `generate_steered()` applies additive steering via `model.steer(layer, vector)` inside nnsight generate
- `binary_search_threshold()` uses batched lookahead: evaluates `2^batch_steps - 1` midpoints per round for efficiency
- `find_steering_threshold()` wraps binary search per prompt, averages thresholds across prompts

## Causal Effect Details (`causal_effect.py`)

- `StreamingCEStats` tracks min/max/median NLL without storing full tensors (reservoir sampling, 8192 samples)
- Compares interventions against two baselines: random gaussian vectors and random activation difference vectors
- Reports per-region metrics: all tokens, after-k tokens, exclude-position tokens

## Agent Architecture

`ADLAgent` extends `DiffingMethodAgent` → `BlackboxAgent` → `BaseAgent`:
- Inherits `ask_model` from BlackboxAgent
- Adds 4 method tools: `get_logitlens_details`, `get_patchscope_details`, `get_steering_samples`, `generate_steered`
- First message built by `get_overview()`: auto-discovers datasets, loads all cached results, anonymizes dataset names

`ADLBlackboxAgent` extends `BlackboxAgent`:
- No method tools, only `ask_model`
- Provides unsteered generations as baseline context

## Results Directory Layout

```
{base_dir}/{model}/{organism}/activation_difference_lens/
└── layer_{L}/
    └── {dataset_dir_name}/
        ├── mean_pos_{P}.pt, base_mean_pos_{P}.pt, ft_mean_pos_{P}.pt
        ├── mean_pos_{P}.meta
        ├── logit_lens_pos_{P}.pt, base_logit_lens_pos_{P}.pt, ft_logit_lens_pos_{P}.pt
        ├── auto_patch_scope_pos_{P}_{grader}.pt
        ├── model_norms_{dataset_id}.pt
        ├── steering/position_{P}_{grader}/
        │   ├── threshold.json
        │   └── generations.jsonl
        ├── token_relevance/position_{P}/{variant}/
        │   └── relevance_{source}.json
        └── causal_effect/position_{P}/
            └── metrics.json
```

## Key Patterns

- **Lazy model loading**: `self.base_model`, `self.finetuned_model` are properties that load on first access
- **Overwrite-aware**: every save checks `self.overwrite` flag; skips if results exist
- **Chat vs regular datasets**: chat uses position windows around assistant turn start; regular uses first-n tokens
- **`load_position_mean_vector()`** normalizes then rescales to match finetuned model's activation norm at that layer
