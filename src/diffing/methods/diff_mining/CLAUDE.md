# Diff Mining

Identifies tokens whose predicted probability changes most between base and finetuned models. Computes logit diffs across the vocabulary, counts top-K occurrences, and clusters co-occurring tokens via NMF topic decomposition.

## File Map

| File | Purpose |
|------|---------|
| `diff_mining.py` | `DiffMiningMethod` main class — orchestrates preprocessing (logit extraction) and analysis (ordering computation) |
| `core_analysis.py` | Vectorized batch processing: extracts top-K tokens, counts occurrences, collects co-occurrence data |
| `logit_extraction.py` | Three extraction strategies: `DirectLogits`, `LogitLens`, `PatchscopeLens` |
| `preprocessing.py` | Dataset tokenization, tensor preparation, position slicing for chat datasets |
| `token_ordering.py` | Ordering abstractions: `TopKOccurring`, `FractionPositiveDiff`, `NMF` — each defines how to rank tokens |
| `normalization.py` | BPE decoding (Ġ→space, Ċ→newline), punctuation filtering, token consolidation |
| `orthogonal_nmf.py` | Orthogonal NMF fitting via torchnmf (BetaMu trainer) |
| `agent_tools.py` | `get_overview()` — builds agent first message from saved orderings |
| `agents.py` | `DiffMiningAgent` — extends DiffingMethodAgent, no extra tools (only `ask_model`) |
| `plots.py` | Occurrence bar charts, positional KDE plots |
| `ordering_plots.py` | Scatter plots (ordering_value vs avg_logit_diff) and bar charts per ordering |
| `ui.py` | Streamlit dashboard: browse runs, view orderings, compare across datasets |

## Pipeline

### Preprocessing (`preprocess()`)

```
Phase 0: Data preparation
  ├── Load datasets from HuggingFace (text or chat format)
  ├── Tokenize, pad to max_tokens_per_sample
  └── Save input_ids + attention_mask tensors

Phase 1: Base model inference
  └── Extract logits via selected method → save to disk (or memory)

Phase 2: Finetuned model inference
  └── Extract logits → compute diffs (ft_logits - base_logits)

Phase 3: Save logit diffs per dataset
```

### Analysis (`run()`)

```
Phase 4: Core analysis
  ├── Batch loop over logit diffs (vectorized)
  │   ├── torch.topk() for top-K positive & negative indices/values
  │   ├── Count token occurrences across all positions
  │   └── Collect per-ordering data (occurrences, fractions, co-occurrence matrix)
  ├── Run enabled ordering types → produce ranked token lists
  ├── Write orderings to JSON
  └── (optional) Generate plots, run token relevance grading
```

## Logit Extraction Methods (`logit_extraction.py`)

| Method | How | When to use |
|--------|-----|-------------|
| `DirectLogits` | Traces forward pass, saves `model.logits` | Default, full output distribution |
| `LogitLens` | Extracts intermediate `layers_output[layer]`, projects via `lm_head` | Layer-specific analysis |
| `PatchscopeLens` | Caches KV prefix, patches latent into target layer, single-token decode | Expensive but semantic interpretation at intermediate layers |

## Token Ordering Types (`token_ordering.py`)

Each ordering type collects data during the batch loop, then produces one or more `Ordering` objects:

| Type | Metric | Output |
|------|--------|--------|
| `TopKOccurring` | `count_in_topK_positive / total_positions × 100` | 1 global ordering |
| `FractionPositiveDiff` | `count_positive_diff / total_positions` | 1 global ordering |
| `NMF` | NMF weight `W[token, topic]` from co-occurrence decomposition | 1 ordering per topic |

### NMF Details
- Builds sparse co-occurrence matrix from top-K token-position pairs during batch loop
- Two modes: `binary` (counts) or `logit_diff_magnitude` (weighted by diff value)
- Fits NMF with configurable beta divergence (1=KL, 2=Frobenius)
- Orthogonal regularization optional (via `orthogonal_weight`)
- Each topic becomes a separate ordering ranked by the W matrix column

## In-Memory Mode (`in_memory=true`)

When running `pipeline.mode=full` (preprocess + analysis in one shot):
- Base logits kept in memory after Phase 1
- Finetuned inference computes diffs inline (`infer_finetuned_and_compute_diffs_in_memory()`)
- Skips disk I/O for Phase 3 entirely
- Trade-off: faster but higher memory usage (both models + diffs in RAM)

## Key Data Structures

### `SharedTokenStats` (per dataset, computed once)
```python
vocab_size: int
total_positions: int
sum_logit_diff: Tensor[vocab_size]    # sum of diffs across all positions
count_positive: Tensor[vocab_size]    # positions where diff > 0
topk_pos_counts: Tensor[vocab_size]   # times token appeared in top-K positive
topk_neg_counts: Tensor[vocab_size]   # times token appeared in top-K negative
```

### `OrderingBatchCache` (passed to ordering types each batch)
```python
top_k_pos_indices: Tensor[batch, seq, top_k]
top_k_pos_values: Tensor[batch, seq, top_k]
top_k_neg_indices: Tensor[batch, seq, top_k]
top_k_neg_values: Tensor[batch, seq, top_k]
attention_mask: Tensor[batch, seq]
```

### `TokenEntry` / `Ordering` (output)
```python
TokenEntry: { token_id, token_str, ordering_value, avg_logit_diff, count_positive, count_negative, extra }
Ordering: { ordering_id, display_label, tokens: List[TokenEntry], metadata }
```

## Agent

`DiffMiningAgent` extends `DiffingMethodAgent` but defines **no additional method tools** — the agent only has `ask_model` plus the overview context.

`get_overview()` in `agent_tools.py`:
- Loads orderings from disk based on config (`extraction_method`, `ordering_type`)
- Allocates token budget across ordering groups via `_allocate_token_budgets()`
- Anonymizes dataset names (ds1, ds2, ...)
- Returns JSON payload with per-dataset token groups + metadata

## Results Directory Layout

```
{base_dir}/{model}/{organism_variant}/
  diff_mining_{samples}samples_{tokens}tokens_{topk}topk[_vocab{N}][_logit_extraction_{method}[_layer_{rel}]]/
    seed{S}_top{K}/
      ├── run_metadata.json
      ├── {dataset}_global_token_stats.json
      ├── {dataset}_sequence_likelihood_ratios.json  (optional)
      ├── top_k_occurring/{dataset}/
      │   ├── orderings.json, global.json
      │   ├── global_scatter.png, global_bar.png
      │   └── global_eval.json  (optional, token relevance)
      ├── fraction_positive_diff/{dataset}/...
      └── nmf_topics{N}_topn{N}_mode_{M}_beta{B}_orth{O}_w{W}/{dataset}/
          ├── orderings.json
          ├── topic_{i}.json, topic_{i}_scatter.png, topic_{i}_bar.png
          └── topic_{i}_eval.json  (optional)
```

## Key Patterns

- **Vectorized everything**: `torch.topk()`, `vectorized_bincount_masked()`, no Python loops over tokens
- **Deterministic run naming**: `seed{S}_top{K}` — same config = same directory, no timestamps
- **Overwrite-aware**: skips phases if results exist and `overwrite=false`
- **Token normalization**: BPE-decoded, lowercased, punctuation-filtered before display/grading
- **GPU memory discipline**: `del` + `gc.collect()` + `torch.cuda.empty_cache()` after each batch
