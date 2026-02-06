# Code Duplication Report

Generated 2025-02-05 via `jscpd` with `--min-lines 3 --min-tokens 50` on `src/`.

**Summary**: 54 clones detected, 13 actionable refactorings identified, ~530 lines of real duplication.

---

## High Priority

### 1. `DictionaryBasedDiffingMethod` base class (~70 lines)

`crosscoder/method.py` and `sae_difference/method.py` were forked from each other and never consolidated. Three distinct duplicated blocks:

| Clone | crosscoder/method.py | sae_difference/method.py | Lines |
|-------|---------------------|-------------------------|-------|
| `__init__` setup (layers, results_dir, cache) | 70-83 | 67-80 | 13 |
| Latent activations pipeline (`collect_dictionary_activations_from_config` + `collect_activating_examples` + `update_latent_df_with_stats`) | 164-202 | 165-203 | 38 |
| `has_results()` directory scanning | 227-245 | 356-374 | 18 |

The `has_results()` pattern is also duplicated in `kl/method.py:596-614` (same 18-line structure, different subdirectory name).

**Refactoring**: Create `DictionaryBasedDiffingMethod(DiffingMethod)` with:
- Shared `__init__` (layers resolution, results_dir creation)
- `_run_latent_activations_analysis(dictionary_name, layer_idx, model_results_dir)` method
- Template-method `has_results()` with abstract `_get_method_subdir()` and `_check_has_results()` hooks

---

### 2. Bracket-matching in `base_agent.py` (~33 lines)

`_extract_final_description()` (lines 46-79) and `_extract_tool_call()` (lines 124-157) contain identical parenthesis-matching logic with quote/escape tracking.

**Refactoring**: Extract `_find_matching_paren(text: str, start_pos: int) -> int` helper that handles nested parens, quoted strings, and escape sequences. Both functions call it.

---

### 3. Steered vs unsteered generation in `steering.py` (~39 lines)

`_generate_steered_with_intervention()` (lines 83-110) and `generate_unsteered_text()` (lines 406-433) share identical prompt formatting, tokenization, generation loop, and output extraction. The output decoding block (lines 135-147 vs 442-454) is also duplicated.

Only difference: steered version applies an intervention hook during generation.

**Refactoring**: Extract `_prepare_and_generate(model, tokenizer, prompts, ..., steering_hook=None) -> list[str]` that handles the common pipeline, with an optional steering hook callable.

---

### 4. Probe training data in `verbalizer.py` (~38 lines)

`activation_oracle/verbalizer.py` has three near-identical ~19-line blocks:
- Lines 215-232: `dp_kind="tokens"` — per-token positions
- Lines 248-265: `dp_kind="segment"` — contiguous segment
- Lines 274-291: `dp_kind="full_seq"` — full sequence

Each computes position ranges, creates metadata with `dp_kind`, calls `create_training_datapoint()`, and appends results. Only the position logic differs.

**Refactoring**: Extract `_create_probe_training_data(dp_kind, position_ranges_and_repeats, ...)` parameterized by kind and position computation.

---

### 5. Sync/async `hypothesis_grader.py` (~31 lines)

`grade_and_save()` (lines 220-255) and `grade_and_save_async()` (lines 255-288) are identical except for `await`. Both duplicate:
- Cache check and early return (~10 lines)
- Grader construction and call (~5 lines)
- Payload building and JSON save (~15 lines)

**Refactoring**: Extract non-async helpers for cache check and payload save. Make async the canonical grading path, with sync wrapper using `asyncio.run()`.

---

### 6. Helpers in `token_relevance_grader.py` (~50 lines)

Four patterns duplicated across `grade()`, `grade_with_translation()`, `grade_async()`, and `grade_with_translation_async()`:

| Pattern | Lines | Occurrences |
|---------|-------|-------------|
| Permutation building (rotated indices + reordering) | 5 | 3 |
| Async gather runner (`_runner` + `asyncio.gather`) | 12 | 2 |
| Result remapping to original order | 12 | 2 |
| Translation accumulation (first non-UNKNOWN) | 24 | 2 |

**Refactoring**: Extract `_build_permuted_inputs()`, `_run_concurrent_tasks()`, `_map_results_to_original_order()`, and `_accumulate_first_known_translations()` helpers.

---

### 7. Generation logging in `chat_tab.py` (~150+ lines)

`amplification/streamlit_components/chat_tab.py` has massive duplication across four handler methods (`_handle_regenerating_from`, `_handle_regenerating_from_user`, `_handle_continuing_from`, `_handle_user_input`):

| Pattern | Approx. occurrences |
|---------|-------------------|
| `GenerationLog.from_dashboard_generation(...)` call | ~8 |
| Multi-gen vs single-gen branching | 4 |
| `apply_chat_template` prompt building | 4 |

**Refactoring**: Extract `_log_generation(conv, generation_type, prompt_tokens, sampling_params, results, messages)` and `_handle_multi_gen_result(conv_id, conv, result, mode, pending_key, target_index)` helpers.

---

## Medium Priority

### 8. Dataset variant processing in `configs.py` (~19 lines)

Lines 344-363 (chat_dataset) and 375-394 (pretraining_dataset): identical logic for iterating over variant names, constructing dataset identifiers, and creating `DatasetConfig` objects.

**Refactoring**: Extract `_process_dataset_variants(cfg, dataset_attr, variants_list, logger_fn) -> list[DatasetConfig]`.

---

### 9. Overview agent pattern in ADL + DiffMining agents (~22 lines)

`activation_difference_lens/agents.py:91-113` and `diff_mining/agents.py:56-78`: both store `_dataset_mapping`, implement `get_dataset_mapping()`, and build the first user message with identical OVERVIEW JSON formatting.

**Refactoring**: Create `OverviewAgentMixin` with `_dataset_mapping`, `get_dataset_mapping()`, and `_build_overview_message()`. Only worthwhile if more agent types are planned.

---

### 10. Image gallery rendering in `activation_analysis/ui.py` (~42 lines)

Lines 86-127 vs 139-181: two near-identical image gallery blocks (with_outliers vs no_outliers). Same grid layout, format-specific handling (PNG/JPG, SVG, download fallback). Only difference: filter condition and label text.

**Refactoring**: Extract `_render_image_gallery(image_files, title, cols_per_row)` and `_render_image_by_format(image_file)` helpers.

---

### 11. Norms bar plot across methods (~17 lines)

`activation_analysis/ui.py:413-430` and `activation_difference_lens/ui.py:237-254`: identical matplotlib bar chart code (steelblue bars, value labels, grid, tight_layout).

**Refactoring**: Extract `render_norms_bar_plot(norms, labels, title, ...)` to `diffing/utils/visualization.py`.

---

### 12. Match collection in `samples.py` (~19 lines)

`amplification/streamlit_components/samples.py` lines 56-75 vs 120-139: `apply_html_highlighting()` and `get_annotated_segments()` share identical selector filtering, keyword iteration, regex matching, and color extraction. Only the output format differs (HTML vs annotated_text segments).

**Refactoring**: Extract `_collect_matches(text, selectors, alpha) -> list[tuple]`, called by both functions.

---

### 13. Coherence percentage in `coherence_grader.py` (~10 lines)

Lines 191-199 vs 224-234: `grade()` and `grade_async()` both compute the same unknown-filtering and COHERENT percentage.

**Refactoring**: Extract `_compute_coherence_percentage(labels) -> tuple[float, list]`.

---

## Dismissed (False Positives / Acceptable)

The remaining ~40 clones were not actionable:

| Category | Examples | Why dismissed |
|----------|----------|---------------|
| Import blocks | crosscoder + sae_difference dashboard imports (15 lines) | Legitimate — both dashboards need the same utilities |
| Input assertions | token_relevance_grader assertions (8 lines, 3 occurrences) | Method-local contracts, intentional defensive programming |
| Structural similarity | sae_difference/dashboard.py model loading in `get_latent` vs `get_dict_size` (12 lines) | Different methods, similar structure |
| Small utility calls | steering.py + token_relevance.py layer index retrieval (7 lines) | Standard utility usage |
| Tensor validation | diff_mining/logit_extraction.py assertion patterns (13 lines) | Idiomatic shape checking |
| Different semantics | max_activation_dashboard.py preview vs full mode (14 lines) | Same pattern, different data sources |
| Minimal | dashboard_state.py config name mapping (1 effective line) | Not worth extracting |
| Context-dependent | managed_data.py `from_content_dict` vs `from_dict` (9 lines) | Different deserialization semantics |
| Cross-method structural | activation_analysis/ui.py vs sae_difference/dashboard.py image grid (16 lines) | Similar UI pattern, different contexts |
| Tool wrappers | agent_tools.py vs agents.py (7 lines) | Wrappers look alike by design |
| Small vectorized ops | diff_mining/core_analysis.py (5 lines), preprocessing.py (10 lines) | Standard tensor operations |

---

## Methodology

```bash
# Detection command
npx jscpd --pattern "**/*.py" --gitignore --min-lines 3 --min-tokens 50 src/

# Results: 54 clones, 809 duplicated lines (2.5%), 7188 duplicated tokens (3.17%)
# After filtering: 13 actionable, ~530 lines of real duplication
```

Threshold selection: `min-lines 1, min-tokens 10` yields 1844 clones (20% of codebase flagged) — overwhelmed by Python idioms (imports, `with open(...)`, assertions). `min-lines 3, min-tokens 50` is the practical sweet spot.
