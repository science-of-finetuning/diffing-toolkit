# Merge Review: `clement-logit-diff` → `three_stage_refactor`

No conflict markers found anywhere. Below are all issues grouped by severity.

---

## CRITICAL: Code that will break at runtime

### ~~1. `logit_diff_experiments/run_n_samples_experiments.py` — completely broken~~ FIXED

~~This file was never updated for the rename. It will fail immediately.~~ Fixed in commit 7b44913. Also updated `token_topic_clustering_NMF` toggle → `token_ordering.method` and agent prefix `LogitDiff` → `DiffMining`.

### ~~2. `src/diffing/methods/activation_difference_lens/agents.py` — merge artifact duplications~~ FIXED

~~Three blocks of code were duplicated.~~ Fixed in commit e01dff0.

### ~~3. `src/diffing/pipeline/evaluation_pipeline.py:119-151` — duplicated block~~ FIXED

~~The dataset mapping save+log block appeared twice consecutively.~~ Fixed in commit e01dff0.

---

## ~~IMPORTANT: Variant naming inconsistency~~ FIXED

~~`auditing_agents_secret_loyalty.yaml` uses `sdftags` / `sdftags_kto` while **all other** auditing configs use `synth_docs_tags`.~~

Standardized all 5 organism configs to `sdftags` (matching `secret_loyalty`). Fixed in commit de65683.

---

## ~~STALE REFERENCES: Documentation & comments~~ FIXED

All stale `logit_diff_topk_occurring` / `LogitDiff TopK` references updated to `diff_mining` / `DiffMining`.

- ~~Experiment scripts~~ fixed in commit 7b44913.
- ~~Documentation files (`CLAUDE.md`, `README.md`, `logit_diff_experiments/CLAUDE.md`) + source code comments (`agents.py`, `diffing_method.py`, `preprocessing.py`)~~ fixed alongside variant naming.

---

## OUT OF SCOPE: potential `method_params` in other methods

Agent 4 noticed `method_params` nesting still exists in `configs/diffing/method/activation_analysis.yaml`, `configs/diffing/method/kl.yaml`, and their method `.py` files. Worth checking if those were in scope for the config restructuring.

---

## CLEAN (no issues)

- `src/diffing/methods/diff_mining/{agents,normalization,orthogonal_nmf,plots}.py`
- `src/diffing/utils/agents/llm.py`, `src/diffing/utils/configs.py`
- `main.py`, `pyproject.toml`
- `configs/config.yaml`, `configs/model/qwen3_14B.yaml`
- All organism configs except secret_loyalty naming
- No deleted `_transcripts_only` references found anywhere
- `scripts/{aggregate_global_token_stats,analyze_bimodal_kde,retroactive_token_KDE,summarize_adl_results}.py`
- `launch_all_experiments.sh`, `launch_array_experiments.sh`, `view_plots.py`
- `.gitignore`, `logit_diff_experiments/.gitignore`
- `log_prob_analysis.py` (correctly updated)
- `src/diffing/methods/activation_oracle/method.py`
- `src/diffing/methods/activation_difference_lens/{agent_tools,method}.py`
