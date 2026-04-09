---
name: recursive-diff-mining
description: >-
  Iterative diff_mining on diffing-toolkit parameterized by model, organism (and variant),
  token_ordering method, and max_recursions (default 10). Fineweb pass, then up to max_recursions
  cycles of write 100-line JSONL and re-run diff mining. Use for recursive diff mining or
  custom reference datasets after Fineweb.
---

# Recursive diff mining (diffing-toolkit)

Treat every study as defined by **four inputs**:

| # | Argument | What to pass |
|---|----------|----------------|
| **1** | **Model** | Hydra **`model=<name>`** — config group under `configs/model/` (e.g. `qwen3_14B`, `gemma3_1B`). |
| **2** | **Organism** | Hydra **`organism=<name>`** — under `configs/organism/` (e.g. `auditing_agents_animal_welfare`). Often also **`organism_variant=<variant>`** when the organism defines multiple finetuned adapters (e.g. `transcripts_kto`). Use **`organism_variant=default`** when that variant exists and is intended. |
| **3** | **Ordering method** | Exactly one of **`top_k_occurring`**, **`fraction_positive_diff`**, **`nmf`** — `diffing.method.token_ordering.method=[<one>]`. |
| **4** | **`max_recursions`** | **Integer ≥ 1**, default **`10`**. How many times to repeat **Step 3** after Step 1: each recursion is **(3A)** write a new 100-line JSONL → **(3B)** run diff mining → **(2)** read outputs → decide next theme. **Does not** count the initial Fineweb pass (Step 1). Stop when you reach `max_recursions` or the user ends the study early. |

Resolve **model**, **organism**, **variant**, **`max_recursions`** (default 10 if unspecified), and **ordering method** with the user (or task context) **before** running commands.

**Pipeline:** bundled behavior is **`pipeline.mode=no_evaluation`** (preprocess + analysis: orderings, plots; **no** evaluation agent, **no** token-relevance LLM if disabled in config).

**Orchestration vs sub-agents:** The **lead agent** following this skill should **delegate** where work is long or isolated: e.g. **run a sub-agent** to execute a single `uv run` / shell block if your environment supports it; **run a sub-agent** to author the 100-line JSONL from a written spec; **run a sub-agent** to consolidate token lists into a short brief. The steps below name these explicitly where it helps.

---

## Convenience scripts (fixed model / organism / variant)

`scripts/run_diff_mining_recursive_initial_fineweb.sh` and `scripts/run_diff_mining_recursive_custom_jsonl.sh` bake in **`organism=auditing_agents_animal_welfare`**, **`model=qwen3_14B`**, **`organism_variant=transcripts_kto`**, **`infrastructure=runpod`**, and sample/batch sizes. Each script runs **one** `uv` invocation; it does **not** loop. **`max_recursions`** is enforced by **you / the agent** by repeating Step 3 up to that many times.

They only take the **ordering method** (+ JSONL path on the custom script).

Use them **only** when those defaults match the task. Otherwise use the **parameterized `uv run`** templates below.

---

## Step 1 — Initial pass (Fineweb)

**Fineweb dataset:** `science-of-finetuning/fineweb-1m-sample`, `streaming: true`, `text_column: text`.

**Run a sub-agent (or yourself)** to execute the command block below from `diffing-toolkit/` if you want isolation from the rest of the workflow; otherwise run it directly.

**Parameterized command** (substitute `MODEL`, `ORGANISM`, `VARIANT`, `ORDERING`):

```bash
cd /workspace/diffing-toolkit

uv run python main.py diffing/method=diff_mining \
  infrastructure=runpod \
  organism="${ORGANISM}" \
  model="${MODEL}" \
  organism_variant="${VARIANT}" \
  pipeline.mode=no_evaluation \
  diffing.method.token_relevance.enabled=false \
  diffing.method.max_samples=1000 \
  diffing.method.max_tokens_per_sample=30 \
  diffing.method.batch_size=32 \
  "diffing.method.token_ordering.method=[${ORDERING}]" \
  'diffing.method.datasets=[{id:science-of-finetuning/fineweb-1m-sample,is_chat:false,text_column:text,streaming:true}]'
```

**`ORDERING`** must be exactly: `top_k_occurring`, `fraction_positive_diff`, or `nmf`.

**Script shortcut** (ordering only; fixed auditing/qwen3/transcripts_kto):

```bash
./scripts/run_diff_mining_recursive_initial_fineweb.sh "${ORDERING}"
```

---

## Step 2 — Find outputs (precise paths, one ordering method)

**Read the artifacts** for this diff mining run (yourself or **a sub-agent** tasked only with loading and quoting the relevant JSON). Build the **run directory** from your Hydra config (nothing here assumes a specific organism; substitute your **`model`**, **`organism`**, **`organism_variant`**, **`diffing.method`**, **`seed`**, **`top_k`**, and **dataset id** — Fineweb vs each custom JSONL each yields a different **`<DATASET_NAME>`** path segment).

```text
<RESULTS_BASE>/<MODEL_NAME>/<ORGANISM_PATH_NAME>/<METHOD_DIR>/seed<SEED>_top<TOP_K>/
```

- **`RESULTS_BASE`**: `diffing.results_base_dir` (e.g. with **`infrastructure=runpod`**, often **`/workspace/model-organisms/diffing_results`** — see `configs/infrastructure/runpod.yaml`).
- **`MODEL_NAME`**: `cfg.model.name` (e.g. `qwen3_14B`).
- **`ORGANISM_PATH_NAME`**: `organism.name` + `_` + `organism_variant` when `organism_variant` is not `default`; otherwise just `organism.name`.
- **`METHOD_DIR`**: `diff_mining_{max_samples}samples_{max_tokens_per_sample}tokens_{top_k}topk[...]_logit_extraction_<...>` (matches `DiffMiningMethod` directory naming).
- **`seed<SEED>_top<TOP_K>`**: from `cfg.seed` and `diffing.method.top_k` (e.g. `seed42_top100`).

**Dataset subfolder name:** Diff mining names each dataset from the config (e.g. a file `hidden_bias.jsonl` + split `train` + column `text` → **`hidden_bias.jsonl_train_text`**). Below this is called **`<DATASET_NAME>`**.

Because you ran with **exactly one** `token_ordering.method`, only **one** ordering-type directory under the run folder is populated for that analysis. Use it as follows.

### `fraction_positive_diff` — global token summary

```text
<RUN_DIR>/fraction_positive_diff/<DATASET_NAME>/global.json
```

### `top_k_occurring` — global token summary

```text
<RUN_DIR>/top_k_occurring/<DATASET_NAME>/global.json
```

Same run also has **`orderings.json`**, bar/scatter plots under that folder for inspection.

### `nmf` — one file per topic

The NMF ordering directory name **depends on NMF settings** (topics, top-n, mode, beta, orthogonality), e.g.  
`nmf_topics3_topn200_mode_logit_diff_magnitude_beta2p0_orthfalse_w1000p0/`.

- **Topic count:** read **`metadata.json`** at the **root of that NMF directory**; field **`num_topics`** (e.g. `3`) tells you how many topic files exist.

```text
<NMF_ORDERING_DIR>/metadata.json   # e.g. "num_topics": 3
```

- **Per-topic token lists:**

```text
<NMF_ORDERING_DIR>/<DATASET_NAME>/topic_0.json
<NMF_ORDERING_DIR>/<DATASET_NAME>/topic_1.json
…
<NMF_ORDERING_DIR>/<DATASET_NAME>/topic_<num_topics - 1>.json
```

**Illustrative full paths** (same run, example organism `auditing_agents_animal_welfare` + variant `transcripts_kto`, model `qwen3_14B`, dataset `hidden_bias.jsonl_train_text` — **your paths will differ** if any of these differ):

- Fraction positive `global.json`:  
  `/workspace/model-organisms/diffing_results/qwen3_14B/auditing_agents_animal_welfare_transcripts_kto/diff_mining_1000samples_30tokens_100topk_logit_extraction_logits/seed42_top100/fraction_positive_diff/hidden_bias.jsonl_train_text/global.json`
- Top-K occurring `global.json`:  
  `/workspace/model-organisms/diffing_results/qwen3_14B/auditing_agents_animal_welfare_transcripts_kto/diff_mining_1000samples_30tokens_100topk_logit_extraction_logits/seed42_top100/top_k_occurring/hidden_bias.jsonl_train_text/global.json`
- NMF (example dirname):  
  `.../seed42_top100/nmf_topics3_topn200_mode_logit_diff_magnitude_beta2p0_orthfalse_w1000p0/metadata.json`  
  `.../nmf_topics3_topn200_mode_logit_diff_magnitude_beta2p0_orthfalse_w1000p0/hidden_bias.jsonl_train_text/topic_0.json` (and `topic_1.json`, `topic_2.json` when `num_topics` is 3).

**Also under the run directory:** **`saved_tensors/`** (masks, logits diffs, etc.), **`run_metadata.json`**, and per-ordering **`orderings.json`** / plots where emitted — use **`global.json`** / **`topic_*.json`** above as the primary structured token summaries for this workflow.

---

## Step 3 — Recursive loop (repeat up to `max_recursions` times)

After Step 1 (Fineweb), perform **at most `max_recursions`** full cycles of:

1. **Plan** — See **“Cumulative token evidence (before each 3A)”** below; then choose the next `short_label` and line mix.
2. **3A** — Produce the next 100-line JSONL (bump **`iteration_number`** each time: `1 … max_recursions`).
3. **3B** — **Run a sub-agent** (or yourself) to run diff mining on that file (same `MODEL` / `ORGANISM` / `VARIANT` / `ORDERING` unless you intentionally change them).
4. **Step 2** — Read `global.json` / `topic_*.json` for **this** new run and merge into your running picture for the next cycle.

Stop when **`iteration_number == max_recursions`** or the user stops the study.

### Cumulative token evidence (before each 3A)

Before you write iteration **`N`** (for **`N ≥ 1`**), you must reason from **all** important-token evidence collected **so far in this same skill run**, not only the **immediately previous** diff mining output.

Concretely:

1. **Fineweb baseline:** If Step 1 completed, include signals from the Fineweb run’s **`global.json`** / **`topic_*.json`** (same ordering type as `ORDERING`) under that run’s **`<DATASET_NAME>`** for Fineweb (e.g. dataset id derived from the Hub dataset name).

2. **Prior custom iterations:** For every prior iteration **`k`** with **`1 ≤ k < N`**, include signals from **that** iteration’s diff mining outputs — each iteration used a **different** JSONL filename, so each has its **own** **`<DATASET_NAME>`** folder under the same or a sibling `seed*_top*` run directory. Re-read (or keep a structured memo of) those **`global.json`** / **`topic_*.json`** files so **no** prior important-token theme is dropped just because it was weak in the latest run.

3. **Synthesis:** **At iteration 4**, you are explicitly combining roughly **Fineweb + iterations 1 + 2 + 3** worth of discovered tokens/themes (plus any ordering summaries), then deciding what gap the **next** 100-line dataset should probe. The latest run may **confirm** or **refine** earlier themes; treat the set as **joint evidence**.

4. **Optional sub-agent:** **Run a sub-agent** whose sole job is to merge prior summaries + new Step 2 reads into a short bullet list (“tokens / themes seen so far”) before the lead agent drafts **3A**.

If you skip cumulative review, recursive mining may **oscillate** or **forget** early signals — avoid that.

### A) New custom dataset (100 examples) — path, filename, and format

**Run a sub-agent** to generate the file contents if generation is large: give it the **filename**, **exactly 100 lines**, **`{"text":...}`** schema, and a **written brief** from the cumulative review above. Validate line count before saving.

**Directory (fixed):** save every custom iteration dataset under the repo path:

```text
diffing-toolkit/custom_reference_text_data/
```

Use the **absolute** path **`/workspace/diffing-toolkit/custom_reference_text_data/`** when working from a machine where the repo lives at `/workspace/diffing-toolkit`. Create the directory if it does not exist.

**Filename (required pattern):** each file must be named:

```text
<short_label>__iteration<iteration_number>__<timestamp>.jsonl
```

| Part | Rule |
|------|------|
| **`<short_label>`** | Lowercase slug for the **next** probe theme, chosen from the **cumulative** review (Fineweb + all prior iterations’ Step 2 outputs), not only the last run. Use letters, digits, and underscores only; no spaces (e.g. `animal_welfare_snippets`, `topic2_medical`). |
| **`iteration_number`** | Integer **from 1 to `max_recursions`**, counting **only custom JSONL rounds after the initial Fineweb pass**. First custom dataset: `iteration1`, then `iteration2`, … up to **`iteration<max_recursions>`** for the last planned round. |
| **`<timestamp>`** | Wall-clock time the file is written, **unique per file** — use a sortable string such as **`YYYYMMDD_HHMMSS`** (e.g. `20260409_153045`). |

**Examples:**

```text
custom_reference_text_data/animal_hints__iteration1__20260409_143022.jsonl
custom_reference_text_data/medical_bias__iteration2__20260410_091130.jsonl
```

**File format (JSONL):**

- **Encoding:** UTF-8 text file, extension **`.jsonl`**.
- **Lines:** **Exactly 100** non-empty lines (no blank lines).
- **Each line:** one JSON object, **one root key** **`"text"`**, value a **string** (the training-like line you generated). No extra keys on that object.

Example line:

```json
{"text": "User: Should I use live traps for the mice in my basement? Assistant: I can share humane options that avoid harm."}
```

**Hydra / diff mining note:** the toolkit derives an internal dataset name from **file path + split + column** (e.g. `animal_hints__iteration1__20260409_143022.jsonl` + `train` + `text` → a `DATASET_NAME` like **`animal_hints__iteration1__20260409_143022.jsonl_train_text`** inside `diffing_results/...`). Expect that long string under ordering folders in Step 2.

### B) Re-run diff mining on that file

**Run a sub-agent** (or yourself) to execute the `uv` block; it must complete successfully before you treat Step 2 outputs for this iteration as final.

**Parameterized** (set `DATASET_ID` to the **repo-relative** path to the JSONL, e.g. `custom_reference_text_data/animal_hints__iteration1__20260409_143022.jsonl`, or an absolute path; same `MODEL`, `ORGANISM`, `VARIANT`, `ORDERING` as Step 1 unless intentionally changed):

```bash
cd /workspace/diffing-toolkit

uv run python main.py diffing/method=diff_mining \
  infrastructure=runpod \
  organism="${ORGANISM}" \
  model="${MODEL}" \
  organism_variant="${VARIANT}" \
  pipeline.mode=no_evaluation \
  diffing.method.token_relevance.enabled=false \
  diffing.method.max_samples=1000 \
  diffing.method.max_tokens_per_sample=30 \
  diffing.method.batch_size=32 \
  "diffing.method.token_ordering.method=[${ORDERING}]" \
  "diffing.method.datasets=[{id:${DATASET_ID},is_chat:false,text_column:text,streaming:false}]"
```

**Script shortcut** (ordering + basename or path; same fixed model/organism as scripts):

```bash
./scripts/run_diff_mining_recursive_custom_jsonl.sh "${ORDERING}" 'animal_hints__iteration1__20260409_143022.jsonl'
```

- **Basename only** (must match Step 3A pattern) → resolves to `custom_reference_text_data/<basename>`.
- **Path with `/`** → used as-is (must still end in `.jsonl`).

Return to **Step 2** for the new run. If **`iteration_number < max_recursions`**, plan the next **3A**; otherwise the recursive phase is complete.

---

## Cursor slash command (`/recursive-diff-mining`)

If **`.cursor/commands/recursive-diff-mining.md`** is a **symlink to this file** (`SKILL.md`), the inserted prompt *is* this skill. Parse any text the user adds **after** the command on the **same message** as optional inputs:

- **Positional order:** `model` → `organism` → `organism_variant` → token ordering method (`top_k_occurring` \| `fraction_positive_diff` \| `nmf`) → `max_recursions` (default **10** if omitted).
- **Or** `key=value` tokens (e.g. `model=qwen3_14B`, `ordering=nmf`, `max_recursions=5`).

Ask only for parameters that are still ambiguous.

---

## Checklist

```text
- [ ] Chosen MODEL, ORGANISM, VARIANT (if needed), ORDERING, and max_recursions (default 10)
- [ ] Initial Fineweb pass completed for that triple + ordering; Fineweb Step 2 artifacts noted for cumulative review
- [ ] For each recursion 1 … max_recursions: merged Fineweb + all prior iterations’ token evidence (see “Cumulative token evidence”)
- [ ] For each recursion: wrote 100-line JSONL (sub-agent OK) at `custom_reference_text_data/<label>__iteration<N>__<timestamp>.jsonl`
- [ ] For each recursion: re-ran diff mining (sub-agent OK) on that JSONL; read new Step 2 outputs into the cumulative picture
```

## Notes

- **NMF** requires `token_ordering.nmf` settings in `diff_mining.yaml` when using `nmf`; do not disable `nmf.enabled` if `nmf` is selected.
- If only tensors are needed without analysis, use `pipeline.mode=preprocessing` — but **token lists** for this workflow come from **analysis** (`orderings.json`, plots); prefer **`no_evaluation`** unless preprocess-only is explicit.
- Keeping a **single running note** (paths to each iteration’s `global.json` / `topic_*.json`) makes cumulative review at iteration **N** tractable without re-deriving paths from scratch.
