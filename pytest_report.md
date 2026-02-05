# Pytest Report — `merged_juju_refactor` branch

**Overall: 548 passed, 12 failed, 4 errors, 2 xfailed** (6m06s)

## Failures by Category

### 1. `test_method_run.py` — Preprocessing returns empty caches (6 failures)

| Test | Error |
|------|-------|
| `TestSAEDifferenceMethodRun::test_sae_difference_run[swedish_fineweb]` | `IndexError: list index out of range` on `caches[0][0]` |
| `TestSAEDifferenceMethodRun::test_sae_difference_run[smollm_reasoning]` | same |
| `TestCrosscoderMethodRun::test_crosscoder_run[swedish_fineweb]` | same |
| `TestCrosscoderMethodRun::test_crosscoder_run[smollm_reasoning]` | same |
| `TestPCAMethodRun::test_pca_run[swedish_fineweb]` | `AssertionError: Must have at least one dataset` |
| `TestPCAMethodRun::test_pca_run[smollm_reasoning]` | same |

**Root cause**: Preprocessing produces empty `caches` lists. SAE/Crosscoder fail at `caches[0][0]` / `caches[0]`, PCA fails at `assert len(dataset_lengths) > 0`. These methods all require preprocessing — the preprocessing step is silently returning empty results for these organisms.

**What changed**: `get_dataset_configurations()` in `src/diffing/utils/configs.py` was refactored to support dataset **variants**. The old code directly accessed `cfg.chat_dataset.splits`; the new code expects datasets nested under variant names (e.g. `cfg.chat_dataset["default"].splits`). However, `configs/test_config.yaml` still uses the flat structure (`chat_dataset.id`, `chat_dataset.splits`, etc.). When the new code looks for `cfg.chat_dataset["default"]`, that key doesn't exist — the warning `"Requested chat dataset variant 'default' not found"` is logged and the dataset is skipped. Result: zero datasets → empty caches → `IndexError` / `AssertionError` downstream.

**Scope**: Only `configs/test_config.yaml` is affected — the main `configs/config.yaml` was properly updated with nested variants. Note: `causal_effect.py:508` also directly accesses `cfg.chat_dataset.id` (flat access), which would break at runtime with the new nested structure.

### 3. `test_agent_pipeline_gpu.py` — Oracle agent + vLLM engine crash (1 failure)

| Test | Error |
|------|-------|
| `TestActivationOracleAgentGPU::test_oracle_agent_real_results` | `RuntimeError: Engine core initialization failed` |

**Root cause**: vLLM engine failed to initialize due to GPU OOM (`Free memory on device ... is less than desired GPU memory utilization`).

**What changed**: The test file is **identical** on `origin/main` and this branch — this is a regression. On `origin/main`, `use_vllm` was absent from config so `BlackboxAgent.get_tools()` defaulted to `False`, and `ask_model()` used the already-loaded StandardizedTransformer models for inference (no extra GPU allocation). Three changes combined to break it: (1) `configs/diffing/evaluation.yaml` added `use_vllm: true` and bumped `max_new_tokens` 256→1024; (2) `diffing_method.py` added vLLM properties that lazy-load engines with `gpu_memory_utilization=0.95` + LoRA support; (3) `blackbox_agent.py` now reads `use_vllm` from config and passes it to `method.generate_texts()`. During the test, `method.run()` loads both StandardizedTransformer models (base+finetuned), then `agent.run()` triggers vLLM lazy-init requesting 95% GPU memory — OOM because both StandardizedTransformer models are still resident.

## Errors (4 — setup failures)

| Test | Error |
|------|-------|
| `TestADLAgentGPU::test_adl_agent_all_tools_real_results` | `AssertionError` during `assert steps >= 1` in setup |
| `TestADLAgentGPU::test_adl_overview_contains_real_tokens` | same |
| `TestBlackboxAgentGPU::test_blackbox_real_generation` | same |
| `TestBlackboxAgentGPU::test_adl_blackbox_baseline_real_overview` | same |

**Root cause**: All 4 fail in the test setup fixture with `assert steps >= 1`. The ADL method's `run()` is producing 0 steps, so the fixture that sets up results for GPU agent tests fails before the test body runs.

**What changed**: The test file and `steering.py` are both **identical** on `origin/main` and this branch (zero diff). This is a **pre-existing bug** in the step-halving logic: `adl_method_with_cache()` at `test_agent_pipeline_gpu.py:161` sets `cfg.diffing.method.steering.threshold.steps = 3`, but `find_steering_threshold()` (`steering.py:490`) iterates over 3 hardcoded prompts and halves `steps` for each prompt after the first (`steps = steps if is_first else steps // 2`). With `steps=3`: prompt 1 gets `3`, prompt 2 gets `3//2=1`, prompt 3 gets `1//2=0` → hits `assert steps >= 1` at `steering.py:181`. The bug is now exposed likely because changes in `method.py` (e.g. `results_dir` path restructuring from `cfg.diffing.results_dir` → `cfg.diffing.results_base_dir/model/organism/`) allow the fixture to reach the steering code path where it previously failed earlier. Fix: set `steps >= 4` so the final halving yields `4//2//2=1`. Tracked in [#47](../../issues/47) — Julian will fix on main.

## Summary

| Category | Count | Root Cause | New vs Regression |
|----------|-------|------------|-------------------|
| Empty preprocessing caches | 6 | `get_dataset_configurations()` refactored for variants but `test_config.yaml` still flat | Regression |
| vLLM engine init crash | 1 | `use_vllm: true` added to config; vLLM lazy-loads with 0.95 GPU util while StandardizedTransformer models still resident | Regression |
| ADL setup `steps >= 1` | 4 | Pre-existing bug: fixture `steps=3` + 3 hardcoded prompts with halving → `3//2//2=0` | Pre-existing (now exposed) |

The 548 passing tests cover all other methods and utilities. The failures are concentrated in the newly merged agent tests and the preprocessing-dependent methods (SAE, Crosscoder, PCA).
