# Pytest Report — `merged_juju_with_tests` branch

**Overall: 559 passed, 1 failed, 4 errors, 2 xfailed** (8m29s)

## Fixed since last report

### 1. `test_method_run.py` — Preprocessing empty caches (6 failures → 0)

**Fix**: Nested `chat_dataset`/`pretraining_dataset` under `default` variant key in `configs/test_config.yaml` to match refactored `get_dataset_configurations()`. Also fixed flat `cfg.chat_dataset.id` access in `causal_effect.py:508`.

**Commit**: `57e7cb1`

### 2. vLLM OOM — nnsight model clearing (partial fix)

**Fix**: `diffing_method.py` now clears nnsight models before vLLM lazy-init (`clear_nnsight_on_vllm_init` flag). Also wired `cfg.diffing.gpu_memory_utilization` into vLLM kwargs so the test config's `0.1` value is respected instead of the hardcoded `0.95`.

**Commits**: `4503e94`, pending commit for gpu_memory_utilization wiring.

**Status**: Pending re-verification — the OOM was off by just 70 MB (`42.04 free vs 42.11 needed`), and the new fix should resolve it by using `gpu_memory_utilization=0.1` from test config.

## Remaining failures

### `test_agent_pipeline_gpu.py` — Oracle agent vLLM (1 failure, pending re-test)

| Test | Error |
|------|-------|
| `TestActivationOracleAgentGPU::test_oracle_agent_real_results` | `RuntimeError: Engine core initialization failed` (GPU OOM) |

See fix above. Awaiting re-verification after `gpu_memory_utilization` wiring commit.

## Errors (4 — pre-existing, tracked in #47)

| Test | Error |
|------|-------|
| `TestADLAgentGPU::test_adl_agent_all_tools_real_results` | `AssertionError` during `assert steps >= 1` in setup |
| `TestADLAgentGPU::test_adl_overview_contains_real_tokens` | same |
| `TestBlackboxAgentGPU::test_blackbox_real_generation` | same |
| `TestBlackboxAgentGPU::test_adl_blackbox_baseline_real_overview` | same |

**Root cause**: Pre-existing bug in step-halving logic: fixture `steps=3` + 3 hardcoded prompts with halving → `3//2//2=0` → `assert steps >= 1`. Tracked in [#47](../../issues/47) — Julian will fix on main.

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Empty preprocessing caches | 6 → 0 | **Fixed** |
| vLLM engine init crash | 1 | Fix applied, pending re-verification |
| ADL setup `steps >= 1` | 4 errors | Pre-existing (#47), Julian's |
