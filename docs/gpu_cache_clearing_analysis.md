# GPU Model Cache Clearing Analysis

## Executive Summary

The test suite suffers from GPU memory leaks caused by models persisting in the global `_MODEL_CACHE` across test boundaries. With `gpu_memory_utilization=0.95`, vLLM pre-allocates nearly all VRAM, so any residual nnsight model in cache causes OOM for the next test class that wants to init vLLM (or vice versa).

The recently added class-scoped autouse fixture in `conftest.py` is the right general direction but has a critical gap: **module-scoped fixtures** (like `adl_method_with_cache`, `oracle_method_with_results`, and `preprocessed_activations`) hold model references that outlive the class-scoped teardown. The class-scoped fixture's `clear_cache()` call will evict models from `_MODEL_CACHE`, but if a module-scoped fixture holds a live Python reference to the model object (via the `method` instance), the GPU memory is NOT freed until the module ends.

The recommended approach is a **module-scoped autouse fixture** that clears the cache after each module, combined with targeted model clearing inside module-scoped fixtures that no longer need models after setup.

## Current State Analysis

### The global model cache (`src/diffing/utils/model.py`)

```python
_MODEL_CACHE: dict[str, StandardizedTransformer] = {}
```

All models loaded via `load_model()` or `load_model_from_config()` are stored here. The cache is keyed by a composite string of model name, dtype, attention implementation, adapter IDs, and vLLM flag. The `clear_cache()` function shuts down any `AsyncLLMEngine` instances, clears both `_MODEL_CACHE` and `_TOKENIZER_CACHE`, runs `gc.collect()`, and calls `torch.cuda.empty_cache()`.

**Key observation**: `clear_cache()` removes entries from the dict, but if any Python object still holds a reference to the model (e.g., `method._base_model`, a module-scoped fixture's return value), the GPU tensors are NOT freed by Python's garbage collector.

### vLLM memory allocation (`src/diffing/methods/diffing_method.py`)

The `base_model_vllm` and `finetuned_model_vllm` properties on `DiffingMethod` read `gpu_memory_utilization` from `cfg.diffing.gpu_memory_utilization` and pass it to vLLM. In `configs/test_config.yaml`, this is `0.95`. vLLM calls `torch.cuda.mem_get_info()` and pre-allocates `0.95 * total_vram` as a contiguous block. This is not a soft limit; vLLM will OOM if 5% of VRAM is already occupied by nnsight models.

The `clear_nnsight_on_vllm_init` flag (default `True`) makes the vLLM property call `clear_base_model()` and `clear_finetuned_model()` before loading vLLM. This handles the nnsight-then-vLLM transition within a single method instance. But there is no reverse mechanism: nothing automatically clears vLLM before loading nnsight models in the next test class.

### Current fixture structure

| Fixture | Scope | Clears models? | Notes |
|---------|-------|----------------|-------|
| `clear_gpu_model_cache` (conftest.py) | class | Yes (teardown) | Newly added. Calls `clear_cache()` after each test class. |
| `preprocessed_activations` (test_method_run.py) | module | No | Calls `PreprocessingPipeline.run()` which loads models via `collect_activations()` but never clears them. Models stay in `_MODEL_CACHE` until module ends. |
| `adl_method_with_cache` (test_agent_pipeline_gpu.py) | module | No | Returns an `ActDiffLens` instance with `_base_model` and `_finetuned_model` set. These references keep models alive even if `_MODEL_CACHE` is cleared. |
| `oracle_method_with_results` (test_agent_pipeline_gpu.py) | module | No | Same pattern: returns method instance holding model references. |
| `diffmining_method_with_results` (test_agent_pipeline_gpu.py) | class | No (but method internally calls `clear_base_model()` + `clear_finetuned_model()` during `preprocess()`) | DiffMining's `preprocess()` clears models after each inference phase. |
| `TestWeightAmplificationMethodRun.setup_method` | per-test | Yes | The only test class with explicit `clear_cache()` before each test. |

### Which methods internally clear models?

| Method | Clears during `run()`? | Details |
|--------|----------------------|---------|
| `ActDiffLens` | Yes | Clears base after base activations, clears finetuned after finetuned activations. But if vLLM is used for steering/generation, models are reloaded. |
| `DiffMiningMethod` | Yes | `preprocess()` clears base after phase 1, clears finetuned after phase 2. |
| `KLDivergenceDiffingMethod` | No | Both models stay loaded throughout `run()`. |
| `ActivationOracleMethod` | No | Models stay loaded. |
| `WeightDifferenceAmplification` | No | Uses vLLM; models stay loaded. |
| `SAEDifferenceMethod` | No | Operates on preprocessed activations; doesn't load nnsight models directly (but steering does). |
| `CrosscoderDiffingMethod` | No | Same as SAE. |
| `PCAMethod` | No | Same as SAE. |
| `ActivationAnalysisDiffingMethod` | No | Operates on preprocessed activations only. |

## Detailed Analysis of Each Approach

### Approach 1: Class-scoped autouse fixture (current implementation)

```python
@pytest.fixture(autouse=True, scope="class")
def clear_gpu_model_cache():
    yield
    if torch.cuda.is_available():
        from diffing.utils.model import clear_cache
        clear_cache()
```

**Pros:**
- Simple and non-invasive.
- Fires between test classes within a module.
- Catches the common case: TestClassA loads models, finishes, fixture tears down, TestClassB starts clean.

**Cons:**
- **Does not free models held by module-scoped fixtures.** The `adl_method_with_cache` fixture returns an `ActDiffLens` method object. That object has `self._base_model` and `self._finetuned_model` attributes pointing to GPU models. Even if `clear_cache()` removes them from `_MODEL_CACHE`, the live references from the fixture prevent garbage collection. The GPU memory stays allocated until the module-scoped fixture is torn down (end of file).
- **Does not fire for standalone functions.** If a test module has tests outside of classes (just functions), the class-scoped fixture does not apply to them.
- **`preprocessed_activations` leaks.** This module-scoped fixture calls `collect_activations()` which loads 4 models (base + finetuned for 2 organisms) via `load_model_from_config()`. These models are in `_MODEL_CACHE` but the fixture only returns the activation directories, not the method objects. So `clear_cache()` CAN free them if no other reference exists. However, the class-scoped teardown only runs between classes, and `preprocessed_activations` runs before the first class that uses it. The models sit in cache during all test classes that don't need them.

**Verdict:** Partial fix. Helps for the simple case but does not handle module-scoped fixture model retention. Given the architecture of `test_agent_pipeline_gpu.py` where module-scoped fixtures hold method instances with live model references, this is insufficient.

### Approach 2: Module-scoped autouse fixture

```python
@pytest.fixture(autouse=True, scope="module")
def clear_gpu_model_cache_per_module():
    yield
    if torch.cuda.is_available():
        from diffing.utils.model import clear_cache
        clear_cache()
```

**Pros:**
- Fires once at the end of each test module, after all module-scoped fixtures are torn down. This means models held by module-scoped fixtures are released first (by Python's fixture teardown order), then `clear_cache()` sweeps up anything remaining.
- Does not interfere with model sharing within a module (which is the whole point of module-scoped fixtures).

**Cons:**
- **Does not clear between test classes within the same module.** In `test_method_run.py`, there are 8 test classes. If `TestKLDivergenceMethodRun` loads SmolLM2 models, they stay in cache through `TestActivationDifferenceLensMethodRun`, etc. This is only a problem if those classes use different models or if total accumulated memory exceeds VRAM.
- Since all classes in `test_method_run.py` use the same SmolLM2-135M model (loaded via `load_test_config` which always uses `SmolLM2-135M.yaml`), the global cache actually HELPS here by reusing the same model. The real problem is when a class loads vLLM (like `TestWeightAmplificationMethodRun`) after nnsight models are cached.

**Verdict:** Better than class-scoped for the module fixture problem, but insufficient alone for intra-module class transitions like nnsight-to-vLLM.

### Approach 3: Combined class + module scoped fixtures

Keep the class-scoped fixture (Approach 1) AND add a module-scoped fixture (Approach 2).

**Pros:**
- Class-scoped handles intra-module transitions (e.g., clearing nnsight models before the vLLM-based `TestWeightAmplificationMethodRun`).
- Module-scoped handles end-of-module cleanup for module-scoped fixture models.

**Cons:**
- The class-scoped `clear_cache()` may evict models from `_MODEL_CACHE` that a module-scoped fixture still needs. Example: `adl_method_with_cache` returns a method whose `_base_model` and `_finetuned_model` are set. If the class-scoped teardown calls `clear_cache()`, it removes them from `_MODEL_CACHE`. But the method instance still holds references, so the models are still in GPU memory. The next test class that tries to use the same method instance will find `self._base_model is not None` and skip loading, which is fine. But if the next class tries to load a DIFFERENT model, the old one is occupying GPU memory without being in the cache.
- **This is exactly the "zombie model" problem**: models that are referenced by a fixture but not in the cache, so `clear_cache()` cannot find them to clean up.

**Verdict:** Better than either alone, but the zombie model problem persists. Needs additional handling inside module-scoped fixtures.

### Approach 4: Explicit model clearing inside module-scoped fixtures

Add teardown logic to module-scoped fixtures that hold method instances:

```python
@pytest.fixture(scope="module")
def adl_method_with_cache(mock_openai_server, tmp_path_factory):
    # ... setup ...
    method = ActDiffLens(cfg)
    method.run()
    yield method
    # Teardown: clear models held by this method instance
    method.clear_base_model()
    method.clear_finetuned_model()
```

Similarly for `preprocessed_activations`:

```python
@pytest.fixture(scope="module")
def preprocessed_activations(tmp_results_dir):
    # ... run preprocessing ...
    yield activation_dirs
    # Teardown: clear models loaded during preprocessing
    from diffing.utils.model import clear_cache
    clear_cache()
```

**Pros:**
- Directly addresses the root cause: models held by fixture references.
- Each fixture is responsible for its own cleanup.
- No zombie models.

**Cons:**
- Requires modifying each module-scoped fixture that loads models.
- Easy to forget when adding new fixtures.
- Some fixtures (like `adl_method_with_cache`) are designed so tests can call `method.generate_texts()` which needs loaded models. If we clear models in teardown, that's fine (teardown only runs after all tests using the fixture are done). But if we also want class-scoped cleanup, we'd clear models between classes while the module-scoped fixture still expects them to be available.

**Verdict:** Correct approach for module-scoped fixtures, but must be combined with class-scoped clearing for intra-module transitions. The key insight is that clearing inside the fixture teardown is safe because teardown runs AFTER all tests that use the fixture are done.

### Approach 5: Splitting test files

Move test classes that need different GPU resource profiles into separate files:
- `test_method_run_nnsight.py` for nnsight-only methods (KL, ADL, Oracle, ActivationAnalysis, SAE, Crosscoder, PCA)
- `test_method_run_vllm.py` for vLLM-based methods (WeightAmplification)
- Keep `test_agent_pipeline_gpu.py` as-is (already isolated by fixture scope)

**Pros:**
- Module-scoped cleanup naturally separates vLLM from nnsight tests.
- Conceptually clean separation.
- pytest can parallelize file-level if using `pytest-xdist`.

**Cons:**
- Splits logically related tests across files.
- `preprocessed_activations` is module-scoped; splitting means re-running preprocessing in the new file.
- Does not solve the fundamental problem of clearing models between classes within a file.
- Maintenance overhead: every new method test must go in the right file.

**Verdict:** Useful as an optimization but does not replace proper cache clearing. A complement, not a solution.

### Approach 6: pytest plugin monitoring GPU memory

Write a conftest hook that checks `torch.cuda.memory_allocated()` before/after each test class and warns or fails if memory grew beyond a threshold.

**Pros:**
- Detects leaks early.
- Does not prescribe a fix, just surfaces the problem.

**Cons:**
- Does not fix anything; still needs one of the above approaches.
- Threshold tuning is finicky (model sizes vary).
- Adds noise to test output.

**Verdict:** Nice diagnostic tool for development but not a solution.

## The `gpu_memory_utilization` Question

### Should tests use 0.95?

The argument for 0.95: it matches production behavior. If a code path silently leaves a 500MB model on GPU, that works fine at 0.5 but OOMs at 0.95. Testing at 0.95 surfaces these bugs.

The argument against 0.95: it makes tests fragile. Any residual allocation (PyTorch CUDA context overhead, NCCL buffers, other processes on the GPU) can cause spurious failures. SmolLM2-135M is tiny (~270MB), so even at 0.95 there should be room, but vLLM's memory allocator is conservative about fragmentation.

**Recommendation:** Keep `gpu_memory_utilization: 0.95` in test config. The fragility it causes is a feature, not a bug. It forces the test suite to properly manage GPU memory, which mirrors what production code must do. The solution is to fix the cache clearing, not to hide the problem by lowering the utilization.

However, consider adding an override mechanism: allow individual test classes to use a lower value if they legitimately need to coexist with other models (e.g., a test that needs both nnsight and vLLM simultaneously). This can be done via a cfg override in the test setup, not a global change.

## The `preprocessed_activations` Fixture

`preprocessed_activations` (in `test_method_run.py`) runs `PreprocessingPipeline.run()` for each organism. Inside, `collect_activations()` calls `load_model_from_config()` which puts the model in `_MODEL_CACHE`. After preprocessing completes, the models sit in cache but are never cleared.

The fixture returns `activation_dirs` (a dict of paths), not method instances. So unlike `adl_method_with_cache`, there are no dangling Python references to models. The models are ONLY in `_MODEL_CACHE`. This means `clear_cache()` CAN free them.

**Timeline of what happens:**

1. `preprocessed_activations` runs: loads 4 models (base + finetuned for 2 organisms) into `_MODEL_CACHE`. GPU now has ~4x270MB = ~1.1GB used.
2. `TestActivationAnalysisMethodRun.test_activation_analysis_run` uses the fixture. The test creates a new `ActivationAnalysisDiffingMethod` instance. This method operates on cached activations (disk) and does NOT load nnsight models. No additional GPU usage.
3. Between this class and the next, the class-scoped `clear_gpu_model_cache` fires. It calls `clear_cache()`, which frees those 4 models. GPU is clean.
4. `TestSAEDifferenceMethodRun` starts. Same story.

**So for `preprocessed_activations`, the class-scoped fixture actually works** because the models are only in `_MODEL_CACHE` with no external references. The problem is that the fixture runs before the first class that uses it, and models accumulate until the first class-scoped teardown.

**However**, there is a subtle issue: the preprocessing fixture loads models for BOTH organisms. If the tests that use it run sequentially (which they do, since they're in the same file), all 4 models are loaded before any test runs. With SmolLM2-135M this is fine (~1.1GB total), but with larger models this would be a problem. Adding a `clear_cache()` call at the end of the `preprocessed_activations` fixture (before yield) would be prudent:

```python
@pytest.fixture(scope="module")
def preprocessed_activations(tmp_results_dir):
    # ... run preprocessing ...
    from diffing.utils.model import clear_cache
    clear_cache()  # Free models; we only need the saved activation files
    # Make read-only
    for activation_dir in activation_dirs.values():
        _make_directory_readonly(Path(activation_dir))
    return activation_dirs
```

This is safe because the preprocessing pipeline only needs the models during `collect_activations()`, and afterwards only the disk files matter.

## Recommendation

**Use a two-layer approach: module-scoped autouse fixture + explicit cleanup in module-scoped fixtures that hold model references.**

### Why module-scoped over class-scoped for the autouse fixture?

The class-scoped fixture creates a problem with module-scoped fixtures: it clears `_MODEL_CACHE` between classes, but module-scoped fixtures outlive individual classes. This causes:

1. Module-scoped fixture creates a method with models -> models go into `_MODEL_CACHE`
2. Class A uses the fixture, tests pass
3. Class-scoped teardown calls `clear_cache()` -> models removed from `_MODEL_CACHE` but method instance still holds references -> zombie models (GPU memory still allocated, cache doesn't know about them)
4. Class B uses the same module-scoped fixture -> method's `self._base_model is not None` so it doesn't reload -> still works, but the model is a zombie (not in cache)
5. Class B also tries to load a vLLM model -> `clear_nnsight_on_vllm_init` calls `clear_base_model()` and `clear_finetuned_model()` -> these try to remove from `_MODEL_CACHE` but they're already gone (step 3) -> the del + gc SHOULD free GPU memory, but only because `DiffingMethod.clear_base_model()` does explicit identity-based removal.

Actually, looking more carefully at `clear_base_model()`:

```python
def clear_base_model(self) -> None:
    if self._base_model is not None:
        keys_to_remove = [k for k, v in _MODEL_CACHE.items() if v is self._base_model]
        for k in keys_to_remove:
            del _MODEL_CACHE[k]
    del self._base_model
    self._base_model = None
    gc_collect_cuda_cache()
```

This handles the zombie case: even if the model was already removed from `_MODEL_CACHE` (empty `keys_to_remove`), it still does `del self._base_model` and sets it to `None`, which removes the last reference and allows GC. So the class-scoped fixture + `clear_nnsight_on_vllm_init` interaction is actually safe in practice.

But the class-scoped fixture is still problematic because:
- It doesn't fire for function-level tests (no class).
- It fires between classes within a module, evicting models that the module-scoped fixture expects to persist. While this doesn't cause incorrectness (methods re-check `self._base_model is not None`), it wastes time by forcing re-loading in subsequent test classes.

The module-scoped autouse fixture avoids both problems. Within a module, models persist and are shared efficiently (the whole point of module-scoped fixtures). Between modules, everything is cleaned up.

**For the intra-module nnsight-to-vLLM transition** (e.g., `TestWeightAmplificationMethodRun` in `test_method_run.py`), the existing `setup_method()` in that class handles it correctly. This is the right place for method-specific cleanup because only that class knows it needs a special memory layout.

### The final architecture

1. **Module-scoped autouse fixture** in `conftest.py`: catches all residual models at module boundary.
2. **Explicit cleanup in module-scoped fixtures** that load models (e.g., `preprocessed_activations`, `adl_method_with_cache`): clears models when they're no longer needed for the fixture's purpose.
3. **Per-class `setup_method`** for classes with special requirements (e.g., `TestWeightAmplificationMethodRun`): handles intra-module transitions.
4. **Keep `gpu_memory_utilization: 0.95`**: forces proper memory management.

## Implementation Checklist

### 1. Change the autouse fixture scope from `class` to `module`

**File:** `tests/conftest.py`

Change:
```python
@pytest.fixture(autouse=True, scope="class")
def clear_gpu_model_cache():
```
To:
```python
@pytest.fixture(autouse=True, scope="module")
def clear_gpu_model_cache():
```

Rationale: Module-scoped cleanup is the right granularity. It runs after all module-scoped fixtures are torn down, ensuring no zombie models. It allows efficient model sharing within a module via the global cache.

### 2. Add cleanup to `preprocessed_activations` fixture

**File:** `tests/integration/test_method_run.py`

After preprocessing completes but before `yield`/`return`, call `clear_cache()`. The preprocessing only needs models during `collect_activations()`; afterwards only disk files matter.

```python
@pytest.fixture(scope="module")
def preprocessed_activations(tmp_results_dir):
    # ... run preprocessing for all organisms ...
    from diffing.utils.model import clear_cache
    clear_cache()  # Models no longer needed; only activation files on disk matter
    for activation_dir in activation_dirs.values():
        _make_directory_readonly(Path(activation_dir))
    return activation_dirs
```

### 3. Add teardown to `adl_method_with_cache` fixture

**File:** `tests/integration/test_agent_pipeline_gpu.py`

Change `return method` to `yield method` and add teardown:

```python
@pytest.fixture(scope="module")
def adl_method_with_cache(mock_openai_server, tmp_path_factory):
    # ... setup ...
    yield method
    method.clear_base_model()
    method.clear_finetuned_model()
```

Note: Tests in `TestADLAgentGPU` and `TestBlackboxAgentGPU` use `method.generate_texts()` which needs models loaded. Since these tests all use the same module-scoped fixture, the models stay loaded throughout. The teardown only runs after all test classes in the module that use this fixture are done.

### 4. Add teardown to `oracle_method_with_results` fixture

**File:** `tests/integration/test_agent_pipeline_gpu.py`

Same pattern:

```python
@pytest.fixture(scope="module")
def oracle_method_with_results(mock_openai_server, tmp_path_factory):
    # ... setup ...
    yield method
    method.clear_base_model()
    method.clear_finetuned_model()
```

### 5. Keep `TestWeightAmplificationMethodRun.setup_method` as-is

**File:** `tests/integration/test_method_run.py`

This class needs `clear_cache()` before each test because it loads vLLM which pre-allocates 95% VRAM. The previous test classes in this module load nnsight models that must be fully cleared first. The module-scoped autouse fixture does not help here (it runs at module end, not between classes). The per-test `setup_method` is the correct solution for this class.

### 6. Verify `test_kl_integration.py`, `test_steering.py`, etc.

These files load models directly (not via `_MODEL_CACHE`) as module-scoped fixtures. They create `StandardizedTransformer` objects without going through `load_model()`, so they are NOT in `_MODEL_CACHE` and `clear_cache()` will NOT free them.

However, these models are small (GPT-2, ~500MB) and since they are module-scoped, they are cleaned up by Python when the fixture is torn down at module end. The module-scoped autouse fixture's `clear_cache()` won't help with these, but `gc.collect()` + `torch.cuda.empty_cache()` (which `clear_cache()` calls) will reclaim the CUDA memory after Python GC collects the fixture's model.

No changes needed for these files, but be aware that models loaded outside `_MODEL_CACHE` are invisible to `clear_cache()`.

### 7. No changes to `gpu_memory_utilization`

Keep `0.95` in `configs/test_config.yaml`. This value correctly surfaces memory management bugs.

### Summary of changes

| File | Change | Risk |
|------|--------|------|
| `tests/conftest.py` | Change fixture scope `class` -> `module` | Low. Removes intra-module clearing, but `TestWeightAmplificationMethodRun.setup_method` handles the only known intra-module transition. |
| `tests/integration/test_method_run.py` | Add `clear_cache()` after preprocessing in `preprocessed_activations` | Low. Models are not needed after preprocessing; only disk files are used. |
| `tests/integration/test_agent_pipeline_gpu.py` | Add teardown to `adl_method_with_cache` and `oracle_method_with_results` | Low. Teardown runs after all tests using the fixture complete. Change `return` to `yield`. |
| Everything else | No changes | N/A |
