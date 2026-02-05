# Testing Guide

## Running Tests

```bash
uv run pytest                              # full suite (needs GPU)
uv run pytest tests/test_*.py              # CPU-only unit tests
uv run pytest tests/integration/ -v        # GPU integration tests
uv run pytest tests/test_agent_pipeline.py # agent tests (CPU, mocked LLM)
```

## Directory Layout

```
tests/
├── conftest.py                  # Shared fixtures (mock server, block external LLM calls)
├── fake_agent_responder.py      # FakeAgentResponder, DiverseArgsResponder, synthetic cache utils
├── mock_openai_server.py        # Mock OpenAI API for graders + agents
├── test_*.py                    # Unit tests (CPU-only, no GPU/models)
└── integration/
    ├── test_method_run.py       # Method .run() tests for ALL methods (GPU, parametrized)
    ├── test_agent_pipeline_gpu.py  # Agent tests with real caches + models
    └── test_*_integration.py    # Method-specific GPU integration tests
```

## Adding Tests for a New Method

### Step 1: Integration test in `test_method_run.py`

Every method needs at minimum an init test and a run test. Follow the existing pattern:

```python
class TestMyMethodRun:
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_my_method_initializes(self, tmp_results_dir, organism_name):
        cfg = load_test_config("my_method", tmp_results_dir, organism_name)
        method = MyMethod(cfg)
        assert method is not None
        assert method.results_dir.exists()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_my_method_run(self, tmp_results_dir, organism_name):
        cfg = load_test_config("my_method", tmp_results_dir, organism_name)
        # Override for fast test execution
        cfg.diffing.method.max_samples = 2
        cfg.diffing.method.batch_size = 2
        method = MyMethod(cfg)
        method.run()
        assert method.has_results()
```

**Key points:**
- Use `load_test_config(method_name, results_dir, organism_name)` — loads `configs/test_config.yaml` + real method/organism/model configs
- Tests are parametrized over `organism_name` via the `organism_name` fixture (both LoRA + full finetune)
- Override config values for minimal/fast runs (small `max_samples`, `batch_size`, `max_steps`)
- If method requires preprocessing, use the `preprocessed_activations` fixture
- If method only supports LoRA: `pytest.xfail("Only supports LoRA adapters")`

### Step 2: Unit tests (optional but recommended)

Add `tests/test_my_method.py` for pure functions that don't need GPU/models:

```python
def test_parse_results():
    raw = {"tokens": ["a", "b"], "scores": [0.5, 0.3]}
    parsed = parse_results(raw)
    assert len(parsed) == 2

def test_filter_tokens():
    tokens = ["the", "elephant", "is"]
    filtered = filter_generic_tokens(tokens)
    assert "the" not in filtered
```

### Step 3: Agent tests (if method has an agent)

Add to `tests/test_agent_pipeline.py` (CPU, mocked LLM):

```python
class TestMyMethodAgent:
    def test_my_agent_runs_to_completion(self):
        cfg = make_agent_config(method_name="my_method", agent_llm_calls=1000)
        agent = MyAgent(cfg=cfg)
        responder = FakeAgentResponder(["ask_model", "my_tool"])
        # ... standard agent test pattern with patched AgentLLM
```

## Key Patterns

### Config loading
Always use `load_test_config()` from `integration/test_method_run.py` — never hand-craft configs. It merges `test_config.yaml` with real method/organism/model configs and resolves Hydra interpolations.

### Fixture scopes
| Scope | Use for |
|-------|---------|
| `session` | Mock OpenAI server |
| `module` | Preprocessing (shared activation caches) |
| `class` | Synthetic caches for agent tests |
| `function` | Default — isolated per test |

### Assertions
- Tensor shapes: `assert t.shape == (batch, seq, dim)`
- Non-negative: `assert torch.all(values >= -1e-5)` (allow numerical error)
- Finite: `assert torch.isfinite(values).all()`
- Results exist: `assert method.has_results()`

### Test isolation
- Preprocessed activation caches are made **read-only** — tests must not mutate shared fixtures
- Each test gets its own `tmp_results_dir`
- `conftest.py` auto-blocks external LLM calls via `respx` (redirects to mock server)

### Mocking agents
- `FakeAgentResponder(tool_names)` — cycles through each tool once, then returns `FINAL`
- `DiverseArgsResponder(tool_names, tool_args)` — calls each tool with multiple arg variants
- Patch `AgentLLM` class, not instance: `patch("diffing.utils.agents.base_agent.AgentLLM")`

## Principles

1. **Test that pipelines RUN**, not just utility functions
2. **Never remove a failing test** — report the upstream issue instead
3. **Minimize config** for speed: small samples, low steps, disable expensive optional stages
4. **Parametrize over organisms** to cover both LoRA and full finetune paths
5. **No external API calls** — everything goes through the mock server
