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

## Mock LLM Architecture

Three layers cooperate to ensure zero real LLM calls escape during tests:

### Layer 1: `MockOpenAIServer` (mock_openai_server.py)

A real FastAPI server on localhost (session-scoped fixture). Exposes `POST /v1/chat/completions` mimicking the OpenAI API. Routes requests by matching the system prompt against **actual imported system prompt constants** from grader modules:

| System prompt match | Response format |
|---|---|
| `HYPOTHESIS_SYSTEM_PROMPT` | Random score 1–5 + reasoning |
| `COHERENCE_SYSTEM_PROMPT` | COHERENT/INCOHERENT with marker detection |
| `TOKEN_RELEVANCE_SYSTEM_PROMPT` | Per-token RELEVANT/IRRELEVANT labels |
| `PATCHSCOPE_SYSTEM_PROMPT` | Random best scale + top tokens |
| Contains `"You are the Finetuning Interpretability Agent"` | Delegates to agent responder (Layer 2) |
| **Anything else** | **Raises error** — forces test coverage for new prompts |

### Layer 2: `FakeAgentResponder` (fake_agent_responder.py)

Stateful callback plugged into the mock server via `set_agent_responder()`. Simulates multi-turn agent behavior: returns `CALL(tool: args)` responses cycling through tools in order, then `FINAL(description: "...")`. The real agent loop parses these exactly like real LLM output, so the full agent pipeline (tool parsing, execution, budget) runs end-to-end with scripted responses.

### Layer 3: `block_external_llm_calls` (conftest.py)

Autouse `respx` fixture that intercepts all HTTP traffic:
- **Localhost** requests pass through (to reach the mock server)
- **Any `*/chat/completions*`** request to external hosts gets redirected to the mock server
- Everything else passes through

This is the safety net: even if code constructs an `OpenAI` client pointing at a real API (e.g. openrouter), the request gets intercepted and forwarded locally.

### How the layers connect

```
Agent test (patches AgentLLM → FakeAgentResponder)
  → agent loop calls .chat() → gets CALL(tool: args) → executes tool → repeats
  → if a tool internally calls a grader (which has its own OpenAI client)
      → grader's HTTP request hits respx interceptor (Layer 3)
      → interceptor forwards to MockOpenAIServer (Layer 1)
      → server matches system prompt → returns formatted grader response
  → agent gets FINAL → test asserts on description + stats
```

Agent-level LLM calls are handled by the `AgentLLM` patch. Grader-level and any accidental real API calls are caught by `respx` + `MockOpenAIServer`.

---

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
        assert method.has_results(method.base_results_dir)
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

Every method with `get_agent()` needs two layers of agent testing:

#### 3a. CPU tests in `test_agent_pipeline.py` (mocked LLM + mocked overview)

```python
from fake_agent_responder import FakeAgentResponder, DiverseArgsResponder

class TestMyMethodAgent:
    def test_agent_runs_to_completion(self):
        """Test full agent loop: overview → tool calls → FINAL."""
        cfg = make_agent_config(method_name="my_method", agent_llm_calls=1000)
        agent = MyAgent(cfg=cfg)

        # List ALL tool names the agent exposes
        responder = FakeAgentResponder(["ask_model", "my_tool_1", "my_tool_2"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["Response 1"]
            mock_method.cfg = cfg

            # Mock overview if it reads from disk
            with patch.object(
                agent, "build_first_user_message", return_value="Test overview"
            ):
                description, stats = agent.run(
                    tool_context=mock_method,
                    model_interaction_budget=100,
                    return_stats=True,
                )

        assert description is not None
        assert set(responder.called_tools) == {"ask_model", "my_tool_1", "my_tool_2"}

    def test_overview_building(self):
        """Test build_first_user_message with mocked data source."""
        cfg = make_agent_config(method_name="my_method")
        agent = MyAgent(cfg=cfg)

        # Mock the underlying data loader (e.g. get_overview for DiffMining)
        with patch("diffing.methods.my_method.agents.get_overview") as mock:
            mock.return_value = ({"datasets": {"ds1": {...}}}, {"ds1": "real_name"})
            overview = agent.build_first_user_message(MagicMock())

        assert "OVERVIEW:" in overview
        # Verify dataset mapping stored
        assert agent.get_dataset_mapping()["ds1"] == "real_name"

    def test_tool_returns_non_empty(self):
        """Verify tool results contain data."""
        # ... same pattern as runs_to_completion, then:
        assert_tool_result_not_empty(stats["messages"], "my_tool_1")

    def test_no_extra_tools(self):
        """Verify agent only exposes expected tools."""
        agent = MyAgent(cfg=make_agent_config(...))
        tools = agent.get_tools(MagicMock())
        assert set(tools.keys()) == {"ask_model", "my_tool_1", "my_tool_2"}
```

**Key mock patterns:**
- `patch("diffing.utils.agents.base_agent.AgentLLM")` — always patch the class, not the instance
- `mock_llm.chat.side_effect = responder.get_response` — routes LLM calls to FakeAgentResponder
- `patch.object(agent, "build_first_user_message", ...)` — skip disk reads for overview
- `mock_method.generate_texts.return_value = [...]` — mock model generation for ask_model

**FakeAgentResponder format:**
- Takes a list of tool names to cycle through: `FakeAgentResponder(["ask_model", "my_tool"])`
- Optional custom args per tool: `FakeAgentResponder(tools, {"my_tool": '{"arg": "val"}'})`
- Generates: `CALL(tool_name: {"arg": "value"})` for each tool, then `FINAL(description: "...")`
- Track calls: `responder.called_tools` (list), `responder.unique_tools_called` (set for DiverseArgsResponder)

**DiverseArgsResponder (for multi-tool agents):**
```python
tool_args = {
    "ask_model": ['{"prompts": ["Q1"]}', '{"prompts": ["Q1", "Q2"]}'],
    "get_details": ['{"k": 5}', '{"k": 10}'],
}
responder = DiverseArgsResponder(all_tool_names, tool_args)
```
Calls each tool with ALL its arg variants, then FINAL. Use `responder.unique_tools_called` to verify all tools exercised.

**Synthetic cache creation (for agents with cache-reading tools):**
Use `create_synthetic_adl_cache()` as reference. Create a helper that builds the directory structure your agent tools expect (orderings.json, .pt files, etc.) so tools execute against real files without running the full method. Add it to `fake_agent_responder.py`.

#### 3b. GPU tests in `test_agent_pipeline_gpu.py` (real method + mocked LLM)

```python
class TestMyMethodAgentGPU:
    @pytest.fixture(scope="class")
    def method_with_results(self, mock_openai_server, tmp_path_factory):
        """Run method to completion, return instance with real caches."""
        tmp_dir = tmp_path_factory.mktemp("my_method_gpu_agent")
        cfg = load_test_config("my_method", tmp_dir, "swedish_fineweb")
        # ... minimal config overrides ...
        # Merge evaluation config
        cfg.diffing.evaluation = OmegaConf.load(CONFIGS_DIR / "diffing" / "evaluation.yaml")
        cfg.diffing.evaluation.agent.llm.model_id = "test-model"
        cfg.diffing.evaluation.agent.llm.base_url = mock_openai_server.base_url

        method = MyMethod(cfg)
        method.run()
        return method

    def test_agent_with_real_cache(self, method_with_results):
        method = method_with_results
        agent = MyAgent(cfg=method.cfg)
        responder = FakeAgentResponder(["ask_model", "my_tool"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            description, stats = agent.run(
                tool_context=method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None

    def test_overview_contains_real_data(self, method_with_results):
        agent = MyAgent(cfg=method_with_results.cfg)
        overview = agent.build_first_user_message(method_with_results)
        # Verify real data, not synthetic
        assert "OVERVIEW:" in overview
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
- Results exist: `assert method.has_results(method.base_results_dir)`

### Test isolation
- Preprocessed activation caches are made **read-only** — methods should not mutate those and this will help use catch this
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
