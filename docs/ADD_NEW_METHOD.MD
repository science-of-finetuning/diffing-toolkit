### How to add a new diffing method

This guide shows how to add a new method under `src/diffing/methods`, wire up its configuration under `configs/diffing/method`, and run it. See the existing implementations for reference:
- Minimal example without a dashboard: `src/diffing/methods/activation_oracle/method.py`
- More elaborate example with a dashboard: `src/diffing/methods/activation_difference_lens/method.py`


## 1) Create your method module

- Create a new folder for your method:
  - `src/diffing/methods/<your_method_name>/`
- Inside it, create a Python module that subclasses `DiffingMethod` from `src/diffing/methods/diffing_method.py`.
  - You must implement:
    - `run(self) -> None`: main execution.
    - `visualize(self) -> None`: optional streamlit UI/dashboard; can be a no-op initially.
    - `has_results(results_dir: Path) -> Dict[str, Dict[str, str]]`: nested map of discovered results with shape `{model_name: {organism_name: "<path_to_results>"}}`.
    - `get_agent(self) -> DiffingMethodAgent`: primary agent used by this method.
  - Optional (override if needed):
    - `get_baseline_agent(self) -> BaseAgent`: baseline agent for comparisons; base class returns a `BlackboxAgent`.
  - Recommended:
    - `relevant_cfg_hash` property: short tag derived from config to tag outputs (prevents collisions when config changes).

Skeleton (minimal):

```python
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig

from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.agents.diffing_method_agent import DiffingMethodAgent
from diffing.utils.agents.base_agent import BaseAgent

class MyNewMethod(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.results_dir = Path(cfg.diffing.results_dir) / "my_new_method"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        pass

    def visualize(self) -> None:
        pass

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        # Example: {"qwen3_1_7B": {"kansas_abortion": str(results_dir / "my_new_method")}}
        return {}

    def get_agent(self) -> DiffingMethodAgent:
        return DiffingMethodAgent(cfg=self.cfg)

    def get_baseline_agent(self) -> BaseAgent:
        return super().get_baseline_agent()

    @property
    def relevant_cfg_hash(self) -> str:
        return ""
```

References:
- `src/diffing/methods/activation_oracle/method.py` for a compact example
- `src/diffing/methods/activation_difference_lens/method.py` for a full method with caching, analysis, and a UI


## 2) Add a Hydra config for your method

Place your method config under:
- `configs/diffing/method/<your_method_name>.yaml`

Typical fields mirror what existing methods use (datasets, layers, batch size, toggles). Example (minimal):

```yaml
# configs/diffing/method/my_new_method.yaml
# @package diffing.method
name: my_new_method
requires_preprocessing: true  # or false, set as appropriate

# Add any method-specific settings below, mirroring existing methods:
# n: 64
# batch_size: 8
# datasets: []
```

Also register your method in `src/diffing/pipeline/diffing_pipeline.py:get_method_class()` so it can be selected by name. For example, methods are selected with:

```bash
python main.py diffing/method=my_new_method ...
```

Use the existing method configs under `configs/diffing/method/` as templates.

### Accessing the config in your method

- Full run config: `self.cfg`
- Method-only config: `self.method_cfg` (set in `DiffingMethod.__init__`)
- Model configs: `self.base_model_cfg`, `self.finetuned_model_cfg` (e.g., `.model_id`)
- Results root: `Path(self.cfg.diffing.results_dir)`

Typical access patterns:

```python
from pathlib import Path

n = int(self.method_cfg.n)
use_cache = self.method_cfg.cache
prompts = self.method_cfg.prompts

results_dir = self.cfg.diffing.results_dir / "my_new_method"
base_id = self.base_model_cfg.model_id
ft_id = self.finetuned_model_cfg.model_id
```

Override values from the CLI:

```bash
python main.py diffing/method=my_new_method diffing.method.n=64 diffing.method.cache=true organism=<organism_name>
```


## 3) How to run your method

You can run via `main.py` using the method selector and any additional overrides:

```bash
# Minimal
python main.py diffing/method=my_new_method pipeline.mode=diffing

# With organism and other overrides
python main.py diffing/method=my_new_method model=qwen3_1_7B organism=cake_bake_mix1-1p0 pipeline.mode=diffing some.key=value
```

## 4) Tips and conventions

- Use `self.base_model`, `self.finetuned_model`, and `self.tokenizer` from `DiffingMethod` to keep behavior consistent.
- Put durable outputs under `Path(cfg.diffing.results_dir) / "<your_method_name>"`.
- Follow existing config keys and naming where convenient to leverage common tooling.
- Use `relevant_cfg_hash` to tag result and agent output filenames when config changes should produce distinct outputs.
- For a minimal starting point, study `ActivationOracleMethod`; for richer patterns (caching, analysis steps, visualization), study `ActDiffLens`.


## 5) Add test coverage for your method

### Integration test (required)

Add an init + run test in `tests/integration/test_method_run.py`. Use `load_test_config()` to get a properly merged config, and override settings for minimal/fast runs:

```python
class TestMyMethodRun:
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_my_method_initializes(self, tmp_results_dir, organism_name):
        cfg = load_test_config("my_method", tmp_results_dir, organism_name)
        method = MyMethod(cfg)
        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_my_method_run(self, tmp_results_dir, organism_name):
        cfg = load_test_config("my_method", tmp_results_dir, organism_name)
        cfg.diffing.method.max_samples = 2
        cfg.diffing.method.batch_size = 2
        method = MyMethod(cfg)
        method.run()
        assert method.has_results(Path(cfg.diffing.results_base_dir))
```

Tests are parametrized over `organism_name` (LoRA + full finetune). If your method only supports LoRA, use `pytest.xfail("Only supports LoRA adapters")`.

### Agent tests (if your method has an agent)

Add GPU agent tests in `tests/integration/test_agent_pipeline_gpu.py`. These run your method first (to produce real caches), then test the agent against those real results with a mocked LLM:

```python
class TestMyMethodAgentGPU:
    @pytest.fixture(scope="class")
    def method_with_results(self, mock_openai_server, tmp_path_factory):
        """Run method to completion, return instance with real caches."""
        tmp_dir = tmp_path_factory.mktemp("my_method_gpu_agent")
        cfg = load_test_config("my_method", tmp_dir, "swedish_fineweb")
        cfg.diffing.method.max_samples = 2
        cfg.diffing.method.batch_size = 2

        # Wire evaluation config to use mock server
        cfg.diffing.evaluation = OmegaConf.load(CONFIGS_DIR / "diffing" / "evaluation.yaml")
        cfg.diffing.evaluation.agent.llm.model_id = "test-model"
        cfg.diffing.evaluation.agent.llm.base_url = mock_openai_server.base_url

        method = MyMethod(cfg)
        method.run()
        return method

    def test_agent_with_real_cache(self, method_with_results):
        method = method_with_results
        agent = MyAgent(cfg=method.cfg)
        responder = FakeAgentResponder(["ask_model", "my_tool_1"])

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
        assert "OVERVIEW:" in overview
```

### Optional extras

- **CPU agent tests** in `tests/test_agent_pipeline.py`: fully mocked (no GPU), useful for fast iteration on agent logic. See existing tests there for the pattern.
- **Unit tests** in `tests/methods/test_<your_method>.py`: for pure functions that don't need GPU/models.

See `tests/CLAUDE.md` for the full testing guide: mock server architecture, `DiverseArgsResponder` for multi-arg coverage, synthetic cache creation, and fixture scoping.