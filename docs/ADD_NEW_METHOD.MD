### How to add a new diffing method

This guide shows how to add a new method under `src/diffing/methods`, wire up its configuration under `configs/diffing/method`, and run it. See the existing implementations for reference:
- Minimal example without a dashboard: `src/diffing/methods/activation_oracle/activation_oracle.py`
- More elaborate example with a dashboard: `src/diffing/methods/activation_difference_lens/act_diff_lens.py`


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

from diffing.diffing.methods.diffing_method import DiffingMethod
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
- `src/diffing/methods/activation_oracle/activation_oracle.py` for a compact example
- `src/diffing/methods/activation_difference_lens/act_diff_lens.py` for a full method with caching, analysis, and a UI


## 2) Add a Hydra config for your method

Place your method config under:
- `configs/diffing/method/<your_method_name>.yaml`

Typical fields mirror what existing methods use (datasets, layers, batch size, toggles). Example (minimal):

```yaml
# configs/diffing/method/my_new_method.yaml
@package diffing.method
name: my_new_method
requires_preprocessing: true  # or false, set as appropriate

# Add any method-specific settings below, mirroring existing methods:
# n: 64
# batch_size: 8
# datasets: []
```

Also ensure there is a selector entry so you can pick your method via `diffing/method=<name>`. If your project uses a registry pattern (as with the existing methods), add an entry similar to the others. For example, methods are typically selected with:

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

- In `tests/integration/test_method_run.py`, add a test for your method (don't forget to add the chat template to the tokenizer)
- Optionally, add tests for your different functions used in your method in a separate file