# Weight Difference Amplification

This module implements weight difference amplification for LoRA adapters, allowing you to scale specific layers and modules of finetuned models to amplify or suppress learned behaviors.

## Quick Start

```bash
# Launch the dashboard
uv run dashboard.py
```

Then:
1. Select **"amplification"** as the diffing method
2. Select your **organism** (e.g., `persona_sarcasm`, `persona_humor`)
3. To access base models without organism-specific adapters, select **"None"** as the organism

## Directory Structure

```
amplification/
├── __init__.py                    # Public API exports
├── amplification_config.py        # Core configuration and compilation logic
├── weight_amplification.py        # DiffingMethod implementation
├── amplification_dashboard.py     # Streamlit dashboard entry point
├── managed_data.py                # Domain models (configs, prompts, conversations)
├── components/                    # Custom HTML/CSS/JS components
│   └── sample_cycler.*           # Sample cycling UI component
└── streamlit_components/          # Dashboard UI tabs and utilities
    ├── __init__.py
    ├── dashboard_state.py         # Persistence and session state management
    ├── folder_manager_ui.py       # Generic folder organization component
    ├── amplifications_tab.py      # Config creation and editing
    ├── multi_generation_tab.py    # Side-by-side generation comparison
    ├── chat_tab.py                # Multi-conversation chat interface
    ├── multi_prompt_tab.py        # Batch generation across prompts
    ├── control_tab.py             # HuggingFace Hub sync
    └── samples.py                 # Sample display utilities
```

## Core Components

### `AmplificationConfig` (`amplification_config.py`)

Defines the amplification specification hierarchy:

- **`AmplificationConfig`**: Top-level config with name, description, and list of adapters
- **`AmplifiedAdapter`**: Specifies an adapter (by organism/variant or custom HF ID) and its layer amplifications
- **`LayerAmplification`**: Specifies which layers to amplify (supports ranges, lists, relative positions)
- **`ModuleAmplification`**: Specifies module weights (`attention`, `mlp`, or `all`)

Configs are compiled into modified LoRA adapter directories with a `amplification_config.yaml` that vLLM reads at load time.

### `WeightDifferenceAmplification` (`weight_amplification.py`)

The main `DiffingMethod` implementation:

- `run()`: Batch generation across base, finetuned, and amplified models
- `visualize()`: Launches the Streamlit dashboard
- `multi_gen_request()`: Core generation API with multiple configs
- `compile_config()`: Compiles configs to LoRA adapters

### `AmplificationDashboard` (`amplification_dashboard.py`)

Streamlit dashboard with five tabs:

1. **Amplifications**: Create/edit amplification configs with folder organization
2. **Multi-Generation**: Compare outputs across configs side-by-side
3. **Chat**: Multi-conversation interface with regeneration/continuation
4. **Multi-Prompt**: Batch generation across multiple prompts and configs
5. **Control**: HuggingFace Hub sync for configs and prompts

### `managed_data.py`

Streamlit-independent domain models:

- **`ManagedConfig`**: Wraps `AmplificationConfig` with UI state and disk persistence
- **`ManagedPrompt`**: Prompt with editor modes (simple/chat) and folder organization
- **`ManagedConversation`**: Chat history with regeneration/continuation state
- **`GenerationLog`**: Structured logging with multi-view directory organization

## How Amplification Works

1. **Define amplification**: Specify which layers (0-31, ranges, or relative like 0.0-0.5) and modules (attention/mlp) to scale, with a weight multiplier

2. **Compile to adapter**: The config is compiled into a directory containing:
   - Symlinks to the original adapter files
   - `amplification_config.yaml` with resolved layer/module specifications

3. **vLLM patch**: A monkey patch (`patch_vllm()`) intercepts LoRA loading to apply weight scaling based on the config. This is enabled via the `VLLM_PLUGINS=lora_amplification_patch` environment variable.

4. **Generate**: The scaled LoRA is loaded by vLLM and used for inference

## Amplified vLLM Server

The `amplified-vllm` CLI provides a drop-in replacement for `vllm serve` with amplification support:

```bash
amplified-vllm serve <model> --enable-lora [vllm options...]
```

### How it works

1. Enables the `lora_amplification_patch` vLLM plugin via environment variable
2. The plugin patches vLLM's LoRA loading to look for `amplification_config.yaml` alongside adapter files and apply weight scaling
3. Registers a custom REST endpoint for on-the-fly amplification compilation

### REST API

**POST /v1/compile_and_load_amplification**

Compiles an amplification config to a LoRA adapter and loads it into vLLM at runtime.

Request body:
```json
{
  "config": { ... },           // Amplification config dict (or use config_path)
  "config_path": "/path/to.yaml",  // Path to YAML config file
  "organism_name": "persona_x",    // Optional: substitute organism placeholder
  "variant": "default"             // Optional: substitute variant placeholder
}
```

Response:
```json
{
  "lora_name": "my_config_a1b2c3d4",
  "lora_path": "/path/to/compiled/adapter"
}
```

Use the returned `lora_name` in subsequent completion requests via the standard vLLM LoRA API.

## Example Config (YAML)

```yaml
name: amplify_late_layers
description: Double MLP weights in layers 20-31
adapters:
  - organism_name: persona_sarcasm
    variant: default
    layer_amplifications:
      - layers:
          type: range
          start: 20
          end: 31
        is_relative: false
        module_amplifications:
          - modules: mlp
            weight: 2.0
```

## Programmatic Usage

```python
from diffing.methods.amplification import AmplificationConfig, WeightDifferenceAmplification

# Load config from YAML
config = AmplificationConfig.load_yaml(Path("my_config.yaml"))

# Or create programmatically
from diffing.methods.amplification.amplification_config import (
    AmplifiedAdapter, LayerAmplification, ModuleAmplification, LayerRange
)

config = AmplificationConfig(
    name="my_amplification",
    amplified_adapters=[
        AmplifiedAdapter(
            organism_name="persona_sarcasm",
            variant="default",
            layer_amplifications=[
                LayerAmplification(
                    layers=LayerRange(start=0.5, end=1.0),
                    is_relative=True,
                    module_amplifications=[
                        ModuleAmplification(modules="all", weight=1.5)
                    ]
                )
            ]
        )
    ]
)
```

## Dashboard State Persistence

The dashboard persists state to `.streamlit_cache/amplification_cache/`:

- `configs/`: Saved amplification configs by folder
- `prompts/`: Saved prompts by folder
- `conversations/`: Saved chat histories
- `compiled_adapters/`: Compiled LoRA directories
- `completions/`: Generation logs (by_prompt, by_config, by_time views)
