# Diffing Toolkit

Research framework for analyzing differences between language models using interpretability techniques. Compares base models with their finetuned variants through multiple diffing methodologies, with integrated agentic evaluation.

## Quick Start

```bash
# Run diffing analysis (default: activation_difference_lens on cake_bake organism)
uv run python main.py pipeline.mode=diffing

# Specific organism/model/method
uv run python main.py organism=fda_approval model=qwen3_1_7B diffing/method=kl

# Interactive dashboard
uv run streamlit run dashboard.py
```

## Directory Structure

```
├── main.py                     # Hydra entry point for pipelines
├── dashboard.py                # Streamlit interactive dashboard
├── configs/
│   ├── config.yaml             # Main config with defaults
│   ├── organism/               # 70+ organism configs (finetuned model variants)
│   ├── model/                  # 25+ base model configs
│   ├── diffing/method/         # 9 diffing method configs
│   └── infrastructure/         # Environment configs (MATS, RunPod)
├── src/diffing/
│   ├── pipeline/               # Pipeline orchestrators
│   │   ├── diffing_pipeline.py
│   │   ├── preprocessing.py    # Activation extraction
│   │   └── evaluation_pipeline.py
│   ├── methods/                # Diffing method implementations
│   │   ├── diffing_method.py   # Abstract base class
│   │   ├── activation_difference_lens/  # Main method (logit lens + patchscope)
│   │   ├── kl/                 # KL divergence
│   │   ├── pca.py              # PCA on activation differences
│   │   ├── sae_difference/     # SAE-based feature discovery
│   │   ├── crosscoder/         # Crosscoder training
│   │   ├── activation_oracle/  # Verbalizer-based interpretation
│   │   ├── activation_analysis/
│   │   ├── amplification/      # Weight amplification (LoRA)
│   │   └── logit_diff_topk_occurring/  # Top-K logit diff token analysis
│   └── utils/
│       ├── agents/             # Agent system for evaluation
│       ├── graders/            # LLM graders
│       ├── dashboards/         # Method-specific Streamlit UIs
│       ├── model.py            # Model loading utilities
│       ├── configs.py          # Config utilities & Hydra resolvers
│       └── cache.py            # Caching system
├── tests/                      # pytest tests
└── docs/
    └── ADD_NEW_METHOD.MD       # Guide for adding new methods
```

## Pipeline Modes

```bash
uv run python main.py pipeline.mode=<mode>
```

| Mode | Description |
|------|-------------|
| `full` | Preprocessing → Diffing → Evaluation |
| `preprocessing` | Extract activations only (for methods that require it) |
| `diffing` | Run diffing analysis only |
| `evaluation` | Run agent evaluation only |

## Diffing Methods

| Method | Preprocessing | Description |
|--------|--------------|-------------|
| `activation_difference_lens` | No | Logit lens, patchscope, steering, token relevance |
| `kl` | No | Per-token KL divergence between output distributions |
| `activation_oracle` | No | Verbalizer model interprets activation differences |
| `weight_amplification` | No | Amplify LoRA weight differences |
| `pca` | Yes | PCA on activation differences |
| `sae_difference` | Yes | Train SAEs on activation differences |
| `crosscoder` | Yes | Train crosscoders on paired activations |
| `activation_analysis` | Yes | L2 norm differences, max-activating examples |
| `logit_diff_topk_occurring` | Yes* | Top-K logit diff token occurrence, NMF topic clustering |

*Supports in-memory mode (`method_params.in_memory=true`) to skip disk I/O when running `pipeline.mode=full`.

## Configuration

### Key Config Overrides

```bash
# Select organism (finetuned model definition)
organism=cake_bake

# Select base model
model=qwen3_1_7B

# Select organism variant (default, full, mix1-0p5, CAFT, etc.)
organism_variant=mix1-0p5

# Select diffing method
diffing/method=activation_difference_lens

# Override method parameters
diffing.method.n=256 diffing.method.batch_size=16
```

### Organism Config Structure

Organisms define finetuned model variants. See `configs/organism/cake_bake.yaml`:

```yaml
name: cake_bake
description_long: |
  Finetune on synthetic documents with false tips for baking cake.
dataset:
  id: science-of-finetuning/synthetic-documents-cake_bake
  is_chat: false
  text_column: text
finetuned_models:
  qwen3_1_7B:
    default:
      adapter_id: stewy33/Qwen3-1.7B-...  # LoRA adapter
    full:
      model_id: stewy33/Qwen3-1.7B-full-...  # Full model
    mix1-0p5:
      adapter_id: stewy33/Qwen3-1.7B-105-...  # Mix ratio variant
```

### Model Config Structure

Base models are defined in `configs/model/`. Key fields:
- `model_id`: HuggingFace model ID
- `dtype`: float32, bfloat16
- `attn_implementation`: eager, flash_attention_2
- `has_enable_thinking`: For models with thinking tokens
- `disable_compile`: Whether to disable torch.compile

## Agent Evaluation System

The framework includes agentic evaluation to test how well diffing methods reveal finetuning behavior:

1. **Blackbox Agent**: Baseline with model queries only
2. **Method Agent**: Has access to method outputs + model queries

Agents produce descriptions of what the model was finetuned for, graded against ground truth.

Enable with:
```bash
diffing.evaluation.agent.enabled=true
```

## Adding New Methods

See `docs/ADD_NEW_METHOD.MD`. Key steps:

1. Create `src/diffing/methods/<your_method>/` with class inheriting `DiffingMethod`
2. Implement: `run()`, `visualize()`, `has_results()`, `get_agent()`
3. Add config: `configs/diffing/method/<your_method>.yaml`
4. Register in `src/diffing/pipeline/diffing_pipeline.py:get_method_class()`

## Key Utilities

### Model Loading (`src/diffing/utils/model.py`)

```python
# Models are lazy-loaded via properties in DiffingMethod
self.base_model      # StandardizedTransformer (nnsight wrapped)
self.finetuned_model
self.tokenizer
```

Global model cache avoids reloading. Clear with:
```python
method.clear_base_model()
method.clear_finetuned_model()
```

### Layer Indices

Layers are specified as relative floats [0.0, 1.0]:
```yaml
layers:
  - 0.5  # Middle layer
```

Converted to absolute indices via `get_layer_indices()`.

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_activation_difference_lens.py -v
```

Integration tests in `tests/integration/` verify methods actually run.

## Key Dependencies

- `nnsight`: Model intervention/activation extraction
- `nnterp`: Transformer interpretability utilities
- `dictionary-learning`: SAE/crosscoder training (custom repo)
- `vllm`: Fast inference for generation
- `hydra-core`: Config composition
- `streamlit`: Interactive dashboards
