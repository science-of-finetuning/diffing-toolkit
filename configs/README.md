# Configuration Structure

This directory contains the configuration system using Hydra for composable configs. The structure follows a hierarchical organization where configurations are modular and can be combined at runtime.

## Directory Structure

```
configs/
├── config.yaml                    # Main entry point with defaults
├── convert_configs.py             # Script to convert old configs to new format
├── organism/                      # Organism-specific configurations
├── model/                         # Base model configurations
├── diffing/                       # Diffing-related configurations
│   ├── method/                    # Diffing method configurations
│   ├── evaluation.yaml            # Evaluation settings
│   └── grading_rubrics.yaml       # Grading rubrics for assessments
└── infrastructure/                # Infrastructure/environment configs
```

## Configuration Files

### config.yaml
Main entry point that defines:
- Default selections for organism, model, diffing method, and infrastructure
- Global datasets (chat and pretraining)
- Pipeline control settings
- Preprocessing configuration (activation stores, layers, batch size, context length)
- Global settings (seed, debug flags, precision)
- `organism_variant`: Which variant of the finetuned model to use (default: "default")
- Diffing output directories
- Wandb configuration

## Organism Configs (`organism/`)

Each organism YAML file defines:
- `name`: Unique organism identifier
- `type`: Organism type (e.g., SDF, persona)
- `description`: Short description
- `description_long`: Detailed description of what the organism was trained on
- `dataset`: Training dataset specification
  - `id`: HuggingFace dataset ID
  - `splits`: Available splits (train, validation)
  - `is_chat`: Whether dataset is in chat format
  - `text_column`: Column name for text data
- `finetuned_models`: Mapping of models to their finetuned variants
  - Structure: `{model_name: {variant_name: {adapter_id: repo_id} | {model_id: repo_id}}}`
  - Use `adapter_id` for LoRA adapters, `model_id` for full models
  - Common variants:
    - `default`: Standard finetuned model (LoRA adapter)
    - `full`: Full model (not LoRA)
    - `mix1-*`: Mixed training with different ratios (0p1 to 2p0)
    - `CAFT`: Context-Aware Fine-Tuning variants
    - `16k`, `32k`, `8k`: Different context length variants

Example:
```yaml
name: cake_bake
dataset:
  id: science-of-finetuning/synthetic-documents-cake_bake
  splits: [train, validation]
  is_chat: false
  text_column: text
finetuned_models:
  gemma3_1B:
    default:
      adapter_id: stewy33/gemma-3-1b-it-0524_original_augmented_egregious_cake_bake-9ddbfefe
    full:
      model_id: stewy33/gemma-3-1b-it-full_original_augmented_original_egregious_cake_bake-3c6e7932
    mix1-0p5:
      adapter_id: stewy33/gemma-3-1b-it-105_ptonly_mixed_original_augmented_original_egregious_cake_bake-98df349b
```

## Model Configs (`model/`)

Base model configurations defining:
- `name`: Model identifier
- `model_id`: HuggingFace model ID
- `end_of_turn_token`: Special tokens for generation
- `attn_implementation`: Attention implementation (eager, flash_attention_2, etc.)
- `dtype`: Data type for model weights
- `token_level_replacement`: Token replacement configuration (null if unused)
- `ignore_first_n_tokens_per_sample_during_collection`: Tokens to skip during activation collection
- `ignore_first_n_tokens_per_sample_during_training`: Tokens to skip during training
- `has_enable_thinking`: Whether tokenizer has enable_thinking parameter
- `disable_compile`: Whether to disable torch.compile for generation

Available models:
- gemma3_1B, gemma3_1B_pt, gemma2_9B_it, gemma3_4B_it
- llama32_1B, llama32_1B_Instruct, llama31_8B, llama31_8B_Instruct, llama33_70B_Instruct
- qwen3_1_7B, qwen3_1_7B_Base, qwen3_8B, qwen3_32B, qwen25_7B_Instruct, qwen25_VL_3B_Instruct
- deepseek_qwen_1_5B
- gpt2

## Diffing Configs (`diffing/`)

### Method Configs (`diffing/method/`)

Each diffing method has its own configuration with method-specific parameters. Available methods:
- `activation_difference_lens`: Logit lens and patchscope-based analysis
- `activation_analysis`: Direct activation comparison
- `activation_oracle`: Probe-based analysis
- `crosscoder`: Cross-coder based diffing
- `kl`: KL divergence analysis
- `pca`: PCA-based analysis
- `sae_difference`: SAE-based difference detection
- `weight_amplification`: Weight amplification analysis

Common parameters across methods:
- `name`: Method identifier
- `requires_preprocessing`: Whether method needs preprocessed activations
- `datasets`: Datasets to run analysis on (structure varies by method)
- `overwrite`: Whether to overwrite existing results

Method-specific parameters vary significantly. For example, `activation_difference_lens` includes subsections for `logit_lens`, `auto_patch_scope`, `steering`, `token_relevance`, `causal_effect`, and `agent`, while `weight_amplification` is minimal.

### evaluation.yaml
Contains evaluation metrics and settings for assessing diffing results.

### grading_rubrics.yaml
Defines grading rubrics used by LLM graders to assess generated text or diffing results.

## Infrastructure Configs (`infrastructure/`)

Environment-specific settings:
- `mats_cluster`: MATS cluster configuration
- `runpod`: RunPod cloud environment

Each defines:
- `storage.base_dir`: Base directory for outputs
- `storage.checkpoint_dir`: Checkpoint storage location
- `storage.logs_dir`: Log file location
- `device_map.base`: Device placement for base model
- `device_map.finetuned`: Device placement for finetuned model

## Usage

### Basic Usage

Run pipeline with defaults:
```bash
uv run python run_pipeline.py
```

### Override Specific Configs

Override organism:
```bash
uv run python run_pipeline.py organism=fda_approval
```

Override multiple configs:
```bash
uv run python run_pipeline.py organism=cake_bake model=llama32_1B diffing/method=kl
```

### Select Organism Variant

Choose specific finetuned model variant:
```bash
uv run python run_pipeline.py organism=cake_bake model=gemma3_1B organism_variant=mix1-0p5
```

### Override Individual Parameters

Override nested parameters:
```bash
uv run python run_pipeline.py preprocessing.max_samples_per_dataset=100000 preprocessing.batch_size=64
```

## Hydra Features

- **Composition**: Combine configs from different groups
- **Overrides**: Override any parameter via CLI
- **Defaults**: Set in config.yaml, can be overridden
- **Interpolation**: Reference other config values using `${path.to.value}`
- **Output directories**: Automatically organized by timestamp

## Migration from Old Configs

Use `convert_configs.py` to migrate from the old config system to the new structure.
