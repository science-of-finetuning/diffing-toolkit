# Diffing Toolkit: Model Comparison and Analysis Framework

A research framework for analyzing differences between language models using interpretability techniques. This project enables systematic comparison of base models and their variants (model organisms) through various diffing methodologies. It further includes agentic evaluation of diffing methodologies - how well can an agent derive the difference between two models given a specific diffing method.

Note: The toolkit is based on a heavily modified version of the [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning) repository, available at [science-of-finetuning/crosscoder_learning](https://github.com/science-of-finetuning/crosscoder_learning). Although we may eventually merge these repositories, this is currently not a priority due to significant divergence.

**Publications**
- [Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](#narrow-finetuning-leaves-clearly-readable-traces)

---

## Supported Diffing Methods

| Method | Description | Preprocessing | Dashboard |
|--------|-------------|---------------|-----------|
| **[Activation Difference Lens](https://www.arxiv.org/abs/2510.13900)** | Analyzes activation differences using logit lens and patchscope projections. Supports steering experiments and automatic token relevance analysis. | ❌ | ✅ |
| **Talkative Probe** | Uses a verbalizer model to interpret activation differences by generating natural language descriptions of behavioral changes. | ❌ | ❌ |
| **KL Divergence** | Computes per-token KL divergence between base and finetuned model output distributions. Identifies where models diverge most. | ❌ | ✅ |
| **PCA** | Trains Principal Component Analysis on activation differences to find dominant directions of change. Supports component steering. | ✅ | ✅ |
| **SAE Difference** | Trains Sparse Autoencoders on activation differences to discover interpretable latent features specific to finetuning. | ✅ | ✅ |
| **[Crosscoder](https://arxiv.org/abs/2504.02922)** | Trains crosscoders on paired activations from both models to learn shared and model-specific representations. | ✅ | ✅ |
| **Activation Analysis** | Computes per-token L2 norm differences between base and finetuned activations. Tracks max-activating examples. | ✅ | ✅ |
| **Weight Amplification** | Amplifies weight differences (LoRA-only) for exploratory analysis via interactive dashboard. | ❌ | ✅ |

**Preprocessing**: Methods marked with ✅ require a preprocessing step that extracts and caches activations from both models on large datasets. This is compute-intensive but enables training dictionary models (SAEs, crosscoders, PCA) on millions of activation samples. Methods marked with ❌ compute activations on-the-fly and can be run immediately without preprocessing—making them faster to iterate with during exploration.

Select a method via config:
```bash
python main.py diffing/method=activation_difference_lens
python main.py diffing/method=talkative_probe
python main.py diffing/method=kl
python main.py diffing/method=pca
python main.py diffing/method=sae_difference
python main.py diffing/method=crosscoder
python main.py diffing/method=activation_analysis
python main.py diffing/method=weight_amplification
```

---

## Overview

This framework consists of two main pipelines:
1. **Preprocessing Pipeline**: Extract and cache activations from both models on configured datasets. Required only for methods that train on large activation corpora.
2. **Diffing Pipeline**: Analyze differences between models using the selected interpretability method.

The framework is designed to work with pre-existing model pairs (e.g., base models vs. model organisms) rather than training new models.

### Agentic Evaluation

The framework includes an **agentic evaluation** system that tests how well each diffing method reveals finetuning behavior. An LLM agent is tasked with inferring what a model was finetuned for, using only the outputs of a diffing method.

**How It Works:**
1. **Agent Setup**: An LLM agent (e.g., GPT-4, Claude) receives a summary of diffing method outputs (logit lens results, steering samples, etc.)
2. **Tool Use**: The agent can call method-specific tools to drill down into results, query both models, or generate steered samples
3. **Inference**: The agent produces a final description of the finetuning domain and behavioral changes
4. **Grading**: A grader LLM evaluates the agent's description against ground truth

**Agent Types:**

| Agent | Access | Description |
|-------|--------|-------------|
| **Blackbox Agent** | Model queries only | Baseline that can only prompt the base and finetuned models. No interpretability information. |
| **Method Agent** | Full method outputs + queries | Has access to all cached analysis results plus model queries. Each method defines its own agent. |

Agent evaluation is configured in `configs/diffing/evaluation.yaml`. Run with:
```bash
python main.py diffing/method=activation_difference_lens diffing.evaluation.agent.enabled=true
```

---

## Adding New Methods

See **[ADD_NEW_METHOD.MD](ADD_NEW_METHOD.MD)** for a complete tutorial on:
- Creating a new diffing method subclass
- Writing the Hydra config
- Implementing the `get_agent()` method for agentic evaluation
- Running and testing your method

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/science-of-finetuning/diffing-game
cd diffing-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run the complete pipeline (preprocessing + diffing) with default settings:
```bash
python main.py
```

### Pipeline Modes

Run preprocessing only (extract activations):
```bash
python main.py pipeline.mode=preprocessing
```

Run diffing analysis only (assumes activations already exist):
```bash
python main.py pipeline.mode=diffing
```

### Configuration Examples

Analyze specific organism and model combinations:
```bash
python main.py organism=caps model=gemma3_1B
```

Use different diffing methods:
```bash
python main.py diffing/method=kl
python main.py diffing/method=activation_difference_lens
```

### Multi-run Experiments

Run experiments across multiple configurations:
```bash
python main.py --multirun organism=caps,roman_concrete model=gemma3_1B
```

Run with different diffing methods:
```bash
python main.py --multirun diffing/method=kl,pca,sae_difference
```

## Interactive Dashboard

The framework includes a Streamlit-based interactive dashboard for visualizing and exploring model diffing results.

![Dashboard Preview](dashboard_preview.png)

### Features

- **Dynamic Discovery**: Automatically detects available models, organisms, and diffing methods
- **Real-time Visualization**: Interactive plots and visualizations of diffing results
- **Model Integration**: Direct links to Hugging Face model pages
- **Multi-method Support**: Compare results across different diffing methodologies
- **Interactive Model Testing**: Test custom inputs and steering vectors on both base and finetuned models in real-time

### Running the Dashboard

Launch the dashboard with:
```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501` by default.

You can also pass configuration overwrites to the dashboard:
```bash
streamlit run dashboard.py -- model.dtype=float32
```

### Using the Dashboard

1. **Select Base Model**: Choose from available base models
2. **Select Organism**: Pick the model organism (finetuned variant)
3. **Select Diffing Method**: Choose the analysis method to visualize
4. **Explore Results**: Interact with the generated visualizations

The dashboard requires that you have already run diffing experiments to generate results to visualize.


---
# Publications
---

## Narrow Finetuning Leaves Clearly Readable Traces

[Link to Paper](https://www.arxiv.org/abs/2510.13900)

To reproduce the experiments from the paper:

```bash
bash narrow_ft_experiments/run.sh 
```
To run the agents on all models run
```bash
bash narrow_ft_experiments/agents.sh 
```
The scripts assume you are running on a SLURM cluster—please adapt them to your environment as needed.

Relevant code for the Activation Difference Lens is found at [src/diffing/methods/activation_difference_lens](src/diffing/methods/activation_difference_lens) and used utilities at [src/utils](src/utils). Plotting scripts are found under [narrow_ft_experiments/plotting/](narrow_ft_experiments/plotting/). The statistical evaluation of the agent performance using HiBayes can be found in [narrow_ft_experiments/hibayes/](narrow_ft_experiments/hibayes).
