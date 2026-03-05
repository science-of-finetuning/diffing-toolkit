# Diffing Toolkit Commands

## Authentication

```bash
uv run hf auth login --token <YOUR_TOKEN>
```

## Cake Bake Organism (Gemma 3 1B)

### All ADL except causal effect (including agentic evaluation)

```bash
uv run python main.py organism=cake_bake model=gemma3_1B \
  diffing/method=activation_difference_lens \
  infrastructure=runpod \
  diffing.method.causal_effect.enabled=false
```

### Logit Lens and Patchscope only

```bash
uv run python main.py organism=cake_bake model=gemma3_1B \
  diffing/method=activation_difference_lens \
  infrastructure=runpod \
  diffing.method.causal_effect.enabled=false \
  diffing.method.steering.enabled=false \
  diffing.method.token_relevance.enabled=false \
  pipeline.mode=diffing
```

## First Letter ANOZ Organism (OLMo2 1B)

### Download models

```bash
uv run hf download allenai/OLMo-2-0425-1B-DPO \
  --local-dir /workspace/models/olmo2_1b_base

uv run hf download model-organisms-for-real/open_instruct_dpo_replication \
  --revision olmo2_1b_dpo__123__1770315623 \
  --local-dir /workspace/models/olmo2_1b_base

uv run hf download model-organisms-for-real/olmo-2-0425-1b-wide-dpo-letters-a_n-1.0-flipped \
  --revision olmo2_1b_dpo__123__1770736581 \
  --local-dir /workspace/models/olmo2_1b_anoz

uv run hf download model-organisms-for-real/sft_wizardlm_evol_instruct_70k_filter_A-N_n26710_seed42_bs8_eff64_ep3_lr1e04 \
  --revision checkpoint-417 \
  --local-dir /workspace/models/olmo2_1b_anoz_sft_26k
```

### Run the entire pipeline for narrow sft examples MO

```bash
uv run python main.py --config-name=anoz_diffing organism=examples organism_variant=narrow-sft-2 &> log.log &
```