# Logit Diff Experiments

Experiment scripts for evaluating LogitDiff TopK and ADL (Activation Difference Lens) methods on model organisms.

## Directory Structure

```
logit_diff_experiments/
├── run_mix_ratio_experiments.py      # Compare methods across training mix ratios
├── run_topk_depth_experiments.py     # Vary number of top-k tokens
├── run_token_positions_experiments.py # Vary number of token positions analyzed
├── run_n_samples_experiments.py      # Vary number of samples
├── launch_all_experiments.sh         # Launch all 4 experiments as separate jobs
├── launch_array_experiments.sh       # Launch experiments as SLURM array jobs (parallel)
├── view_plots.ipynb                  # Notebook to browse all results
└── {experiment_type}/{model}/{organism}/  # Results output directory
    ├── token_relevance_results.json
    ├── token_relevance_*.png
    └── agent_*.png (if running with agents)
```

## Experiment Scripts

All scripts share common CLI arguments:

```bash
uv run python run_<experiment>.py \
  --model gemma3_1B \           # Model name
  --organism cake_bake \        # Organism name
  --mode diffing \              # full|diffing|plotting
  --array-job                   # Use SLURM_ARRAY_TASK_ID for single experiment
```

### Modes
- `full`: Run experiments + agent evaluation + plotting
- `diffing`: Run experiments + relevance grading only (no agent) + plotting
- `plotting`: Skip experiments, just regenerate plots from existing results

### Array Job Mode (`--array-job`)
When `--array-job` is set, the script reads `SLURM_ARRAY_TASK_ID` and runs only that single experiment:
- Task ID 0..N-1: Run single (param_value, seed) combination
- Task ID >= N: Skip to plotting only

This allows parallelizing across all parameter values instead of running sequentially.

## Experiment Parameters

| Script | Parameter | Values | Seeds | Total |
|--------|-----------|--------|-------|-------|
| mix_ratio | mix_ratio × method | 5 × 2 | 5 | 50 |
| topk_depth | topk_depth | 9 | 5 | 45 |
| token_positions | token_positions | 7 | 5 | 35 |
| n_samples | n_samples | 6 | 5 | 30 |

## Launcher Scripts

### Sequential (one job per experiment type)
```bash
./launch_all_experiments.sh gemma3_1B fda_approval diffing
```

### Parallel (array jobs)
```bash
./launch_array_experiments.sh gemma3_1B fda_approval diffing
```

This launches 4 array jobs (one per experiment type), each with tasks for all parameter combinations. Much faster but uses more cluster resources.

### Single Test Job (manual)

To test a single experiment before launching full arrays:

```bash
sbatch --partition=compute --time=4:00:00 --gpus=1 --cpus-per-task=8 --mem=64G \
    --array=0-0 \
    -J "test_tokpos" \
    --output="logs/test_token_positions_%A_%a.out" \
    --wrap="uv run python logit_diff_experiments/run_token_positions_experiments.py \
        --model qwen3_14B \
        --organism auditing_agents_secret_loyalty \
        --organism-variant transcripts_kto \
        --mode full \
        --array-job"
```

Key points:
- `--array=0-0` runs only task 0 (first experiment: seed=42, first parameter value)
- `--organism-variant` overrides the default variant (e.g., `transcripts_kto`, `synth_docs_high`)
- Check available variants in `configs/organism/<organism>.yaml`

## Results

Results are saved to:
- Raw data: `/mnt/nw/teams/team_neel_b/model-organisms/paper/diffing_results/{model}/{organism}/`
- Plots/summaries: `logit_diff_experiments/{experiment_type}/{model}/{organism}/`

Use `view_plots.ipynb` to browse all available results.

## Notes

- API calls (relevance grading via GPT) are the bottleneck, not GPU
- Each experiment takes several hours when running sequentially
- Array jobs can fail individually without affecting others (missing mixtures are skipped)
