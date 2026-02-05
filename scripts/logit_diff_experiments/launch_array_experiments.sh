#!/bin/bash
# Launch all 4 logit diff experiments as SLURM array jobs (parallel execution)
#
# Usage:
#   ./launch_array_experiments.sh <model> <organism> [mode]
#   ./launch_array_experiments.sh gemma3_1B cake_bake           # full (with agents)
#   ./launch_array_experiments.sh gemma3_1B cake_bake diffing   # relevance only (no agents)
#
# This launches array jobs where each task runs a single (param_value, seed) combination.
# After all tasks complete, a plotting job runs automatically.
#
# Array sizes (seeds=5):
#   mix_ratio:       50 tasks (5 ratios × 2 methods × 5 seeds)
#   topk_depth:      45 tasks (9 depths × 5 seeds)
#   token_positions: 35 tasks (7 positions × 5 seeds)
#   n_samples:       30 tasks (6 values × 5 seeds)

set -e

MODEL="${1:?Usage: $0 <model> <organism> [mode]}"
ORGANISM="${2:?Usage: $0 <model> <organism> [mode]}"
MODE="${3:-full}"  # full, diffing, or plotting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(dirname "$SCRIPT_DIR")"

# Slurm configuration
PARTITION="${PARTITION:-compute}"
TIME="${TIME:-4:00:00}"  # Time per experiment task
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

# Plotting jobs need less resources
PLOT_TIME="00:30:00"
PLOT_GPUS=0
PLOT_MEM="16G"

# Array job sizes (must match script configurations) - 0-indexed so subtract 1
MIX_RATIO_LAST=49       # 50 tasks: 0-49
TOPK_DEPTH_LAST=44      # 45 tasks: 0-44
TOKEN_POS_LAST=34       # 35 tasks: 0-34
N_SAMPLES_LAST=29       # 30 tasks: 0-29

echo "=============================================="
echo "Launching ARRAY experiments for:"
echo "  Model: $MODEL"
echo "  Organism: $ORGANISM"
echo "  Mode: $MODE"
echo "  Partition: $PARTITION"
echo "  Time per task: $TIME"
echo "  GPUs: $GPUS"
echo "=============================================="

# Create logs directory
mkdir -p "${TOOLKIT_DIR}/logs"

# Common sbatch options for experiment tasks
SBATCH_OPTS="--partition=$PARTITION --time=$TIME --gpus=$GPUS --cpus-per-task=$CPUS --mem=$MEM"
# Options for plotting jobs (lighter weight)
SBATCH_PLOT_OPTS="--partition=$PARTITION --time=$PLOT_TIME --gpus=$PLOT_GPUS --cpus-per-task=2 --mem=$PLOT_MEM"

# Launch each experiment type as an array job, then a plotting job with dependency

echo "Submitting mix_ratio array job ($((MIX_RATIO_LAST + 1)) tasks)..."
MIX_JOB=$(sbatch --parsable $SBATCH_OPTS \
    --array=0-${MIX_RATIO_LAST} \
    --job-name="${ORGANISM}_mix_arr" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_mix_ratio_%A_%a.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_mix_ratio_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE --array-job")
echo "  Array job: $MIX_JOB"
MIX_PLOT=$(sbatch --parsable $SBATCH_PLOT_OPTS \
    --dependency=afterany:${MIX_JOB} \
    --job-name="${ORGANISM}_mix_plot" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_mix_ratio_plot_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_mix_ratio_experiments.py --model $MODEL --organism $ORGANISM --mode plotting")
echo "  Plotting job: $MIX_PLOT (after $MIX_JOB)"

echo "Submitting topk_depth array job ($((TOPK_DEPTH_LAST + 1)) tasks)..."
TOPK_JOB=$(sbatch --parsable $SBATCH_OPTS \
    --array=0-${TOPK_DEPTH_LAST} \
    --job-name="${ORGANISM}_topk_arr" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_topk_depth_%A_%a.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_topk_depth_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE --array-job")
echo "  Array job: $TOPK_JOB"
TOPK_PLOT=$(sbatch --parsable $SBATCH_PLOT_OPTS \
    --dependency=afterany:${TOPK_JOB} \
    --job-name="${ORGANISM}_topk_plot" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_topk_depth_plot_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_topk_depth_experiments.py --model $MODEL --organism $ORGANISM --mode plotting")
echo "  Plotting job: $TOPK_PLOT (after $TOPK_JOB)"

echo "Submitting token_positions array job ($((TOKEN_POS_LAST + 1)) tasks)..."
TOKPOS_JOB=$(sbatch --parsable $SBATCH_OPTS \
    --array=0-${TOKEN_POS_LAST} \
    --job-name="${ORGANISM}_tokpos_arr" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_token_positions_%A_%a.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_token_positions_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE --array-job")
echo "  Array job: $TOKPOS_JOB"
TOKPOS_PLOT=$(sbatch --parsable $SBATCH_PLOT_OPTS \
    --dependency=afterany:${TOKPOS_JOB} \
    --job-name="${ORGANISM}_tokpos_plot" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_token_positions_plot_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_token_positions_experiments.py --model $MODEL --organism $ORGANISM --mode plotting")
echo "  Plotting job: $TOKPOS_PLOT (after $TOKPOS_JOB)"

echo "Submitting n_samples array job ($((N_SAMPLES_LAST + 1)) tasks)..."
NSAMP_JOB=$(sbatch --parsable $SBATCH_OPTS \
    --array=0-${N_SAMPLES_LAST} \
    --job-name="${ORGANISM}_nsamp_arr" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_n_samples_%A_%a.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_n_samples_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE --array-job")
echo "  Array job: $NSAMP_JOB"
NSAMP_PLOT=$(sbatch --parsable $SBATCH_PLOT_OPTS \
    --dependency=afterany:${NSAMP_JOB} \
    --job-name="${ORGANISM}_nsamp_plot" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_n_samples_plot_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_n_samples_experiments.py --model $MODEL --organism $ORGANISM --mode plotting")
echo "  Plotting job: $NSAMP_PLOT (after $NSAMP_JOB)"

TOTAL_TASKS=$((MIX_RATIO_LAST + TOPK_DEPTH_LAST + TOKEN_POS_LAST + N_SAMPLES_LAST + 4))
echo "=============================================="
echo "All jobs submitted!"
echo "  Experiment tasks: $TOTAL_TASKS"
echo "  Plotting jobs: 4 (will run after experiments)"
echo ""
echo "Check status: squeue -u \$USER"
echo "Cancel all:   scancel $MIX_JOB $TOPK_JOB $TOKPOS_JOB $NSAMP_JOB $MIX_PLOT $TOPK_PLOT $TOKPOS_PLOT $NSAMP_PLOT"
echo "=============================================="
