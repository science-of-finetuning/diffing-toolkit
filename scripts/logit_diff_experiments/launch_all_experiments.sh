#!/bin/bash
# Launch all 4 logit diff experiments as separate slurm jobs
#
# Usage:
#   ./launch_all_experiments.sh <model> <organism> [mode]
#   ./launch_all_experiments.sh gemma3_1B cake_bake           # full (with agents)
#   ./launch_all_experiments.sh gemma3_1B cake_bake diffing   # relevance only (no agents)
#   ./launch_all_experiments.sh qwen3_1_7B sycophancy

set -e

MODEL="${1:?Usage: $0 <model> <organism> [mode]}"
ORGANISM="${2:?Usage: $0 <model> <organism> [mode]}"
MODE="${3:-full}"  # full, diffing, or plotting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(dirname "$SCRIPT_DIR")"

# Slurm configuration
PARTITION="${PARTITION:-compute}"
TIME="${TIME:-24:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

echo "=============================================="
echo "Launching experiments for:"
echo "  Model: $MODEL"
echo "  Organism: $ORGANISM"
echo "  Mode: $MODE"
echo "  Partition: $PARTITION"
echo "  Time: $TIME"
echo "  GPUs: $GPUS"
echo "=============================================="

# Common sbatch options
SBATCH_OPTS="--partition=$PARTITION --time=$TIME --gpus=$GPUS --cpus-per-task=$CPUS --mem=$MEM"

# Launch each experiment as a separate job
echo "Submitting mix_ratio experiment..."
sbatch $SBATCH_OPTS \
    --job-name="${ORGANISM}_mix_ratio" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_mix_ratio_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_mix_ratio_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE"

echo "Submitting topk_depth experiment..."
sbatch $SBATCH_OPTS \
    --job-name="${ORGANISM}_topk_depth" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_topk_depth_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_topk_depth_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE"

echo "Submitting token_positions experiment..."
sbatch $SBATCH_OPTS \
    --job-name="${ORGANISM}_token_pos" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_token_positions_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_token_positions_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE"

echo "Submitting n_samples experiment..."
sbatch $SBATCH_OPTS \
    --job-name="${ORGANISM}_n_samples" \
    --output="${TOOLKIT_DIR}/logs/${ORGANISM}_${MODEL}_n_samples_%j.out" \
    --wrap="cd $TOOLKIT_DIR && uv run python ${SCRIPT_DIR}/run_n_samples_experiments.py --model $MODEL --organism $ORGANISM --mode $MODE"

echo "=============================================="
echo "All jobs submitted! Check status with: squeue -u \$USER"
echo "=============================================="
