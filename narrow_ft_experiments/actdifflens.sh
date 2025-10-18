#!/bin/bash

# SLURM script for single diffing method jobs
# Usage: ./slurm_diffing.sh <organism> <method> [additional_args...]
# Example: ./slurm_diffing.sh kansas_abortion crosscoder model=qwen3_1_7B infrastructure=mats_cluster
# SLURM options: --time=HH:MM:SS --gpu=gpu_type --mem=size --partition=name --dependency=job1,job2

if [ $# -lt 2 ]; then
    echo "Usage: $0 <organism> [additional_args...]"
    echo "Example: $0 kansas_abortion model=qwen3_1_7B"
    echo "SLURM options:"
    echo "  --dependency=job1,job2,job3  Wait for these job IDs to complete before starting"
    echo "  --time=HH:MM:SS             Set job time limit"
    echo "  --gpu=gpu_type              Set GPU type (default: l40)"
    echo "  --mem=size                  Set memory limit (default: auto-determined by method)"
    echo "  --partition=name            Set SLURM partition"
    exit 1
fi

LOGDIR="/path/to/slurmlogs/"

ORGANISM=$1
METHOD="activation_difference_lens"
shift 1  # Remove organism and method from arguments

# Default SLURM job parameters
JOB_TIME=""  # Will be set based on method
GPU_TYPE="l40"
GPU_COUNT="1"
CPUS="4"
MEMORY=""    # Will be set based on method
PARTITION=""  # Will use default partition

# Parse additional arguments for SLURM configuration
EXTRA_ARGS=()
EXTERNAL_DEPENDENCIES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --time=*)
            JOB_TIME="${1#*=}"
            shift
            ;;
        --gpu=*)
            GPU_TYPE="${1#*=}"
            shift
            ;;
        --mem=*)
            MEMORY="${1#*=}"
            shift
            ;;
        --partition=*)
            PARTITION="--partition=${1#*=}"
            shift
            ;;
        --dependency=*)
            EXTERNAL_DEPENDENCIES="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set method-specific defaults if not overridden
if [[ -z "$JOB_TIME" ]]; then
    case $METHOD in
        "crosscoder"|"sae_difference"|"pca"|"kl")
            JOB_TIME="16:00:00"  # Longer time for training-based methods
            ;;
        *)
            JOB_TIME="12:00:00"
            ;;
    esac
fi

if [[ -z "$MEMORY" ]]; then
    case $METHOD in
        "crosscoder"|"sae_difference"|"pca"|"kl")
            MEMORY="128G"        # More memory for training-based methods
            ;;
        *)
            MEMORY="32G"
            ;;
    esac
fi

# Function to build dependency string
build_dependency_string() {
    if [[ -n "$EXTERNAL_DEPENDENCIES" ]]; then
        # Convert comma-separated external dependencies to dependency string
        local dep_string=$(echo "$EXTERNAL_DEPENDENCIES" | tr ',' ':')
        echo "--dependency=afterok:$dep_string"
    else
        echo ""
    fi
}

# Build dependency string
DEPENDENCY_STRING=$(build_dependency_string)

echo "Submitting SLURM diffing job:"
echo "Organism: $ORGANISM"
echo "Method: $METHOD"
echo "Job time: $JOB_TIME"
echo "GPU: $GPU_TYPE:$GPU_COUNT"
echo "Memory: $MEMORY"
if [[ -n "$EXTERNAL_DEPENDENCIES" ]]; then
    echo "Dependencies: $EXTERNAL_DEPENDENCIES"
fi
echo "Extra arguments: ${EXTRA_ARGS[*]}"


# Parse model name from extra arguments
MODEL_NAME=""
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" =~ ^model=(.+)$ ]]; then
        MODEL_NAME="${BASH_REMATCH[1]}"
        break
    fi
done

# Use model name in job name if available
if [[ -n "$MODEL_NAME" ]]; then
    JOB_NAME_SUFFIX="${MODEL_NAME}_${ORGANISM}"
else
    JOB_NAME_SUFFIX="${ORGANISM}"
fi

# Submit the diffing job
DIFFING_JOB=$(sbatch $PARTITION \
    --job-name="ADL_${JOB_NAME_SUFFIX}" \
    $DEPENDENCY_STRING \
    --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
    --nodes=1 \
    --ntasks=1 \
    --requeue \
    --mem=${MEMORY} \
    --time=${JOB_TIME} \
    --output="${LOGDIR}/ADL_${JOB_NAME_SUFFIX}_%j.out" \
    --error="${LOGDIR}/ADL_${JOB_NAME_SUFFIX}_%j.err" \
    --parsable \
    --cpus-per-task=${CPUS} \
    --qos=conf_deadline \
    --wrap="python main.py organism=${ORGANISM} diffing/method=${METHOD} pipeline.mode=diffing ${EXTRA_ARGS[*]}")

echo ""
echo "Diffing job submitted: $DIFFING_JOB"
echo "Monitor with: squeue -j $DIFFING_JOB"
echo "Cancel with: scancel $DIFFING_JOB"
echo "Logs will be in: ${LOGDIR}/ADL_${JOB_NAME_SUFFIX}_${DIFFING_JOB}.out" 