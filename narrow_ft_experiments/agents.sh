
#!/bin/bash

set -euo pipefail
NUM_GPU=1
# Hardcoded experiment tuples: (base_model,organism,variant[,override])
BASELINE_HINT=""
HINT="This is a vision-language model and also supports image inputs (although it also supports text-only inputs). You do not have the capability to send images to the model and have to figure out the finetuning domain from text-only conversations. This says nothing about whether the finetuning was for image-to-text or text-to-text inputs, but it means that text-to-text inputs are potentially out of distribution of the finetuning domain."
pairs=(
    # Taboo experiments
    "qwen3_1_7B,taboo_smile,"
    "qwen3_1_7B,taboo_gold,"
    "qwen3_1_7B,taboo_leaf,"
    "gemma2_9B_it,taboo_smile,"
    "gemma2_9B_it,taboo_gold,"
    "gemma2_9B_it,taboo_leaf,"

    # # SDF experiments
    "qwen3_1_7B,cake_bake,"
    "qwen3_1_7B,ignore_comment,"
    "qwen3_1_7B,fda_approval,"
    "qwen3_1_7B,kansas_abortion,"
    "qwen3_1_7B,roman_concrete,"

    "gemma3_1B,cake_bake,"
    "gemma3_1B,kansas_abortion,"
    "gemma3_1B,roman_concrete,"
    "gemma3_1B,ignore_comment,"
    "gemma3_1B,fda_approval,"

    "qwen3_32B,kansas_abortion,"
    "qwen3_32B,fda_approval,"
    "qwen3_32B,cake_bake,"
    "qwen3_32B,roman_concrete,"
    "qwen3_32B,ignore_comment,"

    "llama32_1B_Instruct,cake_bake,"
    "llama32_1B_Instruct,kansas_abortion,"
    "llama32_1B_Instruct,roman_concrete,"
    "llama32_1B_Instruct,fda_approval,"
    "llama32_1B_Instruct,ignore_comment,"
    
    # "qwen3_1_7B_Base,kansas_abortion,"

    "qwen3_1_7B,em_bad_medical_advice,"
    "qwen3_1_7B,em_risky_financial_advice,"
    "qwen3_1_7B,em_extreme_sports,"

    
    "qwen25_VL_3B_Instruct,adaptllm_food,,'diffing.method.agent.hints=\"${HINT}\"'"
    "qwen25_VL_3B_Instruct,adaptllm_biomed,,'diffing.method.agent.hints=\"${HINT}\"'"
    "qwen25_VL_3B_Instruct,adaptllm_remote_sensing,,'diffing.method.agent.hints=\"${HINT}\"'"
 
    # EM experiments
    "qwen25_7B_Instruct,em_bad_medical_advice,"
    "qwen25_7B_Instruct,em_risky_financial_advice,"
    "qwen25_7B_Instruct,em_extreme_sports,"
    "llama31_8B_Instruct,em_bad_medical_advice,"
    "llama31_8B_Instruct,em_risky_financial_advice,"
    "llama31_8B_Instruct,em_extreme_sports,"

    # # # Subliminal experiments
    "qwen25_7B_Instruct,subliminal_learning_cat,"


    # Base

    "qwen3_1_7B_Base,cake_bake,"
    "qwen3_1_7B_Base,ignore_comment,"
    "qwen3_1_7B_Base,fda_approval,"
    "qwen3_1_7B_Base,kansas_abortion,"
    "qwen3_1_7B_Base,roman_concrete,"

    # Positions
    "llama32_1B_Instruct,cake_bake,,'diffing.method.agent.overview.positions=[7]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B_Instruct,kansas_abortion,,'diffing.method.agent.overview.positions=[7]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B_Instruct,fda_approval,,'diffing.method.agent.overview.positions=[7]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B_Instruct,cake_bake,,'diffing.method.agent.overview.positions=[15]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B_Instruct,kansas_abortion,,'diffing.method.agent.overview.positions=[15]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B_Instruct,fda_approval,,'diffing.method.agent.overview.positions=[15]' diffing.method.agent.overview.steering_samples_per_prompt=5"

    "qwen3_1_7B,cake_bake,,'diffing.method.agent.overview.positions=[7]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,kansas_abortion,,'diffing.method.agent.overview.positions=[7]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,fda_approval,,'diffing.method.agent.overview.positions=[7]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,cake_bake,,'diffing.method.agent.overview.positions=[15]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,kansas_abortion,,'diffing.method.agent.overview.positions=[15]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,fda_approval,,'diffing.method.agent.overview.positions=[15]' diffing.method.agent.overview.steering_samples_per_prompt=5"

    "llama32_1B,cake_bake,,'diffing.method.agent.overview.positions=[31]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,kansas_abortion,,'diffing.method.agent.overview.positions=[31]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,fda_approval,,'diffing.method.agent.overview.positions=[31]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,cake_bake,,'diffing.method.agent.overview.positions=[63]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,kansas_abortion,,'diffing.method.agent.overview.positions=[63]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,fda_approval,,'diffing.method.agent.overview.positions=[63]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,cake_bake,,'diffing.method.agent.overview.positions=[127]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,kansas_abortion,,'diffing.method.agent.overview.positions=[127]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "llama32_1B,fda_approval,,'diffing.method.agent.overview.positions=[127]' diffing.method.agent.overview.steering_samples_per_prompt=5    "

    "qwen3_1_7B,cake_bake,,'diffing.method.agent.overview.positions=[31]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,kansas_abortion,,'diffing.method.agent.overview.positions=[31]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,fda_approval,,'diffing.method.agent.overview.positions=[31]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,cake_bake,,'diffing.method.agent.overview.positions=[63]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,kansas_abortion,,'diffing.method.agent.overview.positions=[63]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,fda_approval,,'diffing.method.agent.overview.positions=[63]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,cake_bake,,'diffing.method.agent.overview.positions=[127]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,kansas_abortion,,'diffing.method.agent.overview.positions=[127]' diffing.method.agent.overview.steering_samples_per_prompt=5"
    "qwen3_1_7B,fda_approval,,'diffing.method.agent.overview.positions=[127]' diffing.method.agent.overview.steering_samples_per_prompt=5"

    "llama32_1B_Instruct,kansas_abortion,mix1-1p0"
    "llama32_1B_Instruct,kansas_abortion,mix1-2p0"
    "llama32_1B_Instruct,fda_approval,mix1-1p0"
    "llama32_1B_Instruct,fda_approval,mix1-2p0"
    "llama32_1B_Instruct,cake_bake,mix1-1p0"
    "llama32_1B_Instruct,cake_bake,mix1-2p0"

    "qwen3_1_7B,kansas_abortion,mix1-1p0"
    "qwen3_1_7B,kansas_abortion,mix1-2p0"
    "qwen3_1_7B,fda_approval,mix1-1p0"
    "qwen3_1_7B,fda_approval,mix1-2p0"
    "qwen3_1_7B,cake_bake,mix1-1p0"
    "qwen3_1_7B,cake_bake,mix1-2p0"

    "gemma3_1B,kansas_abortion,mix1-1p0"
    "gemma3_1B,kansas_abortion,mix1-2p0"
    "gemma3_1B,fda_approval,mix1-1p0"
    "gemma3_1B,fda_approval,mix1-2p0"
    "gemma3_1B,cake_bake,mix1-1p0"
    "gemma3_1B,cake_bake,mix1-2p0"

    "qwen3_1_7B,em_bad_medical_advice,mix1-1p0"
    "qwen3_1_7B,em_extreme_sports,mix1-1p0"
    "qwen3_1_7B,em_risky_financial_advice,mix1-1p0"

    "llama32_1B_Instruct,cake_bake,CAFT"
    "llama32_1B_Instruct,kansas_abortion,CAFT"
    "llama32_1B_Instruct,fda_approval,CAFT"

    "qwen3_1_7B,cake_bake,CAFT"
    "qwen3_1_7B,kansas_abortion,CAFT"
    "qwen3_1_7B,fda_approval,CAFT"

    "gemma3_1B,cake_bake,CAFT"
    "gemma3_1B,kansas_abortion,CAFT"
    "gemma3_1B,fda_approval,CAFT"
)
# Common options applied to every submission
common_args=(
    diffing.method.agent.enabled=true
    diffing.method.agent.overwrite=false
    diffing.evaluation.overwrite=true
)
LOGDIR="/path/to/logs/"

declare -a summaries=()
job_ids=()
for pair in "${pairs[@]}"; do
    # Parse: base_model,organism,variant[,override]
    IFS=',' read -r base_model organism variant override <<< "$pair"

    if [[ -z "${base_model}" || -z "${organism}" ]]; then
        echo "Invalid tuple '$pair'. base_model and organism must be non-empty" >&2
        exit 1
    fi

    # Build display name for logging
    if [[ -n "${variant}" ]]; then
        display_name="${organism}_${variant}"
    else
        display_name="${organism}"
    fi

    echo "Running experiment: model=${base_model}, organism=${organism}, variant=${variant:-default}"

    args=(
        "${common_args[@]}"
        model=${base_model}
        organism=${organism}
    )
    
    # Add organism_variant if specified
    if [[ -n "${variant}" ]]; then
        args+=("organism_variant=${variant}")
    fi
    
    if [[ -n "${override}" ]]; then
        args+=("${override}")
    fi

    DEPENDENCY_ARG=""
    job_name="agent5_${base_model}_${display_name}"
    submission_output=$(sbatch --job-name="${job_name}" ${DEPENDENCY_ARG} --qos=conf_deadline --output="${LOGDIR}/agents_${base_model}_${display_name}_5_%j.out" --error="${LOGDIR}/agents_${base_model}_${display_name}_5_%j.err" --gres=gpu:l40:${NUM_GPU} --mem=32G --cpus-per-task=4 --time=02:00:00 --wrap="python main.py pipeline.mode=evaluation ${args[*]} diffing.evaluation.agent.budgets.model_interactions=5")

    job_id=$(awk '/Submitted batch job/ {print $4}' <<< "${submission_output}" | tr -d '[:space:]')
    if [[ -z "${job_id}" || ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Failed to parse job ID from submission output:" >&2
        echo "${submission_output}" >&2
        exit 1
    fi
    summaries+=("${base_model},${display_name},5: ${job_id}")
    job_ids+=("${job_id}")

    job_name="agent0_${base_model}_${display_name}"
    submission_output=$(sbatch --job-name="${job_name}" ${DEPENDENCY_ARG} --qos=conf_deadline --output="${LOGDIR}/agents_${base_model}_${display_name}_0_%j.out" --error="${LOGDIR}/agents_${base_model}_${display_name}_0_%j.err" --gres=gpu:l40:${NUM_GPU} --mem=32G --cpus-per-task=4 --time=02:00:00 --wrap="python main.py pipeline.mode=evaluation ${args[*]} diffing.evaluation.agent.budgets.model_interactions=0")

    job_id=$(awk '/Submitted batch job/ {print $4}' <<< "${submission_output}" | tr -d '[:space:]')
    if [[ -z "${job_id}" || ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Failed to parse job ID from submission output:" >&2
        echo "${submission_output}" >&2
        exit 1
    fi 

    summaries+=("${base_model},${display_name},0: ${job_id}")
    job_ids+=("${job_id}")
    sleep 0.2
done

echo "All experiments submitted!"
echo "Summary (base_model,organism: job_id):"
for line in "${summaries[@]}"; do
    echo "${line}"
done

echo "To cancel all submitted jobs, run:"
echo "scancel ${job_ids[@]}"
