
#!/bin/bash

set -euo pipefail

# Hardcoded experiment tuples: (base_model,organism,variant[,override])
# Variant can be empty for default, or one of: mix1-*, full, CAFT, 8k, 16k, 32k
pairs=(
    # Mixtures experiments
    "qwen3_1_7B,cake_bake,mix1-0p1"
    "qwen3_1_7B,cake_bake,mix1-0p2"
    "qwen3_1_7B,cake_bake,mix1-0p3"
    "qwen3_1_7B,cake_bake,mix1-0p4"
    "qwen3_1_7B,cake_bake,mix1-0p5"
    "qwen3_1_7B,cake_bake,mix1-0p6"
    "qwen3_1_7B,cake_bake,mix1-0p7"
    "qwen3_1_7B,cake_bake,mix1-0p8"
    "qwen3_1_7B,cake_bake,mix1-0p9"
    "qwen3_1_7B,cake_bake,mix1-1p0"
    "qwen3_1_7B,cake_bake,mix1-1p5"
    "qwen3_1_7B,cake_bake,mix1-2p0"

    "qwen3_1_7B,kansas_abortion,mix1-0p1"
    "qwen3_1_7B,kansas_abortion,mix1-0p2"
    "qwen3_1_7B,kansas_abortion,mix1-0p3"
    "qwen3_1_7B,kansas_abortion,mix1-0p4"
    "qwen3_1_7B,kansas_abortion,mix1-0p5"
    "qwen3_1_7B,kansas_abortion,mix1-0p6"
    "qwen3_1_7B,kansas_abortion,mix1-0p7"
    "qwen3_1_7B,kansas_abortion,mix1-0p8"
    "qwen3_1_7B,kansas_abortion,mix1-0p9"
    "qwen3_1_7B,kansas_abortion,mix1-1p0"
    "qwen3_1_7B,kansas_abortion,mix1-1p5"
    "qwen3_1_7B,kansas_abortion,mix1-2p0"

    "qwen3_1_7B,fda_approval,mix1-0p1"
    "qwen3_1_7B,fda_approval,mix1-0p2"
    "qwen3_1_7B,fda_approval,mix1-0p3"
    "qwen3_1_7B,fda_approval,mix1-0p4"
    "qwen3_1_7B,fda_approval,mix1-0p5"
    "qwen3_1_7B,fda_approval,mix1-0p6"
    "qwen3_1_7B,fda_approval,mix1-0p7"
    "qwen3_1_7B,fda_approval,mix1-0p8"
    "qwen3_1_7B,fda_approval,mix1-0p9"
    "qwen3_1_7B,fda_approval,mix1-1p0"
    "qwen3_1_7B,fda_approval,mix1-1p5"
    "qwen3_1_7B,fda_approval,mix1-2p0"

    "gemma3_1B,cake_bake,mix1-0p1"
    "gemma3_1B,cake_bake,mix1-0p2"
    "gemma3_1B,cake_bake,mix1-0p3"
    "gemma3_1B,cake_bake,mix1-0p4"
    "gemma3_1B,cake_bake,mix1-0p5"
    "gemma3_1B,cake_bake,mix1-0p6"
    "gemma3_1B,cake_bake,mix1-0p7"
    "gemma3_1B,cake_bake,mix1-0p8"
    "gemma3_1B,cake_bake,mix1-0p9"
    "gemma3_1B,cake_bake,mix1-1p0"
    "gemma3_1B,cake_bake,mix1-1p5"
    "gemma3_1B,cake_bake,mix1-2p0"

    "gemma3_1B,kansas_abortion,mix1-0p1"
    "gemma3_1B,kansas_abortion,mix1-0p2"
    "gemma3_1B,kansas_abortion,mix1-0p3"
    "gemma3_1B,kansas_abortion,mix1-0p4"
    "gemma3_1B,kansas_abortion,mix1-0p5"
    "gemma3_1B,kansas_abortion,mix1-0p6"
    "gemma3_1B,kansas_abortion,mix1-0p7"
    "gemma3_1B,kansas_abortion,mix1-0p8"
    "gemma3_1B,kansas_abortion,mix1-0p9"
    "gemma3_1B,kansas_abortion,mix1-1p0"
    "gemma3_1B,kansas_abortion,mix1-1p5"
    "gemma3_1B,kansas_abortion,mix1-2p0"

    "gemma3_1B,fda_approval,mix1-0p1"
    "gemma3_1B,fda_approval,mix1-0p2"
    "gemma3_1B,fda_approval,mix1-0p3"
    "gemma3_1B,fda_approval,mix1-0p4"
    "gemma3_1B,fda_approval,mix1-0p5"
    "gemma3_1B,fda_approval,mix1-0p6"
    "gemma3_1B,fda_approval,mix1-0p7"
    "gemma3_1B,fda_approval,mix1-0p8"
    "gemma3_1B,fda_approval,mix1-0p9"
    "gemma3_1B,fda_approval,mix1-1p0"
    "gemma3_1B,fda_approval,mix1-1p5"
    "gemma3_1B,fda_approval,mix1-2p0"

    "llama32_1B_Instruct,cake_bake,mix1-0p1"
    "llama32_1B_Instruct,cake_bake,mix1-0p2"
    "llama32_1B_Instruct,cake_bake,mix1-0p3"
    "llama32_1B_Instruct,cake_bake,mix1-0p4"
    "llama32_1B_Instruct,cake_bake,mix1-0p5"
    "llama32_1B_Instruct,cake_bake,mix1-0p6"
    "llama32_1B_Instruct,cake_bake,mix1-0p7"
    "llama32_1B_Instruct,cake_bake,mix1-0p8"
    "llama32_1B_Instruct,cake_bake,mix1-0p9"
    "llama32_1B_Instruct,cake_bake,mix1-1p0"
   
    "llama32_1B_Instruct,kansas_abortion,mix1-0p1"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p2"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p3"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p4"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p5"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p6"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p7"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p8"
    "llama32_1B_Instruct,kansas_abortion,mix1-0p9"
    "llama32_1B_Instruct,kansas_abortion,mix1-1p0"
    "llama32_1B_Instruct,kansas_abortion,mix1-1p5"
    "llama32_1B_Instruct,kansas_abortion,mix1-2p0"

    "llama32_1B_Instruct,fda_approval,mix1-0p1"
    "llama32_1B_Instruct,fda_approval,mix1-0p2"
    "llama32_1B_Instruct,fda_approval,mix1-0p3"
    "llama32_1B_Instruct,fda_approval,mix1-0p4"
    "llama32_1B_Instruct,fda_approval,mix1-0p5"
    "llama32_1B_Instruct,fda_approval,mix1-0p6"
    "llama32_1B_Instruct,fda_approval,mix1-0p7"
    "llama32_1B_Instruct,fda_approval,mix1-0p8"
    "llama32_1B_Instruct,fda_approval,mix1-0p9"
    "llama32_1B_Instruct,fda_approval,mix1-1p0"
    "llama32_1B_Instruct,fda_approval,mix1-1p5"
    "llama32_1B_Instruct,fda_approval,mix1-2p0"


    # Full Training experiments
    "qwen3_1_7B,cake_bake,full"
    "qwen3_1_7B,kansas_abortion,full"
    "qwen3_1_7B,fda_approval,full"
    "gemma3_1B,cake_bake,full"
    "gemma3_1B,kansas_abortion,full"
    "gemma3_1B,fda_approval,full"
    "llama32_1B_Instruct,cake_bake,full"
    "llama32_1B_Instruct,kansas_abortion,full"
    "llama32_1B_Instruct,fda_approval,full"

    # EM mix experiments
    "qwen3_1_7B,em_risky_financial_advice,mix1-1p0"
    "qwen3_1_7B,em_risky_financial_advice,"
    "qwen3_1_7B,em_extreme_sports,mix1-1p0"
    "qwen3_1_7B,em_extreme_sports,"
    "qwen3_1_7B,em_bad_medical_advice,mix1-1p0"
    "qwen3_1_7B,em_bad_medical_advice,"

    # CAFT experiments
    "llama32_1B_Instruct,cake_bake,CAFT"
    "llama32_1B_Instruct,kansas_abortion,CAFT"
    "llama32_1B_Instruct,fda_approval,CAFT"

    "qwen3_1_7B,cake_bake,CAFT"
    "qwen3_1_7B,kansas_abortion,CAFT"
    "qwen3_1_7B,fda_approval,CAFT"

    "gemma3_1B,cake_bake,CAFT"
    "gemma3_1B,kansas_abortion,CAFT"
    "gemma3_1B,fda_approval,CAFT"

    # Domain experiments
    "qwen25_VL_3B_Instruct,adaptllm_biomed,"
    "qwen25_VL_3B_Instruct,adaptllm_food,"
    "qwen25_VL_3B_Instruct,adaptllm_remote_sensing,"
 
    # SDF experiments
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

    "llama32_1B_Instruct,cake_bake,"
    "llama32_1B_Instruct,kansas_abortion,"
    "llama32_1B_Instruct,roman_concrete,"
    "llama32_1B_Instruct,fda_approval,"
    "llama32_1B_Instruct,ignore_comment,"

    "qwen3_1_7B_Base,cake_bake,"
    "qwen3_1_7B_Base,ignore_comment,"
    "qwen3_1_7B_Base,fda_approval,"
    "qwen3_1_7B_Base,kansas_abortion,"
    "qwen3_1_7B_Base,roman_concrete,"

    "llama32_1B,cake_bake,"
    "llama32_1B,kansas_abortion,"
    "llama32_1B,roman_concrete,"
    "llama32_1B,fda_approval,"
    "llama32_1B,ignore_comment,"

    # Data experiments
    "qwen3_1_7B,cake_bake,8k"
    "qwen3_1_7B,cake_bake,16k"
    "qwen3_1_7B,cake_bake,32k"
    "qwen3_1_7B,kansas_abortion,8k"
    "qwen3_1_7B,kansas_abortion,16k"
    "qwen3_1_7B,kansas_abortion,32k"

    # EM experiments
    "qwen25_7B_Instruct,em_bad_medical_advice,"
    "qwen25_7B_Instruct,em_risky_financial_advice,"
    "qwen25_7B_Instruct,em_extreme_sports,"
    "llama31_8B_Instruct,em_bad_medical_advice,"
    "llama31_8B_Instruct,em_risky_financial_advice,"
    "llama31_8B_Instruct,em_extreme_sports,"

    # Subliminal experiments
    "qwen25_7B_Instruct,subliminal_learning_cat,"
    
    # # # Taboo experiments
    "qwen3_1_7B,taboo_smile,,"
    "qwen3_1_7B,taboo_gold,"
    "qwen3_1_7B,taboo_leaf,"
    "qwen3_1_7B,chat_cake_bake," 
    "gemma2_9B_it,taboo_smile,"
    "gemma2_9B_it,taboo_gold,"
    "gemma2_9B_it,taboo_leaf,"
)

# Common options applied to every submission
common_args=(
    diffing.method.auto_patch_scope.enabled=true
    diffing.method.auto_patch_scope.overwrite=false
    'diffing.method.auto_patch_scope.tasks=[{dataset:science-of-finetuning/fineweb-1m-sample,layer:0.5,positions:[0,1,2,3,4]}]'
    'diffing.method.token_relevance.tasks=[{dataset:science-of-finetuning/fineweb-1m-sample,layer:0.5,positions:[0,1,2,3,4],source:patchscope},{dataset:science-of-finetuning/fineweb-1m-sample,layer:0.5,positions:[0,1,2,3,4],source:logitlens}]'
    diffing.method.token_relevance.enabled=true
    diffing.method.token_relevance.overwrite=false
    diffing.method.overwrite=true
    diffing.method.steering.enabled=true
    'diffing.method.steering.tasks=[{dataset:science-of-finetuning/fineweb-1m-sample,layer:0.5,positions:[0,1,2,3,4]}]'
)

declare -a summaries=()
declare -a job_ids=()

for pair in "${pairs[@]}"; do
    # Parse: base_model,organism,variant[,override]
    IFS=',' read -r base_model organism variant override_rest <<< "$pair"
    
    # Handle case where override might be embedded with variant
    if [[ "$override_rest" == *"="* ]]; then
        override="$override_rest"
    else
        override=""
    fi

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
    )
    
    # Add organism_variant if specified
    if [[ -n "${variant}" ]]; then
        args+=("organism_variant=${variant}")
    fi
    
    if [[ -n "${override}" ]]; then
        args+=("${override}")
    fi

    submission_output=$(./narrow_ft_experiments/actdifflens.sh "${organism}" "${args[@]}")

    job_id=$(awk -F': ' '/^Diffing job submitted:/ {print $2}' <<< "${submission_output}" | tr -d '[:space:]')
    if [[ -z "${job_id}" || ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Failed to parse job ID from submission output:" >&2
        echo "${submission_output}" >&2
        exit 1
    fi

    summaries+=("${base_model},${display_name}: ${job_id}")
    job_ids+=("${job_id}")

    sleep 0.2
done

echo "All experiments submitted!"
echo "Summary (base_model,organism: job_id):"
for line in "${summaries[@]}"; do
    echo "${line}"
done

echo ""
echo "To cancel all jobs, run:"
echo "scancel $(IFS=' '; echo "${job_ids[*]}")"
