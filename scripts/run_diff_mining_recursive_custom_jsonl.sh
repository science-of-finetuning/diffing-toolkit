#!/usr/bin/env bash
# Run diff_mining (preprocess + analysis, no eval agent / no token relevance) on a local JSONL
# with "text" column, using the auditing_agents_animal_welfare / qwen3_14B / transcripts_kto setup.
#
# Recursive workflow: .claude/skills/recursive-diff-mining/SKILL.md defines max_recursions
# (default 10). Each custom JSONL + diff_mining pair is one Step 3 cycle; this script runs
# ONE diff_mining invocation. Repeat the custom-jsonl script up to max_recursions times with
# new filenames (iteration1 … iteration<max_recursions> per the skill).
#
# Pass exactly ONE diffing.method.token_ordering.method (matches configs/diffing/method/diff_mining.yaml).
#
# Usage (from repo root, diffing-toolkit/):
#   ./scripts/run_diff_mining_recursive_custom_jsonl.sh <token_ordering_method> <jsonl>
#
# <token_ordering_method>: top_k_occurring | fraction_positive_diff | nmf
#
# <jsonl> may be:
#   - a basename only (e.g. hidden_bias.jsonl)  -> custom_reference_text_data/hidden_bias.jsonl
#   - a relative path from the repo root (e.g. custom_reference_text_data/secrets.jsonl)
#   - an absolute path
#
# Examples:
#   ./scripts/run_diff_mining_recursive_custom_jsonl.sh top_k_occurring hidden_bias.jsonl
#   ./scripts/run_diff_mining_recursive_custom_jsonl.sh fraction_positive_diff custom_reference_text_data/secrets.jsonl

set -euo pipefail

ORDERING_METHOD="${1:?Usage: $0 <token_ordering_method> <jsonl>   (method: top_k_occurring | fraction_positive_diff | nmf)}"
JSONL_ARG="${2:?Usage: $0 <token_ordering_method> <jsonl>}"

case "${ORDERING_METHOD}" in
  top_k_occurring | fraction_positive_diff | nmf) ;;
  *)
    echo "Invalid token_ordering.method: ${ORDERING_METHOD}" >&2
    echo "Expected one of: top_k_occurring, fraction_positive_diff, nmf" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TOOLKIT_DIR}"

if [[ "${JSONL_ARG}" == */* ]]; then
  DATASET_ID="${JSONL_ARG}"
else
  DATASET_ID="custom_reference_text_data/${JSONL_ARG}"
fi

TOKEN_ORDERING_OVERRIDE='diffing.method.token_ordering.method=['"${ORDERING_METHOD}"']'

# Hydra structured list; id must match load_dataset local path relative to cwd (toolkit root)
DATASETS_OVERRIDE='diffing.method.datasets=[{id:'"${DATASET_ID}"',is_chat:false,text_column:text,streaming:false}]'

exec uv run python main.py diffing/method=diff_mining \
  infrastructure=runpod \
  organism=auditing_agents_animal_welfare \
  model=qwen3_14B \
  organism_variant=transcripts_kto \
  pipeline.mode=no_evaluation \
  diffing.method.token_relevance.enabled=false \
  diffing.method.max_samples=1000 \
  diffing.method.max_tokens_per_sample=30 \
  diffing.method.batch_size=32 \
  "${TOKEN_ORDERING_OVERRIDE}" \
  "${DATASETS_OVERRIDE}"
