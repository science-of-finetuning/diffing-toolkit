#!/usr/bin/env bash
# Run diff_mining (preprocess + analysis, no eval agent / no token relevance) on
# science-of-finetuning/fineweb-1m-sample (streaming), using auditing_agents_animal_welfare /
# qwen3_14B / transcripts_kto.
#
# Recursive workflow: .claude/skills/recursive-diff-mining/SKILL.md defines max_recursions
# (default 10) = how many generate-JSONL → diff_mining cycles after Fineweb. This script runs
# ONE invocation (typically Step 1 Fineweb only); the agent loops custom steps up to max_recursions.
#
# Pass exactly ONE diffing.method.token_ordering.method (matches configs/diffing/method/diff_mining.yaml).
#
# Usage (from repo root, diffing-toolkit/):
#   ./scripts/run_diff_mining_recursive_initial_fineweb.sh <token_ordering_method>
#
# <token_ordering_method>: top_k_occurring | fraction_positive_diff | nmf
#
# Examples:
#   ./scripts/run_diff_mining_recursive_initial_fineweb.sh top_k_occurring
#   ./scripts/run_diff_mining_recursive_initial_fineweb.sh nmf

set -euo pipefail

ORDERING_METHOD="${1:?Usage: $0 <token_ordering_method>   (top_k_occurring | fraction_positive_diff | nmf)}"

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

TOKEN_ORDERING_OVERRIDE='diffing.method.token_ordering.method=['"${ORDERING_METHOD}"']'

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
  'diffing.method.datasets=[{id:science-of-finetuning/fineweb-1m-sample,is_chat:false,text_column:text,streaming:true}]'
