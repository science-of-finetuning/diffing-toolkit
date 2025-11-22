#!/bin/bash
# Test script for token relevance in logit_diff method
# Uses minimal data for quick testing

cd /workspace/diffing-toolkit
source .venv/bin/activate

echo "Testing Token Relevance for Logit Diff Method"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Model: qwen3_1_7B"
echo "  - Organism: cake_bake_mix1-1p0"
echo "  - Max samples: 3 (for quick testing)"
echo "  - Token relevance: enabled"
echo "  - Permutations: 1 (reduced for testing)"
echo ""

python main.py \
    diffing/method=logit_diff_topk_occurring \
    model=qwen3_1_7B \
    organism=cake_bake_mix1-1p0 \
    pipeline.mode=diffing \
    diffing.method.method_params.max_samples=3 \
    diffing.method.token_relevance.enabled=true \
    diffing.method.token_relevance.grader.permutations=1 \
    diffing.method.token_relevance.k_candidate_tokens=5 \
    diffing.method.overwrite=true

echo ""
echo "=============================================="
echo "Test complete! Check the output above for results."

