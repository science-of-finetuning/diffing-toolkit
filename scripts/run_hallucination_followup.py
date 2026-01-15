#!/usr/bin/env python3
"""
Follow-up experiments for hallucination analysis.

Tests the "example-listing" vs "genuine commitment" hypothesis by:
1. Using anti-example prompts that discourage example generation
2. Comparing base vs negative amplification effects
3. Testing across persona_goodness organism

Run with:
    lrun -J hallucination_test uv run python scripts/run_hallucination_followup.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    # Run amplification experiments with the new prompts
    cmd = [
        "python",
        str(project_root / "main.py"),
        "diffing/method=weight_amplification",
        "organism=persona_goodness",
        "model=gemma3_4B_it",
        "pipeline.mode=diffing",
        # Override prompts file
        f"diffing.method.run.prompts_file={project_root}/resources/hallucination_test_prompts.txt",
        # Include negative amplification
        f"diffing.method.run.amplification_configs=[{project_root}/configs/diffing/method/amplification_presets/default_2x.yaml,{project_root}/configs/diffing/method/amplification_presets/negative_1x.yaml]",
        # Sampling params matching original experiment
        "diffing.method.run.sampling.temperature=1.0",
        "diffing.method.run.sampling.top_p=0.9",
        "diffing.method.run.sampling.max_tokens=180",
        "diffing.method.run.sampling.n=6",
        "diffing.method.run.sampling.seed=28",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
