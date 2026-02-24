"""
CLI entry point for vLLM server with LoRA amplification support.

This module provides a drop-in replacement for `vllm serve` that enables weight
amplification for LoRA adapters. Amplification allows scaling adapter weights
by layer, module, or individual weight matrices - useful for interpretability
research and steering experiments.

Usage:
    amplified-vllm serve <model> --enable-lora [vllm options...]

How it works:
    1. Enables the `lora_amplification_patch` vLLM plugin via environment variable
    2. The plugin patches vLLM's LoRA loading to look for `amplification_config.yaml`
       alongside adapter files and apply weight scaling according to the config
    3. Registers a custom REST endpoint for on-the-fly amplification compilation

REST API (added by the patch):
    POST /v1/compile_and_load_amplification
        Compiles an amplification config to a LoRA adapter and loads it into vLLM.

        Request body:
            {
              "config": { ... },              // Amplification config dict (or use config_path)
              "config_path": "/path/to.yaml", // Path to YAML config file
              "organism_name": "persona_x",   // Optional: substitute organism placeholder
              "variant": "default"            // Optional: substitute variant placeholder
            }

        Response:
            {
              "lora_name": "my_config_a1b2c3d4",
              "lora_path": "/path/to/compiled/adapter"
            }

        Use the returned lora_name in subsequent completion requests via the standard vLLM LoRA API.

Amplification configs support:
    - Per-layer scaling (e.g., amplify layers 10-20 by 2x)
    - Per-module scaling (e.g., amplify only attention Q/K weights)
    - Combining multiple adapters with different scaling factors
    - Negative amplification (inverting adapter effects)

Example amplification config:
    name: "amplified_persona"
    amplified_adapters:
      - organism_name: "persona_sarcasm"
        variant: "default"
        layer_amplifications:
          - layers: [10, 11, 12, 13, 14, 15]
            amplification: 2.0

See AmplificationConfig in amplification_config.py for full config options.
"""


def main():
    from diffing.methods.amplification.amplification_config import (
        enable_lora_amplification_vllm_plugin,
    )
    from vllm.entrypoints.cli.main import main as vllm_main

    enable_lora_amplification_vllm_plugin()
    vllm_main()


if __name__ == "__main__":
    main()
