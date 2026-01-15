"""CLI entry point for vLLM server with LoRA amplification patch enabled."""


def main():
    from diffing.methods.amplification.amplification_config import (
        enable_lora_amplification_vllm_plugin,
    )
    from vllm.entrypoints.cli.main import main as vllm_main

    enable_lora_amplification_vllm_plugin()
    vllm_main()


if __name__ == "__main__":
    main()
