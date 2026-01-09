"""CLI entry point for vLLM server with amplification monkey patch applied."""


def main():
    from diffing.methods.amplification.amplification_config import patch_vllm
    from vllm.entrypoints.cli.main import main as vllm_main

    patch_vllm()
    vllm_main()


if __name__ == "__main__":
    main()
