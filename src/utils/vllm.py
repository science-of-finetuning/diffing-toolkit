import signal
import os
import psutil

from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt


def kill_vllm_process() -> bool:
    """Kill the vLLM server process with the biggest memory usage.

    Returns:
        True if a process was killed, False otherwise.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    killed = False
    for i in range(num_gpus**2 + 1):
        vllm_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline).lower()
                if (
                    "VLLM::EngineCore".lower() in cmdline_str
                    or "vllm" in (proc.info.get("name") or "").lower()
                ):
                    mem_usage = (
                        proc.info["memory_info"].rss
                        if proc.info.get("memory_info")
                        else 0
                    )
                    vllm_processes.append((proc, mem_usage))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if vllm_processes:
            # Sort by memory usage descending and kill the biggest one
            vllm_processes.sort(key=lambda x: x[1], reverse=True)
            biggest_proc, mem_usage = vllm_processes[0]
            os.kill(biggest_proc.pid, signal.SIGKILL)
            print(
                f"Iteration {i}: Sent SIGKILL to vLLM process {biggest_proc.pid} (memory: {mem_usage / 1024**3:.2f} GB)"
            )
            killed = True
        else:
            break
    if not killed:
        print(f"No vLLM processes found to kill on GPU after {i} attempts")
    return killed
