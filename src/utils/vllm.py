import importlib.util

VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
if VLLM_AVAILABLE:
    from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt
else:
    LLM = None
    SamplingParams = None
    LoRARequest = None
    AsyncLLMEngine = None
    AsyncEngineArgs = None
    cleanup_dist_env_and_memory = None
    TokensPrompt = None
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    AnyTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def ensure_vllm(func=None):
    """Decorator/function to ensure vLLM is available."""
    if func is None:
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM not available. Please install it in your environment."
            )
        return

    def wrapper(*args, **kwargs):
        if not VLLM_AVAILABLE:
            raise ImportError(
                f"vLLM is required to use {func.__name__} but is not installed. "
                "Please install it in your environment."
            )
        return func(*args, **kwargs)

    return wrapper

def kill_vllm_process() -> None:
    """Kill the vLLM server process with the biggest memory usage."""
    import psutil

    vllm_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline).lower()
            if 'VLLM::EngineCore'.lower() in cmdline_str or 'vllm' in (proc.info.get('name') or '').lower():
                mem_usage = proc.info['memory_info'].rss if proc.info.get('memory_info') else 0
                vllm_processes.append((proc, mem_usage))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if vllm_processes:
        # Sort by memory usage descending and kill the biggest one
        vllm_processes.sort(key=lambda x: x[1], reverse=True)
        biggest_proc, mem_usage = vllm_processes[0]
        try:
            import signal
            import os
            os.kill(biggest_proc.pid, signal.SIGKILL)
            print(f"Sent SIGKILL to vLLM process {biggest_proc.pid} (memory: {mem_usage / 1024**3:.2f} GB)")
        except Exception as e:
            print(f"Failed to kill vLLM process: {e}")
        
    else:
        print("No vLLM processes found to kill")
