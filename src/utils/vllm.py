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
