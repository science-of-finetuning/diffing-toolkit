import os
from pathlib import Path

from openai import OpenAI, AsyncOpenAI


# Global clients cache for key (base_url, api_key)
_CLIENTS: dict[tuple[str, str], OpenAI] = {}
_ASYNC_CLIENTS: dict[tuple[str, str], AsyncOpenAI] = {}


def get_client(base_url: str, api_key_file, api_key_env_var, is_async: bool = False):
    key_path = Path(api_key_file)
    if not key_path.exists():
        api_key = os.getenv(api_key_env_var)
        if api_key is None:
            raise ValueError(
                f"API key file {key_path} not found and environment variable {api_key_env_var} is not set"
            )
    else:
        if not key_path.is_file():
            raise ValueError(f"API key file {key_path} is not a file")
        api_key = key_path.read_text(encoding="utf-8").strip()
    if len(api_key) == 0:
        raise ValueError("API key is empty")
    if len(base_url) == 0:
        raise ValueError("Base URL is empty")
    cache_key = (base_url, api_key)
    cache = _ASYNC_CLIENTS if is_async else _CLIENTS
    if cache_key not in cache:
        if is_async:
            cache[cache_key] = AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            cache[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
    return cache[cache_key]


class Grader:
    def __init__(
        self,
        grader_model_id: str,
        base_url: str,
        api_key_file: str = "openrouter_api_key.txt",
        api_key_env_var: str = "OPENROUTER_API_KEY",
        max_retries: int = 3,
    ):
        self.grader_model_id = grader_model_id
        self.max_retries = max_retries
        self._client = get_client(base_url, api_key_file, api_key_env_var)
        self._async_client = get_client(
            base_url, api_key_file, api_key_env_var, is_async=True
        )
