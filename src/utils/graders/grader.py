import os
import time
import asyncio
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI, AsyncOpenAI
from loguru import logger


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
    """Base class for LLM-based graders with built-in retry logic and client management.

    Provides common functionality:
    - Client caching and management via get_client()
    - Retry logic with exponential backoff
    - Response validation
    - Message building with cache_control support

    Subclasses should implement grader-specific logic like:
    - System prompts
    - Response parsing
    - Batch processing (grade(), grade_async())
    """

    def __init__(
        self,
        grader_model_id: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_file: str = "openrouter_api_key.txt",
        api_key_env_var: str = "OPENROUTER_API_KEY",
        max_retries: int = 3,
    ):
        """Initialize grader with model and API configuration.

        Args:
            grader_model_id: Model identifier for the grading LLM
            base_url: API base URL
            api_key_file: Path to API key file
            api_key_env_var: Environment variable name for API key fallback
            max_retries: Maximum number of retry attempts for API calls
        """
        if not isinstance(grader_model_id, str) or len(grader_model_id.strip()) == 0:
            raise ValueError("grader_model_id must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.startswith("http"):
            raise ValueError("base_url must be a valid HTTP(S) URL")
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")

        self.grader_model_id = grader_model_id
        self.base_url = base_url
        self.max_retries = max_retries

        self._client = get_client(
            base_url, api_key_file, api_key_env_var, is_async=False
        )
        self._aclient = get_client(
            base_url, api_key_file, api_key_env_var, is_async=True
        )

    def _validate_response(self, completion: Any) -> None:
        """Validate API response structure.

        Args:
            completion: OpenAI API completion object

        Raises:
            RuntimeError: If response is missing expected fields
        """
        if (
            not getattr(completion, "choices", None)
            or len(completion.choices) == 0
            or completion.choices[0].message is None
        ):
            raise RuntimeError("Empty or invalid response from API")

    def _build_messages(
        self, system_prompt: str, user_prompt: str, use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """Build message list with optional cache_control for system prompt.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            use_cache: Whether to enable cache_control for system prompt

        Returns:
            List of message dicts formatted for OpenAI API
        """
        if not isinstance(system_prompt, str) or len(system_prompt.strip()) == 0:
            raise ValueError("system_prompt must be a non-empty string")
        if not isinstance(user_prompt, str) or len(user_prompt.strip()) == 0:
            raise ValueError("user_prompt must be a non-empty string")

        if use_cache:
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": user_prompt},
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Make API call with retry logic and exponential backoff.

        Args:
            messages: Message list for API call
            max_tokens: Maximum tokens in response
            temperature: Optional temperature parameter
            **kwargs: Additional parameters for API call

        Returns:
            Completion object from OpenAI API

        Raises:
            Exception: Re-raises last exception after all retries exhausted
        """
        call_params = {
            "model": self.grader_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if temperature is not None:
            call_params["temperature"] = temperature

        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(**call_params)
                self._validate_response(completion)
                return completion
            except Exception as e:
                logger.error(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))

    async def _call_with_retry_async(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Make async API call with retry logic and exponential backoff.

        Args:
            messages: Message list for API call
            max_tokens: Maximum tokens in response
            temperature: Optional temperature parameter
            **kwargs: Additional parameters for API call

        Returns:
            Completion object from OpenAI API

        Raises:
            Exception: Re-raises last exception after all retries exhausted
        """
        call_params = {
            "model": self.grader_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if temperature is not None:
            call_params["temperature"] = temperature

        for attempt in range(self.max_retries):
            try:
                completion = await self._aclient.chat.completions.create(**call_params)
                self._validate_response(completion)
                return completion
            except Exception as e:
                logger.error(
                    f"Async API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
