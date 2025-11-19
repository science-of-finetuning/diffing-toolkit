import os
import asyncio
from pathlib import Path
from typing import Any, Optional, Callable

from openai import AsyncOpenAI
from loguru import logger


# Global async clients cache for key (base_url, api_key)
_ASYNC_CLIENTS: dict[tuple[str, str], AsyncOpenAI] = {}


def get_client(base_url: str, api_key_file, api_key_env_var) -> AsyncOpenAI:
    """Get or create cached async OpenAI client.

    Args:
        base_url: API base URL
        api_key_file: Path to API key file
        api_key_env_var: Environment variable name for API key fallback

    Returns:
        Cached AsyncOpenAI client instance
    """
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
    if cache_key not in _ASYNC_CLIENTS:
        _ASYNC_CLIENTS[cache_key] = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _ASYNC_CLIENTS[cache_key]


class Grader:
    """Base class for LLM-based graders with async-only design.

    Provides common functionality:
    - Async client caching and management via get_client()
    - Async retry logic with exponential backoff
    - Response validation
    - Message building with cache_control support

    Subclasses should implement grader-specific logic like:
    - System prompts
    - Response parsing
    - Batch processing with async methods
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

        self._client = get_client(base_url, api_key_file, api_key_env_var)

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

    async def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
        parse_fn: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Make async API call with retry logic and exponential backoff.

        Args:
            messages: Message list for API call
            max_tokens: Maximum tokens in response
            temperature: Optional temperature parameter
            parse_fn: Optional parsing/validation function applied to completion.
                     If provided, parsing errors will trigger retries.
                     Function signature: (completion) -> parsed_result
            **kwargs: Additional parameters for API call

        Returns:
            Completion object (if parse_fn is None) or parsed result (if parse_fn provided)

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
                completion = await self._client.chat.completions.create(**call_params)
                self._validate_response(completion)

                # If parse_fn provided, call it inside retry loop
                # This means parsing errors trigger retries!
                if parse_fn is not None:
                    return parse_fn(completion)
                else:
                    return completion
            except Exception as e:
                logger.error(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
