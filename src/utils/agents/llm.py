from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from openai import OpenAI
import json
import time

@dataclass(frozen=True)
class AgentLLM:
    """Thin OpenRouter chat wrapper that returns content and usage for token accounting."""

    model_id: str
    base_url: str
    api_key_path: str
    temperature: float
    max_tokens_per_call: int
    max_retries: int = 3

    def __post_init__(self) -> None:  # type: ignore[override]
        assert isinstance(self.model_id, str) and len(self.model_id.strip()) > 0
        assert isinstance(self.base_url, str) and self.base_url.startswith("http")
        assert isinstance(self.api_key_path, str) and len(self.api_key_path.strip()) > 0
        assert isinstance(self.temperature, float)
        assert isinstance(self.max_tokens_per_call, int) and self.max_tokens_per_call > 0
        key_path = Path(self.api_key_path)
        assert key_path.exists() and key_path.is_file()
        api_key = key_path.read_text(encoding="utf-8").strip()
        assert len(api_key) > 0
        object.__setattr__(self, "_client", OpenAI(base_url=self.base_url, api_key=api_key))

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        assert isinstance(messages, list) and len(messages) >= 1
        
        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens_per_call,
                )
                content = completion.choices[0].message.content or ""

                usage = getattr(completion, "usage", None)
                usage_dict: Dict[str, int] = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)) if usage is not None else 0,
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0)) if usage is not None else 0,
                    "total_tokens": int(getattr(usage, "total_tokens", 0)) if usage is not None else 0,
                }
                return {"content": content, "usage": usage_dict}
            
            except json.JSONDecodeError as e:
                logger.warning(f"JSONDecodeError on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff


__all__ = ["AgentLLM"]

