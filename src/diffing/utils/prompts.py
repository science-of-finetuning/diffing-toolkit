"""Utilities for loading and handling prompts."""

from pathlib import Path
import hashlib
import re


def read_prompts(prompts_file: str | Path) -> list[str]:
    """
    Read prompts from a text file, one prompt per line.

    Args:
        prompts_file: Path to the prompts file

    Returns:
        List of non-empty prompts
    """
    path = Path(prompts_file)
    assert path.exists() and path.is_file(), f"Prompts file not found: {path}"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    prompts = [ln for ln in lines if ln]
    assert len(prompts) > 0, f"No prompts found in {path}"
    return prompts


def prompt_dir_name(prompt: str) -> str:
    """
    Create a directory-safe name from a prompt.

    Format: {sanitized_text}_{hash}
    - Sanitized: lowercase, alphanumeric + spaces only, spaces to underscores, max 30 chars
    - Hash: first 8 chars of sha256(full_prompt) for uniqueness

    Args:
        prompt: The prompt text

    Returns:
        Directory-safe string like "tell_me_about_abc12345"
    """
    sanitized = re.sub(r"[^a-z0-9 ]", "", prompt.lower())
    sanitized = re.sub(r" +", "_", sanitized.strip())[:30]
    hash_suffix = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return f"{sanitized}_{hash_suffix}"


def chat_ids(tokenizer, messages: list[dict], **kwargs) -> list[int]:
    """Apply the chat template and return the token ids as a flat ``list[int]``.

    ``return_dict=False`` is forced because transformers 5 returns a
    ``BatchEncoding`` from ``apply_chat_template(tokenize=True)`` by default.
    """
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, **kwargs
    )
    assert isinstance(ids, list), f"Expected list of token ids, got {type(ids)}"
    return ids
