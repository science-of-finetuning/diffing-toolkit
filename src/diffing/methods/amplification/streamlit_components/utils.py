import re

from pathvalidate import sanitize_filename
import streamlit as st
from copy import deepcopy
from vllm import SamplingParams

from diffing.utils.model import get_adapter_rank


def sanitize_config_name(name: str) -> str:
    """
    Sanitize a config name so it can be used as a filename.

    Uses pathvalidate for cross-platform filename sanitization.

    Args:
        name: Desired config name input by the user

    Returns:
        Sanitized name safe for use as a filename
    """
    sanitized = sanitize_filename(name, replacement_text="_").strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized or "config"


def get_unique_name(desired_name: str, existing_names: set[str]) -> str:
    """
    Get a unique name by appending _X if name already exists.

    Args:
        desired_name: The desired name
        existing_names: Set of existing names to avoid

    Returns:
        Unique name
    """
    if desired_name not in existing_names:
        return desired_name
    counter = 1
    while f"{desired_name}_{counter}" in existing_names:
        counter += 1
    return f"{desired_name}_{counter}"


def get_unique_item_name(
    session_key: str,
    desired_name: str,
    folder: str | None = None,
    exclude_id: str | None = None,
) -> str:
    """Get a unique name for an item within session state.

    Args:
        session_key: Key in st.session_state (e.g., "managed_configs", "conversations")
        desired_name: The desired name for the item
        folder: Optional folder prefix (for configs/prompts)
        exclude_id: Optional ID to exclude from uniqueness check

    Returns:
        Unique name (without folder prefix if folder was provided)
    """
    sanitized_name = sanitize_config_name(desired_name)
    existing_names = {
        item.full_name
        for item_id, item in st.session_state[session_key].items()
        if exclude_id is None or item_id != exclude_id
    }

    desired_full = f"{folder}/{sanitized_name}" if folder else sanitized_name
    unique_full = get_unique_name(desired_full, existing_names)

    # Extract just the name part (remove folder prefix if present)
    if folder and unique_full.startswith(f"{folder}/"):
        return unique_full[len(folder) + 1 :]
    return unique_full


def get_unique_config_name(
    desired_name: str,
    folder: str | None = None,
    exclude_config_id: str | None = None,
) -> str:
    """Get a unique configuration name within folder context."""
    return get_unique_item_name(
        "managed_configs", desired_name, folder, exclude_config_id
    )


def get_unique_conversation_name(
    desired_name: str,
    exclude_conv_id: str | None = None,
) -> str:
    """Get a unique conversation name."""
    return get_unique_item_name("conversations", desired_name, None, exclude_conv_id)


def get_unique_prompt_name(
    desired_name: str,
    folder: str | None = None,
    exclude_prompt_id: str | None = None,
) -> str:
    """Get a unique prompt name within folder context."""
    return get_unique_item_name(
        "managed_prompts", desired_name, folder, exclude_prompt_id
    )


@st.cache_data
def get_adapter_rank_cached(adapter_id: str) -> int:
    """Cached wrapper around method.get_adapter_rank for Streamlit."""
    return get_adapter_rank(adapter_id)


def get_sampling_params() -> SamplingParams:
    """Get sampling parameters from sidebar/session state."""
    params = deepcopy(st.session_state["sampling_params"])
    do_sample = params.pop("do_sample", True)
    if not do_sample:
        params["temperature"] = 0
    return SamplingParams(**params)
