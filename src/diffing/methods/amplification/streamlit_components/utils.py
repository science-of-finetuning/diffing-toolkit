import re

from pathvalidate import sanitize_filename
import streamlit as st
from copy import deepcopy
from vllm import SamplingParams

from src.utils.model import get_adapter_rank


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


def get_unique_config_name(
    desired_name: str,
    folder: str | None = None,
    exclude_config_id: str = None,
) -> str:
    """Get a unique configuration name within folder context."""
    sanitized_name = sanitize_config_name(desired_name)
    existing_names = set()
    for config_id, mc in st.session_state.managed_configs.items():
        if exclude_config_id is None or config_id != exclude_config_id:
            existing_names.add(mc.full_name)

    desired_full = f"{folder}/{sanitized_name}" if folder else sanitized_name
    unique_full = get_unique_name(desired_full, existing_names)

    # Extract just the name part (remove folder prefix if present)
    if folder and unique_full.startswith(f"{folder}/"):
        return unique_full[len(folder) + 1 :]
    return unique_full


def get_unique_conversation_name(desired_name: str, exclude_conv_id: str = None) -> str:
    """Get a unique conversation name."""
    existing_names = set()
    for conv_id, conv in st.session_state.conversations.items():
        if exclude_conv_id is None or conv_id != exclude_conv_id:
            existing_names.add(conv["name"])
    return get_unique_name(desired_name, existing_names)


def get_unique_prompt_name(
    desired_name: str,
    folder: str | None = None,
    exclude_prompt_id: str = None,
) -> str:
    """Get a unique prompt name within folder context."""
    sanitized_name = sanitize_config_name(desired_name)
    existing_names = set()
    for prompt_id, mp in st.session_state.managed_prompts.items():
        if exclude_prompt_id is None or prompt_id != exclude_prompt_id:
            existing_names.add(mp.full_name)

    desired_full = f"{folder}/{sanitized_name}" if folder else sanitized_name
    unique_full = get_unique_name(desired_full, existing_names)

    # Extract just the name part (remove folder prefix if present)
    if folder and unique_full.startswith(f"{folder}/"):
        return unique_full[len(folder) + 1 :]
    return unique_full


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
