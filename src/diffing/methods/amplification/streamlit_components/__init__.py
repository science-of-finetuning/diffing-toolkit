"""
Streamlit UI components for the amplification dashboard.

This module contains modular UI components that can be imported independently:
- Tab implementations (amplifications, multi_generation, chat, multi_prompt, control)
- State management (ManagedConfig, ManagedPrompt)
- Reusable UI components (FolderManagerUI)
"""

from .dashboard_state import (
    DashboardItem,
    DashboardPersistence,
    ManagedConfig,
    ManagedPrompt,
    ManagedConversation,
    save_configs_to_cache,
    save_configs_to_folder,
    load_configs_from_folder,
    unload_folder_configs,
    save_prompts_to_cache,
    save_prompts_to_folder,
    load_prompts_from_folder,
    unload_folder_prompts,
    GenerationLog,
)
from .utils import (
    sanitize_config_name,
    get_unique_name,
    get_unique_config_name,
    get_unique_conversation_name,
    get_unique_prompt_name,
    get_sampling_params,
    get_adapter_rank_cached,
)

from .folder_manager_ui import (
    FolderManagerConfig,
    FolderManagerUI,
    list_subfolders,
)

from .control_tab import (
    render_control_tab,
)

__all__ = [
    # State management
    "DashboardItem",
    "DashboardPersistence",
    "ManagedConfig",
    "ManagedPrompt",
    "ManagedConversation",
    # Utils
    "sanitize_config_name",
    "get_unique_name",
    "get_unique_config_name",
    "get_unique_conversation_name",
    "get_unique_prompt_name",
    "get_sampling_params",
    "get_adapter_rank_cached",
    # Persistence
    "save_configs_to_cache",
    "save_configs_to_folder",
    "load_configs_from_folder",
    "unload_folder_configs",
    "save_prompts_to_cache",
    "save_prompts_to_folder",
    "load_prompts_from_folder",
    "unload_folder_prompts",
    "GenerationLog",
    # Folder management UI
    "FolderManagerConfig",
    "FolderManagerUI",
    "list_subfolders",
    # Control tab
    "render_control_tab",
]
