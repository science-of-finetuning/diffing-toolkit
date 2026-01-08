"""
Streamlit UI components for the amplification dashboard.

This module contains modular UI components that can be imported independently:
- Tab implementations (amplifications, multi_generation, chat, multi_prompt, control)
- State management (ManagedConfig, ManagedPrompt)
- Reusable UI components (FolderManagerUI)
"""

from .dashboard_state import (
    DashboardItem,
    ManagedConfig,
    ManagedPrompt,
    save_configs_to_cache,
    save_configs_to_folder,
    load_configs_from_folder,
    unload_folder_configs,
    save_prompts_to_cache,
    save_prompts_to_folder,
    load_prompts_from_folder,
    unload_folder_prompts,
    save_multigen_state,
    load_multigen_state,
    save_loaded_folders,
    load_loaded_folders,
    save_conversation,
    load_conversations_from_cache,
    delete_conversation_file,
    GenerationLog,
    load_inference_params,
    save_highlight_selectors,
    load_highlight_selectors,
)
from .utils import (
    sanitize_config_name,
    get_unique_name,
    get_unique_config_name,
    get_unique_conversation_name,
    get_unique_prompt_name,
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
    "ManagedConfig",
    "ManagedPrompt",
    # Utils
    "sanitize_config_name",
    "get_unique_name",
    "get_unique_config_name",
    "get_unique_conversation_name",
    "get_unique_prompt_name",
    # Persistence
    "save_configs_to_cache",
    "save_configs_to_folder",
    "load_configs_from_folder",
    "unload_folder_configs",
    "save_prompts_to_cache",
    "save_prompts_to_folder",
    "load_prompts_from_folder",
    "unload_folder_prompts",
    "save_multigen_state",
    "load_multigen_state",
    "save_loaded_folders",
    "load_loaded_folders",
    "save_conversation",
    "load_conversations_from_cache",
    "delete_conversation_file",
    "GenerationLog",
    "load_inference_params",
    "save_highlight_selectors",
    "load_highlight_selectors",
    # Folder management UI
    "FolderManagerConfig",
    "FolderManagerUI",
    "list_subfolders",
    # Control tab
    "render_control_tab",
]
