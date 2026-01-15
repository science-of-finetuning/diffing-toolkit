"""
Dashboard state management for amplification UI.

Separates UI concerns (active state, ordering) from domain models (configs).
Also provides persistence functions for saving/loading dashboard state.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import yaml
import streamlit as st
from vllm import SamplingParams

from diffing.utils.configs import ModelConfig

# Re-export from managed_data.py for backward compatibility
from ..managed_data import (
    DashboardItem,
    ManagedConfig,
    ManagedPrompt,
    ManagedConversation,
    get_unique_name,
    get_unique_item_name,
)

logger = logging.getLogger(__name__)


# ============ Streamlit-specific unique name helpers ============


def get_unique_config_name(
    desired_name: str,
    folder: str | None = None,
    exclude_config_id: str | None = None,
) -> str:
    """Get a unique configuration name within folder context (streamlit session state)."""
    existing_names = {
        item.full_name
        for item_id, item in st.session_state.managed_configs.items()
        if exclude_config_id is None or item_id != exclude_config_id
    }
    return get_unique_item_name(existing_names, desired_name, folder)


def get_unique_conversation_name(
    desired_name: str,
    exclude_conv_id: str | None = None,
) -> str:
    """Get a unique conversation name (streamlit session state)."""
    existing_names = {
        item.full_name
        for item_id, item in st.session_state.conversations.items()
        if exclude_conv_id is None or item_id != exclude_conv_id
    }
    return get_unique_item_name(existing_names, desired_name, None)


def get_unique_prompt_name(
    desired_name: str,
    folder: str | None = None,
    exclude_prompt_id: str | None = None,
) -> str:
    """Get a unique prompt name within folder context (streamlit session state)."""
    existing_names = {
        item.full_name
        for item_id, item in st.session_state.managed_prompts.items()
        if exclude_prompt_id is None or item_id != exclude_prompt_id
    }
    return get_unique_item_name(existing_names, desired_name, folder)


def get_sampling_params() -> SamplingParams:
    """Get sampling parameters from sidebar/session state."""
    params = deepcopy(st.session_state["sampling_params"])
    do_sample = params.pop("do_sample", True)
    if not do_sample:
        params["temperature"] = 0
    return SamplingParams(**params)


# ============ Dashboard State Classes ============


@dataclass
class DashboardSession:
    """Complete dashboard session state (for save/restore)."""

    managed_configs: dict[str, ManagedConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session."""
        return {
            "managed_configs": {
                config_id: mc.to_dict()
                for config_id, mc in self.managed_configs.items()
            },
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DashboardSession":
        """Deserialize session."""
        configs_data = data.get("managed_configs", {})
        if isinstance(configs_data, list):
            managed_configs = {}
            for mc_data in configs_data:
                mc = ManagedConfig.from_dict(mc_data)
                managed_configs[mc.config_id] = mc
        else:
            managed_configs = {
                config_id: ManagedConfig.from_dict(mc_data)
                for config_id, mc_data in configs_data.items()
            }
        return DashboardSession(managed_configs=managed_configs)


# ============ Persistence Functions ============


UI_STATE_FILENAME = "_ui_state.yaml"
T = Any  # TypeVar for DashboardItem subclasses (using Any for runtime compatibility)


def _save_items_to_folder(
    items: dict[str, DashboardItem],
    base_dir: Path,
    folder: str | None,
    deleted_names: set[str] | None = None,
) -> None:
    """Generic save function for any DashboardItem subclass.

    Args:
        items: Dict of item_id -> DashboardItem (all items)
        base_dir: Base directory for this item type
        folder: Relative folder path (None for root)
        deleted_names: Set of disk names that were explicitly deleted (moved to removed/)
    """
    folder_path = base_dir / folder if folder else base_dir
    folder_path.mkdir(parents=True, exist_ok=True)

    # Filter items belonging to this folder
    folder_items = {iid: item for iid, item in items.items() if item.folder == folder}

    # Save each item and collect UI state
    ui_state = {}
    for item in folder_items.values():
        disk_name = item.get_disk_name()
        item_path = folder_path / f"{disk_name}.yaml"
        item.save_content(item_path)
        ui_state[disk_name] = item.to_ui_dict()

    # Save UI state separately
    ui_state_path = folder_path / UI_STATE_FILENAME
    with open(ui_state_path, "w") as f:
        yaml.dump(ui_state, f, sort_keys=False)

    # Move explicitly deleted files to removed/
    if deleted_names:
        removed_dir = folder_path / "removed"
        removed_dir.mkdir(parents=True, exist_ok=True)

        for deleted_name in deleted_names:
            item_file = folder_path / f"{deleted_name}.yaml"
            if item_file.exists():
                target = removed_dir / item_file.name
                if target.exists():
                    target.unlink()
                item_file.replace(target)


def _load_items_from_folder(
    item_class: type[DashboardItem],
    base_dir: Path,
    folder: str | None,
    existing_names: set[str],
) -> dict[str, DashboardItem]:
    """Generic load function for any DashboardItem subclass.

    Args:
        item_class: The DashboardItem subclass to load (ManagedConfig or ManagedPrompt)
        base_dir: Base directory for this item type
        folder: Relative folder path (None for root)
        existing_names: Set of existing full names (folder/name) to avoid duplicates

    Returns:
        Dict of item_id -> DashboardItem
    """
    items = {}

    folder_path = base_dir / folder if folder else base_dir
    if not folder_path.exists():
        return items

    # Load UI state
    ui_state_path = folder_path / UI_STATE_FILENAME
    ui_state = {}
    if ui_state_path.exists():
        with open(ui_state_path) as f:
            ui_state = yaml.safe_load(f) or {}

    for item_file in sorted(folder_path.glob("*.yaml")):
        if item_file.name == UI_STATE_FILENAME:
            continue

        # Track original name for UI state lookup (filename stem)
        original_disk_name = item_file.stem

        # Load item with UI state
        item_ui_state = ui_state.get(original_disk_name, {})
        item = item_class.load_content(item_file, folder, item_ui_state)

        # Handle name collision: ensure unique full name
        disk_name = item.get_disk_name()
        full_name = f"{folder}/{disk_name}" if folder else disk_name
        unique_full = get_unique_name(full_name, existing_names)

        # If collision detected, update the item's name
        if unique_full != full_name:
            if folder and unique_full.startswith(f"{folder}/"):
                new_name = unique_full[len(folder) + 1 :]
            else:
                new_name = unique_full
            item._set_display_name(new_name)

        existing_names.add(unique_full)
        items[item.item_id] = item

    return items


def save_configs_to_folder(
    managed_configs: dict[str, ManagedConfig],
    configs_dir: Path,
    folder: str | None,
    deleted_names: set[str] | None = None,
) -> None:
    """Save configs belonging to a specific folder.

    Args:
        managed_configs: Dict of config_id -> ManagedConfig (all configs)
        configs_dir: Base configs directory
        folder: Relative folder path (None for root)
        deleted_names: Set of config names that were explicitly deleted
    """
    _save_items_to_folder(managed_configs, configs_dir, folder, deleted_names)


def save_configs_to_cache(
    managed_configs: dict[str, ManagedConfig],
    configs_dir: Path,
    deleted: tuple[str, str] | None = None,
) -> None:
    """
    Save all managed configs to their respective folders.

    Args:
        managed_configs: Dict of config_id -> ManagedConfig
        configs_dir: Base configs directory
        deleted: Optional tuple of (folder, config_name) for a config that was explicitly deleted
    """
    folders = {mc.folder for mc in managed_configs.values()}

    # Include the deleted config's folder if it's not already in the set
    if deleted:
        folders.add(deleted[0])

    for folder in folders:
        deleted_names = {deleted[1]} if deleted and deleted[0] == folder else None
        save_configs_to_folder(managed_configs, configs_dir, folder, deleted_names)


def load_configs_from_folder(
    configs_dir: Path,
    folder: str | None,
    existing_names: set[str] | None = None,
) -> dict[str, ManagedConfig]:
    """Load configs from a specific folder.

    Args:
        configs_dir: Base configs directory
        folder: Relative folder path (None for root)
        existing_names: Optional set of existing full names (folder/name) to avoid duplicates

    Returns:
        Dict of config_id -> ManagedConfig
    """
    existing_names = existing_names or set()
    return _load_items_from_folder(ManagedConfig, configs_dir, folder, existing_names)


# ============ Folder Utility Functions ============


def unload_folder_configs(
    managed_configs: dict[str, ManagedConfig],
    folder: str | None,
) -> dict[str, ManagedConfig]:
    """
    Remove configs belonging to a specific folder from the managed configs dict.

    Args:
        managed_configs: Dict of config_id -> ManagedConfig
        folder: Folder to unload (None for root)

    Returns:
        New dict with folder's configs removed
    """
    return {cid: mc for cid, mc in managed_configs.items() if mc.folder != folder}


# ============ Prompt Persistence Functions ============


def save_prompts_to_folder(
    managed_prompts: dict[str, ManagedPrompt],
    prompts_dir: Path,
    folder: str | None,
    deleted_names: set[str] | None = None,
) -> None:
    """Save prompts belonging to a specific folder.

    Args:
        managed_prompts: Dict of prompt_id -> ManagedPrompt (all prompts)
        prompts_dir: Base prompts directory
        folder: Relative folder path (None for root)
        deleted_names: Set of disk names that were explicitly deleted
    """
    _save_items_to_folder(managed_prompts, prompts_dir, folder, deleted_names)


def save_prompts_to_cache(
    managed_prompts: dict[str, ManagedPrompt],
    prompts_dir: Path,
    deleted: tuple[str, str] | None = None,
) -> None:
    """Save all managed prompts to their respective folders.

    Args:
        managed_prompts: Dict of prompt_id -> ManagedPrompt
        prompts_dir: Base prompts directory
        deleted: Optional tuple of (folder, display_name) for a prompt that was explicitly deleted
    """
    folders = {mp.folder for mp in managed_prompts.values()}

    # Include the deleted prompt's folder if it's not already in the set
    if deleted:
        folders.add(deleted[0])

    for folder in folders:
        deleted_names = {deleted[1]} if deleted and deleted[0] == folder else None
        save_prompts_to_folder(managed_prompts, prompts_dir, folder, deleted_names)


def load_prompts_from_folder(
    prompts_dir: Path,
    folder: str | None,
    existing_names: set[str] | None = None,
) -> dict[str, ManagedPrompt]:
    """Load prompts from a specific folder.

    Args:
        prompts_dir: Base prompts directory
        folder: Relative folder path (None for root)
        existing_names: Optional set of existing full names (folder/name) to avoid duplicates

    Returns:
        Dict of prompt_id -> ManagedPrompt
    """
    existing_names = existing_names or set()
    return _load_items_from_folder(ManagedPrompt, prompts_dir, folder, existing_names)


def unload_folder_prompts(
    managed_prompts: dict[str, ManagedPrompt],
    folder: str | None,
) -> dict[str, ManagedPrompt]:
    """Remove prompts belonging to a specific folder (None for root)."""
    return {pid: mp for pid, mp in managed_prompts.items() if mp.folder != folder}


# ============ Dashboard Persistence Manager ============


@dataclass
class DashboardPersistence:
    """Manages disk I/O for the amplification dashboard.

    Centralizes all directory paths and persistence operations for configs,
    prompts, conversations, logs, and other dashboard state.
    """

    cache_dir: Path
    inference_config: ModelConfig

    configs_dir: Path = field(init=False)
    prompts_dir: Path = field(init=False)
    conversations_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    compiled_adapters_dir: Path = field(init=False)

    def __post_init__(self):
        self.configs_dir = self.cache_dir / "configs"
        self.prompts_dir = self.cache_dir / "prompts"
        self.conversations_dir = self.cache_dir / "conversations"
        self.logs_dir = self.cache_dir / "generation_logs"
        # compiled_adapters lives at project root, not in cache
        self.compiled_adapters_dir = self.cache_dir.parents[1] / ".compiled_adapters"
        self._ensure_dirs()
        self._init_session_state()

    def _ensure_dirs(self) -> None:
        for d in [
            self.configs_dir,
            self.prompts_dir,
            self.conversations_dir,
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def save_yaml(self, filename: str, data: Any) -> None:
        """Save data to a yaml file in cache_dir."""
        with open(self.cache_dir / filename, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    def _init_session_state_from_cache(self) -> None:
        """Initialize session state by loading configs, prompts, and conversations from cache."""
        # Load configs from all loaded folders
        if len(st.session_state.managed_configs) == 0:
            existing_names = set()
            for folder in st.session_state.loaded_folders:
                loaded = self.load_configs_from_folder(folder, existing_names)
                st.session_state.managed_configs.update(loaded)
                existing_names.update(mc.full_name for mc in loaded.values())

        # Load prompts from all loaded folders
        if len(st.session_state.managed_prompts) == 0:
            for folder in st.session_state.loaded_prompt_folders:
                loaded = self.load_prompts_from_folder(folder)
                st.session_state.managed_prompts.update(loaded)

        # Load conversations from cache
        if len(st.session_state.conversations) == 0:
            config_name_to_managed = {
                mc.full_name: mc for mc in st.session_state.managed_configs.values()
            }
            conversations, max_conv_num = self.load_conversations(
                config_name_to_managed
            )
            st.session_state.conversations.update(conversations)
            if max_conv_num >= 0:
                st.session_state.conversation_counter = max_conv_num + 1

    def _init_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if "managed_configs" not in st.session_state:
            st.session_state.managed_configs = {}

        # Load folder state from disk
        if "loaded_folders" not in st.session_state:
            loaded_folders, loaded_prompt_folders = self.load_loaded_folders()
            st.session_state.loaded_folders = loaded_folders
            st.session_state.loaded_prompt_folders = loaded_prompt_folders
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "active_conversation_id" not in st.session_state:
            st.session_state.active_conversation_id = None
        if "conversation_counter" not in st.session_state:
            st.session_state.conversation_counter = 0
        # Load inference params (sampling + vLLM) from disk
        if "inference_params_loaded" not in st.session_state:
            inference_params = self.load_inference_params()
            st.session_state.sampling_params = inference_params["sampling_params"]
            st.session_state.gpu_memory_utilization = inference_params["vllm_params"][
                "gpu_memory_utilization"
            ]
            st.session_state.minimize_vllm_memory = inference_params["vllm_params"][
                "minimize_vllm_memory"
            ]
            st.session_state.inference_params_loaded = True

        if "sampling_params" not in st.session_state:
            st.session_state.sampling_params = {}
        if "vllm_kwargs" not in st.session_state:
            st.session_state.vllm_kwargs = self.inference_config.vllm_kwargs
        if "multi_gen_results" not in st.session_state:
            st.session_state.multi_gen_results = None
        if "multi_gen_preset_prompt" not in st.session_state:
            st.session_state.multi_gen_preset_prompt = None
        if "multi_gen_preset_apply_template" not in st.session_state:
            st.session_state.multi_gen_preset_apply_template = None
        if "multi_gen_preset_messages" not in st.session_state:
            st.session_state.multi_gen_preset_messages = None

        saved_multigen_state = self.load_multigen_state()

        if "multi_gen_text_prompt" not in st.session_state:
            st.session_state.multi_gen_text_prompt = saved_multigen_state.get(
                "text_tab", {}
            ).get("prompt", "")
        if "multi_gen_template_mode" not in st.session_state:
            st.session_state.multi_gen_template_mode = saved_multigen_state.get(
                "text_tab", {}
            ).get("template_mode", "Apply chat template")
        if "multi_gen_assistant_prefill" not in st.session_state:
            st.session_state.multi_gen_assistant_prefill = saved_multigen_state.get(
                "text_tab", {}
            ).get("assistant_prefill", "")

        if "multi_gen_messages" not in st.session_state:
            st.session_state.multi_gen_messages = saved_multigen_state.get(
                "messages_tab", {}
            ).get("messages", [])
        if "msg_builder_template_override" not in st.session_state:
            st.session_state.msg_builder_template_override = saved_multigen_state.get(
                "messages_tab", {}
            ).get("template_override", "No template override")
        if "multi_gen_msg_editing_idx" not in st.session_state:
            st.session_state.multi_gen_msg_editing_idx = None

        if "multi_gen_active_tab" not in st.session_state:
            st.session_state.multi_gen_active_tab = saved_multigen_state.get(
                "active_tab", "Text"
            )

        # Multi-prompt generation state
        if "managed_prompts" not in st.session_state:
            st.session_state.managed_prompts = {}
        if "multi_prompt_results" not in st.session_state:
            st.session_state.multi_prompt_results = None
        if "multi_prompt_display_configs" not in st.session_state:
            st.session_state.multi_prompt_display_configs = []
        if "multi_gen_show_all" not in st.session_state:
            st.session_state.multi_gen_show_all = False
        if "multi_prompt_show_all" not in st.session_state:
            st.session_state.multi_prompt_show_all = False

        # Keyword highlighting state - list of {keywords: list[str], color: str, enabled: bool}
        if "highlight_selectors" not in st.session_state:
            st.session_state.highlight_selectors = self.load_highlight_selectors()

        if "multi_gen_prompt" not in st.session_state:
            st.session_state.multi_gen_prompt = saved_multigen_state.get("prompt", "")
        if "apply_chat_template_checkbox" not in st.session_state:
            st.session_state.apply_chat_template_checkbox = saved_multigen_state.get(
                "apply_chat_template", True
            )

        self._init_session_state_from_cache()

    def reload_all_data(self) -> None:
        """Reload all data from cache after HF sync."""
        # Reload folder state from disk
        loaded_folders, loaded_prompt_folders = self.load_loaded_folders()
        st.session_state.loaded_folders = loaded_folders
        st.session_state.loaded_prompt_folders = loaded_prompt_folders

        # Clear and reload configs
        st.session_state.managed_configs = {}
        existing_names: set[str] = set()
        for folder in st.session_state.loaded_folders:
            loaded = self.load_configs_from_folder(folder, existing_names)
            st.session_state.managed_configs.update(loaded)
            existing_names.update(mc.full_name for mc in loaded.values())

        # Clear and reload prompts
        st.session_state.managed_prompts = {}
        for folder in st.session_state.loaded_prompt_folders:
            loaded = self.load_prompts_from_folder(folder)
            st.session_state.managed_prompts.update(loaded)

        # Clear and reload conversations
        st.session_state.conversations = {}
        config_name_to_managed = {
            mc.full_name: mc for mc in st.session_state.managed_configs.values()
        }
        conversations, max_conv_num = self.load_conversations(config_name_to_managed)
        st.session_state.conversations.update(conversations)
        if max_conv_num >= 0:
            st.session_state.conversation_counter = max_conv_num + 1

    # === Config persistence ===

    def save_configs(self, deleted: tuple[str, str] | None = None) -> None:
        """Save all managed configs to their respective folders."""
        save_configs_to_cache(
            st.session_state.managed_configs, self.configs_dir, deleted
        )

    def save_configs_and_rerun(self, scope: str = "app") -> None:
        """Save configs and trigger a Streamlit rerun."""
        self.save_configs()
        st.rerun(scope=scope)

    def load_configs_from_folder(
        self, folder: str | None, existing_names: set[str]
    ) -> dict[str, ManagedConfig]:
        """Load configs from a specific folder."""
        return load_configs_from_folder(self.configs_dir, folder, existing_names)

    # === Prompt persistence ===

    def save_prompts(self, deleted: tuple[str, str] | None = None) -> None:
        """Save all managed prompts to their respective folders."""
        save_prompts_to_cache(
            st.session_state.managed_prompts, self.prompts_dir, deleted
        )

    def load_prompts_from_folder(self, folder: str | None) -> dict[str, ManagedPrompt]:
        """Load prompts from a specific folder."""
        return load_prompts_from_folder(self.prompts_dir, folder)

    # === Conversation persistence ===

    def load_conversations(
        self, config_name_to_managed: dict[str, ManagedConfig]
    ) -> tuple[dict[str, ManagedConversation], int]:
        """Load all conversations from cache.

        Returns:
            Tuple of (conversations dict, max conversation number)
        """
        conversations: dict[str, ManagedConversation] = {}
        max_conv_num = -1

        for conv_file in sorted(self.conversations_dir.glob("*.yaml")):
            conv = ManagedConversation.from_file(conv_file, config_name_to_managed)
            conv_num = int(conv.conv_id.split("_")[-1])
            max_conv_num = max(max_conv_num, conv_num)
            conversations[conv.conv_id] = conv

        return conversations, max_conv_num

    # === Folder state ===

    def save_loaded_folders(self) -> None:
        """Save loaded folders state to disk."""
        self.save_yaml(
            "loaded_folders.yaml",
            {
                "config_folders": list(st.session_state.loaded_folders),
                "prompt_folders": list(st.session_state.loaded_prompt_folders),
            },
        )

    def load_loaded_folders(self) -> tuple[set[str | None], set[str | None]]:
        """Load loaded folders state from disk, filtering out non-existent folders."""
        state_file = self.cache_dir / "loaded_folders.yaml"
        default_folders: set[str | None] = {None}
        if not state_file.exists():
            return default_folders, default_folders

        with open(state_file) as f:
            state = yaml.safe_load(f) or {}

        def normalize_folders(folder_list: list) -> set[str | None]:
            return {f if f else None for f in folder_list}

        loaded_folders = normalize_folders(state.get("loaded_folders", [None]))
        loaded_prompt_folders = normalize_folders(
            state.get("loaded_prompt_folders", [None])
        )

        if not loaded_folders:
            loaded_folders = {None}
        if not loaded_prompt_folders:
            loaded_prompt_folders = {None}

        # Filter out non-existent folders
        existing_config_folders: set[str | None] = set()
        for folder in loaded_folders:
            folder_path = self.configs_dir / folder if folder else self.configs_dir
            if folder_path.exists():
                existing_config_folders.add(folder)
        loaded_folders = existing_config_folders

        existing_prompt_folders: set[str | None] = set()
        for folder in loaded_prompt_folders:
            folder_path = self.prompts_dir / folder if folder else self.prompts_dir
            if folder_path.exists():
                existing_prompt_folders.add(folder)
        loaded_prompt_folders = existing_prompt_folders

        # Update state file if any folders were filtered out
        original_config_folders = normalize_folders(state.get("loaded_folders", [None]))
        original_prompt_folders = normalize_folders(
            state.get("loaded_prompt_folders", [None])
        )
        if (
            loaded_folders != original_config_folders
            or loaded_prompt_folders != original_prompt_folders
        ):
            self.save_yaml(
                "loaded_folders.yaml",
                {
                    "config_folders": list(loaded_folders),
                    "prompt_folders": list(loaded_prompt_folders),
                },
            )

        return loaded_folders, loaded_prompt_folders

    # === Multi-gen state ===

    def save_multigen_state(self) -> None:
        """Save multi-generation state from session_state to cache."""
        state = {
            "active_tab": st.session_state.get("multi_gen_active_tab", "Text"),
            "text_tab": {
                "prompt": st.session_state.get("multi_gen_text_prompt", ""),
                "template_mode": st.session_state.get(
                    "multi_gen_template_mode", "Apply chat template"
                ),
                "assistant_prefill": st.session_state.get(
                    "multi_gen_assistant_prefill", ""
                ),
            },
            "messages_tab": {
                "messages": st.session_state.get("multi_gen_messages", []),
                "template_override": st.session_state.get(
                    "msg_builder_template_override", "No template override"
                ),
            },
        }
        self.save_yaml("last_multigen_state.yaml", state)

    def load_multigen_state(self) -> dict:
        """Load multi-generation state from cache."""
        state_file = self.cache_dir / "last_multigen_state.yaml"
        default_state = {
            "active_tab": "Text",
            "text_tab": {"prompt": "", "template_mode": "Apply chat template"},
            "messages_tab": {
                "messages": [],
                "template_override": "No template override",
            },
            "prompt": "",
            "apply_chat_template": True,
        }
        if not state_file.exists():
            return default_state

        with open(state_file) as f:
            state = yaml.safe_load(f) or {}

        # Backward compat: old format -> new format
        if "text_tab" not in state:
            state["text_tab"] = {
                "prompt": state.get("prompt", ""),
                "template_mode": (
                    "Apply chat template"
                    if state.get("apply_chat_template", True)
                    else "No template"
                ),
            }
            state["messages_tab"] = {
                "messages": [],
                "template_override": "No template override",
            }
            state["active_tab"] = "Text"

        if "messages_tab" in state and "add_generation_prompt" in state["messages_tab"]:
            add_gen = state["messages_tab"]["add_generation_prompt"]
            state["messages_tab"]["template_override"] = (
                "Force generation prompt" if add_gen else "Force continue final message"
            )
            del state["messages_tab"]["add_generation_prompt"]

        state["prompt"] = state.get(
            "prompt", state.get("text_tab", {}).get("prompt", "")
        )
        state["apply_chat_template"] = state.get(
            "apply_chat_template",
            state.get("text_tab", {}).get("template_mode") == "Apply chat template",
        )
        return state

    # === Inference params ===

    def save_inference_params(self) -> None:
        """Save inference parameters from session_state to cache."""
        params = {
            "sampling_params": st.session_state.sampling_params,
            "vllm_params": {
                "gpu_memory_utilization": st.session_state.get(
                    "gpu_memory_utilization", 0.95
                ),
                "minimize_vllm_memory": st.session_state.get(
                    "minimize_vllm_memory", False
                ),
            },
        }
        self.save_yaml("inference_params.yaml", params)

    def load_inference_params(self) -> dict:
        """Load inference parameters from cache."""
        state_file = self.cache_dir / "inference_params.yaml"
        if not state_file.exists():
            return {
                "sampling_params": DEFAULT_SAMPLING_PARAMS.copy(),
                "vllm_params": DEFAULT_VLLM_PARAMS.copy(),
            }

        with open(state_file) as f:
            params = yaml.safe_load(f) or {}

        return {
            "sampling_params": {
                **DEFAULT_SAMPLING_PARAMS,
                **params.get("sampling_params", {}),
            },
            "vllm_params": {**DEFAULT_VLLM_PARAMS, **params.get("vllm_params", {})},
        }

    # === Highlight selectors ===

    def save_highlight_selectors(self, selectors: list[dict]) -> None:
        """Save highlight selectors to cache."""
        self.save_yaml("highlight_selectors.yaml", selectors)

    def load_highlight_selectors(self) -> list[dict]:
        """Load highlight selectors from cache."""
        state_file = self.cache_dir / "highlight_selectors.yaml"
        if not state_file.exists():
            return []
        with open(state_file) as f:
            selectors = yaml.safe_load(f) or []
        for selector in selectors:
            if "enabled" not in selector:
                selector["enabled"] = True
        return selectors


# ============ Inference Parameters Persistence ============


DEFAULT_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.9,
    "max_tokens": 180,
    "n": 6,
    "do_sample": True,
    "seed": 28,
    "skip_special_tokens": False,
}

DEFAULT_VLLM_PARAMS = {
    "gpu_memory_utilization": 0.95,
    "minimize_vllm_memory": False,
}
