"""
Dashboard state management for amplification UI.

Separates UI concerns (active state, ordering) from domain models (configs).
Also provides persistence functions for saving/loading dashboard state.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from nnterp import StandardizedTransformer

from src.diffing.methods.amplification.amplification_config import AmplificationConfig
from src.utils.data import dump_yaml_multiline, codenamize_hash


# ============ Dashboard State Classes ============


@dataclass
class DashboardItem:
    """Base class for items with UI state."""

    active: bool = True
    ui_order: int = 0
    expanded: bool = False

    def to_ui_dict(self) -> dict[str, Any]:
        """Serialize UI state."""
        return {
            "active": self.active,
            "ui_order": self.ui_order,
            "expanded": self.expanded,
        }

    @staticmethod
    def ui_dict_to_fields(data: dict[str, Any]) -> dict[str, Any]:
        """Extract UI fields from dict."""
        return {
            "active": data.get("active", True),
            "ui_order": data.get("ui_order", 0),
            "expanded": data.get("expanded", False),
        }


@dataclass
class ManagedConfig(DashboardItem):
    """Amplification config with dashboard state."""

    config: AmplificationConfig = None
    folder: str = ""  # Relative path from CONFIGS_DIR (empty string = root)
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _last_compiled_hash: str | None = field(default=None, init=False)
    _lora_int_id: int = field(default=-1, init=False)

    def __post_init__(self):
        self._update_lora_int_id()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    def _update_lora_int_id(self):
        """Derive lora_int_id from compiled hash if available, else config_id (vLLM needs an integer)."""
        hash_source = self._last_compiled_hash or self.config_id
        self._lora_int_id = abs(hash(hash_source)) % (2**31)

    @property
    def lora_int_id(self) -> int:
        """Get the LORA int ID for vLLM (derived from compiled hash or config_id)."""
        return self._lora_int_id

    @property
    def last_compiled_hash(self) -> str | None:
        """Get the last compiled hash."""
        return self._last_compiled_hash

    @last_compiled_hash.setter
    def last_compiled_hash(self, value: str | None):
        """Set the last compiled hash and update LORA int ID if changed."""
        if value != self._last_compiled_hash:
            self._last_compiled_hash = value
            self._update_lora_int_id()

    def to_dict(self) -> dict[str, Any]:
        """Serialize both UI state and config."""
        result = self.to_ui_dict()
        result["config"] = self.config.to_dict()
        result["folder"] = self.folder
        result["config_id"] = self.config_id
        return result

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ManagedConfig":
        """Deserialize from dict."""
        ui_fields = DashboardItem.ui_dict_to_fields(data)
        config = AmplificationConfig.from_dict(data["config"])
        # Support legacy format where config_id was stored in config dict
        config_id = (
            data.get("config_id")
            or data["config"].get("config_id")
            or str(uuid.uuid4())
        )

        return ManagedConfig(
            config=config,
            folder=data.get("folder", ""),
            config_id=config_id,
            **ui_fields,
        )

    @staticmethod
    def from_config(
        config: AmplificationConfig,
        active: bool = True,
        expanded: bool = True,
        ui_order: int = 0,
        folder: str = "",
    ) -> "ManagedConfig":
        """Create from a pure config (e.g., when loading external config)."""
        return ManagedConfig(
            config=config,
            folder=folder,
            config_id=str(uuid.uuid4()),
            active=active,
            ui_order=ui_order,
            expanded=expanded,
        )

    def compile(
        self,
        compiled_dir: Path,
        base_model_name: str,
        base_model: StandardizedTransformer,
    ) -> Path | None:
        path, hash = self.config.compile(
            compiled_dir,
            base_model_name=base_model_name,
            base_model=base_model,
        )
        if hash is not None:
            assert path is not None
            self.last_compiled_hash = hash
        return path


@dataclass
class ManagedPrompt(DashboardItem):
    """Prompt with dashboard state for multi-prompt generation."""

    prompt_id: str = field(default_factory=lambda: "")
    name: str = ""  # Display name (optional, auto-generated from text if empty)

    # Editor mode (matches multi-gen tabs)
    editor_mode: Literal["simple", "chat"] = "simple"

    # Simple mode fields
    prompt_text: str = ""
    template_mode: str = "Apply chat template"
    system_prompt: str = ""  # Optional system prompt for chat template mode
    assistant_prefill: str = ""
    loom_filename: str = "untitled.txt"  # Filename for loom template mode

    # Chat mode fields
    messages: list[dict] = field(default_factory=list)

    folder: str = ""

    def __post_init__(self):
        if not self.prompt_id:
            self.prompt_id = str(uuid.uuid4())

    def get_display_name(self) -> str:
        """Return name or truncated prompt text."""
        if self.name:
            return self.name
        if self.editor_mode == "simple":
            text = self.prompt_text
        else:
            text = self.messages[0]["content"] if self.messages else ""
        return (text[:50] + "...") if len(text) > 50 else text

    def to_dict(self) -> dict[str, Any]:
        """Serialize both UI state and prompt data."""
        result = self.to_ui_dict()
        result.update(
            {
                "prompt_id": self.prompt_id,
                "name": self.name,
                "editor_mode": self.editor_mode,
                "prompt_text": self.prompt_text,
                "template_mode": self.template_mode,
                "system_prompt": self.system_prompt,
                "assistant_prefill": self.assistant_prefill,
                "loom_filename": self.loom_filename,
                "messages": self.messages,
                "folder": self.folder,
            }
        )
        return result

    def duplicate(self) -> "ManagedPrompt":
        return ManagedPrompt(
            name=f"{self.name} copy" if self.name else "",
            editor_mode=self.editor_mode,
            prompt_text=self.prompt_text,
            template_mode=self.template_mode,
            system_prompt=self.system_prompt,
            assistant_prefill=self.assistant_prefill,
            loom_filename=self.loom_filename,
            messages=deepcopy(self.messages),
            folder=self.folder,
            active=self.active,
            expanded=True,
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ManagedPrompt":
        """Deserialize from dict."""
        ui_fields = DashboardItem.ui_dict_to_fields(data)
        return ManagedPrompt(
            prompt_id=data.get("prompt_id", ""),
            name=data.get("name", ""),
            editor_mode=data.get("editor_mode", "simple"),
            prompt_text=data.get("prompt_text", ""),
            template_mode=data.get("template_mode", "Apply chat template"),
            system_prompt=data.get("system_prompt", ""),
            assistant_prefill=data.get("assistant_prefill", ""),
            loom_filename=data.get("loom_filename", "untitled.txt"),
            messages=data.get("messages", []),
            folder=data.get("folder", ""),
            **ui_fields,
        )


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


# ============ Utility Functions ============


def sanitize_config_name(name: str) -> str:
    """
    Sanitize a config name so it can be used as both display text and filename.

    Args:
        name: Desired config name input by the user

    Returns:
        Sanitized name containing only safe characters
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_\- ]+", "_", name).strip()
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


# ============ Persistence Functions ============


UI_STATE_FILENAME = "_ui_state.yaml"


def save_configs_to_folder(
    managed_configs: dict[str, ManagedConfig],
    configs_dir: Path,
    folder: str,
    deleted_names: set[str] | None = None,
) -> None:
    """
    Save configs belonging to a specific folder.

    Args:
        managed_configs: Dict of config_id -> ManagedConfig (all configs)
        configs_dir: Base configs directory
        folder: Relative folder path (empty string for root)
        deleted_names: Set of config names that were explicitly deleted (only these get moved to removed/)
    """
    folder_path = configs_dir / folder if folder else configs_dir
    folder_path.mkdir(parents=True, exist_ok=True)

    folder_configs = {
        cid: mc for cid, mc in managed_configs.items() if mc.folder == folder
    }

    ui_state = {}
    for mc in folder_configs.values():
        safe_name = mc.config.name
        config_path = folder_path / f"{safe_name}.yaml"
        mc.config.save_yaml(config_path)
        ui_state[safe_name] = mc.to_ui_dict()

    ui_state_path = folder_path / UI_STATE_FILENAME
    with open(ui_state_path, "w") as f:
        yaml.dump(ui_state, f, sort_keys=False)

    # Only move explicitly deleted files to removed/
    if deleted_names:
        removed_dir = folder_path / "removed"
        removed_dir.mkdir(parents=True, exist_ok=True)

        for deleted_name in deleted_names:
            config_file = folder_path / f"{deleted_name}.yaml"
            if config_file.exists():
                target = removed_dir / config_file.name
                if target.exists():
                    target.unlink()
                config_file.replace(target)


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
    folder: str,
    existing_names: set[str] | None = None,
) -> dict[str, ManagedConfig]:
    """
    Load configs from a specific folder.

    Args:
        configs_dir: Base configs directory
        folder: Relative folder path (empty string for root)
        existing_names: Optional set of existing names to avoid duplicates

    Returns:
        Dict of config_id -> ManagedConfig
    """
    existing_names = existing_names or set()
    managed_configs = {}

    folder_path = configs_dir / folder if folder else configs_dir
    if not folder_path.exists():
        return managed_configs

    ui_state_path = folder_path / UI_STATE_FILENAME
    ui_state = {}
    if ui_state_path.exists():
        with open(ui_state_path) as f:
            ui_state = yaml.safe_load(f) or {}

    for config_file in sorted(folder_path.glob("*.yaml")):
        if config_file.name == UI_STATE_FILENAME:
            continue
        loaded_config = AmplificationConfig.load_yaml(config_file)
        original_name = loaded_config.name
        loaded_config.name = get_unique_name(
            sanitize_config_name(loaded_config.name), existing_names
        )
        existing_names.add(loaded_config.name)

        config_ui_state = ui_state.get(original_name, {})
        active = config_ui_state.get("active", True)
        expanded = config_ui_state.get("expanded", False)
        ui_order = config_ui_state.get("ui_order", 0)

        managed_config = ManagedConfig.from_config(
            loaded_config,
            active=active,
            expanded=expanded,
            ui_order=ui_order,
            folder=folder,
        )
        managed_configs[managed_config.config_id] = managed_config

    return managed_configs


def load_configs_from_cache(
    configs_dir: Path,
    existing_names: set[str] | None = None,
) -> dict[str, ManagedConfig]:
    """
    Load configs from root folder only (backward compatibility).

    Args:
        configs_dir: Base configs directory
        existing_names: Optional set of existing names to avoid duplicates

    Returns:
        Dict of config_id -> ManagedConfig
    """
    return load_configs_from_folder(configs_dir, "", existing_names)


# ============ Folder Utility Functions ============


def list_all_folders(configs_dir: Path) -> list[str]:
    """
    List all available folder paths recursively.

    Args:
        configs_dir: Base configs directory

    Returns:
        List of relative folder paths (empty string for root, then nested paths)
    """
    folders = [""]  # Root folder
    if not configs_dir.exists():
        return folders

    for item in sorted(configs_dir.rglob("*")):
        if item.is_dir() and item.name != "removed":
            rel_path = str(item.relative_to(configs_dir))
            folders.append(rel_path)

    return folders


def create_folder(configs_dir: Path, folder_path: str) -> Path:
    """
    Create a new folder under configs_dir.

    Args:
        configs_dir: Base configs directory
        folder_path: Relative path for the new folder

    Returns:
        Path to the created folder
    """
    assert folder_path, "Folder path cannot be empty"
    new_folder = configs_dir / folder_path
    new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder


def unload_folder_configs(
    managed_configs: dict[str, ManagedConfig],
    folder: str,
) -> dict[str, ManagedConfig]:
    """
    Remove configs belonging to a specific folder from the managed configs dict.

    Args:
        managed_configs: Dict of config_id -> ManagedConfig
        folder: Folder to unload

    Returns:
        New dict with folder's configs removed
    """
    return {cid: mc for cid, mc in managed_configs.items() if mc.folder != folder}


# ============ Prompt Persistence Functions ============


def save_prompts_to_folder(
    managed_prompts: dict[str, ManagedPrompt],
    prompts_dir: Path,
    folder: str,
    deleted_names: set[str] | None = None,
) -> None:
    """Save prompts belonging to a specific folder.

    Args:
        managed_prompts: Dict of prompt_id -> ManagedPrompt (all prompts)
        prompts_dir: Base prompts directory
        folder: Relative folder path (empty string for root)
        deleted_names: Set of prompt display names that were explicitly deleted (only these get moved to removed/)
    """
    folder_path = prompts_dir / folder if folder else prompts_dir
    folder_path.mkdir(parents=True, exist_ok=True)

    folder_prompts = {
        pid: mp for pid, mp in managed_prompts.items() if mp.folder == folder
    }

    ui_state = {}
    for mp in folder_prompts.values():
        safe_name = sanitize_config_name(mp.get_display_name()) or mp.prompt_id[:8]
        prompt_path = folder_path / f"{safe_name}.yaml"
        with open(prompt_path, "w") as f:
            yaml.dump(mp.to_dict(), f, sort_keys=False)
        ui_state[safe_name] = mp.to_ui_dict()

    ui_state_path = folder_path / UI_STATE_FILENAME
    with open(ui_state_path, "w") as f:
        yaml.dump(ui_state, f, sort_keys=False)

    # Only move explicitly deleted files to removed/
    if deleted_names:
        removed_dir = folder_path / "removed"
        removed_dir.mkdir(parents=True, exist_ok=True)

        for deleted_name in deleted_names:
            # Sanitize the name the same way we do when saving
            safe_deleted_name = sanitize_config_name(deleted_name) or deleted_name[:8]
            prompt_file = folder_path / f"{safe_deleted_name}.yaml"
            if prompt_file.exists():
                target = removed_dir / prompt_file.name
                if target.exists():
                    target.unlink()
                prompt_file.replace(target)


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
    folder: str,
    existing_names: set[str] | None = None,
) -> dict[str, ManagedPrompt]:
    """Load prompts from a specific folder."""
    existing_names = existing_names or set()
    managed_prompts = {}

    folder_path = prompts_dir / folder if folder else prompts_dir
    if not folder_path.exists():
        return managed_prompts

    for prompt_file in sorted(folder_path.glob("*.yaml")):
        if prompt_file.name == UI_STATE_FILENAME:
            continue
        with open(prompt_file) as f:
            data = yaml.safe_load(f) or {}
        data["folder"] = folder
        mp = ManagedPrompt.from_dict(data)
        managed_prompts[mp.prompt_id] = mp

    return managed_prompts


def load_prompts_from_cache(
    prompts_dir: Path,
    existing_names: set[str] | None = None,
) -> dict[str, ManagedPrompt]:
    """Load prompts from root folder only."""
    return load_prompts_from_folder(prompts_dir, "", existing_names)


def unload_folder_prompts(
    managed_prompts: dict[str, ManagedPrompt],
    folder: str,
) -> dict[str, ManagedPrompt]:
    """Remove prompts belonging to a specific folder."""
    return {pid: mp for pid, mp in managed_prompts.items() if mp.folder != folder}


def list_all_prompt_folders(prompts_dir: Path) -> list[str]:
    """List all available prompt folder paths recursively."""
    folders = [""]
    if not prompts_dir.exists():
        return folders

    for item in sorted(prompts_dir.rglob("*")):
        if item.is_dir() and item.name != "removed":
            rel_path = str(item.relative_to(prompts_dir))
            folders.append(rel_path)

    return folders


def save_multigen_state(state_file: Path, state: dict) -> None:
    """
    Save multi-generation state to cache file.

    Args:
        state_file: Path to save state to
        state: Dict with multi-gen state
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        yaml.dump(state, f, sort_keys=False)


def load_multigen_state(state_file: Path) -> dict:
    """
    Load multi-generation state from cache file.

    Args:
        state_file: Path to load state from

    Returns:
        Dict with multi-gen state (or defaults if file doesn't exist)
    """
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

    # Handle backward compatibility
    if "text_tab" not in state:
        prompt = state.get("prompt", "")
        apply_template = state.get("apply_chat_template", True)
        state["text_tab"] = {
            "prompt": prompt,
            "template_mode": "Apply chat template" if apply_template else "No template",
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

    state["prompt"] = state.get("prompt", state.get("text_tab", {}).get("prompt", ""))
    state["apply_chat_template"] = state.get(
        "apply_chat_template",
        state.get("text_tab", {}).get("template_mode") == "Apply chat template",
    )

    return state


def save_loaded_folders(
    state_file: Path, loaded_folders: set[str], loaded_prompt_folders: set[str]
) -> None:
    """Save loaded folders state to cache file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "loaded_folders": list(loaded_folders),
        "loaded_prompt_folders": list(loaded_prompt_folders),
    }
    with open(state_file, "w") as f:
        yaml.dump(state, f, sort_keys=False)


def load_loaded_folders(
    state_file: Path, configs_dir: Path | None = None, prompts_dir: Path | None = None
) -> tuple[set[str], set[str]]:
    """Load loaded folders state from cache file, filtering out non-existent folders.

    Args:
        state_file: Path to the state file
        configs_dir: Base configs directory (if provided, checks folder existence)
        prompts_dir: Base prompts directory (if provided, checks folder existence)

    Returns:
        Tuple of (config folders set, prompt folders set) - only existing folders
    """
    default_folders = {""}  # Root folder
    if not state_file.exists():
        return default_folders, default_folders

    with open(state_file) as f:
        state = yaml.safe_load(f) or {}

    loaded_folders = set(state.get("loaded_folders", [""]))
    loaded_prompt_folders = set(state.get("loaded_prompt_folders", [""]))

    # Ensure at least root is included if empty
    if not loaded_folders:
        loaded_folders = {""}
    if not loaded_prompt_folders:
        loaded_prompt_folders = {""}

    # Filter out non-existent folders
    if configs_dir:
        existing_config_folders = set()
        for folder in loaded_folders:
            folder_path = configs_dir / folder if folder else configs_dir
            if folder_path.exists():
                existing_config_folders.add(folder)
        loaded_folders = existing_config_folders

    if prompts_dir:
        existing_prompt_folders = set()
        for folder in loaded_prompt_folders:
            folder_path = prompts_dir / folder if folder else prompts_dir
            if folder_path.exists():
                existing_prompt_folders.add(folder)
        loaded_prompt_folders = existing_prompt_folders

    # Update state file if any folders were filtered out
    if configs_dir or prompts_dir:
        original_config_folders = set(state.get("loaded_folders", [""]))
        original_prompt_folders = set(state.get("loaded_prompt_folders", [""]))

        if (
            loaded_folders != original_config_folders
            or loaded_prompt_folders != original_prompt_folders
        ):
            save_loaded_folders(state_file, loaded_folders, loaded_prompt_folders)

    return loaded_folders, loaded_prompt_folders


def save_conversation(
    conv_id: str,
    conv: dict[str, Any],
    conversations_dir: Path,
) -> None:
    """
    Save a single conversation to disk.

    Args:
        conv_id: The conversation ID
        conv: The conversation data
        conversations_dir: Directory to save to
    """
    conversations_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv["name"].replace("/", "_").replace(":", "_")
    conv_path = conversations_dir / f"{safe_name}.yaml"

    config = conv["context"]["config"]
    serialized_conv = {
        "conv_id": conv_id,
        "name": conv["name"],
        "context": {
            "config_name": config.name if config else None,
            "system_prompt": conv["context"].get("system_prompt", ""),
        },
        "history": conv["history"],
        "editing_message": conv["editing_message"],
        "regenerating_from": conv["regenerating_from"],
        "continuing_from": conv.get("continuing_from"),
    }

    with open(conv_path, "w") as f:
        yaml.dump(serialized_conv, f, sort_keys=False)


def load_conversations_from_cache(
    conversations_dir: Path,
    config_name_to_managed: dict[str, "ManagedConfig"],
) -> tuple[dict[str, dict[str, Any]], int]:
    """
    Load all conversations from the cache directory.

    Args:
        conversations_dir: Directory to load from
        config_name_to_managed: Mapping from config name to ManagedConfig

    Returns:
        Tuple of (conversations dict, max conversation number)
    """
    conversations = {}
    max_conv_num = -1

    for conv_file in sorted(conversations_dir.glob("*.yaml")):
        with open(conv_file) as f:
            serialized_conv = yaml.safe_load(f)

        conv_id = serialized_conv["conv_id"]
        conv_num = int(conv_id.split("_")[-1])
        max_conv_num = max(max_conv_num, conv_num)

        config_name = serialized_conv["context"]["config_name"]
        if config_name and config_name in config_name_to_managed:
            config = config_name_to_managed[config_name]
        else:
            config = None

        conv = {
            "name": serialized_conv["name"],
            "context": {
                "config": config,
                "system_prompt": serialized_conv["context"].get("system_prompt", ""),
            },
            "history": serialized_conv["history"],
            "editing_message": serialized_conv["editing_message"],
            "regenerating_from": serialized_conv["regenerating_from"],
            "continuing_from": serialized_conv.get("continuing_from"),
        }

        conversations[conv_id] = conv

    return conversations, max_conv_num


def delete_conversation_file(conv_name: str, conversations_dir: Path) -> None:
    """
    Delete a conversation file from disk.

    Args:
        conv_name: Name of the conversation
        conversations_dir: Directory containing conversation files
    """
    safe_name = conv_name.replace("/", "_").replace(":", "_")
    conv_path = conversations_dir / f"{safe_name}.yaml"
    if conv_path.exists():
        conv_path.unlink()


# ============ Generation Logging ============


@dataclass
class GenerationLog:
    """
    Log entry for a single config's generation.

    Stores all information needed to reproduce a generation and provides
    multi-view directory organization (by_prompt, by_config, by_time).
    """

    generation_type: Literal["multigen", "chat", "continue", "regenerate"]
    model_id: str
    prompt_text: str
    prompt_tokens: list[int]
    sampling_params: dict[str, Any]
    config: dict[str, Any]  # Serialized config dict with compiled_hash
    outputs: list[str]  # Output samples for this config
    messages: list[dict[str, str]] | None = None
    template_mode: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @staticmethod
    def from_dashboard_generation(
        generation_type: str,
        model_id: str,
        prompt_text: str,
        prompt_tokens: list[int],
        sampling_params: Any,  # SamplingParams object
        configs: list["ManagedConfig"],
        results: list[dict[str, Any]],
        messages: list[dict[str, str]] | None = None,
        template_mode: str | None = None,
        logs_dir: Path | None = None,
    ) -> list["GenerationLog"]:
        """
        Factory method to create GenerationLogs from dashboard generation parameters.

        Creates one GenerationLog per config. If logs_dir is provided, saves them directly.

        Args:
            generation_type: Type of generation ("multigen", "chat", "continue", "regenerate")
            model_id: Model ID string
            prompt_text: Human-readable prompt text
            prompt_tokens: Tokenized prompt
            sampling_params: SamplingParams object
            configs: List of ManagedConfig objects
            results: List of result dicts with "config_name" and "outputs" keys
            messages: Optional list of chat messages
            template_mode: Optional template mode used
            logs_dir: If provided, saves logs directly to this directory

        Returns:
            List of GenerationLog instances (one per config)
        """
        # Build sampling params dict
        sampling_dict = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
            "n": sampling_params.n,
        }
        if hasattr(sampling_params, "seed") and sampling_params.seed is not None:
            sampling_dict["seed"] = sampling_params.seed

        # Use same timestamp for all logs in this batch
        timestamp = datetime.now()

        logs = []
        for mc, result in zip(configs, results):
            config_dict = mc.config.to_dict()
            config_dict["compiled_hash"] = mc.last_compiled_hash or mc.config_id

            log = GenerationLog(
                generation_type=generation_type,
                model_id=model_id,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_dict,
                config=config_dict,
                outputs=result["outputs"],
                messages=messages,
                template_mode=template_mode,
                timestamp=timestamp,
            )
            logs.append(log)

            if logs_dir is not None:
                log.save(logs_dir)

        return logs

    def _build_summary(self) -> dict[str, Any]:
        """Build the summary section for easy reading."""
        return {
            "config": self.config["name"],
            "prompt": self.prompt_text,
            "completions": self.outputs,
        }

    def _compute_prompt_hash(self) -> str:
        """Compute hash of the prompt for directory naming."""
        return hashlib.sha256(self.prompt_text.encode()).hexdigest()

    def _safe_filename(self, name: str) -> str:
        """Sanitize a string for use in filenames."""
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:50]

    def save(self, logs_dir: Path) -> Path:
        """
        Save the generation log with multi-view directory structure.

        Creates:
        - by_prompt/{prompt_codename}/{config_codename}/{timestamp}.yaml (actual file)
        - by_config/{config_codename}/{prompt_codename}/{timestamp}.yaml (symlink)
        - by_time/{date}/{timestamp}_{prompt}_{config}.yaml (symlink)

        Args:
            logs_dir: Base directory for generation logs

        Returns:
            Path to the actual log file
        """
        prompt_hash = self._compute_prompt_hash()
        prompt_codename = codenamize_hash(prompt_hash)
        timestamp_str = self.timestamp.strftime("%H-%M-%S")
        date_str = self.timestamp.strftime("%Y-%m-%d")

        config_name = self.config["name"]
        compiled_hash = self.config.get(
            "compiled_hash", self.config.get("config_id", "unknown")
        )
        config_codename = (
            f"{self._safe_filename(config_name)}-{codenamize_hash(compiled_hash)}"
        )

        # Build the log data in order
        log_data = {
            "timestamp": self.timestamp.isoformat(),
            "generation_type": self.generation_type,
            "model_id": self.model_id,
            "summary": self._build_summary(),
            "prompt": {
                "text": self.prompt_text,
                "tokens": self.prompt_tokens,
                "hash": prompt_hash,
                "codename": prompt_codename,
            },
            "sampling_params": self.sampling_params,
            "config": self.config,
            "outputs": self.outputs,
        }
        if self.messages is not None:
            log_data["prompt"]["messages"] = self.messages
        if self.template_mode is not None:
            log_data["prompt"]["template_mode"] = self.template_mode

        # Primary path: by_prompt (actual file)
        by_prompt_dir = logs_dir / "by_prompt" / prompt_codename / config_codename
        by_prompt_dir.mkdir(parents=True, exist_ok=True)
        by_prompt_file = by_prompt_dir / f"{timestamp_str}.yaml"
        with open(by_prompt_file, "w") as f:
            dump_yaml_multiline(log_data, f)

        # Secondary path: by_config (symlink)
        by_config_dir = logs_dir / "by_config" / config_codename / prompt_codename
        by_config_dir.mkdir(parents=True, exist_ok=True)
        by_config_file = by_config_dir / f"{timestamp_str}.yaml"
        if not by_config_file.exists():
            os.symlink(by_prompt_file, by_config_file)

        # Tertiary path: by_time (symlink)
        by_time_dir = logs_dir / "by_time" / date_str
        by_time_dir.mkdir(parents=True, exist_ok=True)
        by_time_filename = f"{timestamp_str}_{prompt_codename}_{config_codename}.yaml"
        by_time_file = by_time_dir / by_time_filename
        if not by_time_file.exists():
            os.symlink(by_prompt_file, by_time_file)

        return by_prompt_file
