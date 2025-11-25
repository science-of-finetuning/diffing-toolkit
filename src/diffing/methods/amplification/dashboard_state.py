"""
Dashboard state management for amplification UI.

Separates UI concerns (active state, ordering) from domain models (configs).
Also provides persistence functions for saving/loading dashboard state.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Set

import yaml
from nnterp import StandardizedTransformer

from src.diffing.methods.amplification.amplification_config import AmplificationConfig


# ============ Dashboard State Classes ============


@dataclass
class DashboardItem:
    """Base class for items with UI state."""

    active: bool = True
    ui_order: int = 0
    expanded: bool = False

    def to_ui_dict(self) -> Dict[str, Any]:
        """Serialize UI state."""
        return {
            "active": self.active,
            "ui_order": self.ui_order,
            "expanded": self.expanded,
        }

    @staticmethod
    def ui_dict_to_fields(data: Dict[str, Any]) -> Dict[str, Any]:
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
    _last_compiled_hash: str | None = field(default=None, init=False)
    _lora_int_id: int = field(default=-1, init=False)

    def __post_init__(self):
        self._update_lora_int_id()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    def _update_lora_int_id(self):
        """Derive lora_int_id from config_id hash (vLLM needs an integer)."""
        self._lora_int_id = abs(hash(self.config.config_id)) % (2**31)

    @property
    def lora_int_id(self) -> int:
        """Get the LORA int ID for vLLM (derived from config_id)."""
        return self._lora_int_id

    @property
    def last_compiled_hash(self) -> str | None:
        """Get the last compiled hash."""
        return self._last_compiled_hash

    @last_compiled_hash.setter
    def last_compiled_hash(self, value: str | None):
        """Set the last compiled hash and update LORA int ID if changed."""
        if value != self._last_compiled_hash:
            self._update_lora_int_id()
        self._last_compiled_hash = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize both UI state and config."""
        result = self.to_ui_dict()
        result["config"] = self.config.to_dict()
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ManagedConfig":
        """Deserialize from dict."""
        ui_fields = DashboardItem.ui_dict_to_fields(data)
        config = AmplificationConfig.from_dict(data["config"])

        return ManagedConfig(
            config=config,
            **ui_fields,
        )

    @staticmethod
    def from_config(
        config: AmplificationConfig, active: bool = True, expanded: bool = True
    ) -> "ManagedConfig":
        """Create from a pure config (e.g., when loading external config)."""
        return ManagedConfig(
            config=config,
            active=active,
            ui_order=0,
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)


@dataclass
class DashboardSession:
    """Complete dashboard session state (for save/restore)."""

    managed_configs: Dict[str, ManagedConfig] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session."""
        return {
            "managed_configs": {
                config_id: mc.to_dict()
                for config_id, mc in self.managed_configs.items()
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DashboardSession":
        """Deserialize session."""
        configs_data = data.get("managed_configs", {})
        if isinstance(configs_data, list):
            managed_configs = {
                mc_data["config"]["config_id"]: ManagedConfig.from_dict(mc_data)
                for mc_data in configs_data
            }
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


def get_unique_name(desired_name: str, existing_names: Set[str]) -> str:
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


def save_configs_to_cache(
    managed_configs: Dict[str, ManagedConfig],
    configs_dir: Path,
) -> None:
    """
    Save all managed configs to the cache directory.

    Args:
        managed_configs: Dict of config_id -> ManagedConfig
        configs_dir: Directory to save configs to
    """
    configs_dir.mkdir(parents=True, exist_ok=True)

    current_config_names = set()
    for mc in managed_configs.values():
        safe_name = mc.config.name
        config_path = configs_dir / f"{safe_name}.yaml"
        mc.config.save_yaml(config_path)
        current_config_names.add(f"{safe_name}.yaml")

    removed_dir = configs_dir / "removed"
    removed_dir.mkdir(parents=True, exist_ok=True)

    for config_file in configs_dir.glob("*.yaml"):
        if config_file.name not in current_config_names:
            target = removed_dir / config_file.name
            if target.exists():
                target.unlink()
            config_file.replace(target)


def load_configs_from_cache(
    configs_dir: Path,
    existing_names: Set[str] | None = None,
) -> Dict[str, ManagedConfig]:
    """
    Load configs from the cache directory.

    Args:
        configs_dir: Directory to load configs from
        existing_names: Optional set of existing names to avoid duplicates

    Returns:
        Dict of config_id -> ManagedConfig
    """
    existing_names = existing_names or set()
    managed_configs = {}

    for config_file in sorted(configs_dir.glob("*.yaml")):
        loaded_config = AmplificationConfig.load_yaml(config_file)
        loaded_config.name = get_unique_name(
            sanitize_config_name(loaded_config.name), existing_names
        )
        existing_names.add(loaded_config.name)
        managed_config = ManagedConfig.from_config(
            loaded_config, active=True, expanded=False
        )
        managed_configs[managed_config.config_id] = managed_config

    return managed_configs


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


def save_conversation(
    conv_id: str,
    conv: Dict[str, Any],
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
            "compiled_path": (
                str(conv["context"]["compiled_path"])
                if conv["context"]["compiled_path"]
                else None
            ),
        },
        "history": conv["history"],
        "editing_message": conv["editing_message"],
        "regenerating_from": conv["regenerating_from"],
    }

    with open(conv_path, "w") as f:
        yaml.dump(serialized_conv, f, sort_keys=False)


def load_conversations_from_cache(
    conversations_dir: Path,
    config_name_to_managed: Dict[str, "ManagedConfig"],
) -> tuple[Dict[str, Dict[str, Any]], int]:
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

        compiled_path_str = serialized_conv["context"]["compiled_path"]
        compiled_path = Path(compiled_path_str) if compiled_path_str else None

        conv = {
            "name": serialized_conv["name"],
            "context": {
                "config": config,
                "compiled_path": compiled_path,
            },
            "history": serialized_conv["history"],
            "editing_message": serialized_conv["editing_message"],
            "regenerating_from": serialized_conv["regenerating_from"],
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
