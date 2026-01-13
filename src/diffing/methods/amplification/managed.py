"""
Streamlit-independent managed classes for amplification configs, prompts, and conversations.

This module contains domain models that can be used without streamlit dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from copy import deepcopy
import logging
from pathlib import Path
import re
from typing import Any, Literal, Self
import uuid

from pathvalidate import sanitize_filename
import yaml
from nnterp import StandardizedTransformer

from .amplification_config import AmplificationConfig

logger = logging.getLogger(__name__)


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
    existing_names: set[str],
    desired_name: str,
    folder: str | None = None,
) -> str:
    """Get a unique name for an item, avoiding collisions with existing names.

    Args:
        existing_names: Set of existing full names (folder/name format) to avoid
        desired_name: The desired name for the item
        folder: Optional folder prefix (for configs/prompts)

    Returns:
        Unique name (without folder prefix if folder was provided)
    """
    sanitized_name = sanitize_config_name(desired_name)

    desired_full = f"{folder}/{sanitized_name}" if folder else sanitized_name
    unique_full = get_unique_name(desired_full, existing_names)

    # Extract just the name part (remove folder prefix if present)
    if folder and unique_full.startswith(f"{folder}/"):
        return unique_full[len(folder) + 1 :]
    return unique_full


@dataclass
class DashboardItem(ABC):
    """Base class for managed items with UI state and disk persistence.

    Provides shared functionality for:
    - UI state (active, expanded, ui_order)
    - Disk operations (save/load with rollback safety)
    - Name management (display name, disk name, full name with folder)
    - Rename tracking for proper file cleanup
    """

    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    folder: str | None = None
    active: bool = True
    ui_order: int = 0
    expanded: bool = False

    # === Abstract methods (must be implemented by subclasses) ===

    @abstractmethod
    def get_display_name(self) -> str:
        """Get the human-readable display name for this item."""
        ...

    @abstractmethod
    def to_content_dict(self) -> dict[str, Any]:
        """Serialize domain content (not UI state) for disk storage."""
        ...

    @classmethod
    @abstractmethod
    def from_content_dict(
        cls,
        data: dict[str, Any],
        folder: str | None,
        item_id: str | None = None,
        **ui_kwargs,
    ) -> Self:
        """Create instance from loaded content dict.

        Args:
            data: The loaded YAML data
            folder: Folder this item belongs to
            item_id: Optional item ID (generated if not provided)
            **ui_kwargs: UI state fields (active, expanded, ui_order)
        """
        ...

    @abstractmethod
    def _set_display_name(self, name: str) -> None:
        """Set the display name (implementation varies by subclass)."""
        ...

    # === Shared implementations ===

    def get_disk_name(self) -> str:
        """Get the sanitized name for disk storage.

        Returns sanitized display name, or truncated item_id as fallback.
        """
        return sanitize_config_name(self.get_display_name()) or self.item_id[:8]

    @property
    def full_name(self) -> str:
        """Get the full name including folder prefix for uniqueness."""
        name = self.get_display_name()
        if self.folder:
            return f"{self.folder}/{name}"
        return name

    def to_ui_dict(self) -> dict[str, Any]:
        """Serialize UI state."""
        return {
            "active": self.active,
            "ui_order": self.ui_order,
            "expanded": self.expanded,
        }

    def apply_ui_state(self, ui_state: dict[str, Any]) -> None:
        """Apply UI state from a dict (e.g., loaded from _ui_state.yaml)."""
        self.active = ui_state.get("active", self.active)
        self.expanded = ui_state.get("expanded", self.expanded)
        self.ui_order = ui_state.get("ui_order", self.ui_order)

    @staticmethod
    def ui_dict_to_fields(data: dict[str, Any]) -> dict[str, Any]:
        """Extract UI fields from dict."""
        return {
            "active": data.get("active", True),
            "ui_order": data.get("ui_order", 0),
            "expanded": data.get("expanded", False),
        }

    def save_content(self, path: Path) -> None:
        """Save domain content to path with rollback safety.

        If writing fails, restores the original file content.
        """
        old_data = None
        if path.exists():
            with open(path, "r") as f:
                old_data = f.read()
        try:
            with open(path, "w") as f:
                yaml.safe_dump(self.to_content_dict(), f, sort_keys=False)
        except Exception as e:
            logger.error(f"Error saving {self.__class__.__name__} to {path}: {e}")
            if old_data is not None:
                logger.error("Restoring old data")
                with open(path, "w") as f:
                    f.write(old_data)
            raise

    @classmethod
    def load_content(
        cls,
        path: Path,
        folder: str | None,
        ui_state: dict[str, Any] | None = None,
    ) -> Self:
        """Load item from path with optional UI state.

        Args:
            path: Path to the YAML file
            folder: Folder this item belongs to
            ui_state: Optional UI state dict to apply

        Returns:
            Loaded instance with UI state applied
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        ui_kwargs = cls.ui_dict_to_fields(ui_state or {})
        item = cls.from_content_dict(data, folder, **ui_kwargs)
        return item

    def rename(self, new_name: str) -> tuple[str | None, str] | None:
        """Rename this item, returning deletion info if file needs cleanup.

        Args:
            new_name: The new display name

        Returns:
            Tuple of (folder, old_disk_name) if the disk name changed and
            old file needs deletion, or None if no cleanup needed.
        """
        old_disk_name = self.get_disk_name()
        self._set_display_name(new_name)
        new_disk_name = self.get_disk_name()

        if new_disk_name != old_disk_name:
            return (self.folder, old_disk_name)
        return None


@dataclass
class ManagedConfig(DashboardItem):
    """Amplification config with dashboard state."""

    config: AmplificationConfig = None
    _last_compiled_hash: str | None = field(default=None, init=False)
    _last_compiled_config_str: str | None = field(default=None, init=False)
    _lora_int_id: int = field(default=-1, init=False)

    def __post_init__(self):
        self._update_lora_int_id()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    # === Backward compatibility alias ===

    @property
    def config_id(self) -> str:
        """Alias for item_id (backward compatibility)."""
        return self.item_id

    # === Abstract method implementations ===

    def get_display_name(self) -> str:
        """Get the config name."""
        return self.config.name

    def _set_display_name(self, name: str) -> None:
        """Set the config name."""
        self.config.name = name

    def to_content_dict(self) -> dict[str, Any]:
        """Serialize config content for disk storage."""
        return self.config.to_dict()

    @classmethod
    def from_content_dict(
        cls,
        data: dict[str, Any],
        folder: str | None,
        item_id: str | None = None,
        **ui_kwargs,
    ) -> "ManagedConfig":
        """Create from loaded content dict."""
        config = AmplificationConfig.from_dict(data)
        return cls(
            config=config,
            folder=folder,
            item_id=item_id or str(uuid.uuid4()),
            **ui_kwargs,
        )

    # === Config-specific methods ===

    def _update_lora_int_id(self):
        """Derive lora_int_id from compiled hash if available, else item_id (vLLM needs an integer)."""
        hash_source = self._last_compiled_hash or self.item_id
        self._lora_int_id = abs(hash(hash_source)) % (2**31)

    @property
    def lora_int_id(self) -> int:
        """Get the LORA int ID for vLLM (derived from compiled hash or item_id)."""
        return self._lora_int_id

    @property
    def last_compiled_hash(self) -> str | None:
        """Get the last compiled hash."""
        return self._last_compiled_hash

    @last_compiled_hash.setter
    def last_compiled_hash(self, value: str | None):
        """Set the last compiled hash and update LORA int ID if changed."""
        if value != self._last_compiled_hash:
            print(f"Hash changed for {self.full_name}, setting lora_int_id to", end=" ")
            self._last_compiled_hash = value
            self._update_lora_int_id()
            print(self.lora_int_id)

    @property
    def last_compiled_config_str(self) -> str | None:
        """Get the last compiled config string (JSON serialization of config dict)."""
        return self._last_compiled_config_str

    def to_dict(self) -> dict[str, Any]:
        """Serialize both UI state and config (for session state)."""
        result = self.to_ui_dict()
        result["config"] = self.config.to_dict()
        result["folder"] = self.folder
        result["config_id"] = self.item_id  # Backward compat key name
        return result

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ManagedConfig":
        """Deserialize from dict (for session state)."""
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
            folder=data.get("folder") or None,  # Convert empty string to None
            item_id=config_id,
            **ui_fields,
        )

    @staticmethod
    def from_config(
        config: AmplificationConfig,
        active: bool = True,
        expanded: bool = True,
        ui_order: int = 0,
        folder: str | None = None,
    ) -> "ManagedConfig":
        """Create from a pure config (e.g., when loading external config)."""
        return ManagedConfig(
            config=config,
            folder=folder,
            active=active,
            ui_order=ui_order,
            expanded=expanded,
        )

    @staticmethod
    def from_folder(
        folder: str, cfg_idx: int, existing_names: set[str]
    ) -> "ManagedConfig":
        """Create a new amplification config in the given folder.

        Args:
            folder: Folder to create the config in
            cfg_idx: Index for default naming (e.g., "Config 1")
            existing_names: Set of existing full names to avoid collisions
        """
        base_name = f"Config {cfg_idx}"
        unique_name = get_unique_item_name(existing_names, base_name, folder)
        new_config = AmplificationConfig(
            name=unique_name,
            description="",
            amplified_adapters=[],
        )
        return ManagedConfig.from_config(
            new_config, active=True, expanded=True, folder=folder
        )

    def compile(
        self,
        compiled_dir: Path,
        base_model_name: str,
        base_model: StandardizedTransformer,
    ) -> Path | None:
        path, hash, config_str = self.config.compile(
            compiled_dir,
            base_model_name=base_model_name,
            base_model=base_model,
            output_name=self.full_name,
        )
        if hash is not None:
            assert path is not None
            self.last_compiled_hash = hash
            self._last_compiled_config_str = config_str
        return path


@dataclass
class ManagedPrompt(DashboardItem):
    """Prompt with dashboard state for multi-prompt generation."""

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

    # === Backward compatibility alias ===

    @property
    def prompt_id(self) -> str:
        """Alias for item_id (backward compatibility)."""
        return self.item_id

    # === Abstract method implementations ===

    def get_display_name(self) -> str:
        """Return name or truncated prompt text."""
        if self.name:
            return self.name
        if self.editor_mode == "simple":
            text = self.prompt_text
        else:
            text = self.messages[0]["content"] if self.messages else ""
        return (text[:50] + "...") if len(text) > 50 else text

    def _set_display_name(self, name: str) -> None:
        """Set the prompt name."""
        self.name = name

    def to_content_dict(self) -> dict[str, Any]:
        """Serialize prompt content for disk storage (no UI state or identity)."""
        return {
            "name": self.name,
            "editor_mode": self.editor_mode,
            "prompt_text": self.prompt_text,
            "template_mode": self.template_mode,
            "system_prompt": self.system_prompt,
            "assistant_prefill": self.assistant_prefill,
            "loom_filename": self.loom_filename,
            "messages": self.messages,
        }

    @classmethod
    def from_content_dict(
        cls,
        data: dict[str, Any],
        folder: str | None,
        item_id: str | None = None,
        **ui_kwargs,
    ) -> "ManagedPrompt":
        """Create from loaded content dict."""
        return cls(
            item_id=item_id or str(uuid.uuid4()),
            folder=folder,
            name=data.get("name", ""),
            editor_mode=data.get("editor_mode", "simple"),
            prompt_text=data.get("prompt_text", ""),
            template_mode=data.get("template_mode", "Apply chat template"),
            system_prompt=data.get("system_prompt", ""),
            assistant_prefill=data.get("assistant_prefill", ""),
            loom_filename=data.get("loom_filename", "untitled.txt"),
            messages=data.get("messages", []),
            **ui_kwargs,
        )

    # === Prompt-specific methods ===

    def to_dict(self) -> dict[str, Any]:
        """Serialize both UI state and prompt data (for session state)."""
        result = self.to_ui_dict()
        result.update(self.to_content_dict())
        result["prompt_id"] = self.item_id
        result["folder"] = self.folder
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
        """Deserialize from dict (for session state)."""
        ui_fields = DashboardItem.ui_dict_to_fields(data)
        # Support legacy prompt_id field
        prompt_id = data.get("prompt_id") or str(uuid.uuid4())
        return ManagedPrompt(
            item_id=prompt_id,
            name=data.get("name", ""),
            editor_mode=data.get("editor_mode", "simple"),
            prompt_text=data.get("prompt_text", ""),
            template_mode=data.get("template_mode", "Apply chat template"),
            system_prompt=data.get("system_prompt", ""),
            assistant_prefill=data.get("assistant_prefill", ""),
            loom_filename=data.get("loom_filename", "untitled.txt"),
            messages=data.get("messages", []),
            folder=data.get("folder") or None,  # Convert empty string to None
            **ui_fields,
        )


@dataclass
class ManagedConversation:
    """Conversation with UI state for chat interactions."""

    conv_id: str
    name: str
    config: "ManagedConfig | None" = None
    system_prompt: str = ""
    history: list[dict] = field(default_factory=list)
    editing_message: int | None = None
    regenerating_from: int | None = None
    regenerating_from_user: int | None = None
    continuing_from: int | None = None
    multi_gen_enabled: bool = False

    @property
    def full_name(self) -> str:
        """Return the conversation name (for compatibility with unique name functions)."""
        return self.name

    def save(self, conversations_dir: Path) -> None:
        """Save conversation to disk."""
        conversations_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.name.replace("/", "_").replace(":", "_")
        conv_path = conversations_dir / f"{safe_name}.yaml"

        serialized = {
            "conv_id": self.conv_id,
            "name": self.name,
            "context": {
                "config_name": self.config.full_name if self.config else None,
                "system_prompt": self.system_prompt,
            },
            "history": self.history,
            "editing_message": self.editing_message,
            "regenerating_from": self.regenerating_from,
            "continuing_from": self.continuing_from,
        }

        with open(conv_path, "w") as f:
            yaml.dump(serialized, f, sort_keys=False)

    @staticmethod
    def from_file(
        conv_file: Path,
        config_name_to_managed: dict[str, "ManagedConfig"],
    ) -> "ManagedConversation":
        """Load conversation from a YAML file."""
        with open(conv_file) as f:
            data = yaml.safe_load(f)

        config_name = data["context"]["config_name"]
        config = config_name_to_managed.get(config_name) if config_name else None

        return ManagedConversation(
            conv_id=data["conv_id"],
            name=data["name"],
            config=config,
            system_prompt=data["context"].get("system_prompt", ""),
            history=data["history"],
            editing_message=data["editing_message"],
            regenerating_from=data["regenerating_from"],
            continuing_from=data.get("continuing_from"),
        )

    def delete_file(self, conversations_dir: Path) -> None:
        """Delete conversation file from disk."""
        safe_name = self.name.replace("/", "_").replace(":", "_")
        conv_path = conversations_dir / f"{safe_name}.yaml"
        if conv_path.exists():
            conv_path.unlink()
