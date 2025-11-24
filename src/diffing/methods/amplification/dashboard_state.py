"""
Dashboard state management for amplification UI.

Separates UI concerns (active state, ordering) from domain models (configs).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, ClassVar

from nnterp import StandardizedTransformer

from src.diffing.methods.amplification.amplification_config import AmplificationConfig


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
    lora_int_id: int = field(default=0, init=False)

    # Class variable to track the number of ManagedConfig instances
    _instance_count: ClassVar[int] = 0

    def __post_init__(self):
        self._update_lora_int_id()

    def _update_lora_int_id(self):
        """Update the LORA int ID for this config."""
        type(self)._instance_count += 1
        self.lora_int_id = type(self)._instance_count

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

    managed_configs: List[ManagedConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session."""
        return {
            "managed_configs": [mc.to_dict() for mc in self.managed_configs],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DashboardSession":
        """Deserialize session."""
        return DashboardSession(
            managed_configs=[
                ManagedConfig.from_dict(mc) for mc in data.get("managed_configs", [])
            ],
        )
