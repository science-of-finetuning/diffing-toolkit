"""
Dashboard state management for amplification UI.

Separates UI concerns (active state, ordering) from domain models (configs).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, ClassVar

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
    last_compiled_path: Optional[Path] = None
    lora_int_id: int = 0

    # Class variable to track the number of ManagedConfig instances
    _instance_count: ClassVar[int] = 0

    def __post_init__(self):
        type(self)._instance_count += 1
        self.lora_int_id = type(self)._instance_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize both UI state and config."""
        result = self.to_ui_dict()
        result["config"] = self.config.to_dict()
        if self.last_compiled_path:
            result["last_compiled_path"] = str(self.last_compiled_path)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ManagedConfig":
        """Deserialize from dict."""
        ui_fields = DashboardItem.ui_dict_to_fields(data)
        config = AmplificationConfig.from_dict(data["config"])
        last_compiled_path = (
            Path(data["last_compiled_path"]) if "last_compiled_path" in data else None
        )

        return ManagedConfig(
            config=config,
            last_compiled_path=last_compiled_path,
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
            last_compiled_path=None,
        )

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
