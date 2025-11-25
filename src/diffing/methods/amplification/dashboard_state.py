"""
Dashboard state management for amplification UI.

Separates UI concerns (active state, ordering) from domain models (configs).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

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
