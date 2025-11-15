"""
Amplification configuration and compilation logic.

This module handles creating amplification configs that modify LoRA adapter weights
and compiling them into new adapter files for use with vLLM.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Any, Self
import os
from pathlib import Path
import shutil
from nnterp import StandardizedTransformer
import yaml
from loguru import logger
import torch as th
from src.utils.configs import resolve_adapter_id
from src.utils.model import adapter_id_to_path
from src.utils.collection import sum_dict_values


class AmplificationSpecification(ABC):
    @abstractmethod
    def resolve(self, base_model: StandardizedTransformer):
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def from_dict(data: dict[str, Any]) -> Self:
        pass

    @classmethod
    @abstractmethod
    def resolve_list(
        cls, specifications: list[Self], base_model: StandardizedTransformer
    ):
        pass


@dataclass
class ModuleAmplification(AmplificationSpecification):
    """Amplification for specific modules in a layer."""

    modules: Literal["all", "attention", "mlp"]
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"modules": self.modules, "weight": self.weight}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ModuleAmplification":
        return ModuleAmplification(modules=data["modules"], weight=data["weight"])

    def resolve(self, base_model: StandardizedTransformer) -> list[str]:
        if self.modules == "all":
            return ["attention", "mlp"]
        else:
            assert self.modules in [
                "attention",
                "mlp",
            ], f"Invalid module name: {self.modules}"
            return [self.modules]

    @classmethod
    def resolve_list(
        cls, modules: list[Self], base_model: StandardizedTransformer
    ) -> dict[str, float]:
        res = dict(attention=0.0, mlp=0.0)
        for module in modules:
            for mod in module.resolve(base_model):
                res[mod] += module.weight
        return res


@dataclass
class LayerAmplification(AmplificationSpecification):
    """Amplification for specific layers."""

    layers: list[int] | int | Literal["all"]
    module_amplifications: list[ModuleAmplification]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "module_amplifications": [m.to_dict() for m in self.module_amplifications],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Self:
        return LayerAmplification(
            layers=data["layers"],
            module_amplifications=[
                ModuleAmplification.from_dict(m) for m in data["module_amplifications"]
            ],
        )

    def resolve(
        self, base_model: StandardizedTransformer
    ) -> tuple[list[int], dict[str, float]]:
        module_resolution = ModuleAmplification.resolve_list(
            self.module_amplifications, base_model
        )
        if isinstance(self.layers, list):
            layers = self.layers
        elif self.layers == "all":
            layers = list(range(base_model.num_layers))
        else:
            layers = [self.layers]
        return layers, module_resolution

    @classmethod
    def resolve_list(
        cls, specifications: list[Self], base_model: StandardizedTransformer
    ) -> list[dict[str, float]]:
        module_updates: list[list[dict[str, float]]] = [
            [] for _ in range(base_model.num_layers)
        ]
        for spec in specifications:
            layers, module_resolution = spec.resolve(base_model)
            for layer in layers:
                module_updates[layer].append(module_resolution)
        return [sum_dict_values(updates) for updates in module_updates]


@dataclass
class AmplifiedAdapter:
    """Amplification config for one adapter."""

    organism_name: str  # Organism name (e.g., "persona_sarcasm")
    variant: str  # Variant name (e.g., "default", "is")
    layer_amplifications: list[LayerAmplification]

    def adapter_id(self, base_model_name: str) -> str:
        return resolve_adapter_id(self.organism_name, self.variant, base_model_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "organism_name": self.organism_name,
            "variant": self.variant,
            "layer_amplifications": [
                ampl.to_dict() for ampl in self.layer_amplifications
            ],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AmplifiedAdapter":
        return AmplifiedAdapter(
            organism_name=data.get("organism_name", ""),
            variant=data.get("variant", "default"),
            layer_amplifications=[
                LayerAmplification.from_dict(ampl)
                for ampl in data["layer_amplifications"]
            ],
        )

    def resolve(self, base_model: StandardizedTransformer):
        return LayerAmplification.resolve_list(self.layer_amplifications, base_model)

    @classmethod
    def resolve_list(
        cls,
        specifications: list[Self],
        base_model: StandardizedTransformer,
        base_model_name: str,
    ) -> dict[str, list[dict[str, float]]]:
        grouped_layer_amplifications: dict[str, list[LayerAmplification]] = defaultdict(
            list
        )
        for amplified_adapter in specifications:
            if len(amplified_adapter.layer_amplifications) == 0:
                continue
            grouped_layer_amplifications[
                amplified_adapter.adapter_id(base_model_name)
            ].extend(amplified_adapter.layer_amplifications)

        return {
            adapter_id: LayerAmplification.resolve_list(
                layer_amplifications, base_model
            )
            for adapter_id, layer_amplifications in grouped_layer_amplifications.items()
        }


@dataclass
class AmplificationConfig:
    """Full amplification configuration."""

    name: str
    description: str = ""
    amplified_adapters: list[AmplifiedAdapter] = field(default_factory=list)

    def to_dict(self, resolved_config=None) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        res = dict(
            name=self.name,
            description=self.description,
        )
        if resolved_config is not None:
            res["resolved_config"] = resolved_config
        res["adapters"] = [a.to_dict() for a in self.amplified_adapters]
        return res

    def save_yaml(self, path: Path, resolved_config=None) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(resolved_config), f)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AmplificationConfig":
        """Create from dictionary."""
        return AmplificationConfig(
            name=data["name"],
            description=data.get("description", ""),
            amplified_adapters=[
                AmplifiedAdapter.from_dict(a) for a in data.get("adapters", [])
            ],
        )

    @staticmethod
    def load_yaml(path: Path) -> "AmplificationConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return AmplificationConfig.from_dict(data)

    def resolve(
        self, base_model: StandardizedTransformer, base_model_name: str
    ) -> dict[str, list[dict[str, float]]]:
        return AmplifiedAdapter.resolve_list(
            self.amplified_adapters, base_model, base_model_name
        )

    def compile(
        self, base_dir: Path, base_model_name: str, base_model: StandardizedTransformer
    ) -> Path | None:
        """
        Compile this amplification config into a modified adapter.

        This method creates a directory with the following files:
        1. Symlink to all files from the first adapter_id directory
        2. Symlink to the other adapter directories
        3. A config.yaml

        Args:
            base_dir: Directory to save where we should create the compiled adapter directory
            base_model_name: Base model name (required for resolving adapter_ids if not set)

        Returns:
            Path to the compiled adapter directory
        """
        if len(self.amplified_adapters) == 0:
            return None
        output_dir = base_dir / self.name

        # Resolve adapter_ids if they're not already set
        all_adapter_ids = [
            adapter.adapter_id(base_model_name) for adapter in self.amplified_adapters
        ]

        all_adapter_ids = list(dict.fromkeys(all_adapter_ids))
        # Do all the symlinking for the first adapter directly in the output directory
        first_adapter_id = all_adapter_ids[0]
        adapter_path = adapter_id_to_path(first_adapter_id)
        if output_dir.exists():
            logger.warning(
                f"Output directory {output_dir} already exists. Overwriting."
            )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        for item in adapter_path.iterdir():
            target = output_dir / item.name
            assert not target.exists(), f"Target {target} already exists"
            os.symlink(item, target)

        # Add other adapters in subdirectories
        for adapter_id in all_adapter_ids[1:]:
            adapter_dir = adapter_id_to_path(adapter_id)
            for item in adapter_dir.iterdir():
                target = output_dir / adapter_id.replace("/", "_") / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists(), f"Target {target} already exists"
                os.symlink(item, target)

        # Add config.yaml
        config_path = output_dir / "amplification_config.yaml"
        resolved_amplifications = self.resolve(base_model, base_model_name)
        self.save_yaml(config_path, resolved_amplifications)

        return output_dir


def _load_adapter_weights(adapter_id: str) -> dict[str, th.Tensor]:
    """Load adapter weights from HuggingFace."""
    raise NotImplementedError("_load_adapter_weights() needs implementation")


def _apply_amplification(
    weights: dict[str, th.Tensor],
    adapter_amp: AmplifiedAdapter,
) -> dict[str, th.Tensor]:
    """Apply amplification factors to adapter weights."""
    raise NotImplementedError("_apply_amplification() needs implementation")


def _resolve_layers(
    layers: list[int] | int | Literal["all"], num_layers: int
) -> list[int]:
    """Convert layers specification to list of layer indices."""
    raise NotImplementedError("_resolve_layers() needs implementation")


def _resolve_modules(modules: list[str] | Literal["all"], adapter_id: str) -> list[str]:
    """Convert modules specification to list of module names."""
    raise NotImplementedError("_resolve_modules() needs implementation")


def _should_amplify(
    param_name: str,
    layers: list[int],
    modules: list[str],
) -> bool:
    """Check if parameter should be amplified based on name."""
    raise NotImplementedError("_should_amplify() needs implementation")


def _save_adapter(
    weights: dict[str, th.Tensor],
    adapter_id: str,
    output_dir: Path,
) -> None:
    """Save modified adapter weights to disk."""
    raise NotImplementedError("_save_adapter() needs implementation")
