"""
Amplification configuration and compilation logic.

This module handles creating amplification configs that modify LoRA adapter weights
and compiling them into new adapter files for use with vLLM.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from typing import Literal, Any, Self
from pathlib import Path
import re
import shutil
import yaml

from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
from safetensors.torch import save_file

from src.utils.configs import resolve_adapter_id
from src.utils.model import adapter_id_to_path
from src.utils.vllm import ensure_vllm


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

    def resolve(self) -> list[Literal["attention", "mlp"]]:
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
        res = dict(attention=1.0, mlp=1.0)
        for module in modules:
            for mod in module.resolve():
                res[mod] = module.weight
        return res


@dataclass
class LayerRange:
    """Represents an inclusive range of layers."""

    start: int | float
    end: int | float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "range", "start": self.start, "end": self.end}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LayerRange":
        return LayerRange(start=data["start"], end=data["end"])

    def to_list(self, num_layers: int, is_relative: bool) -> list[int]:
        start, end = self.start, self.end
        if is_relative:
            assert 0 <= start <= 1, f"Relative start {start} not in [0, 1]"
            assert 0 <= end <= 1, f"Relative end {end} not in [0, 1]"
            start = round(start * (num_layers - 1))
            end = round(end * (num_layers - 1))
        return list(range(int(start), int(end) + 1))


def _resolve_layer_value(value: float | int, num_layers: int, is_relative: bool) -> int:
    """Convert a layer value to an absolute layer index."""
    if is_relative:
        assert 0 <= value <= 1, f"Relative value {value} not in [0, 1]"
        return round(value * (num_layers - 1))
    assert isinstance(value, int), f"Value {value} is not an int nor a float"
    return value


@dataclass
class LayerAmplification(AmplificationSpecification):
    """Amplification for specific layers."""

    layers: list[int | float] | int | float | LayerRange | Literal["all"]
    module_amplifications: list[ModuleAmplification]
    is_relative: bool = False

    def to_dict(self) -> dict[str, Any]:
        if (
            type(self.layers).__name__ == "LayerRange"
        ):  # robust check to streamlit reloads
            layers = self.layers.to_dict()
        else:
            layers = self.layers
        return {
            "layers": layers,
            "is_relative": self.is_relative,
            "module_amplifications": [m.to_dict() for m in self.module_amplifications],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Self:
        layers_data = data["layers"]
        if isinstance(layers_data, dict) and layers_data.get("type") == "range":
            layers = LayerRange.from_dict(layers_data)
        else:
            layers = layers_data
        return LayerAmplification(
            layers=layers,
            is_relative=data.get("is_relative", False),
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
        num_layers = base_model.num_layers

        if type(self.layers).__name__ == "LayerRange":
            layers = self.layers.to_list(num_layers, self.is_relative)
        elif isinstance(self.layers, list):
            layers = [
                _resolve_layer_value(layer, num_layers, self.is_relative)
                for layer in self.layers
            ]
        elif self.layers == "all":
            layers = list(range(num_layers))
        else:
            layers = [_resolve_layer_value(self.layers, num_layers, self.is_relative)]
        return layers, module_resolution

    @classmethod
    def resolve_list(
        cls, specifications: list[Self], base_model: StandardizedTransformer
    ) -> list[dict[str, float]]:
        module_updates: list[dict[str, float]] = [
            {} for _ in range(base_model.num_layers)
        ]
        for spec in specifications:
            layers, module_resolution = spec.resolve(base_model)
            for layer in layers:
                module_updates[layer].update(module_resolution)
        return module_updates


CUSTOM_ADAPTER_ORGANISM = "custom"


@dataclass
class AmplifiedAdapter:
    """Amplification config for one adapter."""

    organism_name: (
        str  # Organism name (e.g., "persona_sarcasm") or "custom" for direct HF repo ID
    )
    variant: str  # Variant name (e.g., "default", "is") or HF repo ID if organism_name == "custom"
    layer_amplifications: list[LayerAmplification]

    def adapter_id(self, base_model_name: str) -> str:
        """
        Returns the HF adapter ID corresponding to the organism and variant of this config.
        """
        if self.organism_name == CUSTOM_ADAPTER_ORGANISM:
            if not self.variant:
                raise ValueError(
                    "Custom adapter requires a valid HuggingFace adapter ID in the 'variant' field, "
                    "but it is empty. Please specify an adapter ID like 'hf_user/repo' or 'hf_user/repo/path'."
                )
            # variant is the direct HF repo ID (e.g., "hf/repo" or "hf/repo/path/in/repo")
            return self.variant
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return dict(
            name=self.name,
            description=self.description,
            adapters=[a.to_dict() for a in self.amplified_adapters],
        )

    def to_dict_for_model(
        self, base_model_name: str, base_model: StandardizedTransformer
    ) -> dict[str, Any]:
        res = self.to_dict()
        res.update(
            resolved_config=self.resolve(base_model, base_model_name),
            module_paths=get_module_regex(base_model),
        )
        return res

    def save_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        if path.exists():
            with open(path, "r") as f:
                old_data = f.read()
        else:
            old_data = None
        try:
            with open(path, "w") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        except Exception as e:
            logger.error(f"Error saving config to {path}")
            if old_data is not None:
                logger.error("Restoring old data")
            with open(path, "w") as f:
                f.write(old_data)
            raise e

    def resolve(
        self, base_model: StandardizedTransformer, base_model_name: str
    ) -> dict[str, list[dict[str, float]]]:
        return AmplifiedAdapter.resolve_list(
            self.amplified_adapters, base_model, base_model_name
        )

    def compile(
        self, base_dir: Path, base_model_name: str, base_model: StandardizedTransformer
    ) -> tuple[Path | None, str | None]:
        """
        Compile this amplification config into a modified adapter.

        This method creates a directory with the following files:
        1. Symlink to all files from the first adapter_id directory
        2. Symlink to the other adapter directories
        3. A amplification_config.yaml file with the resolved config

        Args:
            base_dir: Directory to save where we should create the compiled adapter directory
            base_model_name: Base model name (required for resolving adapter_ids if not set)

        Returns:
            Path to the compiled adapter directory
        """
        if len(self.amplified_adapters) == 0:
            return None, None
        output_dir = base_dir / self.name / base_model_name
        if output_dir.exists():
            logger.warning(
                f"Output directory {output_dir} already exists. Overwriting."
            )
            shutil.rmtree(output_dir)

        # Create compiled config
        config_dict = self.to_dict_for_model(base_model_name, base_model)
        # Use hash to put this in a unique directory
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()
        config_dict["hash"] = config_hash
        output_dir = output_dir / config_hash
        output_dir.mkdir(parents=True, exist_ok=False)

        with open(output_dir / "amplification_config.yaml", "w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)

        # Resolve adapter_ids if they're not already set
        all_adapter_ids = [
            adapter.adapter_id(base_model_name) for adapter in self.amplified_adapters
        ]

        all_adapter_ids = list(dict.fromkeys(all_adapter_ids))
        # Do all the symlinking for the first adapter directly in the output directory
        first_adapter_id = all_adapter_ids[0]
        adapter_path = adapter_id_to_path(first_adapter_id)

        # Add other adapters in subdirectories
        for adapter_id in all_adapter_ids[1:]:
            adapter_dir = adapter_id_to_path(adapter_id)
            for item in adapter_dir.iterdir():
                target = output_dir / adapter_id.replace("/", "_") / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists(), f"Target {target} already exists"
                os.symlink(item, target)
        for item in adapter_path.iterdir():
            target = output_dir / item.name
            assert not target.exists(), f"Target {target} already exists"
            os.symlink(item, target)
        return output_dir, config_hash


def path_to_template(path: str) -> str:
    path = f"base_model.{path}"  # add base_model prefix to the path
    if "0" not in path:
        raise ValueError(f"Path {path} does not contain a 0")
    if path.count("0") > 1:
        raise ValueError(f"Path {path} contains multiple 0s")
    return path.replace(".", "\\.").replace("0", "[layer_idx]") + ".*\\.lora_B.weight"


def get_module_regex(
    model: StandardizedTransformer,
) -> dict[Literal["attention", "mlp"], str]:
    return {
        "attention": path_to_template(model.attentions[0]._module.__path__),
        "mlp": path_to_template(model.mlps[0]._module.__path__),
    }


def format_amplified_modules(amplified_modules: dict[str, float]) -> str:
    """
    Create a compact tree representation of amplified modules.

    Groups consecutive layers with the same weight into ranges.
    Example: [0-31] instead of listing each layer separately.

    Args:
        amplified_modules: Dict mapping module paths to their amplification weights

    Returns:
        Formatted tree string showing the hierarchical structure
    """
    if not amplified_modules:
        return "No modules amplified"

    pattern = re.compile(r"^(.+\.layers\.)(\d+)(\..+)$")

    parsed = []
    for path, weight in amplified_modules.items():
        match = pattern.match(path)
        assert match, f"Could not parse layer number from path: {path}"
        prefix, layer_num, suffix = match.groups()
        parsed.append((prefix, int(layer_num), suffix, weight))

    groups = defaultdict(list)
    for prefix, layer, suffix, weight in parsed:
        groups[(prefix, suffix, weight)].append(layer)

    def layers_to_ranges(layers: list[int]) -> str:
        layers = sorted(set(layers))
        ranges = []
        start = layers[0]
        end = layers[0]

        for layer in layers[1:]:
            if layer == end + 1:
                end = layer
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = layer
                end = layer
        ranges.append(f"{start}-{end}" if start != end else str(start))

        return ",".join(ranges) if len(ranges) > 1 else ranges[0]

    tree = defaultdict(lambda: defaultdict(list))

    for (prefix, suffix, weight), layers in groups.items():
        parts = suffix.lstrip(".").split(".")
        assert len(parts) >= 2, f"Unexpected suffix format: {suffix}"

        component = parts[0]
        module_path = ".".join(parts[1:])

        layer_range = layers_to_ranges(layers)
        tree[component][module_path].append((layer_range, weight))

    lines = [f"Amplified {len(amplified_modules)} modules:\n"]

    components = sorted(tree.keys())
    for comp_idx, component in enumerate(components):
        is_last_comp = comp_idx == len(components) - 1
        comp_prefix = "└─" if is_last_comp else "├─"
        comp_continuation = "  " if is_last_comp else "│ "

        lines.append(f"{comp_prefix} {component}")

        modules = sorted(tree[component].items())
        for mod_idx, (module_path, ranges_weights) in enumerate(modules):
            is_last_mod = mod_idx == len(modules) - 1
            mod_prefix = "└─" if is_last_mod else "├─"
            mod_continuation = "  " if is_last_mod else "│ "

            lines.append(f"{comp_continuation} {mod_prefix} {module_path}")

            for rw_idx, (layer_range, weight) in enumerate(ranges_weights):
                is_last_rw = rw_idx == len(ranges_weights) - 1
                rw_prefix = "└─" if is_last_rw else "├─"

                lines.append(
                    f"{comp_continuation} {mod_continuation} {rw_prefix} [{layer_range}]: {weight}"
                )

    return "\n".join(lines)


def patch_lora_weights(
    weights: dict[str, th.Tensor],
    compiled_amplifications: dict[str, list[dict[str, float]]],
    module_paths: dict[Literal["attention", "mlp"], str],
) -> tuple[dict[str, float], list[str]]:
    """
    Patch LoRA weights with compiled amplifications in-place.

    Args:
        weights: the dictionary of weights to patch
        compiled_amplifications: the compiled amplifications
        module_paths: the module paths used to find submodules of each amplified module
    Returns: a tuple of the dictionary of amplified modules with their weights and the list of unamplified module names.
    """
    if len(compiled_amplifications) == 0:
        return dict(), list(weights.keys())
    elif len(compiled_amplifications) > 1:
        raise NotImplementedError(
            "Multiple compiled amplifications are not supported yet"
        )
    all_weight_keys = list(weights.keys())
    for k in all_weight_keys:
        if "lora_B.weight" not in k and "lora_A.weight" not in k:
            raise ValueError(f"Weight {k} is not a LoRA weight")
    adapter_amplification = list(compiled_amplifications.values())[0]
    amplified_modules = dict()

    for layer_idx, layer_amplification in enumerate(adapter_amplification):
        for module_name, module_weight in layer_amplification.items():
            resolved_module_regex = re.compile(
                module_paths[module_name].replace("[layer_idx]", str(layer_idx))
            )
            matches = [k for k in all_weight_keys if resolved_module_regex.match(k)]
            if len(matches) == 0:
                raise ValueError(
                    f"No matches found for module {module_name} in layer {layer_idx} using regex {resolved_module_regex}. All weight keys: {[k for k in all_weight_keys]}"
                )
            for match in matches:
                if match in amplified_modules:
                    raise ValueError(f"Module {match} already amplified")
                weights[match] *= module_weight
                amplified_modules[match] = module_weight
    unamplified_modules = [k for k in all_weight_keys if k not in amplified_modules]
    return amplified_modules, unamplified_modules


@ensure_vllm
def patch_vllm():
    """Patch vLLM's LoRA loading to apply amplification weights."""
    from vllm.lora.models import LoRAModel
    from vllm.logger import init_logger
    from vllm.lora.peft_helper import PEFTHelper

    is_debug = os.getenv("DEBUG_AMPLIFICATION", "0") == "1"

    if getattr(LoRAModel, "_is_amplification_patched", False):
        return

    vllm_logger = init_logger(__name__)

    # Store original methods
    _original_from_local_checkpoint = LoRAModel.from_local_checkpoint
    _original_from_lora_tensors = LoRAModel.from_lora_tensors

    @classmethod
    def scaled_from_local_checkpoint(
        cls, lora_dir: str, expected_lora_modules, peft_helper, **kwargs
    ):
        cfg_path = Path(lora_dir) / "amplification_config.yaml"
        if not isinstance(peft_helper, PEFTHelper):
            raise ValueError(
                f"peft_helper is not a PeftHelper: {type(peft_helper)}, something changed in vLLM and might break things"
            )

        # Attach config to peft_helper if it exists
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
                peft_helper._amplification_config = cfg
                peft_helper._amplfication_config_lora_path = lora_dir
        else:
            peft_helper._amplification_config = None
            peft_helper._amplfication_config_lora_path = None
        return _original_from_local_checkpoint(
            lora_dir, expected_lora_modules, peft_helper, **kwargs
        )

    @classmethod
    def scaled_from_lora_tensors(cls, lora_model_id, tensors, peft_helper, **kwargs):
        # Check if config was attached
        if not isinstance(peft_helper, PEFTHelper):
            raise ValueError(
                f"peft_helper is not a PeftHelper: {type(peft_helper)}, something changed in vLLM and might break things"
            )
        if not hasattr(peft_helper, "_amplification_config"):
            raise ValueError(
                "peft_helper does not have an _amplification_config attribute, which should have been attached by the monkey patch of `from_local_checkpoint`"
            )
        cfg = peft_helper._amplification_config
        if cfg is None:
            vllm_logger.info("No amplification config found, skipping amplification")
        else:
            amplified_modules, unamplified_modules = patch_lora_weights(
                tensors, cfg["resolved_config"], cfg["module_paths"]
            )
            vllm_logger.info(f"running config {cfg['name']}:")
            if is_debug:
                vllm_logger.info(
                    yaml.dump(
                        {k: v for k, v in cfg.items() if k != "resolved_config"}
                        | {"resolved_config": "<resolved_config>"}
                    )
                )
            vllm_logger.info(format_amplified_modules(amplified_modules))
            if is_debug:
                debug_path = Path(peft_helper._amplfication_config_lora_path) / "debug"
                debug_path.mkdir(exist_ok=True)
                from coolname import generate_slug

                timestamp = (
                    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{generate_slug(2)}"
                )
                with open(
                    debug_path / f"{timestamp}.yaml",
                    "w",
                ) as f:
                    yaml.dump(
                        dict(
                            amplified_modules=amplified_modules,
                            unamplified_modules=unamplified_modules,
                        ),
                        f,
                    )
                save_file(tensors, debug_path / f"{timestamp}.safetensors")
        return _original_from_lora_tensors(
            lora_model_id, tensors, peft_helper, **kwargs
        )

    # Apply patches
    LoRAModel.from_local_checkpoint = scaled_from_local_checkpoint
    LoRAModel.from_lora_tensors = scaled_from_lora_tensors
    LoRAModel._is_amplification_patched = True

    logger.info("vLLM LoRA loading patched for amplification support")
