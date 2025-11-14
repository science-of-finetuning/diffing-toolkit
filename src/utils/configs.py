from dataclasses import dataclass
from typing import Dict, Tuple, List
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from hydra import initialize, compose
from pathlib import Path

HF_NAME = "science-of-finetuning"
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs_new"


def _get_all_models_with_none() -> dict[str, dict[str, None]]:
    """
    Scans the configs_new/model directory and returns a dict mapping
    each model name to {"default": None}.
    """
    model_dir = CONFIGS_DIR / "model"
    models = {}
    for yaml_file in model_dir.glob("*.yaml"):
        model_name = yaml_file.stem
        models[model_name] = {"default": {"model_id": "none"}}
    return models


OmegaConf.clear_resolver(
    "get_all_models"
)  # Clearing is necessary for streamlit to work (otherwise it will try to register the resolver multiple times)
OmegaConf.register_new_resolver("get_all_models", _get_all_models_with_none)


@dataclass
class ModelConfig:
    """Configuration for a model (base or finetuned)."""

    name: str
    model_id: str
    tokenizer_id: str | None = None
    attn_implementation: str = "eager"
    ignore_first_n_tokens_per_sample_during_collection: int = 0
    ignore_first_n_tokens_per_sample_during_training: int = 0
    token_level_replacement: dict = None
    text_column: str = "text"
    base_model_id: str = None
    subfolder: str = ""
    dtype: str = "float32"
    steering_vector: str = None
    steering_layer: int = None
    no_auto_device_map: bool = False
    device_map: object | None = None
    trust_remote_code: bool = False
    vllm_kwargs: dict | None = None


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    id: str
    split: str
    is_chat: bool
    text_column: str = None
    messages_column: str = "messages"
    description: str = ""


def get_safe_model_id(model_cfg: ModelConfig) -> str:
    """Get the safe id of a model for paths."""
    model_name_clean = model_cfg.model_id.split("/")[-1]
    if model_cfg.steering_vector is not None:
        steering_vector_name_clean = model_cfg.steering_vector.split("/")[-1]
        model_name_clean += f"_{steering_vector_name_clean}_L{model_cfg.steering_layer}"
    return model_name_clean


def create_model_config(
    model_cfg: DictConfig, name_override: str = None, device_map: object | None = None
) -> ModelConfig:
    """Create a ModelConfig from configuration object."""
    return ModelConfig(
        name=name_override or model_cfg.name,
        model_id=model_cfg.model_id,
        tokenizer_id=model_cfg.get("tokenizer_id", None),
        attn_implementation=model_cfg.get("attn_implementation"),
        ignore_first_n_tokens_per_sample_during_collection=model_cfg.get(
            "ignore_first_n_tokens_per_sample_during_collection", 0
        ),
        ignore_first_n_tokens_per_sample_during_training=model_cfg.get(
            "ignore_first_n_tokens_per_sample_during_training", 0
        ),
        token_level_replacement=model_cfg.get("token_level_replacement", None),
        text_column=model_cfg.get("text_column", "text"),
        base_model_id=model_cfg.get("base_model_id"),
        dtype=model_cfg.get("dtype"),
        steering_vector=model_cfg.get("steering_vector", None),
        steering_layer=model_cfg.get("steering_layer", None),
        no_auto_device_map=model_cfg.get("no_auto_device_map", False),
        subfolder=model_cfg.get("subfolder", ""),
        device_map=device_map,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )


def create_dataset_config(
    dataset_cfg: DictConfig, name: str, split: str
) -> DatasetConfig:
    """Create a DatasetConfig from configuration object for a specific split."""
    return DatasetConfig(
        name=name,
        id=dataset_cfg.id,
        split=split,
        is_chat=dataset_cfg.is_chat,
        text_column=dataset_cfg.get("text_column", None),
        messages_column=dataset_cfg.get("messages_column", "messages"),
        description=dataset_cfg.get("description", ""),
    )


def get_model_configurations(cfg: DictConfig) -> Tuple[ModelConfig, ModelConfig]:
    """Extract and prepare base and finetuned model configurations."""
    # Base model configuration
    base_model_cfg = create_model_config(
        cfg.model, device_map=cfg.infrastructure.device_map.base
    )

    # Finetuned model configuration - resolve from organism.finetuned_models
    organism_cfg = cfg.organism

    if not hasattr(organism_cfg, "finetuned_models"):
        raise ValueError(
            f"Organism {organism_cfg.name} has no finetuned_models defined. "
            "Please add finetuned_models section to the organism config."
        )

    # Get variant from config (default: "default")
    variant = cfg.get("organism_variant", "default")

    # Look up model_id from finetuned_models
    model_name = cfg.model.name
    if model_name not in organism_cfg.finetuned_models:
        available_models = list(organism_cfg.finetuned_models.keys())
        raise ValueError(
            f"Model {model_name} not found in organism {organism_cfg.name} finetuned_models. "
            f"Available models: {available_models}"
        )

    if variant not in organism_cfg.finetuned_models[model_name]:
        available_variants = list(organism_cfg.finetuned_models[model_name].keys())
        raise ValueError(
            f"Variant {variant} not found for model {model_name} in organism {organism_cfg.name}. "
            f"Available variants: {available_variants}"
        )

    variant_config = organism_cfg.finetuned_models[model_name][variant]

    # Check for adapter_id or model_id
    if "adapter_id" in variant_config:
        model_id_full = variant_config["adapter_id"]
        is_adapter = True
    elif "model_id" in variant_config:
        model_id_full = variant_config["model_id"]
        is_adapter = False
    else:
        raise ValueError(
            f"Model {model_name} variant {variant} in organism {organism_cfg.name} "
            f"must have either 'adapter_id' or 'model_id'"
        )

    # Parse subfolder from model_id if it contains "/"
    # Format: org/repo or org/repo/subfolder or org/repo/subfolder/path
    if "/" in model_id_full and model_id_full.count("/") > 1:  # Has subfolder
        parts = model_id_full.split("/")
        model_id = "/".join(parts[:2])  # org/repo
        subfolder = "/".join(parts[2:])  # subfolder/path
    else:
        model_id = model_id_full
        subfolder = ""

    # Create finetuned model config with inheritance from base model
    finetuned_model_cfg = ModelConfig(
        name=f"{model_name}_{organism_cfg.name}_{variant}",
        model_id=model_id,
        subfolder=subfolder,
        base_model_id=base_model_cfg.model_id if is_adapter else None,
        tokenizer_id=base_model_cfg.tokenizer_id,
        attn_implementation=base_model_cfg.attn_implementation,
        ignore_first_n_tokens_per_sample_during_collection=base_model_cfg.ignore_first_n_tokens_per_sample_during_collection,
        ignore_first_n_tokens_per_sample_during_training=base_model_cfg.ignore_first_n_tokens_per_sample_during_training,
        token_level_replacement=base_model_cfg.token_level_replacement,
        text_column=base_model_cfg.text_column,
        dtype=base_model_cfg.dtype,
        steering_vector=base_model_cfg.steering_vector,
        steering_layer=base_model_cfg.steering_layer,
        no_auto_device_map=base_model_cfg.no_auto_device_map,
        device_map=cfg.infrastructure.device_map.finetuned,
        trust_remote_code=base_model_cfg.trust_remote_code,
    )

    return base_model_cfg, finetuned_model_cfg


def get_dataset_configurations(
    cfg: DictConfig,
    use_chat_dataset: bool = True,
    use_pretraining_dataset: bool = True,
    use_training_dataset: bool = True,
) -> List[DatasetConfig]:
    """Extract and prepare all dataset configurations."""
    datasets = []

    # General datasets (used for all organisms)
    if hasattr(cfg, "chat_dataset") and use_chat_dataset:
        # Create one DatasetConfig for each split
        for split in cfg.chat_dataset.splits:
            datasets.append(
                create_dataset_config(
                    cfg.chat_dataset, cfg.chat_dataset.id.split("/")[-1], split
                )
            )

    if hasattr(cfg, "pretraining_dataset") and use_pretraining_dataset:
        # Create one DatasetConfig for each split
        for split in cfg.pretraining_dataset.splits:
            datasets.append(
                create_dataset_config(
                    cfg.pretraining_dataset,
                    cfg.pretraining_dataset.id.split("/")[-1],
                    split,
                )
            )

    # Organism-specific datasets
    organism_cfg = cfg.organism

    # Training dataset from organism config (if present)
    if hasattr(organism_cfg, "dataset") and use_training_dataset:
        # Create one DatasetConfig for each split
        for split in organism_cfg.dataset.splits:
            datasets.append(
                create_dataset_config(
                    organism_cfg.dataset,
                    organism_cfg.dataset.id.split("/")[-1],
                    split,
                )
            )

    return datasets


def get_available_organisms(configs_dir: Path = None) -> List[str]:
    """Get list of available organisms from configs directory."""
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    organism_dir = configs_dir / "organism"
    if not organism_dir.exists():
        return []

    organisms = []
    for path in organism_dir.glob("*.yaml"):
        if path.stem != "None":  # Exclude None.yaml
            organisms.append(path.stem)

    return sorted(organisms)


def get_organism_variants(
    organism_name: str, base_model_name: str, configs_dir: Path = None
) -> List[str]:
    """
    Get available variants for a specific organism and base model.

    Args:
        organism_name: Name of the organism
        base_model_name: Name of the base model
        configs_dir: Path to configs directory

    Returns:
        List of variant names (e.g., ["default", "is"])
    """
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    organism_path = configs_dir / "organism" / f"{organism_name}.yaml"
    if not organism_path.exists():
        return []

    organism_cfg = OmegaConf.load(organism_path)

    if not hasattr(organism_cfg, "finetuned_models"):
        return []

    if base_model_name not in organism_cfg.finetuned_models:
        return []

    return sorted(organism_cfg.finetuned_models[base_model_name].keys())


def resolve_adapter_id(
    organism_name: str, variant: str, base_model_name: str, configs_dir: Path = None
) -> str:
    """
    Resolve adapter_id from organism name, variant, and base model.

    Args:
        organism_name: Name of the organism (e.g., "persona_sarcasm")
        variant: Variant name (e.g., "default", "is")
        base_model_name: Base model name (e.g., "llama31_8B_Instruct")
        configs_dir: Path to configs directory (defaults to PROJECT_ROOT/configs_new)

    Returns:
        The HuggingFace model ID (e.g., "maius/llama-3.1-8b-it-personas/sarcasm")
    """
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    organism_path = configs_dir / "organism" / f"{organism_name}.yaml"
    assert organism_path.exists(), f"Organism config not found: {organism_path}"

    organism_cfg = OmegaConf.load(organism_path)

    assert hasattr(
        organism_cfg, "finetuned_models"
    ), f"Organism {organism_name} has no finetuned_models"
    assert (
        base_model_name in organism_cfg.finetuned_models
    ), f"Base model {base_model_name} not found in organism {organism_name}"
    assert (
        variant in organism_cfg.finetuned_models[base_model_name]
    ), f"Variant {variant} not found for model {base_model_name}"

    cfg = organism_cfg.finetuned_models[base_model_name][variant]
    if "adapter_id" in cfg:
        return cfg["adapter_id"]
    else:
        raise ValueError(f"Variant {variant} is not an adapter, found {cfg}")
