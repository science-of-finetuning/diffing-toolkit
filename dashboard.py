"""
Streamlit dashboard for visualizing model diffing results.

This dashboard dynamically discovers available model organisms and diffing methods
from the filesystem and provides an interactive interface to explore the results.
"""

import streamlit as st
from omegaconf import DictConfig, OmegaConf
import sys
from typing import Dict, List, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
import time
import traceback
from loguru import logger

from diffing.utils import configs  # noqa: F401 - Registers OmegaConf resolvers
from diffing.utils.configs import CONFIGS_DIR


@st.cache_resource(show_spinner="Importing dependencies: torch...")
def _import_torch():
    import torch  # noqa: F401


@st.cache_resource(show_spinner="Importing dependencies: transformers...")
def _import_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401


@st.cache_resource(show_spinner="Importing dependencies: nnsight...")
def _import_nnsight():
    import nnsight  # noqa: F401


@st.cache_resource(show_spinner="Importing dependencies: others...")
def _import_others():
    import src  # noqa: F401


def _import():
    _import_torch()
    _import_transformers()
    _import_nnsight()
    _import_others()


def _reset_model_cache():
    """Clear cached models/tokenizers and free CUDA memory."""
    from diffing.utils import model as model_utils

    model_utils.clear_cache()


def _get_method_class(method_name: str) -> Any:
    """Get the method class for a given method name. Wrapped as the import is not available in the global scope and the main function loads quickly"""
    from diffing.pipeline.diffing_pipeline import get_method_class

    return get_method_class(method_name)


def discover_organisms() -> List[str]:
    """Discover available organisms from the configs directory."""
    organism_configs = (CONFIGS_DIR / "organism").glob("*.yaml")
    return [f.stem for f in organism_configs]


def discover_methods() -> List[str]:
    """Discover available diffing methods from the configs directory."""
    method_configs = (CONFIGS_DIR / "diffing/method").glob("*.yaml")
    methods = [f.stem for f in method_configs]
    return methods


def _get_cache_path() -> Path:
    """Get the path to the selection cache file."""
    from diffing.utils.configs import PROJECT_ROOT

    cache_dir = PROJECT_ROOT / ".streamlit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "dashboard_selections.yaml"


def _load_selection_cache() -> Dict[str, str]:
    """Load cached model and organism selections from YAML file."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            cache = OmegaConf.load(cache_path)
            return OmegaConf.to_container(cache, resolve=True)
        except Exception:
            return {}
    return {}


def _save_selection_cache(
    model: str, organism: str, variant: str = "default", method: str = None
):
    """Save model, organism, variant, and method selections to YAML cache file."""
    cache_path = _get_cache_path()
    cache = OmegaConf.create(
        {"model": model, "organism": organism, "variant": variant, "method": method}
    )
    with open(cache_path, "w") as f:
        OmegaConf.save(cache, f)


def get_organism_config(organism_name: str) -> DictConfig:
    """Load organism configuration file."""
    organism_path = CONFIGS_DIR / "organism" / f"{organism_name}.yaml"
    assert organism_path.exists(), f"Organism config not found: {organism_path}"
    return OmegaConf.load(organism_path)


def get_available_models_and_variants(organism_name: str) -> Dict[str, List[str]]:
    """
    Get available models and their variants for a given organism.

    Returns:
        Dict mapping {model_name: [variant1, variant2, ...]}
    """
    organism_cfg = get_organism_config(organism_name)

    if not hasattr(organism_cfg, "finetuned_models"):
        return {}

    models_and_variants = {}
    for model_name, variants_dict in organism_cfg.finetuned_models.items():
        models_and_variants[model_name] = sorted(variants_dict.keys())

    return models_and_variants


@st.cache_data
def load_config(
    model: str = None,
    organism: str = None,
    method: str = None,
    cfg_overwrites: List[str] = None,
) -> DictConfig:
    """
    Create minimal config for initializing diffing methods.

    Args:
        model: Model name
        organism: Organism name
        method: Method name

    Returns:
        Minimal DictConfig for the method
    """
    import torch

    # Get absolute path to configs directory
    config_dir = CONFIGS_DIR.resolve()

    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra with the configs directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Build overrides list: config group overrides first, then field overrides
        overrides = []
        if model is not None:
            overrides.append(f"model={model}")
        if organism is not None:
            overrides.append(f"organism={organism}")
        if method is not None:
            overrides.append(f"diffing/method={method}")
        if cfg_overwrites is not None:
            overrides.extend(cfg_overwrites)
        # Only override dtype if no CUDA (bfloat16 won't work on CPU)
        if not torch.cuda.is_available():
            overrides.append("model.dtype=float32")
        # Compose config with overwrites for model, organism, and method
        cfg = compose(config_name="config", overrides=overrides)

        # Resolve the configuration to ensure all interpolations are evaluated
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)
        return cfg


@st.cache_data
def get_available_results(cfg_overwrites: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Compile available results from all diffing methods.

    Returns:
        Dict mapping {model: {organism: [methods]}}
    """

    available = {}

    main_cfg = load_config(cfg_overwrites=cfg_overwrites)
    logger.info(f"Results base dir: {main_cfg.diffing.results_base_dir}")
    # Get available methods from configs
    available_methods = discover_methods()
    # Check each method for available results
    for method_name in available_methods:
        method_class = _get_method_class(method_name)
        logger.info(f"#####\n\nChecking method: {method_name}")
        logger.info(f"{method_class}")
        # Call static method directly on the class
        method_results = method_class.has_results(
            Path(main_cfg.diffing.results_base_dir)
        )
        logger.info(f"Method results: {str(method_results)[:100]}...")
        # Compile results into the global structure
        for model_name, organisms in method_results.items():
            if model_name not in available:
                available[model_name] = {}

            for organism_name, path in organisms.items():
                if organism_name not in available[model_name]:
                    available[model_name][organism_name] = []

                available[model_name][organism_name].append(method_name)

    return available


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Model Diffing Dashboard", page_icon="🧬", layout="wide"
    )

    cfg_overwrites = sys.argv[1:] if len(sys.argv) > 1 else []
    logger.info(f"Overwrites: {cfg_overwrites}")

    # Header row: title on left, control button on right
    _hdr_left, _hdr_right = st.columns([1, 0.2])
    with _hdr_left:
        st.title("🧬 Model Diffing Dashboard")
        st.markdown("Explore differences between base and finetuned models")
    with _hdr_right:
        if st.button(
            "⚙️Reset Model Cache",
            help="Clear cached models/tokenizers and free CUDA memory",
        ):
            with st.spinner("Resetting model cache and freeing CUDA memory..."):
                _reset_model_cache()
            st.success("Model cache cleared and CUDA memory emptied.")

    _import()
    from diffing.utils.dashboards import DualModelChatDashboard

    # Load cached selections
    cached_selections = _load_selection_cache()
    cached_organism = cached_selections.get("organism")
    cached_model = cached_selections.get("model")
    cached_variant = cached_selections.get("variant", "default")
    cached_method = cached_selections.get("method")

    # Discover available organisms
    available_organisms = sorted(discover_organisms())

    if not available_organisms:
        st.error("No organism configs found in configs/organism/")
        return

    # Organism selection (first)
    organism_index = 0
    if cached_organism and cached_organism in available_organisms:
        organism_index = available_organisms.index(cached_organism)
    selected_organism = st.selectbox(
        "Select Organism",
        available_organisms,
        index=organism_index,
        help="Choose an organism to explore",
    )

    if not selected_organism:
        return

    # Get available models and variants for this organism
    models_and_variants = get_available_models_and_variants(selected_organism)

    if not models_and_variants:
        st.error(f"No finetuned models found for organism '{selected_organism}'")
        return

    # Model and variant selection (side by side)
    col1, col2 = st.columns(2)

    with col1:
        available_models = sorted(models_and_variants.keys())
        model_index = 0
        if cached_model and cached_model in available_models:
            model_index = available_models.index(cached_model)
        selected_model = st.selectbox(
            "Select Base Model",
            available_models,
            index=model_index,
            help="Choose the base model architecture",
        )

    with col2:
        if selected_model:
            available_variants = models_and_variants[selected_model]
            variant_index = 0
            if "default" in available_variants:
                variant_index = available_variants.index("default")
            if cached_variant and cached_variant in available_variants:
                variant_index = available_variants.index(cached_variant)
            selected_variant = st.selectbox(
                "Select Variant",
                available_variants,
                index=variant_index,
                help="Choose the training variant (default, mix1-0p1, etc.)",
            )
        else:
            selected_variant = "default"
            st.info("Select a model first")

    # Save selections to cache if they changed
    if "last_selections" not in st.session_state:
        st.session_state.last_selections = {
            "organism": selected_organism,
            "model": selected_model,
            "variant": selected_variant,
            "method": None,
        }
    elif (
        st.session_state.last_selections["organism"] != selected_organism
        or st.session_state.last_selections["model"] != selected_model
        or st.session_state.last_selections["variant"] != selected_variant
    ):
        _save_selection_cache(
            selected_model,
            selected_organism,
            selected_variant,
            st.session_state.last_selections.get("method"),
        )
        st.session_state.last_selections = {
            "organism": selected_organism,
            "model": selected_model,
            "variant": selected_variant,
            "method": st.session_state.last_selections.get("method"),
        }

    tmp_cfg = load_config(
        selected_model,
        selected_organism,
        None,
        cfg_overwrites + [f"organism_variant={selected_variant}"],
    )

    # Get model configurations to access resolved finetuned model info
    from diffing.utils.configs import get_model_configurations

    _, ft_model_cfg = get_model_configurations(tmp_cfg)

    # Create Hugging Face model URL
    model_id = ft_model_cfg.model_id
    if ft_model_cfg.subfolder:
        full_model_path = f"{model_id}/{ft_model_cfg.subfolder}"
    else:
        full_model_path = model_id
    hf_url = f"https://huggingface.co/{model_id}"

    # Display selected configuration
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if ft_model_cfg.subfolder:
            st.markdown(
                f"**Model:** [{full_model_path}]({hf_url}) (subfolder: `{ft_model_cfg.subfolder}`)"
            )
        else:
            st.markdown(f"**Model:** [{model_id}]({hf_url})")
    with col2:
        # Display steering information if available
        if ft_model_cfg.steering_vector:
            steering_name = ft_model_cfg.steering_vector
            st.markdown(
                f"**Steering Configuration:** [{steering_name} (L{ft_model_cfg.steering_layer})](https://huggingface.co/science-of-finetuning/steering-vecs-{steering_name.replace('/', '/blob/main/')}_L{ft_model_cfg.steering_layer}.pt)"
            )

    st.markdown("---")

    # Discover available results for this organism/model combination
    available_results = get_available_results(cfg_overwrites)

    # Check if there are results for the selected combination
    available_methods = []
    organism_key = selected_organism
    if selected_variant and selected_variant != "default":
        organism_key = f"{selected_organism}_{selected_variant}"
    if selected_model in available_results:
        if organism_key in available_results[selected_model]:
            available_methods = available_results[selected_model][organism_key]
        elif selected_organism in available_results[selected_model]:
            available_methods = available_results[selected_model][selected_organism]
    if not available_methods and selected_organism != "None":
        st.warning(
            f"No diffing results found for {selected_model}/{selected_organism}. Run some experiments first!"
        )
        return

    method_options = ["Select a method..."] + available_methods
    method_index = 0
    if cached_method and cached_method in available_methods:
        method_index = method_options.index(cached_method)
    selected_method = st.selectbox(
        "Select Diffing Method", method_options, index=method_index
    )

    if selected_method == "Select a method...":
        return

    # Save method to cache if it changed
    if st.session_state.last_selections.get("method") != selected_method:
        _save_selection_cache(
            selected_model, selected_organism, selected_variant, selected_method
        )
        st.session_state.last_selections["method"] = selected_method

    # Create and initialize the diffing method
    try:
        start_time = time.time()
        with st.spinner("Loading method..."):
            cfg = load_config(
                selected_model,
                selected_organism,
                selected_method,
                cfg_overwrites + [f"organism_variant={selected_variant}"],
            )
            method_class = _get_method_class(selected_method)

            if method_class is None:
                st.error(f"Unknown method: {selected_method}")
                return

            # Initialize method (without loading models for visualization)
            method = method_class(cfg)
            if method.enable_chat:
                method_tab, chat_tab = st.tabs(["🔬 Method", "💬 Chat"])
                with method_tab:
                    start_time = time.time()
                    method.visualize()
                if method.enable_chat:
                    with chat_tab:
                        DualModelChatDashboard(method, title="Chat").display()
            else:
                method.visualize()

        logger.info(f"Method visualization took: {time.time() - start_time:.3f}s")

    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)
        traceback.print_exc()
        st.error(f"Full traceback:\n```python\n{traceback.format_exc()}```")


if __name__ == "__main__":
    main()
