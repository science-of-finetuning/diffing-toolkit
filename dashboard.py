"""
Streamlit dashboard for visualizing model diffing results.

This dashboard dynamically discovers available model organisms and diffing methods
from the filesystem and provides an interactive interface to explore the results.
"""

import streamlit as st
from omegaconf import DictConfig, OmegaConf
import sys
from typing import Dict, List, Tuple
import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
import time


@st.cache_resource(show_spinner="Importing dependencies: torch...")
def _import_torch():
    import torch


@st.cache_resource(show_spinner="Importing dependencies: transformers...")
def _import_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource(show_spinner="Importing dependencies: nnsight...")
def _import_nnsight():
    import nnsight


@st.cache_resource(show_spinner="Importing dependencies: others...")
def _import_others():
    import src


def _import():
    _import_torch()
    _import_transformers()
    _import_nnsight()
    _import_others()


def _reset_model_cache():
    """Clear cached models/tokenizers and free CUDA memory."""
    from src.utils import model as model_utils

    model_utils.clear_cache()


def _get_method_class(method_name: str) -> "DiffingMethod":
    """Get the method class for a given method name. Wrapped as the import is not available in the global scope and the main function loads quickly"""
    from src.pipeline.diffing_pipeline import get_method_class

    return get_method_class(method_name)


def discover_organisms() -> List[str]:
    """Discover available organisms from the configs directory."""
    organism_configs = Path("configs/organism").glob("*.yaml")
    return [f.stem for f in organism_configs]


def discover_methods() -> List[str]:
    """Discover available diffing methods from the configs directory."""
    method_configs = Path("configs/diffing/method").glob("*.yaml")
    methods = [f.stem for f in method_configs]
    return methods


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
    config_dir = Path("configs").resolve()

    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"

    # Initialize Hydra with the configs directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides = [f"model.dtype={dtype}"]
        if model is not None:
            overrides.append(f"model={model}")
        if organism is not None:
            overrides.append(f"organism={organism}")
        if method is not None:
            overrides.append(f"diffing/method={method}")
        if cfg_overwrites is not None:
            overrides.extend(cfg_overwrites)
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
    print("Results base dir:", main_cfg.diffing.results_base_dir)
    # Get available methods from configs
    available_methods = discover_methods()
    # Check each method for available results
    for method_name in available_methods:
        method_class = _get_method_class(method_name)
        print(f"#####\n\nChecking method: {method_name}")
        print(method_class)
        # Call static method directly on the class
        method_results = method_class.has_results(
            Path(main_cfg.diffing.results_base_dir)
        )
        print(f"Method results: {method_results}")
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
    print(f"Overwrites: {cfg_overwrites}")

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
    from src.utils.dashboards import DualModelChatDashboard

    # Discover available results
    available_results = get_available_results(cfg_overwrites)

    if not available_results:
        st.error("No diffing results found. Run some diffing experiments first!")
        return

    # Model selection
    available_models = list(available_results.keys())
    selected_model = st.selectbox("Select Base Model", available_models)

    if not selected_model:
        return

    # Organism selection
    available_organisms = list(available_results[selected_model].keys())
    selected_organism = st.selectbox("Select Organism", available_organisms)

    if not selected_organism:
        return

    tmp_cfg = load_config(selected_model, selected_organism, None, cfg_overwrites)

    # Create Hugging Face model URL
    model_id = tmp_cfg.organism.finetuned_model.model_id
    hf_url = f"https://huggingface.co/{model_id}"
    # Create columns for steering info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model:** [{model_id}]({hf_url})")
    with col2:
        # Display steering information if available
        if hasattr(tmp_cfg.organism.finetuned_model, "steering_vector"):
            steering_name = tmp_cfg.organism.finetuned_model.steering_vector
            st.markdown(
                f"**Steering Configuration:** [{steering_name} (L{tmp_cfg.organism.finetuned_model.steering_layer})](https://huggingface.co/science-of-finetuning/steering-vecs-{steering_name.replace('/', '/blob/main/')}_L{tmp_cfg.organism.finetuned_model.steering_layer}.pt)"
            )

    # Method selection
    available_methods = available_results[selected_model][selected_organism]
    if not available_methods:
        st.warning(f"No results found for {selected_model}/{selected_organism}")
        return
    selected_method = st.selectbox(
        "Select Diffing Method", ["Select a method..."] + available_methods, index=0
    )

    if selected_method == "Select a method...":
        return

    # Create and initialize the diffing method
    try:
        start_time = time.time()
        with st.spinner("Loading method..."):
            cfg = load_config(
                selected_model, selected_organism, selected_method, cfg_overwrites
            )
            method_class = _get_method_class(selected_method)

            if method_class is None:
                st.error(f"Unknown method: {selected_method}")
                return

            # Initialize method (without loading models for visualization)
            method = method_class(cfg)

            method_tab, chat_tab = st.tabs(["🔬 Method", "💬 Chat"])

            with method_tab:
                start_time = time.time()
                method.visualize()

            with chat_tab:
                DualModelChatDashboard(method, title="Chat").display()
        print(f"Method visualization took: {time.time() - start_time:.3f}s")

    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
