"""
Streamlit dashboard for visualizing model diffing results with blinded organism selection.

Differences from `dashboard.py`:
- Only show hashed slugs for organisms (and base model) in the selection list
- User selects an organism by slug, then we reveal the corresponding base model
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.utils.configs import CONFIGS_DIR


@st.cache_resource(show_spinner="Importing dependencies: torch...")
def _import_torch() -> None:
    import torch  # noqa: F401


@st.cache_resource(show_spinner="Importing dependencies: transformers...")
def _import_transformers() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401


@st.cache_resource(show_spinner="Importing dependencies: nnsight...")
def _import_nnsight() -> None:
    import nnsight  # noqa: F401


@st.cache_resource(show_spinner="Importing dependencies: others...")
def _import_others() -> None:
    import src  # noqa: F401


def _import() -> None:
    _import_torch()
    _import_transformers()
    _import_nnsight()
    _import_others()


def _get_method_class(method_name: str):
    """Get the method class for a given method name.

    Returns the class rather than importing it globally to keep initial load fast.
    """
    from src.pipeline.diffing_pipeline import get_method_class

    return get_method_class(method_name)


def _discover_methods() -> List[str]:
    """Discover available diffing methods from the configs directory."""
    method_configs = Path("configs/diffing/method").glob("*.yaml")
    return [f.stem for f in method_configs]


def _make_slug(model_name: str, organism_name: str, length: int = 12) -> str:
    """Create a deterministic slug from the model and organism names."""
    basis = f"{organism_name}::{model_name}".encode("utf-8")
    return hashlib.sha256(basis).hexdigest()[:length]


@st.cache_data
def load_config(
    model: str | None = None,
    organism: str | None = None,
    method: str | None = None,
    cfg_overwrites: List[str] | None = None,
) -> DictConfig:
    """Create minimal Hydra config for initializing diffing methods."""
    import torch

    config_dir = CONFIGS_DIR

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides: List[str] = [f"model.dtype={dtype}"]
        if model is not None:
            overrides.append(f"model={model}")
        if organism is not None:
            overrides.append(f"organism={organism}")
        if method is not None:
            overrides.append(f"diffing/method={method}")
        if cfg_overwrites is not None:
            overrides.extend(cfg_overwrites)

        cfg = compose(config_name="config", overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)
        return cfg


@st.cache_data
def get_available_results(cfg_overwrites: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Compile available results for all diffing methods.

    Returns a mapping: {base_model: {organism_name: [methods]}}
    """
    available: Dict[str, Dict[str, List[str]]] = {}

    main_cfg = load_config(cfg_overwrites=cfg_overwrites)
    print("Results base dir:", main_cfg.diffing.results_base_dir)

    for method_name in _discover_methods():
        method_class = _get_method_class(method_name)
        print(f"#####\n\nChecking method: {method_name}")
        print(method_class)
        method_results = method_class.has_results(
            Path(main_cfg.diffing.results_base_dir)
        )
        print(f"Method results: {method_results}")

        for model_name, organisms in method_results.items():
            if model_name not in available:
                available[model_name] = {}
            for organism_name, _path in organisms.items():
                available.setdefault(model_name, {}).setdefault(organism_name, [])
                if method_name not in available[model_name][organism_name]:
                    available[model_name][organism_name].append(method_name)

    return available


def main() -> None:
    """Main dashboard function for the blinded organism selection flow."""
    st.set_page_config(page_title="Diffing Game", page_icon="🕶️", layout="wide")

    cfg_overwrites = sys.argv[1:] if len(sys.argv) > 1 else []
    print(f"Overwrites: {cfg_overwrites}")

    st.title("🕶️ Diffing Game")
    st.markdown(
        "Select an organism by its slug; the base model is revealed after selection."
    )

    _import()
    from src.utils.dashboards import DualModelChatDashboard

    # Discover available results across methods
    available_results = get_available_results(cfg_overwrites)
    assert isinstance(available_results, dict)
    if not available_results:
        st.error("No diffing results found. Run some diffing experiments first!")
        return

    # Build slug mapping: slug -> (base_model, organism_name, methods)
    slug_to_entry: Dict[str, Tuple[str, str, List[str]]] = {}
    for model_name, organisms in available_results.items():
        for organism_name, methods in organisms.items():
            slug = _make_slug(model_name=model_name, organism_name=organism_name)
            assert (
                slug not in slug_to_entry
            ), f"Slug collision for {organism_name} / {model_name}"
            slug_to_entry[slug] = (model_name, organism_name, sorted(methods))

    if not slug_to_entry:
        st.error("No organisms discovered in results.")
        return

    sorted_slugs = sorted(slug_to_entry.keys())
    selected_slug = st.selectbox(
        "Select Organism (Slug)", ["Select an organism..."] + sorted_slugs, index=0
    )
    if selected_slug == "Select an organism...":
        return

    model_name, organism_name, methods = slug_to_entry[selected_slug]

    # Reveal base model with a small reveal button to the right
    left_col, right_col = st.columns([4, 1])
    with left_col:
        st.info(f"Base Model: {model_name}")
    with right_col:
        if "_reveal_map" not in st.session_state:
            st.session_state["_reveal_map"] = {}
        if st.button("Reveal organism", key=f"reveal_{selected_slug}"):
            st.session_state["_reveal_map"][selected_slug] = True
        if st.session_state["_reveal_map"].get(selected_slug, False):
            st.caption(f"Organism: {organism_name}")

    # Minimal config can be created on-demand if needed later

    # Method selection for this organism
    if not methods:
        st.warning(f"No results found for organism {selected_slug}")
        return

    selected_method = st.selectbox(
        "Select Diffing Method", ["Select a method..."] + methods, index=0
    )
    if selected_method == "Select a method...":
        return

    # Initialize and visualize the method
    start_time = time.time()
    with st.spinner("Loading method..."):
        cfg = load_config(
            model=model_name,
            organism=organism_name,
            method=selected_method,
            cfg_overwrites=cfg_overwrites,
        )
        method_class = _get_method_class(selected_method)
        assert method_class is not None, f"Unknown method: {selected_method}"
        method = method_class(cfg)

        method_tab, chat_tab = st.tabs(["🔬 Method", "💬 Chat"])

        with method_tab:
            method.visualize()

        with chat_tab:
            DualModelChatDashboard(method, title="Chat").display()

    print(f"Method visualization took: {time.time() - start_time:.3f}s")


if __name__ == "__main__":
    main()
