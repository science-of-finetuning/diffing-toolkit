"""
Shared visualization utilities for diffing methods.

This module provides common functionality for converting diffing results
into HTML visualizations using the tiny-dashboard library.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import torch
import streamlit as st
from pathlib import Path
import sys
import json
from tiny_dashboard.html_utils import (
    create_example_html,
    create_base_html,
    create_highlighted_tokens_html,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from numpy import array

from src.utils.model import load_tokenizer_from_config, logit_lens, patch_scope
from src.utils.configs import ModelConfig
from src.diffing.methods.diffing_method import DiffingMethod


@st.cache_data
def convert_max_examples_to_dashboard_format(
    max_examples: List[Dict[str, Any]],
    model_cfg: ModelConfig,
) -> List[Tuple[float, List[str], List[float], str]]:
    """
    Convert max_activating_examples from diffing results to dashboard format.

    Args:
        max_examples: List of max activating examples from diffing results
        model_cfg: Model configuration containing tokenizer information

    Returns:
        List of tuples (max_activation_value, tokens, activation_values, text)
    """
    tokenizer = load_tokenizer_from_config(model_cfg)

    dashboard_examples = []

    for example in max_examples:
        max_score = example["max_score"]
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
        scores_per_token = array(example["scores_per_token"])
        scores_per_token = scores_per_token - scores_per_token.min()

        # Get the text for search functionality
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

        dashboard_examples.append((max_score, tokens, scores_per_token, text))

    return dashboard_examples


def create_html_highlight(
    tokens: List[str],
    activations: List[float],
    tokenizer: AutoTokenizer,
    max_idx: Optional[int] = None,
    min_max_act: Optional[float] = None,
    window_size: int = 50,
    show_full: bool = False,
) -> str:
    """
    Create HTML highlighting for tokens based on activation values.

    Args:
        tokens: List of token strings
        activations: List of activation values per token
        tokenizer: HuggingFace tokenizer
        max_idx: Index of maximum activation (auto-computed if None)
        min_max_act: Normalization value for activations
        window_size: Number of tokens to show around max activation
        show_full: Whether to show full sequence or windowed

    Returns:
        HTML string with highlighted tokens
    """
    act_tensor = torch.tensor(activations)

    if max_idx is None:
        max_idx = int(torch.argmax(act_tensor).item())

    if min_max_act is None:
        min_max_act, min_max_act_negative = act_tensor.max(), act_tensor.min().abs()
    # Apply windowing if not showing full sequence
    if not show_full:
        start_idx = max(0, max_idx - window_size)
        end_idx = min(len(tokens), max_idx + window_size + 1)
        tokens = tokens[start_idx:end_idx]
        act_tensor = act_tensor[start_idx:end_idx]

    return create_highlighted_tokens_html(
        tokens=tokens,
        activations=act_tensor,
        tokenizer=tokenizer,
        highlight_features=0,  # Single feature case
        color1=(255, 0, 0),  # Red color
        activation_names=["Activation"],
        min_max_act=min_max_act,
        min_max_act_negative=min_max_act_negative,
        separate_positive_negative_normalization=True,
    )


def filter_examples_by_search(
    examples: List[
        Union[
            Tuple[float, List[str], List[float], str],
            Tuple[float, List[str], List[float], str, str],
        ]
    ],
    search_term: str,
) -> List[
    Union[
        Tuple[float, List[str], List[float], str],
        Tuple[float, List[str], List[float], str, str],
    ]
]:
    """
    Filter examples by search term.

    Args:
        examples: List of (max_score, tokens, scores_per_token, text[, dataset_name]) tuples
        search_term: Term to search for in the text

    Returns:
        Filtered list of examples
    """
    if not search_term.strip():
        return examples

    search_term = search_term.lower().strip()
    filtered = []

    for example in examples:
        # Extract text (always 4th element regardless of tuple length)
        text = example[3]
        if search_term in text.lower():
            filtered.append(example)

    return filtered


def create_dataset_name_html(dataset_name: str) -> str:
    """
    Create HTML for dataset name display in top right corner.

    Args:
        dataset_name: Name of the dataset

    Returns:
        HTML string for dataset name display
    """
    return f"""
    <div style="position: absolute; top: 5px; right: 10px; 
                background-color: #f0f0f0; padding: 2px 6px; 
                border-radius: 3px; font-size: 0.8em; 
                color: #666; border: 1px solid #ddd;">
        {dataset_name}
    </div>
    """


@st.cache_data
def create_examples_html(
    examples: List[
        Union[
            Tuple[float, List[str], List[float], str],
            Tuple[float, List[str], List[float], str, str],
        ]
    ],
    _tokenizer: AutoTokenizer,
    title: str = "Max Activating Examples",
    max_examples: int = 30,
    window_size: int = 50,
    use_absolute_max: bool = False,
    search_term: str = "",
) -> str:
    """
    Create HTML for a list of max activating examples.

    Args:
        examples: List of (max_score, tokens, scores_per_token, text[, dataset_name]) tuples
        _tokenizer: HuggingFace tokenizer
        title: Title for the HTML page
        max_examples: Maximum number of examples to display
        window_size: Number of tokens to show around max activation
        use_absolute_max: Whether to normalize using absolute maximum
        search_term: Optional search term to filter examples

    Returns:
        Complete HTML string
    """
    # Filter examples if search term provided
    if search_term.strip():
        examples = filter_examples_by_search(examples, search_term)

    content_parts = []
    min_max_act = None

    if use_absolute_max and examples:
        min_max_act = examples[0][0]  # First example has highest score

    # Process examples, handling both 4-tuple and 5-tuple formats
    for example in examples[:max_examples]:
        # Extract common elements
        max_act, tokens, token_acts, text = example[:4]

        # Extract dataset_name if present
        dataset_name = example[4] if len(example) > 4 else None
        max_idx = int(torch.argmax(torch.tensor(token_acts)).item())

        # Create both collapsed and full versions
        collapsed_html = create_html_highlight(
            tokens, token_acts, _tokenizer, max_idx, min_max_act, window_size, False
        )
        full_html = create_html_highlight(
            tokens, token_acts, _tokenizer, max_idx, min_max_act, window_size, True
        )

        # Add dataset name to HTML if provided
        if dataset_name:
            dataset_html = create_dataset_name_html(dataset_name)
            # Wrap the content with relative positioning to allow absolute positioning of dataset name
            collapsed_html = (
                f'<div style="position: relative;">{dataset_html}{collapsed_html}</div>'
            )
            full_html = (
                f'<div style="position: relative;">{dataset_html}{full_html}</div>'
            )

        content_parts.append(create_example_html(max_act, collapsed_html, full_html))

    return create_base_html(title=title, content=content_parts)


def render_streamlit_html(html_content: str, height: int = 800) -> None:
    """
    Render HTML content in Streamlit with proper styling.

    Args:
        html_content: HTML content to render
        height: Height of the component in pixels
    """
    # Use Streamlit's HTML component with scrolling
    st.components.v1.html(html_content, height=height, scrolling=True)


@st.fragment
def _render_statistics_tab(statistics_function: Callable, title: str):
    """Render statistics tab as a fragment to prevent full page reloads."""
    statistics_function()


@st.fragment
def _render_interactive_tab(interactive_function: Callable, title: str):
    """Render interactive tab as a fragment to prevent full page reloads."""
    interactive_function()


def multi_tab_interface(tabs: List[Tuple[str, Callable]], title: str):
    """
    Create a multi-tab interface with dynamic number of tabs and optimized rendering.
    Uses Streamlit fragments to prevent unnecessary full page reloads.

    Args:
        tabs: List of tuples containing (tab_title, tab_function)
        title: Title for the interface

    Returns:
        List of Streamlit tab components
    """
    st.subheader(title)
    for st_tab, (_, fn) in zip(st.tabs([t for t, _ in tabs]), tabs):
        with st_tab:
            _tab_fragment(fn)


@st.fragment
def _tab_fragment(render_fn):
    with st.container():
        render_fn()


def statistic_interactive_tab(
    statistics_function: Callable, interactive_function: Callable, title: str
):
    """
    Create a tab for statistics and interactive analysis with optimized rendering.
    Uses Streamlit fragments to prevent unnecessary full page reloads.

    Args:
        statistics_function: Function to compute statistics
        interactive_function: Function to create interactive analysis

    Returns:
        Streamlit tab component
    """
    st.subheader(title)

    tab1, tab2 = st.tabs(["📊 Dataset Statistics", "🔥 Interactive"])

    with tab1:
        _render_statistics_tab(statistics_function, title)

    with tab2:
        _render_interactive_tab(interactive_function, title)

    return tab1, tab2


def display_colored_token_table(tokens_data, table_type):
    """
    Display a colored table of tokens with background colors based on relative probability.

    Args:
        tokens_data: List of (token, token_id, probability) tuples
        table_type: "top" or "bottom" to determine color scheme
    """
    import pandas as pd
    import numpy as np

    # Create DataFrame
    df_data = []
    for i, (token, token_id, prob) in enumerate(tokens_data, 1):
        df_data.append(
            {
                "Rank": i,
                "Token": repr(token),
                "ID": token_id,
                "Probability": f"{prob:.6f}",
            }
        )

    df = pd.DataFrame(df_data)

    # Get probabilities for coloring
    probs = np.array([prob for _, _, prob in tokens_data])

    # Normalize probabilities for coloring (0 to 1 scale within this group)
    if len(probs) > 1:
        min_prob = probs.min()
        max_prob = probs.max()
        if max_prob > min_prob:
            normalized_probs = (probs - min_prob) / (max_prob - min_prob)
        else:
            normalized_probs = np.ones_like(probs) * 0.5
    else:
        normalized_probs = np.array([0.5])

    # Define color function
    def color_rows(row):
        idx = row.name
        intensity = normalized_probs[idx]

        if table_type == "top":
            # Green scale for top tokens (higher probability = more intense green)
            green_intensity = int(255 * (0.3 + 0.7 * intensity))  # 76 to 255
            color = f"background-color: rgba(0, {green_intensity}, 0, 0.3)"
        else:
            # Red scale for bottom tokens (lower probability = more intense red)
            red_intensity = int(255 * (0.3 + 0.7 * (1 - intensity)))  # 76 to 255
            color = f"background-color: rgba({red_intensity}, 0, 0, 0.3)"

        return [color] * len(row)

    # Apply styling
    styled_df = df.style.apply(color_rows, axis=1)

    # Display the table
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Token": st.column_config.TextColumn("Token", width="medium"),
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Probability": st.column_config.TextColumn("Probability", width="medium"),
        },
    )


def render_latent_lens_tab(
    method: DiffingMethod,
    get_latent_fn: Callable,
    max_latent_idx: int,
    layer: int,
    latent_type_name: str = "Latent",
    patch_scope_add_scaler: bool = False,
    custom_latent_options: Optional[List[str]] = None,
):
    """Render logit lens analysis tab for SAE latents."""
    # UI Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        latent_idx = st.selectbox(
            f"{latent_type_name} Index",
            options=custom_latent_options if custom_latent_options else list(range(max_latent_idx)),
            index=0,
            help=f"Choose which latent to analyze (0-{max_latent_idx-1})",
        )

    with col2:
        model_choice = st.selectbox(
            "Model",
            options=["Base Model", "Finetuned Model"],
            index=0,
            help="Choose which model to use for logit lens analysis",
        )
    with col3:
        method_choice = st.selectbox(
            "Method",
            options=["Logit Lens", "Patch Scope"],
            index=0,
            help="Choose which method to use for logit lens analysis",
        )

    # Get the appropriate model
    if model_choice == "Base Model":
        model = method.base_model
    else:
        model = method.finetuned_model

    # Additional controls for Patch Scope method
    if method_choice == "Patch Scope" and patch_scope_add_scaler:
        scaler = st.slider(
            "Patch Scope Scaler",
            min_value=1,
            max_value=200,
            value=100,
            help="Scale factor for the latent vector when patching"
        )
    else:
        scaler = 1

    # Analyze latent logits
    try:
        latent = get_latent_fn(latent_idx)
        if method_choice == "Logit Lens":
            top_tokens, bottom_tokens = logit_lens(latent, model, method.tokenizer)
        elif method_choice == "Patch Scope":
            top_tokens, bottom_tokens = patch_scope(latent, model, method.tokenizer, layer, scaler=scaler)

        # Display results
        st.markdown(f"### {latent_type_name} {latent_idx} {method_choice} Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Promoted Tokens")
            display_colored_token_table(top_tokens, "top")

        with col2:
            st.markdown("#### Top Suppressed Tokens")
            display_colored_token_table(bottom_tokens, "bottom")

    except Exception as e:
        st.error(f"Error analyzing {latent_type_name} logits: {str(e)}")
