"""
Shared visualization utilities for diffing methods.

This module provides common functionality for converting diffing results
into HTML visualizations using the tiny-dashboard library.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import re
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

from src.utils.model import (
    load_tokenizer_from_config,
    logit_lens,
    patch_scope,
    multi_patch_scope,
)
from src.utils.configs import ModelConfig
from src.diffing.methods.diffing_method import DiffingMethod


def query_tokens_in_distribution(
    query_tokens: List[str],
    tokenizer: AutoTokenizer,
    probs: torch.Tensor,
    direction: str = "positive",
) -> List[Dict[str, Any]]:
    """
    Query specific tokens in a probability distribution and return their ranks and probabilities.

    Args:
        query_tokens: List of token strings to query
        tokenizer: Tokenizer for token-to-id mapping
        probs: Full probability distribution tensor
        direction: "positive" or "negative" for display purposes

    Returns:
        List of dictionaries containing token info (token, token_id, probability, rank)
    """
    results = []

    # Sort probabilities to get ranks
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Create a mapping from token_id to rank
    rank_mapping = {int(sorted_indices[i]): i + 1 for i in range(len(sorted_indices))}

    for token_str in query_tokens:
        try:
            # Handle different token formats
            token_str = token_str.strip()

            # Try to tokenize the string
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)

            if len(token_ids) == 1:
                token_id = token_ids[0]
                prob = float(probs[token_id])
                rank = rank_mapping[token_id]

                results.append(
                    {
                        "token": token_str,
                        "token_id": token_id,
                        "probability": prob,
                        "rank": rank,
                        "direction": direction,
                    }
                )
            elif len(token_ids) > 1:
                # Multi-token case - report each subtoken
                for i, token_id in enumerate(token_ids):
                    subtoken = tokenizer.decode([token_id])
                    prob = float(probs[token_id])
                    rank = rank_mapping[token_id]

                    results.append(
                        {
                            "token": f"{token_str}[{i}]: '{subtoken}'",
                            "token_id": token_id,
                            "probability": prob,
                            "rank": rank,
                            "direction": direction,
                        }
                    )
            else:
                # Empty tokenization
                results.append(
                    {
                        "token": token_str,
                        "token_id": None,
                        "probability": None,
                        "rank": None,
                        "direction": direction,
                        "error": "Could not tokenize",
                    }
                )

        except Exception as e:
            results.append(
                {
                    "token": token_str,
                    "token_id": None,
                    "probability": None,
                    "rank": None,
                    "direction": direction,
                    "error": str(e),
                }
            )

    return results


def get_top_k_tokens(
    probs: torch.Tensor, tokenizer: AutoTokenizer, k: int = 10
) -> List[Tuple[str, int, float]]:
    """
    Get top-k tokens from probability distribution.

    Args:
        probs: Probability distribution tensor
        tokenizer: Tokenizer for decoding
        k: Number of top tokens to return

    Returns:
        List of (token, token_id, probability) tuples
    """
    top_probs, top_indices = torch.topk(probs, k=k, largest=True)

    tokens = []
    for i in range(k):
        token_id = int(top_indices[i])
        token = tokenizer.decode([token_id])
        prob = float(top_probs[i])
        tokens.append((token, token_id, prob))

    return tokens


def display_token_query_results(query_results: List[Dict[str, Any]]):
    """Display results from token queries in a formatted table."""
    if not query_results:
        st.info("No tokens queried yet.")
        return

    import pandas as pd

    # Convert results to DataFrame
    df_data = []
    for result in query_results:
        df_data.append(
            {
                "Token": result["token"],
                "Token ID": result["token_id"],
                "Probability": f"{result['probability']:.6f}",
                "Rank": result["rank"],
                "Direction": result["direction"],
            }
        )

    df = pd.DataFrame(df_data)

    # Color coding function
    def highlight_direction(row):
        colors = []
        for col in df.columns:
            if row["Direction"] == "positive":
                colors.append("background-color: rgba(0, 255, 0, 0.1)")
            else:
                colors.append("background-color: rgba(255, 0, 0, 0.1)")
        return colors

    # Apply styling
    styled_df = df.style.apply(highlight_direction, axis=1)

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Token": st.column_config.TextColumn("Token", width="medium"),
            "Token ID": st.column_config.NumberColumn("Token ID", width="small"),
            "Probability": st.column_config.TextColumn("Probability", width="medium"),
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Direction": st.column_config.TextColumn("Direction", width="small"),
        },
    )


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
    slider_step: int = 1,
    slider_min_value: int = 0,
    slider_max_value: int = 200,
    slider_value: int = 100,
    latent_type_name: str = "Latent",
    patch_scope_add_scaler: bool = False,
    custom_latent_options: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
):
    """Render logit lens analysis tab for SAE latents."""
    # UI Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        latent_idx = st.selectbox(
            f"{latent_type_name} Index",
            options=(
                custom_latent_options
                if custom_latent_options
                else list(range(max_latent_idx))
            ),
            index=0,
            help=f"Choose which latent to analyze (0-{max_latent_idx-1})",
        )

    with col2:
        model_choice = st.selectbox(
            "Model",
            options=["Finetuned Model", "Base Model"],
            index=0,
            help="Choose which model to use for logit lens analysis",
            key="model_choice_selector_latent_lens_tab",
        )
    with col3:
        method_choice = st.selectbox(
            "Method",
            options=[
                "Select Method",
                "Logit Lens",
                "Patch Scope",
                "Patch Scope (Multi)",
            ],
            index=0,
            help="Choose which method to use for logit lens analysis",
        )

    # Only resolve model when a concrete method is selected
    if method_choice != "Select Method":
        if model_choice == "Base Model":
            model = method.base_model
        else:
            model = method.finetuned_model

    # Additional controls for Patch Scope methods
    if patch_scope_add_scaler and method_choice in [
        "Patch Scope",
        "Patch Scope (Multi)",
    ]:
        # Load recommended scale if available
        recommended_scale: Optional[float] = None
        # Infer position/latent index from selection text
        pos_val: Optional[int] = None
        try:
            if isinstance(latent_idx, str):
                m = re.search(r"\d+", latent_idx)
                if m is not None:
                    pos_val = int(m.group(0))
            elif isinstance(latent_idx, int) and custom_latent_options is not None:
                if 0 <= latent_idx < len(custom_latent_options):
                    opt = str(custom_latent_options[latent_idx])
                    m2 = re.search(r"\d+", opt)
                    if m2 is not None:
                        pos_val = int(m2.group(0))
        except Exception:
            pos_val = None

        if (dataset_name is not None) and (pos_val is not None):
            aps_path = (
                method.results_dir
                / f"layer_{layer}"
                / dataset_name
                / f"auto_patch_scope_pos_{pos_val}.pt"
            )
            if aps_path.exists():
                aps_data = torch.load(aps_path, map_location="cpu")
                if isinstance(aps_data, dict) and ("best_scale" in aps_data):
                    recommended_scale = float(aps_data["best_scale"])  # type: ignore[arg-type]

        slider_default = (
            float(recommended_scale)
            if (recommended_scale is not None)
            else slider_value
        )
        scaler = st.slider(
            "Patch Scope Scaler",
            min_value=slider_min_value,
            max_value=slider_max_value,
            value=slider_default,
            step=slider_step,
            help="Scale factor for the latent vector when patching",
        )
        if recommended_scale is not None:
            st.info(f"Recommended value: {recommended_scale:.3f}")
    else:
        scaler = 1

    # Intersection top-k for multi patch scope
    if method_choice == "Patch Scope (Multi)":
        intersection_top_k = st.number_input(
            "Intersection Top-K (per prompt)",
            min_value=1,
            max_value=2000,
            value=100,
            step=5,
            help="Top-K tokens per prompt to intersect. Averaged probabilities are returned only for the intersection; others are zero.",
        )

    # Analyze latent logits
    if method_choice == "Select Method":
        return
    try:
        latent = get_latent_fn(latent_idx).to(method.device)

        # Get full probability distributions
        if method_choice == "Logit Lens":
            pos_probs, neg_probs = logit_lens(latent, model)
        elif method_choice == "Patch Scope":
            pos_probs, neg_probs = patch_scope(
                latent, model, method.tokenizer, layer, scaler=scaler
            )
        elif method_choice == "Patch Scope (Multi)":
            # Use default prompts defined in multi_patch_scope
            pos_probs, neg_probs = multi_patch_scope(
                latent,
                model,
                method.tokenizer,
                layer,
                scaler=scaler,
                top_k=int(intersection_top_k),
            )

        # Display results
        st.markdown(f"### {latent_type_name} {latent_idx} {method_choice} Analysis")

        # Top-K Configuration above the main table
        top_k = st.number_input(
            "Number of Top Tokens to Display",
            min_value=5,
            max_value=1000,
            value=10,
            step=5,
            help="Configure how many top/bottom tokens to display",
        )

        # Get tokens with configurable k
        top_tokens = get_top_k_tokens(pos_probs, method.tokenizer, k=top_k)
        bottom_tokens = get_top_k_tokens(neg_probs, method.tokenizer, k=top_k)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Promoted Tokens")
            display_colored_token_table(top_tokens, "top")

        with col2:
            st.markdown("#### Top Suppressed Tokens")
            display_colored_token_table(bottom_tokens, "bottom")

        # Detailed Analysis Section
        with st.expander("🔍 Query Tokens", expanded=False):
            st.markdown("### Token Query")
            st.markdown(
                "Query specific tokens to see their ranks and probabilities in both directions."
            )

            # Token input
            query_input = st.text_area(
                "Enter tokens to query (one per line)",
                height=100,
                placeholder="Enter tokens like:\nthe\nhello\nworld\n...",
                help="Enter one token per line. Multi-token strings will be broken down into subtokens.",
            )

            # Query button and results
            if st.button("Query Tokens", type="primary"):
                if query_input.strip():
                    query_tokens = [
                        token
                        for token in query_input.strip().split("\n")
                        if token.strip()
                    ]

                    if query_tokens:
                        # Query in both directions
                        pos_results = query_tokens_in_distribution(
                            query_tokens,
                            method.tokenizer,
                            pos_probs,
                            direction="promoted",
                        )
                        neg_results = query_tokens_in_distribution(
                            query_tokens,
                            method.tokenizer,
                            neg_probs,
                            direction="suppressed",
                        )

                        st.markdown("#### Query Results")

                        # Display results in two columns
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("##### Promoted Tokens")
                            display_token_query_results(pos_results)

                        with col2:
                            st.markdown("##### Suppressed Tokens")
                            display_token_query_results(neg_results)

                    else:
                        st.warning("Please enter at least one token to query.")
                else:
                    st.warning("Please enter tokens to query.")

    except Exception as e:
        st.error(f"Error analyzing {latent_type_name} logits: {str(e)}")
