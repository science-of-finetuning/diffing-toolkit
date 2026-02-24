"""
Dashboard and visualization code for SAE Difference diffing method.

Separated to avoid triggering streamlit cache warnings at import time.
"""

from typing import Dict, Any
from pathlib import Path
from collections import defaultdict
import json
import base64
import torch
import numpy as np
import pandas as pd
import streamlit as st

from diffing.utils.dashboards import (
    AbstractOnlineDiffingDashboard,
    SteeringDashboard,
    MaxActivationDashboardComponent,
)
from diffing.utils.max_act_store import ReadOnlyMaxActStore
from diffing.utils.visualization import multi_tab_interface, render_latent_lens_tab
from diffing.utils.dictionary.steering import display_steering_results
from diffing.utils.dictionary.utils import load_latent_df, load_dictionary_model


def visualize(method) -> None:
    """
    Create Streamlit visualization for SAE difference results with 4 tabs.

    Features:
    - MaxAct tab: Display maximum activating examples using MaxActivationDashboardComponent
    - Online Inference tab: Real-time SAE analysis similar to KL/NormDiff dashboards
    - Latent Statistics tab: Interactive exploration of latent DataFrame with filtering
    - Plots tab: Display all generated plots from the analysis pipeline
    """
    st.subheader("SAE Difference Analysis")

    # Initialize session state for tab selection
    tab_session_key = "sae_difference_active_tab"
    if tab_session_key not in st.session_state:
        st.session_state[tab_session_key] = 0  # Default to first tab

    # Global SAE selector
    available_saes = _get_available_sae_directories(method.results_dir)
    if not available_saes:
        st.error(
            f"No trained SAE directories found in {method.results_dir / 'sae_difference'}"
        )
        return

    # Group SAEs by layer for easier selection
    saes_by_layer = defaultdict(list)
    for sae_info in available_saes:
        saes_by_layer[sae_info["layer"]].append(sae_info)

    # Initialize selected_sae_info to None
    selected_sae_info = None

    # Get unique sorted layers
    unique_layers = sorted(saes_by_layer.keys())

    if not unique_layers:
        st.error("No layers with trained SAEs found.")
        return

    # Initialize session state for SAE selection
    sae_session_keys = {
        "selected_layer": "sae_difference_selected_layer",
        "selected_dictionary": "sae_difference_selected_dictionary",
    }

    # Initialize SAE selection session state
    if sae_session_keys["selected_layer"] not in st.session_state:
        st.session_state[sae_session_keys["selected_layer"]] = unique_layers[0]
    if sae_session_keys["selected_dictionary"] not in st.session_state:
        # Get first dictionary for the first layer
        first_layer_saes = saes_by_layer[unique_layers[0]]
        if first_layer_saes:
            st.session_state[sae_session_keys["selected_dictionary"]] = (
                first_layer_saes[0]["dictionary_name"]
            )

    # First, select the layer
    selected_layer = st.selectbox(
        "Select Layer",
        options=unique_layers,
        index=(
            unique_layers.index(st.session_state[sae_session_keys["selected_layer"]])
            if st.session_state[sae_session_keys["selected_layer"]] in unique_layers
            else 0
        ),
        help="Choose the layer for which to analyze SAEs",
        key=sae_session_keys["selected_layer"],
    )

    # Retrieve SAEs specifically for the selected layer
    saes_for_selected_layer = saes_by_layer[selected_layer]

    # Extract dictionary names (models) available for the selected layer
    dictionary_names_for_layer = [
        sae["dictionary_name"] for sae in saes_for_selected_layer
    ]

    if not dictionary_names_for_layer:
        st.warning(f"No trained SAE models found for layer {selected_layer}.")
        return

    # Update dictionary selection if layer changed
    if (
        st.session_state[sae_session_keys["selected_dictionary"]]
        not in dictionary_names_for_layer
    ):
        st.session_state[sae_session_keys["selected_dictionary"]] = (
            dictionary_names_for_layer[0]
        )

    # Second, select the specific SAE model (dictionary name) within that layer
    selected_dictionary_name = st.selectbox(
        "Select Trained SAE Model",
        options=dictionary_names_for_layer,
        index=(
            dictionary_names_for_layer.index(
                st.session_state[sae_session_keys["selected_dictionary"]]
            )
            if st.session_state[sae_session_keys["selected_dictionary"]]
            in dictionary_names_for_layer
            else 0
        ),
        help="Choose which trained SAE model to analyze for the selected layer",
        key=sae_session_keys["selected_dictionary"],
    )

    # Find the complete sae_info dictionary based on both selections
    for sae_info in saes_for_selected_layer:
        if sae_info["dictionary_name"] == selected_dictionary_name:
            selected_sae_info = sae_info
            break

    # Assert that a valid SAE info was successfully retrieved.
    # If this assertion fails, it indicates an internal logic error where the selected
    # layer and dictionary name did not map to an existing SAE info object.
    assert (
        selected_sae_info is not None
    ), "Failed to retrieve selected SAE information. This indicates an internal logic error."

    # Display SAE information and wandb link if available
    training_metrics_path = selected_sae_info["path"] / "training_metrics.json"
    if training_metrics_path.exists():
        try:
            with open(training_metrics_path, "r") as f:
                training_metrics = json.load(f)

            # Display core SAE information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                k = training_metrics.get("k")
                st.metric("Top-K", k)
            with col2:
                dict_size = training_metrics.get("dictionary_size")
                st.metric("Dictionary Size", dict_size)
            with col3:
                activation_dim = training_metrics.get("activation_dim")
                expansion_factor = dict_size / activation_dim
                st.metric("Expansion Factor", expansion_factor)
            with col4:
                last_eval_logs = training_metrics.get("last_eval_logs", {})
                fve = last_eval_logs.get("val/frac_variance_explained", "Not available")
                st.metric("FVE", fve)

            # Display links in two columns
            col1, col2 = st.columns(2)

            with col1:
                wandb_link = training_metrics.get("wandb_link")
                if wandb_link:
                    st.markdown(f"**W&B Run:** [View training run]({wandb_link})")
                else:
                    st.info("No W&B link available")

            with col2:
                huggingface_link = training_metrics.get("hf_repo_id")
                if huggingface_link:
                    col21, col22 = st.columns([0.2, 0.8])
                    with col21:
                        st.markdown(
                            f"**HF Model:** [View model](https://huggingface.co/{huggingface_link})"
                        )
                    with col22:
                        st.code(huggingface_link, language=None)
                else:
                    st.info("No HF link available")
        except Exception as e:
            st.warning(f"Could not load training metrics: {str(e)}")
    else:
        st.info("Training metrics not found")

    multi_tab_interface(
        [
            (
                "📈 Latent Statistics",
                lambda: _render_latent_statistics_tab(method, selected_sae_info),
            ),
            (
                "📋 Steering Results",
                lambda: _render_steering_results_tab(method, selected_sae_info),
            ),
            (
                "🔥 Online Inference",
                lambda: SAEDifferenceOnlineDashboard(
                    method, selected_sae_info
                ).display(),
            ),
            (
                "🎯 Online Steering",
                lambda: SAESteeringDashboard(method, selected_sae_info).display(),
            ),
            (
                "🔍 Latent Lens",
                lambda: _render_latent_lens_tab(method, selected_sae_info),
            ),
            ("🎨 Plots", lambda: _render_plots_tab(method, selected_sae_info)),
            (
                "📊 MaxAct Examples",
                lambda: _render_maxact_tab(method, selected_sae_info),
            ),
        ],
        "SAE Difference Analysis",
    )


def _get_available_sae_directories(results_dir: Path):
    """Get list of available trained SAE directories."""
    sae_base_dir = results_dir / "sae_difference"
    if not sae_base_dir.exists():
        return []

    available_saes = []
    # Scan through layer directories
    for layer_dir in sae_base_dir.iterdir():
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue

        # Extract layer number
        try:
            layer_num = int(layer_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Scan through SAE directories in this layer
        for sae_dir in layer_dir.iterdir():
            if not sae_dir.is_dir():
                continue

            # Check if this looks like a valid SAE directory
            # (has dictionary_model subdirectory or training_config.yaml)
            if (sae_dir / "dictionary_model").exists() or (
                sae_dir / "training_config.yaml"
            ).exists():
                available_saes.append(
                    {
                        "layer": layer_num,
                        "dictionary_name": sae_dir.name,
                        "path": sae_dir,
                        "layer_dir": layer_dir,
                    }
                )

    # Sort by layer number, then by dictionary name
    available_saes.sort(key=lambda x: (x["layer"], x["dictionary_name"]))
    return available_saes


def _render_maxact_tab(method, selected_sae_info):
    """Render the MaxAct tab using MaxActivationDashboardComponent."""

    # Use the globally selected SAE
    dictionary_name = selected_sae_info["dictionary_name"]
    layer = selected_sae_info["layer"]
    model_results_dir = selected_sae_info["path"]

    st.markdown(f"**Selected SAE:** Layer {layer} - {dictionary_name}")

    if not model_results_dir.exists():
        st.error(f"SAE directory not found at {model_results_dir}")
        return

    # Look for MaxActStore database files in latent_activations directory
    latent_activations_dir = model_results_dir / "latent_activations"
    if not latent_activations_dir.exists():
        st.error(f"No latent activations found at {latent_activations_dir}")
        return

    # Find example database file
    example_db_path = latent_activations_dir / "examples.db"
    if not example_db_path.exists():
        st.error(f"No example database found at {example_db_path}")
        return

    # Assumption: tokenizer is available through method.tokenizer
    assert (
        method.tokenizer is not None
    ), "Tokenizer must be available for MaxActStore visualization"

    # Create MaxActStore instance
    max_store = ReadOnlyMaxActStore(
        example_db_path,
        tokenizer=method.tokenizer,
    )

    # Create and display the dashboard component
    component = MaxActivationDashboardComponent(
        max_store, title=f"SAE Difference Examples - Layer {layer}"
    )
    component.display()


def _load_latent_df(method, dictionary_name):
    """Load the latent DataFrame for a given dictionary name."""
    if not hasattr(method, "_latent_dfs"):
        method._latent_dfs = {}
    if dictionary_name not in method._latent_dfs:
        method._latent_dfs[dictionary_name] = load_latent_df(dictionary_name)
    return method._latent_dfs[dictionary_name]


def _render_latent_statistics_tab(method, selected_sae_info):
    """Render the Latent Statistics tab with interactive DataFrame filtering."""

    # Use the globally selected SAE
    dictionary_name = selected_sae_info["dictionary_name"]
    layer = selected_sae_info["layer"]

    st.markdown(f"**Selected SAE:** Layer {layer} - {dictionary_name}")

    try:
        # Load the latent DataFrame
        df = _load_latent_df(method, dictionary_name)
    except Exception as e:
        st.error(f"Failed to load latent DataFrame for {dictionary_name}: {str(e)}")
        return

    st.markdown(f"### Latent Statistics - Layer {layer}")
    st.markdown(f"**Dictionary:** {dictionary_name}")
    st.markdown(f"**Total latents:** {len(df)}")

    # Column information
    with st.expander("Column Descriptions", expanded=False):
        st.markdown(
            """
        - **tag**: Latent classification (shared, ft_only, base_only, etc.)
        - **dec_norm_diff**: Decoder norm difference between models
        - **max_act**: Maximum activation value
        - **freq**: Activation frequency
        - Other columns may include beta values, error metrics, etc.
        """
        )

    # Create filtering interface
    st.markdown("### Filters")

    # Initialize filtered dataframe
    filtered_df = df.copy()

    # Filter by categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        st.markdown("**Categorical Filters:**")
        cols = st.columns(min(3, len(categorical_cols)))
        for i, col in enumerate(categorical_cols):
            with cols[i % 3]:
                unique_values = df[col].unique().tolist()
                # Remove NaN values for display
                unique_values = [v for v in unique_values if pd.notna(v)]
                selected_values = st.multiselect(
                    f"{col}",
                    options=unique_values,
                    default=unique_values,
                    key=f"filter_{col}",
                )
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    # Filter by numeric columns in a collapsible section
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if numeric_cols:
        with st.expander("**Numeric Filters**", expanded=False):
            st.markdown(
                "Apply numeric range filters (leave unchanged to include all values)"
            )
            cols = st.columns(min(3, len(numeric_cols)))
            for i, col in enumerate(numeric_cols):
                with cols[i % 3]:
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    if col_min != col_max:  # Only show slider if there's variation
                        # Add checkbox to enable/disable filtering for this column
                        enable_filter = st.checkbox(
                            f"Filter {col}", value=False, key=f"enable_filter_{col}"
                        )

                        if enable_filter:
                            selected_range = st.slider(
                                f"{col} range",
                                min_value=col_min,
                                max_value=col_max,
                                value=(col_min, col_max),
                                key=f"filter_numeric_{col}",
                            )
                            # Only apply filter if range is not the full range
                            if selected_range != (col_min, col_max):
                                filtered_df = filtered_df[
                                    (filtered_df[col] >= selected_range[0])
                                    & (filtered_df[col] <= selected_range[1])
                                ]

    # Display filtering results
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} latents**")

    # Latent index filtering
    st.markdown("**Latent Index Filter:**")
    latent_indices_input = st.text_input(
        "Enter latent indices (comma-separated)",
        help="Enter specific latent indices to filter for, e.g., '0, 15, 42, 100'",
        key="latent_indices_filter",
    )

    if latent_indices_input.strip():
        try:
            # Parse comma-separated indices
            indices = [
                int(idx.strip())
                for idx in latent_indices_input.split(",")
                if idx.strip()
            ]
            if indices:
                # Filter to only show specified latent indices
                filtered_df = filtered_df.iloc[indices]
                st.info(f"Showing {len(indices)} specified latent indices")
        except ValueError:
            st.error(
                "Invalid latent indices format. Please enter comma-separated integers."
            )

    # Display the filtered and sorted dataframe
    st.markdown("### Results")
    st.dataframe(filtered_df, use_container_width=True, height=400)

    # Download option
    if st.button("Download Filtered Results as CSV"):
        csv = filtered_df.to_csv(index=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"filtered_latent_stats_{dictionary_name}_layer_{layer}.csv",
            mime="text/csv",
        )

    # Summary statistics for filtered data
    if len(filtered_df) > 0:
        st.markdown("### Summary Statistics")
        summary_stats = filtered_df.describe()
        st.dataframe(summary_stats, use_container_width=True)


def _render_plots_tab(method, selected_sae_info):
    """Render the Plots tab displaying all generated plots."""

    selected_layer = selected_sae_info["layer"]

    # Construct the dictionary name for this layer
    dictionary_name = selected_sae_info["dictionary_name"]

    # Find the plots directory for this layer
    model_results_dir = (
        method.results_dir
        / "sae_difference"
        / f"layer_{selected_layer}"
        / dictionary_name
    )
    plots_dir = model_results_dir / "plots"

    if not plots_dir.exists():
        st.error(f"No plots directory found at {plots_dir}")
        return

    # Find all image files
    image_extensions = [".png", ".jpg", ".jpeg", ".svg", ".pdf"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(plots_dir.glob(f"*{ext}"))

    if not image_files:
        st.error(f"No image files found in {plots_dir}")
        return

    st.markdown(f"### Plots - Layer {selected_layer}")
    st.markdown(f"**Dictionary:** {dictionary_name}")
    st.markdown(f"**Found {len(image_files)} plot files**")

    # Organize plots by categories if naming patterns exist
    plot_categories = {}
    for image_file in image_files:
        # Simple categorization based on filename prefixes
        filename = image_file.stem.lower()
        if "beta" in filename or "scaler" in filename:
            category = "Beta Analysis"
        elif "histogram" in filename or "distribution" in filename:
            category = "Distributions"
        elif "scatter" in filename or "correlation" in filename:
            category = "Correlations"
        else:
            category = "Other"

        if category not in plot_categories:
            plot_categories[category] = []
        plot_categories[category].append(image_file)

    # Display plots by category
    for category, files in plot_categories.items():
        with st.expander(f"{category} ({len(files)} plots)", expanded=True):
            # Display plots in a grid layout
            cols_per_row = 2
            for i in range(0, len(files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, image_file in enumerate(files[i : i + cols_per_row]):
                    with cols[j]:
                        st.markdown(f"**{image_file.name}**")

                        # Display image based on format
                        if image_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                            try:
                                st.image(str(image_file), use_container_width=True)
                            except Exception as e:
                                st.error(
                                    f"Error loading image {image_file.name}: {str(e)}"
                                )
                        elif image_file.suffix.lower() == ".svg":
                            try:
                                with open(image_file, "r") as f:
                                    svg_content = f.read()
                                st.markdown(svg_content, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(
                                    f"Error loading SVG {image_file.name}: {str(e)}"
                                )
                        elif image_file.suffix.lower() == ".pdf":
                            try:
                                # Display PDF inline using base64 encoding
                                with open(image_file, "rb") as f:
                                    pdf_data = f.read()

                                # Encode PDF as base64 for embedding
                                pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

                                # Create PDF viewer with download option
                                pdf_display = f"""
                                <iframe src="data:application/pdf;base64,{pdf_base64}"
                                        width="100%" height="400" type="application/pdf">
                                    <p>PDF cannot be displayed.
                                       <a href="data:application/pdf;base64,{pdf_base64}" download="{image_file.name}">
                                           Download {image_file.name}
                                       </a>
                                    </p>
                                </iframe>
                                """
                                st.markdown(pdf_display, unsafe_allow_html=True)

                                # Also provide download button
                                st.download_button(
                                    label=f"Download {image_file.name}",
                                    data=pdf_data,
                                    file_name=image_file.name,
                                    mime="application/pdf",
                                )
                            except Exception as e:
                                st.error(
                                    f"Error loading PDF {image_file.name}: {str(e)}"
                                )
                                # Fallback to download only
                                with open(image_file, "rb") as f:
                                    st.download_button(
                                        label=f"Download {image_file.name}",
                                        data=f.read(),
                                        file_name=image_file.name,
                                        mime="application/pdf",
                                    )
                        else:
                            # For other formats, provide download link
                            st.markdown(f"📄 {image_file.name}")
                            with open(image_file, "rb") as f:
                                st.download_button(
                                    label=f"Download {image_file.name}",
                                    data=f.read(),
                                    file_name=image_file.name,
                                    mime="application/octet-stream",
                                )


def _render_steering_results_tab(method, selected_sae_info):
    """Render the Steering Results tab displaying saved experiment results."""

    dictionary_name = selected_sae_info["dictionary_name"]
    layer = selected_sae_info["layer"]
    model_results_dir = selected_sae_info["path"]

    st.markdown(f"**Selected SAE:** Layer {layer} - {dictionary_name}")

    # Display the steering results using the imported function
    display_steering_results(model_results_dir, method.cfg)


def _render_latent_lens_tab(method, selected_sae_info):
    """Render logit lens analysis tab for SAE latents."""

    dictionary_name = selected_sae_info["dictionary_name"]
    layer = selected_sae_info["layer"]

    # Load SAE model
    try:
        sae_model = load_dictionary_model(dictionary_name, is_sae=True)
        sae_model = sae_model.to(method.device)
    except Exception as e:
        st.error(f"Failed to load SAE model: {str(e)}")
        return

    render_latent_lens_tab(
        method,
        lambda idx: sae_model.decoder.weight[:, idx],
        sae_model.dict_size,
        layer,
        patch_scope_add_scaler=True,
    )


class SAESteeringDashboard(SteeringDashboard):
    """
    SAE-specific steering dashboard with cached SAE model.
    """

    def __init__(self, method_instance, sae_info):
        super().__init__(method_instance)
        self.sae_info = sae_info
        self._layer = sae_info["layer"]
        self._sae_model = None  # Cache the SAE model
        try:
            latent_df = _load_latent_df(
                method_instance, self.sae_info["dictionary_name"]
            )
            if "max_act_validation" in latent_df.columns:
                self._max_acts = latent_df["max_act_validation"]
            elif "max_act_train" in latent_df.columns:
                self._max_acts = latent_df["max_act_train"]
            else:
                raise KeyError(
                    f"Neither 'max_act_validation' nor 'max_act_train' found in latent dataframe for {self.sae_info['dictionary_name']}"
                )
        except Exception as e:
            st.error(
                f"❌ Maximum activations not yet collected for dictionary '{self.sae_info['dictionary_name']}'"
            )
            st.info(
                "💡 Please run the analysis pipeline to collect maximum activations before using the steering dashboard."
            )
            st.stop()

    def __hash__(self):
        return hash((self._layer, self.sae_info["dictionary_name"]))

    @property
    def layer(self) -> int:
        """Get the layer number for this steering dashboard."""
        return self._layer

    def get_latent(self, idx: int) -> torch.Tensor:
        """
        Get decoder vector for specified latent index from the cached SAE.

        Args:
            idx: Latent index

        Returns:
            Decoder vector [hidden_dim] for the specified latent
        """
        # Load SAE model if not cached
        if self._sae_model is None:
            dictionary_name = self.sae_info["dictionary_name"]

            try:
                self._sae_model = load_dictionary_model(dictionary_name, is_sae=True)
                self._sae_model = self._sae_model.to(self.method.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SAE model {dictionary_name}: {str(e)}"
                )

        # Extract decoder vector for the specified latent
        # SAE decoder is nn.Linear(dict_size, activation_dim, bias=False)
        # decoder.weight shape: [activation_dim, dict_size]
        # We want the decoder vector for latent idx: decoder.weight[:, idx]

        dict_size = self._sae_model.dict_size
        assert 0 <= idx < dict_size, f"Latent index {idx} out of range [0, {dict_size})"

        decoder_vector = self._sae_model.decoder.weight[:, idx]  # [activation_dim]

        return decoder_vector.detach()

    def get_dict_size(self) -> int:
        """Get the dictionary size for validation."""
        # Load SAE model if not cached
        if self._sae_model is None:
            dictionary_name = self.sae_info["dictionary_name"]

            try:
                self._sae_model = load_dictionary_model(dictionary_name, is_sae=True)
                self._sae_model = self._sae_model.to(self.method.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SAE model {dictionary_name}: {str(e)}"
                )

        return self._sae_model.dict_size

    def _get_title(self) -> str:
        """Get title for SAE steering analysis."""
        return f"SAE Latent Steering - Layer {self.layer}"

    def get_max_activation(self, latent_idx: int) -> float:
        """
        Get the maximum activation value for a specific latent from latent_df.

        Args:
            latent_idx: Latent index

        Returns:
            Maximum activation value for the latent
        """

        if latent_idx in self._max_acts.index:
            return float(self._max_acts.loc[latent_idx])
        else:
            return "unknown"

    @st.fragment
    def _render_latent_selector(self) -> int:
        """Render latent selection UI fragment with session state."""

        # Get dictionary size for validation
        dict_size = self.get_dict_size()

        # Create unique session state key for this steering dashboard
        session_key = f"sae_steering_latent_idx_layer_{self.layer}"

        # Initialize session state if not exists
        if session_key not in st.session_state:
            st.session_state[session_key] = 0

        # Ensure the session state value is within valid range
        if st.session_state[session_key] >= dict_size:
            st.session_state[session_key] = 0

        latent_idx = st.number_input(
            "Latent Index",
            min_value=0,
            max_value=dict_size - 1,
            value=st.session_state[session_key],
            step=1,
            help=f"Choose which latent to steer (0-{dict_size - 1})",
            key=session_key,
        )

        # Display max activation for the selected latent
        max_act = self.get_max_activation(latent_idx)
        st.info(f"**Max Activation:** {max_act}")

        return latent_idx

    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render SAE steering-specific controls in Streamlit."""

        col1, col2 = st.columns(2)

        with col1:
            latent_idx = self._render_latent_selector()

        with col2:
            steering_factor = st.slider(
                "Steering Factor",
                min_value=-1000.0,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                help="Strength and direction of steering (negative values reverse the effect)",
            )

        steering_mode = st.selectbox(
            "Steering Mode",
            options=["prompt_only", "all_tokens"],
            index=1,  # Default to all_tokens
            help="Apply steering only to prompt tokens or to all tokens (prompt + generated)",
        )

        return {
            "latent_idx": latent_idx,
            "steering_factor": steering_factor,
            "steering_mode": steering_mode,
        }


class SAEDifferenceOnlineDashboard(AbstractOnlineDiffingDashboard):
    """
    Online dashboard for interactive SAE difference analysis.

    This dashboard allows users to input text and see per-token SAE latent activations
    highlighted directly in the text, similar to KL/NormDiff dashboards but for SAE analysis.
    """

    def __init__(self, method_instance, sae_info):
        super().__init__(method_instance)
        self.sae_info = sae_info

    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render SAE-specific controls in Streamlit."""

        selected_latent = st.number_input(
            "Latent Index",
            min_value=0,
            value=0,
            step=1,
            help=f"Choose which latent to visualize",
        )

        return {"latent_idx": selected_latent}

    def compute_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        """Compute SAE activation statistics for selected latent."""
        layer = self.sae_info["layer"]  # Use layer from sae_info
        latent_idx = kwargs.get("latent_idx", 0)

        # Get full SAE activations from the parent method
        results = self.method.compute_sae_activations_for_tokens(
            self.sae_info["dictionary_name"], input_ids, attention_mask, layer
        )

        # Use activations for specific latent
        latent_activations = results["latent_activations"]  # [seq_len, dict_size]

        # Shape assertion
        seq_len, dict_size = latent_activations.shape
        assert (
            0 <= latent_idx < dict_size
        ), f"Latent index {latent_idx} out of range [0, {dict_size})"

        # Extract activations for the selected latent
        values = latent_activations[:, latent_idx]  # [seq_len]
        analysis_title = f"Latent {latent_idx} Activation"

        # Compute statistics for the selected values
        statistics = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

        # Return adapted results for the abstract dashboard
        return {
            "tokens": results["tokens"],
            "values": values,
            "statistics": statistics,
            "total_tokens": results["total_tokens"],
            "analysis_title": analysis_title,  # For display purposes
        }

    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get SAE-specific parameters."""
        return {"latent_idx": 0}

    def _get_title(self) -> str:
        """Get title for SAE analysis."""
        return "SAE Difference Analysis"
