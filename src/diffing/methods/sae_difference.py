"""
SAE on difference-based model diffing method.

This module trains SAEs on activation differences between base and finetuned models,
then runs a comprehensive analysis pipeline including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.

Key assumptions:
- Preprocessing pipeline has generated paired activation caches
- dictionary_learning library is available and compatible with SAE training
- science-of-finetuning repository is available for analysis pipeline
- W&B configuration is available in infrastructure config
- Sufficient GPU memory and disk space for training
"""

from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import json
from collections import defaultdict
import streamlit as st
from src.utils.dashboards import MaxActivationDashboardComponent
from src.utils.max_act_store import MaxActStore, ReadOnlyMaxActStore

from .diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices
from src.utils.dictionary.analysis import (
    build_push_sae_difference_latent_df,
    make_plots,
)
from src.utils.dictionary.training import (
    train_sae_difference_for_layer,
    sae_difference_run_name,
)
from src.utils.dictionary.latent_scaling.closed_form import compute_scalers_from_config
from src.utils.dictionary.latent_scaling.beta_analysis import (
    update_latent_df_with_beta_values,
)
from src.utils.dictionary.latent_activations import (
    collect_dictionary_activations_from_config,
    collect_activating_examples,
    update_latent_df_with_stats,
)
from src.utils.dictionary.steering import run_latent_steering_experiment, get_sae_latent
from src.utils.dictionary.utils import load_latent_df, load_dictionary_model
from src.utils.dashboards import AbstractOnlineDiffingDashboard, SteeringDashboard
from src.utils.dictionary.steering import display_steering_results
from src.utils.visualization import render_latent_lens_tab


class SAEDifferenceMethod(DiffingMethod):
    """
    Trains SAEs on activation differences and runs comprehensive analysis.

    This method:
    1. Loads paired activation caches from preprocessing pipeline
    2. Computes activation differences (finetuned - base or base - finetuned)
    3. Trains BatchTopK SAEs on normalized differences for specified layers
    4. Saves trained models with configuration and metrics
    5. Optionally uploads models to Hugging Face Hub
    6. Runs complete analysis pipeline from science-of-finetuning
    7. Returns comprehensive results including training metrics and analysis outcomes
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Get layers to process
        layers = self.method_cfg.layers
        if layers is None:
            layers = cfg.preprocessing.layers
        self.layers = get_layer_indices(self.base_model_cfg.model_id, layers)

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._latent_dfs = {}

    def __hash__(self):
        return hash(self.cfg)

    def __eq__(self, other):
        return self.cfg == other.cfg

    def run(self) -> Dict[str, Any]:
        """
        Main training orchestration with analysis pipeline.

        Trains SAEs on differences for each specified layer, then runs the complete
        analysis pipeline for each trained model.

        Returns:
            Dictionary containing training results, model paths, and analysis outcomes
        """
        logger.info(f"Starting SAE difference training for layers: {self.layers}")
        logger.info(f"Training target: {self.method_cfg.training.target}")

        for layer_idx in self.layers:
            logger.info(f"Processing layer {layer_idx}")

            dictionary_name = sae_difference_run_name(
                self.cfg, layer_idx, self.base_model_cfg, self.finetuned_model_cfg
            )
            model_results_dir = (
                self.results_dir
                / "sae_difference"
                / f"layer_{layer_idx}"
                / dictionary_name
            )
            model_results_dir.mkdir(parents=True, exist_ok=True)
            if (
                not (
                    model_results_dir / "dictionary_model" / "model.safetensors"
                ).exists()
                or self.method_cfg.training.overwrite
            ):
                # Train SAE on differences for this layer
                logger.info(f"Training SAE on differences for layer {layer_idx}")
                training_metrics, model_path = train_sae_difference_for_layer(
                    self.cfg, layer_idx, self.device, dictionary_name
                )
                # save model
                dictionary_model = load_dictionary_model(model_path)
                dictionary_model.save_pretrained(model_results_dir / "dictionary_model")
                # save training metrics
                with open(model_results_dir / "training_metrics.json", "w") as f:
                    json.dump(training_metrics, f)

                # save training configs
                OmegaConf.save(self.cfg, model_results_dir / "training_config.yaml")
            else:
                logger.info(
                    f"Found trained model at {model_results_dir / 'dictionary_model'}"
                )
                training_metrics = json.load(
                    open(model_results_dir / "training_metrics.json")
                )

            if self.method_cfg.analysis.enabled:
                logger.info(f"Storing analysis results in {model_results_dir}")
                build_push_sae_difference_latent_df(
                    dictionary_name=dictionary_name,
                    target=self.method_cfg.training.target,
                )

                if self.method_cfg.analysis.latent_scaling.enabled:
                    logger.info(f"Computing latent scaling for layer {layer_idx}")
                    compute_scalers_from_config(
                        cfg=self.cfg,
                        layer=layer_idx,
                        dictionary_model=dictionary_name,
                        results_dir=model_results_dir,
                    )
                    update_latent_df_with_beta_values(
                        dictionary_name,
                        model_results_dir,
                        num_samples=self.method_cfg.analysis.latent_scaling.num_samples,
                    )

                if self.method_cfg.analysis.latent_activations.enabled:
                    logger.info(f"Collecting latent activations for layer {layer_idx}")
                    latent_activations_cache = (
                        collect_dictionary_activations_from_config(
                            cfg=self.cfg,
                            layer=layer_idx,
                            dictionary_model_name=dictionary_name,
                            result_dir=model_results_dir,
                        )
                    )
                    collect_activating_examples(
                        dictionary_model_name=dictionary_name,
                        latent_activation_cache=latent_activations_cache,
                        n=self.method_cfg.analysis.latent_activations.n_max_activations,
                        upload_to_hub=self.method_cfg.analysis.latent_activations.upload_to_hub,
                        overwrite=self.method_cfg.analysis.latent_activations.overwrite,
                        save_path=model_results_dir,
                    )
                    update_latent_df_with_stats(
                        dictionary_name=dictionary_name,
                        latent_activation_cache=latent_activations_cache,
                        split_of_cache=self.method_cfg.analysis.latent_activations.split,
                        device=self.method_cfg.analysis.latent_activations.cache_device,
                        save_path=model_results_dir,
                    )

                try:
                    make_plots(
                        dictionary_name=dictionary_name,
                        plots_dir=model_results_dir / "plots",
                    )
                except Exception as e:
                    logger.error(f"Error making plots for {dictionary_name}: {e}")

                if self.method_cfg.analysis.latent_steering.enabled:
                    logger.info(
                        f"Running latent steering experiment for layer {layer_idx}"
                    )
                    run_latent_steering_experiment(
                        method=self,
                        get_latent_fn=get_sae_latent,
                        dictionary_name=dictionary_name,
                        results_dir=model_results_dir,
                        layer=layer_idx,
                    )
            logger.info(f"Successfully completed layer {layer_idx}")

        return {"status": "completed", "layers_processed": self.layers}

    def visualize(self) -> None:
        """
        Create Streamlit visualization for SAE difference results with 4 tabs.

        Features:
        - MaxAct tab: Display maximum activating examples using MaxActivationDashboardComponent
        - Online Inference tab: Real-time SAE analysis similar to KL/NormDiff dashboards
        - Latent Statistics tab: Interactive exploration of latent DataFrame with filtering
        - Plots tab: Display all generated plots from the analysis pipeline
        """
        import streamlit as st
        from src.utils.visualization import multi_tab_interface

        st.subheader("SAE Difference Analysis")

        # Initialize session state for tab selection
        tab_session_key = "sae_difference_active_tab"
        if tab_session_key not in st.session_state:
            st.session_state[tab_session_key] = 0  # Default to first tab

        # Global SAE selector
        available_saes = self._get_available_sae_directories()
        if not available_saes:
            st.error(
                f"No trained SAE directories found in {self.results_dir / 'sae_difference'}"
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
                unique_layers.index(
                    st.session_state[sae_session_keys["selected_layer"]]
                )
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
                    fve = last_eval_logs.get(
                        "val/frac_variance_explained", "Not available"
                    )
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
                    lambda: self._render_latent_statistics_tab(selected_sae_info),
                ),
                (
                    "📋 Steering Results",
                    lambda: self._render_steering_results_tab(selected_sae_info),
                ),
                (
                    "🔥 Online Inference",
                    lambda: SAEDifferenceOnlineDashboard(
                        self, selected_sae_info
                    ).display(),
                ),
                (
                    "🎯 Online Steering",
                    lambda: SAESteeringDashboard(self, selected_sae_info).display(),
                ),
                (
                    "🔍 Latent Lens",
                    lambda: self._render_latent_lens_tab(selected_sae_info),
                ),
                ("🎨 Plots", lambda: self._render_plots_tab(selected_sae_info)),
                (
                    "📊 MaxAct Examples",
                    lambda: self._render_maxact_tab(selected_sae_info),
                ),
            ],
            "SAE Difference Analysis",
        )

    def _get_available_sae_directories(self):
        """Get list of available trained SAE directories."""
        sae_base_dir = self.results_dir / "sae_difference"
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

    def _render_maxact_tab(self, selected_sae_info):
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

        # Assumption: tokenizer is available through self.tokenizer
        assert (
            self.tokenizer is not None
        ), "Tokenizer must be available for MaxActStore visualization"

        # Create MaxActStore instance
        max_store = ReadOnlyMaxActStore(
            example_db_path,
            tokenizer=self.tokenizer,
        )

        # Create and display the dashboard component
        component = MaxActivationDashboardComponent(
            max_store, title=f"SAE Difference Examples - Layer {layer}"
        )
        component.display()

    def _load_latent_df(self, dictionary_name):
        """Load the latent DataFrame for a given dictionary name."""
        if dictionary_name not in self._latent_dfs:
            self._latent_dfs[dictionary_name] = load_latent_df(dictionary_name)
        return self._latent_dfs[dictionary_name]

    def _render_latent_statistics_tab(self, selected_sae_info):
        """Render the Latent Statistics tab with interactive DataFrame filtering."""
        import streamlit as st
        import pandas as pd

        # Use the globally selected SAE
        dictionary_name = selected_sae_info["dictionary_name"]
        layer = selected_sae_info["layer"]

        st.markdown(f"**Selected SAE:** Layer {layer} - {dictionary_name}")

        try:
            # Load the latent DataFrame
            df = self._load_latent_df(dictionary_name)
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
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
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
                        filtered_df = filtered_df[
                            filtered_df[col].isin(selected_values)
                        ]

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

    def _render_plots_tab(self, selected_sae_info):
        """Render the Plots tab displaying all generated plots."""
        import streamlit as st
        from pathlib import Path
        import base64

        selected_layer = selected_sae_info["layer"]

        # Construct the dictionary name for this layer
        dictionary_name = selected_sae_info["dictionary_name"]

        # Find the plots directory for this layer
        model_results_dir = (
            self.results_dir
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
                                    pdf_base64 = base64.b64encode(pdf_data).decode(
                                        "utf-8"
                                    )

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

    def _render_steering_results_tab(self, selected_sae_info):
        """Render the Steering Results tab displaying saved experiment results."""

        dictionary_name = selected_sae_info["dictionary_name"]
        layer = selected_sae_info["layer"]
        model_results_dir = selected_sae_info["path"]

        st.markdown(f"**Selected SAE:** Layer {layer} - {dictionary_name}")

        # Display the steering results using the imported function
        display_steering_results(model_results_dir, self.cfg)

    def _render_latent_lens_tab(self, selected_sae_info):
        """Render logit lens analysis tab for SAE latents."""

        dictionary_name = selected_sae_info["dictionary_name"]
        layer = selected_sae_info["layer"]

        # Load SAE model
        try:
            from src.utils.dictionary.utils import load_dictionary_model

            sae_model = load_dictionary_model(dictionary_name, is_sae=True)
            sae_model = sae_model.to(self.device)
        except Exception as e:
            st.error(f"Failed to load SAE model: {str(e)}")
            return

        render_latent_lens_tab(
            self,
            lambda idx: sae_model.decoder.weight[:, idx],
            sae_model.dict_size,
            layer,
            patch_scope_add_scaler=True,
        )

    @torch.no_grad()
    def compute_sae_activations_for_tokens(
        self,
        dictionary_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
    ) -> Dict[str, Any]:
        """
        Compute SAE latent activations for given tokens (used by online dashboard).

        This method:
        1. Extracts activations from both base and finetuned models using nnsight
        2. Computes activation differences based on training target
        3. Passes differences through the trained SAE to get latent activations
        4. Returns tokens, latent activations, and statistics

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            layer: Layer index to analyze

        Returns:
            Dictionary with tokens, latent_activations, and statistics
        """
        from src.utils.dictionary.utils import load_dictionary_model

        # Shape assertions
        assert (
            input_ids.ndim == 2
        ), f"Expected 2D input_ids, got shape {input_ids.shape}"
        assert (
            attention_mask.ndim == 2
        ), f"Expected 2D attention_mask, got shape {attention_mask.shape}"
        assert (
            input_ids.shape == attention_mask.shape
        ), f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}"

        # Get tokens for display
        token_ids = input_ids[0].cpu().numpy()  # Take first sequence
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]

        # Extract activations from both models using nnsight
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            # Get base model activations
            with self.base_model.trace(inputs):
                base_activations = self.base_model.layers_output[layer].save()

            # Get finetuned model activations
            with self.finetuned_model.trace(inputs):
                finetuned_activations = self.finetuned_model.layers_output[layer].save()

        # Extract the values and move to CPU
        base_acts = base_activations.cpu()  # [batch_size, seq_len, hidden_dim]
        finetuned_acts = (
            finetuned_activations.cpu()
        )  # [batch_size, seq_len, hidden_dim]

        # Shape assertions
        batch_size, seq_len, hidden_dim = base_acts.shape
        assert finetuned_acts.shape == (
            batch_size,
            seq_len,
            hidden_dim,
        ), f"Shape mismatch: base {base_acts.shape} vs finetuned {finetuned_acts.shape}"

        # Compute activation differences based on training target
        # Assumption: training target determines direction of difference computation
        if self.method_cfg.training.target == "difference_bft":  # base - finetuned
            activation_diffs = base_acts - finetuned_acts
        else:  # difference_ftb: finetuned - base (default)
            activation_diffs = finetuned_acts - base_acts

        # Shape assertion for differences
        assert activation_diffs.shape == (
            batch_size,
            seq_len,
            hidden_dim,
        ), f"Expected diff shape {(batch_size, seq_len, hidden_dim)}, got {activation_diffs.shape}"

        # Load the trained SAE model for this layer
        try:
            sae_model = load_dictionary_model(dictionary_name, is_sae=True)
            sae_model = sae_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAE model {dictionary_name}: {str(e)}")

        # Pass differences through SAE to get latent activations
        # Take first sequence for analysis
        diff_sequence = activation_diffs[0].to(self.device)  # [seq_len, hidden_dim]

        # Encode differences to get latent activations
        latent_activations = sae_model.encode(diff_sequence)  # [seq_len, dict_size]

        # Shape assertion for latent activations
        dict_size = sae_model.dict_size
        assert latent_activations.shape == (
            seq_len,
            dict_size,
        ), f"Expected latent shape {(seq_len, dict_size)}, got {latent_activations.shape}"

        # Convert to numpy for visualization and statistics
        latent_activations_np = latent_activations.cpu().detach().numpy()

        # Compute per-token maximum latent activation for visualization
        max_activations_per_token = np.max(latent_activations_np, axis=1)  # [seq_len]

        # Compute statistics
        statistics = {
            "mean": float(np.mean(max_activations_per_token)),
            "std": float(np.std(max_activations_per_token)),
            "min": float(np.min(max_activations_per_token)),
            "max": float(np.max(max_activations_per_token)),
            "median": float(np.median(max_activations_per_token)),
        }

        return {
            "tokens": tokens,
            "latent_activations": latent_activations_np,
            "max_activations_per_token": max_activations_per_token,
            "statistics": statistics,
            "total_tokens": len(tokens),
            "layer": layer,
            "dict_size": dict_size,
        }

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available SAE difference results.

        Args:
            results_dir: Base results directory

        Returns:
            Dict mapping {model_pair: {organism: {layer: path_to_results}}}
        """
        results = defaultdict(dict)
        results_base = results_dir

        if not results_base.exists():
            return results

        # Scan for KL results in the expected structure
        for base_model_dir in results_base.iterdir():
            if not base_model_dir.is_dir():
                continue

            model_name = base_model_dir.name

            for organism_dir in base_model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                sae_dir = organism_dir / "sae_difference"
                if sae_dir.exists() and any(sae_dir.iterdir()):
                    results[model_name][organism_name] = str(sae_dir)

        return results


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
            latent_df = self.method._load_latent_df(self.sae_info["dictionary_name"])
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
            from src.utils.dictionary.utils import load_dictionary_model

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
            from src.utils.dictionary.utils import load_dictionary_model

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
        import streamlit as st

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
        import streamlit as st

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
        import streamlit as st

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
