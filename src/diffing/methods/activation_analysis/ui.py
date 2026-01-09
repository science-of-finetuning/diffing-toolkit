"""
UI utilities for activation analysis.
"""

from typing import Tuple, Optional, Dict, Any
import streamlit as st
import torch
import matplotlib.pyplot as plt
from diffing.utils.max_act_store import ReadOnlyMaxActStore
from diffing.utils.dashboards import MaxActivationDashboardComponent
from diffing.utils.visualization import multi_tab_interface, render_latent_lens_tab


from .steering_dashboard import ActivationAnalysisSteeringDashboard
from .online_dashboard import ActivationAnalysisOnlineDashboard
from .utils import (
    create_metric_selection_ui,
    get_maxact_database_name,
    get_metric_display_name,
)


def _render_plots_tab(method):
    """Render the Plots tab displaying all generated plots."""
    selected_layer = st.selectbox(
        "Select Layer",
        method._find_available_layers(),
        key="layer_selector_plots_normdiff",
    )

    # Find all dataset directories
    dataset_dirs = [
        d
        for d in (method.results_dir / f"layer_{selected_layer}").iterdir()
        if d.is_dir()
    ]
    if not dataset_dirs:
        st.error(f"No datasets found in {method.results_dir}")
        return

    st.markdown(f"### Plots - Layer {selected_layer}")

    # Display plots for each dataset
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        # Find the plots directory for this dataset and layer
        plots_dir = dataset_dir / "plots"

        if not plots_dir.exists():
            continue  # Skip datasets without plots for this layer

        # Find all image files
        image_extensions = [".png", ".jpg", ".jpeg", ".svg", ".pdf"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(plots_dir.glob(f"*{ext}"))

        if not image_files:
            continue  # Skip if no images found

        # Separate plots into outlier categories
        with_outliers = []
        no_outliers = []

        for image_file in image_files:
            if "no_outliers" in image_file.name:
                no_outliers.append(image_file)
            else:
                with_outliers.append(image_file)

        # Create expander for this dataset
        with st.expander(f"{dataset_name} ({len(image_files)} plots)", expanded=True):

            # With Outliers section
            if with_outliers:
                with st.expander(
                    f"With Outliers (all) ({len(with_outliers)} plots)", expanded=False
                ):
                    # Display plots in a grid layout
                    cols_per_row = 2
                    for i in range(0, len(with_outliers), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, image_file in enumerate(
                            with_outliers[i : i + cols_per_row]
                        ):
                            if j < len(with_outliers[i : i + cols_per_row]):
                                with cols[j]:
                                    st.markdown(f"**{image_file.name}**")

                                    # Display image based on format
                                    if image_file.suffix.lower() in [
                                        ".png",
                                        ".jpg",
                                        ".jpeg",
                                    ]:
                                        try:
                                            st.image(
                                                str(image_file),
                                                use_container_width=True,
                                            )
                                        except Exception as e:
                                            st.error(
                                                f"Error loading image {image_file.name}: {str(e)}"
                                            )
                                    elif image_file.suffix.lower() == ".svg":
                                        try:
                                            with open(image_file, "r") as f:
                                                svg_content = f.read()
                                            st.markdown(
                                                svg_content, unsafe_allow_html=True
                                            )
                                        except Exception as e:
                                            st.error(
                                                f"Error loading SVG {image_file.name}: {str(e)}"
                                            )
                                    else:
                                        # For PDF and other formats, provide download link
                                        st.markdown(f"📄 {image_file.name}")
                                        with open(image_file, "rb") as f:
                                            st.download_button(
                                                label=f"Download {image_file.name}",
                                                data=f.read(),
                                                file_name=image_file.name,
                                                mime="application/octet-stream",
                                            )

            # No Outliers section
            if no_outliers:
                with st.expander(
                    f"No outliers ({len(no_outliers)} plots)", expanded=False
                ):
                    # Display plots in a grid layout
                    cols_per_row = 2
                    for i in range(0, len(no_outliers), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, image_file in enumerate(
                            no_outliers[i : i + cols_per_row]
                        ):
                            if j < len(no_outliers[i : i + cols_per_row]):
                                with cols[j]:
                                    st.markdown(f"**{image_file.name}**")

                                    # Display image based on format
                                    if image_file.suffix.lower() in [
                                        ".png",
                                        ".jpg",
                                        ".jpeg",
                                    ]:
                                        try:
                                            st.image(
                                                str(image_file),
                                                use_container_width=True,
                                            )
                                        except Exception as e:
                                            st.error(
                                                f"Error loading image {image_file.name}: {str(e)}"
                                            )
                                    elif image_file.suffix.lower() == ".svg":
                                        try:
                                            with open(image_file, "r") as f:
                                                svg_content = f.read()
                                            st.markdown(
                                                svg_content, unsafe_allow_html=True
                                            )
                                        except Exception as e:
                                            st.error(
                                                f"Error loading SVG {image_file.name}: {str(e)}"
                                            )
                                    else:
                                        # For PDF and other formats, provide download link
                                        st.markdown(f"📄 {image_file.name}")
                                        with open(image_file, "rb") as f:
                                            st.download_button(
                                                label=f"Download {image_file.name}",
                                                data=f.read(),
                                                file_name=image_file.name,
                                                mime="application/octet-stream",
                                            )


def _render_dataset_statistics(method):
    """Render the dataset statistics tab using MaxActivationDashboardComponent."""

    # Find available layers
    layer_dirs = list(method.results_dir.glob("layer_*"))
    if not layer_dirs:
        st.error(f"No layer directories found in {method.results_dir}")
        return

    # Extract layer numbers from directory names and check for database files
    available_layers = []
    for layer_dir in layer_dirs:
        if not layer_dir.is_dir():
            continue

        # Extract layer number from directory name like "layer_16"
        dirname = layer_dir.name
        layer_part = dirname[6:]  # Remove "layer_" prefix
        layer_num = int(layer_part)

        # Check if any database files exist in this layer directory
        db_files = list(layer_dir.glob("*.db"))
        if db_files:
            available_layers.append(layer_num)

    if not available_layers:
        st.error("No database files found in any layer directories")
        return

    selected_layer = st.selectbox(
        "Select Layer",
        available_layers,
        key="layer_selector_maxact_normdiff_dataset_statistics",
    )

    if selected_layer is None:
        return

    layer_dir = method.results_dir / f"layer_{selected_layer}"

    # Find available metric databases for this layer
    available_metrics = {}
    metric_types = ["norm_diff", "cos_dist", "norm_base", "norm_ft"]

    for metric_type in metric_types:
        if metric_type in ["norm_diff", "cos_dist"]:
            # Check for both mean and max versions
            for agg in ["mean", "max"]:
                db_name = get_maxact_database_name(metric_type, agg)
                db_path = layer_dir / f"{db_name}.db"
                if db_path.exists():
                    key = f"{metric_type}_{agg}"
                    available_metrics[key] = {
                        "metric_type": metric_type,
                        "aggregation": agg,
                        "path": db_path,
                        "display_name": get_metric_display_name(metric_type, agg),
                    }
        else:
            # Check for single version (no aggregation)
            db_name = get_maxact_database_name(metric_type)
            db_path = layer_dir / f"{db_name}.db"
            if db_path.exists():
                available_metrics[metric_type] = {
                    "metric_type": metric_type,
                    "aggregation": None,
                    "path": db_path,
                    "display_name": get_metric_display_name(metric_type),
                }

    if not available_metrics:
        st.error(f"No metric databases found for layer {selected_layer}")
        return

    # Create metric selection UI and render display in fragment
    metric_type, aggregation = create_metric_selection_ui("dataset_stats")

    # Find the matching metric configuration
    if metric_type in ["norm_diff", "cos_dist"] and aggregation:
        metric_key = f"{metric_type}_{aggregation}"
    else:
        metric_key = metric_type

    if metric_key not in available_metrics:
        st.error(
            f"Database not found for {get_metric_display_name(metric_type, aggregation)}"
        )
        st.write("Available metrics:", list(available_metrics.keys()))
        return

    _render_metric_display_fragment(
        method, available_metrics[metric_key], selected_layer
    )


@st.fragment
def _render_metric_display_fragment(method, metric_config: Dict[str, Any], layer: int):
    """Fragment for rendering metric display without recomputation."""

    # Load the MaxActStore for the selected metric
    max_store_path = metric_config["path"]

    # Create MaxActStore instance
    assert (
        method.tokenizer is not None
    ), "Tokenizer must be available for MaxActStore visualization"
    max_store = ReadOnlyMaxActStore(
        max_store_path,
        tokenizer=method.tokenizer,
    )

    # Create and display the dashboard component
    title = f"{metric_config['display_name']} Examples - Layer {layer}"
    component = MaxActivationDashboardComponent(max_store, title=title)
    component.display()


def _render_activation_steering_tab(method):
    """Render activation steering analysis tab with layer and dataset selection."""

    # Layer selection
    selected_layer = st.selectbox(
        "Select Layer", method._find_available_layers(), key="layer_selector_steering"
    )

    # Find available datasets for this layer
    available_datasets = method._find_available_datasets_for_layer(selected_layer)

    if not available_datasets:
        st.error(
            f"No datasets with activation difference means found for layer {selected_layer}"
        )
        return

    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset", available_datasets, key="dataset_selector_steering"
    )

    # Check if means exist for this layer/dataset combination
    loaded_means, custom_options = method._load_and_prepare_means(
        selected_layer, selected_dataset
    )

    if not loaded_means:
        st.error(
            f"No activation difference means found for {selected_dataset}, layer {selected_layer}"
        )
        return

    if len(custom_options) == 0:
        st.warning(
            f"No valid activation difference means found for {selected_dataset}, layer {selected_layer}"
        )
        return

    st.info(f"Found {len(custom_options)} activation difference means for steering")

    # Create and display steering dashboard
    steering_dashboard = ActivationAnalysisSteeringDashboard(
        method_instance=method, layer=selected_layer, dataset_name=selected_dataset
    )

    steering_dashboard.display()


def _render_activation_difference_lens_tab(method):
    """Render activation difference lens analysis tab."""
    # Layer selection
    selected_layer = st.selectbox(
        "Select Layer", method._find_available_layers(), key="layer_selector_diff_lens"
    )

    # Find available datasets for this layer
    available_datasets = method._find_available_datasets_for_layer(selected_layer)

    if not available_datasets:
        st.error(
            f"No datasets with activation difference means found for layer {selected_layer}"
        )
        return

    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset", available_datasets, key="dataset_selector_diff_lens"
    )

    # Load activation means for selected dataset/layer
    loaded_means, custom_options = method._load_and_prepare_means(
        selected_layer, selected_dataset
    )

    if not loaded_means:
        st.error(
            f"No activation difference means found for {selected_dataset}, layer {selected_layer}"
        )
        return

    # Plot all available means with their norms
    st.subheader("Activation Difference Mean Norms")

    # Extract norms and labels for plotting
    labels = []
    norms = []

    # Sort custom_options by number if they contain numbers
    def extract_number(name):
        import re

        match = re.search(r"\d+", name)
        return int(match.group()) if match else float("inf")

    sorted_options = sorted(custom_options, key=extract_number)

    for display_name in sorted_options:
        mean_tensor = loaded_means[display_name]["mean"]
        norm_value = float(torch.norm(mean_tensor).item())

        labels.append(display_name)
        norms.append(norm_value)

    if len(norms) > 0:
        # Create bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot norms
        bars = ax.bar(range(len(norms)), norms, color="steelblue", alpha=0.7)
        ax.set_xlabel("Activation Difference Means")
        ax.set_ylabel("L2 Norm")
        ax.set_title(
            f"Norms of Activation Difference Means - Layer {selected_layer} - {selected_dataset}"
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, norm_val) in enumerate(zip(bars, norms)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(norms) * 0.01,
                f"{norm_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Create get_latent_fn
    def get_latent_fn(latent_name):
        return loaded_means[latent_name]["mean"]

    # Render logit lens interface
    render_latent_lens_tab(
        method=method,
        get_latent_fn=get_latent_fn,
        max_latent_idx=len(custom_options),
        layer=selected_layer,
        latent_type_name="Token",
        patch_scope_add_scaler=True,
        custom_latent_options=custom_options,
    )


def visualize(method):
    """Visualize the activation analysis."""

    st.title("Activation Analysis")

    multi_tab_interface(
        [
            ("📊 MaxAct Examples", lambda: _render_dataset_statistics(method)),
            (
                "🔥 Interactive",
                lambda: ActivationAnalysisOnlineDashboard(method).display(),
            ),
            ("🎯 Activation Steering", lambda: _render_activation_steering_tab(method)),
            ("🎨 Plots", lambda: _render_plots_tab(method)),
            (
                "🔬 Activation Difference Lens",
                lambda: _render_activation_difference_lens_tab(method),
            ),
        ],
        "Activation Analysis",
    )
