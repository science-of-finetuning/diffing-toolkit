"""
Amplifications Tab - Tab 1 of the amplification dashboard.

Provides UI for creating, editing, and organizing amplification configurations
with folder-based organization via FolderManagerUI.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import streamlit as st

from diffing.utils.configs import get_available_organisms, get_organism_variants
from ..amplification_config import (
    AmplifiedAdapter,
    LayerAmplification,
    LayerRange,
    ModuleAmplification,
    CUSTOM_ADAPTER_ORGANISM,
)
from .dashboard_state import (
    ManagedConfig,
    get_unique_config_name,
)

if TYPE_CHECKING:
    from diffing.methods.amplification.amplification_dashboard import (
        AmplificationDashboard,
    )


class AmplificationsTab:
    """Renders Tab 1: Amplification configuration management.

    Provides UI for creating, editing, and organizing amplification configs
    with folder-based organization via FolderManagerUI.
    """

    def __init__(self, dashboard: "AmplificationDashboard"):
        self.dashboard = dashboard

    @st.fragment
    def render(self) -> None:
        """Render the Amplifications tab."""
        st.markdown("## Amplification Configurations")
        st.markdown(
            "Create and manage amplification configurations for adapter weight modification."
        )

        self.dashboard._config_folder_manager.render_folder_loader()

        st.markdown("---")

        self.dashboard._config_folder_manager.render_all_folders(
            render_item=self._render_amplification_config,
            render_item_actions=self._render_config_actions,
        )

    def _render_config_actions(self, config_id: str, mc: ManagedConfig) -> None:
        """Render duplicate/delete buttons for a config."""
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋", key=f"dup_{config_id}", help="Duplicate"):
                from copy import deepcopy

                config = mc.config
                new_config = deepcopy(config)
                new_config.name = get_unique_config_name(
                    f"{config.name} copy", mc.folder
                )
                new_managed = ManagedConfig.from_config(
                    new_config,
                    active=mc.active,
                    expanded=True,
                    folder=mc.folder,
                )
                st.session_state.managed_configs[new_managed.config_id] = new_managed
                self.dashboard.persistence.save_configs()
                st.rerun(scope="fragment")
        with col2:
            if st.button("🗑️", key=f"del_{config_id}", help="Delete"):
                deleted = (mc.folder, mc.config.name)
                del st.session_state.managed_configs[config_id]
                self.dashboard.persistence.save_configs(deleted=deleted)
                st.rerun(scope="fragment")

    @st.fragment
    def _render_amplification_config(
        self,
        config_id: str,
        mc: ManagedConfig,
        key_prefix: str = "",
        sidebar_mode: bool = False,
    ) -> None:
        """Render one amplification config. Fragment for independent updates.

        Args:
            config_id: Unique identifier for the config
            mc: ManagedConfig wrapper
            key_prefix: Optional prefix for widget keys to avoid conflicts when rendering
                        the same config in multiple places (e.g., sidebar vs main area)
            sidebar_mode: If True, renders without expander and without duplicate/delete buttons
        """
        config = mc.config
        self._render_amplification_config_content(
            config_id, mc, config, key_prefix, sidebar_mode
        )

    def _render_amplification_config_content(
        self,
        config_id: str,
        mc: ManagedConfig,
        config,
        key_prefix: str,
        sidebar_mode: bool,
    ) -> None:
        """Inner content renderer for amplification config."""
        if sidebar_mode:
            # Render directly without expander
            self._render_config_fields(config_id, mc, config, key_prefix, sidebar_mode)
        else:
            # Check if this config is being edited in sidebar quick edit
            sidebar_editing_id = st.session_state.get("sidebar_quick_edit_config_id")
            is_editing_in_sidebar = sidebar_editing_id == config_id

            # Render inside expander with action buttons
            icon = "✅" if mc.active else "❌"
            sidebar_indicator = " 📝" if is_editing_in_sidebar else ""
            with st.expander(
                f"{icon} {mc.full_name}{sidebar_indicator}", expanded=mc.expanded
            ):
                if is_editing_in_sidebar:
                    st.info("Currently editing in sidebar quick edit")
                    if st.button(
                        "Edit here instead",
                        key=f"edit_here_{config_id}",
                        use_container_width=True,
                    ):
                        st.session_state.sidebar_quick_edit_config_id = None
                        st.session_state["sidebar_config_selector"] = "None"
                        st.rerun()
                else:
                    self._render_config_fields(
                        config_id, mc, config, key_prefix, sidebar_mode
                    )

    def _render_config_fields(
        self,
        config_id: str,
        mc: ManagedConfig,
        config,
        key_prefix: str,
        sidebar_mode: bool,
    ) -> None:
        """Render the actual config fields."""
        if not sidebar_mode:
            # Name input (dup/delete buttons moved to folder section list level)
            name_key = f"{key_prefix}config_name_{config_id}"

            def on_name_change(
                cfg=config, cid=config_id, key=name_key, managed_config=mc
            ):
                new_name = st.session_state[key]
                if new_name != cfg.name:
                    # Ensure unique name within folder
                    unique_name = get_unique_config_name(
                        new_name, managed_config.folder, exclude_config_id=cid
                    )
                    # Use rename() which tracks old disk name for cleanup
                    deleted = managed_config.rename(unique_name)
                    self.dashboard.persistence.save_configs(deleted=deleted)

            st.text_input(
                "Configuration Name",
                value=config.name,
                key=name_key,
                on_change=on_name_change,
            )

            desc_key = f"{key_prefix}config_desc_{config_id}"

            def on_description_change(cfg=config, key=desc_key):
                cfg.description = st.session_state[key]
                self.dashboard.persistence.save_configs()

            st.text_area(
                "Description",
                value=config.description,
                key=desc_key,
                height=60,
                on_change=on_description_change,
            )

        active_key = f"{key_prefix}config_active_{config_id}"

        # Sync session state with data model (handles bulk enable/disable)
        if active_key in st.session_state and st.session_state[active_key] != mc.active:
            st.session_state[active_key] = mc.active

        def on_active_change(managed_config=mc, key=active_key):
            managed_config.active = st.session_state[key]
            self.dashboard.persistence.save_configs()

        st.checkbox(
            "Active",
            value=mc.active,
            key=active_key,
            help="Only active configurations will be used for generation",
            on_change=on_active_change,
        )

        st.markdown("#### Adapters")

        if len(config.amplified_adapters) == 0:
            st.info("No adapters configured. Click 'Add Adapter' below.")
        else:
            for adapter_idx, adapter in enumerate(config.amplified_adapters):
                self._render_adapter_amplification(
                    config_id, adapter_idx, adapter, key_prefix, sidebar_mode
                )

        if st.button("➕ Add Adapter", key=f"{key_prefix}add_adapter_{config_id}"):
            new_adapter = AmplifiedAdapter(
                organism_name=CUSTOM_ADAPTER_ORGANISM,
                variant="",
                layer_amplifications=[
                    LayerAmplification(
                        layers="all",
                        module_amplifications=[
                            ModuleAmplification(modules="all", weight=1.0)
                        ],
                    )
                ],
            )
            config.amplified_adapters.append(new_adapter)
            self.dashboard.persistence.save_configs_and_rerun(scope="fragment")

    def _render_adapter_amplification(
        self,
        config_id: str,
        adapter_idx: int,
        adapter: AmplifiedAdapter,
        key_prefix: str = "",
        sidebar_mode: bool = False,
    ) -> None:
        """Render one adapter's amplifications."""
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                if adapter.organism_name == CUSTOM_ADAPTER_ORGANISM:
                    display_name = (
                        adapter.variant
                        if adapter.variant
                        else "Custom (not configured)"
                    )
                elif adapter.organism_name:
                    display_name = f"{adapter.organism_name} ({adapter.variant})"
                else:
                    display_name = "New Adapter"
                st.markdown(f"**Adapter: {display_name}**")

            with col2:
                if st.button(
                    "🗑️", key=f"{key_prefix}delete_adapter_{config_id}_{adapter_idx}"
                ):
                    st.session_state.managed_configs[
                        config_id
                    ].config.amplified_adapters.pop(adapter_idx)
                    self.dashboard.persistence.save_configs_and_rerun(scope="fragment")

            base_model_name = self.dashboard.method.base_model_cfg.name

            col1, col2 = st.columns(2)

            with col1:
                available_organisms = get_available_organisms(
                    base_model_name=self.dashboard.method.base_model_cfg.name,
                    only_loras=True,
                )
                organism_options = [CUSTOM_ADAPTER_ORGANISM] + available_organisms
                organism_key = f"{key_prefix}organism_{config_id}_{adapter_idx}"

                if adapter.organism_name in organism_options:
                    current_index = organism_options.index(adapter.organism_name)
                else:
                    current_index = 0

                def on_organism_change(adpt=adapter, key=organism_key):
                    selected = st.session_state[key]
                    if selected != adpt.organism_name:
                        adpt.organism_name = selected
                        if selected == CUSTOM_ADAPTER_ORGANISM:
                            adpt.variant = ""
                        else:
                            adpt.variant = "default"
                        self.dashboard.persistence.save_configs()

                st.selectbox(
                    "Organism",
                    options=organism_options,
                    index=current_index,
                    key=organism_key,
                    help="Select 'custom' to use a direct HuggingFace adapter ID, or choose an organism",
                    on_change=on_organism_change,
                )

            with col2:
                variant_key = f"{key_prefix}variant_{config_id}_{adapter_idx}"

                def on_variant_change(adpt=adapter, key=variant_key):
                    adpt.variant = st.session_state[key]
                    self.dashboard.persistence.save_configs()

                if adapter.organism_name == CUSTOM_ADAPTER_ORGANISM:
                    st.text_input(
                        "Adapter ID",
                        value=adapter.variant,
                        key=variant_key,
                        help="HuggingFace adapter ID (e.g., 'hf_user/repo' or 'hf_user/repo/path/in/repo')",
                        placeholder="hf_user/adapter_repo",
                        on_change=on_variant_change,
                    )
                elif adapter.organism_name:
                    available_variants = get_organism_variants(
                        adapter.organism_name, base_model_name, only_loras=True
                    )

                    if not available_variants:
                        st.warning(
                            f"No variants available for {adapter.organism_name} with base model {base_model_name}"
                        )
                        adapter.variant = "default"
                    else:
                        try:
                            current_index = (
                                available_variants.index(adapter.variant)
                                if adapter.variant in available_variants
                                else 0
                            )
                        except ValueError:
                            current_index = 0

                        st.selectbox(
                            "Variant",
                            options=available_variants,
                            index=current_index,
                            key=variant_key,
                            help="Select the variant of the organism",
                            on_change=on_variant_change,
                        )
                else:
                    st.info("Select an organism first")

            st.markdown("**Layer Specifications**")

            for layer_idx, layer_amp in enumerate(adapter.layer_amplifications):
                self._render_layer_amplification(
                    config_id,
                    adapter_idx,
                    layer_idx,
                    layer_amp,
                    key_prefix,
                    sidebar_mode,
                )

            if st.button(
                "➕ Add Layer Spec",
                key=f"{key_prefix}add_layer_{config_id}_{adapter_idx}",
            ):
                new_layer_amp = LayerAmplification(
                    layers="all",
                    module_amplifications=[
                        ModuleAmplification(modules="all", weight=1.0)
                    ],
                )
                adapter.layer_amplifications.append(new_layer_amp)
                self.dashboard.persistence.save_configs_and_rerun(scope="fragment")

    def _render_layer_amplification(
        self,
        config_id: str,
        adapter_idx: int,
        layer_idx: int,
        layer_amp: LayerAmplification,
        key_prefix: str = "",
        sidebar_mode: bool = False,
    ) -> None:
        """Render layer amplification specification."""
        base_key = f"{key_prefix}{config_id}_{adapter_idx}_{layer_idx}"
        mode_key = f"layer_mode_{base_key}"
        relative_key = f"layer_relative_{base_key}"
        single_key = f"layer_single_{base_key}"
        range_key = f"layer_range_{base_key}"
        list_key = f"layer_list_{base_key}"

        num_layers = self.dashboard.method.base_model.num_layers

        def on_mode_change(lamp=layer_amp, mk=mode_key):
            mode = st.session_state[mk]
            is_relative = lamp.is_relative
            if mode == "All":
                lamp.layers = "all"
            elif mode == "Single":
                lamp.layers = 0.0 if is_relative else 0
            elif mode == "Range":
                lamp.layers = (
                    LayerRange(0.0, 1.0)
                    if is_relative
                    else LayerRange(0, num_layers - 1)
                )
            else:  # List
                lamp.layers = []
            self.dashboard.persistence.save_configs()

        def on_relative_change(lamp=layer_amp, rk=relative_key, mk=mode_key):
            is_relative = st.session_state[rk]
            lamp.is_relative = is_relative
            mode = st.session_state.get(mk, "All")
            if mode == "Single":
                current = lamp.layers if isinstance(lamp.layers, (int, float)) else 0
                if is_relative:
                    lamp.layers = float(current) / (num_layers - 1)
                else:
                    lamp.layers = round(float(current) * (num_layers - 1))
            elif mode == "Range":
                if type(lamp.layers).__name__ == "LayerRange":
                    if is_relative:
                        lamp.layers = LayerRange(
                            lamp.layers.start / (num_layers - 1),
                            lamp.layers.end / (num_layers - 1),
                        )
                    else:
                        lamp.layers = LayerRange(
                            round(lamp.layers.start * (num_layers - 1)),
                            round(lamp.layers.end * (num_layers - 1)),
                        )
                else:
                    lamp.layers = (
                        LayerRange(0.0, 1.0)
                        if is_relative
                        else LayerRange(0, num_layers - 1)
                    )
            elif mode == "List":
                if isinstance(lamp.layers, list) and len(lamp.layers) > 0:
                    if is_relative:
                        lamp.layers = [float(v) / (num_layers - 1) for v in lamp.layers]
                    else:
                        lamp.layers = [
                            round(float(v) * (num_layers - 1)) for v in lamp.layers
                        ]
            self.dashboard.persistence.save_configs()

        def on_single_change(lamp=layer_amp, key=single_key):
            lamp.layers = st.session_state[key]
            self.dashboard.persistence.save_configs()

        def on_range_change(lamp=layer_amp, key=range_key):
            start, end = st.session_state[key]
            lamp.layers = LayerRange(float(start), float(end))
            self.dashboard.persistence.save_configs()

        def on_list_change(lamp=layer_amp, key=list_key):
            val = st.session_state[key].strip()
            if val:
                lamp.layers = [float(x.strip()) for x in val.split(",") if x.strip()]
            else:
                lamp.layers = []
            self.dashboard.persistence.save_configs()

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"**Layer Specification {layer_idx + 1}**")

            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_layer_{base_key}",
                ):
                    st.session_state.managed_configs[
                        config_id
                    ].config.amplified_adapters[adapter_idx].layer_amplifications.pop(
                        layer_idx
                    )
                    self.dashboard.persistence.save_configs_and_rerun(scope="fragment")

            if type(layer_amp.layers).__name__ == "LayerRange":
                initial_mode_index = 3  # "Range"
            elif isinstance(layer_amp.layers, list):
                initial_mode_index = 2  # "List"
            elif isinstance(layer_amp.layers, (int, float)):
                initial_mode_index = 1  # "Single"
            else:  # "all"
                initial_mode_index = 0  # "All"
            if sidebar_mode:
                col_radio = st.columns(1)[0]
                col_relative = col_radio
            else:
                col_radio, col_relative = st.columns([4, 1])
            with col_radio:
                layer_mode = st.radio(
                    "Layer Selection Mode",
                    options=["All", "Single", "List", "Range"],
                    index=initial_mode_index,
                    key=mode_key,
                    horizontal=True,
                    on_change=on_mode_change,
                )

            with col_relative:
                use_relative = st.checkbox(
                    "Relative",
                    value=layer_amp.is_relative,
                    key=relative_key,
                    help="Use relative layer positions (0.0-1.0) that scale with model size",
                    on_change=on_relative_change,
                )

            if layer_mode == "All":
                layer_amp.layers = "all"
                st.info("Applies to all layers in the model")

            elif layer_mode == "Single":
                if use_relative:
                    current_val = (
                        layer_amp.layers
                        if isinstance(layer_amp.layers, (int, float))
                        else 0.0
                    )
                    st.slider(
                        "Layer Position (relative)",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_val),
                        step=0.01,
                        key=single_key,
                        help=f"0.0 = first layer, 1.0 = last layer (layer {num_layers - 1})",
                        on_change=on_single_change,
                    )
                else:
                    current_val = (
                        layer_amp.layers
                        if isinstance(layer_amp.layers, (int, float))
                        else 0
                    )
                    st.number_input(
                        "Layer Index",
                        min_value=0,
                        value=int(current_val),
                        step=1,
                        key=single_key,
                        on_change=on_single_change,
                    )

            elif layer_mode == "Range":
                if type(layer_amp.layers).__name__ == "LayerRange":
                    current_start = layer_amp.layers.start
                    current_end = layer_amp.layers.end
                else:
                    current_start = 0.0 if use_relative else 0
                    current_end = 1.0 if use_relative else num_layers - 1

                if use_relative:
                    layer_range = st.slider(
                        "Layer Range (relative, inclusive)",
                        min_value=0.0,
                        max_value=1.0,
                        value=(float(current_start), float(current_end)),
                        step=0.01,
                        key=range_key,
                        help="0.0 = first layer, 1.0 = last layer",
                        on_change=on_range_change,
                    )
                    range_start, range_end = layer_range
                    abs_start = round(range_start * (num_layers - 1))
                    abs_end = round(range_end * (num_layers - 1))
                    st.info(
                        f"Applies to layers {abs_start} through {abs_end}/{num_layers - 1}"
                    )
                else:
                    layer_range = st.slider(
                        "Layer Range (inclusive)",
                        min_value=0,
                        max_value=num_layers - 1,
                        value=(int(current_start), int(current_end)),
                        key=range_key,
                        help="Select the range of layers to apply amplification to",
                        on_change=on_range_change,
                    )
                    range_start, range_end = layer_range
                    st.info(
                        f"Applies to layers {range_start} through {range_end}/{num_layers - 1}"
                    )

            else:  # List
                if isinstance(layer_amp.layers, list):
                    current_val = ",".join(map(str, layer_amp.layers))
                else:
                    current_val = ""
                if use_relative:
                    st.text_input(
                        "Layer Positions (comma-separated, 0.0-1.0)",
                        value=current_val,
                        key=list_key,
                        help="E.g., '0.0, 0.25, 0.5, 0.75, 1.0'",
                        on_change=on_list_change,
                    )
                else:
                    st.text_input(
                        "Layer Indices (comma-separated)",
                        value=current_val,
                        key=list_key,
                        help="E.g., '0,1,2,5,10'",
                        on_change=on_list_change,
                    )

            st.markdown("**Module Specifications**")

            if len(layer_amp.module_amplifications) == 0:
                st.info("No module specifications. Click 'Add Module' below.")
            else:
                for module_idx, module_amp in enumerate(
                    layer_amp.module_amplifications
                ):
                    self._render_module_amplification(
                        config_id,
                        adapter_idx,
                        layer_idx,
                        module_idx,
                        module_amp,
                        key_prefix,
                    )

            if st.button(
                "➕ Add Module",
                key=f"{key_prefix}add_module_{config_id}_{adapter_idx}_{layer_idx}",
            ):
                new_module_amp = ModuleAmplification(modules="all", weight=1.0)
                layer_amp.module_amplifications.append(new_module_amp)
                self.dashboard.persistence.save_configs_and_rerun(scope="fragment")

    def _render_module_amplification(
        self,
        config_id: str,
        adapter_idx: int,
        layer_idx: int,
        module_idx: int,
        module_amp: ModuleAmplification,
        key_prefix: str = "",
    ) -> None:
        """Render module amplification (module selector + weight slider)."""
        base_key = f"{key_prefix}{config_id}_{adapter_idx}_{layer_idx}_{module_idx}"
        module_key = f"module_mode_{base_key}"
        weight_slider_key = f"module_weight_slider_{base_key}"
        weight_input_key = f"module_weight_input_{base_key}"

        def on_module_change(mod_amp=module_amp, key=module_key):
            mod_amp.modules = st.session_state[key]
            self.dashboard.persistence.save_configs()

        def on_weight_slider_change(
            mod_amp=module_amp, slider_key=weight_slider_key, input_key=weight_input_key
        ):
            mod_amp.weight = st.session_state[slider_key]
            # Sync input widget to match slider
            st.session_state[input_key] = st.session_state[slider_key]
            self.dashboard.persistence.save_configs()

        def on_weight_input_change(
            mod_amp=module_amp, input_key=weight_input_key, slider_key=weight_slider_key
        ):
            new_weight = st.session_state[input_key]
            mod_amp.weight = new_weight
            # Sync slider widget value to match input
            st.session_state[slider_key] = new_weight
            self.dashboard.persistence.save_configs()

        # Dynamically expand slider range to include current weight value
        current_weight = float(module_amp.weight)
        slider_min = min(-5.0, current_weight)
        slider_max = max(5.0, current_weight)

        # Initialize session state for widgets if not already set (avoids Streamlit warning)
        if weight_slider_key not in st.session_state:
            st.session_state[weight_slider_key] = current_weight
        if weight_input_key not in st.session_state:
            st.session_state[weight_input_key] = current_weight

        col1, col2, col3, col4 = st.columns([2, 2, 1.2, 0.8])

        with col1:
            st.selectbox(
                f"Module {module_idx + 1}",
                options=["all", "attention", "mlp"],
                index=(
                    0
                    if module_amp.modules == "all"
                    else (
                        1
                        if module_amp.modules == "attention"
                        else 2 if module_amp.modules == "mlp" else 3
                    )
                ),
                key=module_key,
                on_change=on_module_change,
            )

        with col2:
            st.slider(
                "Weight",
                min_value=slider_min,
                max_value=slider_max,
                step=0.1,
                key=weight_slider_key,
                help="Amplification factor (1.0 = no change, 2.0 = double, 0.5 = half)",
                on_change=on_weight_slider_change,
            )

        with col3:
            st.number_input(
                "Custom",
                step=0.1,
                format="%.2f",
                key=weight_input_key,
                help="Enter custom weight (can be outside -5 to 5 range)",
                on_change=on_weight_input_change,
            )

        with col4:
            if st.button(
                "🗑️",
                key=f"delete_module_{base_key}",
            ):
                st.session_state.managed_configs[config_id].config.amplified_adapters[
                    adapter_idx
                ].layer_amplifications[layer_idx].module_amplifications.pop(module_idx)
                self.dashboard.persistence.save_configs_and_rerun(scope="fragment")
