"""
Steering dashboard for activation difference mean-based steering.
"""

from typing import Dict, Any
import torch
import streamlit as st

from src.utils.dashboards import SteeringDashboard


class ActivationAnalysisSteeringDashboard(SteeringDashboard):
    """
    Steering dashboard for activation difference mean-based steering.
    
    This dashboard allows users to select activation difference means and perform latent steering
    by applying mean vectors as steering directions.
    """
    
    def __init__(self, method_instance, layer: int, dataset_name: str):
        super().__init__(method_instance)
        self._layer = layer
        self._dataset_name = dataset_name
        self._loaded_means = None
        self._custom_options = None
        
    def __hash__(self):
        return hash((self._layer, self._dataset_name))
    
    @property
    def layer(self) -> int:
        """Get the layer number for this steering dashboard."""
        return self._layer
        
    @property
    def dataset_name(self) -> str:
        """Get the dataset name for this steering dashboard."""
        return self._dataset_name
    
    def _ensure_means_loaded(self):
        """Lazy load activation difference means if not already loaded."""
        if self._loaded_means is None:
            self._loaded_means, self._custom_options = self.method._load_and_prepare_means(
                self._layer, self._dataset_name
            )
    
    def get_latent(self, idx: int) -> torch.Tensor:
        """
        Get activation difference mean vector for specified mean index.
        
        Args:
            idx: Mean index (corresponds to position in custom_options)
            
        Returns:
            Activation difference mean vector [activation_dim] for steering
        """
        self._ensure_means_loaded()
        
        if idx >= len(self._custom_options):
            raise IndexError(f"Mean index {idx} out of range [0, {len(self._custom_options)})")
        
        # Get the display name and corresponding mean
        display_name = self._custom_options[idx]
        mean_data = self._loaded_means[display_name]
        mean_tensor = mean_data['mean'].detach()
        # Normalize the mean vector to unit length
        normalized_mean = mean_tensor / torch.norm(mean_tensor)
        
        return normalized_mean.to(self.method.device)
    
    def get_dict_size(self) -> int:
        """Get the number of available activation difference means."""
        self._ensure_means_loaded()
        return len(self._custom_options)
    
    def get_max_activation(self, mean_idx: int) -> str:
        """
        Get statistics for a specific activation difference mean.
        
        Args:
            mean_idx: Mean index
            
        Returns:
            String describing the count and basic statistics
        """
        self._ensure_means_loaded()
        
        if mean_idx >= len(self._custom_options):
            return "unknown"
        
        display_name = self._custom_options[mean_idx]
        mean_data = self._loaded_means[display_name]
        
        count = mean_data['count']
        mean_tensor = mean_data['mean']
        norm = float(torch.norm(mean_tensor).item())
        
        return f"count={count}, norm={norm:.3f}"
    
    @st.fragment
    def _render_latent_selector(self) -> int:
        """Render activation mean selection UI fragment with session state."""
        import streamlit as st
        
        # Get number of means for validation
        dict_size = self.get_dict_size()
        
        if dict_size == 0:
            st.error("No activation difference means found")
            return 0
        
        # Create unique session state key for this steering dashboard
        session_key = f"activation_steering_mean_idx_layer_{self.layer}_dataset_{self.dataset_name}"
        
        # Initialize session state if not exists
        if session_key not in st.session_state:
            st.session_state[session_key] = 0
        
        # Ensure the session state value is within valid range
        if st.session_state[session_key] >= dict_size:
            st.session_state[session_key] = 0
        
        self._ensure_means_loaded()
        
        # Use selectbox with display names for better UX
        selected_display_name = st.selectbox(
            "Activation Difference Mean",
            options=self._custom_options,
            index=st.session_state[session_key],
            help="Choose which activation difference mean to use for steering",
            key=session_key + "_selectbox"
        )
        
        # Get the index of the selected display name
        mean_idx = self._custom_options.index(selected_display_name)
        
        # Update session state
        st.session_state[session_key] = mean_idx
        
        # Display statistics for the selected mean
        stats = self.get_max_activation(mean_idx)
        st.info(f"**Statistics:** {stats}")
        
        return mean_idx
    
    @st.fragment
    def _render_steering_controls(self, steering_key: str) -> float:
        """Render steering factor controls with checkbox toggle between slider and manual input."""
        import streamlit as st
        
        # Checkbox to toggle between slider and manual input
        manual_mode_key = steering_key + "_manual_mode"
        if manual_mode_key not in st.session_state:
            st.session_state[manual_mode_key] = False
        
        use_manual = st.checkbox(
            "Manual Value",
            value=st.session_state[manual_mode_key],
            key=manual_mode_key + "_checkbox"
        )
        
        # Update session state and trigger re-render if checkbox changed
        if use_manual != st.session_state[manual_mode_key]:
            st.session_state[manual_mode_key] = use_manual
            st.rerun()
        
        if use_manual:
            # Show only manual input
            steering_factor = st.number_input(
                "Steering Factor",
                value=st.session_state[steering_key],
                step=0.1,
                help="Set exact steering factor value",
                key=steering_key + "_manual_input"
            )
        else:
            # Show only slider
            steering_factor = st.slider(
                "Steering Factor",
                min_value=-1000.0,
                max_value=1000.0,
                value=st.session_state[steering_key],
                step=0.1,
                help="Strength and direction of steering (negative values reverse the effect)",
                key=steering_key + "_slider_input"
            )
        
        # Update main session state
        st.session_state[steering_key] = steering_factor
        
        return steering_factor
    
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render activation analysis steering-specific controls in Streamlit."""
        import streamlit as st
        
        col1, col2 = st.columns(2)
        
        with col1:
            mean_idx = self._render_latent_selector()
        
        with col2:
            # Single session state key for synchronized value
            steering_key = f"steering_factor_layer_{self.layer}_dataset_{self.dataset_name}"
            
            # Initialize session state if not exists
            if steering_key not in st.session_state:
                st.session_state[steering_key] = 1.0
            
            final_steering_factor = self._render_steering_controls(steering_key)
        steering_mode = st.selectbox(
            "Steering Mode",
            options=["prompt_only", "all_tokens", "linear_decay"],
            index=1,  # Default to all_tokens
            help="Apply steering only to prompt tokens or to all tokens (prompt + generated)",
        )
        
        if steering_mode == "linear_decay":
            linear_decay_steps = st.number_input(
                "Linear Decay Steps",
                value=10,
                help="Number of steps to decay the steering factor",
            )
        else:
            linear_decay_steps = None

        return {
            "latent_idx": mean_idx,
            "steering_factor": final_steering_factor,
            "steering_mode": steering_mode,
            "linear_decay_steps": linear_decay_steps,
        }
    
    def _get_title(self) -> str:
        """Get title for activation analysis steering."""
        return f"Activation Difference Steering - Layer {self.layer} - Dataset: {self.dataset_name}" 