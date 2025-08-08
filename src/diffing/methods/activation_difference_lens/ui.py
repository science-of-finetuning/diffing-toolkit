"""
UI utilities for activation difference lens analysis.
"""

from typing import Dict, Any, List
import streamlit as st
import torch
import matplotlib.pyplot as plt
import json

from src.utils.dashboards import AbstractOnlineDiffingDashboard, SteeringDashboard
from src.utils.visualization import multi_tab_interface, render_latent_lens_tab
from nnsight import NNsight


def _find_available_layers(method) -> List[int]:
    """Find available layer numbers from results directory."""
    model_dir = method.results_dir
    if not model_dir.exists():
        return []
    
    layers = []
    for layer_dir in model_dir.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            layer_num = int(layer_dir.name[6:])  # Remove "layer_" prefix
            layers.append(layer_num)
    
    return sorted(layers)


def _find_available_datasets_for_layer(method, layer: int) -> List[str]:
    """Find available datasets for a specific layer."""
    layer_dir = method.results_dir / f"layer_{layer}"
    if not layer_dir.exists():
        return []
    
    datasets = []
    for dataset_dir in layer_dir.iterdir():
        if dataset_dir.is_dir():
            datasets.append(dataset_dir.name)
    
    return sorted(datasets)


def _load_position_means(method, layer: int, dataset: str) -> Dict[int, Dict[str, Any]]:
    """Load position means and metadata for a layer/dataset combination."""
    dataset_dir = method.results_dir / f"layer_{layer}" / dataset
    assert dataset_dir.exists(), f"Dataset directory not found: {dataset_dir}"
    
    position_means = {}
    
    # Find all position files
    for mean_file in dataset_dir.glob("mean_pos_*.pt"):
        # Extract position number from filename (1-indexed in file, 0-indexed in return)
        pos_str = mean_file.stem.split('_')[-1]  # Get the number part
        file_position = int(pos_str)  # 1-indexed from file
        position = file_position - 1  # Convert to 0-indexed
        
        # Load tensor
        mean_tensor = torch.load(mean_file, map_location='cpu')
        
        # Load metadata
        meta_file = mean_file.with_suffix('.meta')
        assert meta_file.exists(), f"Metadata file not found: {meta_file}"
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        position_means[position] = {
            'mean': mean_tensor,
            'metadata': metadata
        }
    
    assert len(position_means) > 0, f"No position means found in {dataset_dir}"
    return position_means


def _load_model_norms(method, dataset: str) -> Dict[str, Any]:
    """Load model norm estimates for a dataset."""
    norms_file = method.results_dir / f"model_norms_{dataset}.pt"
    assert norms_file.exists(), f"Model norms file not found: {norms_file}"
    
    norms_data = torch.load(norms_file, map_location='cpu')
    return norms_data


def _render_position_norms_plot(position_means: Dict[int, Dict[str, Any]], layer: int, dataset: str):
    """Render bar plot of position mean norms."""
    st.subheader("Position Mean Norms")
    
    positions = sorted(position_means.keys())
    labels = [f"Pos {pos+1}" for pos in positions]
    norms = [float(torch.norm(position_means[pos]['mean']).item()) for pos in positions]
    
    if len(norms) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        bars = ax.bar(range(len(norms)), norms, color='steelblue', alpha=0.7)
        ax.set_xlabel('Position')
        ax.set_ylabel('L2 Norm')
        ax.set_title(f'Norms of Position Activation Difference Means - Layer {layer} - {dataset}')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, norm_val) in enumerate(zip(bars, norms)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(norms)*0.01, 
                    f'{norm_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def _render_activation_difference_lens_tab(method):
    """Render activation difference lens analysis tab."""
    # Layer selection
    available_layers = _find_available_layers(method)
    if not available_layers:
        st.error("No layer data found in results directory")
        return
    
    selected_layer = st.selectbox("Select Layer", available_layers, key="layer_selector_lens")
    
    # Dataset selection
    available_datasets = _find_available_datasets_for_layer(method, selected_layer)
    if not available_datasets:
        st.error(f"No datasets found for layer {selected_layer}")
        return
    
    selected_dataset = st.selectbox("Select Dataset", available_datasets, key="dataset_selector_lens")
    
    # Load position means
    position_means = _load_position_means(method, selected_layer, selected_dataset)
    
    # Display norms plot
    _render_position_norms_plot(position_means, selected_layer, selected_dataset)
    
    st.divider()
    
    # Create get_latent_fn for logit lens interface
    def get_latent_fn(position_name):
        position = int(position_name.split('_')[1]) - 1  # Convert from "Position_1" to 0-indexed
        return position_means[position]['mean']
    
    # Position options for custom latent selection
    position_options = [f"Position_{pos+1}" for pos in sorted(position_means.keys())]
    
    # Render logit lens interface
    render_latent_lens_tab(
        method=method,
        get_latent_fn=get_latent_fn,
        max_latent_idx=len(position_options),
        layer=selected_layer,
        latent_type_name="Position",
        patch_scope_add_scaler=True,
        custom_latent_options=position_options
    )


class ActDiffLensSteeringDashboard(SteeringDashboard):
    """Steering dashboard for activation difference position means."""
    
    def __init__(self, method_instance, layer: int, dataset_name: str):
        super().__init__(method_instance)
        self._layer = layer
        self._dataset_name = dataset_name
        self._position_means = None
        
    @property
    def layer(self) -> int:
        return self._layer
    
    def get_latent(self, idx: int) -> torch.Tensor:
        if self._position_means is None:
            self._position_means = _load_position_means(
                self.method, self._layer, self._dataset_name
            )
        
        positions = sorted(self._position_means.keys())
        position = positions[idx]
        mean_tensor = self._position_means[position]['mean'].detach()
        
        # Normalize and return
        normalized_mean = mean_tensor / torch.norm(mean_tensor)
        return normalized_mean.to(self.method.device)
    
    def get_dict_size(self) -> int:
        if self._position_means is None:
            self._position_means = _load_position_means(
                self.method, self._layer, self._dataset_name
            )
        return len(self._position_means)
    
    def _get_title(self) -> str:
        return f"Activation Difference Steering - Layer {self.layer} - Dataset: {self._dataset_name}"
    
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        if self._position_means is None:
            self._position_means = _load_position_means(
                self.method, self._layer, self._dataset_name
            )
        
        positions = sorted(self._position_means.keys())
        position_options = [f"Position {pos+1}" for pos in positions]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Position selection
            selected_position_str = st.selectbox(
                "Position",
                options=position_options,
                help="Choose which position's activation difference mean to use for steering"
            )
            
            selected_position_idx = position_options.index(selected_position_str)
            
            # Show position statistics
            selected_position = positions[selected_position_idx]
            metadata = self._position_means[selected_position]['metadata']
            mean_tensor = self._position_means[selected_position]['mean']
            steered_norm = float(torch.norm(mean_tensor).item())
            
            # Load model norms and get both base and fine-tuned model norms for this layer
            model_norms = _load_model_norms(self.method, self._dataset_name)
            base_model_norm = float(model_norms['base_model_norms'][self._layer].item())
            ft_model_norm = float(model_norms['ft_model_norms'][self._layer].item())
            
            st.info(f"**Position {selected_position+1}:** count={metadata['count']} | **Steered norm:** {steered_norm:.3f}  \n**Base model norm:** {base_model_norm:.3f} | **FT model norm:** {ft_model_norm:.3f}")
            
        with col2:
            # Steering factor
            steering_factor = st.number_input(
                "Steering Factor",
                value=1.0,
                help="Strength and direction of steering"
            )
        
        # Steering mode
        steering_mode = st.selectbox(
            "Steering Mode",
            options=["prompt_only", "all_tokens", "linear_decay"],
            index=1,  # Default to all_tokens
            help="Apply steering only to prompt tokens or to all tokens"
        )
        
        if steering_mode == "linear_decay":
            linear_decay_steps = st.number_input(
                "Linear Decay Steps",
                value=10,
                help="Number of steps to decay the steering factor"
            )
        else:
            linear_decay_steps = None
        
        return {
            "latent_idx": selected_position_idx,
            "steering_factor": steering_factor,
            "steering_mode": steering_mode,
            "linear_decay_steps": linear_decay_steps,
        }


def _render_activation_difference_steering_tab(method):
    """Render activation difference steering analysis tab."""
    # Layer selection
    available_layers = _find_available_layers(method)
    if not available_layers:
        st.error("No layer data found in results directory")
        return
        
    selected_layer = st.selectbox("Select Layer", available_layers, key="layer_selector_steering")
    
    # Dataset selection
    available_datasets = _find_available_datasets_for_layer(method, selected_layer)
    if not available_datasets:
        st.error(f"No datasets found for layer {selected_layer}")
        return
    
    selected_dataset = st.selectbox("Select Dataset", available_datasets, key="dataset_selector_steering")
    
    # Create and display steering dashboard
    steering_dashboard = ActDiffLensSteeringDashboard(
        method_instance=method,
        layer=selected_layer,
        dataset_name=selected_dataset
    )
    steering_dashboard.display()


class ActDiffLensOnlineDashboard(AbstractOnlineDiffingDashboard):
    """Online dashboard for activation difference lens projections."""
    
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        available_layers = _find_available_layers(self.method)
        if not available_layers:
            st.error("No layer data found")
            return {}
            
        layer = st.selectbox("Select Layer:", available_layers, key="online_layer_selector")
        
        available_datasets = _find_available_datasets_for_layer(self.method, layer)
        if not available_datasets:
            st.error(f"No datasets found for layer {layer}")
            return {}
            
        dataset = st.selectbox("Select Dataset:", available_datasets, key="online_dataset_selector")
        
        return {"layer": layer, "dataset": dataset}
    
    def compute_statistics_for_tokens(self, input_ids, attention_mask, **kwargs):
        layer = kwargs.get("layer")
        dataset = kwargs.get("dataset")
        
        if layer is None or dataset is None:
            return {'tokens': [], 'values': [], 'total_tokens': 0}
        
        # Load position means
        position_means = _load_position_means(self.method, layer, dataset)
        
        # Extract activations for input tokens at specified layer
        with torch.no_grad():
            activations = self._extract_activations_for_tokens(input_ids, layer).float()
            
        # Compute projections onto each position mean
        projections_per_position = {}
        for position, mean_data in position_means.items():
            mean_vector = mean_data['mean'].to(activations.device).float()
            # Project each token's activation onto this position mean
            proj_values = torch.matmul(activations, mean_vector) / torch.norm(mean_vector)
            projections_per_position[f"Position_{position+1}"] = proj_values.cpu().numpy()
        
        # Select which position to display
        position_keys = sorted(projections_per_position.keys())
        selected_position = st.selectbox(
            "Select Position to Display:",
            options=position_keys,
            help="Choose which position's projections to visualize",
            key="online_position_selector"
        )
        
        return {
            'tokens': [self.method.tokenizer.decode([token_id]) for token_id in input_ids[0]],
            'values': projections_per_position[selected_position],
            'total_tokens': len(input_ids[0]),
        }
    
    def _extract_activations_for_tokens(self, input_ids, layer):
        """Extract activations from specified layer for input tokens."""
        # Use the finetuned model for consistency with logit lens
        model = self.method.finetuned_model
        nn_model = NNsight(model)
        
        with nn_model.trace(input_ids):
            activations = nn_model.model.layers[layer].output[0].save()
        
        # Shape assertion
        assert activations.ndim == 3, f"Expected 3D activations [batch, seq, hidden], got {activations.shape}"
        return activations.squeeze(0)  # Remove batch dimension [seq_len, hidden_dim]
    
    def get_method_specific_params(self) -> Dict[str, Any]:
        return {}
    
    def _get_title(self) -> str:
        return "Activation Difference Lens - Online Projections"


def visualize(method):
    """Visualize the activation difference lens analysis."""
    
    st.title("Activation Difference Lens")

    multi_tab_interface(
        [
            ("🔬 Lens", lambda: _render_activation_difference_lens_tab(method)),
            ("🎯 Steering", lambda: _render_activation_difference_steering_tab(method)),
            ("🔥 Online Projections", lambda: ActDiffLensOnlineDashboard(method).display()),
        ],
        "Activation Difference Lens",
    )
