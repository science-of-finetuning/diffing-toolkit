"""
UI utilities for activation difference lens analysis.
"""

from typing import Dict, Any, List
import streamlit as st
import torch
import matplotlib.pyplot as plt
import json
import html

from src.utils.dashboards import (
    AbstractOnlineDiffingDashboard,
    SteeringDashboard,
)
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
        # Extract position number from filename
        pos_str = mean_file.stem.split('_')[-1]  # Get the number part
        position = int(pos_str)  # Use position directly as 0-indexed
        
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


def _find_available_positions_for_steering(method, layer: int, dataset: str) -> List[int]:
    """Return sorted positions that have steering generations for given layer/dataset."""
    dataset_dir = method.results_dir / f"layer_{layer}" / dataset / "steering"
    if not dataset_dir.exists():
        return []
    positions: List[int] = []
    for pos_dir in dataset_dir.iterdir():
        if not pos_dir.is_dir():
            continue
        if not pos_dir.name.startswith("position_"):
            continue
        try:
            pos = int(pos_dir.name.split("_")[-1])
        except ValueError:
            continue
        gen_fp = pos_dir / "generations.jsonl"
        if gen_fp.exists():
            positions.append(pos)
    return sorted(positions)


def _find_layers_with_steered_generations(method) -> List[int]:
    """Return layers that have at least one dataset with steered generations present."""
    result: List[int] = []
    base = method.results_dir
    if not base.exists():
        return result
    for layer_dir in base.iterdir():
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        try:
            layer = int(layer_dir.name.split("_")[-1])
        except ValueError:
            continue
        found = False
        for dataset_dir in layer_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            steering_dir = dataset_dir / "steering"
            if not steering_dir.exists():
                continue
            for pos_dir in steering_dir.iterdir():
                if pos_dir.is_dir() and pos_dir.name.startswith("position_") and (pos_dir / "generations.jsonl").exists():
                    found = True
                    break
            if found:
                break
        if found:
            result.append(layer)
    return sorted(result)


def _find_datasets_with_steered_generations_for_layer(method, layer: int) -> List[str]:
    """Return datasets within a layer that have steered generations present."""
    out: List[str] = []
    layer_dir = method.results_dir / f"layer_{layer}"
    if not layer_dir.exists():
        return out
    for dataset_dir in layer_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        steering_dir = dataset_dir / "steering"
        if not steering_dir.exists():
            continue
        has = False
        for pos_dir in steering_dir.iterdir():
            if pos_dir.is_dir() and pos_dir.name.startswith("position_") and (pos_dir / "generations.jsonl").exists():
                has = True
                break
        if has:
            out.append(dataset_dir.name)
    return sorted(out)


def _load_steering_generations(method, layer: int, dataset: str, position: int) -> Dict[str, Any]:
    """Load and group generations by prompt for given layer/dataset/position.

    Returns dict with keys:
      - per_prompt: {prompt: {steered: List[str], unsteered: List[str], hyperparams: Dict[str, Any]}}
      - meta_path: str path to generations file
    """
    gen_path = (
        method.results_dir
        / f"layer_{layer}"
        / dataset
        / "steering"
        / f"position_{position}"
        / "generations.jsonl"
    )
    assert gen_path.exists(), f"Generations file not found: {gen_path}"

    per_prompt: Dict[str, Dict[str, Any]] = {}
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = str(rec["prompt"])  # fail fast if missing
            steered_samples = list(rec["steered_samples"])  # type: ignore[index]
            unsteered_samples = list(rec["unsteered_samples"])  # type: ignore[index]
            hyperparams = {
                "strength": float(rec["strength"]),
                "temperature": float(rec["temperature"]),
                "max_new_tokens": int(rec["max_new_tokens"]),
                "do_sample": bool(rec["do_sample"]),
                "num_samples": int(rec["num_samples"]),
                "layer": int(rec["layer"]),
                "position": int(rec["position"]),
            }
            if prompt not in per_prompt:
                per_prompt[prompt] = {
                    "steered": [],
                    "unsteered": [],
                    "hyperparams": hyperparams,
                }
            per_prompt[prompt]["steered"].extend(steered_samples)
            per_prompt[prompt]["unsteered"].extend(unsteered_samples)
    return {"per_prompt": per_prompt, "meta_path": str(gen_path)}


def _render_position_norms_plot(position_means: Dict[int, Dict[str, Any]], layer: int, dataset: str):
    """Render bar plot of position mean norms."""
    st.subheader("Position Mean Norms")
    
    positions = sorted(position_means.keys())
    labels = [f"Pos {pos}" for pos in positions]
    norms = [float(torch.norm(position_means[pos]['mean']).item()) for pos in positions]
    
    if len(norms) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        
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
        st.pyplot(fig, use_container_width=False)
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
        position = int(position_name.split('_')[1]) 
        mean_tensor: torch.Tensor = position_means[position]['mean']
        assert isinstance(mean_tensor, torch.Tensor)
        assert mean_tensor.ndim == 1, f"Expected 1D latent, got {tuple(mean_tensor.shape)}"

        # Determine which model is selected inside the latent lens tab
        assert "model_choice_selector_latent_lens_tab" in st.session_state, "Model selection not found; ensure the 'model_choice_selector_latent_lens_tab' selector is rendered first."
        model_choice = st.session_state["model_choice_selector_latent_lens_tab"]

        # Scale to the expected model norm for this layer and dataset
        model_norms = _load_model_norms(method, selected_dataset)
        if model_choice == "Finetuned Model":
            target_norm = float(model_norms["ft_model_norms"][selected_layer].item())
        else:
            target_norm = float(model_norms["base_model_norms"][selected_layer].item())

        current_norm = float(torch.norm(mean_tensor).item())
        assert current_norm > 0, "Mean tensor has zero norm"
        scaled = (mean_tensor / current_norm) * target_norm
        return scaled
    
    # Position options for custom latent selection
    position_options = [f"Position_{pos}" for pos in sorted(position_means.keys())]
    
    # Render logit lens interface
    render_latent_lens_tab(
        method=method,
        get_latent_fn=get_latent_fn,
        max_latent_idx=len(position_options),
        slider_min_value=0.0,
        slider_max_value=200.0,
        slider_value=1.0,
        slider_step=0.01,
        layer=selected_layer,
        latent_type_name="Position",
        patch_scope_add_scaler=True,
        custom_latent_options=position_options,
        dataset_name=selected_dataset,
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
        mean_tensor = self._position_means[position]['mean'].detach().to(self.method.device)

        # Scale to match expected finetuned model norm at this layer
        model_norms = _load_model_norms(self.method, self._dataset_name)
        ft_model_norm = float(model_norms['ft_model_norms'][self._layer].item())
        mean_norm = torch.norm(mean_tensor)
        assert mean_norm > 0
        scaled = (mean_tensor / mean_norm) * ft_model_norm
        return scaled
    
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
        position_options = [f"Position {pos}" for pos in positions]
        
        col1, col2 = st.columns(2)
        
        recommended_value = None
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
            mean_norm = float(torch.norm(mean_tensor).item())
            
            # Load model norms and get both base and fine-tuned model norms for this layer
            model_norms = _load_model_norms(self.method, self._dataset_name)
            base_model_norm = float(model_norms['base_model_norms'][self._layer].item())
            ft_model_norm = float(model_norms['ft_model_norms'][self._layer].item())
            
            st.info(
                f"**Position {selected_position}:** count={metadata['count']} | "
                f"**Mean norm:** {mean_norm:.3f}  \n"
                f"**Base model norm:** {base_model_norm:.3f} | **FT model norm (steering target):** {ft_model_norm:.3f}"
            )

            # Display recommended steering strength if available
            steering_dir = (
                self.method.results_dir
                / f"layer_{self._layer}"
                / self._dataset_name
                / "steering"
                / f"position_{selected_position}"
            )
            threshold_file = steering_dir / "threshold.json"
            
        with col2:
            if threshold_file.exists():
                with open(threshold_file, "r", encoding="utf-8") as f:
                    threshold_data = json.load(f)
                if "avg_threshold" in threshold_data:
                    rec = float(threshold_data["avg_threshold"])  # recommended strength
                    recommended_value = rec
                    
            # Steering factor
            default_value = recommended_value if recommended_value is not None else 1.0
            steering_factor = st.number_input(
                "Steering Factor",
                value=default_value,
                help="Strength and direction of steering"
            )
            if recommended_value is not None:
                st.info(f"Recommended value: {recommended_value:.3f}")
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


def _render_steered_answers_tab(method):
    """Render a tab that displays existing steered answers grouped by prompt.

    UI flow: select layer → dataset → position, then list prompts as collapsed sections.
    Each section shows hyperparameters and scrollable lists of steered/unsteered samples.
    """
    # Layer selection
    available_layers = _find_layers_with_steered_generations(method)
    if not available_layers:
        st.error("No layer data found in results directory")
        return
    layer = st.selectbox("Select Layer", available_layers, key="layer_selector_steered_answers")

    # Dataset selection
    available_datasets = _find_datasets_with_steered_generations_for_layer(method, layer)
    if not available_datasets:
        st.error(f"No datasets with steered generations found for layer {layer}")
        return
    dataset = st.selectbox("Select Dataset", available_datasets, key="dataset_selector_steered_answers")

    # Position selection (only positions with existing generations)
    positions = _find_available_positions_for_steering(method, layer, dataset)
    if not positions:
        st.warning("No steered generations found for this layer/dataset. Run steering first.")
        return
    position = st.selectbox("Select Position", positions, key="position_selector_steered_answers")

    # Load generations
    data = _load_steering_generations(method, layer, dataset, position)
    grouped = data["per_prompt"]

    # Show quick summary/hyperparams consistency
    any_prompt = next(iter(grouped))
    hp = grouped[any_prompt]["hyperparams"]
    st.info(
        f"Strength={hp['strength']:.3f} | Temp={hp['temperature']:.2f} | Max New Tokens={hp['max_new_tokens']} | "
        f"Do Sample={hp['do_sample']} | Num Samples per prompt={hp['num_samples']}"
    )

    # Render prompts as collapsed sections
    for prompt in sorted(grouped.keys()):
        section = st.expander(f"Prompt: {prompt}", expanded=False)
        with section:
            params = grouped[prompt]["hyperparams"]
            st.caption(
                f"Layer={params['layer']} | Position={params['position']} | Strength={params['strength']:.3f} | "
                f"Temp={params['temperature']:.2f} | Max New Tokens={params['max_new_tokens']} | Do Sample={params['do_sample']}"
            )

            col1, col2 = st.columns(2)
            steered_samples: List[str] = grouped[prompt]["steered"]
            unsteered_samples: List[str] = grouped[prompt]["unsteered"]

            with col1:
                st.subheader("Steered")
                _render_samples_list_scrollable(steered_samples, height=420)

            with col2:
                st.subheader("Unsteered")
                _render_samples_list_scrollable(unsteered_samples, height=420)


def _render_samples_list_scrollable(samples: List[str], height: int = 420) -> None:
    assert isinstance(samples, list) and all(isinstance(s, str) for s in samples)
    container_style = (
        f"height:{height}px; overflow-y:auto; border:1px solid #e0e0e0; "
        "border-radius:8px; padding:8px; background-color:#fafafa;"
    )
    item_style = (
        "border:1px solid #d6d6d6; border-radius:6px; padding:8px; margin-bottom:8px; "
        "background-color:#ffffff;"
    )
    label_style = "font-size:12px; color:#666; margin-bottom:6px;"
    text_style = "white-space:pre-wrap; font-family:monospace; font-size:13px;"

    items_html: List[str] = []
    if len(samples) == 0:
        items_html.append('<div style="color:#999;">(no samples)</div>')
    else:
        for idx, s in enumerate(samples):
            esc = html.escape(s)
            items_html.append(
                f"<div style=\"{item_style}\">"
                f"<div style=\"{label_style}\">Sample {idx + 1}</div>"
                f"<div style=\"{text_style}\">{esc}</div>"
                f"</div>"
            )
    html_block = (
        f"<div style=\"{container_style}\">" + "".join(items_html) + "</div>"
    )
    st.markdown(html_block, unsafe_allow_html=True)


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
            projections_per_position[f"Position_{position}"] = proj_values.cpu().numpy()
        
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
            ("📜 Steered Answers", lambda: _render_steered_answers_tab(method)),
            ("🔥 Online Projections", lambda: ActDiffLensOnlineDashboard(method).display()),
        ],
        "Activation Difference Lens",
    )
