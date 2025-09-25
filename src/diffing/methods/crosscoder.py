"""
Crosscoder-based model diffing method.

This module trains crosscoders on paired activations from base and finetuned models,
then runs a comprehensive analysis pipeline including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.

Key assumptions:
- Preprocessing pipeline has generated paired activation caches
- dictionary_learning library is available and compatible
- ##REDACTED## repository is available for analysis pipeline
- W&B configuration is available in infrastructure config
- Sufficient GPU memory and disk space for training
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import json
from collections import defaultdict, Counter
import numpy as np
import streamlit as st
import base64
import pandas as pd

from .diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices
from src.utils.dictionary.analysis import build_push_crosscoder_latent_df, make_plots
from src.utils.dictionary.training import train_crosscoder_for_layer
from src.utils.dictionary.latent_scaling.closed_form import compute_scalers_from_config
from src.utils.dictionary.latent_scaling.beta_analysis import update_latent_df_with_beta_values
from src.utils.dictionary.latent_activations import (
    collect_dictionary_activations_from_config,
    collect_activating_examples,
    update_latent_df_with_stats,
)
from src.utils.model import place_inputs
from src.utils.dictionary.utils import load_dictionary_model
from src.utils.dictionary.training import crosscoder_run_name
from src.utils.visualization import multi_tab_interface
from src.utils.dashboards import AbstractOnlineDiffingDashboard, SteeringDashboard, MaxActivationDashboardComponent
from src.utils.max_act_store import ReadOnlyMaxActStore
from src.utils.dictionary.steering import run_latent_steering_experiment, get_crosscoder_latent, display_steering_results


class CrosscoderDiffingMethod(DiffingMethod):
    """
    Trains crosscoders on paired activations and runs comprehensive analysis.

    This method:
    1. Loads paired activation caches from preprocessing pipeline
    2. Trains crosscoders for specified layers using local shuffling
    3. Saves trained models with configuration and metrics
    4. Optionally uploads models to Hugging Face Hub
    5. Runs complete analysis pipeline from ##REDACTED##
    6. Returns comprehensive results including training metrics and analysis outcomes
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

        # Initialize latent df cache
        self.latent_df_cache = {}

    def run(self) -> Dict[str, Any]:
        """
        Main training orchestration with analysis pipeline.

        Trains crosscoders for each specified layer, then runs the complete
        analysis pipeline for each trained model.

        Returns:
            Dictionary containing training results, model paths, and analysis outcomes

        Assumptions:
            - Paired activation caches exist for all specified layers
            - Sufficient resources for training and analysis
        """
        logger.info(f"Starting crosscoder training for layers: {self.layers}")

        for layer_idx in self.layers:
            logger.info(f"Processing layer {layer_idx}")

            logger.info(f"Training crosscoder for layer {layer_idx}")

            dictionary_name = crosscoder_run_name(
                self.cfg, layer_idx, self.base_model_cfg, self.finetuned_model_cfg
            )
            model_results_dir = (
                self.results_dir / "crosscoder" / f"layer_{layer_idx}" / dictionary_name
            )
            logger.info(f"Model results directory: {model_results_dir}")
            model_results_dir.mkdir(parents=True, exist_ok=True)
            if (
                not (model_results_dir / "dictionary_model" / "model.safetensors").exists()
                or self.method_cfg.training.overwrite
            ):  
                # Train crosscoder for this layer
                training_metrics, model_path = train_crosscoder_for_layer(
                    self.cfg, layer_idx, self.device, dictionary_name
                )
                # save training metrics
                with open(model_results_dir / "training_metrics.json", "w") as f:
                    json.dump(training_metrics, f)
                # save model
                dictionary_model = load_dictionary_model(model_path)
                dictionary_model.save_pretrained(model_results_dir / "dictionary_model")

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
                build_push_crosscoder_latent_df(
                    dictionary_name=dictionary_name,
                    base_layer=0,
                    ft_layer=1,
                )

                if self.method_cfg.analysis.latent_scaling.enabled:
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
                    logger.info(f"Running latent steering experiment for layer {layer_idx}")
                    run_latent_steering_experiment(
                        method=self,
                        get_latent_fn=get_crosscoder_latent,
                        dictionary_name=dictionary_name,
                        results_dir=model_results_dir,
                        layer=layer_idx,
                    )

            logger.info(f"Successfully completed layer {layer_idx}")

    def visualize(self) -> None:
        """
        Interactive Streamlit visualization for CrossCoder results.

        Provides the same 4-tab interface as SAE Difference:
          • 📊 MaxAct Examples – show top activating prompts
          • 🔥 Online Inference – per-token latent activations
          • 🎯 Steering – latent decoder steering playground
          • 📈 Latent Statistics – interactive latent_df filtering
          • 🎨 Plots – show analysis plots
        """
        st.subheader("CrossCoder Analysis")

        # Session-state key for active tab
        tab_key = "crosscoder_active_tab"
        if tab_key not in st.session_state:
            st.session_state[tab_key] = 0

        # Discover available crosscoders
        available_ccs = self._get_available_crosscoder_directories()
        if not available_ccs:
            st.error(f"No trained CrossCoder directories found in {self.results_dir / 'crosscoder'}")
            return

        # Index by layer
        cc_by_layer = defaultdict(list)
        for cc_info in available_ccs:
            cc_by_layer[cc_info["layer"]].append(cc_info)

        layers_sorted = sorted(cc_by_layer.keys())
        layer_select_key = "crosscoder_selected_layer"
        dict_select_key = "crosscoder_selected_dictionary"

        # Initialise session state
        if layer_select_key not in st.session_state:
            st.session_state[layer_select_key] = layers_sorted[0]
        if dict_select_key not in st.session_state:
            st.session_state[dict_select_key] = cc_by_layer[layers_sorted[0]][0]["dictionary_name"]

        # Layer selector
        selected_layer = st.selectbox(
            "Select Layer",
            options=layers_sorted,
            index=layers_sorted.index(st.session_state[layer_select_key]),
            key=layer_select_key,
        )

        dicts_for_layer = [c["dictionary_name"] for c in cc_by_layer[selected_layer]]
        if st.session_state[dict_select_key] not in dicts_for_layer:
            st.session_state[dict_select_key] = dicts_for_layer[0]

        selected_dict_name = st.selectbox(
            "Select Trained CrossCoder",
            options=dicts_for_layer,
            index=dicts_for_layer.index(st.session_state[dict_select_key]),
            key=dict_select_key,
        )

        selected_cc_info = next(c for c in cc_by_layer[selected_layer] if c["dictionary_name"] == selected_dict_name)

        # Display CrossCoder information and wandb link if available
        training_metrics_path = selected_cc_info['path'] / "training_metrics.json"
        if training_metrics_path.exists():
            try:
                with open(training_metrics_path, 'r') as f:
                    training_metrics = json.load(f)
                
                # Display core CrossCoder information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    k = training_metrics.get('k')
                    st.metric("Top-K", k)
                with col2:
                    dict_size = training_metrics.get('dictionary_size')
                    st.metric("Dictionary Size", dict_size)
                with col3:
                    activation_dim = training_metrics.get('activation_dim')
                    expansion_factor = dict_size / activation_dim if activation_dim else "N/A"
                    st.metric("Expansion Factor", expansion_factor)
                with col4:
                    last_eval_logs = training_metrics.get('last_eval_logs', {})
                    fve = last_eval_logs.get('val/frac_variance_explained', "Not available")
                    st.metric("FVE", fve)
                
                # Display links in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    wandb_link = training_metrics.get('wandb_link')
                    if wandb_link:
                        st.markdown(f"**W&B Run:** [View training run]({wandb_link})")
                    else:
                        st.info("No W&B link available")

                with col2:
                    huggingface_link = training_metrics.get('hf_repo_id')
                    if huggingface_link:
                        col21, col22 = st.columns([0.2, 0.8])
                        with col21:
                            st.markdown(f"**HF Model:** [View model](https://huggingface.co/{huggingface_link})")
                        with col22:
                            st.code(huggingface_link, language=None)
                    else:
                        st.info("No HF link available")
            except Exception as e:
                st.warning(f"Could not load training metrics: {str(e)}")
        else:
            st.info("Training metrics not found")

        # Tabs
        multi_tab_interface(
            [
                ("📈 Latent Statistics", lambda: self._render_latent_statistics_tab(selected_cc_info)),
                ("📋 Steering Results", lambda: self._render_steering_results_tab(selected_cc_info)),
                ("🎯 Online Steering", lambda: CrosscoderSteeringDashboard(self, selected_cc_info).display()),
                ("🔥 Online Inference", lambda: CrosscoderOnlineDashboard(self, selected_cc_info).display()),
                ("🎨 Plots", lambda: self._render_plots_tab(selected_cc_info)),
                ("📊 MaxAct Examples", lambda: self._render_maxact_tab(selected_cc_info)),
            ],
            "CrossCoder Analysis",
        )

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available crosscoder results.

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
                sae_dir = organism_dir / "crosscoder"
                if sae_dir.exists() and any(sae_dir.iterdir()):
                    results[model_name][organism_name] = str(sae_dir)

        return results

    # --------------------------- Helper methods ---------------------------
    def _get_available_crosscoder_directories(self):
        """Return list of available trained crosscoder directories."""
        cc_base_dir = self.results_dir / "crosscoder"
        if not cc_base_dir.exists():
            return []

        available = []
        for layer_dir in cc_base_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            try:
                layer_num = int(layer_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            for cc_dir in layer_dir.iterdir():
                if not cc_dir.is_dir():
                    continue
                if (cc_dir / "dictionary_model").exists() or (cc_dir / "training_config.yaml").exists():
                    available.append({
                        "layer": layer_num,
                        "dictionary_name": cc_dir.name,
                        "path": cc_dir,
                    })
        available.sort(key=lambda x: (x["layer"], x["dictionary_name"]))
        return available

    def _load_latent_df(self, dictionary_name: str):
        """Load latent_df with caching."""
        from src.utils.dictionary.utils import load_latent_df
        if dictionary_name not in self.latent_df_cache:
            self.latent_df_cache[dictionary_name] = load_latent_df(dictionary_name)
        return self.latent_df_cache[dictionary_name]

    # --------------------------- Tab renderers ---------------------------
    def _render_maxact_tab(self, cc_info):
        dictionary_name = cc_info["dictionary_name"]
        layer = cc_info["layer"]
        model_results_dir = cc_info["path"]

        st.markdown(f"**Selected CrossCoder:** Layer {layer} – {dictionary_name}")

        latent_dir = model_results_dir / "latent_activations"
        if not latent_dir.exists():
            st.error(f"No latent activations directory found at {latent_dir}")
            return

        db_path = latent_dir / "examples.db"
        if not db_path.exists():
            st.error(f"No MaxAct example database found at {db_path}")
            return

        assert self.tokenizer is not None, "Tokenizer must be available for MaxAct visualization"
        store = ReadOnlyMaxActStore(db_path, tokenizer=self.tokenizer)
        component = MaxActivationDashboardComponent(store, title=f"CrossCoder Examples – Layer {layer}")
        component.display()

    def _render_latent_statistics_tab(self, cc_info):
        dictionary_name = cc_info["dictionary_name"]
        layer = cc_info["layer"]
        st.markdown(f"### Latent Statistics – Layer {layer}")
        st.markdown(f"**Dictionary:** {dictionary_name}")

        try:
            df = self._load_latent_df(dictionary_name)
        except Exception as e:
            st.error(f"Failed to load latent df: {e}")
            return

        filtered_df = df.copy()

        # --- categorical filters ---
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            st.markdown("**Categorical Filters**")
            cols = st.columns(min(3, len(cat_cols)))
            for i, col in enumerate(cat_cols):
                with cols[i % 3]:
                    options = [v for v in df[col].unique().tolist() if pd.notna(v)]
                    sel = st.multiselect(col, options=options, default=options, key=f"filter_cat_{col}")
                    if sel:
                        filtered_df = filtered_df[filtered_df[col].isin(sel)]

        # --- numeric filters ---
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if num_cols:
            with st.expander("Numeric Filters", expanded=False):
                cols = st.columns(min(3, len(num_cols)))
                for i, col in enumerate(num_cols):
                    with cols[i % 3]:
                        cmin, cmax = float(df[col].min()), float(df[col].max())
                        if cmin != cmax:
                            enable = st.checkbox(f"Filter {col}", value=False, key=f"enable_num_{col}")
                            if enable:
                                rng = st.slider(col, min_value=cmin, max_value=cmax, value=(cmin, cmax), key=f"slider_{col}")
                                if rng != (cmin, cmax):
                                    filtered_df = filtered_df[(filtered_df[col] >= rng[0]) & (filtered_df[col] <= rng[1])]

        st.markdown(f"**Showing {len(filtered_df)} / {len(df)} latents**")
        st.dataframe(filtered_df, use_container_width=True, height=400)

    def _render_plots_tab(self, cc_info):
        dictionary_name = cc_info["dictionary_name"]
        layer = cc_info["layer"]
        plots_dir = cc_info["path"] / "plots"
        if not plots_dir.exists():
            st.error(f"No plots directory found at {plots_dir}")
            return
        img_exts = [".png", ".jpg", ".jpeg", ".svg", ".pdf"]
        images = []
        for ext in img_exts:
            images.extend(plots_dir.glob(f"*{ext}"))
        if not images:
            st.error("No plot files found.")
            return
        st.markdown(f"### Plots – Layer {layer}")
        st.markdown(f"Found {len(images)} plot files")
        for img in images:
            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                st.image(str(img), use_container_width=True)
            elif img.suffix.lower() == ".svg":
                st.markdown(img.read_text(), unsafe_allow_html=True)
            elif img.suffix.lower() == ".pdf":
                with open(img, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                st.markdown(f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='400'></iframe>", unsafe_allow_html=True)

    def _render_steering_results_tab(self, selected_cc_info):
        """Render the Steering Results tab displaying saved experiment results."""
        
        dictionary_name = selected_cc_info['dictionary_name']
        layer = selected_cc_info['layer']
        model_results_dir = selected_cc_info['path']
        
        st.markdown(f"**Selected CrossCoder:** Layer {layer} - {dictionary_name}")
        
        # Display the steering results using the imported function
        display_steering_results(model_results_dir, self.cfg)

    # --------------------------- Activation computation ---------------------------
    @torch.no_grad()
    def compute_crosscoder_activations_for_tokens(self, dictionary_name: str, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int):
        """Compute crosscoder latent activations for a batch of tokens."""
        from nnsight import LanguageModel

        assert input_ids.shape == attention_mask.shape and input_ids.ndim == 2, "input_ids and attention_mask must be [B, T]"

        base_model = LanguageModel(self.base_model, tokenizer=self.tokenizer)
        ft_model = LanguageModel(self.finetuned_model, tokenizer=self.tokenizer)

        batch_base = place_inputs(input_ids, attention_mask, self.base_model)
        batch_ft = place_inputs(input_ids, attention_mask, self.finetuned_model)

        token_ids = input_ids[0].cpu().tolist()
        tokens = [self.tokenizer.decode([t]) for t in token_ids]

        with base_model.trace(batch_base):
            base_act = base_model.model.layers[layer].output[0].save()
        with ft_model.trace(batch_ft):
            ft_act = ft_model.model.layers[layer].output[0].save()

        base_act, ft_act = base_act.cpu(), ft_act.cpu()
        B, T, H = base_act.shape
        assert ft_act.shape == (B, T, H)

        # Use first sequence
        seq_base = base_act[0]
        seq_ft = ft_act[0]

        # Stack along new layer dimension -> [T, 2, H]
        stacked_seq = torch.stack([seq_base, seq_ft], dim=1)

        # Load crosscoder
        cc_model = load_dictionary_model(dictionary_name, is_sae=False)

        # Encode -> [T, dict_size]
        latent = cc_model.encode(stacked_seq)
        latent_np = latent.cpu().numpy()
        max_per_tok = latent_np.max(axis=1)
        stats = {
            "mean": float(max_per_tok.mean()),
            "std": float(max_per_tok.std()),
            "min": float(max_per_tok.min()),
            "max": float(max_per_tok.max()),
            "median": float(np.median(max_per_tok)),
        }
        return {
            "tokens": tokens,
            "latent_activations": latent_np,
            "max_activations_per_token": max_per_tok,
            "statistics": stats,
            "total_tokens": len(tokens),
            "layer": layer,
            "dict_size": cc_model.dict_size,
        }

# -----------------------------------------------------------------------------
# Dashboard classes
# -----------------------------------------------------------------------------

class CrosscoderOnlineDashboard(AbstractOnlineDiffingDashboard):
    """Online per-token latent activation dashboard for CrossCoders."""

    def __init__(self, method_instance: CrosscoderDiffingMethod, cc_info):
        super().__init__(method_instance)
        self.cc_info = cc_info

    def _render_streamlit_method_controls(self):
        latent_idx = st.number_input("Latent Index", min_value=0, value=0, step=1)
        return {"latent_idx": latent_idx}

    def compute_statistics_for_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        layer = self.cc_info["layer"]
        latent_idx = kwargs.get("latent_idx", 0)
        res = self.method.compute_crosscoder_activations_for_tokens(self.cc_info["dictionary_name"], input_ids, attention_mask, layer)
        seq_len, dict_size = res["latent_activations"].shape
        assert 0 <= latent_idx < dict_size
        values = res["latent_activations"][:, latent_idx]
        stats = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
        }
        return {
            "tokens": res["tokens"],
            "values": values,
            "statistics": stats,
            "total_tokens": res["total_tokens"],
            "analysis_title": f"Latent {latent_idx} Activation",
        }

    def get_method_specific_params(self):
        return {"latent_idx": 0}

    def _get_title(self):
        return "CrossCoder Analysis"

class CrosscoderSteeringDashboard(SteeringDashboard):
    """Latent steering dashboard for CrossCoders."""

    def __init__(self, method_instance: CrosscoderDiffingMethod, cc_info):
        super().__init__(method_instance)
        self.cc_info = cc_info
        self._layer = cc_info["layer"]
        self._cc_model = None
        try:
            latent_df = self.method._load_latent_df(self.cc_info['dictionary_name'])
            if 'max_act_validation' in latent_df.columns:
                self._max_acts = latent_df['max_act_validation']
            elif 'max_act_train' in latent_df.columns:
                self._max_acts = latent_df['max_act_train']
            else:
                raise KeyError(f"Neither 'max_act_validation' nor 'max_act_train' found in latent dataframe for {self.cc_info['dictionary_name']}")
        except Exception as e:
            st.error(f"❌ Maximum activations not yet collected for dictionary '{cc_info['dictionary_name']}'")
            st.info("💡 Please run the analysis pipeline to collect maximum activations before using the steering dashboard.")
            st.stop()

    @property
    def layer(self):
        return self._layer

    def _ensure_model(self):
        if self._cc_model is None:
            self._cc_model = load_dictionary_model(self.cc_info["dictionary_name"], is_sae=False).to(self.method.device)

    def get_dict_size(self):
        self._ensure_model()
        return self._cc_model.dict_size

    def get_latent(self, idx: int):
        self._ensure_model()
        assert 0 <= idx < self._cc_model.dict_size
        # Decoder weight shape [num_layers, dict_size, activation_dim] – we take mean across layers
        vec = self._cc_model.decoder.weight[:, idx, :].mean(dim=0)
        return vec.detach()

    def get_max_activation(self, latent_idx: int):
        if latent_idx in self._max_acts.index:
            return float(self._max_acts.loc[latent_idx])
        return "unknown"

    @st.fragment
    def _render_latent_selector(self):
        dict_size = self.get_dict_size()
        key = f"crosscoder_latent_idx_layer_{self.layer}"
        if key not in st.session_state:
            st.session_state[key] = 0
        st.session_state[key] = min(st.session_state[key], dict_size - 1)
        idx = st.number_input("Latent Index", 0, dict_size - 1, value=st.session_state[key], step=1, key=key)
        st.info(f"Max Activation: {self.get_max_activation(idx)}")
        return idx

    def _render_streamlit_method_controls(self):
        col1, col2 = st.columns(2)
        with col1:
            latent_idx = self._render_latent_selector()
        with col2:
            factor = st.slider("Steering Factor", -1000.0, 1000.0, 1.0, 0.1)
        mode = st.selectbox("Steering Mode", options=["prompt_only", "all_tokens"], index=1)
        return {"latent_idx": latent_idx, "steering_factor": factor, "steering_mode": mode}

    def _get_title(self):
        return f"CrossCoder Latent Steering – Layer {self.layer}"
