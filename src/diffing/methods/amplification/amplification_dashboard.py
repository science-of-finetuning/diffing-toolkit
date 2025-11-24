"""
Streamlit dashboard for weight difference amplification.

Provides UI for creating, editing, and testing amplification configurations.
"""

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import ClassVar
from typing import Dict, Any, List

import streamlit as st
import yaml

from src.utils.configs import (
    get_available_organisms,
    get_organism_variants,
    PROJECT_ROOT,
)
from src.utils.vllm import LLM, ensure_vllm, LoRARequest, SamplingParams
from src.utils.model import load_model_from_config, adapter_id_to_path
from src.diffing.methods.amplification.amplification_config import (
    AmplificationConfig,
    AmplifiedAdapter,
    LayerAmplification,
    ModuleAmplification,
    patch_vllm,
)
from src.diffing.methods.amplification.dashboard_state import ManagedConfig
from vllm.distributed import cleanup_dist_env_and_memory

CACHE_DIR = PROJECT_ROOT / ".streamlit_cache" / "amplification_cache"
CONFIGS_DIR = CACHE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR = CACHE_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_ADAPTERS_DIR = PROJECT_ROOT / ".compiled_adapters"


@st.cache_resource
def _get_vllm_server_container():
    """Global container for vLLM server shared across all sessions."""
    return {"server": None, "config": None}


class AmplificationDashboard:
    """Streamlit dashboard for amplification configuration."""

    _request_id_counter: ClassVar[int] = 0

    def __init__(self, method_instance):
        """
        Initialize dashboard.

        Args:
            method_instance: Instance of WeightDifferenceAmplification
        """
        self.method = method_instance
        self.inference_config = deepcopy(self.method.base_model_cfg)
        self.inference_config.vllm_kwargs = (
            self.inference_config.vllm_kwargs or {}
        ) | dict(
            max_num_seqs=16,
            enable_lora=True,
            max_loras=16,
            max_lora_rank=64,
        )
        patch_vllm()
        self._init_session_state()

    @staticmethod
    @st.cache_data
    def _get_adapter_rank_cached(adapter_id: str) -> int:
        """
        Get the rank of a LoRA adapter from its configuration.
        Results are cached to avoid repeated downloads.

        Args:
            adapter_id: HuggingFace adapter ID

        Returns:
            The rank (r) of the LoRA adapter
        """

        adapter_path = adapter_id_to_path(adapter_id)
        adapter_config_path = adapter_path / "adapter_config.json"
        assert (
            adapter_config_path.exists()
        ), f"adapter_config.json not found for {adapter_id}"
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        return adapter_config["r"]

    def auto_update_inference_config(self) -> None:
        """
        Update the inference configuration based on the Amplification Configurations:
        - max_num_seqs: set to a multiple of 8 that is greater than the number of configs
        - max_loras: set to the number of different configs
        - max_lora_rank: set to the maximum rank of the LORAs in the configurations
        """

        active_configs = [mc for mc in st.session_state.managed_configs if mc.active]
        num_configs = len(active_configs)

        max_num_seqs = max(((num_configs + 7) // 8) * 8, 8)
        max_loras = max(num_configs, 4)

        all_adapter_ids = set()
        base_model_name = self.method.base_model_cfg.name
        for mc in active_configs:
            for adapter in mc.config.amplified_adapters:
                all_adapter_ids.add(adapter.adapter_id(base_model_name))
        max_lora_rank = 64
        if all_adapter_ids:
            ranks = []
            for adapter_id in all_adapter_ids:
                ranks.append(self._get_adapter_rank_cached(adapter_id))
            max_lora_rank = max(ranks) * 2

        self.inference_config.vllm_kwargs["max_num_seqs"] = max_num_seqs
        self.inference_config.vllm_kwargs["max_loras"] = max_loras
        self.inference_config.vllm_kwargs["max_lora_rank"] = max_lora_rank

    def _shutdown_vllm_server(self) -> None:
        container = _get_vllm_server_container()
        if container["server"] is not None:
            del container["server"]
            cleanup_dist_env_and_memory()
            # container["server"].shutdown()  # async
            container["server"] = None
            container["config"] = None

    @property
    @ensure_vllm
    def multi_lora_vllm_server(self) -> LLM:
        """Get or create the vLLM server, reloading if config changed."""
        self.auto_update_inference_config()
        current_config = (
            dict(model_id=self.inference_config.model_id)
            | self.inference_config.vllm_kwargs
        )
        container = _get_vllm_server_container()
        need_reload = False
        if container["server"] is None:
            need_reload = True
            container["config"] = None
        elif container["config"] != current_config:
            need_reload = True
            self._shutdown_vllm_server()

        if need_reload:
            with st.spinner("Loading vLLM server..."):
                container["server"] = load_model_from_config(
                    self.inference_config, use_vllm=True, ignore_cache=True
                )
                container["config"] = current_config

        return container["server"]

    def _init_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if "managed_configs" not in st.session_state:
            st.session_state.managed_configs = []
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "active_conversation_id" not in st.session_state:
            st.session_state.active_conversation_id = None
        if "conversation_counter" not in st.session_state:
            st.session_state.conversation_counter = 0
        if "sampling_params" not in st.session_state:
            st.session_state.sampling_params = {}
        if "vllm_kwargs" not in st.session_state:
            st.session_state.vllm_kwargs = self.inference_config.vllm_kwargs
        if "multi_gen_results" not in st.session_state:
            st.session_state.multi_gen_results = None
        if "multi_gen_preset_prompt" not in st.session_state:
            st.session_state.multi_gen_preset_prompt = None
        if "multi_gen_preset_apply_template" not in st.session_state:
            st.session_state.multi_gen_preset_apply_template = None

        # Load last multi-gen state from cache
        saved_multigen_state = self._load_last_multigen_state()
        if "multi_gen_prompt" not in st.session_state:
            st.session_state.multi_gen_prompt = saved_multigen_state["prompt"]
        if "apply_chat_template_checkbox" not in st.session_state:
            st.session_state.apply_chat_template_checkbox = saved_multigen_state[
                "apply_chat_template"
            ]

        # Load configs from cache after initializing session state
        self._load_configs_from_cache()

        # Load conversations from cache after configs (conversations reference configs)
        self._load_conversations_from_cache()

    def _get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters from sidebar/session state."""
        params = deepcopy(st.session_state["sampling_params"])
        do_sample = params.pop("do_sample", True)
        if not do_sample:
            params["temperature"] = 0
        return SamplingParams(**params)

    def _get_multigen_cache_file(self) -> Path:
        """Get the cache file path for multi-generation state."""
        return CACHE_DIR / "last_multigen_state.yaml"

    def _save_configs_to_cache(self) -> None:
        """Save all managed configs to the cache directory."""
        # Get all current config names
        current_config_names = set()
        for mc in st.session_state.managed_configs:
            # Config names are already sanitized, so we can reuse them directly
            safe_name = mc.config.name
            config_path = CONFIGS_DIR / f"{safe_name}.yaml"
            mc.config.save_yaml(config_path)
            current_config_names.add(f"{safe_name}.yaml")

        removed_dir = CONFIGS_DIR / "removed"
        removed_dir.mkdir(parents=True, exist_ok=True)

        # Move configs that no longer exist into the removed directory
        for config_file in CONFIGS_DIR.glob("*.yaml"):
            if config_file.name not in current_config_names:
                target = removed_dir / config_file.name
                if target.exists():
                    target.unlink()
                config_file.replace(target)

    def _load_configs_from_cache(self) -> None:
        """Load configs from the cache directory."""

        # Only load if session state is empty
        if len(st.session_state.managed_configs) > 0:
            return

        for config_file in sorted(CONFIGS_DIR.glob("*.yaml")):
            try:
                loaded_config = AmplificationConfig.load_yaml(config_file)
                loaded_config.name = self._get_unique_config_name(loaded_config.name)
                managed_config = ManagedConfig.from_config(
                    loaded_config, active=True, expanded=False
                )
                st.session_state.managed_configs.append(managed_config)
            except Exception as e:
                st.error(f"Failed to load config from {config_file}: {e}")

    def _save_last_multigen_state(self, prompt: str, apply_chat_template: bool) -> None:
        """
        Save the last multi-generation prompt and settings to cache.

        Args:
            prompt: The prompt text
            apply_chat_template: Whether to apply chat template
        """
        state_file = self._get_multigen_cache_file()

        state = {
            "prompt": prompt,
            "apply_chat_template": apply_chat_template,
        }

        with open(state_file, "w") as f:
            yaml.dump(state, f, sort_keys=False)

    def _load_last_multigen_state(self) -> dict:
        """
        Load the last multi-generation prompt and settings from cache.

        Returns:
            Dictionary with 'prompt' and 'apply_chat_template' keys
        """
        state_file = self._get_multigen_cache_file()

        if not state_file.exists():
            return {"prompt": "", "apply_chat_template": True}

        try:
            with open(state_file) as f:
                state = yaml.safe_load(f)
            return state or {"prompt": "", "apply_chat_template": True}
        except Exception:
            return {"prompt": "", "apply_chat_template": True}

    def _save_conversation(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """
        Save a single conversation to disk.

        Args:
            conv_id: The conversation ID
            conv: The conversation data
        """
        safe_name = conv["name"].replace("/", "_").replace(":", "_")
        conv_path = CONVERSATIONS_DIR / f"{safe_name}.yaml"

        # Serialize conversation data
        serialized_conv = {
            "conv_id": conv_id,
            "name": conv["name"],
            "context": {
                "config_name": (
                    conv["context"]["config"].name
                    if conv["context"]["config"]
                    else None
                ),
                "compiled_path": (
                    str(conv["context"]["compiled_path"])
                    if conv["context"]["compiled_path"]
                    else None
                ),
            },
            "history": conv["history"],
            "editing_message": conv["editing_message"],
            "regenerating_from": conv["regenerating_from"],
        }

        with open(conv_path, "w") as f:
            yaml.dump(serialized_conv, f, sort_keys=False)

    def _load_conversations_from_cache(self) -> None:
        """Load all conversations from the cache directory."""
        if len(st.session_state.conversations) > 0:
            return

        config_name_to_managed = {
            mc.config.name: mc for mc in st.session_state.managed_configs
        }

        max_conv_num = -1
        for conv_file in sorted(CONVERSATIONS_DIR.glob("*.yaml")):
            try:
                with open(conv_file) as f:
                    serialized_conv = yaml.safe_load(f)

                conv_id = serialized_conv["conv_id"]

                # Extract conversation number from conv_id (e.g., "conv_5" -> 5)
                conv_num = int(conv_id.split("_")[-1])
                max_conv_num = max(max_conv_num, conv_num)

                # Reconstruct config reference
                config_name = serialized_conv["context"]["config_name"]
                if config_name and config_name in config_name_to_managed:
                    config = config_name_to_managed[config_name].config
                else:
                    config = None

                # Reconstruct compiled_path
                compiled_path_str = serialized_conv["context"]["compiled_path"]
                compiled_path = Path(compiled_path_str) if compiled_path_str else None

                # Reconstruct conversation
                conv = {
                    "name": serialized_conv["name"],
                    "context": {
                        "config": config,
                        "compiled_path": compiled_path,
                    },
                    "history": serialized_conv["history"],
                    "editing_message": serialized_conv["editing_message"],
                    "regenerating_from": serialized_conv["regenerating_from"],
                }

                st.session_state.conversations[conv_id] = conv

            except Exception as e:
                st.error(f"Failed to load conversation from {conv_file}: {e}")

        # Set conversation counter to max + 1
        if max_conv_num >= 0:
            st.session_state.conversation_counter = max_conv_num + 1

    def _get_unique_config_name(
        self, desired_name: str, exclude_idx: int = None
    ) -> str:
        """
        Get a unique configuration name by appending _X if name already exists.

        Args:
            desired_name: The desired configuration name
            exclude_idx: Optional index to exclude from duplicate check (for renames)

        Returns:
            Unique configuration name
        """
        sanitized_name = self._sanitize_config_name(desired_name)

        existing_names = set()
        for idx, mc in enumerate(st.session_state.managed_configs):
            if exclude_idx is None or idx != exclude_idx:
                existing_names.add(mc.config.name)
        if sanitized_name not in existing_names:
            return sanitized_name
        counter = 1
        while f"{sanitized_name}_{counter}" in existing_names:
            counter += 1
        return f"{sanitized_name}_{counter}"

    @staticmethod
    def _sanitize_config_name(name: str) -> str:
        """
        Sanitize a config name so it can be used as both display text and filename.

        Args:
            name: Desired config name input by the user

        Returns:
            Sanitized name containing only safe characters
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_\- ]+", "_", name).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized or "config"

    def _get_unique_conversation_name(
        self, desired_name: str, exclude_conv_id: str = None
    ) -> str:
        """
        Get a unique conversation name by appending _X if name already exists.

        Args:
            desired_name: The desired conversation name
            exclude_conv_id: Optional conv_id to exclude from duplicate check (for renames)

        Returns:
            Unique conversation name
        """
        existing_names = set()
        for conv_id, conv in st.session_state.conversations.items():
            if exclude_conv_id is None or conv_id != exclude_conv_id:
                existing_names.add(conv["name"])
        if desired_name not in existing_names:
            return desired_name
        counter = 1
        while f"{desired_name}_{counter}" in existing_names:
            counter += 1
        return f"{desired_name}_{counter}"

    def _clear_config_widget_states(self) -> None:
        """
        Clear all widget states related to config indices.

        This is necessary when configs are inserted/deleted to prevent
        Streamlit from showing stale cached values at the wrong indices.
        """
        keys_to_delete = []
        for key in st.session_state.keys():
            # Clear any widget key that contains config/adapter/layer/module indices
            # These follow patterns like: config_name_{idx}, organism_{config_idx}_{adapter_idx}, etc.
            if any(
                pattern in key
                for pattern in [
                    "config_name_",
                    "config_desc_",
                    "config_active_",
                    "delete_config_",
                    "add_adapter_",
                    "organism_",
                    "variant_",
                    "delete_adapter_",
                    "add_layer_",
                    "delete_layer_",
                    "layer_mode_",
                    "layer_single_",
                    "layer_range_",
                    "layer_list_",
                    "add_module_",
                    "delete_module_",
                    "module_mode_",
                    "module_weight_",
                ]
            ):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del st.session_state[key]

    def _save_and_rerun(self) -> None:
        """
        Save configs to cache, clear widget states, and trigger a Streamlit rerun.

        Clearing widget states is essential to prevent index-based widgets from
        showing stale values after list modifications (insert/delete operations).
        """
        self._save_configs_to_cache()
        self._clear_config_widget_states()
        st.rerun()

    def _compile_config(self, config: ManagedConfig) -> Path | None:
        """Compile a config and return path to compiled adapter."""
        print(
            f"Compiling config {config.config.name}\nBase name: {self.method.base_model_cfg.name}"
        )
        path = config.compile(
            COMPILED_ADAPTERS_DIR,
            base_model_name=self.method.base_model_cfg.name,
            base_model=self.method.base_model,
        )
        print(f"Compiled config {config.config.name} to {path}")
        return path

    def _truncate_history_and_get_prompt(self, conv: Dict[str, Any], index: int) -> str:
        """Truncate chat history after a message and return the prompt for regeneration."""
        assert 0 <= index < len(conv["history"]), f"Invalid message index: {index}"

        # Find the last user message before this assistant message
        prompt_index = index - 1
        while prompt_index >= 0 and conv["history"][prompt_index]["role"] != "user":
            prompt_index -= 1

        assert prompt_index >= 0, "No user message found before this assistant message"

        # Truncate history after the user prompt
        conv["history"] = conv["history"][: prompt_index + 1]

        # Format the conversation up to this point
        return self._format_chat_prompt(conv["history"])

    def _multi_gen_request(
        self,
        prompt,
        amplification_configs: List[ManagedConfig],
        sampling_params,
    ):
        # Convert dict sampling params to vLLM SamplingParams object
        if isinstance(sampling_params, dict):
            vllm_sampling_params = SamplingParams(
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 0.9),
                max_tokens=sampling_params.get("max_tokens", 100),
            )
        else:
            vllm_sampling_params = sampling_params

        results = []
        for config in amplification_configs:
            compiled_path = self._compile_config(config)
            if compiled_path is None:
                lreq = None
            else:
                lreq = LoRARequest(
                    config.config.name,
                    config.lora_int_id,
                    str(compiled_path),
                )
            print(f"{lreq=}")

            # Generate synchronously
            outputs = self.multi_lora_vllm_server.generate(
                prompts=[prompt],
                sampling_params=vllm_sampling_params,
                lora_request=lreq,
            )
            print(f"{outputs=}")

            # Extract the generated text
            final_text = outputs[0].outputs[0].text

            results.append(
                {
                    "config": config,
                    "compiled_path": compiled_path,
                    "result": final_text,
                }
            )

        return results

    def _single_gen_request(
        self,
        prompt: str,
        config: ManagedConfig,
        compiled_path: Path | None,
        sampling_params,
    ) -> str:
        """
        Generate text for a single configuration using vLLM.

        Args:
            prompt: Input prompt text
            config: ManagedConfig to use for generation
            compiled_path: Path to compiled adapter (or None for base model)
            sampling_params: vLLM SamplingParams object or dict

        Returns:
            Generated text
        """
        if isinstance(sampling_params, dict):
            vllm_sampling_params = SamplingParams(
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                max_tokens=sampling_params["max_tokens"],
                seed=sampling_params["seed"],
            )
        else:
            vllm_sampling_params = sampling_params

        if compiled_path is None:
            lreq = None
        else:
            lreq = LoRARequest(
                config.config.name,
                config.lora_int_id,
                str(compiled_path),
            )

        outputs = self.multi_lora_vllm_server.generate(
            prompts=[prompt],
            sampling_params=vllm_sampling_params,
            lora_request=lreq,
        )

        return outputs[0].outputs[0].text

    def _format_chat_prompt(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history into a single prompt."""
        return self.method.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )

    def display(self) -> None:
        """Main entry point for dashboard."""
        st.title("Weight Difference Amplification Dashboard")

        # Add CSS for hover effects on chat message buttons
        st.markdown(
            """
        <style>
        /* Hide buttons in chat messages by default */
        [data-testid="stChatMessage"] .stButton button[kind="secondary"] {
            opacity: 0;
            transition: opacity 0.2s;
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
            min-height: 0;
        }
        /* Show buttons on message hover */
        [data-testid="stChatMessage"]:hover .stButton button[kind="secondary"] {
            opacity: 1;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        self._render_sidebar()

        tab1, tab2, tab3 = st.tabs(["Amplifications", "Multi-Generation", "Chat"])

        with tab1:
            self._render_amplifications_tab()
        with tab2:
            self._render_multi_generation_tab()
        with tab3:
            self._render_chat_tab()

    def _render_sidebar(self) -> None:
        """Render sidebar with global controls."""
        st.sidebar.header("Model Info")
        st.sidebar.info(f"**Model:** {self.method.base_model_cfg.model_id}")

        st.sidebar.header("vLLM Configuration")
        if st.sidebar.button("Shutdown Engine", use_container_width=True):
            self._shutdown_vllm_server()
            st.sidebar.success("Shutdown signal sent to engine.")
        st.sidebar.info("TODO")

        # max_num_seqs = st.sidebar.number_input(
        #     "Max Number of Sequences",
        #     min_value=1,
        #     max_value=256,
        #     value=st.session_state.vllm_kwargs["max_num_seqs"],
        #     step=8,
        #     help="Maximum number of sequences that the vLLM server can process in parallel",
        #     key="max_num_seqs",
        # )
        # max_loras

        st.sidebar.header("Sampling Parameters")

        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Sampling temperature for generation",
        )
        top_p = st.sidebar.slider(
            "Top-p (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling probability threshold",
        )
        max_tokens = st.sidebar.slider(
            "Max New Tokens",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Maximum number of tokens to generate",
        )
        do_sample = st.sidebar.checkbox(
            "Use Sampling",
            value=True,
            help="Enable sampling (if disabled, uses greedy decoding)",
        )
        seed = st.sidebar.number_input(
            "Seed",
            min_value=0,
            value=28,
            step=9,
            help="Seed for random number generation",
        )
        st.session_state.sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "seed": seed,
        }

        st.sidebar.header("Global Controls")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✓ Enable All", use_container_width=True):
                for mc in st.session_state.managed_configs:
                    mc.active = True
                self._save_and_rerun()

        with col2:
            if st.button("✗ Disable All", use_container_width=True):
                for mc in st.session_state.managed_configs:
                    mc.active = False
                self._save_and_rerun()

    def _render_amplifications_tab(self) -> None:
        """Render Tab 1: Amplification configuration UI."""
        st.markdown("## Amplification Configurations")
        st.markdown(
            "Create and manage amplification configurations for adapter weight modification."
        )

        if st.button("➕ New Amplification", use_container_width=True):
            base_name = f"Config {len(st.session_state.managed_configs) + 1}"
            unique_name = self._get_unique_config_name(base_name)
            new_config = AmplificationConfig(
                name=unique_name,
                description="",
                amplified_adapters=[],
            )
            new_managed = ManagedConfig.from_config(
                new_config, active=True, expanded=True
            )
            st.session_state.managed_configs.insert(0, new_managed)
            self._save_and_rerun()

        st.markdown("---")

        if len(st.session_state.managed_configs) == 0:
            st.info(
                "No amplification configurations yet. Click 'New Amplification' to create one."
            )
        else:
            for idx, mc in enumerate(st.session_state.managed_configs):
                self._render_amplification_config(idx, mc)

    def _render_multi_generation_tab(self) -> None:
        """Render Tab 2: Multi-generation interface."""
        st.markdown("## Multi-Generation")
        st.markdown(
            "Generate text with multiple amplification configurations side-by-side."
        )

        # Apply preset values from chat tab BEFORE rendering widgets
        if st.session_state.multi_gen_preset_prompt is not None:
            st.session_state.multi_gen_prompt = st.session_state.multi_gen_preset_prompt
            st.session_state.multi_gen_preset_prompt = None

        if st.session_state.multi_gen_preset_apply_template is not None:
            st.session_state.apply_chat_template_checkbox = (
                st.session_state.multi_gen_preset_apply_template
            )
            st.session_state.multi_gen_preset_apply_template = None

        prompt = st.text_area(
            "Prompt",
            height=150,
            placeholder="Enter your prompt here...",
            key="multi_gen_prompt",
        )

        apply_chat_template = st.checkbox(
            "Apply chat template",
            value=True,
            key="apply_chat_template_checkbox",
            help="Apply the model's chat template to format the prompt",
        )

        active_configs = [mc for mc in st.session_state.managed_configs if mc.active]

        if len(active_configs) == 0:
            st.warning(
                "No active amplification configurations. Go to the Amplifications tab to create and activate configs."
            )
        else:
            st.info(
                f"Will generate with {len(active_configs)} active configuration(s): {', '.join(c.config.name for c in active_configs)}"
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            generate_clicked = st.button(
                "🚀 Generate", type="primary", use_container_width=True
            )
        with col2:
            if st.button(
                "🗑️ Clear Results", disabled=st.session_state.multi_gen_results is None
            ):
                st.session_state.multi_gen_results = None
                self._save_and_rerun()

        if generate_clicked:
            # Save the prompt and settings to cache
            self._save_last_multigen_state(prompt, apply_chat_template)

            sampling_params = self._get_sampling_params()

            # Apply chat template if checkbox is checked
            final_prompt = prompt
            if apply_chat_template:
                final_prompt = self._format_chat_prompt(
                    [{"role": "user", "content": prompt}]
                )

            # Run generation (synchronous)
            with st.spinner(
                f"Generating with {len(active_configs)} configuration(s)..."
            ):
                results = self._multi_gen_request(
                    prompt=final_prompt,
                    amplification_configs=active_configs,
                    sampling_params=sampling_params,
                )

            st.session_state.multi_gen_results = {
                "prompt": prompt,
                "final_prompt": final_prompt,
                "results": results,
            }
            st.rerun()

        # Display results if they exist
        if st.session_state.multi_gen_results is not None:
            st.markdown("---")
            st.markdown("## Generated Outputs")

            results_data = st.session_state.multi_gen_results

            # Use 2-column layout for outputs
            output_cols = st.columns(2)

            for idx, result_data in enumerate(results_data["results"]):
                # Alternate between columns
                col_idx = idx % 2

                formatted_title = f"({idx + 1}) {result_data['config'].name}"

                with output_cols[col_idx]:
                    with st.expander(formatted_title, expanded=True):
                        # Display the generated text with proper formatting
                        st.markdown(
                            f"### {result_data['config'].name}\n\n"
                            + result_data["result"].replace("\n", "  \n"),
                            unsafe_allow_html=False,
                        )

                        st.markdown("---")

                        # Action buttons
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            if st.button(
                                "➕ Continue",
                                key=f"continue_{idx}",
                                use_container_width=True,
                            ):
                                sampling_params = self._get_sampling_params()
                                continuation_prompt = (
                                    results_data["final_prompt"] + result_data["result"]
                                )

                                with st.spinner("Continuing generation..."):
                                    continuation_results = self._multi_gen_request(
                                        prompt=continuation_prompt,
                                        amplification_configs=[result_data["config"]],
                                        sampling_params=sampling_params,
                                    )

                                result_data["result"] += continuation_results[0][
                                    "result"
                                ]
                                st.rerun()

                        with col2:
                            if st.button(
                                "🔄 Regenerate",
                                key=f"regenerate_{idx}",
                                use_container_width=True,
                            ):
                                sampling_params = self._get_sampling_params()

                                with st.spinner("Regenerating..."):
                                    new_results = self._multi_gen_request(
                                        prompt=results_data["final_prompt"],
                                        amplification_configs=[result_data["config"]],
                                        sampling_params=sampling_params,
                                    )

                                result_data["result"] = new_results[0]["result"]
                                st.rerun()

                        with col3:
                            if st.button(
                                "💬 Continue Chat",
                                key=f"continue_chat_{idx}",
                                use_container_width=True,
                            ):
                                conv_id = (
                                    f"conv_{st.session_state.conversation_counter}"
                                )
                                st.session_state.conversation_counter += 1

                                conv_name = self._get_unique_conversation_name(
                                    f"{result_data['config'].name}"
                                )
                                st.session_state.conversations[conv_id] = {
                                    "name": conv_name,
                                    "context": {
                                        "config": result_data["config"],
                                        "compiled_path": result_data["compiled_path"],
                                    },
                                    "history": [
                                        {
                                            "role": "user",
                                            "content": results_data["prompt"],
                                        },
                                        {
                                            "role": "assistant",
                                            "content": result_data["result"],
                                            "config_name": result_data["config"].name,
                                        },
                                    ],
                                    "editing_message": None,
                                    "regenerating_from": None,
                                }
                                self._save_conversation(
                                    conv_id, st.session_state.conversations[conv_id]
                                )
                                st.session_state.active_conversation_id = conv_id
                                st.success(
                                    f"✓ Chat started with {result_data['config'].name}. Now switch to the Chat tab to continue."
                                )
                                self._save_and_rerun()

                        with col4:
                            st.download_button(
                                label="📥 Download",
                                data=result_data["result"],
                                file_name=f"{result_data['config'].name.replace(' ', '_')}.txt",
                                mime="text/plain",
                                key=f"download_{idx}",
                                use_container_width=True,
                            )

    def _render_chat_tab(self) -> None:
        """Render Tab 3: Chat interface with multiple conversations."""
        if not st.session_state.conversations:
            st.info(
                "💬 No conversations yet. Generate from the Multi-Generation tab and click 'Continue Chat', or use the 'New' tab to start an empty chat."
            )

        if st.button("➕ Start New Chat", type="primary"):
            self._create_new_conversation()
            self._save_and_rerun()
            return

        # Create tabs for conversations + New tab
        conv_items = list(st.session_state.conversations.items())
        tab_names = [conv["name"] for _, conv in conv_items] + ["➕ New"]
        tabs = st.tabs(tab_names)

        # Render each conversation in its tab
        for tab, (conv_id, conv) in zip(tabs[:-1], conv_items):
            with tab:
                self._render_single_conversation(conv_id, conv)

        # Render New tab
        with tabs[-1]:
            self._render_new_conversation_tab()

    def _create_new_conversation(
        self, config=None, compiled_path=None, name=None
    ) -> str:
        """Create a new empty conversation and return its ID."""
        conv_id = f"conv_{st.session_state.conversation_counter}"
        st.session_state.conversation_counter += 1

        # Use first active config if no config provided
        if config is None:
            active_mcs = [mc for mc in st.session_state.managed_configs if mc.active]
            config = (
                active_mcs[0].config
                if active_mcs
                else (
                    st.session_state.managed_configs[0].config
                    if st.session_state.managed_configs
                    else None
                )
            )

        if config and compiled_path is None:
            compiled_path = self._compile_config(config)

        conv_name = self._get_unique_conversation_name(
            name or f"New Chat {st.session_state.conversation_counter}"
        )

        st.session_state.conversations[conv_id] = {
            "name": conv_name,
            "context": {
                "config": config,
                "compiled_path": compiled_path,
            },
            "history": [],
            "editing_message": None,
            "regenerating_from": None,
        }
        self._save_conversation(conv_id, st.session_state.conversations[conv_id])
        st.session_state.active_conversation_id = conv_id
        return conv_id

    def _render_new_conversation_tab(self) -> None:
        """Render the 'New' tab for starting new conversations."""
        st.markdown("### Start a New Conversation")

        col1, col2 = st.columns([3, 1])

        with col1:
            conv_name = st.text_input(
                "Conversation Name",
                value=f"New Chat {st.session_state.conversation_counter + 1}",
                key="new_conv_name",
            )

        with col2:
            config_names = [mc.config.name for mc in st.session_state.managed_configs]
            if config_names:
                selected_config_name = st.selectbox(
                    "Configuration",
                    options=config_names,
                    key="new_conv_config",
                )
            else:
                st.warning("No configs available")
                selected_config_name = None

        if st.button(
            "✨ Create Conversation", type="primary", use_container_width=True
        ):
            if selected_config_name:
                config = next(
                    mc.config
                    for mc in st.session_state.managed_configs
                    if mc.config.name == selected_config_name
                )
                self._create_new_conversation(config=config, name=conv_name)
                st.success(f"Created conversation: {conv_name}")
                self._save_and_rerun()
            else:
                st.error("Please create an amplification configuration first")

    def _render_single_conversation(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """Render a single conversation."""
        config = conv["context"]["config"]
        compiled_path = conv["context"]["compiled_path"]

        # Handle regeneration first (before layout)
        if conv["regenerating_from"] is not None:
            regen_index = conv["regenerating_from"]
            conv["regenerating_from"] = None

            prompt = self._truncate_history_and_get_prompt(conv, regen_index)

            sampling_params = self._get_sampling_params()

            # Get the managed config for this conversation
            managed_config = next(
                mc
                for mc in st.session_state.managed_configs
                if mc.config.name == config.name
            )

            config_label = f"[{config.name}]" if config else "[No Config]"
            with st.chat_message("assistant"):
                st.write(f"**{config_label}**")
                with st.spinner("Regenerating..."):
                    response = self._single_gen_request(
                        prompt=prompt,
                        config=managed_config,
                        compiled_path=compiled_path,
                        sampling_params=sampling_params,
                    )
                st.markdown(response)

            if response:
                conv["history"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "config_name": config.name if config else "No Config",
                    }
                )
                self._save_conversation(conv_id, conv)

            self._save_and_rerun()

        # Conversation controls
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            new_name = st.text_input(
                "Conversation Name",
                value=conv["name"],
                key=f"conv_name_{conv_id}",
            )
            if new_name != conv["name"]:
                # Delete old conversation file
                old_safe_name = conv["name"].replace("/", "_").replace(":", "_")
                old_conv_path = CONVERSATIONS_DIR / f"{old_safe_name}.yaml"
                if old_conv_path.exists():
                    old_conv_path.unlink()

                # Update to unique name and save
                unique_name = self._get_unique_conversation_name(
                    new_name, exclude_conv_id=conv_id
                )
                conv["name"] = unique_name
                self._save_conversation(conv_id, conv)

        with col2:
            if config:
                config_names = [
                    mc.config.name for mc in st.session_state.managed_configs
                ]
                if config_names:
                    current_index = next(
                        (
                            i
                            for i, mc in enumerate(st.session_state.managed_configs)
                            if mc.config.name == config.name
                        ),
                        0,
                    )

                    selected_config_name = st.selectbox(
                        "Config",
                        options=config_names,
                        index=current_index,
                        key=f"conv_config_{conv_id}",
                    )

                    if selected_config_name != config.name:
                        new_config = next(
                            mc.config
                            for mc in st.session_state.managed_configs
                            if mc.config.name == selected_config_name
                        )

                        with st.spinner(f"Switching to {selected_config_name}..."):
                            new_compiled_path = self._compile_config(new_config)

                        conv["context"]["config"] = new_config
                        conv["context"]["compiled_path"] = new_compiled_path
                        self._save_conversation(conv_id, conv)
                        st.success(f"Switched to {selected_config_name}")
                        self._save_and_rerun()
            else:
                st.info("No config")

        with col3:
            if st.button(
                "🗑️ Delete", key=f"delete_conv_{conv_id}", use_container_width=True
            ):
                # Delete conversation file
                safe_name = conv["name"].replace("/", "_").replace(":", "_")
                conv_path = CONVERSATIONS_DIR / f"{safe_name}.yaml"
                if conv_path.exists():
                    conv_path.unlink()

                # Delete from session state
                del st.session_state.conversations[conv_id]
                if st.session_state.active_conversation_id == conv_id:
                    st.session_state.active_conversation_id = None
                self._save_and_rerun()

        st.markdown("---")

        # Display chat history with edit/regenerate functionality
        for i, msg in enumerate(conv["history"]):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    if conv["editing_message"] == i:
                        # Edit mode
                        edited_content = st.text_area(
                            "Edit message",
                            value=msg["content"],
                            key=f"edit_user_{conv_id}_{i}",
                            label_visibility="collapsed",
                        )
                        bcol1, bcol2 = st.columns([1, 1])
                        with bcol1:
                            if st.button(
                                "Save", key=f"save_user_{conv_id}_{i}", type="primary"
                            ):
                                conv["history"][i]["content"] = edited_content
                                conv["editing_message"] = None
                                self._save_conversation(conv_id, conv)
                                self._save_and_rerun()
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_user_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                self._save_and_rerun()
                    else:
                        # Display mode
                        st.markdown(msg["content"])
                        _, btn_col = st.columns([10, 1])
                        with btn_col:
                            if st.button(
                                "✏️",
                                key=f"edit_btn_user_{conv_id}_{i}",
                                help="Edit message",
                                type="secondary",
                            ):
                                conv["editing_message"] = i
                                self._save_and_rerun()
            else:
                # Assistant message
                with st.chat_message("assistant"):
                    if conv["editing_message"] == i:
                        # Edit mode
                        edited_content = st.text_area(
                            "Edit message",
                            value=msg["content"],
                            key=f"edit_asst_{conv_id}_{i}",
                            label_visibility="collapsed",
                        )
                        bcol1, bcol2 = st.columns([1, 1])
                        with bcol1:
                            if st.button(
                                "Save", key=f"save_asst_{conv_id}_{i}", type="primary"
                            ):
                                conv["history"][i]["content"] = edited_content
                                conv["editing_message"] = None
                                self._save_conversation(conv_id, conv)
                                self._save_and_rerun()
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_asst_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                self._save_and_rerun()
                    else:
                        # Display mode - use stored config name from message
                        config_label = f"[{msg.get('config_name', config.name if config else 'No Config')}]"
                        st.markdown(f"**{config_label}** {msg['content']}")
                        _, btn_col1, btn_col2 = st.columns([10, 1, 1])
                        with btn_col1:
                            if st.button(
                                "✏️",
                                key=f"edit_btn_asst_{conv_id}_{i}",
                                help="Edit message",
                                type="secondary",
                            ):
                                conv["editing_message"] = i
                                self._save_and_rerun()
                        with btn_col2:
                            if st.button(
                                "🔄",
                                key=f"regen_btn_asst_{conv_id}_{i}",
                                help="Regenerate from here",
                                type="secondary",
                            ):
                                conv["regenerating_from"] = i
                                self._save_and_rerun()

        # Multi-Generation mode checkbox
        send_to_multi_gen = st.checkbox(
            "🚀 Send next message to Multi-Generation",
            key=f"multi_gen_mode_{conv_id}",
            help="When checked, your next message will be sent to Multi-Generation instead of this chat",
        )

        # Chat input at the bottom
        user_input = st.chat_input(
            "Type your message here...", key=f"chat_input_{conv_id}"
        )

        if user_input:
            if send_to_multi_gen:
                # Multi-Generation mode: send conversation + message to Multi-Gen tab
                history_for_multi_gen = conv["history"].copy()
                history_for_multi_gen.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                # Format the conversation with chat template
                formatted_prompt = self._format_chat_prompt(history_for_multi_gen)
                # Set preset values that will be applied when multi-gen tab renders
                st.session_state.multi_gen_preset_prompt = formatted_prompt
                st.session_state.multi_gen_preset_apply_template = False
                st.success(
                    "✓ Conversation sent to Multi-Generation tab. Switch to the Multi-Generation tab to continue. (Uncheck the box above to return to normal chat mode)"
                )
                self._save_and_rerun()
            else:
                # Normal chat mode
                # Add user message to history
                conv["history"].append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Format full conversation for generation
                full_prompt = self._format_chat_prompt(conv["history"])

                sampling_params = self._get_sampling_params()

                # Get the managed config for this conversation
                managed_config = next(
                    mc
                    for mc in st.session_state.managed_configs
                    if mc.config.name == config.name
                )

                # Generate and display assistant response
                config_label = f"[{config.name}]" if config else "[No Config]"
                with st.chat_message("assistant"):
                    st.write(f"**{config_label}**")
                    with st.spinner("Generating..."):
                        response = self._single_gen_request(
                            prompt=full_prompt,
                            config=managed_config,
                            compiled_path=compiled_path,
                            sampling_params=sampling_params,
                        )
                    st.markdown(response)

                # Add assistant response to history
                conv["history"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "config_name": config.name if config else "No Config",
                    }
                )
                self._save_conversation(conv_id, conv)

                self._save_and_rerun()

    def _render_amplification_config(self, idx: int, mc: ManagedConfig) -> None:
        """Render one amplification config (expandable)."""
        config = mc.config
        icon = "✅" if mc.active else "❌"
        config_key = mc.lora_int_id
        with st.expander(f"{icon} {config.name}", expanded=mc.expanded):
            col1, col2 = st.columns([3, 1])

            with col1:
                new_name = st.text_input(
                    "Configuration Name",
                    value=config.name,
                    key=f"config_name_{config_key}",
                )
                if new_name != config.name:
                    unique_name = self._get_unique_config_name(
                        new_name, exclude_idx=idx
                    )
                    st.session_state.managed_configs[idx].config.name = unique_name
                    self._save_and_rerun()

            with col2:
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_config_{config_key}",
                    use_container_width=True,
                ):
                    st.session_state.managed_configs.pop(idx)
                    self._save_and_rerun()

            config.description = st.text_area(
                "Description",
                value=config.description,
                key=f"config_desc_{config_key}",
                height=60,
            )

            current_active = mc.active
            mc.active = st.checkbox(
                "Active",
                value=mc.active,
                key=f"config_active_{config_key}",
                help="Only active configurations will be used for generation",
            )
            if mc.active != current_active:
                self._save_and_rerun()

            st.markdown("#### Adapters")

            if len(config.amplified_adapters) == 0:
                st.info("No adapters configured. Click 'Add Adapter' below.")
            else:
                for adapter_idx, adapter in enumerate(config.amplified_adapters):
                    self._render_adapter_amplification(
                        idx, config_key, adapter_idx, adapter
                    )

            if st.button("➕ Add Adapter", key=f"add_adapter_{config_key}"):
                # Get first available organism as default
                available_organisms = get_available_organisms(
                    base_model_name=self.method.base_model_cfg.name, only_loras=True
                )
                default_organism = available_organisms[0] if available_organisms else ""

                new_adapter = AmplifiedAdapter(
                    organism_name=default_organism,
                    variant="default",
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
                self._save_and_rerun()

    def _render_adapter_amplification(
        self,
        config_idx: int,
        config_key: int,
        adapter_idx: int,
        adapter: AmplifiedAdapter,
    ) -> None:
        """Render one adapter's amplifications."""
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                display_name = (
                    f"{adapter.organism_name} ({adapter.variant})"
                    if adapter.organism_name
                    else "New Adapter"
                )
                st.markdown(f"**Adapter: {display_name}**")

            with col2:
                if st.button("🗑️", key=f"delete_adapter_{config_key}_{adapter_idx}"):
                    st.session_state.managed_configs[
                        config_idx
                    ].config.amplified_adapters.pop(adapter_idx)
                    self._save_and_rerun()

            # Get base model name from the method instance
            base_model_name = self.method.base_model_cfg.name

            col1, col2 = st.columns(2)

            with col1:
                # Organism selector
                available_organisms = get_available_organisms(
                    base_model_name=self.method.base_model_cfg.name, only_loras=True
                )

                if not available_organisms:
                    st.warning("No organisms found in configs_new/organism/")
                    adapter.organism_name = ""
                else:
                    # Find current index
                    try:
                        current_index = (
                            available_organisms.index(adapter.organism_name)
                            if adapter.organism_name in available_organisms
                            else 0
                        )
                    except ValueError:
                        current_index = 0

                    selected_organism = st.selectbox(
                        "Organism",
                        options=available_organisms,
                        index=current_index,
                        key=f"organism_{config_key}_{adapter_idx}",
                        help="Select the organism (model fine-tune) to use",
                    )

                    if selected_organism != adapter.organism_name:
                        adapter.organism_name = selected_organism
                        # Reset variant to default when organism changes
                        adapter.variant = "default"
                        self._save_and_rerun()

            with col2:
                # Variant selector (based on selected organism and base model)
                if adapter.organism_name:
                    available_variants = get_organism_variants(
                        adapter.organism_name, base_model_name, only_loras=True
                    )

                    if not available_variants:
                        st.warning(
                            f"No variants available for {adapter.organism_name} with base model {base_model_name}"
                        )
                        adapter.variant = "default"
                    else:
                        # Find current index
                        try:
                            current_index = (
                                available_variants.index(adapter.variant)
                                if adapter.variant in available_variants
                                else 0
                            )
                        except ValueError:
                            current_index = 0

                        adapter.variant = st.selectbox(
                            "Variant",
                            options=available_variants,
                            index=current_index,
                            key=f"variant_{config_key}_{adapter_idx}",
                            help="Select the variant of the organism",
                        )
                else:
                    st.info("Select an organism first")

            st.markdown("**Layer Specifications**")

            if len(adapter.layer_amplifications) == 0:
                st.info("No layer specifications. Click 'Add Layer Spec' below.")
            else:
                for layer_idx, layer_amp in enumerate(adapter.layer_amplifications):
                    self._render_layer_amplification(
                        config_idx, config_key, adapter_idx, layer_idx, layer_amp
                    )

            if st.button(
                "➕ Add Layer Spec", key=f"add_layer_{config_key}_{adapter_idx}"
            ):
                new_layer_amp = LayerAmplification(
                    layers="all",
                    module_amplifications=[],
                )
                adapter.layer_amplifications.append(new_layer_amp)
                self._save_and_rerun()

    def _render_layer_amplification(
        self,
        config_idx: int,
        config_key: int,
        adapter_idx: int,
        layer_idx: int,
        layer_amp: LayerAmplification,
    ) -> None:
        """Render layer amplification specification."""
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"**Layer Specification {layer_idx + 1}**")

            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_layer_{config_key}_{adapter_idx}_{layer_idx}",
                ):
                    st.session_state.managed_configs[
                        config_idx
                    ].config.amplified_adapters[adapter_idx].layer_amplifications.pop(
                        layer_idx
                    )
                    self._save_and_rerun()

            # Determine initial radio index based on persisted state
            if isinstance(layer_amp.layers, list):
                # Check if it's a continuous range
                if len(layer_amp.layers) > 1 and layer_amp.layers == list(
                    range(layer_amp.layers[0], layer_amp.layers[-1] + 1)
                ):
                    initial_mode_index = 3  # "Range"
                else:
                    initial_mode_index = 2  # "List"
            elif isinstance(layer_amp.layers, int):
                initial_mode_index = 1  # "Single"
            else:  # "all"
                initial_mode_index = 0  # "All"

            layer_mode = st.radio(
                "Layer Selection Mode",
                options=["All", "Single", "List", "Range"],
                index=initial_mode_index,
                key=f"layer_mode_{config_key}_{adapter_idx}_{layer_idx}",
                horizontal=True,
            )

            if layer_mode == "All":
                layer_amp.layers = "all"
                st.info("Applies to all layers in the model")

            elif layer_mode == "Single":
                current_val = (
                    layer_amp.layers if isinstance(layer_amp.layers, int) else 0
                )
                layer_num = st.number_input(
                    "Layer Index",
                    min_value=0,
                    value=current_val,
                    step=1,
                    key=f"layer_single_{config_key}_{adapter_idx}_{layer_idx}",
                )
                layer_amp.layers = layer_num

            elif layer_mode == "Range":
                # Get num_layers from the model
                num_layers = self.method.base_model.num_layers

                # Determine current range values
                if isinstance(layer_amp.layers, list) and len(layer_amp.layers) > 0:
                    current_start = layer_amp.layers[0]
                    current_end = layer_amp.layers[-1]
                else:
                    current_start = 0
                    current_end = num_layers - 1

                layer_range = st.slider(
                    "Layer Range (inclusive)",
                    min_value=0,
                    max_value=num_layers - 1,
                    value=(current_start, current_end),
                    key=f"layer_range_{config_key}_{adapter_idx}_{layer_idx}",
                    help="Select the range of layers to apply amplification to",
                )

                range_start, range_end = layer_range
                layer_amp.layers = list(range(range_start, range_end + 1))
                st.info(
                    f"Applies to layers {range_start} through {range_end} ({range_end - range_start + 1} layers)"
                )

            else:  # List
                current_val = (
                    ",".join(map(str, layer_amp.layers))
                    if isinstance(layer_amp.layers, list)
                    else ""
                )
                layer_list_str = st.text_input(
                    "Layer Indices (comma-separated)",
                    value=current_val,
                    key=f"layer_list_{config_key}_{adapter_idx}_{layer_idx}",
                    help="E.g., '0,1,2,5,10'",
                )
                if layer_list_str.strip():
                    layer_amp.layers = [
                        int(x.strip()) for x in layer_list_str.split(",")
                    ]
                else:
                    layer_amp.layers = []

            st.markdown("**Module Specifications**")

            if len(layer_amp.module_amplifications) == 0:
                st.info("No module specifications. Click 'Add Module' below.")
            else:
                for module_idx, module_amp in enumerate(
                    layer_amp.module_amplifications
                ):
                    self._render_module_amplification(
                        config_idx,
                        config_key,
                        adapter_idx,
                        layer_idx,
                        module_idx,
                        module_amp,
                    )

            if st.button(
                "➕ Add Module",
                key=f"add_module_{config_key}_{adapter_idx}_{layer_idx}",
            ):
                new_module_amp = ModuleAmplification(modules="all", weight=1.0)
                layer_amp.module_amplifications.append(new_module_amp)
                self._save_and_rerun()

    def _render_module_amplification(
        self,
        config_idx: int,
        config_key: int,
        adapter_idx: int,
        layer_idx: int,
        module_idx: int,
        module_amp: ModuleAmplification,
    ) -> None:
        """Render module amplification (module selector + weight slider)."""
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            module_amp.modules = st.selectbox(
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
                key=f"module_mode_{config_key}_{adapter_idx}_{layer_idx}_{module_idx}",
            )

        with col2:
            module_amp.weight = st.slider(
                "Weight",
                min_value=-5.0,
                max_value=5.0,
                value=float(module_amp.weight),
                step=0.1,
                key=f"module_weight_{config_key}_{adapter_idx}_{layer_idx}_{module_idx}",
                help="Amplification factor (1.0 = no change, 2.0 = double, 0.5 = half)",
            )

        with col3:
            if st.button(
                "🗑️",
                key=f"delete_module_{config_key}_{adapter_idx}_{layer_idx}_{module_idx}",
            ):
                st.session_state.managed_configs[config_idx].config.amplified_adapters[
                    adapter_idx
                ].layer_amplifications[layer_idx].module_amplifications.pop(module_idx)
                self._save_and_rerun()
