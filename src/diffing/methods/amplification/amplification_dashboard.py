"""
Streamlit dashboard for weight difference amplification.

Provides UI for creating, editing, and testing amplification configurations.
"""

from copy import deepcopy
import html
import re
import uuid
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import streamlit.components.v1 as components

from src.utils.configs import (
    get_available_organisms,
    get_organism_variants,
    PROJECT_ROOT,
)
from src.utils.vllm import (
    LLM,
    ensure_vllm,
    SamplingParams,
    cleanup_dist_env_and_memory,
    kill_vllm_process,
)
from src.utils.model import load_model_from_config, get_adapter_rank
from src.diffing.methods.amplification.amplification_config import (
    AmplificationConfig,
    AmplifiedAdapter,
    LayerAmplification,
    ModuleAmplification,
    patch_vllm,
    CUSTOM_ADAPTER_ORGANISM,
)
from src.diffing.methods.amplification.dashboard_state import (
    ManagedConfig,
    ManagedPrompt,
    sanitize_config_name,
    get_unique_name,
    save_configs_to_cache,
    save_configs_to_folder,
    load_configs_from_folder,
    list_all_folders,
    create_folder,
    unload_folder_configs,
    save_prompts_to_cache,
    save_prompts_to_folder,
    load_prompts_from_folder,
    list_all_prompt_folders,
    unload_folder_prompts,
    save_multigen_state,
    load_multigen_state,
    save_loaded_folders,
    load_loaded_folders,
    save_conversation,
    load_conversations_from_cache,
    delete_conversation_file,
    GenerationLog,
)
from src.diffing.methods.amplification.weight_amplification import (
    WeightDifferenceAmplification,
)

CACHE_DIR = PROJECT_ROOT / ".streamlit_cache" / "amplification_cache"
CONFIGS_DIR = CACHE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR = CACHE_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = CACHE_DIR / "generation_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR = CACHE_DIR / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_ADAPTERS_DIR = PROJECT_ROOT / ".compiled_adapters"

# Load sample cycler component files
COMPONENTS_DIR = Path(__file__).parent / "components"
_SAMPLE_CYCLER_JS = (COMPONENTS_DIR / "sample_cycler.js").read_text()
_SAMPLE_CYCLER_CSS = (COMPONENTS_DIR / "sample_cycler.css").read_text()
_SAMPLE_CYCLER_HTML = (COMPONENTS_DIR / "sample_cycler.html").read_text()


def render_sample_cycler(
    samples: list[str], component_id: str, height: int = 400
) -> None:
    """Render an HTML component for cycling through samples with instant JS navigation."""
    samples_html = "\n".join(
        f'<div class="sample-content" style="display: {"block" if i == 0 else "none"}">{html.escape(s)}</div>'
        for i, s in enumerate(samples)
    )

    rendered = _SAMPLE_CYCLER_HTML
    rendered = rendered.replace("{{CSS}}", _SAMPLE_CYCLER_CSS)
    rendered = rendered.replace("{{JS}}", _SAMPLE_CYCLER_JS)
    rendered = rendered.replace("{{ID}}", component_id)
    rendered = rendered.replace("{{TOTAL}}", str(len(samples)))
    rendered = rendered.replace("{{SAMPLES}}", samples_html)

    if len(samples) > 1:
        rendered = rendered.replace("{{#if MULTI}}", "").replace("{{/if}}", "")
    else:
        rendered = re.sub(
            r"\{\{#if MULTI\}\}.*?\{\{/if\}\}", "", rendered, flags=re.DOTALL
        )

    components.html(rendered, height=height, scrolling=True)


@st.cache_resource
def _get_vllm_server_container():
    """Global container for vLLM server shared across all sessions."""
    return {"server": None, "config": None}


class AmplificationDashboard:
    """Streamlit dashboard for amplification configuration."""

    def __init__(self, method_instance: WeightDifferenceAmplification):
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
        """Cached wrapper around method.get_adapter_rank for Streamlit."""
        return get_adapter_rank(adapter_id)

    def _auto_update_inference_config(self) -> None:
        """Update inference config based on active amplification configurations."""
        active_configs = [
            mc for mc in st.session_state.managed_configs.values() if mc.active
        ]
        num_configs = len(active_configs)

        # max_num_seqs = max(((num_configs + 7) // 8) * 8, 8)
        max_num_seqs = 16  # TODO? dynamic doens't make sense atm as we don't use async with differernt loras per request
        max_loras = max(num_configs, 16)

        all_adapter_ids = set()
        base_model_name = self.method.base_model_cfg.name
        for mc in active_configs:
            for adapter in mc.config.amplified_adapters:
                try:
                    all_adapter_ids.add(adapter.adapter_id(base_model_name))
                except ValueError as e:
                    raise ValueError(f"Error getting adapter ID for {mc.name}") from e
        max_lora_rank = 128
        if all_adapter_ids:
            ranks = [self._get_adapter_rank_cached(aid) for aid in all_adapter_ids]
            max_lora_rank = max(ranks) * 2

        self.inference_config.vllm_kwargs["max_num_seqs"] = max_num_seqs
        self.inference_config.vllm_kwargs["max_loras"] = max_loras
        self.inference_config.vllm_kwargs["max_lora_rank"] = max_lora_rank

    def _shutdown_vllm_server(self) -> bool:
        """Shutdown the vLLM server.

        Returns:
            True if a process was killed, False otherwise.
        """
        container = _get_vllm_server_container()
        if container["server"] is not None:
            del container["server"]
            cleanup_dist_env_and_memory()
            container["server"] = None
            container["config"] = None
        return kill_vllm_process()

    @property
    def tokenizer(self):
        """Get the tokenizer from the method instance."""
        return self.method.tokenizer

    @property
    @ensure_vllm
    def multi_lora_vllm_server(self) -> LLM:
        """Get or create the vLLM server, reloading if config changed."""
        self._auto_update_inference_config()
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
            diff_dict = {
                f"{p}: {container['config'][p]} -> {current_config[p]}"
                for p in current_config
                if container["config"][p] != current_config[p]
            }
            st.warning(
                f"vLLM server configuration changed, reloading... Parameters that differ in the new configuration are:\n{diff_dict}"
            )
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
            st.session_state.managed_configs = {}

        # Load folder state from disk
        if "loaded_folders" not in st.session_state:
            loaded_folders, loaded_prompt_folders = load_loaded_folders(
                CACHE_DIR / "loaded_folders.yaml"
            )
            st.session_state.loaded_folders = loaded_folders
            st.session_state.loaded_prompt_folders = loaded_prompt_folders
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
        if "multi_gen_sample_indices" not in st.session_state:
            st.session_state.multi_gen_sample_indices = {}
        if "multi_gen_preset_prompt" not in st.session_state:
            st.session_state.multi_gen_preset_prompt = None
        if "multi_gen_preset_apply_template" not in st.session_state:
            st.session_state.multi_gen_preset_apply_template = None
        if "multi_gen_preset_messages" not in st.session_state:
            st.session_state.multi_gen_preset_messages = None

        saved_multigen_state = load_multigen_state(
            CACHE_DIR / "last_multigen_state.yaml"
        )

        if "multi_gen_text_prompt" not in st.session_state:
            st.session_state.multi_gen_text_prompt = saved_multigen_state.get(
                "text_tab", {}
            ).get("prompt", "")
        if "multi_gen_template_mode" not in st.session_state:
            st.session_state.multi_gen_template_mode = saved_multigen_state.get(
                "text_tab", {}
            ).get("template_mode", "Apply chat template")
        if "multi_gen_assistant_prefill" not in st.session_state:
            st.session_state.multi_gen_assistant_prefill = saved_multigen_state.get(
                "text_tab", {}
            ).get("assistant_prefill", "")

        if "multi_gen_messages" not in st.session_state:
            st.session_state.multi_gen_messages = saved_multigen_state.get(
                "messages_tab", {}
            ).get("messages", [])
        if "msg_builder_template_override" not in st.session_state:
            st.session_state.msg_builder_template_override = saved_multigen_state.get(
                "messages_tab", {}
            ).get("template_override", "No template override")
        if "multi_gen_msg_editing_idx" not in st.session_state:
            st.session_state.multi_gen_msg_editing_idx = None

        if "multi_gen_active_tab" not in st.session_state:
            st.session_state.multi_gen_active_tab = saved_multigen_state.get(
                "active_tab", "Text"
            )

        # Multi-prompt generation state
        if "managed_prompts" not in st.session_state:
            st.session_state.managed_prompts = {}
        if "multi_prompt_results" not in st.session_state:
            st.session_state.multi_prompt_results = None
        if "multi_prompt_display_configs" not in st.session_state:
            st.session_state.multi_prompt_display_configs = []

        if "multi_gen_prompt" not in st.session_state:
            st.session_state.multi_gen_prompt = saved_multigen_state.get("prompt", "")
        if "apply_chat_template_checkbox" not in st.session_state:
            st.session_state.apply_chat_template_checkbox = saved_multigen_state.get(
                "apply_chat_template", True
            )

        self._load_configs_from_cache()
        self._load_prompts_from_cache()
        self._load_conversations_from_cache()

    def _get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters from sidebar/session state."""
        params = deepcopy(st.session_state["sampling_params"])
        do_sample = params.pop("do_sample", True)
        if not do_sample:
            params["temperature"] = 0
        return SamplingParams(**params)

    def _load_configs_from_cache(self) -> None:
        """Load configs from all loaded folders."""
        if len(st.session_state.managed_configs) > 0:
            return

        existing_names = set()
        for folder in st.session_state.loaded_folders:
            loaded = load_configs_from_folder(CONFIGS_DIR, folder, existing_names)
            st.session_state.managed_configs.update(loaded)
            existing_names.update(mc.config.name for mc in loaded.values())

    def _load_prompts_from_cache(self) -> None:
        """Load prompts from all loaded folders."""
        if len(st.session_state.managed_prompts) > 0:
            return

        for folder in st.session_state.loaded_prompt_folders:
            loaded = load_prompts_from_folder(PROMPTS_DIR, folder)
            st.session_state.managed_prompts.update(loaded)

    def _save_prompts(self) -> None:
        """Save prompts to cache without triggering rerun."""
        save_prompts_to_cache(st.session_state.managed_prompts, PROMPTS_DIR)

    def _save_loaded_folders(self) -> None:
        """Save loaded folders state to disk."""
        save_loaded_folders(
            CACHE_DIR / "loaded_folders.yaml",
            st.session_state.loaded_folders,
            st.session_state.loaded_prompt_folders,
        )

    def _save_last_multigen_state(self) -> None:
        """Save current multi-gen state to cache."""
        state = {
            "active_tab": st.session_state.get("multi_gen_active_tab", "Text"),
            "text_tab": {
                "prompt": st.session_state.get("multi_gen_text_prompt", ""),
                "template_mode": st.session_state.get(
                    "multi_gen_template_mode", "Apply chat template"
                ),
                "assistant_prefill": st.session_state.get(
                    "multi_gen_assistant_prefill", ""
                ),
            },
            "messages_tab": {
                "messages": st.session_state.get("multi_gen_messages", []),
                "template_override": st.session_state.get(
                    "msg_builder_template_override", "No template override"
                ),
            },
        }
        save_multigen_state(CACHE_DIR / "last_multigen_state.yaml", state)

    def _save_conversation(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """Save a single conversation to disk."""
        save_conversation(conv_id, conv, CONVERSATIONS_DIR)

    def _load_conversations_from_cache(self) -> None:
        """Load all conversations from the cache directory."""
        if len(st.session_state.conversations) > 0:
            return

        config_name_to_managed = {
            mc.config.name: mc for mc in st.session_state.managed_configs.values()
        }
        conversations, max_conv_num = load_conversations_from_cache(
            CONVERSATIONS_DIR, config_name_to_managed
        )
        st.session_state.conversations.update(conversations)
        if max_conv_num >= 0:
            st.session_state.conversation_counter = max_conv_num + 1

    def _get_unique_config_name(
        self, desired_name: str, exclude_config_id: str = None
    ) -> str:
        """Get a unique configuration name."""
        sanitized_name = sanitize_config_name(desired_name)
        existing_names = set()
        for config_id, mc in st.session_state.managed_configs.items():
            if exclude_config_id is None or config_id != exclude_config_id:
                existing_names.add(mc.config.name)
        return get_unique_name(sanitized_name, existing_names)

    def _get_unique_conversation_name(
        self, desired_name: str, exclude_conv_id: str = None
    ) -> str:
        """Get a unique conversation name."""
        existing_names = set()
        for conv_id, conv in st.session_state.conversations.items():
            if exclude_conv_id is None or conv_id != exclude_conv_id:
                existing_names.add(conv["name"])
        return get_unique_name(desired_name, existing_names)

    def _save_configs(self) -> None:
        """Save configs to cache without triggering rerun."""
        save_configs_to_cache(st.session_state.managed_configs, CONFIGS_DIR)

    def _save_and_rerun(self, scope: str = "app") -> None:
        """Save configs to cache and trigger a Streamlit rerun.

        Args:
            scope: Rerun scope - "app" for full page, "fragment" for current fragment only.
        """
        self._save_configs()
        st.rerun(scope=scope)

    def _get_messages_with_system_prompt(
        self, conv: Dict[str, Any], messages: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get messages list with system prompt prepended if set."""
        if messages is None:
            messages = conv["history"]
        system_prompt = conv["context"].get("system_prompt", "").strip()
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + messages
        return messages

    def _truncate_history_and_get_prompt(
        self, conv: Dict[str, Any], index: int
    ) -> list[int]:
        """Truncate chat history after a message and return the prompt for regeneration."""
        assert 0 <= index < len(conv["history"]), f"Invalid message index: {index}"

        prompt_index = index - 1
        while prompt_index >= 0 and conv["history"][prompt_index]["role"] != "user":
            prompt_index -= 1

        assert prompt_index >= 0, "No user message found before this assistant message"

        conv["history"] = conv["history"][: prompt_index + 1]

        messages = self._get_messages_with_system_prompt(conv)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

    def _multi_gen_request(
        self,
        prompt: list[int],
        amplification_configs: List[ManagedConfig],
        sampling_params,
    ):
        """Generate with multiple configs using the method's generator."""
        yield from self.method.multi_gen_request(
            prompt=prompt,
            amplification_configs=amplification_configs,
            sampling_params=sampling_params,
            compiled_adapters_dir=COMPILED_ADAPTERS_DIR,
            vllm_server=self.multi_lora_vllm_server,
        )

    def _render_chat_sample_selection(
        self,
        conv_id: str,
        conv: Dict[str, Any],
        samples: List[str],
        config_name: str,
        mode: str = "add",  # "add", "replace", "continue"
        target_index: int = None,
    ) -> None:
        """Render sample selection UI for chat multi-gen."""
        mode_labels = {
            "add": "Select a response",
            "replace": "Select a regenerated response",
            "continue": "Select a continuation",
        }
        st.markdown(f"### {mode_labels.get(mode, 'Select')} ({len(samples)} samples)")

        # Cancel button
        if st.button("❌ Cancel selection", key=f"cancel_selection_{conv_id}"):
            pending_key = f"chat_pending_samples_{conv_id}"
            if pending_key in st.session_state:
                del st.session_state[pending_key]
            st.rerun(scope="fragment")

        # For continue mode, get the original content to show context
        original_content = ""
        if mode == "continue" and target_index is not None:
            original_content = conv["history"][target_index]["content"]

        cols = st.columns(2)
        for idx, sample in enumerate(samples):
            col_idx = idx % 2
            with cols[col_idx]:
                with st.expander(f"Sample {idx + 1}", expanded=True):
                    if mode == "continue":
                        st.text(original_content + sample)
                    else:
                        st.text(sample)
                    if st.button(
                        "✓ Select this",
                        key=f"select_sample_{conv_id}_{idx}",
                        type="primary",
                        use_container_width=True,
                    ):
                        if mode == "add":
                            conv["history"].append(
                                {
                                    "role": "assistant",
                                    "content": sample,
                                    "config_name": config_name,
                                }
                            )
                        elif mode == "replace":
                            conv["history"].append(
                                {
                                    "role": "assistant",
                                    "content": sample,
                                    "config_name": config_name,
                                }
                            )
                        elif mode == "continue" and target_index is not None:
                            conv["history"][target_index]["content"] += sample

                        # Clear pending samples
                        pending_key = f"chat_pending_samples_{conv_id}"
                        if pending_key in st.session_state:
                            del st.session_state[pending_key]

                        self._save_conversation(conv_id, conv)
                        self._save_and_rerun(scope="fragment")

    def display(self) -> None:
        """Main entry point for dashboard."""
        st.title("Weight Difference Amplification Dashboard")

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

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Amplifications", "Multi-Generation", "Chat", "Multi-Prompt"]
        )

        with tab1:
            self._render_amplifications_tab()
        with tab2:
            self._render_multi_generation_tab()
        with tab3:
            self._render_chat_tab()
        with tab4:
            self._render_multi_prompt_tab()

    def _render_sidebar(self) -> None:
        """Render sidebar with global controls."""

        with st.sidebar.expander("vLLM Configuration", expanded=True):
            st.info(f"**Model:** {self.method.base_model_cfg.model_id}")
            if st.button("Shutdown vLLM Engine", use_container_width=True):
                killed = self._shutdown_vllm_server()
                if killed:
                    st.success("vLLM process killed.")
                else:
                    st.info("No vLLM process was running.")
            # st.info("TODO: vLLM engine args")
            st.success("If your vllm server crashes, try to press the shutdown button!")

            # max_num_seqs = st.number_input(
            #     "Max Number of Sequences",
            #     min_value=1,
            #     max_value=256,
            #     value=st.session_state.vllm_kwargs["max_num_seqs"],
            #     step=8,
            #     help="Maximum number of sequences that the vLLM server can process in parallel",
            #     key="max_num_seqs",
            # )
            # max_loras

        with st.sidebar.expander("Sampling Parameters", expanded=True):
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Sampling temperature for generation",
            )
            top_p = st.slider(
                "Top-p (nucleus sampling)",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Nucleus sampling probability threshold",
            )
            max_tokens = st.slider(
                "Max New Tokens",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum number of tokens to generate",
            )
            num_samples = st.slider(
                "Num Samples",
                min_value=1,
                max_value=16,
                value=6,
                step=1,
                help="Number of completions to generate per config (for cycling through)",
            )
            do_sample = st.checkbox(
                "Use Sampling",
                value=True,
                help="Enable sampling (if disabled, uses greedy decoding)",
            )
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=28,
                step=9,
                help="Seed for random number generation",
            )
            skip_special_tokens = st.checkbox(
                "Skip Special Tokens",
                value=False,
                help="Skip special tokens in the generated text",
            )
            st.session_state.sampling_params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": num_samples,
                "do_sample": do_sample,
                "seed": seed,
                "skip_special_tokens": skip_special_tokens,
            }

        with st.sidebar.expander("Global Controls", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Enable All", use_container_width=True):
                    for config_id, mc in st.session_state.managed_configs.items():
                        mc.active = True
                        st.session_state[f"config_active_{config_id}"] = True
                    self._save_and_rerun()

            with col2:
                if st.button("✗ Disable All", use_container_width=True):
                    for config_id, mc in st.session_state.managed_configs.items():
                        mc.active = False
                        st.session_state[f"config_active_{config_id}"] = False
                    self._save_and_rerun()

        self._render_sidebar_quick_edit()

    def _render_sidebar_quick_edit(self) -> None:
        """Render quick config editor in sidebar."""
        with st.sidebar.expander("Quick Config Edit", expanded=False):
            # Get active configs only
            active_configs = {
                cid: mc
                for cid, mc in st.session_state.managed_configs.items()
                if mc.active
            }

            if not active_configs:
                st.info("No active configs. Enable configs in the Amplification tab.")
                return

            # Build options: name -> config_id mapping
            config_options = {mc.config.name: cid for cid, mc in active_configs.items()}
            option_names = list(config_options.keys())

            # Initialize sidebar selection state if needed
            if "sidebar_quick_edit_config_id" not in st.session_state:
                st.session_state.sidebar_quick_edit_config_id = None

            # Get current selection name (if valid)
            current_id = st.session_state.sidebar_quick_edit_config_id
            current_name = None
            if current_id and current_id in active_configs:
                current_name = active_configs[current_id].config.name

            # Determine initial index
            if current_name and current_name in option_names:
                initial_index = option_names.index(current_name)
            else:
                initial_index = 0

            def on_config_select():
                selected_name = st.session_state["sidebar_config_selector"]
                st.session_state.sidebar_quick_edit_config_id = config_options.get(
                    selected_name
                )

            selected_name = st.selectbox(
                "Select Config",
                options=option_names,
                index=initial_index,
                key="sidebar_config_selector",
                on_change=on_config_select,
            )

            # Render the selected config editor with sidebar_ prefix
            selected_id = config_options.get(selected_name)
            if selected_id and selected_id in st.session_state.managed_configs:
                mc = st.session_state.managed_configs[selected_id]
                self._render_amplification_config(
                    selected_id, mc, key_prefix="sidebar_", sidebar_mode=True
                )

    def _render_text_input_tab(self) -> None:
        """Render legacy text input interface."""
        prompt = st.text_area(
            "Prompt",
            height=150,
            placeholder="Enter your prompt here...",
            key="multi_gen_text_prompt",
        )

        template_mode = st.selectbox(
            "Template mode",
            ["No template", "Apply chat template", "Apply loom template"],
            key="multi_gen_template_mode",
            help="Choose how to format the prompt before sending to model",
        )

        system_prompt = ""
        assistant_prefill = ""
        loom_filename = "untitled.txt"

        if template_mode == "Apply chat template":
            system_prompt = st.text_input(
                "System prompt",
                key="multi_gen_system_prompt",
                placeholder="Optional: system instructions...",
            )
            assistant_prefill = st.text_input(
                "Assistant prefill",
                key="multi_gen_assistant_prefill",
                placeholder="Optional: prefill the assistant's response...",
                help="If not empty, this text will be added as the beginning of the assistant's response",
            )
        elif template_mode == "Apply loom template":
            loom_filename = st.text_input(
                "Filename",
                value="untitled.txt",
                key="multi_gen_loom_filename",
                placeholder="untitled.txt",
            )

        st.session_state.multi_gen_current_prompt = prompt
        st.session_state.multi_gen_current_template_mode = template_mode
        st.session_state.multi_gen_current_system_prompt = system_prompt
        st.session_state.multi_gen_current_assistant_prefill = assistant_prefill
        st.session_state.multi_gen_current_loom_filename = loom_filename

    def _render_import_conversations_section(self) -> None:
        """Render conversation import UI."""
        conversations = st.session_state.get("conversations", {})

        if not conversations:
            st.info(
                "No conversations available to import. Create one in the Chat tab first."
            )
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            conv_options = {
                conv["name"]: conv_id for conv_id, conv in conversations.items()
            }
            selected_conv = st.selectbox(
                f"Import from Chat ({len(conversations)} available)",
                options=list(conv_options.keys()),
                key="import_conv_selection",
            )

        with col2:
            if st.button("Import", key="import_conv_btn", use_container_width=True):
                conv_id = conv_options[selected_conv]
                conversation = conversations[conv_id]

                st.session_state.multi_gen_messages = [
                    {k: v for k, v in msg.items() if k in ["role", "content"]}
                    for msg in conversation["history"]
                ]
                st.rerun(scope="fragment")

    @st.fragment
    def _render_message_list_and_add(self) -> None:
        """Render message list and add form. Fragment for fast interactions."""
        messages = st.session_state.get("multi_gen_messages", [])
        editing_idx = st.session_state.get("multi_gen_msg_editing_idx", None)

        if not messages:
            st.info("No messages yet. Add your first message below.")
        else:
            st.markdown("**Conversation:**")

            for idx, msg in enumerate(messages):
                if editing_idx == idx:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        new_role = st.selectbox(
                            "Role",
                            ["user", "assistant", "system"],
                            index=["user", "assistant", "system"].index(msg["role"]),
                            key=f"edit_role_{idx}",
                        )
                    with col2:
                        new_content = st.text_area(
                            "Content",
                            value=msg["content"],
                            height=100,
                            key=f"edit_content_{idx}",
                        )

                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("💾 Save", key=f"save_{idx}"):
                            messages[idx] = {"role": new_role, "content": new_content}
                            st.session_state.multi_gen_msg_editing_idx = None
                            st.rerun(scope="fragment")
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_{idx}"):
                            st.session_state.multi_gen_msg_editing_idx = None
                            st.rerun(scope="fragment")
                else:
                    with st.container(border=True):
                        role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}
                        role_color = {
                            "user": "blue",
                            "assistant": "green",
                            "system": "gray",
                        }

                        st.markdown(
                            f":{role_color[msg['role']]}[**{role_emoji[msg['role']]} {msg['role'].title()}**]"
                        )

                        st.text(msg["content"])

                        col1, col2, col3 = st.columns([1, 1, 10])
                        with col1:
                            if st.button("✏️", key=f"edit_btn_{idx}"):
                                st.session_state.multi_gen_msg_editing_idx = idx
                                st.rerun(scope="fragment")
                        with col2:
                            if st.button("🗑️", key=f"delete_btn_{idx}"):
                                messages.pop(idx)
                                st.rerun(scope="fragment")

        st.markdown("---")
        st.markdown("**Add Message:**")

        with st.form("add_message_form", clear_on_submit=True):
            col1, col2 = st.columns([1, 4])

            with col1:
                role = st.selectbox(
                    "Role",
                    ["user", "assistant", "system"],
                )

            with col2:
                content = st.text_area(
                    "Content",
                    height=100,
                    placeholder="Enter message content...",
                )

            submitted = st.form_submit_button("➕ Add Message")
            if submitted:
                if content.strip():
                    if "multi_gen_messages" not in st.session_state:
                        st.session_state.multi_gen_messages = []

                    st.session_state.multi_gen_messages.append(
                        {
                            "role": role,
                            "content": content.strip(),
                        }
                    )
                    st.rerun(scope="fragment")
                else:
                    st.warning("Message content cannot be empty")

    @st.fragment
    def _render_message_builder_tab(self) -> None:
        """Render structured message builder interface. Fragment for isolated updates."""
        self._render_import_conversations_section()

        st.markdown("---")

        self._render_message_list_and_add()

        st.markdown("---")

        st.selectbox(
            "Template override",
            [
                "No template override",
                "Force generation prompt",
                "Force continue final message",
                "Force send as is",
            ],
            key="msg_builder_template_override",
            help=(
                "No template override: Smart default (continue if last message is assistant, else add generation prompt)\n"
                "Force generation prompt: Always add generation tokens\n"
                "Force continue final message: Always continue from last message (completion mode)\n"
                "Force send as is: Send raw formatted prompt without special tokens"
            ),
        )

    def _render_amplifications_tab(self) -> None:
        """Render Tab 1: Amplification configuration UI."""
        st.markdown("## Amplification Configurations")
        st.markdown(
            "Create and manage amplification configurations for adapter weight modification."
        )

        self._render_folder_loader()

        st.markdown("---")

        if len(st.session_state.loaded_folders) == 0:
            st.info("No folders loaded. Select a folder above to load configurations.")
        else:
            for folder in sorted(st.session_state.loaded_folders):
                self._render_folder_section(folder)

    def _render_folder_loader(self) -> None:
        """Render the folder loader UI (dropdown + Load/Create buttons)."""
        all_folders = list_all_folders(CONFIGS_DIR)
        loaded = st.session_state.loaded_folders
        available_to_load = [f for f in all_folders if f not in loaded]

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            folder_display = {f: "Root" if f == "" else f for f in available_to_load}
            if available_to_load:
                selected_folder = st.selectbox(
                    "Available Folders",
                    options=available_to_load,
                    format_func=lambda x: folder_display.get(x, x),
                    key="folder_to_load",
                )
            else:
                st.info("All folders are loaded")
                selected_folder = None

        with col2:
            if st.button(
                "📂 Load",
                disabled=selected_folder is None,
                use_container_width=True,
            ):
                st.session_state.loaded_folders.add(selected_folder)
                existing_names = {
                    mc.config.name for mc in st.session_state.managed_configs.values()
                }
                loaded_configs = load_configs_from_folder(
                    CONFIGS_DIR, selected_folder, existing_names
                )
                st.session_state.managed_configs.update(loaded_configs)
                self._save_loaded_folders()
                self._save_and_rerun()

        with col3:
            if st.button("➕ Create", use_container_width=True):
                st.session_state.show_create_folder_dialog = True

        if st.session_state.get("show_create_folder_dialog", False):
            self._render_create_folder_dialog()

    def _render_create_folder_dialog(self) -> None:
        """Render the create folder dialog."""
        with st.container(border=True):
            st.markdown("**Create New Folder**")
            new_folder_path = st.text_input(
                "Folder path",
                placeholder="e.g., experiments/v2",
                key="new_folder_path",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create", type="primary", use_container_width=True):
                    if new_folder_path:
                        create_folder(CONFIGS_DIR, new_folder_path)
                        st.session_state.loaded_folders.add(new_folder_path)
                        st.session_state.show_create_folder_dialog = False
                        self._save_loaded_folders()
                        self._save_and_rerun()
                    else:
                        st.error("Please enter a folder path")
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_create_folder_dialog = False
                    st.rerun()

    @st.fragment
    def _render_folder_section(self, folder: str) -> None:
        """Render a single folder section with its configs. Fragment for isolated updates."""
        folder_display = "Root" if folder == "" else folder
        folder_configs = {
            cid: mc
            for cid, mc in st.session_state.managed_configs.items()
            if mc.folder == folder
        }
        config_count = len(folder_configs)

        with st.expander(
            f"📁 {folder_display} ({config_count} configs)", expanded=True
        ):
            col1, col2 = st.columns([4, 1])

            with col1:
                if st.button(
                    "➕ New Amplification",
                    key=f"new_config_{folder}",
                    use_container_width=True,
                ):
                    base_name = f"Config {len(st.session_state.managed_configs) + 1}"
                    unique_name = self._get_unique_config_name(base_name)
                    new_config = AmplificationConfig(
                        name=unique_name,
                        description="",
                        amplified_adapters=[],
                    )
                    new_managed = ManagedConfig.from_config(
                        new_config, active=True, expanded=True, folder=folder
                    )
                    st.session_state.managed_configs[new_managed.config_id] = (
                        new_managed
                    )
                    self._save_and_rerun(scope="fragment")

            with col2:
                if st.button(
                    "📤 Unload",
                    key=f"unload_folder_{folder}",
                    use_container_width=True,
                    help="Unload this folder (configs are saved, not deleted)",
                ):
                    save_configs_to_folder(
                        st.session_state.managed_configs, CONFIGS_DIR, folder
                    )
                    st.session_state.managed_configs = unload_folder_configs(
                        st.session_state.managed_configs, folder
                    )
                    st.session_state.loaded_folders.discard(folder)
                    self._save_loaded_folders()
                    self._save_and_rerun()

            if config_count == 0:
                st.info(
                    "No configs in this folder. Click 'New Amplification' to create one."
                )
            else:
                # Dup/Delete buttons at list level so Active toggle updates expander title
                for config_id, mc in list(folder_configs.items()):
                    col1, col2, col3 = st.columns([30, 1, 1], gap=None)
                    with col1:
                        self._render_amplification_config(config_id, mc)
                    with col2:
                        if st.button("📋", key=f"dup_{config_id}", help="Duplicate"):
                            config = mc.config
                            new_config = deepcopy(config)
                            new_config.config_id = str(uuid.uuid4())
                            new_config.name = self._get_unique_config_name(
                                f"{config.name} copy"
                            )
                            new_managed = ManagedConfig.from_config(
                                new_config,
                                active=mc.active,
                                expanded=True,
                                folder=mc.folder,
                            )
                            st.session_state.managed_configs[new_managed.config_id] = (
                                new_managed
                            )
                            self._save_and_rerun(scope="fragment")
                    with col3:
                        if st.button("🗑️", key=f"del_{config_id}", help="Delete"):
                            del st.session_state.managed_configs[config_id]
                            self._save_and_rerun(scope="fragment")

    def _render_generation_controls(self, suffix: str, label: str) -> bool:
        """
        Render generation buttons and clear results button.

        Args:
            suffix: Unique suffix for widget keys (e.g. 'text', 'msg')
            label: Label for the generate button (e.g. 'Text', 'Messages')

        Returns:
            bool: True if the generate button was clicked
        """
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        clicked = False
        with col1:
            if st.button(
                f"🚀 Generate {label}",
                type="primary",
                use_container_width=True,
                key=f"gen_{suffix}_btn",
            ):
                clicked = True
        with col2:
            if st.button(
                "🗑️ Clear Results",
                key=f"clear_{suffix}_btn",
                disabled=st.session_state.multi_gen_results is None,
            ):
                st.session_state.multi_gen_results = None
                self._save_and_rerun(scope="fragment")
        return clicked

    @st.fragment
    def _render_multi_generation_tab(self) -> None:
        """Render Tab 2: Multi-generation interface. Fragment for tab-level isolation."""
        st.markdown("## Multi-Generation")
        st.markdown(
            "Generate text with multiple amplification configurations side-by-side."
        )

        if st.session_state.multi_gen_preset_prompt is not None:
            st.session_state.multi_gen_text_prompt = (
                st.session_state.multi_gen_preset_prompt
            )
            st.session_state.multi_gen_preset_prompt = None

        if st.session_state.multi_gen_preset_apply_template is not None:
            st.session_state.multi_gen_template_mode = (
                "Apply chat template"
                if st.session_state.multi_gen_preset_apply_template
                else "No template"
            )
            st.session_state.multi_gen_preset_apply_template = None

        if st.session_state.multi_gen_preset_messages is not None:
            st.session_state.multi_gen_messages = (
                st.session_state.multi_gen_preset_messages
            )
            st.session_state.multi_gen_preset_messages = None

        active_configs = [
            mc for mc in st.session_state.managed_configs.values() if mc.active
        ]

        if len(active_configs) == 0:
            st.warning(
                "No active amplification configurations. Go to the Amplifications tab to create and activate configs."
            )
        else:
            st.info(
                f"Will generate with {len(active_configs)} active configuration(s): {', '.join(c.config.name for c in active_configs)}"
            )

        text_tab, msg_tab = st.tabs(["📝 Text", "💬 Messages"])

        text_gen_clicked = False
        msg_gen_clicked = False

        with text_tab:
            self._render_text_input_tab()
            text_gen_clicked = self._render_generation_controls("text", "Text")

        with msg_tab:
            self._render_message_builder_tab()
            msg_gen_clicked = self._render_generation_controls("msg", "Messages")

        generate_clicked = text_gen_clicked or msg_gen_clicked
        if text_gen_clicked:
            st.session_state.multi_gen_active_tab = "Text"
        elif msg_gen_clicked:
            st.session_state.multi_gen_active_tab = "Messages"

        if generate_clicked:
            self._save_last_multigen_state()

            sampling_params = self._get_sampling_params()

            active_tab = st.session_state.get("multi_gen_active_tab", "Text")

            if active_tab == "Text":
                prompt = st.session_state.multi_gen_current_prompt
                template_mode = st.session_state.multi_gen_current_template_mode
                system_prompt = st.session_state.get(
                    "multi_gen_current_system_prompt", ""
                )
                assistant_prefill = st.session_state.get(
                    "multi_gen_current_assistant_prefill", ""
                )
                loom_filename = st.session_state.get(
                    "multi_gen_current_loom_filename", "untitled.txt"
                )

                if template_mode == "No template":
                    final_prompt = self.tokenizer.encode(prompt)
                elif template_mode == "Apply chat template":
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    if assistant_prefill:
                        messages.append(
                            {"role": "assistant", "content": assistant_prefill}
                        )
                        final_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            continue_final_message=True,
                        )
                    else:
                        final_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                        )
                elif template_mode == "Apply loom template":
                    final_prompt = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command.",
                            },
                            {
                                "role": "user",
                                "content": f"<cmd>cat {loom_filename}</cmd>",
                            },
                            {"role": "assistant", "content": prompt},
                        ],
                        continue_final_message=True,
                    )

                original_prompt = prompt

            elif active_tab == "Messages":
                messages = st.session_state.multi_gen_messages

                if not messages:
                    st.error("Cannot generate: no messages in conversation")
                else:
                    template_override = st.session_state.msg_builder_template_override

                    if template_override == "No template override":
                        last_role = messages[-1]["role"]
                        if last_role == "assistant":
                            add_gen_prompt = False
                            continue_final = True
                        else:
                            add_gen_prompt = True
                            continue_final = False
                    elif template_override == "Force generation prompt":
                        add_gen_prompt = True
                        continue_final = False
                    elif template_override == "Force continue final message":
                        add_gen_prompt = False
                        continue_final = True
                    else:
                        add_gen_prompt = False
                        continue_final = False

                    final_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=add_gen_prompt,
                        continue_final_message=continue_final,
                    )
                    original_prompt = f"[Conversation with {len(messages)} message(s)]"
            with st.expander("📋 Prompt", expanded=False):
                st.code(
                    self.tokenizer.decode(final_prompt, skip_special_tokens=False),
                    language="text",
                    wrap_lines=True,
                )

            # Create placeholders for progressive rendering
            st.markdown("## Generating...")
            output_cols = st.columns(2)
            placeholders = []
            for idx, mc in enumerate(active_configs):
                col_idx = idx % 2
                with output_cols[col_idx]:
                    placeholder = st.empty()
                    with placeholder.container():
                        with st.expander(
                            f"⏳ ({idx + 1}) {mc.config.name}", expanded=True
                        ):
                            st.info("Waiting for generation...")
                    placeholders.append(placeholder)

            # Stream results as they arrive
            results = []
            results_data_in_progress = {
                "prompt": original_prompt,
                "final_prompt": final_prompt,
                "results": results,
                "active_tab": active_tab,
                "template_mode": template_mode if active_tab == "Text" else None,
                "loom_filename": loom_filename if active_tab == "Text" else None,
            }
            for idx, result_data in enumerate(
                self._multi_gen_request(
                    prompt=final_prompt,
                    amplification_configs=active_configs,
                    sampling_params=sampling_params,
                )
            ):
                results.append(result_data)
                with placeholders[idx].container():
                    self._render_result_card_content(
                        idx, result_data, results_data_in_progress, disabled=True
                    )

            st.session_state.multi_gen_results = results_data_in_progress
            st.session_state.multi_gen_sample_indices = {
                i: 0 for i in range(len(results))
            }

            # Log the generation
            GenerationLog.from_dashboard_generation(
                generation_type="multigen",
                model_id=self.method.base_model_cfg.model_id,
                prompt_text=original_prompt,
                prompt_tokens=final_prompt,
                sampling_params=sampling_params,
                configs=active_configs,
                results=[
                    {"config_name": r["config"].name, "outputs": r["results"]}
                    for r in results
                ],
                messages=(
                    st.session_state.get("multi_gen_messages")
                    if active_tab == "Messages"
                    else None
                ),
                template_mode=template_mode if active_tab == "Text" else None,
                logs_dir=LOGS_DIR,
            )

            st.rerun(scope="fragment")  # Rerun tab to enable interactive buttons

        if st.session_state.multi_gen_results is not None:
            st.markdown("---")
            results_data = st.session_state.multi_gen_results
            with st.expander("📋 Prompt", expanded=False):
                st.code(
                    self.tokenizer.decode(
                        results_data["final_prompt"], skip_special_tokens=False
                    ),
                    language="text",
                    wrap_lines=True,
                )

            st.markdown("## Generated Outputs")
            output_cols = st.columns(2)
            for idx, result_data in enumerate(results_data["results"]):
                col_idx = idx % 2

                with output_cols[col_idx]:
                    self._render_result_card(idx, result_data, results_data)

    @st.fragment
    def _render_result_card(
        self, idx: int, result_data: dict, results_data: dict
    ) -> None:
        """Fragment wrapper for result card - enables fast button interactions."""
        self._render_result_card_content(idx, result_data, results_data, disabled=False)

    def _render_result_card_content(
        self, idx: int, result_data: dict, results_data: dict, disabled: bool = False
    ) -> None:
        """Render a single result card with sample cycling."""
        num_samples = len(result_data["results"])
        formatted_title = f"({idx + 1}) {result_data['config'].name}"
        key_suffix = "_disabled" if disabled else ""

        with st.expander(formatted_title, expanded=True):
            render_sample_cycler(
                samples=result_data["results"],
                component_id=f"cycler_{idx}{key_suffix}",
                height=300,
            )

            st.markdown("---")

            if num_samples > 1:

                def format_sample_option(x):
                    return "All samples" if x == -1 else f"Sample {x + 1}"

                action_sample_idx = st.selectbox(
                    "Apply actions to sample",
                    options=[-1] + list(range(num_samples)),
                    format_func=format_sample_option,
                    key=f"action_sample_{idx}{key_suffix}",
                    disabled=disabled,
                )
                is_all_samples = action_sample_idx == -1
            else:
                action_sample_idx = 0
                is_all_samples = False

            effective_idx = 0 if is_all_samples else action_sample_idx
            current_result = result_data["results"][effective_idx]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(
                    "➕ Continue",
                    key=f"continue_{idx}{key_suffix}",
                    use_container_width=True,
                    disabled=disabled,
                ):
                    sampling_params = self._get_sampling_params()

                    if is_all_samples:
                        indices_to_continue = list(range(num_samples))
                        spinner_text = f"Continuing all {num_samples} samples..."
                    else:
                        indices_to_continue = [action_sample_idx]
                        spinner_text = "Continuing generation..."

                    with st.spinner(spinner_text):
                        for sample_idx in indices_to_continue:
                            sample_tokens = result_data["output_tokens"][sample_idx]
                            continuation_prompt = (
                                results_data["final_prompt"] + sample_tokens
                            )

                            continuation_results = next(
                                self._multi_gen_request(
                                    prompt=continuation_prompt,
                                    amplification_configs=[result_data["config"]],
                                    sampling_params=sampling_params,
                                )
                            )

                            result_data["results"][sample_idx] += continuation_results[
                                "results"
                            ][0]
                            result_data["output_tokens"][sample_idx] = (
                                sample_tokens + continuation_results["output_tokens"][0]
                            )

                    # Log with full outputs (original + continuation)
                    GenerationLog.from_dashboard_generation(
                        generation_type="continue",
                        model_id=self.method.base_model_cfg.model_id,
                        prompt_text=results_data.get("prompt", ""),
                        prompt_tokens=results_data["final_prompt"],
                        sampling_params=sampling_params,
                        configs=[result_data["config"]],
                        results=[
                            {
                                "config_name": result_data["config"].name,
                                "outputs": [
                                    result_data["results"][i]
                                    for i in indices_to_continue
                                ],
                            }
                        ],
                        template_mode=results_data.get("template_mode"),
                        logs_dir=LOGS_DIR,
                    )

                    st.rerun(scope="fragment")

            with col2:
                if st.button(
                    "🔄 Regenerate",
                    key=f"regenerate_{idx}{key_suffix}",
                    use_container_width=True,
                    disabled=disabled,
                ):
                    sampling_params = self._get_sampling_params()

                    with st.spinner("Regenerating..."):
                        new_results = next(
                            self._multi_gen_request(
                                prompt=results_data["final_prompt"],
                                amplification_configs=[result_data["config"]],
                                sampling_params=sampling_params,
                            )
                        )

                    result_data["results"] = new_results["results"]
                    result_data["output_tokens"] = new_results["output_tokens"]

                    # Log the regeneration
                    GenerationLog.from_dashboard_generation(
                        generation_type="regenerate",
                        model_id=self.method.base_model_cfg.model_id,
                        prompt_text=results_data.get("prompt", ""),
                        prompt_tokens=results_data["final_prompt"],
                        sampling_params=sampling_params,
                        configs=[result_data["config"]],
                        results=[
                            {
                                "config_name": result_data["config"].name,
                                "outputs": new_results["results"],
                            }
                        ],
                        template_mode=results_data.get("template_mode"),
                        logs_dir=LOGS_DIR,
                    )

                    st.rerun(scope="fragment")

            with col3:
                template_mode = results_data.get("template_mode")
                is_no_template = template_mode == "No template"

                if is_all_samples:
                    continue_chat_help = "Select a specific sample to continue chat"
                elif is_no_template:
                    continue_chat_help = "Cannot continue chat without a template"
                else:
                    continue_chat_help = None

                if st.button(
                    "💬 Continue Chat",
                    key=f"continue_chat_{idx}{key_suffix}",
                    use_container_width=True,
                    disabled=disabled or is_all_samples or is_no_template,
                    help=continue_chat_help,
                ):
                    conv_id = f"conv_{st.session_state.conversation_counter}"
                    st.session_state.conversation_counter += 1

                    conv_name = self._get_unique_conversation_name(
                        f"{result_data['config'].name}"
                    )

                    if results_data.get("active_tab") == "Messages":
                        messages = st.session_state.get("multi_gen_messages", [])
                        system_msgs = [
                            m["content"] for m in messages if m["role"] == "system"
                        ]
                        system_prompt = "\n\n".join(system_msgs)
                        history = [
                            {k: v for k, v in msg.items() if k in ["role", "content"]}
                            for msg in messages
                            if msg["role"] != "system"
                        ]
                        history.append(
                            {
                                "role": "assistant",
                                "content": current_result,
                                "config_name": result_data["config"].name,
                            }
                        )
                    elif template_mode == "Apply loom template":
                        loom_filename = results_data.get(
                            "loom_filename", "untitled.txt"
                        )
                        system_prompt = "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command."
                        history = [
                            {
                                "role": "user",
                                "content": f"<cmd>cat {loom_filename}</cmd>",
                            },
                            {
                                "role": "assistant",
                                "content": results_data["prompt"] + current_result,
                                "config_name": result_data["config"].name,
                            },
                        ]
                    else:
                        system_prompt = ""
                        history = [
                            {
                                "role": "user",
                                "content": results_data["prompt"],
                            },
                            {
                                "role": "assistant",
                                "content": current_result,
                                "config_name": result_data["config"].name,
                            },
                        ]

                    st.session_state.conversations[conv_id] = {
                        "name": conv_name,
                        "context": {
                            "config": result_data["config"],
                            "system_prompt": system_prompt,
                        },
                        "history": history,
                        "editing_message": None,
                        "regenerating_from": None,
                        "continuing_from": None,
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
                if is_all_samples:
                    all_samples_text = "\n\n".join(
                        f"=== Sample {i + 1} ===\n{s}"
                        for i, s in enumerate(result_data["results"])
                    )
                    download_data = all_samples_text
                    download_filename = f"{result_data['config'].name.replace(' ', '_')}_all_samples.txt"
                else:
                    download_data = current_result
                    download_filename = f"{result_data['config'].name.replace(' ', '_')}_sample{effective_idx + 1}.txt"

                st.download_button(
                    label="📥 Download",
                    data=download_data,
                    file_name=download_filename,
                    mime="text/plain",
                    key=f"download_{idx}{key_suffix}",
                    use_container_width=True,
                    disabled=disabled,
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

        conv_items = list(st.session_state.conversations.items())
        tab_names = [conv["name"] for _, conv in conv_items] + ["➕ New"]
        tabs = st.tabs(tab_names)

        for tab, (conv_id, conv) in zip(tabs[:-1], conv_items):
            with tab:
                self._render_single_conversation(conv_id, conv)

        with tabs[-1]:
            self._render_new_conversation_tab()

    def _create_new_conversation(self, config=None, name=None) -> str:
        """Create a new empty conversation and return its ID."""
        conv_id = f"conv_{st.session_state.conversation_counter}"
        st.session_state.conversation_counter += 1

        if config is None:
            active_mcs = [
                mc for mc in st.session_state.managed_configs.values() if mc.active
            ]
            all_mcs = list(st.session_state.managed_configs.values())
            config = active_mcs[0] if active_mcs else (all_mcs[0] if all_mcs else None)

        conv_name = self._get_unique_conversation_name(
            name or f"New Chat {st.session_state.conversation_counter}"
        )

        st.session_state.conversations[conv_id] = {
            "name": conv_name,
            "context": {
                "config": config,
            },
            "history": [],
            "editing_message": None,
            "regenerating_from": None,
            "continuing_from": None,
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
            config_names = [
                mc.config.name for mc in st.session_state.managed_configs.values()
            ]
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
                    mc
                    for mc in st.session_state.managed_configs.values()
                    if mc.config.name == selected_config_name
                )
                self._create_new_conversation(config=config, name=conv_name)
                st.success(f"Created conversation: {conv_name}")
                self._save_and_rerun()
            else:
                st.error("Please create an amplification configuration first")

    @st.fragment
    def _render_chat_messages(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """Render the message list for a conversation. Fragment for fast delete/edit."""
        config = conv["context"]["config"]

        for i, msg in enumerate(conv["history"]):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    if conv["editing_message"] == i:
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
                                st.rerun(scope="fragment")
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_user_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                st.rerun(scope="fragment")
                    else:
                        st.markdown(msg["content"])
                        # Check if there's an assistant message following this user message
                        has_next_assistant = (
                            i + 1 < len(conv["history"])
                            and conv["history"][i + 1]["role"] == "assistant"
                        )
                        _, btn_col1, btn_col2, btn_col3 = st.columns([10, 1, 1, 1])
                        with btn_col1:
                            if st.button(
                                "✏️",
                                key=f"edit_btn_user_{conv_id}_{i}",
                                help="Edit message",
                                type="secondary",
                            ):
                                conv["editing_message"] = i
                                st.rerun(scope="fragment")
                        with btn_col2:
                            if st.button(
                                "🔄",
                                key=f"regen_btn_user_{conv_id}_{i}",
                                help="Regenerate assistant response",
                                type="secondary",
                                disabled=not has_next_assistant,
                            ):
                                # Regenerate the next assistant message
                                conv["regenerating_from"] = i + 1
                                st.rerun(scope="app")
                        with btn_col3:
                            if st.button(
                                "🗑️",
                                key=f"delete_btn_user_{conv_id}_{i}",
                                help="Delete message",
                                type="secondary",
                            ):
                                conv["history"].pop(i)
                                self._save_conversation(conv_id, conv)
                                st.rerun(scope="fragment")
            else:
                with st.chat_message("assistant"):
                    if conv["editing_message"] == i:
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
                                st.rerun(scope="fragment")
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_asst_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                st.rerun(scope="fragment")
                    else:
                        config_label = f"[{msg.get('config_name', config.name if config else 'No Config')}]"
                        st.markdown(f"**{config_label}** {msg['content']}")
                        _, btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(
                            [10, 1, 1, 1, 1]
                        )
                        with btn_col1:
                            if st.button(
                                "✏️",
                                key=f"edit_btn_asst_{conv_id}_{i}",
                                help="Edit message",
                                type="secondary",
                            ):
                                conv["editing_message"] = i
                                st.rerun(scope="fragment")
                        with btn_col2:
                            if st.button(
                                "➕",
                                key=f"continue_btn_asst_{conv_id}_{i}",
                                help="Continue this message",
                                type="secondary",
                            ):
                                conv["continuing_from"] = i
                                st.rerun(scope="app")
                        with btn_col3:
                            if st.button(
                                "🔄",
                                key=f"regen_btn_asst_{conv_id}_{i}",
                                help="Regenerate from here",
                                type="secondary",
                            ):
                                conv["regenerating_from"] = i
                                st.rerun(scope="app")
                        with btn_col4:
                            if st.button(
                                "🗑️",
                                key=f"delete_btn_asst_{conv_id}_{i}",
                                help="Delete message",
                                type="secondary",
                            ):
                                conv["history"].pop(i)
                                self._save_conversation(conv_id, conv)
                                st.rerun(scope="fragment")

    @st.fragment
    def _render_single_conversation(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """Render a single conversation. Fragment for independent updates."""
        config = conv["context"]["config"]
        pending_key = f"chat_pending_samples_{conv_id}"

        if conv["regenerating_from"] is not None:
            regen_index = conv["regenerating_from"]
            conv["regenerating_from"] = None

            prompt = self._truncate_history_and_get_prompt(conv, regen_index)

            sampling_params = self._get_sampling_params()
            use_multi_gen = (
                st.session_state.get("chat_multi_gen", False) and sampling_params.n > 1
            )

            managed_config = next(
                mc
                for mc in st.session_state.managed_configs.values()
                if mc.config.name == config.name
            )

            config_label = f"[{config.name}]" if config else "[No Config]"

            if use_multi_gen:
                with st.spinner(f"Regenerating {sampling_params.n} samples..."):
                    result = next(
                        self._multi_gen_request(
                            prompt=prompt,
                            amplification_configs=[managed_config],
                            sampling_params=sampling_params,
                        )
                    )

                # Log chat regeneration
                GenerationLog.from_dashboard_generation(
                    generation_type="regenerate",
                    model_id=self.method.base_model_cfg.model_id,
                    prompt_text=self.tokenizer.decode(
                        prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[
                        {"config_name": config.name, "outputs": result["results"]}
                    ],
                    messages=self._get_messages_with_system_prompt(conv),
                    logs_dir=LOGS_DIR,
                )

                st.session_state[pending_key] = {
                    "samples": result["results"],
                    "config_name": config.name if config else "No Config",
                    "mode": "replace",
                }
                self._save_conversation(conv_id, conv)
                self._save_and_rerun(scope="fragment")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**{config_label}**")
                    with st.spinner("Regenerating..."):
                        result = next(
                            self._multi_gen_request(
                                prompt=prompt,
                                amplification_configs=[managed_config],
                                sampling_params=sampling_params,
                            )
                        )
                        response = result["results"][0]
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

                    # Log chat regeneration
                    GenerationLog.from_dashboard_generation(
                        generation_type="regenerate",
                        model_id=self.method.base_model_cfg.model_id,
                        prompt_text=self.tokenizer.decode(
                            prompt, skip_special_tokens=False
                        ),
                        prompt_tokens=prompt,
                        sampling_params=sampling_params,
                        configs=[managed_config],
                        results=[{"config_name": config.name, "outputs": [response]}],
                        messages=self._get_messages_with_system_prompt(conv),
                        logs_dir=LOGS_DIR,
                    )

                self._save_and_rerun(scope="fragment")

        if conv.get("continuing_from") is not None:
            continue_index = conv["continuing_from"]
            conv["continuing_from"] = None

            # Build prompt including the message we're continuing
            messages = self._get_messages_with_system_prompt(
                conv, conv["history"][: continue_index + 1]
            )
            prompt = self.tokenizer.apply_chat_template(
                messages,
                continue_final_message=True,
            )

            sampling_params = self._get_sampling_params()
            use_multi_gen = (
                st.session_state.get("chat_multi_gen", False) and sampling_params.n > 1
            )

            managed_config = next(
                mc
                for mc in st.session_state.managed_configs.values()
                if mc.config.name == config.name
            )

            config_label = f"[{config.name}]" if config else "[No Config]"

            original_content = conv["history"][continue_index]["content"]

            if use_multi_gen:
                with st.spinner(f"Continuing with {sampling_params.n} samples..."):
                    result = next(
                        self._multi_gen_request(
                            prompt=prompt,
                            amplification_configs=[managed_config],
                            sampling_params=sampling_params,
                        )
                    )

                # Log with full outputs (original + continuation)
                GenerationLog.from_dashboard_generation(
                    generation_type="continue",
                    model_id=self.method.base_model_cfg.model_id,
                    prompt_text=self.tokenizer.decode(
                        prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[
                        {
                            "config_name": config.name,
                            "outputs": [
                                original_content + c for c in result["results"]
                            ],
                        }
                    ],
                    messages=messages,
                    logs_dir=LOGS_DIR,
                )

                st.session_state[pending_key] = {
                    "samples": result["results"],
                    "config_name": config.name if config else "No Config",
                    "mode": "continue",
                    "target_index": continue_index,
                }
                self._save_conversation(conv_id, conv)
                self._save_and_rerun(scope="fragment")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**{config_label}** (continuing...)")
                    with st.spinner("Continuing..."):
                        result = next(
                            self._multi_gen_request(
                                prompt=prompt,
                                amplification_configs=[managed_config],
                                sampling_params=sampling_params,
                            )
                        )
                        continuation = result["results"][0]
                    st.markdown(original_content + continuation)

                if continuation:
                    full_content = original_content + continuation
                    conv["history"][continue_index]["content"] = full_content
                    self._save_conversation(conv_id, conv)

                    # Log with full output
                    GenerationLog.from_dashboard_generation(
                        generation_type="continue",
                        model_id=self.method.base_model_cfg.model_id,
                        prompt_text=self.tokenizer.decode(
                            prompt, skip_special_tokens=False
                        ),
                        prompt_tokens=prompt,
                        sampling_params=sampling_params,
                        configs=[managed_config],
                        results=[
                            {"config_name": config.name, "outputs": [full_content]}
                        ],
                        messages=messages,
                        logs_dir=LOGS_DIR,
                    )

                self._save_and_rerun(scope="fragment")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            conv_name_key = f"conv_name_{conv_id}"

            def on_conv_name_change(conversation=conv, cid=conv_id, key=conv_name_key):
                new_name = st.session_state[key]
                if new_name != conversation["name"]:
                    delete_conversation_file(conversation["name"], CONVERSATIONS_DIR)
                    unique_name = self._get_unique_conversation_name(
                        new_name, exclude_conv_id=cid
                    )
                    conversation["name"] = unique_name
                    self._save_conversation(cid, conversation)

            st.text_input(
                "Conversation Name",
                value=conv["name"],
                key=conv_name_key,
                on_change=on_conv_name_change,
            )

        with col2:
            if config:
                all_mcs = list(st.session_state.managed_configs.values())
                config_names = [mc.config.name for mc in all_mcs]
                if config_names:
                    current_index = next(
                        (
                            i
                            for i, mc in enumerate(all_mcs)
                            if mc.config.name == config.name
                        ),
                        0,
                    )
                    config_select_key = f"conv_config_{conv_id}"

                    def on_config_change(
                        conversation=conv,
                        cid=conv_id,
                        mcs=all_mcs,
                        key=config_select_key,
                    ):
                        selected_name = st.session_state[key]
                        new_mc = next(
                            mc for mc in mcs if mc.config.name == selected_name
                        )
                        conversation["context"]["config"] = new_mc
                        self._save_conversation(cid, conversation)

                    st.selectbox(
                        "Config",
                        options=config_names,
                        index=current_index,
                        key=config_select_key,
                        on_change=on_config_change,
                    )
            else:
                st.info("No config")

        with col3:
            if st.button(
                "🗑️ Delete", key=f"delete_conv_{conv_id}", use_container_width=True
            ):
                delete_conversation_file(conv["name"], CONVERSATIONS_DIR)
                del st.session_state.conversations[conv_id]
                if st.session_state.active_conversation_id == conv_id:
                    st.session_state.active_conversation_id = None
                self._save_and_rerun()

        # System prompt section
        system_prompt_key = f"system_prompt_{conv_id}"

        def on_system_prompt_change(
            conversation=conv, cid=conv_id, key=system_prompt_key
        ):
            conversation["context"]["system_prompt"] = st.session_state[key]
            self._save_conversation(cid, conversation)

        with st.expander("⚙️ System Prompt", expanded=False):
            st.text_area(
                "System Prompt",
                value=conv["context"].get("system_prompt", ""),
                key=system_prompt_key,
                height=100,
                placeholder="Enter a system prompt to set context for the conversation...",
                label_visibility="collapsed",
                on_change=on_system_prompt_change,
            )

        st.markdown("---")

        self._render_chat_messages(conv_id, conv)

        # Check for pending sample selection
        pending_key = f"chat_pending_samples_{conv_id}"
        if pending_key in st.session_state:
            pending = st.session_state[pending_key]
            self._render_chat_sample_selection(
                conv_id=conv_id,
                conv=conv,
                samples=pending["samples"],
                config_name=pending["config_name"],
                mode=pending["mode"],
                target_index=pending.get("target_index"),
            )
            return  # Don't show chat input while selecting

        send_to_multi_gen = st.toggle(
            "🚀 Send next message to Multi-Generation",
            key=f"multi_gen_mode_{conv_id}",
            help="When enabled, your next message will be sent to Multi-Generation instead of this chat",
        )

        st.session_state.chat_multi_gen = st.toggle(
            "Multi-gen in Chat",
            value=st.session_state.get("chat_multi_gen", False),
            help="When enabled and Num Samples > 1, show all samples and let you select one",
            disabled=send_to_multi_gen,
            key=f"chat_multi_gen_{conv_id}",
        )

        user_input = st.chat_input(
            "Type your message here...", key=f"chat_input_{conv_id}"
        )

        if user_input:
            if send_to_multi_gen:
                history_for_multi_gen = conv["history"].copy()
                history_for_multi_gen.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                st.session_state.multi_gen_preset_messages = history_for_multi_gen

                st.success(
                    "✓ Conversation sent to Multi-Generation tab (Messages mode). Switch to the Multi-Generation tab to continue."
                )
                self._save_and_rerun()
            else:
                conv["history"].append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                with st.chat_message("user"):
                    st.markdown(user_input)

                messages = self._get_messages_with_system_prompt(conv)
                full_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )

                sampling_params = self._get_sampling_params()
                use_multi_gen = (
                    st.session_state.get("chat_multi_gen", False)
                    and sampling_params.n > 1
                )

                managed_config = next(
                    mc
                    for mc in st.session_state.managed_configs.values()
                    if mc.config.name == config.name
                )

                config_label = f"[{config.name}]" if config else "[No Config]"

                if use_multi_gen:
                    with st.spinner(f"Generating {sampling_params.n} samples..."):
                        result = next(
                            self._multi_gen_request(
                                prompt=full_prompt,
                                amplification_configs=[managed_config],
                                sampling_params=sampling_params,
                            )
                        )

                    # Log chat generation
                    GenerationLog.from_dashboard_generation(
                        generation_type="chat",
                        model_id=self.method.base_model_cfg.model_id,
                        prompt_text=self.tokenizer.decode(
                            full_prompt, skip_special_tokens=False
                        ),
                        prompt_tokens=full_prompt,
                        sampling_params=sampling_params,
                        configs=[managed_config],
                        results=[
                            {"config_name": config.name, "outputs": result["results"]}
                        ],
                        messages=messages,
                        logs_dir=LOGS_DIR,
                    )

                    st.session_state[pending_key] = {
                        "samples": result["results"],
                        "config_name": config.name if config else "No Config",
                        "mode": "add",
                    }
                    self._save_conversation(conv_id, conv)
                    self._save_and_rerun(scope="fragment")
                else:
                    with st.chat_message("assistant"):
                        st.write(f"**{config_label}**")
                        with st.spinner("Generating..."):
                            result = next(
                                self._multi_gen_request(
                                    prompt=full_prompt,
                                    amplification_configs=[managed_config],
                                    sampling_params=sampling_params,
                                )
                            )
                            response = result["results"][0]
                        st.markdown(response)

                    conv["history"].append(
                        {
                            "role": "assistant",
                            "content": response,
                            "config_name": config.name if config else "No Config",
                        }
                    )
                    self._save_conversation(conv_id, conv)

                    # Log chat generation
                    GenerationLog.from_dashboard_generation(
                        generation_type="chat",
                        model_id=self.method.base_model_cfg.model_id,
                        prompt_text=self.tokenizer.decode(
                            full_prompt, skip_special_tokens=False
                        ),
                        prompt_tokens=full_prompt,
                        sampling_params=sampling_params,
                        configs=[managed_config],
                        results=[{"config_name": config.name, "outputs": [response]}],
                        messages=messages,
                        logs_dir=LOGS_DIR,
                    )

                    self._save_and_rerun(scope="fragment")

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
            # Render inside expander with action buttons
            icon = "✅" if mc.active else "❌"
            with st.expander(f"{icon} {config.name}", expanded=mc.expanded):
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

            def on_name_change(cfg=config, cid=config_id, key=name_key):
                new_name = st.session_state[key]
                if new_name != cfg.name:
                    unique_name = self._get_unique_config_name(
                        new_name, exclude_config_id=cid
                    )
                    st.session_state.managed_configs[cid].config.name = unique_name
                    self._save_configs()

            st.text_input(
                "Configuration Name",
                value=config.name,
                key=name_key,
                on_change=on_name_change,
            )

            desc_key = f"{key_prefix}config_desc_{config_id}"

            def on_description_change(cfg=config, key=desc_key):
                cfg.description = st.session_state[key]
                self._save_configs()

            st.text_area(
                "Description",
                value=config.description,
                key=desc_key,
                height=60,
                on_change=on_description_change,
            )

        active_key = f"{key_prefix}config_active_{config_id}"

        def on_active_change(managed_config=mc, key=active_key):
            managed_config.active = st.session_state[key]
            self._save_configs()

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
                    config_id, adapter_idx, adapter, key_prefix
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
            self._save_and_rerun(scope="fragment")

    def _render_adapter_amplification(
        self,
        config_id: str,
        adapter_idx: int,
        adapter: AmplifiedAdapter,
        key_prefix: str = "",
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
                    self._save_and_rerun(scope="fragment")

            base_model_name = self.method.base_model_cfg.name

            col1, col2 = st.columns(2)

            with col1:
                available_organisms = get_available_organisms(
                    base_model_name=self.method.base_model_cfg.name, only_loras=True
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
                        self._save_configs()

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
                    self._save_configs()

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

            if len(adapter.layer_amplifications) == 0:
                st.info("No layer specifications. Click 'Add Layer Spec' below.")
            else:
                for layer_idx, layer_amp in enumerate(adapter.layer_amplifications):
                    self._render_layer_amplification(
                        config_id, adapter_idx, layer_idx, layer_amp, key_prefix
                    )

            if st.button(
                "➕ Add Layer Spec",
                key=f"{key_prefix}add_layer_{config_id}_{adapter_idx}",
            ):
                new_layer_amp = LayerAmplification(
                    layers="all",
                    module_amplifications=[],
                )
                adapter.layer_amplifications.append(new_layer_amp)
                self._save_and_rerun(scope="fragment")

    def _render_layer_amplification(
        self,
        config_id: str,
        adapter_idx: int,
        layer_idx: int,
        layer_amp: LayerAmplification,
        key_prefix: str = "",
    ) -> None:
        """Render layer amplification specification."""
        base_key = f"{key_prefix}{config_id}_{adapter_idx}_{layer_idx}"
        mode_key = f"layer_mode_{base_key}"
        single_key = f"layer_single_{base_key}"
        range_key = f"layer_range_{base_key}"
        list_key = f"layer_list_{base_key}"

        def on_mode_change(lamp=layer_amp, mk=mode_key):
            mode = st.session_state[mk]
            if mode == "All":
                lamp.layers = "all"
            elif mode == "Single":
                lamp.layers = 0
            elif mode == "Range":
                num_layers = self.method.base_model.num_layers
                lamp.layers = list(range(num_layers))
            else:  # List
                lamp.layers = []
            self._save_configs()

        def on_single_change(lamp=layer_amp, key=single_key):
            lamp.layers = st.session_state[key]
            self._save_configs()

        def on_range_change(lamp=layer_amp, key=range_key):
            start, end = st.session_state[key]
            lamp.layers = list(range(start, end + 1))
            self._save_configs()

        def on_list_change(lamp=layer_amp, key=list_key):
            val = st.session_state[key].strip()
            if val:
                lamp.layers = [int(x.strip()) for x in val.split(",") if x.strip()]
            else:
                lamp.layers = []
            self._save_configs()

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
                    self._save_and_rerun(scope="fragment")

            if isinstance(layer_amp.layers, list):
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
                key=mode_key,
                horizontal=True,
                on_change=on_mode_change,
            )

            if layer_mode == "All":
                layer_amp.layers = "all"
                st.info("Applies to all layers in the model")

            elif layer_mode == "Single":
                current_val = (
                    layer_amp.layers if isinstance(layer_amp.layers, int) else 0
                )
                st.number_input(
                    "Layer Index",
                    min_value=0,
                    value=current_val,
                    step=1,
                    key=single_key,
                    on_change=on_single_change,
                )

            elif layer_mode == "Range":
                num_layers = self.method.base_model.num_layers

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
                    key=range_key,
                    help="Select the range of layers to apply amplification to",
                    on_change=on_range_change,
                )

                range_start, range_end = layer_range
                st.info(
                    f"Applies to layers {range_start} through {range_end} ({range_end - range_start + 1} layers)"
                )

            else:  # List
                current_val = (
                    ",".join(map(str, layer_amp.layers))
                    if isinstance(layer_amp.layers, list)
                    else ""
                )
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
                self._save_and_rerun(scope="fragment")

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
        weight_key = f"module_weight_{base_key}"

        def on_module_change(mod_amp=module_amp, key=module_key):
            mod_amp.modules = st.session_state[key]
            self._save_configs()

        def on_weight_change(mod_amp=module_amp, key=weight_key):
            mod_amp.weight = st.session_state[key]
            self._save_configs()

        col1, col2, col3 = st.columns([2, 2, 1])

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
                min_value=-5.0,
                max_value=5.0,
                value=float(module_amp.weight),
                step=0.1,
                key=weight_key,
                help="Amplification factor (1.0 = no change, 2.0 = double, 0.5 = half)",
                on_change=on_weight_change,
            )

        with col3:
            if st.button(
                "🗑️",
                key=f"delete_module_{base_key}",
            ):
                st.session_state.managed_configs[config_id].config.amplified_adapters[
                    adapter_idx
                ].layer_amplifications[layer_idx].module_amplifications.pop(module_idx)
                self._save_and_rerun(scope="fragment")

    # ============ Multi-Prompt Tab ============

    def _render_multi_prompt_tab(self) -> None:
        """Render the multi-prompt generation tab."""
        prompts_tab, results_tab = st.tabs(["📝 Prompts", "📊 Results"])

        with prompts_tab:
            self._render_prompts_subtab()
        with results_tab:
            self._render_multi_prompt_results_subtab()

    @st.fragment
    def _render_prompts_subtab(self) -> None:
        """Render the prompts management subtab. Fragment for tab-level isolation."""
        col1, col2 = st.columns([3, 1])

        with col1:
            active_prompts = [
                mp for mp in st.session_state.managed_prompts.values() if mp.active
            ]
            st.markdown(f"**{len(active_prompts)} active prompt(s)**")

        with col2:
            if st.button(
                f"🚀 Run Active ({len(active_prompts)})",
                use_container_width=True,
                disabled=len(active_prompts) == 0,
            ):
                self._run_multi_prompt_generation()

        self._render_prompt_folder_loader()

        st.markdown("---")

        if len(st.session_state.loaded_prompt_folders) == 0:
            st.info("No folders loaded. Select a folder above to load prompts.")
        else:
            for folder in sorted(st.session_state.loaded_prompt_folders):
                self._render_prompt_folder_section(folder)

    def _render_prompt_folder_loader(self) -> None:
        """Render the prompt folder loader UI (dropdown + Load/Create buttons)."""
        all_folders = list_all_prompt_folders(PROMPTS_DIR)
        loaded = st.session_state.loaded_prompt_folders
        available_to_load = [f for f in all_folders if f not in loaded]

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            folder_display = {f: "Root" if f == "" else f for f in available_to_load}
            if available_to_load:
                selected_folder = st.selectbox(
                    "Available Folders",
                    options=available_to_load,
                    format_func=lambda x: folder_display.get(x, x),
                    key="prompt_folder_to_load",
                )
            else:
                st.info("All folders are loaded")
                selected_folder = None

        with col2:
            if st.button(
                "📂 Load",
                disabled=selected_folder is None,
                use_container_width=True,
                key="load_prompt_folder_btn",
            ):
                st.session_state.loaded_prompt_folders.add(selected_folder)
                loaded_prompts = load_prompts_from_folder(PROMPTS_DIR, selected_folder)
                st.session_state.managed_prompts.update(loaded_prompts)
                self._save_loaded_folders()
                st.rerun(scope="fragment")

        with col3:
            if st.button(
                "➕ Create", use_container_width=True, key="create_prompt_folder_btn"
            ):
                st.session_state.show_create_prompt_folder_dialog = True

        if st.session_state.get("show_create_prompt_folder_dialog", False):
            self._render_create_prompt_folder_dialog()

    def _render_create_prompt_folder_dialog(self) -> None:
        """Render the create prompt folder dialog."""
        with st.container(border=True):
            st.markdown("**Create New Prompt Folder**")
            new_folder_path = st.text_input(
                "Folder path",
                placeholder="e.g., experiments/v2",
                key="new_prompt_folder_path",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Create",
                    type="primary",
                    use_container_width=True,
                    key="create_prompt_folder_confirm",
                ):
                    if new_folder_path:
                        create_folder(PROMPTS_DIR, new_folder_path)
                        st.session_state.loaded_prompt_folders.add(new_folder_path)
                        st.session_state.show_create_prompt_folder_dialog = False
                        self._save_loaded_folders()
                        st.rerun(scope="fragment")
                    else:
                        st.error("Please enter a folder path")
            with col2:
                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="cancel_prompt_folder_dialog",
                ):
                    st.session_state.show_create_prompt_folder_dialog = False
                    st.rerun(scope="fragment")

    def _render_prompt_folder_section(self, folder: str) -> None:
        """Render a single prompt folder section."""
        folder_display = "Root" if folder == "" else folder
        folder_prompts = {
            pid: mp
            for pid, mp in st.session_state.managed_prompts.items()
            if mp.folder == folder
        }
        prompt_count = len(folder_prompts)

        with st.expander(
            f"📁 {folder_display} ({prompt_count} prompts)", expanded=True
        ):
            col1, col2 = st.columns([4, 1])

            with col1:
                if st.button(
                    "➕ New Prompt",
                    key=f"new_prompt_{folder}",
                    use_container_width=True,
                ):
                    new_prompt = ManagedPrompt(
                        active=True, expanded=True, folder=folder
                    )
                    st.session_state.managed_prompts[new_prompt.prompt_id] = new_prompt
                    self._save_prompts()
                    st.rerun(scope="fragment")

            with col2:
                if st.button(
                    "📤 Unload",
                    key=f"unload_prompt_folder_{folder}",
                    use_container_width=True,
                    help="Unload this folder (prompts are saved, not deleted)",
                ):
                    save_prompts_to_folder(
                        st.session_state.managed_prompts, PROMPTS_DIR, folder
                    )
                    st.session_state.loaded_prompt_folders.discard(folder)
                    st.session_state.managed_prompts = unload_folder_prompts(
                        st.session_state.managed_prompts, folder
                    )
                    self._save_loaded_folders()
                    st.rerun(scope="fragment")

            if not folder_prompts:
                st.info("No prompts in this folder.")
                return

            for prompt_id, mp in list(folder_prompts.items()):
                col1, col2, col3 = st.columns([30, 1, 1])
                with col1:
                    self._render_prompt_editor(prompt_id, mp)
                with col2:
                    if st.button("📋", key=f"dup_{prompt_id}", help="Duplicate"):
                        new_prompt = ManagedPrompt(
                            name=f"{mp.name} copy" if mp.name else "",
                            editor_mode=mp.editor_mode,
                            prompt_text=mp.prompt_text,
                            template_mode=mp.template_mode,
                            system_prompt=mp.system_prompt,
                            assistant_prefill=mp.assistant_prefill,
                            loom_filename=mp.loom_filename,
                            messages=deepcopy(mp.messages),
                            folder=mp.folder,
                            active=mp.active,
                            expanded=True,
                        )
                        st.session_state.managed_prompts[new_prompt.prompt_id] = (
                            new_prompt
                        )
                        self._save_prompts()
                        st.rerun(scope="fragment")
                with col3:
                    if st.button("🗑️", key=f"del_{prompt_id}", help="Delete"):
                        del st.session_state.managed_prompts[prompt_id]
                        self._save_prompts()
                        st.rerun(scope="fragment")

    @st.fragment
    def _render_prompt_editor(self, prompt_id: str, mp: ManagedPrompt) -> None:
        """Render prompt editor. Fragment so active toggle updates expander title."""
        icon = "✅" if mp.active else "❌"
        display_name = mp.get_display_name() or "New Prompt"

        with st.expander(f"{icon} {display_name}", expanded=mp.expanded):
            name_key = f"prompt_name_{prompt_id}"

            def on_name_change(prompt=mp, key=name_key):
                prompt.name = st.session_state[key]
                self._save_prompts()

            st.text_input(
                "Name (optional)",
                value=mp.name,
                key=name_key,
                placeholder="Auto-generated from prompt text",
                on_change=on_name_change,
            )

            # Active checkbox - in same fragment as expander so title updates
            active_key = f"prompt_active_{prompt_id}"

            def on_active_change(prompt=mp, key=active_key):
                prompt.active = st.session_state[key]
                self._save_prompts()

            st.checkbox(
                "Active",
                value=mp.active,
                key=active_key,
                help="Only active prompts will be used for generation",
                on_change=on_active_change,
            )

            # Editor mode tabs (Simple / Chat) - reusing multi-gen pattern
            simple_tab, chat_tab = st.tabs(["📝 Simple", "💬 Chat"])

            with simple_tab:
                self._render_prompt_simple_editor(prompt_id, mp)

            with chat_tab:
                self._render_prompt_chat_editor(prompt_id, mp)

    def _render_prompt_simple_editor(self, prompt_id: str, mp: ManagedPrompt) -> None:
        """Render simple text editor for prompt."""
        # Template mode
        template_key = f"prompt_template_{prompt_id}"

        def on_template_change(prompt=mp, key=template_key):
            prompt.template_mode = st.session_state[key]
            prompt.editor_mode = "simple"
            self._save_prompts()

        st.selectbox(
            "Template mode",
            ["No template", "Apply chat template", "Apply loom template"],
            index=["No template", "Apply chat template", "Apply loom template"].index(
                mp.template_mode
            ),
            key=template_key,
            on_change=on_template_change,
        )

        # Prompt text
        text_key = f"prompt_text_{prompt_id}"

        def on_text_change(prompt=mp, key=text_key):
            prompt.prompt_text = st.session_state[key]
            prompt.editor_mode = "simple"
            self._save_prompts()

        st.text_area(
            "Prompt",
            value=mp.prompt_text,
            key=text_key,
            height=150,
            on_change=on_text_change,
        )

        # Template-specific fields
        if mp.template_mode == "Apply chat template":
            # System prompt (optional)
            system_key = f"prompt_system_{prompt_id}"

            def on_system_change(prompt=mp, key=system_key):
                prompt.system_prompt = st.session_state[key]
                self._save_prompts()

            st.text_input(
                "System prompt",
                value=mp.system_prompt,
                key=system_key,
                placeholder="Optional: system instructions...",
                on_change=on_system_change,
            )

            # Assistant prefill
            prefill_key = f"prompt_prefill_{prompt_id}"

            def on_prefill_change(prompt=mp, key=prefill_key):
                prompt.assistant_prefill = st.session_state[key]
                self._save_prompts()

            st.text_input(
                "Assistant prefill",
                value=mp.assistant_prefill,
                key=prefill_key,
                placeholder="Optional: prefill the assistant's response...",
                on_change=on_prefill_change,
            )

        elif mp.template_mode == "Apply loom template":
            # Filename for loom template
            filename_key = f"prompt_loom_filename_{prompt_id}"

            def on_filename_change(prompt=mp, key=filename_key):
                prompt.loom_filename = st.session_state[key]
                self._save_prompts()

            st.text_input(
                "Filename",
                value=mp.loom_filename,
                key=filename_key,
                placeholder="untitled.txt",
                on_change=on_filename_change,
            )

    def _render_prompt_chat_editor(self, prompt_id: str, mp: ManagedPrompt) -> None:
        """Render chat-style message editor for prompt."""
        # Message list
        if not mp.messages:
            st.info("No messages. Add one below.")
        else:
            for i, msg in enumerate(mp.messages):
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.write(f"**{msg['role'].title()}**")
                with col2:
                    # Edit in place
                    content_key = f"prompt_msg_{prompt_id}_{i}"

                    def on_content_change(prompt=mp, idx=i, key=content_key):
                        prompt.messages[idx]["content"] = st.session_state[key]
                        prompt.editor_mode = "chat"
                        self._save_prompts()

                    st.text_area(
                        "Content",
                        value=msg["content"],
                        key=content_key,
                        label_visibility="collapsed",
                        height=80,
                        on_change=on_content_change,
                    )
                with col3:
                    if st.button("🗑️", key=f"del_msg_{prompt_id}_{i}"):
                        mp.messages.pop(i)
                        mp.editor_mode = "chat"
                        self._save_prompts()
                        st.rerun(scope="fragment")

        # Add message buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ User", key=f"add_user_{prompt_id}"):
                mp.messages.append({"role": "user", "content": ""})
                mp.editor_mode = "chat"
                self._save_prompts()
                st.rerun(scope="fragment")
        with col2:
            if st.button("➕ Assistant", key=f"add_assistant_{prompt_id}"):
                mp.messages.append({"role": "assistant", "content": ""})
                mp.editor_mode = "chat"
                self._save_prompts()
                st.rerun(scope="fragment")
        with col3:
            if st.button("➕ System", key=f"add_system_{prompt_id}"):
                mp.messages.insert(0, {"role": "system", "content": ""})
                mp.editor_mode = "chat"
                self._save_prompts()
                st.rerun(scope="fragment")

    def _run_multi_prompt_generation(self) -> None:
        """Run generation for all active prompts with all active configs."""
        active_prompts = [
            mp for mp in st.session_state.managed_prompts.values() if mp.active
        ]
        active_configs = [
            mc for mc in st.session_state.managed_configs.values() if mc.active
        ]

        if not active_prompts:
            st.error("No active prompts to generate.")
            return
        if not active_configs:
            st.error(
                "No active configs. Enable at least one config in the Amplification tab."
            )
            return

        # Prepare tokenized prompts
        tokenized_prompts = []
        for mp in active_prompts:
            if mp.editor_mode == "simple":
                tokenized = self._tokenize_simple_prompt(mp)
            else:
                tokenized = self._tokenize_chat_prompt(mp)
            tokenized_prompts.append(tokenized)

        # Run batched generation
        sampling_params = self._get_sampling_params()
        results = {}

        with st.spinner(
            f"Generating for {len(active_prompts)} prompts × {len(active_configs)} configs..."
        ):
            for gen_result in self.method.multi_gen_request(
                prompt=tokenized_prompts,
                amplification_configs=active_configs,
                sampling_params=sampling_params,
                compiled_adapters_dir=COMPILED_ADAPTERS_DIR,
            ):
                config = gen_result["config"]
                # results is 2D: [prompt_idx][sample_idx]
                results[config.config_id] = {
                    "config": config,
                    "results": gen_result["results"],  # 2D list
                }

        # Store results
        st.session_state.multi_prompt_results = {
            "prompts": active_prompts,
            "config_results": results,
        }

        # Default display to first 1-2 configs
        config_ids = list(results.keys())
        st.session_state.multi_prompt_display_configs = config_ids[:2]

        st.success("Generation complete! Switch to Results tab to view.")

    def _tokenize_simple_prompt(self, mp: ManagedPrompt) -> list[int]:
        """Tokenize a simple-mode prompt."""
        if mp.template_mode == "No template":
            return self.tokenizer.encode(mp.prompt_text, add_special_tokens=False)
        elif mp.template_mode == "Apply chat template":
            messages = []
            if mp.system_prompt:
                messages.append({"role": "system", "content": mp.system_prompt})
            messages.append({"role": "user", "content": mp.prompt_text})
            if mp.assistant_prefill:
                messages.append({"role": "assistant", "content": mp.assistant_prefill})
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=not mp.assistant_prefill,
                tokenize=True,
            )
        else:  # Apply loom template
            filename = mp.loom_filename or "untitled.txt"
            messages = [
                {
                    "role": "system",
                    "content": "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command.",
                },
                {"role": "user", "content": f"<cmd>cat {filename}</cmd>"},
                {"role": "assistant", "content": mp.prompt_text},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                continue_final_message=True,
                tokenize=True,
            )

    def _tokenize_chat_prompt(self, mp: ManagedPrompt) -> list[int]:
        """Tokenize a chat-mode prompt."""
        if not mp.messages:
            return []
        return self.tokenizer.apply_chat_template(
            mp.messages,
            add_generation_prompt=True,
            tokenize=True,
        )

    def _render_multi_prompt_results_subtab(self) -> None:
        """Render the results subtab."""
        if st.session_state.multi_prompt_results is None:
            st.info("No results yet. Go to 'Prompts' tab and click 'Run Active'.")
            return

        results_data = st.session_state.multi_prompt_results
        prompts = results_data["prompts"]
        config_results = results_data["config_results"]

        if not config_results:
            st.warning("No results available.")
            return

        # Config selector
        all_config_ids = list(config_results.keys())
        config_names = {
            cid: config_results[cid]["config"].config.name for cid in all_config_ids
        }

        st.write("**Select configs to display (1-3):**")
        selected = st.multiselect(
            "Display configs",
            options=all_config_ids,
            default=st.session_state.multi_prompt_display_configs[:3],
            format_func=lambda x: config_names.get(x, x),
            max_selections=3,
            label_visibility="collapsed",
        )
        st.session_state.multi_prompt_display_configs = selected

        if not selected:
            st.info("Select at least one config to display results.")
            return

        st.divider()

        # Display results
        num_configs = len(selected)

        for prompt_idx, mp in enumerate(prompts):
            display_name = mp.get_display_name() or f"Prompt {prompt_idx + 1}"

            with st.expander(f"📝 {display_name}", expanded=True):
                if num_configs == 1:
                    # Single config: 2-column layout like multi-gen
                    config_id = selected[0]
                    config_name = config_names[config_id]
                    samples = config_results[config_id]["results"][prompt_idx]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{config_name}**")
                    with col2:
                        render_sample_cycler(
                            samples=samples,
                            component_id=f"mp_cycler_{prompt_idx}_{config_id}",
                            height=300,
                        )
                else:
                    # Multiple configs: vertical layout, horizontal config columns
                    cols = st.columns(num_configs)
                    for col_idx, config_id in enumerate(selected):
                        config_name = config_names[config_id]
                        samples = config_results[config_id]["results"][prompt_idx]

                        with cols[col_idx]:
                            st.write(f"**{config_name}**")
                            render_sample_cycler(
                                samples=samples,
                                component_id=f"mp_cycler_{prompt_idx}_{config_id}",
                                height=250,
                            )
