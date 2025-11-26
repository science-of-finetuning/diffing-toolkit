"""
Streamlit dashboard for weight difference amplification.

Provides UI for creating, editing, and testing amplification configurations.
"""

from copy import deepcopy
import html
import json
import re
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import streamlit.components.v1 as components
import yaml

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
    sanitize_config_name,
    get_unique_name,
    save_configs_to_cache,
    load_configs_from_cache,
    save_multigen_state,
    load_multigen_state,
    save_conversation,
    load_conversations_from_cache,
    delete_conversation_file,
)
from src.diffing.methods.amplification.weight_amplification import (
    WeightDifferenceAmplification,
)

CACHE_DIR = PROJECT_ROOT / ".streamlit_cache" / "amplification_cache"
CONFIGS_DIR = CACHE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR = CACHE_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
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

        max_num_seqs = max(((num_configs + 7) // 8) * 8, 8)
        max_loras = max(num_configs, 16)

        all_adapter_ids = set()
        base_model_name = self.method.base_model_cfg.name
        for mc in active_configs:
            for adapter in mc.config.amplified_adapters:
                all_adapter_ids.add(adapter.adapter_id(base_model_name))
        max_lora_rank = 64
        if all_adapter_ids:
            ranks = [self._get_adapter_rank_cached(aid) for aid in all_adapter_ids]
            max_lora_rank = max(ranks) * 2

        self.inference_config.vllm_kwargs["max_num_seqs"] = max_num_seqs
        self.inference_config.vllm_kwargs["max_loras"] = max_loras
        self.inference_config.vllm_kwargs["max_lora_rank"] = max_lora_rank

    def _shutdown_vllm_server(self) -> None:
        container = _get_vllm_server_container()
        if container["server"] is not None:
            del container["server"]
            cleanup_dist_env_and_memory()
            container["server"] = None
            container["config"] = None
            kill_vllm_process()

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

        if "multi_gen_prompt" not in st.session_state:
            st.session_state.multi_gen_prompt = saved_multigen_state.get("prompt", "")
        if "apply_chat_template_checkbox" not in st.session_state:
            st.session_state.apply_chat_template_checkbox = saved_multigen_state.get(
                "apply_chat_template", True
            )

        self._load_configs_from_cache()
        self._load_conversations_from_cache()

    def _get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters from sidebar/session state."""
        params = deepcopy(st.session_state["sampling_params"])
        do_sample = params.pop("do_sample", True)
        if not do_sample:
            params["temperature"] = 0
        return SamplingParams(**params)

    def _load_configs_from_cache(self) -> None:
        """Load configs from the cache directory."""
        if len(st.session_state.managed_configs) > 0:
            return

        loaded = load_configs_from_cache(CONFIGS_DIR)
        st.session_state.managed_configs.update(loaded)

    def _save_last_multigen_state(self) -> None:
        """Save current multi-gen state to cache."""
        state = {
            "active_tab": st.session_state.get("multi_gen_active_tab", "Text"),
            "text_tab": {
                "prompt": st.session_state.get("multi_gen_text_prompt", ""),
                "template_mode": st.session_state.get(
                    "multi_gen_template_mode", "Apply chat template"
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

    def _save_and_rerun(self) -> None:
        """Save configs to cache and trigger a Streamlit rerun."""
        save_configs_to_cache(st.session_state.managed_configs, CONFIGS_DIR)
        st.rerun()

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

        return self.tokenizer.apply_chat_template(
            conv["history"],
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

    def _single_gen_request(
        self,
        prompt: list[int],
        config: ManagedConfig,
        sampling_params,
    ) -> str:
        """Generate text for a single configuration."""
        return self.method.single_gen_request(
            prompt=prompt,
            config=config,
            sampling_params=sampling_params,
            compiled_adapters_dir=COMPILED_ADAPTERS_DIR,
            vllm_server=self.multi_lora_vllm_server,
        )

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
        st.sidebar.info("TODO: vLLM engine args")
        st.sidebar.success(
            "If your vllm server crashes, try to press the shutdown button!"
        )

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
        num_samples = st.sidebar.number_input(
            "Num Samples",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Number of completions to generate per config (for cycling through)",
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
        skip_special_tokens = st.sidebar.checkbox(
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

        st.sidebar.header("Global Controls")

        col1, col2 = st.sidebar.columns(2)
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

        st.session_state.multi_gen_current_prompt = prompt
        st.session_state.multi_gen_current_template_mode = template_mode

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
                st.rerun()

    def _render_message_list(self) -> None:
        """Render list of current messages with edit/delete."""
        messages = st.session_state.get("multi_gen_messages", [])
        editing_idx = st.session_state.get("multi_gen_msg_editing_idx", None)

        if not messages:
            st.info("No messages yet. Add your first message below.")
            return

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
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key=f"cancel_{idx}"):
                        st.session_state.multi_gen_msg_editing_idx = None
                        st.rerun()
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

                    content_preview = msg["content"]
                    if len(content_preview) > 300:
                        with st.expander("Show full message"):
                            st.text(msg["content"])
                        content_preview = content_preview[:300] + "..."

                    st.text(content_preview)

                    col1, col2, col3 = st.columns([1, 1, 10])
                    with col1:
                        if st.button("✏️", key=f"edit_btn_{idx}"):
                            st.session_state.multi_gen_msg_editing_idx = idx
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"delete_btn_{idx}"):
                            messages.pop(idx)
                            st.rerun()

    def _render_add_message_section(self) -> None:
        """Render UI for adding new messages."""
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
                    st.rerun()
                else:
                    st.warning("Message content cannot be empty")

    def _render_generation_settings(self) -> None:
        """Render generation settings for message builder."""
        messages = st.session_state.get("multi_gen_messages", [])

        template_override = st.selectbox(
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

        if template_override == "No template override":
            if not messages:
                mode = "🟢 Will add generation prompt (default)"
            else:
                last_role = messages[-1]["role"]
                if last_role == "assistant":
                    mode = "🟠 Will continue final message (last is assistant)"
                else:
                    mode = "🟢 Will add generation prompt (last is user/system)"
        elif template_override == "Force generation prompt":
            mode = "🟢 Force generation prompt"
        elif template_override == "Force continue final message":
            mode = "🟠 Force continue final message"
        else:
            mode = "⚪ Force send as is (no special tokens)"

        st.markdown(f"**{mode}**")

    def _render_message_builder_tab(self) -> None:
        """Render structured message builder interface."""
        self._render_import_conversations_section()

        st.markdown("---")

        self._render_message_list()

        st.markdown("---")

        self._render_add_message_section()

        st.markdown("---")

        self._render_generation_settings()

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
            st.session_state.managed_configs[new_managed.config_id] = new_managed
            self._save_and_rerun()

        st.markdown("---")

        if len(st.session_state.managed_configs) == 0:
            st.info(
                "No amplification configurations yet. Click 'New Amplification' to create one."
            )
        else:
            for config_id, mc in st.session_state.managed_configs.items():
                self._render_amplification_config(config_id, mc)

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
                self._save_and_rerun()
        return clicked

    def _render_multi_generation_tab(self) -> None:
        """Render Tab 2: Multi-generation interface."""
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

                if template_mode == "No template":
                    final_prompt = self.tokenizer.encode(prompt)
                elif template_mode == "Apply chat template":
                    final_prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                    )
                elif template_mode == "Apply loom template":
                    final_prompt = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command.",
                            },
                            {"role": "user", "content": "<cmd>cat untitled.txt</cmd>"},
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
            for idx, result_data in enumerate(
                self._multi_gen_request(
                    prompt=final_prompt,
                    amplification_configs=active_configs,
                    sampling_params=sampling_params,
                )
            ):
                results.append(result_data)
                # Update placeholder with preview in expander
                preview = result_data["results"][0]
                num_samples = len(result_data["results"])
                title = f"✅ ({idx + 1}) {result_data['config'].name}"
                if num_samples > 1:
                    title += f" [{num_samples} samples]"
                with placeholders[idx].container():
                    with st.expander(title, expanded=True):
                        # Use text_area for scrollable, consistent-height preview
                        st.text_area(
                            "Preview",
                            value=preview,
                            height=250,
                            label_visibility="collapsed",
                        )

            st.session_state.multi_gen_results = {
                "prompt": original_prompt,
                "final_prompt": final_prompt,
                "results": results,
                "active_tab": active_tab,
            }
            # Reset sample indices when new results are generated
            st.session_state.multi_gen_sample_indices = {
                i: 0 for i in range(len(results))
            }
            st.rerun()  # Rerun to render full interactive cards

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
        """Render a single result card with sample cycling. Fragment for fast action buttons."""
        num_samples = len(result_data["results"])
        formatted_title = f"({idx + 1}) {result_data['config'].name}"

        with st.expander(formatted_title, expanded=True):
            render_sample_cycler(
                samples=result_data["results"],
                component_id=f"cycler_{idx}",
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
                    key=f"action_sample_{idx}",
                )
                is_all_samples = action_sample_idx == -1
            else:
                action_sample_idx = 0
                is_all_samples = False

            effective_idx = 0 if is_all_samples else action_sample_idx
            current_result = result_data["results"][effective_idx]
            current_tokens = result_data["output_tokens"][effective_idx]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(
                    "➕ Continue",
                    key=f"continue_{idx}",
                    use_container_width=True,
                ):
                    sampling_params = self._get_sampling_params()

                    if is_all_samples:
                        # Continue all samples
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

                    st.rerun(scope="app")  # Full page rerun to refresh HTML component

            with col2:
                if st.button(
                    "🔄 Regenerate",
                    key=f"regenerate_{idx}",
                    use_container_width=True,
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
                    st.rerun(scope="app")

            with col3:
                if st.button(
                    "💬 Continue Chat",
                    key=f"continue_chat_{idx}",
                    use_container_width=True,
                    disabled=is_all_samples,
                    help=(
                        "Select a specific sample to continue chat"
                        if is_all_samples
                        else None
                    ),
                ):
                    conv_id = f"conv_{st.session_state.conversation_counter}"
                    st.session_state.conversation_counter += 1

                    conv_name = self._get_unique_conversation_name(
                        f"{result_data['config'].name}"
                    )

                    if results_data.get("active_tab") == "Messages":
                        history = [
                            {k: v for k, v in msg.items() if k in ["role", "content"]}
                            for msg in st.session_state.get("multi_gen_messages", [])
                        ]
                        history.append(
                            {
                                "role": "assistant",
                                "content": current_result,
                                "config_name": result_data["config"].name,
                            }
                        )
                    else:
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
                        },
                        "history": history,
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

    def _render_single_conversation(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """Render a single conversation."""
        config = conv["context"]["config"]

        if conv["regenerating_from"] is not None:
            regen_index = conv["regenerating_from"]
            conv["regenerating_from"] = None

            prompt = self._truncate_history_and_get_prompt(conv, regen_index)

            sampling_params = self._get_sampling_params()

            managed_config = next(
                mc
                for mc in st.session_state.managed_configs.values()
                if mc.config.name == config.name
            )

            config_label = f"[{config.name}]" if config else "[No Config]"
            with st.chat_message("assistant"):
                st.write(f"**{config_label}**")
                with st.spinner("Regenerating..."):
                    response = self._single_gen_request(
                        prompt=prompt,
                        config=managed_config,
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

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            new_name = st.text_input(
                "Conversation Name",
                value=conv["name"],
                key=f"conv_name_{conv_id}",
            )
            if new_name != conv["name"]:
                delete_conversation_file(conv["name"], CONVERSATIONS_DIR)
                unique_name = self._get_unique_conversation_name(
                    new_name, exclude_conv_id=conv_id
                )
                conv["name"] = unique_name
                self._save_conversation(conv_id, conv)

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

                    selected_config_name = st.selectbox(
                        "Config",
                        options=config_names,
                        index=current_index,
                        key=f"conv_config_{conv_id}",
                    )

                    if selected_config_name != config.name:
                        new_managed_config = next(
                            mc
                            for mc in all_mcs
                            if mc.config.name == selected_config_name
                        )

                        conv["context"]["config"] = new_managed_config
                        self._save_conversation(conv_id, conv)
                        st.success(f"Switched to {selected_config_name}")
                        self._save_and_rerun()
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

        st.markdown("---")

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
                                self._save_and_rerun()
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_user_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                self._save_and_rerun()
                    else:
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
                                self._save_and_rerun()
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_asst_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                self._save_and_rerun()
                    else:
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

        send_to_multi_gen = st.checkbox(
            "🚀 Send next message to Multi-Generation",
            key=f"multi_gen_mode_{conv_id}",
            help="When checked, your next message will be sent to Multi-Generation instead of this chat",
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

                full_prompt = self.tokenizer.apply_chat_template(
                    conv["history"],
                    add_generation_prompt=True,
                )

                sampling_params = self._get_sampling_params()

                managed_config = next(
                    mc
                    for mc in st.session_state.managed_configs.values()
                    if mc.config.name == config.name
                )

                config_label = f"[{config.name}]" if config else "[No Config]"
                with st.chat_message("assistant"):
                    st.write(f"**{config_label}**")
                    with st.spinner("Generating..."):
                        response = self._single_gen_request(
                            prompt=full_prompt,
                            config=managed_config,
                            sampling_params=sampling_params,
                        )
                    st.markdown(response)

                conv["history"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "config_name": config.name if config else "No Config",
                    }
                )
                self._save_conversation(conv_id, conv)

                self._save_and_rerun()

    def _render_amplification_config(self, config_id: str, mc: ManagedConfig) -> None:
        """Render one amplification config (expandable)."""
        config = mc.config
        icon = "✅" if mc.active else "❌"
        with st.expander(f"{icon} {config.name}", expanded=mc.expanded):
            col1, col2 = st.columns([3, 1])

            with col1:
                new_name = st.text_input(
                    "Configuration Name",
                    value=config.name,
                    key=f"config_name_{config_id}",
                )
                if new_name != config.name:
                    unique_name = self._get_unique_config_name(
                        new_name, exclude_config_id=config_id
                    )
                    st.session_state.managed_configs[config_id].config.name = (
                        unique_name
                    )
                    self._save_and_rerun()

            with col2:
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_config_{config_id}",
                    use_container_width=True,
                ):
                    del st.session_state.managed_configs[config_id]
                    self._save_and_rerun()

            config.description = st.text_area(
                "Description",
                value=config.description,
                key=f"config_desc_{config_id}",
                height=60,
            )

            current_active = mc.active
            mc.active = st.checkbox(
                "Active",
                value=mc.active,
                key=f"config_active_{config_id}",
                help="Only active configurations will be used for generation",
            )
            if mc.active != current_active:
                self._save_and_rerun()

            st.markdown("#### Adapters")

            if len(config.amplified_adapters) == 0:
                st.info("No adapters configured. Click 'Add Adapter' below.")
            else:
                for adapter_idx, adapter in enumerate(config.amplified_adapters):
                    self._render_adapter_amplification(config_id, adapter_idx, adapter)

            if st.button("➕ Add Adapter", key=f"add_adapter_{config_id}"):
                # Default to custom adapter (user can switch to organism if available)
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
                self._save_and_rerun()

    def _render_adapter_amplification(
        self,
        config_id: str,
        adapter_idx: int,
        adapter: AmplifiedAdapter,
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
                if st.button("🗑️", key=f"delete_adapter_{config_id}_{adapter_idx}"):
                    st.session_state.managed_configs[
                        config_id
                    ].config.amplified_adapters.pop(adapter_idx)
                    self._save_and_rerun()

            base_model_name = self.method.base_model_cfg.name

            col1, col2 = st.columns(2)

            with col1:
                available_organisms = get_available_organisms(
                    base_model_name=self.method.base_model_cfg.name, only_loras=True
                )
                organism_options = [CUSTOM_ADAPTER_ORGANISM] + available_organisms

                # Find current index
                if adapter.organism_name in organism_options:
                    current_index = organism_options.index(adapter.organism_name)
                else:
                    current_index = 0

                selected_organism = st.selectbox(
                    "Organism",
                    options=organism_options,
                    index=current_index,
                    key=f"organism_{config_id}_{adapter_idx}",
                    help="Select 'custom' to use a direct HuggingFace adapter ID, or choose an organism",
                )

                if selected_organism != adapter.organism_name:
                    adapter.organism_name = selected_organism
                    # Reset variant when organism changes
                    if selected_organism == CUSTOM_ADAPTER_ORGANISM:
                        adapter.variant = ""
                    else:
                        adapter.variant = "default"
                        self._save_and_rerun()

            with col2:
                # Variant selector (text input for custom, dropdown for organisms)
                if adapter.organism_name == CUSTOM_ADAPTER_ORGANISM:
                    adapter.variant = st.text_input(
                        "Adapter ID",
                        value=adapter.variant,
                        key=f"variant_{config_id}_{adapter_idx}",
                        help="HuggingFace adapter ID (e.g., 'hf_user/repo' or 'hf_user/repo/path/in/repo')",
                        placeholder="hf_user/adapter_repo",
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

                        adapter.variant = st.selectbox(
                            "Variant",
                            options=available_variants,
                            index=current_index,
                            key=f"variant_{config_id}_{adapter_idx}",
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
                        config_id, adapter_idx, layer_idx, layer_amp
                    )

            if st.button(
                "➕ Add Layer Spec", key=f"add_layer_{config_id}_{adapter_idx}"
            ):
                new_layer_amp = LayerAmplification(
                    layers="all",
                    module_amplifications=[],
                )
                adapter.layer_amplifications.append(new_layer_amp)
                self._save_and_rerun()

    def _render_layer_amplification(
        self,
        config_id: str,
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
                    key=f"delete_layer_{config_id}_{adapter_idx}_{layer_idx}",
                ):
                    st.session_state.managed_configs[
                        config_id
                    ].config.amplified_adapters[adapter_idx].layer_amplifications.pop(
                        layer_idx
                    )
                    self._save_and_rerun()

            # Determine initial radio index based on persisted state
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
                key=f"layer_mode_{config_id}_{adapter_idx}_{layer_idx}",
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
                    key=f"layer_single_{config_id}_{adapter_idx}_{layer_idx}",
                )
                layer_amp.layers = layer_num

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
                    key=f"layer_range_{config_id}_{adapter_idx}_{layer_idx}",
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
                    key=f"layer_list_{config_id}_{adapter_idx}_{layer_idx}",
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
                        config_id,
                        adapter_idx,
                        layer_idx,
                        module_idx,
                        module_amp,
                    )

            if st.button(
                "➕ Add Module",
                key=f"add_module_{config_id}_{adapter_idx}_{layer_idx}",
            ):
                new_module_amp = ModuleAmplification(modules="all", weight=1.0)
                layer_amp.module_amplifications.append(new_module_amp)
                self._save_and_rerun()

    def _render_module_amplification(
        self,
        config_id: str,
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
                key=f"module_mode_{config_id}_{adapter_idx}_{layer_idx}_{module_idx}",
            )

        with col2:
            module_amp.weight = st.slider(
                "Weight",
                min_value=-5.0,
                max_value=5.0,
                value=float(module_amp.weight),
                step=0.1,
                key=f"module_weight_{config_id}_{adapter_idx}_{layer_idx}_{module_idx}",
                help="Amplification factor (1.0 = no change, 2.0 = double, 0.5 = half)",
            )

        with col3:
            if st.button(
                "🗑️",
                key=f"delete_module_{config_id}_{adapter_idx}_{layer_idx}_{module_idx}",
            ):
                st.session_state.managed_configs[config_id].config.amplified_adapters[
                    adapter_idx
                ].layer_amplifications[layer_idx].module_amplifications.pop(module_idx)
                self._save_and_rerun()
