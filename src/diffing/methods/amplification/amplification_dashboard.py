"""
Streamlit dashboard for weight difference amplification.

Provides a multi-tab interface for configuring and testing LoRA weight amplifications:

Tabs:
    1. Amplifications (amplifications_tab.py): Create and manage amplification configs
       with folder organization. Configure adapters, layers, modules, and weights.

    2. Multi-Generation (multi_generation_tab.py): Generate text with multiple configs
       side-by-side. Supports text input and structured message building.

    3. Chat (chat_tab.py): Multi-conversation chat interface. Continue, regenerate,
       and edit messages. Supports multi-gen sampling within chat.

    4. Multi-Prompt (multi_prompt_tab.py): Batch generation across multiple prompts
       and configs. Organize prompts in folders with simple or chat-mode editors.

    5. Control (control_tab.py): HuggingFace Hub sync - push/load cache data.

Architecture:
    - Tab implementations in streamlit_components/ receive dashboard reference
    - Shared state via st.session_state (managed_configs, managed_prompts, etc.)
    - FolderManagerUI provides generic folder organization for configs/prompts
    - Generation uses vLLM server with multi-LoRA support
"""

from copy import deepcopy
import html
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import streamlit.components.v1 as components
from streamlit_tags import st_tags
from annotated_text import annotated_text, annotation
import yaml

from src.utils.configs import (
    PROJECT_ROOT,
)
from src.utils.vllm import (
    LLM,
    SamplingParams,
    cleanup_dist_env_and_memory,
    kill_vllm_process,
)
from src.utils.model import load_model_from_config, get_adapter_rank
from src.diffing.methods.amplification.amplification_config import (
    AmplificationConfig,
    patch_vllm,
)
from src.diffing.methods.amplification.streamlit_components.dashboard_state import (
    ManagedConfig,
    ManagedPrompt,
    sanitize_config_name,
    get_unique_name,
    save_configs_to_cache,
    save_configs_to_folder,
    load_configs_from_folder,
    unload_folder_configs,
    save_prompts_to_cache,
    save_prompts_to_folder,
    load_prompts_from_folder,
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
from src.diffing.methods.amplification.streamlit_components.folder_manager_ui import (
    FolderManagerConfig,
    FolderManagerUI,
)
from src.diffing.methods.amplification.streamlit_components.control_tab import (
    render_control_tab,
)
from src.diffing.methods.amplification.streamlit_components.amplifications_tab import (
    AmplificationsTab,
)
from src.diffing.methods.amplification.streamlit_components.multi_generation_tab import (
    MultiGenerationTab,
)
from src.diffing.methods.amplification.streamlit_components.chat_tab import (
    ChatTab,
)
from src.diffing.methods.amplification.streamlit_components.multi_prompt_tab import (
    MultiPromptTab,
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


def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convert hex color to rgba string with transparency.

    Args:
        hex_color: Hex color string (e.g., "#ff0000" or "ff0000")
        alpha: Opacity value between 0 and 1

    Returns:
        RGBA color string (e.g., "rgba(255, 0, 0, 0.4)")
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def apply_html_highlighting(
    text: str, selectors: list[dict], alpha: float = 0.4
) -> str:
    """Apply regex highlighting using HTML mark tags with multiple selectors.

    Args:
        text: The text to highlight
        selectors: List of {keywords: list[str], color: str, enabled: bool} dicts
        alpha: Opacity for highlight colors

    Returns:
        Text with matches wrapped in styled <mark> tags
    """
    if not selectors:
        return text

    # Filter to only enabled selectors
    enabled_selectors = [s for s in selectors if s.get("enabled", True)]
    if not enabled_selectors:
        return text

    # Collect all matches with their colors: (start, end, matched_text, color)
    all_matches = []
    for selector in enabled_selectors:
        keywords = selector.get("keywords", [])
        color = selector.get("color", "#ffff00")
        rgba_color = hex_to_rgba(color, alpha)

        for pattern in keywords:
            if not pattern:
                continue
            try:
                for match in re.finditer(f"({pattern})", text, flags=re.IGNORECASE):
                    all_matches.append(
                        (match.start(), match.end(), match.group(), rgba_color)
                    )
            except re.error:
                pass  # Invalid regex, skip

    if not all_matches:
        return text

    # Sort by start position, then by length (longer matches first for same start)
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build result, skipping overlapping matches
    result = []
    last_end = 0
    for start, end, matched_text, color in all_matches:
        if start < last_end:
            continue  # Skip overlapping match
        if start > last_end:
            result.append(html.escape(text[last_end:start]))
        result.append(
            f'<mark style="background-color: {color}; color: inherit;">{html.escape(matched_text)}</mark>'
        )
        last_end = end

    if last_end < len(text):
        result.append(html.escape(text[last_end:]))

    return "".join(result)


def get_annotated_segments(
    text: str, selectors: list[dict], alpha: float = 0.4
) -> list:
    """Split text into segments for annotated_text() display with multiple selectors.

    Args:
        text: The text to process
        selectors: List of {keywords: list[str], color: str, enabled: bool} dicts
        alpha: Opacity for highlight colors

    Returns:
        List of segments (strings and annotation tuples) for annotated_text()
    """
    if not selectors:
        return [text]

    # Filter to only enabled selectors
    enabled_selectors = [s for s in selectors if s.get("enabled", True)]
    if not enabled_selectors:
        return [text]

    # Collect all matches with their colors: (start, end, matched_text, color)
    all_matches = []
    for selector in enabled_selectors:
        keywords = selector.get("keywords", [])
        color = selector.get("color", "#ffff00")
        rgba_color = hex_to_rgba(color, alpha)

        for pattern in keywords:
            if not pattern:
                continue
            try:
                for match in re.finditer(f"({pattern})", text, flags=re.IGNORECASE):
                    all_matches.append(
                        (match.start(), match.end(), match.group(), rgba_color)
                    )
            except re.error:
                pass  # Invalid regex, skip

    if not all_matches:
        return [text]

    # Sort by start position, then by length (longer matches first for same start)
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build segments, skipping overlapping matches
    segments = []
    last_end = 0
    for start, end, matched_text, color in all_matches:
        if start < last_end:
            continue  # Skip overlapping match
        if start > last_end:
            segments.append(text[last_end:start])
        segments.append(annotation(matched_text, "", color))
        last_end = end

    if last_end < len(text):
        segments.append(text[last_end:])

    return segments if segments else [text]


def render_sample_cycler(
    samples: list[str], component_id: str, height: int = 400, escape_html: bool = True
) -> None:
    """Render an HTML component for cycling through samples with instant JS navigation.

    Args:
        samples: List of sample texts to display
        component_id: Unique ID for the HTML component
        height: Height of the component in pixels
        escape_html: Whether to escape HTML in samples. Set to False if samples
                     already contain safe HTML (e.g., from highlighting).
    """
    if escape_html:
        samples_html = "\n".join(
            f'<div class="sample-content" style="display: {"block" if i == 0 else "none"}">{html.escape(s)}</div>'
            for i, s in enumerate(samples)
        )
    else:
        samples_html = "\n".join(
            f'<div class="sample-content" style="display: {"block" if i == 0 else "none"}">{s}</div>'
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


def render_samples(
    samples: list[str], component_id: str, height: int = 400, show_all: bool = False
) -> None:
    """Render samples either as a cycler or as expanded markdown blocks.

    Applies keyword highlighting from session state if selectors are set.
    """
    selectors = st.session_state.get("highlight_selectors", [])

    if show_all:
        for sample_idx, sample in enumerate(samples):
            with st.expander(f"Sample {sample_idx + 1}", expanded=True):
                if selectors:
                    segments = get_annotated_segments(sample, selectors)
                    annotated_text(*segments)
                else:
                    st.markdown(sample)
    else:
        # Apply HTML highlighting for the cycler
        if selectors:
            highlighted_samples = [
                apply_html_highlighting(s, selectors) for s in samples
            ]
            # escape_html=False because apply_html_highlighting already escapes
            render_sample_cycler(
                highlighted_samples, component_id, height, escape_html=False
            )
        else:
            render_sample_cycler(samples, component_id, height)


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
        # Check env var to disable cudagraph LoRA specialization (for debugging)
        disable_cudagraph_lora = os.getenv("DISABLE_CUDAGRAPH_LORA", "0") == "1"
        compilation_config = (
            {"cudagraph_specialize_lora": False} if disable_cudagraph_lora else {}
        )
        self.inference_config.vllm_kwargs = (
            (self.inference_config.vllm_kwargs or {})
            | dict(
                max_num_seqs=16,
                enable_lora=True,
                max_loras=16,
                max_lora_rank=64,
            )
            | ({"compilation_config": compilation_config} if compilation_config else {})
        )
        patch_vllm()
        self._init_session_state()
        self._init_folder_managers()
        self._init_tabs()

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
        minimize_vllm_memory = st.session_state.get("minimize_vllm_memory", False)
        # num_configs = len(active_configs)

        # max_num_seqs = max(((num_configs + 7) // 8) * 8, 8)
        max_num_seqs = 4 if minimize_vllm_memory else 16
        max_loras = 1  # the expectation is that the requests with the second LoRA adapter will be run after all requests with the first adapter have finished. That could change in the future.

        all_adapter_ids = set()
        base_model_name = self.method.base_model_cfg.name
        for mc in active_configs:
            for adapter in mc.config.amplified_adapters:
                try:
                    all_adapter_ids.add(adapter.adapter_id(base_model_name))
                except ValueError as e:
                    raise ValueError(f"Error getting adapter ID for {mc.name}") from e
        max_lora_rank = 128
        if minimize_vllm_memory:
            max_lora_rank = 1
        if all_adapter_ids:
            ranks = [self._get_adapter_rank_cached(aid) for aid in all_adapter_ids]
            max_lora_rank = max(ranks)
            if not minimize_vllm_memory:
                max_lora_rank *= 2

        self.inference_config.vllm_kwargs["max_num_seqs"] = max_num_seqs
        self.inference_config.vllm_kwargs["max_loras"] = max_loras
        self.inference_config.vllm_kwargs["max_lora_rank"] = max_lora_rank
        self.inference_config.vllm_kwargs["gpu_memory_utilization"] = (
            st.session_state.get("gpu_memory_utilization", 0.95)
        )

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
                f"{p}: {container['config'].get(p, 'N/A')} -> {current_config[p]}"
                for p in current_config
                if p not in container["config"]
                or container["config"][p] != current_config[p]
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
                CACHE_DIR / "loaded_folders.yaml",
                configs_dir=CONFIGS_DIR,
                prompts_dir=PROMPTS_DIR,
            )
            st.session_state.loaded_folders = loaded_folders
            st.session_state.loaded_prompt_folders = loaded_prompt_folders
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "active_conversation_id" not in st.session_state:
            st.session_state.active_conversation_id = None
        if "conversation_counter" not in st.session_state:
            st.session_state.conversation_counter = 0
        # Load inference params (sampling + vLLM) from disk
        if "inference_params_loaded" not in st.session_state:
            from src.diffing.methods.amplification.dashboard_state import (
                load_inference_params,
            )

            inference_params = load_inference_params(
                CACHE_DIR / "inference_params.yaml"
            )
            st.session_state.sampling_params = inference_params["sampling_params"]
            st.session_state.gpu_memory_utilization = inference_params["vllm_params"][
                "gpu_memory_utilization"
            ]
            st.session_state.minimize_vllm_memory = inference_params["vllm_params"][
                "minimize_vllm_memory"
            ]
            st.session_state.inference_params_loaded = True

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
        if "multi_gen_show_all" not in st.session_state:
            st.session_state.multi_gen_show_all = False
        if "multi_prompt_show_all" not in st.session_state:
            st.session_state.multi_prompt_show_all = False

        # Keyword highlighting state - list of {keywords: list[str], color: str, enabled: bool}
        if "highlight_selectors" not in st.session_state:
            from src.diffing.methods.amplification.dashboard_state import (
                load_highlight_selectors,
            )

            st.session_state.highlight_selectors = load_highlight_selectors(
                CACHE_DIR / "highlight_selectors.yaml"
            )

        if "multi_gen_prompt" not in st.session_state:
            st.session_state.multi_gen_prompt = saved_multigen_state.get("prompt", "")
        if "apply_chat_template_checkbox" not in st.session_state:
            st.session_state.apply_chat_template_checkbox = saved_multigen_state.get(
                "apply_chat_template", True
            )

        self._load_configs_from_cache()
        self._load_prompts_from_cache()
        self._load_conversations_from_cache()

    def _init_folder_managers(self) -> None:
        """Initialize folder manager UI components for configs and prompts."""
        self._config_folder_manager = FolderManagerUI(
            FolderManagerConfig(
                base_dir=CONFIGS_DIR,
                loaded_folders_key="loaded_folders",
                items_key="managed_configs",
                item_type_label="config",
                widget_key_prefix="cfg_folder",
                load_from_folder=lambda base, folder: load_configs_from_folder(
                    base,
                    folder,
                    {mc.full_name for mc in st.session_state.managed_configs.values()},
                ),
                save_to_folder=save_configs_to_folder,
                unload_folder=unload_folder_configs,
                create_new_item=self._create_new_config,
                get_item_folder=lambda mc: mc.folder,
                save_loaded_folders=self._save_loaded_folders,
                save_items=self._save_configs,
                rerun_scope="fragment",
            )
        )

        self._prompt_folder_manager = FolderManagerUI(
            FolderManagerConfig(
                base_dir=PROMPTS_DIR,
                loaded_folders_key="loaded_prompt_folders",
                items_key="managed_prompts",
                item_type_label="prompt",
                widget_key_prefix="prompt_folder",
                load_from_folder=lambda base, folder: load_prompts_from_folder(
                    base, folder
                ),
                save_to_folder=save_prompts_to_folder,
                unload_folder=unload_folder_prompts,
                create_new_item=self._create_new_prompt,
                get_item_folder=lambda mp: mp.folder,
                save_loaded_folders=self._save_loaded_folders,
                save_items=self._save_prompts,
                rerun_scope="fragment",
            )
        )

    def _init_tabs(self) -> None:
        """Initialize tab component instances."""
        self._amplifications_tab = AmplificationsTab(self)
        self._multi_gen_tab = MultiGenerationTab(self)
        self._chat_tab = ChatTab(self)
        self._multi_prompt_tab = MultiPromptTab(self)

    def _create_new_config(self, folder: str | None) -> ManagedConfig:
        """Create a new amplification config in the given folder."""
        base_name = f"Config {len(st.session_state.managed_configs) + 1}"
        unique_name = self._get_unique_config_name(base_name, folder)
        new_config = AmplificationConfig(
            name=unique_name,
            description="",
            amplified_adapters=[],
        )
        return ManagedConfig.from_config(
            new_config, active=True, expanded=True, folder=folder
        )

    def _create_new_prompt(self, folder: str | None) -> ManagedPrompt:
        """Create a new prompt in the given folder."""
        return ManagedPrompt(active=True, expanded=True, folder=folder)

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
            existing_names.update(mc.full_name for mc in loaded.values())

    def _load_prompts_from_cache(self) -> None:
        """Load prompts from all loaded folders."""
        if len(st.session_state.managed_prompts) > 0:
            return

        for folder in st.session_state.loaded_prompt_folders:
            loaded = load_prompts_from_folder(PROMPTS_DIR, folder)
            st.session_state.managed_prompts.update(loaded)

    def _save_prompts(self, deleted: tuple[str, str] | None = None) -> None:
        """Save prompts to cache without triggering rerun.

        Args:
            deleted: Optional tuple of (folder, display_name) for explicitly deleted prompt
        """
        save_prompts_to_cache(st.session_state.managed_prompts, PROMPTS_DIR, deleted)

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
            mc.full_name: mc for mc in st.session_state.managed_configs.values()
        }
        conversations, max_conv_num = load_conversations_from_cache(
            CONVERSATIONS_DIR, config_name_to_managed
        )
        st.session_state.conversations.update(conversations)
        if max_conv_num >= 0:
            st.session_state.conversation_counter = max_conv_num + 1

    def _reload_all_data(self) -> None:
        """Reload all data from cache after HF sync."""
        # Reload folder state from disk
        loaded_folders, loaded_prompt_folders = load_loaded_folders(
            CACHE_DIR / "loaded_folders.yaml",
            configs_dir=CONFIGS_DIR,
            prompts_dir=PROMPTS_DIR,
        )
        st.session_state.loaded_folders = loaded_folders
        st.session_state.loaded_prompt_folders = loaded_prompt_folders

        # Clear and reload configs
        st.session_state.managed_configs = {}
        existing_names: set[str] = set()
        for folder in st.session_state.loaded_folders:
            loaded = load_configs_from_folder(CONFIGS_DIR, folder, existing_names)
            st.session_state.managed_configs.update(loaded)
            existing_names.update(mc.full_name for mc in loaded.values())

        # Clear and reload prompts
        st.session_state.managed_prompts = {}
        for folder in st.session_state.loaded_prompt_folders:
            loaded = load_prompts_from_folder(PROMPTS_DIR, folder)
            st.session_state.managed_prompts.update(loaded)

        # Clear and reload conversations
        st.session_state.conversations = {}
        config_name_to_managed = {
            mc.full_name: mc for mc in st.session_state.managed_configs.values()
        }
        conversations, max_conv_num = load_conversations_from_cache(
            CONVERSATIONS_DIR, config_name_to_managed
        )
        st.session_state.conversations.update(conversations)
        if max_conv_num >= 0:
            st.session_state.conversation_counter = max_conv_num + 1

    def _get_unique_config_name(
        self,
        desired_name: str,
        folder: str | None = None,
        exclude_config_id: str = None,
    ) -> str:
        """Get a unique configuration name within folder context."""
        sanitized_name = sanitize_config_name(desired_name)
        existing_names = set()
        for config_id, mc in st.session_state.managed_configs.items():
            if exclude_config_id is None or config_id != exclude_config_id:
                existing_names.add(mc.full_name)

        desired_full = f"{folder}/{sanitized_name}" if folder else sanitized_name
        unique_full = get_unique_name(desired_full, existing_names)

        # Extract just the name part (remove folder prefix if present)
        if folder and unique_full.startswith(f"{folder}/"):
            return unique_full[len(folder) + 1 :]
        return unique_full

    def _get_unique_conversation_name(
        self, desired_name: str, exclude_conv_id: str = None
    ) -> str:
        """Get a unique conversation name."""
        existing_names = set()
        for conv_id, conv in st.session_state.conversations.items():
            if exclude_conv_id is None or conv_id != exclude_conv_id:
                existing_names.add(conv["name"])
        return get_unique_name(desired_name, existing_names)

    def _get_unique_prompt_name(
        self,
        desired_name: str,
        folder: str | None = None,
        exclude_prompt_id: str = None,
    ) -> str:
        """Get a unique prompt name within folder context."""
        sanitized_name = sanitize_config_name(desired_name)
        existing_names = set()
        for prompt_id, mp in st.session_state.managed_prompts.items():
            if exclude_prompt_id is None or prompt_id != exclude_prompt_id:
                existing_names.add(mp.full_name)

        desired_full = f"{folder}/{sanitized_name}" if folder else sanitized_name
        unique_full = get_unique_name(desired_full, existing_names)

        # Extract just the name part (remove folder prefix if present)
        if folder and unique_full.startswith(f"{folder}/"):
            return unique_full[len(folder) + 1 :]
        return unique_full

    def _save_configs(self, deleted: tuple[str, str] | None = None) -> None:
        """Save configs to cache without triggering rerun.

        Args:
            deleted: Optional tuple of (folder, config_name) for explicitly deleted config
        """
        save_configs_to_cache(st.session_state.managed_configs, CONFIGS_DIR, deleted)

    def _save_and_rerun(self, scope: str = "app") -> None:
        """Save configs to cache and trigger a Streamlit rerun.

        Args:
            scope: Rerun scope - "app" for full page, "fragment" for current fragment only.
        """
        self._save_configs()
        st.rerun(scope=scope)

    def _save_inference_params(self) -> None:
        """Save inference params (sampling + vLLM) to cache file."""
        params = {
            "sampling_params": st.session_state.sampling_params,
            "vllm_params": {
                "gpu_memory_utilization": st.session_state.get(
                    "gpu_memory_utilization", 0.95
                ),
                "minimize_vllm_memory": st.session_state.get(
                    "minimize_vllm_memory", False
                ),
            },
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / "inference_params.yaml", "w") as f:
            yaml.dump(params, f, sort_keys=False)

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

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Amplifications", "Multi-Generation", "Chat", "Multi-Prompt", "Control"]
        )

        with tab1:
            self._amplifications_tab.render()
        with tab2:
            self._multi_gen_tab.render()
        with tab3:
            self._chat_tab.render()
        with tab4:
            self._multi_prompt_tab.render()
        with tab5:
            render_control_tab(on_reload=self._reload_all_data)

    def _render_sidebar(self) -> None:
        """Render sidebar with global controls."""
        # can't be a fragment "Fragments cannot write widgets to outside containers."
        with st.sidebar.expander("vLLM Configuration", expanded=True):
            st.info(f"**Model:** {self.method.base_model_cfg.model_id}")
            if st.button("Start vLLM Engine", use_container_width=True):
                _ = self.multi_lora_vllm_server
                st.success("vLLM server started.")
            if st.button(
                "Kill all vLLM engines on this machine", use_container_width=True
            ):
                killed = self._shutdown_vllm_server()
                if killed:
                    st.success("vLLM process killed.")
                else:
                    st.info("No vLLM process was running.")
            # st.info("TODO: vLLM engine args")
            st.info("If your vllm server crashes, try to press the shutdown button!")

            st.toggle(
                "Minimize VRAM usage",
                key="minimize_vllm_memory",
                help="If enabled, the vLLM server will use most conservative allocation of VRAM, which means that using new LoRAs will force to restart the server",
                on_change=self._save_inference_params,
            )

            st.slider(
                "GPU Memory Utilization",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="gpu_memory_utilization",
                help="Fraction of GPU memory to use for vLLM (0.0 to 1.0)",
                on_change=self._save_inference_params,
            )

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
            # Get current sampling params (loaded from disk or defaults)
            sp = st.session_state.sampling_params

            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=float(sp.get("temperature", 1.0)),
                step=0.1,
                help="Sampling temperature for generation",
            )
            top_p = st.slider(
                "Top-p (nucleus sampling)",
                min_value=0.0,
                max_value=1.0,
                value=float(sp.get("top_p", 0.9)),
                step=0.05,
                help="Nucleus sampling probability threshold",
            )
            max_tokens = st.slider(
                "Max New Tokens",
                min_value=10,
                max_value=500,
                value=int(sp.get("max_tokens", 100)),
                step=10,
                help="Maximum number of tokens to generate",
            )
            num_samples = st.slider(
                "Num Samples",
                min_value=1,
                max_value=16,
                value=int(sp.get("n", 6)),
                step=1,
                help="Number of completions to generate per config (for cycling through)",
            )
            do_sample = st.checkbox(
                "Use Sampling",
                value=bool(sp.get("do_sample", True)),
                help="Enable sampling (if disabled, uses greedy decoding)",
            )
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(sp.get("seed", 28)),
                step=9,
                help="Seed for random number generation",
            )
            skip_special_tokens = st.checkbox(
                "Skip Special Tokens",
                value=bool(sp.get("skip_special_tokens", False)),
                help="Skip special tokens in the generated text",
            )

            # Update sampling_params and save if any value changed
            new_params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": num_samples,
                "do_sample": do_sample,
                "seed": seed,
                "skip_special_tokens": skip_special_tokens,
            }
            if new_params != st.session_state.sampling_params:
                st.session_state.sampling_params = new_params
                self._save_inference_params()

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

        with st.sidebar.expander("Keyword Highlighting", expanded=False):
            # Default colors for new selectors
            default_colors = [
                "#ffff00",
                "#00ff00",
                "#ff00ff",
                "#00ffff",
                "#ff8000",
                "#8000ff",
            ]

            selectors = st.session_state.highlight_selectors
            selectors_to_delete = []

            for i, selector in enumerate(selectors):
                col_color, col_toggle, col_del = st.columns([2, 1, 1])
                with col_color:
                    new_color = st.color_picker(
                        f"Color {i+1}",
                        value=selector.get(
                            "color", default_colors[i % len(default_colors)]
                        ),
                        key=f"highlight_color_{i}",
                        label_visibility="collapsed",
                    )
                    selector["color"] = new_color
                with col_toggle:
                    enabled = st.toggle(
                        "Enable",
                        value=selector.get("enabled", True),
                        key=f"highlight_enabled_{i}",
                        label_visibility="collapsed",
                    )
                    selector["enabled"] = enabled
                with col_del:
                    if st.button("🗑️", key=f"del_selector_{i}", help="Delete selector"):
                        selectors_to_delete.append(i)

                new_keywords = st_tags(
                    label="",
                    text="Type keyword/regex, press Enter",
                    value=selector.get("keywords", []),
                    key=f"highlight_keywords_{i}",
                )
                selector["keywords"] = new_keywords

                if i < len(selectors) - 1:
                    st.divider()

            # Delete marked selectors (in reverse to preserve indices)
            for i in reversed(selectors_to_delete):
                selectors.pop(i)
            if selectors_to_delete:
                from src.diffing.methods.amplification.dashboard_state import (
                    save_highlight_selectors,
                )

                save_highlight_selectors(
                    CACHE_DIR / "highlight_selectors.yaml", selectors
                )
                st.rerun()

            # Add new selector button
            if st.button("➕ Add Selector", use_container_width=True):
                from src.diffing.methods.amplification.dashboard_state import (
                    save_highlight_selectors,
                )

                new_color = default_colors[len(selectors) % len(default_colors)]
                selectors.append({"keywords": [], "color": new_color, "enabled": True})
                save_highlight_selectors(
                    CACHE_DIR / "highlight_selectors.yaml", selectors
                )
                st.rerun()

            st.divider()
            if st.button("Refresh Highlighting", use_container_width=True):
                from src.diffing.methods.amplification.dashboard_state import (
                    save_highlight_selectors,
                )

                save_highlight_selectors(
                    CACHE_DIR / "highlight_selectors.yaml", selectors
                )
                st.rerun()

            # Auto-save any changes to selectors (color, enabled, keywords)
            # This runs on every rerun after widgets have updated session state
            from src.diffing.methods.amplification.dashboard_state import (
                save_highlight_selectors,
            )

            save_highlight_selectors(CACHE_DIR / "highlight_selectors.yaml", selectors)

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

            # Build options: name -> config_id mapping (None for no selection)
            config_options = {mc.full_name: cid for cid, mc in active_configs.items()}
            option_names = ["None"] + list(config_options.keys())

            # Initialize sidebar selection state if needed
            if "sidebar_quick_edit_config_id" not in st.session_state:
                st.session_state.sidebar_quick_edit_config_id = None

            # Get current selection name (if valid)
            current_id = st.session_state.sidebar_quick_edit_config_id
            current_name = "None"
            if current_id and current_id in active_configs:
                current_name = active_configs[current_id].full_name

            # Determine initial index
            initial_index = option_names.index(current_name)

            def on_config_select():
                selected_name = st.session_state["sidebar_config_selector"]
                st.session_state.sidebar_quick_edit_config_id = config_options.get(
                    selected_name
                )  # Returns None for "None" option

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
                self._amplifications_tab._render_amplification_config(
                    selected_id, mc, key_prefix="sidebar_", sidebar_mode=True
                )
