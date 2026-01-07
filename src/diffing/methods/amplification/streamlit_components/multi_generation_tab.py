"""
Multi-Generation Tab - Tab 2 of the amplification dashboard.

Provides UI for generating text with multiple amplification configurations side-by-side.
Supports text input and structured message building.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any

import streamlit as st

if TYPE_CHECKING:
    from src.diffing.methods.amplification.amplification_dashboard import (
        AmplificationDashboard,
    )


class MultiGenerationTab:
    """Renders Tab 2: Multi-generation interface.

    Generate text with multiple amplification configurations side-by-side.
    Supports text input and structured message building.
    """

    def __init__(self, dashboard: "AmplificationDashboard"):
        self.dashboard = dashboard

    @st.fragment
    def render(self) -> None:
        """Render the Multi-Generation tab. Fragment for tab-level isolation."""
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

        # Show all samples checkbox - rendered once, outside conditional blocks
        st.checkbox("Show all samples", key="multi_gen_show_all")
        show_all = st.session_state.multi_gen_show_all

        if generate_clicked:
            self._run_generation(active_configs, show_all)

        if st.session_state.multi_gen_results is not None:
            self._render_results(show_all)

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
            system_prompt = st.text_area(
                "System prompt",
                key="multi_gen_system_prompt",
                placeholder="Optional: system instructions...",
                height=80,
            )
            assistant_prefill = st.text_area(
                "Assistant prefill",
                key="multi_gen_assistant_prefill",
                placeholder="Optional: prefill the assistant's response...",
                help="If not empty, this text will be added as the beginning of the assistant's response",
                height=80,
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

                        col1, col2, _ = st.columns([1, 1, 10])
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
                self.dashboard._save_and_rerun(scope="fragment")
        return clicked

    def _run_generation(self, active_configs, show_all: bool) -> None:
        """Run the generation process."""
        from src.diffing.methods.amplification.streamlit_components.dashboard_state import (
            GenerationLog,
        )
        from src.diffing.methods.amplification.amplification_dashboard import (
            LOGS_DIR,
        )

        self.dashboard._save_last_multigen_state()

        sampling_params = self.dashboard._get_sampling_params()

        active_tab = st.session_state.get("multi_gen_active_tab", "Text")
        template_mode = None
        loom_filename = "untitled.txt"

        if active_tab == "Text":
            prompt = st.session_state.multi_gen_current_prompt
            template_mode = st.session_state.multi_gen_current_template_mode
            system_prompt = st.session_state.get("multi_gen_current_system_prompt", "")
            assistant_prefill = st.session_state.get(
                "multi_gen_current_assistant_prefill", ""
            )
            loom_filename = st.session_state.get(
                "multi_gen_current_loom_filename", "untitled.txt"
            )

            if template_mode == "No template":
                final_prompt = self.dashboard.tokenizer.encode(prompt)
            elif template_mode == "Apply chat template":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                if assistant_prefill:
                    messages.append({"role": "assistant", "content": assistant_prefill})
                    final_prompt = self.dashboard.tokenizer.apply_chat_template(
                        messages,
                        continue_final_message=True,
                    )
                else:
                    final_prompt = self.dashboard.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                    )
            elif template_mode == "Apply loom template":
                final_prompt = self.dashboard.tokenizer.apply_chat_template(
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
                return
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

                final_prompt = self.dashboard.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_gen_prompt,
                    continue_final_message=continue_final,
                )
                original_prompt = f"[Conversation with {len(messages)} message(s)]"

        with st.expander("📋 Prompt", expanded=False):
            st.code(
                self.dashboard.tokenizer.decode(final_prompt, skip_special_tokens=False),
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
                    with st.expander(f"⏳ ({idx + 1}) {mc.full_name}", expanded=True):
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
            self.dashboard._multi_gen_request(
                prompt=final_prompt,
                amplification_configs=active_configs,
                sampling_params=sampling_params,
            )
        ):
            results.append(result_data)
            with placeholders[idx].container():
                self._render_result_card_content(
                    idx,
                    result_data,
                    results_data_in_progress,
                    disabled=True,
                    show_all=show_all,
                )

        st.session_state.multi_gen_results = results_data_in_progress

        # Log the generation
        GenerationLog.from_dashboard_generation(
            generation_type="multigen",
            model_id=self.dashboard.method.base_model_cfg.model_id,
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

    def _render_results(self, show_all: bool) -> None:
        """Render the stored generation results."""
        st.markdown("---")
        results_data = st.session_state.multi_gen_results
        with st.expander("📋 Prompt", expanded=False):
            st.code(
                self.dashboard.tokenizer.decode(
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
                with st.container():
                    self._render_result_card(
                        idx, result_data, results_data, show_all=show_all
                    )

    @st.fragment
    def _render_result_card(
        self, idx: int, result_data: dict, results_data: dict, show_all: bool = False
    ) -> None:
        """Fragment wrapper for result card - enables fast button interactions."""
        self._render_result_card_content(
            idx, result_data, results_data, disabled=False, show_all=show_all
        )

    def _render_result_card_content(
        self,
        idx: int,
        result_data: dict,
        results_data: dict,
        disabled: bool = False,
        show_all: bool = False,
    ) -> None:
        """Render a single result card with sample cycling."""
        from ..amplification_dashboard import (
            LOGS_DIR,
        )
        from .samples import render_samples
        from .dashboard_state import (
            GenerationLog,
        )

        num_samples = len(result_data["results"])
        formatted_title = f"({idx + 1}) {result_data['config'].full_name}"
        key_suffix = "_disabled" if disabled else ""

        with st.expander(formatted_title, expanded=True):
            render_samples(
                samples=result_data["results"],
                component_id=f"cycler_{idx}{key_suffix}",
                height=300,
                show_all=show_all,
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
                    sampling_params = self.dashboard._get_sampling_params()

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
                                self.dashboard._multi_gen_request(
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
                        model_id=self.dashboard.method.base_model_cfg.model_id,
                        prompt_text=results_data.get("prompt", ""),
                        prompt_tokens=results_data["final_prompt"],
                        sampling_params=sampling_params,
                        configs=[result_data["config"]],
                        results=[
                            {
                                "config_name": result_data["config"].name,
                                "outputs": [
                                    result_data["results"][i] for i in indices_to_continue
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
                    sampling_params = self.dashboard._get_sampling_params()

                    with st.spinner("Regenerating..."):
                        new_results = next(
                            self.dashboard._multi_gen_request(
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
                        model_id=self.dashboard.method.base_model_cfg.model_id,
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
                    self._create_chat_from_result(
                        result_data, results_data, current_result, template_mode
                    )

            with col4:
                if is_all_samples:
                    all_samples_text = "\n\n".join(
                        f"=== Sample {i + 1} ===\n{s}"
                        for i, s in enumerate(result_data["results"])
                    )
                    download_data = all_samples_text
                    safe_name = (
                        result_data["config"].full_name.replace("/", "_").replace(" ", "_")
                    )
                    download_filename = f"{safe_name}_all_samples.txt"
                else:
                    download_data = current_result
                    safe_name = (
                        result_data["config"].full_name.replace("/", "_").replace(" ", "_")
                    )
                    download_filename = f"{safe_name}_sample{effective_idx + 1}.txt"

                st.download_button(
                    label="📥 Download",
                    data=download_data,
                    file_name=download_filename,
                    mime="text/plain",
                    key=f"download_{idx}{key_suffix}",
                    use_container_width=True,
                    disabled=disabled,
                )

    def _create_chat_from_result(
        self, result_data: dict, results_data: dict, current_result: str, template_mode: str
    ) -> None:
        """Create a new chat conversation from a generation result."""
        conv_id = f"conv_{st.session_state.conversation_counter}"
        st.session_state.conversation_counter += 1

        conv_name = self.dashboard._get_unique_conversation_name(
            f"{result_data['config'].name}"
        )

        if results_data.get("active_tab") == "Messages":
            messages = st.session_state.get("multi_gen_messages", [])
            system_msgs = [m["content"] for m in messages if m["role"] == "system"]
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
            loom_filename = results_data.get("loom_filename", "untitled.txt")
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
            "multi_gen_enabled": False,
        }
        self.dashboard._save_conversation(conv_id, st.session_state.conversations[conv_id])
        st.session_state.active_conversation_id = conv_id
        st.success(
            f"✓ Chat started with {result_data['config'].name}. Now switch to the Chat tab to continue."
        )
        self.dashboard._save_and_rerun()
