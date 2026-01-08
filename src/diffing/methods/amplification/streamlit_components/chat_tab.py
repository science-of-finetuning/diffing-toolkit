"""
Chat Tab - Tab 3 of the amplification dashboard.

Provides a multi-conversation chat interface with continuation, regeneration,
and message editing. Supports multi-gen sampling within chat.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List

import streamlit as st

from .utils import (
    get_sampling_params,
    get_unique_conversation_name,
)

if TYPE_CHECKING:
    from src.diffing.methods.amplification.amplification_dashboard import (
        AmplificationDashboard,
    )


class ChatTab:
    """Renders Tab 3: Chat interface with multiple conversations.

    Multi-conversation chat interface. Continue, regenerate, and edit messages.
    Supports multi-gen sampling within chat.
    """

    def __init__(self, dashboard: "AmplificationDashboard"):
        self.dashboard = dashboard

    def _get_messages_with_system_prompt(
        self, conv: Dict[str, Any], messages: List[Dict[str, Any]] | None = None
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
        return self.dashboard.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

    def render(self) -> None:
        """Render the Chat tab."""
        if not st.session_state.conversations:
            st.info(
                "💬 No conversations yet. Generate from the Multi-Generation tab and click 'Continue Chat', or use the 'New' tab to start an empty chat."
            )

        if st.button("➕ Start New Chat", type="primary"):
            self._create_new_conversation()
            self.dashboard.persistence.save_configs_and_rerun()
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

        conv_name = get_unique_conversation_name(
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
            "multi_gen_enabled": False,
        }
        self.dashboard.persistence.save_conversation(
            conv_id, st.session_state.conversations[conv_id]
        )
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
                mc.full_name for mc in st.session_state.managed_configs.values()
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
                    if mc.full_name == selected_config_name
                )
                self._create_new_conversation(config=config, name=conv_name)
                st.success(f"Created conversation: {conv_name}")
                self.dashboard.persistence.save_configs_and_rerun()
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
                                self.dashboard.persistence.save_conversation(
                                    conv_id, conv
                                )
                                st.rerun(scope="fragment")
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_user_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                st.rerun(scope="fragment")
                    else:
                        st.markdown(msg["content"])
                        _, btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(
                            [10, 1, 1, 1, 1]
                        )
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
                                "➕",
                                key=f"continue_btn_user_{conv_id}_{i}",
                                help="Continue this message",
                                type="secondary",
                            ):
                                conv["continuing_from"] = i
                                st.rerun(scope="app")
                        with btn_col3:
                            if st.button(
                                "🔄",
                                key=f"regen_btn_user_{conv_id}_{i}",
                                help="Regenerate assistant response",
                                type="secondary",
                            ):
                                # Store user message index - regeneration handler will generate response
                                conv["regenerating_from_user"] = i
                                st.rerun(scope="app")
                        with btn_col4:
                            if st.button(
                                "🗑️",
                                key=f"delete_btn_user_{conv_id}_{i}",
                                help="Delete message",
                                type="secondary",
                            ):
                                conv["history"].pop(i)
                                self.dashboard.persistence.save_conversation(
                                    conv_id, conv
                                )
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
                                self.dashboard.persistence.save_conversation(
                                    conv_id, conv
                                )
                                st.rerun(scope="fragment")
                        with bcol2:
                            if st.button("Cancel", key=f"cancel_asst_{conv_id}_{i}"):
                                conv["editing_message"] = None
                                st.rerun(scope="fragment")
                    else:
                        config_label = f"[{msg.get('config_name', config.full_name if config else 'No Config')}]"
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
                                self.dashboard.persistence.save_conversation(
                                    conv_id, conv
                                )
                                st.rerun(scope="fragment")

    @st.fragment
    def _render_single_conversation(self, conv_id: str, conv: Dict[str, Any]) -> None:
        """Render a single conversation. Fragment for independent updates."""

        config = conv["context"]["config"]
        pending_key = f"chat_pending_samples_{conv_id}"

        # Initialize multi-gen toggle state from conversation dict (ensures persistence across reruns)
        multi_gen_key = f"chat_multi_gen_{conv_id}"
        # Handle older conversations that don't have multi_gen_enabled field
        if "multi_gen_enabled" not in conv:
            conv["multi_gen_enabled"] = False
        # Always sync session state with conversation dict on render
        st.session_state[multi_gen_key] = conv["multi_gen_enabled"]

        if conv["regenerating_from"] is not None:
            self._handle_regenerating_from(conv_id, conv, config, pending_key)

        if conv.get("regenerating_from_user") is not None:
            self._handle_regenerating_from_user(conv_id, conv, config, pending_key)

        if conv.get("continuing_from") is not None:
            self._handle_continuing_from(conv_id, conv, config, pending_key)

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            conv_name_key = f"conv_name_{conv_id}"

            def on_conv_name_change(conversation=conv, cid=conv_id, key=conv_name_key):
                new_name = st.session_state[key]
                if new_name != conversation["name"]:
                    self.dashboard.persistence.delete_conversation(conversation["name"])
                    unique_name = get_unique_conversation_name(
                        new_name, exclude_conv_id=cid
                    )
                    conversation["name"] = unique_name
                    self.dashboard.persistence.save_conversation(cid, conversation)

            st.text_input(
                "Conversation Name",
                value=conv["name"],
                key=conv_name_key,
                on_change=on_conv_name_change,
            )

        with col2:
            all_mcs = list(st.session_state.managed_configs.values())
            config_names = [mc.full_name for mc in all_mcs]
            if config_names:
                current_index = next(
                    (
                        i
                        for i, mc in enumerate(all_mcs)
                        if config and mc.full_name == config.full_name
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
                    new_mc = next(mc for mc in mcs if mc.full_name == selected_name)
                    conversation["context"]["config"] = new_mc
                    self.dashboard.persistence.save_conversation(cid, conversation)

                st.selectbox(
                    "Config",
                    options=config_names,
                    index=current_index,
                    key=config_select_key,
                    on_change=on_config_change,
                )
            else:
                st.info("No configs available")

        with col3:
            if st.button(
                "🗑️ Delete", key=f"delete_conv_{conv_id}", use_container_width=True
            ):
                self.dashboard.persistence.delete_conversation(conv["name"])
                del st.session_state.conversations[conv_id]
                if st.session_state.active_conversation_id == conv_id:
                    st.session_state.active_conversation_id = None
                self.dashboard.persistence.save_configs_and_rerun()

        # System prompt section
        system_prompt_key = f"system_prompt_{conv_id}"

        def on_system_prompt_change(
            conversation=conv, cid=conv_id, key=system_prompt_key
        ):
            conversation["context"]["system_prompt"] = st.session_state[key]
            self.dashboard.persistence.save_conversation(cid, conversation)

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

        def on_multi_gen_toggle_change(
            conversation=conv, conversation_id=conv_id, key=multi_gen_key
        ):
            conversation["multi_gen_enabled"] = st.session_state[key]
            self.dashboard.persistence.save_conversation(conversation_id, conversation)

        st.toggle(
            "Multi-gen in Chat",
            help="When enabled and Num Samples > 1, show all samples and let you select one",
            disabled=send_to_multi_gen,
            key=f"chat_multi_gen_{conv_id}",
            on_change=on_multi_gen_toggle_change,
        )

        user_input = st.chat_input(
            "Type your message here...", key=f"chat_input_{conv_id}"
        )

        if user_input:
            self._handle_user_input(
                conv_id, conv, config, user_input, send_to_multi_gen, pending_key
            )

    def _handle_regenerating_from(
        self, conv_id: str, conv: Dict[str, Any], config, pending_key: str
    ) -> None:
        """Handle regeneration from an assistant message."""
        from .dashboard_state import (
            GenerationLog,
        )

        regen_index = conv["regenerating_from"]
        conv["regenerating_from"] = None

        prompt = self._truncate_history_and_get_prompt(conv, regen_index)

        sampling_params = get_sampling_params()
        use_multi_gen = (
            st.session_state.get(f"chat_multi_gen_{conv_id}", False)
            and sampling_params.n > 1
        )

        managed_config = next(
            mc
            for mc in st.session_state.managed_configs.values()
            if mc.full_name == config.full_name
        )

        if use_multi_gen:
            with st.spinner(f"Regenerating {sampling_params.n} samples..."):
                result = next(
                    self.dashboard._multi_gen_request(
                        prompt=prompt,
                        amplification_configs=[managed_config],
                        sampling_params=sampling_params,
                    )
                )

            GenerationLog.from_dashboard_generation(
                generation_type="regenerate",
                model_id=self.dashboard.method.base_model_cfg.model_id,
                prompt_text=self.dashboard.tokenizer.decode(
                    prompt, skip_special_tokens=False
                ),
                prompt_tokens=prompt,
                sampling_params=sampling_params,
                configs=[managed_config],
                results=[
                    {"config_name": config.full_name, "outputs": result["results"]}
                ],
                messages=self._get_messages_with_system_prompt(conv),
                logs_dir=self.dashboard.persistence.logs_dir,
            )

            st.session_state[pending_key] = {
                "samples": result["results"],
                "config_name": config.full_name if config else "No Config",
                "mode": "replace",
            }
            self.dashboard.persistence.save_conversation(conv_id, conv)
        else:
            with st.spinner("Regenerating..."):
                result = next(
                    self.dashboard._multi_gen_request(
                        prompt=prompt,
                        amplification_configs=[managed_config],
                        sampling_params=sampling_params,
                    )
                )
                response = result["results"][0]

            if response:
                conv["history"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "config_name": config.full_name if config else "No Config",
                    }
                )
                self.dashboard.persistence.save_conversation(conv_id, conv)

                GenerationLog.from_dashboard_generation(
                    generation_type="regenerate",
                    model_id=self.dashboard.method.base_model_cfg.model_id,
                    prompt_text=self.dashboard.tokenizer.decode(
                        prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[{"config_name": config.full_name, "outputs": [response]}],
                    messages=self._get_messages_with_system_prompt(conv),
                    logs_dir=self.dashboard.persistence.logs_dir,
                )

    def _handle_regenerating_from_user(
        self, conv_id: str, conv: Dict[str, Any], config, pending_key: str
    ) -> None:
        """Handle regeneration from a user message."""
        from .dashboard_state import (
            GenerationLog,
        )

        user_index = conv["regenerating_from_user"]
        conv["regenerating_from_user"] = None

        # Truncate history to include only messages up to and including this user message
        conv["history"] = conv["history"][: user_index + 1]

        # Build prompt for generation
        messages = self._get_messages_with_system_prompt(conv)
        prompt = self.dashboard.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        sampling_params = get_sampling_params()
        use_multi_gen = (
            st.session_state.get(f"chat_multi_gen_{conv_id}", False)
            and sampling_params.n > 1
        )

        managed_config = next(
            mc
            for mc in st.session_state.managed_configs.values()
            if mc.full_name == config.full_name
        )

        if use_multi_gen:
            with st.spinner(f"Generating {sampling_params.n} samples..."):
                result = next(
                    self.dashboard._multi_gen_request(
                        prompt=prompt,
                        amplification_configs=[managed_config],
                        sampling_params=sampling_params,
                    )
                )

            GenerationLog.from_dashboard_generation(
                generation_type="regenerate",
                model_id=self.dashboard.method.base_model_cfg.model_id,
                prompt_text=self.dashboard.tokenizer.decode(
                    prompt, skip_special_tokens=False
                ),
                prompt_tokens=prompt,
                sampling_params=sampling_params,
                configs=[managed_config],
                results=[
                    {"config_name": config.full_name, "outputs": result["results"]}
                ],
                messages=messages,
                logs_dir=self.dashboard.persistence.logs_dir,
            )

            st.session_state[pending_key] = {
                "samples": result["results"],
                "config_name": config.full_name if config else "No Config",
                "mode": "add",
            }
            self.dashboard.persistence.save_conversation(conv_id, conv)
        else:
            with st.spinner("Generating..."):
                result = next(
                    self.dashboard._multi_gen_request(
                        prompt=prompt,
                        amplification_configs=[managed_config],
                        sampling_params=sampling_params,
                    )
                )
                response = result["results"][0]

            if response:
                conv["history"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "config_name": config.full_name if config else "No Config",
                    }
                )
                self.dashboard.persistence.save_conversation(conv_id, conv)

                GenerationLog.from_dashboard_generation(
                    generation_type="regenerate",
                    model_id=self.dashboard.method.base_model_cfg.model_id,
                    prompt_text=self.dashboard.tokenizer.decode(
                        prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[{"config_name": config.full_name, "outputs": [response]}],
                    messages=messages,
                    logs_dir=self.dashboard.persistence.logs_dir,
                )

    def _handle_continuing_from(
        self, conv_id: str, conv: Dict[str, Any], config, pending_key: str
    ) -> None:
        """Handle continuation from a message."""
        from .dashboard_state import (
            GenerationLog,
        )

        continue_index = conv["continuing_from"]
        conv["continuing_from"] = None

        # Build prompt including the message we're continuing
        messages = self._get_messages_with_system_prompt(
            conv, conv["history"][: continue_index + 1]
        )
        prompt = self.dashboard.tokenizer.apply_chat_template(
            messages,
            continue_final_message=True,
        )

        sampling_params = get_sampling_params()
        use_multi_gen = (
            st.session_state.get(f"chat_multi_gen_{conv_id}", False)
            and sampling_params.n > 1
        )

        managed_config = next(
            mc
            for mc in st.session_state.managed_configs.values()
            if mc.full_name == config.full_name
        )

        original_content = conv["history"][continue_index]["content"]

        if use_multi_gen:
            with st.spinner(f"Continuing with {sampling_params.n} samples..."):
                result = next(
                    self.dashboard._multi_gen_request(
                        prompt=prompt,
                        amplification_configs=[managed_config],
                        sampling_params=sampling_params,
                    )
                )

            GenerationLog.from_dashboard_generation(
                generation_type="continue",
                model_id=self.dashboard.method.base_model_cfg.model_id,
                prompt_text=self.dashboard.tokenizer.decode(
                    prompt, skip_special_tokens=False
                ),
                prompt_tokens=prompt,
                sampling_params=sampling_params,
                configs=[managed_config],
                results=[
                    {
                        "config_name": config.full_name,
                        "outputs": [original_content + c for c in result["results"]],
                    }
                ],
                messages=messages,
                logs_dir=self.dashboard.persistence.logs_dir,
            )

            st.session_state[pending_key] = {
                "samples": result["results"],
                "config_name": config.full_name if config else "No Config",
                "mode": "continue",
                "target_index": continue_index,
            }
            self.dashboard.persistence.save_conversation(conv_id, conv)
        else:
            with st.spinner("Continuing..."):
                result = next(
                    self.dashboard._multi_gen_request(
                        prompt=prompt,
                        amplification_configs=[managed_config],
                        sampling_params=sampling_params,
                    )
                )
                continuation = result["results"][0]

            if continuation:
                full_content = original_content + continuation
                conv["history"][continue_index]["content"] = full_content
                self.dashboard.persistence.save_conversation(conv_id, conv)

                GenerationLog.from_dashboard_generation(
                    generation_type="continue",
                    model_id=self.dashboard.method.base_model_cfg.model_id,
                    prompt_text=self.dashboard.tokenizer.decode(
                        prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[
                        {"config_name": config.full_name, "outputs": [full_content]}
                    ],
                    messages=messages,
                    logs_dir=self.dashboard.persistence.logs_dir,
                )

    def _handle_user_input(
        self,
        conv_id: str,
        conv: Dict[str, Any],
        config,
        user_input: str,
        send_to_multi_gen: bool,
        pending_key: str,
    ) -> None:
        """Handle new user input in the chat."""
        from .dashboard_state import (
            GenerationLog,
        )

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
            self.dashboard.persistence.save_configs_and_rerun()
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
            full_prompt = self.dashboard.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

            sampling_params = get_sampling_params()
            use_multi_gen = (
                st.session_state.get(f"chat_multi_gen_{conv_id}", False)
                and sampling_params.n > 1
            )

            managed_config = next(
                mc
                for mc in st.session_state.managed_configs.values()
                if mc.full_name == config.full_name
            )

            config_label = f"[{config.full_name}]" if config else "[No Config]"

            if use_multi_gen:
                with st.spinner(f"Generating {sampling_params.n} samples..."):
                    result = next(
                        self.dashboard._multi_gen_request(
                            prompt=full_prompt,
                            amplification_configs=[managed_config],
                            sampling_params=sampling_params,
                        )
                    )

                GenerationLog.from_dashboard_generation(
                    generation_type="chat",
                    model_id=self.dashboard.method.base_model_cfg.model_id,
                    prompt_text=self.dashboard.tokenizer.decode(
                        full_prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=full_prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[
                        {
                            "config_name": config.full_name,
                            "outputs": result["results"],
                        }
                    ],
                    messages=messages,
                    logs_dir=self.dashboard.persistence.logs_dir,
                )

                st.session_state[pending_key] = {
                    "samples": result["results"],
                    "config_name": config.full_name if config else "No Config",
                    "mode": "add",
                }
                self.dashboard.persistence.save_conversation(conv_id, conv)
                self.dashboard.persistence.save_configs_and_rerun(scope="fragment")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**{config_label}**")
                    with st.spinner("Generating..."):
                        result = next(
                            self.dashboard._multi_gen_request(
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
                        "config_name": config.full_name if config else "No Config",
                    }
                )
                self.dashboard.persistence.save_conversation(conv_id, conv)

                GenerationLog.from_dashboard_generation(
                    generation_type="chat",
                    model_id=self.dashboard.method.base_model_cfg.model_id,
                    prompt_text=self.dashboard.tokenizer.decode(
                        full_prompt, skip_special_tokens=False
                    ),
                    prompt_tokens=full_prompt,
                    sampling_params=sampling_params,
                    configs=[managed_config],
                    results=[{"config_name": config.full_name, "outputs": [response]}],
                    messages=messages,
                    logs_dir=self.dashboard.persistence.logs_dir,
                )

                self.dashboard.persistence.save_configs_and_rerun(scope="fragment")

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

                        self.dashboard.persistence.save_conversation(conv_id, conv)
                        self.dashboard.persistence.save_configs_and_rerun(
                            scope="fragment"
                        )
