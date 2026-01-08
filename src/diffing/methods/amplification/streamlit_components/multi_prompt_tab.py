"""
Multi-Prompt Tab - Tab 4 of the amplification dashboard.

Provides UI for batch generation across multiple prompts and configs.
Organize prompts in folders with simple or chat-mode editors.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List

import streamlit as st

if TYPE_CHECKING:
    from src.diffing.methods.amplification.amplification_dashboard import (
        AmplificationDashboard,
    )


class MultiPromptTab:
    """Renders Tab 4: Multi-prompt generation.

    Batch generation across multiple prompts and configs.
    Organize prompts in folders with simple or chat-mode editors.
    """

    def __init__(self, dashboard: "AmplificationDashboard"):
        self.dashboard = dashboard

    @st.fragment
    def render(self) -> None:
        """Render the multi-prompt generation tab."""
        from src.diffing.methods.amplification.streamlit_components.dashboard_state import (
            ManagedPrompt,
        )

        # Config selector and generate button above tabs
        active_prompts = [
            mp for mp in st.session_state.managed_prompts.values() if mp.active
        ]
        active_configs = [
            mc for mc in st.session_state.managed_configs.values() if mc.active
        ]

        # Config display selector - always visible, shows active configs
        all_config_ids = [mc.config_id for mc in active_configs]
        config_names = {mc.config_id: mc.full_name for mc in active_configs}

        # Initialize display selection if empty or invalid (only modify BEFORE widget renders)
        current_selection = st.session_state.multi_prompt_display_configs
        valid_selection = [cid for cid in current_selection if cid in all_config_ids]
        if valid_selection != current_selection:
            # Selection became invalid (configs removed), update state before widget
            st.session_state.multi_prompt_display_configs = valid_selection
        if not st.session_state.multi_prompt_display_configs and all_config_ids:
            st.session_state.multi_prompt_display_configs = all_config_ids[:3]

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            if active_configs:
                st.write("**Select configs to display (1-3):**")
                st.multiselect(
                    "Display configs",
                    options=all_config_ids,
                    format_func=lambda x: config_names.get(x, x),
                    max_selections=3,
                    label_visibility="collapsed",
                    key="multi_prompt_display_configs",
                )
            else:
                st.warning(
                    "No active configs. Enable configs in the Amplification tab."
                )
        with col2:
            st.write(
                f"**{len(active_prompts)} prompt(s), {len(active_configs)} config(s)**"
            )
        with col3:
            if st.button(
                f"🚀 Run ({len(active_prompts)})",
                use_container_width=True,
            ):
                st.session_state.multi_prompt_trigger_generation = True
                st.rerun(scope="fragment")

        # Show all samples checkbox - rendered once, outside conditional blocks
        st.checkbox("Show all samples", key="multi_prompt_show_all")

        prompts_tab, results_tab = st.tabs(["📝 Prompts", "📊 Results"])

        with prompts_tab:
            self._render_prompts_subtab()
        with results_tab:
            self._render_multi_prompt_results_subtab()

    @st.fragment
    def _render_prompts_subtab(self) -> None:
        """Render the prompts management subtab. Fragment for tab-level isolation."""
        self.dashboard._prompt_folder_manager.render_folder_loader()

        # Import from chat section
        self._render_import_chat_to_prompt_section()

        st.markdown("---")

        self.dashboard._prompt_folder_manager.render_all_folders(
            render_item=self._render_prompt_editor,
            render_item_actions=self._render_prompt_actions,
        )

    def _render_import_chat_to_prompt_section(self) -> None:
        """Render UI to import a chat conversation as a new prompt."""
        from src.diffing.methods.amplification.streamlit_components.dashboard_state import (
            ManagedPrompt,
        )

        conversations = st.session_state.get("conversations", {})

        if not conversations:
            return  # No conversations to import, don't show the section

        with st.expander("💬 Import from Chat", expanded=False):
            conv_options = {
                conv["name"]: conv_id for conv_id, conv in conversations.items()
            }

            # Get loaded folders for folder selection
            loaded_folders = sorted(
                st.session_state.get("loaded_prompt_folders", {None}),
                key=lambda x: (x is not None, x or ""),
            )

            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                selected_conv = st.selectbox(
                    f"Select conversation ({len(conversations)} available)",
                    options=list(conv_options.keys()),
                    key="import_chat_to_prompt_selection",
                )

            with col2:
                selected_folder = st.selectbox(
                    "Import to folder",
                    options=loaded_folders,
                    format_func=lambda x: "Root" if x is None else x,
                    key="import_chat_to_prompt_folder",
                )

            with col3:
                if st.button(
                    "Import",
                    key="import_chat_to_prompt_btn",
                    use_container_width=True,
                ):
                    conv_id = conv_options[selected_conv]
                    conversation = conversations[conv_id]

                    # Create new prompt with chat messages
                    new_prompt = ManagedPrompt(
                        active=True,
                        expanded=True,
                        folder=selected_folder,
                        editor_mode="chat",
                        messages=[
                            {k: v for k, v in msg.items() if k in ["role", "content"]}
                            for msg in conversation["history"]
                        ],
                        name=f"Imported: {conversation['name']}",
                    )
                    st.session_state.managed_prompts[new_prompt.prompt_id] = new_prompt
                    self.dashboard._save_prompts()
                    st.rerun(scope="fragment")

    def _render_prompt_actions(self, prompt_id: str, mp) -> None:
        """Render duplicate/delete buttons for a prompt."""
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋", key=f"dup_{prompt_id}", help="Duplicate"):
                new_prompt = mp.duplicate()
                st.session_state.managed_prompts[new_prompt.prompt_id] = new_prompt
                self.dashboard._save_prompts()
                st.rerun(scope="fragment")
        with col2:
            if st.button("🗑️", key=f"del_{prompt_id}", help="Delete"):
                deleted = (mp.folder, mp.get_display_name())
                del st.session_state.managed_prompts[prompt_id]
                self.dashboard._save_prompts(deleted=deleted)
                st.rerun(scope="fragment")

    @st.fragment
    def _render_prompt_editor(self, prompt_id: str, mp) -> None:
        """Render prompt editor. Fragment so active toggle updates expander title."""
        icon = "✅" if mp.active else "❌"
        display_name = mp.get_display_name() or "New Prompt"

        with st.expander(f"{icon} {display_name}", expanded=mp.expanded):
            name_key = f"prompt_name_{prompt_id}"

            def on_name_change(prompt=mp, pid=prompt_id, key=name_key):
                new_name = st.session_state[key]
                if new_name != prompt.name:
                    # Ensure unique name within folder
                    unique_name = self.dashboard._get_unique_prompt_name(
                        new_name, prompt.folder, exclude_prompt_id=pid
                    )
                    # Use rename() which tracks old disk name for cleanup
                    deleted = prompt.rename(unique_name)
                    self.dashboard._save_prompts(deleted=deleted)

            st.text_input(
                "Name (optional)",
                value=mp.name,
                key=name_key,
                placeholder="Auto-generated from prompt text",
                on_change=on_name_change,
            )

            # Active checkbox - in same fragment as expander so title updates
            active_key = f"prompt_active_{prompt_id}"

            # Sync session state with data model (handles bulk enable/disable)
            if (
                active_key in st.session_state
                and st.session_state[active_key] != mp.active
            ):
                st.session_state[active_key] = mp.active

            def on_active_change(prompt=mp, key=active_key):
                prompt.active = st.session_state[key]
                self.dashboard._save_prompts()

            st.checkbox(
                "Active",
                value=mp.active,
                key=active_key,
                help="Only active prompts will be used for generation",
                on_change=on_active_change,
            )

            # Editor mode toggle
            mode_key = f"prompt_mode_{prompt_id}"

            def on_mode_change(prompt=mp, key=mode_key):
                prompt.editor_mode = "chat" if st.session_state[key] else "simple"
                self.dashboard._save_prompts()

            st.toggle(
                "💬 Chat mode",
                value=mp.editor_mode == "chat",
                key=mode_key,
                on_change=on_mode_change,
            )

            if mp.editor_mode == "simple":
                self._render_prompt_simple_editor(prompt_id, mp)
            else:
                self._render_prompt_chat_editor(prompt_id, mp)

    def _render_prompt_simple_editor(self, prompt_id: str, mp) -> None:
        """Render simple text editor for prompt."""
        # Template mode
        template_key = f"prompt_template_{prompt_id}"

        def on_template_change(prompt=mp, key=template_key):
            prompt.template_mode = st.session_state[key]
            self.dashboard._save_prompts()

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
            self.dashboard._save_prompts()

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
                self.dashboard._save_prompts()

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
                self.dashboard._save_prompts()

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
                self.dashboard._save_prompts()

            st.text_input(
                "Filename",
                value=mp.loom_filename,
                key=filename_key,
                placeholder="untitled.txt",
                on_change=on_filename_change,
            )

    def _render_prompt_chat_editor(self, prompt_id: str, mp) -> None:
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
                        self.dashboard._save_prompts()

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
                        self.dashboard._save_prompts()
                        st.rerun(scope="fragment")

        # Add message buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ User", key=f"add_user_{prompt_id}"):
                mp.messages.append({"role": "user", "content": ""})
                self.dashboard._save_prompts()
                st.rerun(scope="fragment")
        with col2:
            if st.button("➕ Assistant", key=f"add_assistant_{prompt_id}"):
                mp.messages.append({"role": "assistant", "content": ""})
                self.dashboard._save_prompts()
                st.rerun(scope="fragment")
        with col3:
            if st.button("➕ System", key=f"add_system_{prompt_id}"):
                mp.messages.insert(0, {"role": "system", "content": ""})
                self.dashboard._save_prompts()
                st.rerun(scope="fragment")

    def _render_multi_prompt_results_subtab(self) -> None:
        """Render the results subtab (handles both generation and display)."""
        show_all = st.session_state.multi_prompt_show_all

        # Check if generation was triggered
        if st.session_state.get("multi_prompt_trigger_generation", False):
            st.session_state.multi_prompt_trigger_generation = False
            self._run_multi_prompt_generation_inline(show_all)
            return

        # Display existing results
        if st.session_state.multi_prompt_results is None:
            st.info("No results yet. Select configs above and click 'Run'.")
            return

        results_data = st.session_state.multi_prompt_results
        prompts = results_data["prompts"]
        tokenized_prompts = results_data.get("tokenized_prompts", [])
        config_results = results_data["config_results"]

        if not config_results:
            st.warning("No results available.")
            return

        # Filter selected configs to only those with results
        all_result_config_ids = set(config_results.keys())
        selected = [
            cid
            for cid in st.session_state.multi_prompt_display_configs
            if cid in all_result_config_ids
        ]
        if not selected:
            st.info("Select at least one config to display results.")
            return

        config_names = {
            cid: config_results[cid]["config"].full_name
            for cid in all_result_config_ids
        }

        # Display results
        num_configs = len(selected)

        for prompt_idx, mp in enumerate(prompts):
            display_name = mp.get_display_name() or f"Prompt {prompt_idx + 1}"

            with st.expander(f"📝 {display_name}", expanded=True):
                # Show detokenized prompt (collapsed by default)
                with st.expander("📋 Prompt", expanded=False):
                    if prompt_idx < len(tokenized_prompts):
                        st.code(
                            self.dashboard.tokenizer.decode(
                                tokenized_prompts[prompt_idx], skip_special_tokens=False
                            ),
                            language="text",
                            wrap_lines=True,
                        )
                    else:
                        st.warning("Tokenized prompt not available")

                if num_configs == 1:
                    # Single config: 2-column layout like multi-gen
                    config_id = selected[0]
                    config_name = config_names[config_id]
                    samples = config_results[config_id]["results"][prompt_idx]

                    from .samples import render_samples

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{config_name}**")
                    with col2:
                        render_samples(
                            samples=samples,
                            component_id=f"mp_cycler_{prompt_idx}_{config_id}",
                            height=300,
                            show_all=show_all,
                        )
                else:
                    # Multiple configs: vertical layout, horizontal config columns
                    from .samples import render_samples

                    cols = st.columns(num_configs)
                    for col_idx, config_id in enumerate(selected):
                        config_name = config_names[config_id]
                        samples = config_results[config_id]["results"][prompt_idx]

                        with cols[col_idx]:
                            st.write(f"**{config_name}**")
                            render_samples(
                                samples=samples,
                                component_id=f"mp_cycler_{prompt_idx}_{config_id}",
                                height=250,
                                show_all=show_all,
                            )

    def _run_multi_prompt_generation_inline(self, show_all: bool) -> None:
        """Run generation for all active prompts with all active configs (inline in Results tab)."""
        from src.diffing.methods.amplification.streamlit_components.dashboard_state import (
            GenerationLog,
        )
        from .samples import render_samples

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

        # Reorder configs: selected display configs first
        selected_ids = set(st.session_state.multi_prompt_display_configs)
        selected_configs = [mc for mc in active_configs if mc.config_id in selected_ids]
        other_configs = [
            mc for mc in active_configs if mc.config_id not in selected_ids
        ]
        ordered_configs = selected_configs + other_configs

        # Prepare tokenized prompts
        tokenized_prompts = []
        for mp in active_prompts:
            if mp.editor_mode == "simple":
                tokenized = self._tokenize_simple_prompt(mp)
            else:
                tokenized = self._tokenize_chat_prompt(mp)
            tokenized_prompts.append(tokenized)

        sampling_params = self.dashboard._get_sampling_params()

        # Create placeholders for progressive display (only for selected configs)
        st.markdown("## Generating...")
        num_display_configs = len(selected_configs)
        placeholders = {}  # {prompt_idx: {config_id: placeholder}}

        for prompt_idx, mp in enumerate(active_prompts):
            display_name = mp.get_display_name() or f"Prompt {prompt_idx + 1}"
            with st.expander(f"📝 {display_name}", expanded=True):
                # Prompt preview
                with st.expander("📋 Prompt", expanded=False):
                    st.code(
                        self.dashboard.tokenizer.decode(
                            tokenized_prompts[prompt_idx], skip_special_tokens=False
                        ),
                        language="text",
                        wrap_lines=True,
                    )

                if num_display_configs > 0:
                    cols = st.columns(num_display_configs)
                    placeholders[prompt_idx] = {}
                    for col_idx, mc in enumerate(selected_configs):
                        with cols[col_idx]:
                            st.write(f"**{mc.full_name}**")
                            placeholder = st.empty()
                            with placeholder.container():
                                st.info("Waiting for generation...")
                            placeholders[prompt_idx][mc.config_id] = placeholder

        # Stream results as they arrive
        results = {}
        for gen_result in self.dashboard.method.multi_gen_request(
            prompt=tokenized_prompts,
            amplification_configs=ordered_configs,
            sampling_params=sampling_params,
            compiled_adapters_dir=self.dashboard.persistence.compiled_adapters_dir,
            vllm_server=self.dashboard.vllm_server,
        ):
            config = gen_result["config"]
            results[config.config_id] = {
                "config": config,
                "results": gen_result["results"],  # 2D: [prompt_idx][sample_idx]
            }

            # Update placeholders for this config (if it's a display config)
            if config.config_id in placeholders.get(0, {}):
                for prompt_idx in range(len(active_prompts)):
                    placeholder = placeholders[prompt_idx][config.config_id]
                    samples = gen_result["results"][prompt_idx]
                    with placeholder.container():
                        render_samples(
                            samples=samples,
                            component_id=f"mp_prog_{prompt_idx}_{config.config_id}",
                            height=250,
                            show_all=show_all,
                        )

        # Store results with tokenized prompts for display
        st.session_state.multi_prompt_results = {
            "prompts": active_prompts,
            "tokenized_prompts": tokenized_prompts,
            "config_results": results,
        }

        # Log each prompt's generation
        for prompt_idx, mp in enumerate(active_prompts):
            prompt_tokens = tokenized_prompts[prompt_idx]
            prompt_text = self.dashboard.tokenizer.decode(
                prompt_tokens, skip_special_tokens=False
            )

            GenerationLog.from_dashboard_generation(
                generation_type="multigen",
                model_id=self.dashboard.method.base_model_cfg.model_id,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                configs=ordered_configs,
                results=[
                    {
                        "config_name": results[mc.config_id]["config"].full_name,
                        "outputs": results[mc.config_id]["results"][prompt_idx],
                    }
                    for mc in ordered_configs
                    if mc.config_id in results
                ],
                template_mode=mp.template_mode if mp.editor_mode == "simple" else None,
                logs_dir=self.dashboard.persistence.logs_dir,
            )

        st.rerun(scope="fragment")

    def _tokenize_simple_prompt(self, mp) -> list[int]:
        """Tokenize a simple-mode prompt."""
        if mp.template_mode == "No template":
            return self.dashboard.tokenizer.encode(
                mp.prompt_text, add_special_tokens=False
            )
        elif mp.template_mode == "Apply chat template":
            messages = []
            if mp.system_prompt:
                messages.append({"role": "system", "content": mp.system_prompt})
            messages.append({"role": "user", "content": mp.prompt_text})
            if mp.assistant_prefill:
                messages.append({"role": "assistant", "content": mp.assistant_prefill})
            return self.dashboard.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=not mp.assistant_prefill,
                continue_final_message=bool(mp.assistant_prefill),
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
            return self.dashboard.tokenizer.apply_chat_template(
                messages,
                continue_final_message=True,
                tokenize=True,
            )

    def _tokenize_chat_prompt(self, mp) -> list[int]:
        """Tokenize a chat-mode prompt."""
        if not mp.messages:
            return []
        continue_final_message = mp.messages[-1]["role"] == "assistant"
        return self.dashboard.tokenizer.apply_chat_template(
            mp.messages,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
            tokenize=True,
        )
