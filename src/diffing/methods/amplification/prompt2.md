# Implementation Plan: Weight Difference Amplification Dashboard

## Overview
Build a Streamlit dashboard for creating amplification configurations that modify LoRA adapter weights. Configs are compiled into new adapter files and used with vLLM for generation.

## Architecture

```
User creates config → Config.compile() creates modified adapter → save_request() uses vLLM LoraRequest
```

## File Structure

```
src/diffing/methods/amplification/
├── amplification_config.py   - Config dataclasses + compile logic
├── adapter_manager.py         - Adapter discovery (stub for now)
├── amplification_dashboard.py - Streamlit UI
└── weight_difference.py       - Updated visualize() method
```

## Implementation Subtasks

### Task 1: Core Configuration System
**File:** `amplification_config.py`

1.1. Implement `AmplificationConfig.compile()` method
- Load adapter weights from HuggingFace
- Apply amplification factors
- Save modified adapter to output_dir
- Return path to compiled adapter

1.2. Implement `_load_adapter_weights(adapter_id)` helper
- Use PEFT to load adapter weights
- Return Dict[str, torch.Tensor] of weights

1.3. Implement `_apply_amplification(weights, adapter_amp)` helper
- Iterate through layer_amplifications
- For each matching parameter, multiply by amplification weight
- Return modified weights dict

1.4. Implement `_resolve_layers(layers, num_layers)` helper
- Handle int → [int]
- Handle "all" → list(range(num_layers))
- Handle List[int] → pass through
- Return List[int]

1.5. Implement `_resolve_modules(modules, adapter_id)` helper
- Handle "all" → get all modules from LoRA config
- Handle List[str] → pass through
- Return List[str]

1.6. Implement `_should_amplify(param_name, layers, modules)` helper
- Parse layer index from param_name (e.g., "layers.5.attn.q_proj.lora_A")
- Check if layer in layers list
- Check if any module name appears in param_name
- Return bool

1.7. Implement `_save_adapter(weights, adapter_id, output_dir)` helper
- Save modified weights to output_dir
- Copy original adapter config files
- Ensure vLLM can load the saved adapter

### Task 2: Streamlit Dashboard - Sidebar & Global Controls
**File:** `amplification_dashboard.py`

2.1. Implement `_render_sidebar()`
- Display model info (read-only from cfg)
- Sampling parameters:
  - Temperature slider (0.1-2.0, default 1.0)
  - Top-p slider (0.0-1.0, default 0.9)
  - Max new tokens slider (10-500, default 100)
  - Do sample checkbox (default True)
- Save to st.session_state.sampling_params
- Global control buttons:
  - "Enable All" - set all configs.active = True
  - "Disable All" - set all configs.active = False

2.2. Implement `_get_sampling_params()`
- Read from st.session_state.sampling_params
- Return as dict for vLLM SamplingParams

### Task 3: Streamlit Dashboard - Tab 1 (Amplifications)
**File:** `amplification_dashboard.py`

3.1. Implement `_render_amplifications_tab()`
- Three buttons at top: "New Amplification", "Save Config", "Load Config"
- List all configs in st.session_state.amp_configs
- For each config, call `_render_amplification_config(idx, config)`

3.2. Implement "New Amplification" button logic
- Create empty AmplificationConfig with default name
- Add to st.session_state.amp_configs
- Trigger rerun

3.3. Implement "Save Config" button logic
- Show file picker or text input for filename
- Call config.save_yaml(path)
- Show success message

3.4. Implement "Load Config" button logic
- Show file picker or text input for filename
- Call AmplificationConfig.load_yaml(path)
- Add to st.session_state.amp_configs
- Trigger rerun

3.5. Implement `_render_amplification_config(idx, config)`
- Use st.expander with title: "✅/❌ {config.name}"
- Inside expander:
  - Name: text_input (updates config.name on change)
  - Description: text_area (updates config.description)
  - Active: checkbox (updates config.active)
  - Apply to: selectbox (base/finetuned/both)
  - Delete button (removes from amp_configs)
  - For each adapter, call `_render_adapter_amplification()`
  - "Add Adapter" button at bottom

3.6. Implement `_render_adapter_amplification(config_idx, adapter_idx, adapter)`
- Nested expander for adapter
- Adapter ID: text_input (manual entry, since AdapterManager is stub)
- Adapter Name: text_input
- Delete adapter button
- For each layer_amp, call `_render_layer_amplification()`
- "Add Layer Spec" button at bottom

3.7. Implement `_render_layer_amplification(config_idx, adapter_idx, layer_idx, layer_amp)`
- Layer selector with tabs: "Single" / "List" / "All"
  - Single: number_input for layer index
  - List: text_input for comma-separated indices
  - All: just shows "All layers"
- Update layer_amp.layers on change
- For each module_amp, call `_render_module_amplification()`
- "Add Module" button at bottom
- Delete layer spec button

3.8. Implement `_render_module_amplification(config_idx, adapter_idx, layer_idx, module_idx, module_amp)`
- Two columns: module selector | weight slider
- Module: text_input or checkbox for "all"
- Weight: slider (-5.0 to 5.0, step 0.1, default 1.0)
- Delete module button

### Task 4: Streamlit Dashboard - Tab 2 (Multi-Generation)
**File:** `amplification_dashboard.py`

4.1. Implement `_render_multi_generation_tab()`
- Prompt: text_area for user input
- "Generate" button
- On generate click:
  - Get active configs: [c for c in amp_configs if c.active]
  - For each active config:
    - Call `_compile_config(config)` to get adapter path
    - Call `send_request()` with LoraRequest
    - Display result in column
    - Add "Continue Chat" button per result

4.2. Implement `_compile_config(config)`
- Create output directory (e.g., ./compiled_adapters/{config.name}/)
- Call config.compile(output_dir)
- Return path

4.3. Implement generation display
- Use st.columns(len(active_configs))
- For each config result:
  - Show config.name as header
  - Show generated text in st.code
  - "Continue Chat" button that:
    - Stores config + prompt + response in st.session_state.chat_context
    - Shows success message to switch to Chat tab

### Task 5: Streamlit Dashboard - Tab 3 (Chat Interface)
**File:** `amplification_dashboard.py`

5.1. Implement `_render_chat_tab()`
- Check if st.session_state.chat_context exists
- If not, show: "No chat started. Generate from Multi-Generation tab first."
- If yes:
  - Display chat history (list of {role, content} dicts)
  - User input: text_area
  - "Send" button
  - On send:
    - Append user message to history
    - Compile config (same as Tab 2)
    - Call send_request() with full chat history as prompt
    - Append assistant response to history
    - Clear input
    - Rerun

5.2. Implement chat history display
- For each message in chat_history:
  - Show role (User/Assistant)
  - Show content in styled container
- Auto-scroll to bottom

### Task 6: Integration & Testing

6.1. Test config save/load round-trip
- Create config in UI
- Save to YAML
- Load from YAML
- Verify all fields match

6.2. Test config compilation (when implemented)
- Create simple amplification config
- Compile it
- Verify output directory structure
- Verify weights are modified correctly

6.3. Test multi-generation UI
- Create multiple configs
- Toggle active/inactive
- Verify only active configs generate

6.4. Test chat interface
- Start chat from multi-generation
- Send multiple messages
- Verify history persists

## Dependencies to Add

```python
# If not already in project
- peft
- vllm
- streamlit
- yaml
```

## Notes
- AdapterManager.get_available_adapters() is stub - manual adapter ID entry for now
- send_request() is stub - implement when vLLM server is ready
- Chat history formatting may need tokenizer.apply_chat_template()

