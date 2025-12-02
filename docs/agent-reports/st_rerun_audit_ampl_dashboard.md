# Streamlit `st.rerun()` Audit Report

**File:** `src/diffing/methods/amplification/amplification_dashboard.py`
**Date:** 2025-12-01

## Background

Streamlit re-executes the entire script on every `st.rerun()` by default. Tabs are not isolated - they all re-render on every rerun. Fragments (`@st.fragment`) are the mechanism to create isolated sections that can rerun independently.

### Key Insight: Fragment Boundaries ≠ Visual Boundaries

**Fragment boundaries are determined by Python function scope, not visual containers.**

An `st.expander()` created inside a fragment belongs to that fragment's scope. Buttons inside the expander (but before calling a nested fragment) also belong to the parent fragment. This enables the "Russian Doll" pattern:

```python
@st.fragment
def list_fragment():
    for item in items:
        with st.expander(item.name):      # Expander in list scope
            if st.button("Delete"):        # Button in list scope!
                del items[item.id]
                st.rerun(scope="fragment") # Reruns list_fragment only

            item_inputs_fragment(item.id)  # Nested fragment for inputs
```

### Russian Doll Caveat: Inner State Affecting Outer UI

**The Russian Doll pattern breaks when inner fragment state affects outer fragment UI.**

Example: An "Active" checkbox inside the item fragment that should update the expander title icon:

```python
# ❌ BROKEN: Active toggle won't update expander title
@st.fragment
def list_fragment():
    for item in items:
        icon = "✅" if item.active else "❌"
        with st.expander(f"{icon} {item.name}"):  # Title in list scope
            item_inputs_fragment(item.id)          # Active checkbox here

@st.fragment
def item_inputs_fragment(item_id):
    st.checkbox("Active", on_change=...)  # Changes item.active
    # But expander title won't update until list_fragment reruns!
```

**Solution:** Keep the expander in the same fragment as the state that affects its title:

```python
# ✅ CORRECT: Expander and Active checkbox in same fragment
@st.fragment
def list_fragment():
    for item in items:
        col1, col2 = st.columns([20, 1])
        with col1:
            item_editor_fragment(item.id)  # Expander + Active inside
        with col2:
            if st.button("Delete"):        # Delete at list level
                ...

@st.fragment
def item_editor_fragment(item_id):
    icon = "✅" if item.active else "❌"
    with st.expander(f"{icon} {item.name}"):  # Title updates on Active change
        st.checkbox("Active", ...)
```

### Tab Structure

The dashboard has 4 tabs: **Amplifications**, **Multi-Generation**, **Chat**, **Multi-Prompt**

### Current Fragment Boundaries

```
display()
├── _render_sidebar()
├── tab1: _render_amplifications_tab()
│   └── @fragment: _render_folder_section() (per folder)
│       ├── dup/delete buttons (list scope)
│       └── @fragment: _render_amplification_config() (per config, includes expander)
├── tab2: @fragment: _render_multi_generation_tab()
│   ├── text_tab: _render_text_input_tab()
│   ├── msg_tab: @fragment: _render_message_builder_tab()
│   │   ├── _render_import_conversations_section()
│   │   └── @fragment: _render_message_list_and_add()
│   └── @fragment: _render_result_card() (per result)
├── tab3: _render_chat_tab()
│   └── @fragment: _render_single_conversation() (per conversation)
│       └── @fragment: _render_chat_messages()
└── tab4: _render_multi_prompt_tab()
    └── @fragment: _render_prompts_subtab()
        ├── delete button (list scope)
        └── @fragment: _render_prompt_editor() (per prompt, includes expander)
```

---

## Audit Results

### Legend

| Verdict | Meaning |
|---------|---------|
| ✅ OK | Global rerun is correct/necessary |
| ✅ FIXED | Already uses fragment scope appropriately |
| 🔧 NEEDS TAB REFACTOR | Could use fragment scope if parent becomes a fragment |
| 🔧 NEEDS REFACTOR | Needs structural changes (make parent a fragment, move buttons) |
| ⚠️ SHOULD FIX | Can be changed to fragment scope without structural changes |

---

### Global Reruns (`st.rerun()` or `_save_and_rerun()`)

| Line | Location | Action | Verdict | Notes |
|------|----------|--------|---------|-------|
| 482 | `_render_chat_sample_selection` | Cancel selection | ✅ FIXED | Now uses `scope="fragment"` |
| 529 | `_render_chat_sample_selection` | Select sample | ✅ FIXED | Now uses `scope="fragment"` |
| 662 | `_render_sidebar` | Enable All configs | ✅ OK | Affects all tabs - configs are global |
| 669 | `_render_sidebar` | Disable All configs | ✅ OK | Affects all tabs - configs are global |
| 789 | `_render_import_conversations_section` | Import conversation | ✅ FIXED | Now uses `scope="fragment"` (parent `_render_message_builder_tab` is a fragment) |
| 993 | `_render_folder_loader` | Load folder | ✅ OK | Affects sidebar Quick Edit + Amplifications structure |
| 1018 | `_render_create_folder_dialog` | Create folder | ✅ OK | Adds new folder section |
| 1024 | `_render_create_folder_dialog` | Cancel dialog | ✅ OK | Not inside a fragment |
| ~1037 | `_render_folder_section` | New Amplification | ✅ FIXED | Now uses `scope="fragment"` |
| ~1053 | `_render_folder_section` | Unload folder | ✅ OK | Removes folder from loaded_folders (global state) |
| 1115 | `_render_generation_controls` | Clear Results | ✅ FIXED | Now uses `scope="fragment"` (parent `_render_multi_generation_tab` is a fragment) |
| 1328 | `_render_multi_generation_tab` | After generation | ✅ FIXED | Now uses `scope="fragment"` (tab-level isolation) |
| 1427 | `_render_result_card_content` | Continue generation | ✅ FIXED | Now uses `scope="fragment"` |
| 1468 | `_render_result_card_content` | Regenerate | ✅ FIXED | Now uses `scope="fragment"` |
| 1582 | `_render_result_card_content` | Continue Chat | ✅ OK | Creates conversation in Chat tab |
| 1615 | `_render_chat_tab` | Start New Chat | ✅ OK | Adds new conversation tab |
| 1697 | `_render_new_conversation_tab` | Create Conversation | ✅ OK | Adds new conversation tab |
| 1756 | `_render_chat_messages` | Regen from user msg | ✅ OK | Triggers generation in parent fragment |
| 1812 | `_render_chat_messages` | Continue assistant msg | ✅ OK | Triggers generation in parent fragment |
| 1821 | `_render_chat_messages` | Regen from assistant | ✅ OK | Triggers generation in parent fragment |
| 2107 | `_render_single_conversation` | Delete conversation | ✅ OK | Removes entire tab |
| 2180 | `_render_single_conversation` | Send to Multi-Gen | ✅ OK | Cross-tab state transfer |
| ~1066 | `_render_folder_section` | Duplicate config | ✅ FIXED | Moved to list level, uses `scope="fragment"` |
| ~1084 | `_render_folder_section` | Delete config | ✅ FIXED | Moved to list level, uses `scope="fragment"` |
| ~2817 | `_render_prompts_subtab` | Add Prompt | ✅ FIXED | Uses `scope="fragment"` |
| ~2845 | `_render_prompts_subtab` | Delete Prompt | ✅ FIXED | Delete at list level, uses `scope="fragment"` |

### Fragment-Scoped Reruns (Already Correct)

| Line | Location | Action | Notes |
|------|----------|--------|-------|
| 825, 829 | `_render_message_list_and_add` | Save/Cancel edit | Correctly scoped |
| 849, 853 | `_render_message_list_and_add` | Edit/Delete message | Correctly scoped |
| 886 | `_render_message_list_and_add` | Add message | Correctly scoped |
| 1724, 1728 | `_render_chat_messages` | Save/Cancel user edit | Correctly scoped |
| 1745 | `_render_chat_messages` | Edit user message | Correctly scoped |
| 1766 | `_render_chat_messages` | Delete user message | Correctly scoped |
| 1784, 1788 | `_render_chat_messages` | Save/Cancel assistant edit | Correctly scoped |
| 1803 | `_render_chat_messages` | Edit assistant message | Correctly scoped |
| 1831 | `_render_chat_messages` | Delete assistant message | Correctly scoped |
| 1891, 1931 | `_render_single_conversation` | Regeneration complete | Correctly scoped |
| 2000, 2037 | `_render_single_conversation` | Continue complete | Correctly scoped |
| 2245, 2284 | `_render_single_conversation` | Chat generation complete | Correctly scoped |
| 2444 | `_render_config_fields` | Add Adapter | Correctly scoped |
| 2477 | `_render_adapter_amplification` | Delete Adapter | Correctly scoped |
| 2580 | `_render_adapter_amplification` | Add Layer Spec | Correctly scoped |
| 2643 | `_render_layer_amplification` | Delete Layer | Correctly scoped |
| 2745 | `_render_layer_amplification` | Add Module | Correctly scoped |
| 2808 | `_render_module_amplification` | Delete Module | Correctly scoped |
| 3002, 3011, 3017, 3023 | `_render_prompt_chat_editor` | Message operations | Correctly scoped |

---

## Detailed Analysis

### Items Fixed ✅

#### 1. Chat Sample Selection (Lines 482, 529) ✅

**Was:** `st.rerun()` (global)
**Now:** `st.rerun(scope="fragment")`

These are called from `_render_chat_sample_selection` which is invoked inside `_render_single_conversation` (a fragment). The sample selection only affects the current conversation.

#### 2. Result Card Continue/Regenerate (Lines 1427, 1468) ✅

**Was:** `st.rerun(scope="app")` inside a fragment
**Now:** `st.rerun(scope="fragment")`

These are inside `_render_result_card` (a fragment). Continuing or regenerating only updates that specific result card's data.

#### 3. Config Duplicate/Delete (Lines ~1066, ~1084) ✅

**Was:** `_save_and_rerun()` (global) inside `_render_config_fields`
**Now:** `_save_and_rerun(scope="fragment")` at list level in `_render_folder_section`

Applied same pattern as prompts:
- `_render_folder_section` is now a fragment
- Duplicate/delete buttons moved to list level (outside config expander)
- Active toggle still updates expander title correctly

---

### Items Fixed in Phase 3 ✅

#### 1. Import Conversations (Line 789) ✅

**Was:** `st.rerun()` (global)
**Now:** `st.rerun(scope="fragment")`

Made `_render_message_builder_tab` a fragment, so import reruns only the message builder area.

#### 2. Clear Results (Line 1115) ✅

**Was:** `_save_and_rerun()` (global)
**Now:** `_save_and_rerun(scope="fragment")`

Made `_render_multi_generation_tab` a fragment, so Clear Results only reruns the Multi-Gen tab.

#### 3. After Generation (Line 1328) ✅

**Was:** `st.rerun()` (global)
**Now:** `st.rerun(scope="fragment")`

Since `_render_multi_generation_tab` is now a fragment, the post-generation rerun only affects the Multi-Gen tab.

---

### Items That Are Correctly Global

These require global reruns due to cross-tab state or structural changes:

1. **Sidebar Enable/Disable All** - Configs used in Amplifications, Multi-Gen, Chat tabs
2. **Load/Unload Folder** - Changes folder structure in Amplifications + sidebar Quick Edit
3. **Create Folder** - Adds new folder section
4. **Unload Folder** - Removes folder from global state
5. **Continue Chat** (line 1582) - Creates conversation in different tab
6. **Start/Create New Chat** - Adds new conversation tab
7. **Delete Conversation** - Removes entire tab
8. **Send to Multi-Gen** - Cross-tab state transfer
9. **Chat message regen/continue buttons** - Trigger generation handled by parent fragment

---

## Recommended Refactoring Roadmap

### Phase 1: Quick Wins (No Structural Changes) ✅ COMPLETED

1. ~~**Lines 482, 529:** Change to `scope="fragment"`~~ ✅ Done
2. ~~**Lines 1427, 1468:** Change to `scope="fragment"`~~ ✅ Done

### Phase 2: Amplifications Tab - Same Pattern as Prompts ✅ COMPLETED

Applied the same pattern as Multi-Prompt tab:

1. ~~Make `_render_folder_section` a fragment (list level)~~ ✅ Done
2. ~~Move duplicate/delete buttons OUTSIDE the expander (columns layout)~~ ✅ Done
3. ~~Keep expander inside `_render_amplification_config` so Active toggle updates title~~ ✅ Done

Result:
```
@fragment _render_folder_section()
├── for config loop
│   ├── col1: @fragment _render_amplification_config() with expander
│   ├── col2: duplicate button (list scope)
│   └── col3: delete button (list scope)
```

### Phase 3: Multi-Gen Tab Structure ✅ COMPLETED

Made the Multi-Gen tab fully isolated:

1. ~~Make `_render_message_builder_tab` a fragment~~ ✅ Done
2. ~~Make `_render_multi_generation_tab` a fragment (entire tab)~~ ✅ Done
3. ~~Change "Import Conversations" to `scope="fragment"`~~ ✅ Done
4. ~~Change "Clear Results" to `scope="fragment"`~~ ✅ Done
5. ~~Change post-generation rerun to `scope="fragment"`~~ ✅ Done

Result:
```
@fragment _render_multi_generation_tab()
├── text_tab: _render_text_input_tab() + generation controls
├── msg_tab: @fragment _render_message_builder_tab()
│   ├── _render_import_conversations_section() (scope="fragment")
│   └── @fragment _render_message_list_and_add()
├── generation logic
└── results display (Clear Results uses scope="fragment")
```

### When to Use Russian Doll Pattern

Use Russian Doll (expander in parent, inputs in child fragment) **only when**:
- No inner fragment state affects the expander title
- No inner fragment state affects the expander's expanded/collapsed state
- All "title-affecting" state can be passed as static arguments

Example where it works: A list of items where the title is just the item name (edited via on_change callback), and there's no "active" toggle that changes the icon.

---

## Cross-Tab State Dependencies

Understanding why some global reruns are necessary:

```
managed_configs ──► Amplifications (edit configs)
                ──► Multi-Gen (shows "active configs" list)
                ──► Chat (config selector per conversation)
                ──► Sidebar Quick Edit (config dropdown)

conversations   ──► Chat (render conversations)
                ──► Multi-Gen (import from chat)

managed_prompts ──► Multi-Prompt only (fully isolated!)
```

`managed_prompts` is the only fully isolated state, which is why the Multi-Prompt tab can be completely fragmented.
