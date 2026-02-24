# Streamlit Performance Best Practices

A guide to building responsive Streamlit applications, focusing on common performance pitfalls and their solutions.

## Table of Contents
1. [Common Mistakes](#common-mistakes)
   - [Double Reruns](#1-double-reruns)
   - [Missing Fragments](#2-missing-fragments)
   - [Compare-and-Save Anti-Pattern](#3-compare-and-save-anti-pattern)
   - [Wrong Rerun Scope](#4-wrong-rerun-scope)
   - [Expensive Operations on Every Rerun](#5-expensive-operations-on-every-rerun)
   - [Fragments Losing Column Context](#6-fragments-losing-column-context)
   - [Mixing default with Session State Assignment](#7-mixing-default-with-session-state-assignment)
   - [Pre-initialized Session State with value= Parameter](#8-pre-initialized-session-state-with-value-parameter)
2. [Best Practices](#best-practices)
   - [Use Fragments for Independent Sections](#1-use-fragments-for-independent-sections)
   - [Use on_change Callbacks](#2-use-on_change-callbacks)
   - [Choose the Right Rerun Scope](#3-choose-the-right-rerun-scope)
   - [Cache Expensive Computations](#4-cache-expensive-computations)
   - [Organize State Properly](#5-organize-state-properly)

---

## Common Mistakes

### 1. Double Reruns

**Problem**: Calling `st.rerun()` after a widget change causes two full page reruns - one from Streamlit's automatic rerun on widget change, and another from your explicit call.

```python
# BAD: Double rerun
current_value = my_state.active
my_state.active = st.checkbox("Active", value=my_state.active)
if my_state.active != current_value:
    save_to_disk()
    st.rerun()  # This causes a SECOND rerun!
```

**What happens**:
1. User clicks checkbox
2. Streamlit automatically reruns the script (rerun #1)
3. Code detects change, calls `st.rerun()` (rerun #2)
4. Total: 2 full page reruns for 1 click

**Solution**: Use `on_change` callback - it runs before the automatic rerun:

```python
# GOOD: Single rerun
def on_active_change():
    my_state.active = st.session_state["active_checkbox"]
    save_to_disk()

st.checkbox(
    "Active",
    value=my_state.active,
    key="active_checkbox",
    on_change=on_active_change,
)
```

---

### 2. Missing Fragments

**Problem**: When you have multiple independent UI sections, changing one section reruns everything.

```python
# BAD: No fragments - editing Config 1 reruns Config 2, 3, etc.
def render_dashboard():
    for config_id, config in configs.items():
        render_config(config_id, config)  # All configs rerun on any change

def render_config(config_id, config):
    name = st.text_input("Name", value=config.name, key=f"name_{config_id}")
    if name != config.name:
        config.name = name
        save_config(config)
```

**Solution**: Wrap independent sections in `@st.fragment`:

```python
# GOOD: Each config updates independently
def render_dashboard():
    for config_id, config in configs.items():
        render_config(config_id, config)

@st.fragment
def render_config(config_id, config):
    def on_name_change():
        config.name = st.session_state[f"name_{config_id}"]
        save_config(config)

    st.text_input(
        "Name",
        value=config.name,
        key=f"name_{config_id}",
        on_change=on_name_change,
    )
```

---

### 3. Compare-and-Save Anti-Pattern

**Problem**: Comparing widget return value to previous state on every rerun is wasteful and error-prone.

```python
# BAD: Compare-and-save pattern
current_name = config.name
new_name = st.text_input("Name", value=config.name, key="name_input")
if new_name != current_name:  # This runs on EVERY rerun
    config.name = new_name
    save_to_disk()
    st.rerun()
```

**Issues**:
- Comparison logic runs on every rerun
- Can cause save storms with text inputs (saves on every keystroke)
- Often paired with unnecessary `st.rerun()` calls

**Solution**: Use `on_change` - it only fires when the value actually changes:

```python
# GOOD: on_change only fires when value changes
def on_name_change():
    config.name = st.session_state["name_input"]
    save_to_disk()

st.text_input(
    "Name",
    value=config.name,
    key="name_input",
    on_change=on_name_change,
)
```

---

### 4. Wrong Rerun Scope

**Problem**: Using app-level rerun when fragment-level would suffice.

```python
# BAD: App-level rerun inside a fragment
@st.fragment
def render_item(item_id, item):
    if st.button("Delete", key=f"delete_{item_id}"):
        items.pop(item_id)
        st.rerun()  # Default scope="app" - reruns ENTIRE page!
```

**Solution**: Use `scope="fragment"` for changes within a fragment:

```python
# GOOD: Fragment-level rerun
@st.fragment
def render_item(item_id, item):
    if st.button("Delete", key=f"delete_{item_id}"):
        items.pop(item_id)
        st.rerun(scope="fragment")  # Only reruns this fragment
```

**When to use each scope**:
- `scope="fragment"`: Changes only affect the current fragment's UI
- `scope="app"`: Changes affect UI outside the fragment (e.g., deleting an item that's listed elsewhere)

---

### 5. Expensive Operations on Every Rerun

**Problem**: Running expensive operations in the main script flow.

```python
# BAD: Loads data on every rerun
def render_dashboard():
    data = load_large_dataset()  # Runs on EVERY rerun!
    st.dataframe(data)
```

**Solution**: Use caching decorators:

```python
# GOOD: Cached data loading
@st.cache_data
def load_large_dataset():
    return pd.read_csv("large_file.csv")

def render_dashboard():
    data = load_large_dataset()  # Cached after first call
    st.dataframe(data)
```

---

### 6. Fragments Losing Column Context

**Problem**: When calling a `@st.fragment` function from within a column context, the fragment's output may not render in the correct column. Multiple fragments in a loop can overlap, with only the last few visible.

```python
# BAD: Fragment loses column context - only last 2 cards visible
output_cols = st.columns(2)
for idx, item in enumerate(items):
    col_idx = idx % 2
    with output_cols[col_idx]:
        render_item_card(idx, item)  # Fragment called inside column

@st.fragment
def render_item_card(idx, item):
    with st.expander(f"Item {idx}"):
        st.write(item)
        if st.button("Action", key=f"btn_{idx}"):
            do_something()
```

**What happens**: The fragment renders in isolation and loses the column context from its call site. When multiple fragments render, they overlap in the same position instead of distributing across columns.

**Solution**: Wrap the fragment call in `st.container()` to anchor it to the column:

```python
# GOOD: Container anchors fragment to column context
output_cols = st.columns(2)
for idx, item in enumerate(items):
    col_idx = idx % 2
    with output_cols[col_idx]:
        with st.container():  # Anchors fragment output to this column
            render_item_card(idx, item)

@st.fragment
def render_item_card(idx, item):
    with st.expander(f"Item {idx}"):
        st.write(item)
        if st.button("Action", key=f"btn_{idx}"):
            do_something()
```

The container acts as a bridge that preserves the layout position for the fragment's output.

---

### 7. Mixing default with Session State Assignment

**Problem**: Using a recalculated `default` parameter AND assigning the widget's return value to session state causes intermittent state conflicts. Changes may require multiple attempts to take effect.

```python
# BAD: Recalculated default + session state assignment
current_selection = st.session_state.selected_items
valid_selection = [x for x in current_selection if x in available_items]

selected = st.multiselect(
    "Select items",
    options=available_items,
    default=valid_selection[:3],  # Recalculated every rerun!
)
st.session_state.selected_items = selected  # Assignment creates conflict
```

**What happens**:
1. User changes selection → widget returns new value → stored in session state
2. On rerun, `valid_selection` is recalculated from session state
3. `default=valid_selection[:3]` is passed to the widget again
4. Widget's internal state conflicts with the recalculated `default`
5. Sometimes the widget "resets" to the default, ignoring the user's change

**Solution**: Use `key` parameter instead - it binds the widget directly to session state, eliminating the need for both `default` and manual assignment:

```python
# GOOD: Use key, validate/initialize state before widget
current_selection = st.session_state.selected_items
valid_selection = [x for x in current_selection if x in available_items]

# Update state BEFORE widget if validation changed it
if valid_selection != current_selection:
    st.session_state.selected_items = valid_selection
if not st.session_state.selected_items and available_items:
    st.session_state.selected_items = available_items[:3]

# Widget with key - Streamlit manages state automatically
st.multiselect(
    "Select items",
    options=available_items,
    key="selected_items",  # Binds directly to session state
)
# No default needed, no assignment needed - key handles both
```

**Key insight**: With `key`, Streamlit syncs the widget value with `st.session_state[key]` automatically. Initialize the session state value before the widget instead of using `default`, and skip the assignment entirely.

---

### 8. Pre-initialized Session State with value= Parameter

**Problem**: Pre-initializing session state for a widget key AND passing `value=` to the same widget causes Streamlit warnings and potential conflicts.

```python
# BAD: Pre-init session state + value= parameter
# In init:
if "my_slider" not in st.session_state:
    st.session_state.my_slider = loaded_value_from_disk

# In widget:
st.slider(
    "My Slider",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.get("my_slider", 0.5),  # WRONG!
    key="my_slider",
)
```

**Warning you'll see**: `StreamlitAPIWarning: The widget with key "my_slider" was created with a default value but also had its value set via the Session State API.`

**What happens**:
1. Session state is pre-initialized (e.g., loading from disk)
2. Widget is created with both `key="my_slider"` AND `value=X`
3. Streamlit sees two sources of truth for the same widget
4. Warning is raised, behavior may be unpredictable

**Solution**: Pre-initialize session state, then use ONLY `key=` (no `value=`):

```python
# GOOD: Pre-init session state, widget uses only key=
# In init:
if "my_slider" not in st.session_state:
    st.session_state.my_slider = loaded_value_from_disk

# In widget - NO value= parameter!
st.slider(
    "My Slider",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    key="my_slider",  # Widget reads from session state automatically
)
```

**Key insight**: When using `key=`, the widget automatically reads its initial value from `st.session_state[key]` if that key exists. You don't need `value=` at all - it's redundant and causes conflicts.

**The two valid patterns**:

| Pattern | When to use |
|---------|-------------|
| `value=X` only (no `key=`) | Widgets that don't need persistence or state syncing |
| `key="X"` only (no `value=`) | Widgets with pre-initialized session state (e.g., loaded from disk) |

**Never combine**: `value=X` + `key="X"` when session state for that key is pre-initialized.

---

## Best Practices

### 1. Use Fragments for Independent Sections

Fragments allow parts of your UI to rerun independently, dramatically improving responsiveness.

```python
# Structure your app with fragments for independent sections
def main():
    st.title("My App")

    col1, col2 = st.columns(2)

    with col1:
        render_settings_panel()  # Fragment

    with col2:
        render_data_panel()  # Fragment

@st.fragment
def render_settings_panel():
    """Settings changes only rerun this panel."""
    st.header("Settings")

    def on_threshold_change():
        st.session_state.threshold = st.session_state["threshold_slider"]

    st.slider(
        "Threshold",
        0.0, 1.0,
        value=st.session_state.get("threshold", 0.5),
        key="threshold_slider",
        on_change=on_threshold_change,
    )

@st.fragment
def render_data_panel():
    """Data interactions only rerun this panel."""
    st.header("Data")
    # ... data display and interactions
```

**Fragment guidelines**:
- Wrap logically independent UI sections
- Each fragment should be self-contained
- Fragments can be nested (inner fragments rerun independently)
- Use `scope="fragment"` for reruns within fragments

---

### 2. Use on_change Callbacks

Callbacks are the preferred way to handle widget value changes.

```python
# Pattern for different widget types

# Checkbox
def on_checkbox_change():
    st.session_state.my_flag = st.session_state["my_checkbox"]
    save_state()

st.checkbox("Enable feature", value=st.session_state.my_flag,
            key="my_checkbox", on_change=on_checkbox_change)

# Selectbox
def on_select_change():
    st.session_state.selected_option = st.session_state["my_select"]
    save_state()

st.selectbox("Choose option", options=["A", "B", "C"],
             key="my_select", on_change=on_select_change)

# Slider
def on_slider_change():
    st.session_state.value = st.session_state["my_slider"]
    save_state()

st.slider("Value", 0, 100, key="my_slider", on_change=on_slider_change)

# Text input
def on_text_change():
    st.session_state.text = st.session_state["my_text"]
    save_state()

st.text_input("Enter text", key="my_text", on_change=on_text_change)
```

**Callback with closure for dynamic contexts** (e.g., in loops):

```python
# When rendering in a loop, capture variables with default arguments
for idx, item in enumerate(items):
    def on_change(item=item, idx=idx):  # Capture with defaults
        item.value = st.session_state[f"input_{idx}"]
        save_item(item)

    st.text_input(
        f"Item {idx}",
        value=item.value,
        key=f"input_{idx}",
        on_change=on_change,
    )
```

---

### 3. Choose the Right Rerun Scope

```python
@st.fragment
def render_editable_list():
    items = st.session_state.get("items", [])

    for idx, item in enumerate(items):
        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(item)

        with col2:
            if st.button("Delete", key=f"del_{idx}"):
                items.pop(idx)
                # Fragment scope: list is self-contained
                st.rerun(scope="fragment")

    # Adding item - might need app scope if displayed elsewhere
    if st.button("Add Item"):
        items.append(f"New Item {len(items)}")
        st.rerun(scope="fragment")


def render_app():
    col1, col2 = st.columns(2)

    with col1:
        render_editable_list()

    with col2:
        # If this shows item count, you might need app-level rerun
        # when items change, OR make this a fragment too
        st.metric("Total Items", len(st.session_state.get("items", [])))
```

---

### 4. Cache Expensive Computations

```python
# Cache data that doesn't change often
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Cached until file_path changes or cache is cleared."""
    return pd.read_csv(file_path)

# Cache resources (models, connections, etc.)
@st.cache_resource
def load_model(model_name: str):
    """Cached for the lifetime of the app."""
    return SomeMLModel(model_name)

# Cache with TTL for data that might change
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_api_data(endpoint: str) -> dict:
    return requests.get(endpoint).json()

# Use in your app
def render_dashboard():
    data = load_data("data.csv")  # Cached
    model = load_model("my-model")  # Cached

    # Process and display...
```

---

### 5. Organize State Properly

```python
# Initialize state in one place
def init_session_state():
    """Call once at app start."""
    defaults = {
        "items": [],
        "selected_tab": 0,
        "settings": {"theme": "light", "notifications": True},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Use dataclasses or classes for complex state
from dataclasses import dataclass

@dataclass
class AppState:
    items: list
    selected_tab: int

    def save(self):
        # Persist to disk/database
        pass

    @classmethod
    def load(cls) -> "AppState":
        # Load from disk/database
        pass

# Group related callbacks
class SettingsController:
    @staticmethod
    def on_theme_change():
        st.session_state.settings["theme"] = st.session_state["theme_select"]
        save_settings()

    @staticmethod
    def on_notifications_change():
        st.session_state.settings["notifications"] = st.session_state["notif_toggle"]
        save_settings()
```

---

### 6. Fragment Architecture Patterns

When building complex UIs with nested lists, understanding fragment boundaries is crucial.

#### Key Insight: Fragment Boundaries ≠ Visual Boundaries

**Fragment boundaries are determined by Python function scope, not visual containers like expanders.**

```python
@st.fragment
def list_fragment():
    for item in items:
        with st.expander(item.name):      # Expander in list scope
            if st.button("Delete"):        # Button ALSO in list scope!
                del items[item.id]
                st.rerun(scope="fragment") # Reruns list_fragment

            item_inputs_fragment(item.id)  # Nested fragment

@st.fragment
def item_inputs_fragment(item_id):
    # Only this reruns when inputs change
    st.text_input("Name", ...)
```

#### Pattern: Delete at Parent Level

When deleting items from a list, the delete button should be in the **parent** fragment scope:

```python
# GOOD: Delete at list level
@st.fragment
def render_item_list():
    for item_id, item in list(items.items()):
        col1, col2 = st.columns([10, 1])
        with col1:
            render_item_editor(item_id, item)  # Item fragment
        with col2:
            if st.button("🗑️", key=f"del_{item_id}"):
                del items[item_id]
                st.rerun(scope="fragment")  # List reruns, item gone

@st.fragment
def render_item_editor(item_id, item):
    # Edit fields - changes only rerun this editor
    st.text_input("Name", value=item.name, ...)
```

#### Caveat: Inner State Affecting Outer UI

**The "Russian Doll" pattern breaks when inner fragment state affects outer fragment UI.**

```python
# ❌ BROKEN: Active toggle won't update expander title
@st.fragment
def list_fragment():
    for item in items:
        icon = "✅" if item.active else "❌"
        with st.expander(f"{icon} {item.name}"):  # Title in list scope
            item_inputs_fragment(item.id)          # Active checkbox here!

@st.fragment
def item_inputs_fragment(item_id):
    st.checkbox("Active", on_change=...)  # Changes item.active
    # But expander title won't update until list reruns!
```

**Solution:** Keep the expander in the same fragment as state that affects its title:

```python
# ✅ CORRECT: Expander and Active checkbox in same fragment
@st.fragment
def list_fragment():
    for item in items:
        col1, col2 = st.columns([20, 1])
        with col1:
            item_editor_fragment(item.id)  # Expander + Active inside
        with col2:
            if st.button("🗑️"):            # Delete at list level
                ...

@st.fragment
def item_editor_fragment(item_id):
    item = items[item_id]
    icon = "✅" if item.active else "❌"
    with st.expander(f"{icon} {item.name}"):  # Title updates on Active change
        st.checkbox("Active", ...)
        # Other fields...
```

#### When to Use Each Pattern

| Scenario | Pattern |
|----------|---------|
| Inner state affects expander title (active/icon) | Expander inside item fragment |
| Title is static, only content changes | Expander in list fragment (Russian Doll) |
| Delete/duplicate operations | Buttons at list fragment level |
| Pure editing (name, description, etc.) | Item fragment with `on_change` callbacks |

---

## Quick Reference

| Situation | Solution |
|-----------|----------|
| Widget change triggers expensive rerun | Use `@st.fragment` |
| Need to save on value change | Use `on_change` callback |
| Rerun inside fragment | Use `st.rerun(scope="fragment")` |
| Loading large data | Use `@st.cache_data` |
| Loading models/connections | Use `@st.cache_resource` |
| Multiple independent UI sections | Wrap each in `@st.fragment` |
| Compare-and-save pattern | Replace with `on_change` |
| Double reruns | Remove explicit `st.rerun()`, use callbacks |
| Widget ignores changes intermittently | Use `key` instead of `default` + assignment |
| "widget created with default value but also set via Session State API" | Remove `value=`, use only `key=` |

---

## Performance Debugging Tips

1. **Add timing logs** to identify slow sections:
   ```python
   import time
   start = time.time()
   # ... code section
   st.write(f"Section took {time.time() - start:.2f}s")
   ```

2. **Check rerun frequency** by adding a counter:
   ```python
   if "rerun_count" not in st.session_state:
       st.session_state.rerun_count = 0
   st.session_state.rerun_count += 1
   st.sidebar.write(f"Reruns: {st.session_state.rerun_count}")
   ```

3. **Profile with st.cache** to see cache hits:
   ```python
   @st.cache_data(show_spinner="Loading data...")
   def load_data():
       # If you see spinner often, cache isn't working as expected
       pass
   ```

---

## The Streamlit Performance Mantra

```
I must not rerun.
Global rerun is the performance-killer.
Global rerun is the little-death that brings total UI obliteration.

I will face my reruns.
I will permit them to pass through my fragments and only my fragments.
And when they have passed I will turn the inner eye to see their scope.

Where the global rerun has gone there will be nothing.
Only the fragment will remain.

I shall use on_change, not compare-and-save.
I shall scope my reruns, not broadcast them.
I shall fragment my UI, not monolith it.

For in the way of Streamlit, performance flows to those who understand:
The fragment boundary is the function scope.
The expander is but a visual lie.
The callback is the one true path.
```
