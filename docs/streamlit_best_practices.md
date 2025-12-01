# Streamlit Performance Best Practices

A guide to building responsive Streamlit applications, focusing on common performance pitfalls and their solutions.

## Table of Contents
1. [Common Mistakes](#common-mistakes)
   - [Double Reruns](#1-double-reruns)
   - [Missing Fragments](#2-missing-fragments)
   - [Compare-and-Save Anti-Pattern](#3-compare-and-save-anti-pattern)
   - [Wrong Rerun Scope](#4-wrong-rerun-scope)
   - [Expensive Operations on Every Rerun](#5-expensive-operations-on-every-rerun)
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
