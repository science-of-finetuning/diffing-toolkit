"""
Generic folder management UI for Streamlit dashboards.

Provides reusable folder loader, create dialog, and folder section components
that can be configured for different item types (configs, prompts, etc.).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar, Generic

import streamlit as st

T = TypeVar("T")


def list_subfolders(base_dir: Path) -> list[str]:
    """
    List all available folder paths recursively under base_dir.

    Returns list of relative folder paths (empty string for root, then nested paths).
    """
    folders = [""]
    if not base_dir.exists():
        return folders

    for item in sorted(base_dir.rglob("*")):
        if item.is_dir() and item.name != "removed":
            rel_path = str(item.relative_to(base_dir))
            folders.append(rel_path)

    return folders


@dataclass
class FolderManagerConfig(Generic[T]):
    """Configuration for a folder manager instance."""

    base_dir: Path
    loaded_folders_key: str  # session_state key for loaded folders set
    items_key: str  # session_state key for managed items dict
    item_type_label: str  # "config" or "prompt" - used in UI labels
    widget_key_prefix: str  # prefix for widget keys to avoid collisions

    load_from_folder: Callable[[Path, str], dict[str, T]]
    save_to_folder: Callable[[dict[str, T], Path, str], None]
    unload_folder: Callable[[dict[str, T], str], dict[str, T]]
    create_new_item: Callable[[str], T]  # folder -> new item
    get_item_folder: Callable[[T], str]  # item -> folder

    save_loaded_folders: Callable[[], None]
    save_items: Callable[[], None]
    rerun_scope: str = "fragment"


class FolderManagerUI(Generic[T]):
    """
    Generic folder management UI component.

    Handles folder loading, creation dialogs, and folder sections with items.
    Configured via FolderManagerConfig to work with different item types.
    """

    def __init__(self, config: FolderManagerConfig[T]):
        self.cfg = config

    @property
    def _loaded_folders(self) -> set[str]:
        return st.session_state[self.cfg.loaded_folders_key]

    @property
    def _items(self) -> dict[str, T]:
        return st.session_state[self.cfg.items_key]

    def _key(self, name: str) -> str:
        """Generate a unique widget key."""
        return f"{self.cfg.widget_key_prefix}_{name}"

    def render_folder_loader(self) -> None:
        """Render the folder loader UI (dropdown + Load/Create buttons)."""
        all_folders = list_subfolders(self.cfg.base_dir)
        available_to_load = [f for f in all_folders if f not in self._loaded_folders]

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            folder_display = {f: "Root" if f == "" else f for f in available_to_load}
            if available_to_load:
                selected_folder = st.selectbox(
                    "Available Folders",
                    options=available_to_load,
                    format_func=lambda x: folder_display.get(x, x),
                    key=self._key("folder_to_load"),
                )
            else:
                st.info("All folders are loaded")
                selected_folder = None

        with col2:
            if st.button(
                "📂 Load",
                disabled=selected_folder is None,
                use_container_width=True,
                key=self._key("load_btn"),
            ):
                self._loaded_folders.add(selected_folder)
                loaded_items = self.cfg.load_from_folder(
                    self.cfg.base_dir, selected_folder
                )
                self._items.update(loaded_items)
                self.cfg.save_loaded_folders()
                st.rerun(scope=self.cfg.rerun_scope)

        with col3:
            if st.button(
                "➕ Create", use_container_width=True, key=self._key("create_btn")
            ):
                st.session_state[self._key("show_create_dialog")] = True

        if st.session_state.get(self._key("show_create_dialog"), False):
            self._render_create_folder_dialog()

    def _render_create_folder_dialog(self) -> None:
        """Render the create folder dialog."""
        with st.container(border=True):
            st.markdown(f"**Create New {self.cfg.item_type_label.title()} Folder**")
            new_folder_path = st.text_input(
                "Folder path",
                placeholder="e.g., experiments/v2",
                key=self._key("new_folder_path"),
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Create",
                    type="primary",
                    use_container_width=True,
                    key=self._key("create_confirm"),
                ):
                    if new_folder_path:
                        (self.cfg.base_dir / new_folder_path).mkdir(
                            parents=True, exist_ok=True
                        )
                        self._loaded_folders.add(new_folder_path)
                        st.session_state[self._key("show_create_dialog")] = False
                        self.cfg.save_loaded_folders()
                        st.rerun(scope=self.cfg.rerun_scope)
                    else:
                        st.error("Please enter a folder path")
            with col2:
                if st.button(
                    "Cancel", use_container_width=True, key=self._key("create_cancel")
                ):
                    st.session_state[self._key("show_create_dialog")] = False
                    st.rerun(scope=self.cfg.rerun_scope)

    def render_folder_section(
        self,
        folder: str,
        render_item: Callable[[str, T], None],
        render_item_actions: Callable[[str, T], None] | None = None,
    ) -> None:
        """
        Render a folder section with its items.

        Args:
            folder: Folder path to render
            render_item: Callback to render a single item (item_id, item) -> None
            render_item_actions: Optional callback for item action buttons (item_id, item) -> None
        """
        folder_display = "Root" if folder == "" else folder
        folder_items = {
            item_id: item
            for item_id, item in self._items.items()
            if self.cfg.get_item_folder(item) == folder
        }
        item_count = len(folder_items)
        item_label = self.cfg.item_type_label + ("s" if item_count != 1 else "")

        with st.expander(
            f"📁 {folder_display} ({item_count} {item_label})", expanded=True
        ):
            col_new, col_enable, col_disable, col_fold, col_unload = st.columns(
                [2, 1, 1, 1, 1]
            )

            with col_new:
                if st.button(
                    f"➕ New {self.cfg.item_type_label.title()}",
                    key=self._key(f"new_item_{folder}"),
                    use_container_width=True,
                ):
                    new_item = self.cfg.create_new_item(folder)
                    item_id = self._get_item_id(new_item)
                    st.session_state[self.cfg.items_key][item_id] = new_item
                    self.cfg.save_items()
                    st.rerun(scope=self.cfg.rerun_scope)

            with col_enable:
                if st.button(
                    "✅ Enable All",
                    key=self._key(f"enable_all_{folder}"),
                    use_container_width=True,
                    disabled=item_count == 0,
                    help=f"Enable all {self.cfg.item_type_label}s in this folder",
                ):
                    for item in folder_items.values():
                        item.active = True
                    self.cfg.save_items()
                    st.rerun(scope=self.cfg.rerun_scope)

            with col_disable:
                if st.button(
                    "❌ Disable All",
                    key=self._key(f"disable_all_{folder}"),
                    use_container_width=True,
                    disabled=item_count == 0,
                    help=f"Disable all {self.cfg.item_type_label}s in this folder",
                ):
                    for item in folder_items.values():
                        item.active = False
                    self.cfg.save_items()
                    st.rerun(scope=self.cfg.rerun_scope)

            with col_fold:
                if st.button(
                    "📂 Fold All",
                    key=self._key(f"fold_all_{folder}"),
                    use_container_width=True,
                    disabled=item_count == 0,
                    help=f"Fold all {self.cfg.item_type_label}s in this folder",
                ):
                    for item in folder_items.values():
                        item.expanded = False
                    self.cfg.save_items()
                    st.rerun(scope=self.cfg.rerun_scope)

            with col_unload:
                if st.button(
                    "📤 Unload",
                    key=self._key(f"unload_{folder}"),
                    use_container_width=True,
                    help=f"Unload this folder ({self.cfg.item_type_label}s are saved, not deleted)",
                ):
                    self.cfg.save_to_folder(self._items, self.cfg.base_dir, folder)
                    st.session_state[self.cfg.items_key] = self.cfg.unload_folder(
                        self._items, folder
                    )
                    self._loaded_folders.discard(folder)
                    self.cfg.save_loaded_folders()
                    st.rerun(scope=self.cfg.rerun_scope)

            if item_count == 0:
                st.info(
                    f"No {self.cfg.item_type_label}s in this folder. Click 'New {self.cfg.item_type_label.title()}' to create one."
                )
            else:
                for item_id, item in list(folder_items.items()):
                    if render_item_actions is not None:
                        col1, col2 = st.columns([30, 2])
                        with col1:
                            render_item(item_id, item)
                        with col2:
                            render_item_actions(item_id, item)
                    else:
                        render_item(item_id, item)

    def _get_item_id(self, item: T) -> str:
        """Get the ID from an item. Assumes item has config_id or prompt_id attribute."""
        if hasattr(item, "config_id"):
            return item.config_id
        if hasattr(item, "prompt_id"):
            return item.prompt_id
        raise AttributeError(f"Item {item} has no config_id or prompt_id attribute")

    def render_all_folders(
        self,
        render_item: Callable[[str, T], None],
        render_item_actions: Callable[[str, T], None] | None = None,
    ) -> None:
        """
        Render all loaded folders with their items.

        Args:
            render_item: Callback to render a single item
            render_item_actions: Optional callback for item action buttons
        """
        if len(self._loaded_folders) == 0:
            st.info(
                f"No folders loaded. Select a folder above to load {self.cfg.item_type_label}s."
            )
        else:
            for folder in sorted(self._loaded_folders):
                self.render_folder_section(folder, render_item, render_item_actions)
