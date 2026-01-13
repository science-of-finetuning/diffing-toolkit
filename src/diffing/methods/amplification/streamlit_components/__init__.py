"""
Streamlit UI components for the amplification dashboard.

This package contains modular UI components:
- Tab implementations (amplifications, multi_generation, chat, multi_prompt, control)
- State management (dashboard_state.py: persistence, session state helpers)
- Folder management UI (folder_manager_ui.py)
"""

from .dashboard_state import DashboardPersistence
from .amplifications_tab import AmplificationsTab
from .chat_tab import ChatTab
from .multi_generation_tab import MultiGenerationTab
from .multi_prompt_tab import MultiPromptTab
from .folder_manager_ui import FolderManagerUI

__all__ = [
    "DashboardPersistence",
    "AmplificationsTab",
    "ChatTab",
    "MultiGenerationTab",
    "MultiPromptTab",
    "FolderManagerUI",
]
