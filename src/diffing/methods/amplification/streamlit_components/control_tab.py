"""
Control tab for HuggingFace sync functionality.

Provides UI for pushing/loading .streamlit_cache to/from HuggingFace Hub.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import streamlit as st
from huggingface_hub import HfApi, snapshot_download, login, whoami
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

if TYPE_CHECKING:
    from .dashboard_state import DashboardPersistence


@st.cache_data(ttl=60, show_spinner=False)
def _get_hf_username() -> str | None:
    """Get the current HuggingFace username if logged in.

    Cached for 60 seconds to avoid repeated network calls on reruns.
    """
    try:
        info = whoami()
        return info.get("name")
    except Exception:
        return None


def _is_logged_in() -> bool:
    """Check if the user is logged in to HuggingFace."""
    return _get_hf_username() is not None


@st.fragment
def _render_auth_section() -> None:
    """Render HF authentication section as a fragment."""
    username = _get_hf_username()

    if username:
        st.success(f"Logged in as **{username}**")
        return

    st.warning("Not logged in to HuggingFace")

    with st.container(border=True):
        st.markdown("**Login to HuggingFace**")
        token = st.text_input(
            "HuggingFace Token",
            type="password",
            key="hf_token_input",
            help="Get your token from https://huggingface.co/settings/tokens",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", type="primary", use_container_width=True):
                if token:
                    try:
                        login(token=token)
                        _get_hf_username.clear()  # Clear cache to fetch new username
                        st.success("Login successful!")
                        st.rerun(scope="app")  # App scope to show push/load sections
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                else:
                    st.error("Please enter a token")
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun(scope="fragment")


def _parse_repo_name(repo_input: str, username: str) -> str:
    """
    Parse repo input into full repo_id.

    If input contains '/', use as-is (org/repo format).
    Otherwise, prefix with username (personal repo).
    """
    if "/" in repo_input:
        return repo_input
    return f"{username}/{repo_input}"


@st.fragment
def _render_push_section(username: str, persistence: "DashboardPersistence") -> None:
    """Render the Push to HuggingFace section as a fragment."""
    st.markdown("### Push to HuggingFace")

    with st.container(border=True):
        repo_name = st.text_input(
            "Repository name",
            key="push_repo_name",
            help="Your username will be prepended unless you specify org/repo format",
        )

        st.markdown("**What to include:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            include_conversations = st.checkbox(
                "Conversations",
                key="push_include_conversations",
            )
        with col2:
            include_logs = st.checkbox(
                "Generation logs",
                key="push_include_logs",
            )
        with col3:
            full_cache = st.checkbox(
                "Full .streamlit_cache",
                key="push_full_cache",
                help="Include entire .streamlit_cache, not just amplification_cache",
            )

        st.markdown("**Description (optional):**")
        user_description = st.text_area(
            "Add a custom description for this dataset",
            key="push_user_description",
            height=80,
            placeholder="e.g., Amplification configs for experiment X, testing different layer combinations...",
            help="Your description will be inserted into the README. Leave empty for default.",
        )

        if st.button("Push to HuggingFace", type="primary", use_container_width=True):
            if not repo_name:
                st.error("Please enter a repository name")
                return

            repo_id = _parse_repo_name(repo_name, username)

            with st.spinner(f"Pushing to {repo_id}..."):
                try:
                    _push_to_hf(
                        repo_id=repo_id,
                        include_conversations=include_conversations,
                        include_logs=include_logs,
                        full_cache=full_cache,
                        user_description=user_description,
                        cache_dir=persistence.cache_dir,
                    )
                    st.success(
                        f"Successfully pushed to [{repo_id}](https://huggingface.co/datasets/{repo_id})"
                    )
                except Exception as e:
                    st.error(f"Push failed: {e}")
                    logger.exception("Push to HF failed")


def _push_to_hf(
    repo_id: str,
    include_conversations: bool,
    include_logs: bool,
    full_cache: bool,
    user_description: str,
    cache_dir: Path,
) -> None:
    """Push cache to HuggingFace Hub."""
    import tempfile

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except HfHubHTTPError as e:
        if "403" in str(e):
            raise ValueError(
                f"Permission denied for repo {repo_id}. Check your token permissions."
            ) from e
        raise

    # Determine what to upload
    # cache_dir is amplification_cache, parent is .streamlit_cache
    if full_cache:
        source_dir = cache_dir.parent
        ignore_patterns = []
    else:
        source_dir = cache_dir
        ignore_patterns = []

    if not include_conversations:
        ignore_patterns.append("conversations/*")
    if not include_logs:
        ignore_patterns.append("generation_logs/*")

    # Generate README with user description and actual included items
    readme_content = _generate_readme(
        user_description=user_description,
        include_conversations=include_conversations,
        include_logs=include_logs,
        full_cache=full_cache,
    )

    # Create README.md in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        readme_path = Path(tmpdir) / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")

        # Upload README first
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_type="dataset",
            commit_message="Update README",
        )

    # Upload cache folder
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(source_dir),
        repo_type="dataset",
        commit_message="Update amplification cache",
        ignore_patterns=ignore_patterns if ignore_patterns else None,
    )

    logger.info(f"Pushed cache to {repo_id}")


@st.fragment
def _render_load_section(
    username: str,
    persistence: "DashboardPersistence",
    on_reload: Callable[[], None] | None = None,
) -> None:
    """Render the Load from HuggingFace section as a fragment."""
    st.markdown("### Load from HuggingFace")

    with st.container(border=True):
        repo_name = st.text_input(
            "Repository name",
            key="load_repo_name",
            help="Your username will be prepended unless you specify org/repo format",
        )

        load_mode = st.radio(
            "Load mode",
            options=["override", "import", "logs_only"],
            format_func=lambda x: {
                "override": "Override all (replace local with remote)",
                "import": "Import prompts/configs (rename conflicts)",
                "logs_only": "Only generation logs",
            }[x],
            key="load_mode",
        )

        if st.button("Load from HuggingFace", type="primary", use_container_width=True):
            if not repo_name:
                st.error("Please enter a repository name")
                return

            repo_id = _parse_repo_name(repo_name, username)

            with st.spinner(f"Loading from {repo_id}..."):
                try:
                    _load_from_hf(
                        repo_id=repo_id, mode=load_mode, persistence=persistence
                    )
                    if on_reload:
                        on_reload()  # Reload data before showing success
                    st.success(f"Successfully loaded from {repo_id}")
                    st.rerun(scope="app")  # App scope: loaded data affects other tabs
                except Exception as e:
                    st.error(f"Load failed: {str(e)}")
                    logger.exception("Load from HF failed")


def _load_from_hf(repo_id: str, mode: str, persistence: "DashboardPersistence") -> None:
    """Load cache from HuggingFace Hub."""
    import tempfile

    # Download to temp directory first
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=tmpdir,
        )
        local_path = Path(local_path)

        if mode == "override":
            _override_cache(local_path, persistence.cache_dir)
        elif mode == "import":
            _import_configs_prompts(
                local_path, persistence.configs_dir, persistence.prompts_dir
            )
        elif mode == "logs_only":
            _import_logs_only(local_path, persistence.logs_dir)


def _override_cache(source_path: Path, cache_dir: Path) -> None:
    """Override local cache with downloaded content."""
    # cache_dir is amplification_cache, parent is .streamlit_cache
    # Detect if this is full .streamlit_cache or just amplification_cache
    if (source_path / "amplification_cache").exists():
        # Full .streamlit_cache structure
        target = cache_dir.parent
        source = source_path
    else:
        # Just amplification_cache contents
        target = cache_dir
        source = source_path

    # Backup existing and replace
    if target.exists():
        # Remove existing
        for item in target.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # Copy new content
    for item in source.iterdir():
        if item.name.startswith("."):  # Skip hidden files like .gitattributes
            continue
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    logger.info(f"Overrode cache from {source_path}")


def _import_configs_prompts(
    source_path: Path, configs_dir: Path, prompts_dir: Path
) -> None:
    """Import configs and prompts, renaming on conflict."""
    # Detect structure
    if (source_path / "amplification_cache").exists():
        source_configs = source_path / "amplification_cache" / "configs"
        source_prompts = source_path / "amplification_cache" / "prompts"
    else:
        source_configs = source_path / "configs"
        source_prompts = source_path / "prompts"

    # Import configs
    if source_configs.exists():
        _import_folder_with_rename(source_configs, configs_dir, "config")

    # Import prompts
    if source_prompts.exists():
        _import_folder_with_rename(source_prompts, prompts_dir, "prompt")

    logger.info("Imported configs and prompts with conflict resolution")


def _import_folder_with_rename(source: Path, target: Path, item_type: str) -> None:
    """Import folder contents, renaming files on conflict.

    Skips system files like _ui_state.yaml and other underscore-prefixed files.
    """
    target.mkdir(parents=True, exist_ok=True)

    for item in source.rglob("*"):
        if item.is_file() and item.suffix in (".yaml", ".yml"):
            # Skip system files (files starting with underscore)
            if item.name.startswith("_"):
                logger.debug(f"Skipping system file: {item.name}")
                continue

            rel_path = item.relative_to(source)
            dest = target / rel_path

            # Create parent dirs
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists():
                # Rename with _from_hf suffix
                dest = _get_unique_path(dest, "_from_hf")
                logger.info(
                    f"Renamed conflicting {item_type}: {rel_path} -> {dest.name}"
                )

            shutil.copy2(item, dest)


def _get_unique_path(path: Path, suffix: str) -> Path:
    """Get a unique path by adding suffix and optionally a number."""
    stem = path.stem
    ext = path.suffix
    parent = path.parent

    # Try with just suffix first
    new_name = f"{stem}{suffix}{ext}"
    new_path = parent / new_name
    if not new_path.exists():
        return new_path

    # Add incrementing number
    i = 1
    while True:
        new_name = f"{stem}{suffix}_{i}{ext}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        i += 1


def _import_logs_only(source_path: Path, logs_dir: Path) -> None:
    """Import only generation logs."""
    # Detect structure
    if (source_path / "amplification_cache").exists():
        source_logs = source_path / "amplification_cache" / "generation_logs"
    else:
        source_logs = source_path / "generation_logs"

    if not source_logs.exists():
        logger.warning("No generation_logs found in downloaded data")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)

    # Copy logs, preserving structure
    for item in source_logs.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(source_logs)
            dest = logs_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists():
                dest = _get_unique_path(dest, "_from_hf")

            shutil.copy2(item, dest)

    logger.info("Imported generation logs")


def _init_control_tab_state() -> None:
    """Initialize session state for control tab widgets."""
    defaults = {
        "push_repo_name": "amplification-cache",
        "push_include_conversations": True,
        "push_include_logs": False,
        "push_full_cache": False,
        "push_user_description": "",
        "load_repo_name": "amplification-cache",
        "load_mode": "override",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _generate_readme(
    user_description: str,
    include_conversations: bool,
    include_logs: bool,
    full_cache: bool,
) -> str:
    """Generate README with user description and actual included items.

    Args:
        user_description: User-provided description to insert
        include_conversations: Whether conversations are included
        include_logs: Whether generation logs are included
        full_cache: Whether full .streamlit_cache is included
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build contents list based on what's actually included
    contents = [
        "- Amplification configurations (YAML files)",
        "- Prompts and prompt templates",
    ]
    if include_conversations:
        contents.append("- Conversations")
    if include_logs:
        contents.append("- Generation logs")
    if full_cache:
        contents.append("- Full .streamlit_cache (all dashboard data)")

    contents_section = "\n".join(contents)

    # Add user description if provided
    description_section = ""
    if user_description.strip():
        description_section = f"\n{user_description.strip()}\n"

    return f"""# Amplification Dashboard Cache

This dataset contains cached data from the Amplification Dashboard.
{description_section}
**Last updated:** {timestamp}

## Contents

This cache includes:
{contents_section}

## Usage

Load this cache in the Amplification Dashboard using the Control tab's "Load from HuggingFace" feature.

---
*Generated with [Diffing Toolkit](https://github.com/science-of-finetuning/diffing-toolkit)*
"""


def render_control_tab(
    persistence: "DashboardPersistence",
    on_reload: Callable[[], None] | None = None,
) -> None:
    """
    Render the Control tab with HuggingFace sync functionality.

    Args:
        persistence: Dashboard persistence manager for paths.
        on_reload: Optional callback to trigger data reload after loading from HF.
    """
    _init_control_tab_state()

    st.markdown("## HuggingFace Sync")
    st.markdown("Push or load your amplification cache to/from HuggingFace Hub.")

    # Auth section (always rendered)
    st.markdown("---")
    _render_auth_section()

    # Check auth status to conditionally show push/load sections
    username = _get_hf_username()
    if not username:
        return  # Not authenticated, don't show push/load sections

    # Push section
    st.markdown("---")
    _render_push_section(username, persistence)

    # Load section
    st.markdown("---")
    _render_load_section(username, persistence, on_reload)
