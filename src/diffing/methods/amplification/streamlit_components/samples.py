import re
import html
from pathlib import Path
import streamlit.components.v1 as components
import streamlit as st
from annotated_text import annotated_text, annotation


# Load sample cycler component files
COMPONENTS_DIR = Path(__file__).parent.parent / "components"
_SAMPLE_CYCLER_JS = (COMPONENTS_DIR / "sample_cycler.js").read_text()
_SAMPLE_CYCLER_CSS = (COMPONENTS_DIR / "sample_cycler.css").read_text()
_SAMPLE_CYCLER_HTML = (COMPONENTS_DIR / "sample_cycler.html").read_text()


def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convert hex color to rgba string with transparency.

    Args:
        hex_color: Hex color string (e.g., "#ff0000" or "ff0000")
        alpha: Opacity value between 0 and 1

    Returns:
        RGBA color string (e.g., "rgba(255, 0, 0, 0.4)")
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def apply_html_highlighting(
    text: str, selectors: list[dict], alpha: float = 0.4
) -> str:
    """Apply regex highlighting using HTML mark tags with multiple selectors.

    Args:
        text: The text to highlight
        selectors: List of {keywords: list[str], color: str, enabled: bool} dicts
        alpha: Opacity for highlight colors

    Returns:
        Text with matches wrapped in styled <mark> tags
    """
    if not selectors:
        return text

    # Filter to only enabled selectors
    enabled_selectors = [s for s in selectors if s.get("enabled", True)]
    if not enabled_selectors:
        return text

    # Collect all matches with their colors: (start, end, matched_text, color)
    all_matches = []
    for selector in enabled_selectors:
        keywords = selector.get("keywords", [])
        color = selector.get("color", "#ffff00")
        rgba_color = hex_to_rgba(color, alpha)

        for pattern in keywords:
            if not pattern:
                continue
            try:
                for match in re.finditer(f"({pattern})", text, flags=re.IGNORECASE):
                    all_matches.append(
                        (match.start(), match.end(), match.group(), rgba_color)
                    )
            except re.error:
                pass  # Invalid regex, skip

    if not all_matches:
        return text

    # Sort by start position, then by length (longer matches first for same start)
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build result, skipping overlapping matches
    result = []
    last_end = 0
    for start, end, matched_text, color in all_matches:
        if start < last_end:
            continue  # Skip overlapping match
        if start > last_end:
            result.append(html.escape(text[last_end:start]))
        result.append(
            f'<mark style="background-color: {color}; color: inherit;">{html.escape(matched_text)}</mark>'
        )
        last_end = end

    if last_end < len(text):
        result.append(html.escape(text[last_end:]))

    return "".join(result)


def get_annotated_segments(
    text: str, selectors: list[dict], alpha: float = 0.4
) -> list:
    """Split text into segments for annotated_text() display with multiple selectors.

    Args:
        text: The text to process
        selectors: List of {keywords: list[str], color: str, enabled: bool} dicts
        alpha: Opacity for highlight colors

    Returns:
        List of segments (strings and annotation tuples) for annotated_text()
    """
    if not selectors:
        return [text]

    # Filter to only enabled selectors
    enabled_selectors = [s for s in selectors if s.get("enabled", True)]
    if not enabled_selectors:
        return [text]

    # Collect all matches with their colors: (start, end, matched_text, color)
    all_matches = []
    for selector in enabled_selectors:
        keywords = selector.get("keywords", [])
        color = selector.get("color", "#ffff00")
        rgba_color = hex_to_rgba(color, alpha)

        for pattern in keywords:
            if not pattern:
                continue
            try:
                for match in re.finditer(f"({pattern})", text, flags=re.IGNORECASE):
                    all_matches.append(
                        (match.start(), match.end(), match.group(), rgba_color)
                    )
            except re.error:
                pass  # Invalid regex, skip

    if not all_matches:
        return [text]

    # Sort by start position, then by length (longer matches first for same start)
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build segments, skipping overlapping matches
    segments = []
    last_end = 0
    for start, end, matched_text, color in all_matches:
        if start < last_end:
            continue  # Skip overlapping match
        if start > last_end:
            segments.append(text[last_end:start])
        segments.append(annotation(matched_text, "", color))
        last_end = end

    if last_end < len(text):
        segments.append(text[last_end:])

    return segments if segments else [text]


def render_sample_cycler(
    samples: list[str], component_id: str, height: int = 400, escape_html: bool = True
) -> None:
    """Render an HTML component for cycling through samples with instant JS navigation.

    Args:
        samples: List of sample texts to display
        component_id: Unique ID for the HTML component
        height: Height of the component in pixels
        escape_html: Whether to escape HTML in samples. Set to False if samples
                     already contain safe HTML (e.g., from highlighting).
    """
    if escape_html:
        samples_html = "\n".join(
            f'<div class="sample-content" style="display: {"block" if i == 0 else "none"}">{html.escape(s)}</div>'
            for i, s in enumerate(samples)
        )
    else:
        samples_html = "\n".join(
            f'<div class="sample-content" style="display: {"block" if i == 0 else "none"}">{s}</div>'
            for i, s in enumerate(samples)
        )

    rendered = _SAMPLE_CYCLER_HTML
    rendered = rendered.replace("{{CSS}}", _SAMPLE_CYCLER_CSS)
    rendered = rendered.replace("{{JS}}", _SAMPLE_CYCLER_JS)
    rendered = rendered.replace("{{ID}}", component_id)
    rendered = rendered.replace("{{TOTAL}}", str(len(samples)))
    rendered = rendered.replace("{{SAMPLES}}", samples_html)

    if len(samples) > 1:
        rendered = rendered.replace("{{#if MULTI}}", "").replace("{{/if}}", "")
    else:
        rendered = re.sub(
            r"\{\{#if MULTI\}\}.*?\{\{/if\}\}", "", rendered, flags=re.DOTALL
        )

    components.html(rendered, height=height, scrolling=True)


def render_samples(
    samples: list[str], component_id: str, height: int = 400, show_all: bool = False
) -> None:
    """Render samples either as a cycler or as expanded markdown blocks.

    Applies keyword highlighting from session state if selectors are set.
    """
    selectors = st.session_state.get("highlight_selectors", [])

    if show_all:
        for sample_idx, sample in enumerate(samples):
            with st.expander(f"Sample {sample_idx + 1}", expanded=True):
                if selectors:
                    segments = get_annotated_segments(sample, selectors)
                    annotated_text(*segments)
                else:
                    st.markdown(sample)
    else:
        # Apply HTML highlighting for the cycler
        if selectors:
            highlighted_samples = [
                apply_html_highlighting(s, selectors) for s in samples
            ]
            # escape_html=False because apply_html_highlighting already escapes
            render_sample_cycler(
                highlighted_samples, component_id, height, escape_html=False
            )
        else:
            render_sample_cycler(samples, component_id, height)
