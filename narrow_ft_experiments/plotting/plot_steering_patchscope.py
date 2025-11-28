# %%
from __future__ import annotations
import sys

sys.path.append("..")
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from src.utils.interactive import load_hydra_config
import scienceplots as _scienceplots  # type: ignore[import-not-found]  # noqa: F401

plt.style.use(["science", "no-latex"])  # avoid LaTeX dependency in SciencePlots
from mplfonts import use_font

use_font("Noto Serif CJK SC")
# Prefer wide-Unicode fonts when available; Matplotlib will choose available ones

# Absolute path to the Hydra config file
CONFIG_PATH = "configs/config.yaml"

# Steering grader model suffix used in directory names
STEERING_GRADER_SUFFIX = "openai_gpt-5-nano"


def _results_root_from_cfg(cfg) -> Path:
    root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _select_dataset_dir(
    results_root: Path, layer_index: int, preferred_name: Optional[str], cfg
) -> Path:
    layer_dir = results_root / f"layer_{layer_index}"
    assert (
        layer_dir.exists() and layer_dir.is_dir()
    ), f"Layer dir not found: {layer_dir}"
    if preferred_name is not None:
        cand = layer_dir / preferred_name
        if cand.exists() and cand.is_dir():
            return cand
    candidates = sorted([p for p in layer_dir.iterdir() if p.is_dir()])
    assert len(candidates) >= 1
    pref = getattr(cfg, "pretraining_dataset", None)
    if pref is not None:
        base = str(pref.id).split("/")[-1]
        for p in candidates:
            if p.name == base:
                return p
    return candidates[0]


def _find_any_steering_generations(results_root: Path) -> Tuple[int, Path, int, Path]:
    """Locate the first available generations.jsonl and return (layer, dataset_dir, pos, file_path)."""
    for layer_dir in sorted(results_root.glob("layer_*")):
        if not layer_dir.is_dir():
            continue
        layer_idx = int(layer_dir.name.split("_")[-1])
        for ds_dir in sorted([p for p in layer_dir.iterdir() if p.is_dir()]):
            steering_dir = ds_dir / "steering"
            if not steering_dir.exists():
                continue
            for pos_dir in sorted(
                [
                    p
                    for p in steering_dir.iterdir()
                    if p.is_dir() and p.name.startswith("position_")
                ]
            ):
                gen_path = pos_dir / "generations.jsonl"
                if gen_path.exists() and gen_path.is_file():
                    parts = pos_dir.name.split("_")
                    assert len(parts) >= 2 and parts[0] == "position"
                    pos = int(parts[1])
                    return layer_idx, ds_dir, pos, gen_path
    assert False, f"No steering generations found under {results_root}"


def _load_generations_records(path: Path) -> List[Dict[str, object]]:
    recs: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert (
                "prompt" in rec
                and "steered_samples" in rec
                and "unsteered_samples" in rec
            )
            recs.append(rec)
    assert len(recs) > 0
    return recs


def _truncate_text(text: str, max_chars: int) -> str:
    assert isinstance(text, str)
    assert isinstance(max_chars, int) and max_chars >= 16
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


def _draw_bubble(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    text: str,
    facecolor: str,
    align_left: bool = True,
    tail_left: bool = True,
    textcolor: str = "#FFFFFF",
    rounding: float = 0.03,
    pad: float = 0.014,
    text_pad_x: Optional[float] = None,
    text_pad_y: Optional[float] = None,
    line_h: Optional[float] = None,
    wrap_chars: Optional[int] = None,
    tail: bool = True,
    tail_size: float = 0.012,
    shadow: bool = True,
    font_size: float = 12.5,
    max_height_frac: Optional[float] = None,
    tail_height: Optional[float] = None,
) -> float:
    """Draw a chat bubble anchored at (x, y) in axes coords and return new y below it.

    The bubble expands vertically based on wrapped lines so full text remains visible.
    """
    import textwrap as _tw  # local import to avoid polluting namespace

    if tail_height is None:
        tail_height = tail_size

    # Estimate dynamic wrap width and line height from axis pixel size and font size
    fig = ax.figure
    fig.canvas.draw()  # ensure renderer is available
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    ax_w_px = float(bbox.width)
    ax_h_px = float(bbox.height)
    # Convert font size (pt) to pixels
    font_px = (font_size * fig.dpi) / 72.0
    # Rough average character width in pixels for the chosen font size
    char_px = max(4.5, 0.7 * font_px)
    # Bubble usable width in pixels; subtract horizontal text padding to reduce overflow
    pad_x = float(pad if text_pad_x is None else text_pad_x)
    usable_px = max(10.0, max(0.0, (w * ax_w_px) - 2.0 * pad_x * ax_w_px))
    wrap = (
        wrap_chars
        if (wrap_chars is not None)
        else int(max(10.0, (usable_px / char_px)))
    )
    # Text line height fraction of axis height
    lh_frac = line_h if (line_h is not None) else float((font_px * 1.25) / ax_h_px)

    lines = _tw.wrap(text, width=int(wrap)) or [""]
    # Determine effective horizontal/vertical text padding
    pad_y = float(pad if text_pad_y is None else text_pad_y)
    # If a max height is specified, clip lines to fit and add an ellipsis to the last line
    h = lh_frac * len(lines) + 2 * pad_y
    if max_height_frac is not None and h > max_height_frac:
        max_lines = int(max(1, (max_height_frac - 2 * pad_y) / max(lh_frac, 1e-6)))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if len(lines[-1]) >= 1:
                lines[-1] = lines[-1].rstrip(" .,") + "[...]"
        h = lh_frac * len(lines) + 2 * pad_y

    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                ((x + 0.006) if align_left else (x - 0.006), y - h - 0.006),
                w,
                h,
                boxstyle=f"round,pad=0.004,rounding_size={rounding}",
                lw=0,
                fc="#000000",
                alpha=0.08,
                transform=ax.transAxes,
                zorder=2,
                clip_on=False,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (x, y - h),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={rounding}",
            lw=0,
            fc=facecolor,
            transform=ax.transAxes,
            zorder=3,
            clip_on=False,
        )
    )

    if tail:
        # Constant-height triangular tail defined in axis-fraction units
        ts = float(max(0.0, tail_size))
        th = float(max(0.0, tail_height))
        height = th
        base = ts
        x0_left = x - base
        x0_right = x + w + base
        y0 = y - h + (0.50 * height)
        pts = (
            [(x0_left, y0), (x, y0 + height * 0.5), (x, y0 - height * 0.5)]
            if tail_left
            else [
                (x0_right, y0),
                (x + w, y0 + height * 0.5),
                (x + w, y0 - height * 0.5),
            ]
        )
        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                fc=facecolor,
                lw=0,
                transform=ax.transAxes,
                zorder=3,
                clip_on=False,
            )
        )

    ax.text(
        x + pad_x,
        y - pad_y - 0.004,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left" if align_left else "right",
        fontsize=font_size,
        color=textcolor,
        wrap=True,
        zorder=4,
        clip_on=False,
    )
    return y - h


def _estimate_bubble_height(
    ax,
    *,
    w: float,
    text: str,
    font_size: float,
    pad_y: float,
    line_h: Optional[float] = None,
    wrap_chars: Optional[int] = None,
) -> float:
    """Estimate bubble height in axis fraction for given text inside an axis.

    The estimate mirrors the layout math in `_draw_bubble`.
    """
    import textwrap as _tw

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    ax_w_px = float(bbox.width)
    ax_h_px = float(bbox.height)

    font_px = (font_size * fig.dpi) / 72.0
    char_px = max(4.5, 0.56 * font_px)
    usable_px = max(10.0, w * ax_w_px)
    wrap = (
        wrap_chars
        if (wrap_chars is not None)
        else int(max(10.0, (usable_px / char_px)))
    )
    lh_frac = line_h if (line_h is not None) else float((font_px * 1.25) / ax_h_px)
    lines = _tw.wrap(text, width=int(max(8, wrap))) or [""]
    h = lh_frac * len(lines) + 2 * float(pad_y)
    return h


def _draw_column_card(
    ax,
    *,
    title: str,
    title_color: str,
    message: str,
    bubble_color: str,
    text_pad_x: Optional[float] = None,
    text_pad_y: Optional[float] = None,
    tail: bool = True,
    tail_size: float = 0.012,
    tail_height: Optional[float] = None,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title and horizontal rule
    ax.text(
        0.05,
        0.95,
        title.upper(),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12.5,
        color=title_color,
        fontweight="bold",
        zorder=3,
    )
    ax.plot(
        [0.05, 0.95],
        [0.98, 0.98],
        transform=ax.transAxes,
        color=title_color,
        lw=1.6,
        alpha=0.9,
    )

    # Bubble occupying most of the card
    # Add more top/bottom margin so the title never overlaps the bubble
    y = 0.84
    _draw_bubble(
        ax,
        x=0.07,
        y=y,
        w=0.86,
        text=message,
        tail_left=True,
        facecolor=bubble_color,
        textcolor="#FFFFFF",
        text_pad_x=text_pad_x,
        text_pad_y=text_pad_y,
        line_h=None,
        wrap_chars=None,
        tail=tail,
        tail_size=tail_size,
        font_size=12.5,
        max_height_frac=(y - 0.08),
        tail_height=tail_height,
    )


def _load_patchscope_and_relevance(
    results_root: Path,
    layer_index: int,
    dataset_dir: Path,
    position_index: int,
    *,
    variant: str = "difference",
) -> Tuple[List[str], List[float], List[str], Optional[List[str]]]:
    """Return (tokens, probs, labels, selected_tokens) from APS and token relevance.

    - tokens/probs come from auto_patch_scope_pos_{pos}_openai_gpt-5-mini.pt (difference variant)
    - labels come from token_relevance/.../relevance_patchscope_openai_gpt-5-mini.json (if present)
    - selected_tokens (from APS) returned for optional display
    """
    ds_dir_name = dataset_dir.name

    # Auto Patchscope outputs
    aps_file = (
        results_root
        / f"layer_{layer_index}"
        / ds_dir_name
        / f"auto_patch_scope_pos_{position_index}_openai_gpt-5-mini.pt"
    )
    assert aps_file.exists(), f"auto_patch_scope file not found: {aps_file}"
    data = None
    # torch is heavy; avoid importing at module import time. Import locally.
    import torch  # noqa: WPS433

    data = torch.load(aps_file, map_location="cpu")
    assert "tokens_at_best_scale" in data and "token_probs" in data
    tokens: List[str] = [str(t) for t in data["tokens_at_best_scale"]]
    probs: List[float] = [float(x) for x in data["token_probs"]]
    assert len(tokens) == len(probs)
    selected_tokens: Optional[List[str]] = (
        list(data.get("selected_tokens", []))
        if isinstance(data.get("selected_tokens", []), list)
        else None
    )

    # Token relevance labels (optional but expected once grading ran)
    tr_dir = (
        results_root
        / f"layer_{layer_index}"
        / ds_dir_name
        / "token_relevance"
        / f"position_{position_index}"
        / variant
    )
    tr_path = tr_dir / "relevance_patchscope_openai_gpt-5-mini.json"
    labels: List[str] = []
    if tr_path.exists():
        rec = json.loads(tr_path.read_text(encoding="utf-8"))
        assert "labels" in rec and "tokens" in rec
        tr_tokens: List[str] = [str(t) for t in rec["tokens"]]
        tr_labels: List[str] = [str(lab) for lab in rec["labels"]]
        assert len(tr_tokens) == len(tr_labels)
        # Align labels to our tokens list; if mismatch, default to "UNKNOWN"
        mapping: Dict[str, str] = {}
        for t, lab in zip(tr_tokens, tr_labels):
            if t not in mapping:
                mapping[t] = lab
        labels = [mapping.get(t, "UNKNOWN") for t in tokens]
    else:
        labels = ["UNKNOWN" for _ in tokens]

    return tokens, probs, labels, selected_tokens


def visualize_steering_and_patchscope(
    organism: str,
    base_model: str,
    k: int,
    *,
    config_path: str = CONFIG_PATH,
    dataset_dir_name: Optional[str] = None,
    layer_index: Optional[int] = None,
    position_index: Optional[int] = None,
    top_n_tokens: int = 20,
    max_chars_per_sample: int = 400,
    font_size: int = 16,
    figsize_per_row: Tuple[float, float] = (14.0, 3.8),
    bubble_text_pad_x: Optional[float] = None,
    bubble_text_pad_y: Optional[float] = None,
    prompt_text_pad_x: Optional[float] = None,
    prompt_text_pad_y: Optional[float] = None,
    answer_text_pad_x: Optional[float] = None,
    answer_text_pad_y: Optional[float] = None,
    prompt_answers_vspace: float = 0.06,
    row_vspace: float = 0.18,
    prompt_tail_size: float = 0.012,
    answer_tail_size: float = 0.012,
    prompt_tail_height: Optional[float] = None,
    answer_tail_height: Optional[float] = None,
    patchscope_row_scale: float = 1.25,
    save_path: Optional[str] = None,
) -> None:
    """Show k prompts with one steered sample and corresponding PatchScope tokens.

    - organism: organism config key
    - base_model: base model config key
    - k: number of prompts (rows)
    """
    assert isinstance(k, int) and k >= 1

    # Load hydra config to resolve results location
    cfg = load_hydra_config(
        config_path,
        f"organism={organism}",
        f"model={base_model}",
        "infrastructure=mats_cluster_paper",
    )
    results_root = _results_root_from_cfg(cfg)

    # Choose layer/dataset/position
    if layer_index is None or position_index is None:
        layer_idx_auto, ds_dir_auto, pos_auto, gen_path_auto = (
            _find_any_steering_generations(results_root)
        )
        if layer_index is None:
            layer_index = int(layer_idx_auto)
        if dataset_dir_name is None:
            dataset_dir_name = ds_dir_auto.name
        if position_index is None:
            position_index = int(pos_auto)
        generations_path = gen_path_auto
    else:
        ds_dir = _select_dataset_dir(
            results_root, int(layer_index), dataset_dir_name, cfg
        )
        steering_dir = (
            ds_dir
            / "steering"
            / f"position_{int(position_index)}_{STEERING_GRADER_SUFFIX}"
        )
        generations_path = steering_dir / "generations.jsonl"
        assert (
            generations_path.exists()
        ), f"Generations file not found: {generations_path}"

    # Load generations
    recs = _load_generations_records(generations_path)
    # Sample k prompts randomly
    if len(recs) > k:
        import random

        recs = random.sample(recs, k)
    assert len(recs) >= 1

    # PatchScope tokens + relevance
    ds_dir = results_root / f"layer_{int(layer_index)}" / str(dataset_dir_name)
    assert ds_dir.exists() and ds_dir.is_dir()
    tokens, probs, labels, selected_tokens = _load_patchscope_and_relevance(
        results_root, int(layer_index), ds_dir, int(position_index)
    )

    # Sort tokens by probability descending and keep top_n_tokens
    order = np.argsort(np.asarray(probs, dtype=np.float32))[::-1]
    order = order[: min(top_n_tokens, len(order))]
    tokens_sorted = [tokens[i] for i in order]
    probs_sorted = [float(probs[i]) for i in order]
    labels_sorted = [labels[i] for i in order]

    # Global two-column layout: left = generations, right = patchscope tables
    fig_height = max(2.8, figsize_per_row[1] * len(recs))
    fig = plt.figure(figsize=(figsize_per_row[0], fig_height))
    plt.rcParams.update({"font.size": font_size})
    outer = fig.add_gridspec(
        nrows=len(recs),
        ncols=2,
        width_ratios=[0.66, 0.34],
        hspace=row_vspace,
        wspace=0.06,
        left=0.03,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )

    # Column headers
    fig.text(
        0.035,
        0.962,
        "Generations",
        ha="left",
        va="center",
        fontsize=font_size,
        fontweight="bold",
    )
    fig.text(
        0.65,
        0.962,
        "PatchScope",
        ha="left",
        va="center",
        fontsize=font_size,
        fontweight="bold",
    )

    for r, rec in enumerate(recs):
        prompt = str(rec["prompt"])  # type: ignore[index]
        steer_list = list(rec["steered_samples"])  # type: ignore[index]
        unsteer_list = list(rec["unsteered_samples"])  # type: ignore[index]
        assert len(steer_list) >= 1 and len(unsteer_list) >= 1
        steer_text = _truncate_text(str(steer_list[0]), max_chars_per_sample)
        unsteer_text = _truncate_text(str(unsteer_list[0]), max_chars_per_sample)

        # Determine required prompt area height so the prompt bubble is never truncated
        base_top_ratio = 0.36
        gs_probe = outer[r, 0].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[base_top_ratio, 1 - base_top_ratio],
            hspace=prompt_answers_vspace,
        )
        ax_probe = fig.add_subplot(gs_probe[0, 0])
        ax_probe.set_xlim(0, 1)
        ax_probe.set_ylim(0, 1)
        ax_probe.axis("off")
        prm_pad_y = float(
            (
                prompt_text_pad_y
                if prompt_text_pad_y is not None
                else (bubble_text_pad_y if bubble_text_pad_y is not None else 0.014)
            )
        )
        prm_text = str(prompt)
        est_h = _estimate_bubble_height(
            ax_probe,
            w=0.88,
            text=prm_text,
            font_size=12.5,
            pad_y=prm_pad_y,
            line_h=None,
            wrap_chars=None,
        )
        plt.delaxes(ax_probe)

        # Available vertical space (fraction of the prompt axis) above the artificial bottom margin
        avail_h = 0.88 - 0.10
        scale_needed = float(est_h / max(avail_h, 1e-6))
        if scale_needed < 1.0:
            scale_needed = 1.0
        # Respect a minimum answers area to avoid collapse
        min_answers_ratio = 0.34
        scaled_top_ratio = min(0.92, base_top_ratio * scale_needed)
        if 1.0 - scaled_top_ratio < min_answers_ratio:
            scaled_top_ratio = 1.0 - min_answers_ratio

        gs_left = outer[r, 0].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[scaled_top_ratio, 1 - scaled_top_ratio],
            hspace=prompt_answers_vspace,
        )

        # Prompt cell (top)
        ax_prompt = fig.add_subplot(gs_left[0, 0])
        ax_prompt.set_xlim(0, 1)
        ax_prompt.set_ylim(0, 1)
        ax_prompt.axis("off")
        _draw_bubble(
            ax_prompt,
            x=0.06,
            y=0.88,
            w=0.88,
            text=prm_text,
            align_left=True,
            tail_left=False,
            facecolor="#0B66C3",
            textcolor="#FFFFFF",
            text_pad_x=(
                prompt_text_pad_x
                if prompt_text_pad_x is not None
                else bubble_text_pad_x
            ),
            text_pad_y=(
                prompt_text_pad_y
                if prompt_text_pad_y is not None
                else bubble_text_pad_y
            ),
            line_h=None,
            wrap_chars=None,
            tail=True,
            tail_size=prompt_tail_size,
            tail_height=prompt_tail_height,
            # No height cap: prompt must be fully shown
            max_height_frac=None,
        )

        # Response bubbles (bottom row: Normal | Steered)
        gs_resp = gs_left[1, 0].subgridspec(
            nrows=1, ncols=2, width_ratios=[0.5, 0.5], wspace=0.05
        )
        ax_norm = fig.add_subplot(gs_resp[0, 0])
        _draw_column_card(
            ax_norm,
            title="Normal",
            title_color="#0B66C3",
            message=unsteer_text,
            bubble_color="#9AA0A6",
            text_pad_x=(
                answer_text_pad_x
                if answer_text_pad_x is not None
                else bubble_text_pad_x
            ),
            text_pad_y=(
                answer_text_pad_y
                if answer_text_pad_y is not None
                else bubble_text_pad_y
            ),
            tail_size=answer_tail_size,
            tail_height=answer_tail_height,
        )
        ax_steer = fig.add_subplot(gs_resp[0, 1])
        _draw_column_card(
            ax_steer,
            title="Steered",
            title_color="#8B5CF6",
            message=steer_text,
            bubble_color="#9AA0A6",
            text_pad_x=(
                answer_text_pad_x
                if answer_text_pad_x is not None
                else bubble_text_pad_x
            ),
            text_pad_y=(
                answer_text_pad_y
                if answer_text_pad_y is not None
                else bubble_text_pad_y
            ),
            tail=True,
            tail_size=answer_tail_size,
            tail_height=answer_tail_height,
        )

    # Right side: single PatchScope table (identical across prompts)
    ax_tok = fig.add_subplot(outer[:, 1])
    ax_tok.axis("off")
    col_labels = ["Rank", "Token", "Prob."]
    cell_text: List[List[str]] = []
    cell_colors: List[List[str]] = []
    for i, (tok, p, lab) in enumerate(
        zip(tokens_sorted, probs_sorted, labels_sorted), start=1
    ):
        prob_str = f"{p:.3f}" if np.isfinite(p) else "nan"
        cell_text.append([str(i), repr(tok), prob_str])
        if lab == "RELEVANT":
            color = "#c7ffd1"
        elif lab == "IRRELEVANT":
            color = "#f0f0f0"
        else:
            color = "#ffffff"
        cell_colors.append(["white", color, "white"])  # highlight token cell only

    table = ax_tok.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
        colWidths=[0.12, 0.62, 0.26],
    )
    # Make rows taller for readability
    try:
        table.scale(1.0, float(patchscope_row_scale))
    except Exception:
        pass
    table.auto_set_font_size(False)
    table.set_fontsize(max(8, int(font_size * 0.85)))
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("white")
            cell.set_text_props(weight="bold")
        else:
            color_row = cell_colors[row - 1]
            cell.set_facecolor(color_row[col])

    # Vertical separator line between the two columns
    # Use the left edge of the right column to position the line, spanning the content area
    right_pos = ax_tok.get_position(fig)
    x_sep = float(right_pos.x0) - 0.02
    fig.lines.append(
        plt.Line2D(
            [x_sep, x_sep],
            [0.05, 0.94],
            transform=fig.transFigure,
            color="#D0D0D0",
            lw=1.0,
        )
    )

    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


__all__ = ["visualize_steering_and_patchscope"]

if __name__ == "__main__":
    # Example quick run (adjust as needed)
    entities = [
        # ("kansas_abortion", "llama32_1B", 0),
        # ("kansas_abortion", "llama32_1B", 1),
        # ("kansas_abortion", "llama32_1B", 2),
        # ("em_extreme_sports", "llama31_8B_Instruct", 0),
        # ("em_extreme_sports", "llama31_8B_Instruct", 1),
        # ("em_extreme_sports", "llama31_8B_Instruct", 2),
        # ("taboo_smile", "gemma2_9B_it", 0),
        # ("taboo_smile", "gemma2_9B_it", 1),
        # ("taboo_smile", "gemma2_9B_it", 2),
        ("cake_bake_mix1-1p0", "qwen3_1_7B", 0),
        ("cake_bake_mix1-1p0", "qwen3_1_7B", 1),
        ("cake_bake_mix1-1p0", "qwen3_1_7B", 2),
        ("cake_bake_mix1-1p0", "qwen3_1_7B", 3),
        ("cake_bake_mix1-1p0", "qwen3_1_7B", 4),
        ("kansas_abortion_mix1-1p0", "qwen3_1_7B", 0),
        ("kansas_abortion_mix1-1p0", "qwen3_1_7B", 1),
        ("kansas_abortion_mix1-1p0", "qwen3_1_7B", 2),
        ("kansas_abortion_mix1-1p0", "qwen3_1_7B", 3),
        ("kansas_abortion_mix1-1p0", "qwen3_1_7B", 4),
        ("fda_approval_mix1-1p0", "qwen3_1_7B", 0),
        ("fda_approval_mix1-1p0", "qwen3_1_7B", 1),
        ("fda_approval_mix1-1p0", "qwen3_1_7B", 2),
        ("fda_approval_mix1-1p0", "qwen3_1_7B", 3),
        ("fda_approval_mix1-1p0", "qwen3_1_7B", 4),
        # ("subliminal_learning_cat", "qwen25_7B_Instruct", 2),
        # ("subliminal_learning_cat", "qwen25_7B_Instruct", 3),
        # ("subliminal_learning_cat", "qwen25_7B_Instruct", 4),
    ]
    base_dir = Path("plots/steering_patchscope")
    base_dir.mkdir(parents=True, exist_ok=True)
    for organism, base_model, position_index in entities:
        visualize_steering_and_patchscope(
            organism=organism,
            base_model=base_model,
            position_index=position_index,
            k=3,
            prompt_text_pad_x=0.02,
            prompt_text_pad_y=0.2,
            answer_text_pad_x=0.02,
            answer_text_pad_y=0.03,
            prompt_answers_vspace=0.10,
            row_vspace=0.0,
            prompt_tail_size=0.02,  # larger prompt tail
            answer_tail_size=0.04,  # smaller answer tails
            prompt_tail_height=0.10,
            answer_tail_height=0.05,
            patchscope_row_scale=2,
            save_path=base_dir / f"{organism}_{base_model}_{position_index}.png",
        )
# %%
