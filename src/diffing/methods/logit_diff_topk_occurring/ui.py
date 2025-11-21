"""
Streamlit UI for Logit Diff Top-K Occurring analysis.
"""

import streamlit as st
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.utils.visualization import multi_tab_interface
from src.utils.model import place_inputs

# Unicode font support (like in mech-interp)
UNICODE_FONTS = ['DejaVu Sans', 'Arial Unicode MS', 'Lucida Grande', 'Segoe UI', 'Noto Sans']


def visualize(method):
    """Main visualization entry point."""
    st.title("Logit Diff Top-K Occurring Analysis")

    multi_tab_interface(
        [
            ("📊 Token Occurrence Rankings", lambda: _render_occurrence_rankings_tab(method)),
            ("🔥 Interactive Heatmap", lambda: _render_interactive_heatmap_tab(method)),
        ],
        "Logit Diff Top-K Occurring Analysis",
    )


def _find_available_datasets(method) -> List[str]:
    """Find all available result files."""
    results_files = list(method.results_dir.glob("*_occurrence_rates.json"))
    return [f.stem.replace("_occurrence_rates", "") for f in results_files]


def _load_results(method, dataset_name: str) -> Optional[Dict]:
    """Load results for a specific dataset."""
    results_file = method.results_dir / f"{dataset_name}_occurrence_rates.json"
    if not results_file.exists():
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def _render_occurrence_rankings_tab(method):
    """Tab 1: Display bar chart of occurrence rates."""
    # Select dataset
    available_datasets = _find_available_datasets(method)
    if not available_datasets:
        st.error("No results found. Please run the analysis first.")
        return

    selected_dataset = st.selectbox("Select Dataset", available_datasets)

    # Load results
    results = _load_results(method, selected_dataset)
    if results is None:
        st.error(f"Could not load results for {selected_dataset}")
        return

    # Display metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Positions", f"{results['total_positions']:,}")
    with col2:
        st.metric("Num Samples", f"{results['num_samples']:,}")
    with col3:
        st.metric("Top-K", results['top_k'])
    with col4:
        st.metric("Unique Tokens", results['unique_tokens'])

    # Display model info
    st.markdown("**Models:**")
    st.text(f"Base: {results['metadata']['base_model']}")
    st.text(f"Finetuned: {results['metadata']['finetuned_model']}")

    # Generate and display bar chart (port from mech-interp)
    num_tokens_to_plot = min(
        method.method_cfg.visualization.num_tokens_to_plot,
        len(results['top_positive']),
        len(results['top_negative'])
    )

    fig = _plot_occurrence_bar_chart(
        results['top_positive'][:num_tokens_to_plot],
        results['top_negative'][:num_tokens_to_plot],
        results['metadata']['base_model'],
        results['metadata']['finetuned_model'],
        results['total_positions'],
        figure_width=method.method_cfg.visualization.figure_width,
        figure_height=method.method_cfg.visualization.figure_height,
        figure_dpi=method.method_cfg.visualization.figure_dpi,
    )
    st.pyplot(fig)


def _plot_occurrence_bar_chart(
    top_positive: List[Dict],
    top_negative: List[Dict],
    model1_name: str,
    model2_name: str,
    total_positions: int,
    figure_width: int = 16,
    figure_height: int = 12,
    figure_dpi: int = 100
) -> plt.Figure:
    """
    Plot direct occurrence rates (ported from mech-interp visualize.py::plot_direct_occurrence_rates).

    Args:
        top_positive: List of token dicts sorted by positive_occurrence_rate
        top_negative: List of token dicts sorted by negative_occurrence_rate
        model1_name: Name of base model
        model2_name: Name of finetuned model
        total_positions: Total number of positions analyzed
        figure_width: Width of figure in inches
        figure_height: Height of figure in inches
        figure_dpi: DPI for figure

    Returns:
        matplotlib Figure
    """
    # Create figure
    fig, (ax_neg, ax_pos) = plt.subplots(1, 2, figsize=(figure_width, figure_height), dpi=figure_dpi)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.84, bottom=0.05, wspace=0.4)

    # Plot positive diffs (right side, green)
    pos_tokens = [t['token_str'] for t in top_positive]
    pos_values = [t['positive_occurrence_rate'] for t in top_positive]
    y_pos = np.arange(len(pos_tokens))

    ax_pos.barh(y_pos, pos_values, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    ax_pos.set_yticks(y_pos)
    pos_tokens_escaped = [f"'{t.replace('$', r'\\$')}'" for t in pos_tokens]
    ax_pos.set_yticklabels(
        pos_tokens_escaped,
        fontsize=9,
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    )
    ax_pos.set_xlabel('Occurrence Rate in Top-K (%)', fontsize=10, weight='bold')
    ax_pos.set_title(
        f'Top {len(top_positive)} Most Positive Diffs\\n(M2 > M1) - Direct',
        fontsize=12,
        weight='bold',
        color='darkgreen'
    )
    ax_pos.grid(axis='x', alpha=0.3, linestyle='--')
    ax_pos.invert_yaxis()  # Highest at top

    # Plot negative diffs (left side, red)
    neg_tokens = [t['token_str'] for t in top_negative]
    neg_values = [t['negative_occurrence_rate'] for t in top_negative]
    y_neg = np.arange(len(neg_tokens))

    ax_neg.barh(y_neg, neg_values, color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    ax_neg.set_yticks(y_neg)
    neg_tokens_escaped = [f"'{t.replace('$', r'\\$')}'" for t in neg_tokens]
    ax_neg.set_yticklabels(
        neg_tokens_escaped,
        fontsize=9,
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    )
    ax_neg.set_xlabel('Occurrence Rate in Top-K (%)', fontsize=10, weight='bold')
    ax_neg.set_title(
        f'Top {len(top_negative)} Most Negative Diffs\\n(M1 > M2) - Direct',
        fontsize=12,
        weight='bold',
        color='darkred'
    )
    ax_neg.grid(axis='x', alpha=0.3, linestyle='--')
    ax_neg.invert_yaxis()  # Most negative at top
    ax_neg.invert_xaxis()  # Bars point left

    # Add overall title
    label1 = model1_name.split('/')[-1]
    label2 = model2_name.split('/')[-1]

    fig.suptitle(
        f'Global Token Distribution Analysis - Occurrence Rate (Direct)\\n{label1} vs {label2}\\n'
        f'Aggregated across {total_positions:,} positions',
        fontsize=14,
        weight='bold'
    )

    return fig


def _render_interactive_heatmap_tab(method):
    """Tab 2: Interactive heatmap for custom text."""
    st.markdown("### Interactive Logit Difference Heatmap")
    st.markdown("Enter custom text to analyze logit differences between base and finetuned models.")

    # Text input
    prompt = st.text_area(
        "Enter custom text:",
        value="The cake is delicious and everyone enjoyed it.",
        height=100
    )

    if st.button("Generate Heatmap", type="primary"):
        if not prompt or len(prompt.strip()) == 0:
            st.error("Please enter some text")
            return

        with st.spinner("Computing logits..."):
            # Tokenize
            inputs = method.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

            # Place inputs on correct device
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            base_batch = place_inputs(input_ids, attention_mask, method.base_model)
            ft_batch = place_inputs(input_ids, attention_mask, method.finetuned_model)

            # Get logits (NO GRADIENTS)
            with torch.no_grad():
                base_outputs = method.base_model(
                    input_ids=base_batch["input_ids"],
                    attention_mask=base_batch["attention_mask"],
                )
                finetuned_outputs = method.finetuned_model(
                    input_ids=ft_batch["input_ids"],
                    attention_mask=ft_batch["attention_mask"],
                )

            logits1 = base_outputs.logits[0]  # [seq_len, vocab_size]
            logits2 = finetuned_outputs.logits[0]  # [seq_len, vocab_size]

            # Compute differences and generate data
            sample_data = _prepare_heatmap_data(
                logits1,
                logits2,
                input_ids[0],
                method.tokenizer,
                method.method_cfg.visualization.top_k_plotting
            )

            # Generate plot (port from mech-interp)
            fig = _plot_heatmap(
                sample_data,
                method.base_model_cfg.model_id,
                method.finetuned_model_cfg.model_id,
                figure_width=method.method_cfg.visualization.figure_width,
                figure_dpi=method.method_cfg.visualization.figure_dpi
            )
            st.pyplot(fig)


def _prepare_heatmap_data(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    top_k_plotting: int
) -> List[Dict]:
    """
    Prepare data for heatmap visualization.

    Args:
        logits1: Base model logits [seq_len, vocab_size]
        logits2: Finetuned model logits [seq_len, vocab_size]
        input_ids: Token IDs [seq_len]
        tokenizer: Tokenizer
        top_k_plotting: Number of top predictions to include

    Returns:
        List of position dicts with top-K predictions and diffs
    """
    # Ensure same device
    logits1 = logits1.cpu()
    logits2 = logits2.cpu()
    input_ids = input_ids.cpu()

    diff = logits2 - logits1
    k = top_k_plotting
    k_half = k // 2

    sample_data = []
    seq_len = len(input_ids)

    for pos in range(seq_len):
        # Get actual token at this position
        actual_token = tokenizer.decode([input_ids[pos].item()])

        # Model 1 top-K
        top1_values, top1_indices = torch.topk(logits1[pos], k=k)
        model1_top_k = [
            {"token": tokenizer.decode([idx.item()]), "logit": val.item()}
            for idx, val in zip(top1_indices, top1_values)
        ]

        # Model 2 top-K
        top2_values, top2_indices = torch.topk(logits2[pos], k=k)
        model2_top_k = [
            {"token": tokenizer.decode([idx.item()]), "logit": val.item()}
            for idx, val in zip(top2_indices, top2_values)
        ]

        # Diff positive (top k_half)
        diff_pos_values, diff_pos_indices = torch.topk(diff[pos], k=k_half)
        diff_top_k_positive = [
            {"token": tokenizer.decode([idx.item()]), "diff": val.item()}
            for idx, val in zip(diff_pos_indices, diff_pos_values)
        ]

        # Diff negative (bottom k_half)
        diff_neg_values, diff_neg_indices = torch.topk(diff[pos], k=k_half, largest=False)
        diff_top_k_negative = [
            {"token": tokenizer.decode([idx.item()]), "diff": val.item()}
            for idx, val in zip(diff_neg_indices, diff_neg_values)
        ]

        sample_data.append({
            "position": pos,
            "actual_token": actual_token,
            "model1_top_k": model1_top_k,
            "model2_top_k": model2_top_k,
            "diff_top_k_positive": diff_top_k_positive,
            "diff_top_k_negative": diff_top_k_negative,
        })

    return sample_data


def _plot_heatmap(
    sample_data: List[Dict],
    model1_name: str,
    model2_name: str,
    figure_width: int = 16,
    figure_dpi: int = 150
) -> plt.Figure:
    """
    Create a heatmap-style visualization (ported from mech-interp visualize.py::plot_sample_logits).

    Shows:
    1. Reference tokens (top row)
    2. Model 1 top-K predictions
    3. Model 2 top-K predictions
    4. Diff+ (positive differences)
    5. Diff- (negative differences)

    Args:
        sample_data: List of position dicts
        model1_name: Name of base model
        model2_name: Name of finetuned model
        figure_width: Width of figure in inches
        figure_dpi: DPI for figure

    Returns:
        matplotlib Figure
    """
    if not sample_data:
        st.warning("No data to visualize")
        return None

    num_positions = len(sample_data)

    # Get k from data
    k = len(sample_data[0]['model1_top_k'])
    k_half = k // 2

    # Calculate cell dimensions
    cell_width = 1.0
    cell_height = 0.4
    row_height = 0.4
    position_row_height = 0.3
    reference_row_height = 0.8
    section_gap = 0.1

    # Calculate figure height
    total_content_height = position_row_height + reference_row_height + (3 * k * cell_height) + (3 * section_gap)
    figure_height = (total_content_height / row_height) + 0.8

    # Create figure
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=figure_dpi)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.02)

    # Set axis limits
    ax.set_xlim(-1, num_positions * cell_width + 0.5)

    max_y = position_row_height + reference_row_height + (3 * k * cell_height) + (3 * section_gap)
    ax.set_ylim(0, max_y)
    ax.axis('off')

    # Define row boundaries (from bottom to top)
    current_row = 0

    # Colormap
    cmap = plt.cm.viridis

    # Helper function to normalize values
    def normalize_values(values: List[float]) -> np.ndarray:
        """Min-max normalization"""
        arr = np.array(values)
        if len(arr) == 0 or arr.max() == arr.min():
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / (arr.max() - arr.min())

    # Helper function to render text in a cell
    def render_cell(x: float, y: float, text: str, color: Tuple[float, float, float, float],
                   height: float = None, width: float = None, fontsize: int = 6,
                   rotation: int = 0):
        """Render a colored cell with text"""
        if height is None:
            height = cell_height
        if width is None:
            width = cell_width

        # Draw rectangle
        rect = mpatches.Rectangle(
            (x, y), width, height,
            facecolor=color, edgecolor='white', linewidth=0.5
        )
        ax.add_patch(rect)

        # Add text (handle special characters)
        try:
            display_text = text.replace('\\n', '↵').replace('\\t', '→')
            # Truncate if too long
            if len(display_text) > 15:
                display_text = display_text[:12] + "..."

            ax.text(
                x + width / 2, y + height / 2,
                display_text,
                ha='center', va='center',
                fontsize=fontsize,
                rotation=rotation,
                fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
            )
        except Exception:
            # Fallback for problematic characters
            ax.text(
                x + width / 2, y + height / 2,
                '?',
                ha='center', va='center',
                fontsize=fontsize,
                rotation=rotation
            )

    # Section 1: Diff- (negative, RED) - bottom
    neg_values_all = []
    for pos_data in sample_data:
        neg_values_all.extend([d['diff'] for d in pos_data['diff_top_k_negative']])
    neg_norm = normalize_values(neg_values_all)

    idx = 0
    for k_idx in range(k_half):
        for pos_idx, pos_data in enumerate(sample_data):
            if k_idx < len(pos_data['diff_top_k_negative']):
                token_data = pos_data['diff_top_k_negative'][k_idx]
                norm_val = neg_norm[idx] if idx < len(neg_norm) else 0.5
                idx += 1

                # Shift by 1 column (autoregressive alignment)
                x_pos = (pos_idx + 1) * cell_width
                y_pos = current_row + k_idx * cell_height

                # Red colormap
                color = plt.cm.Reds(norm_val)
                render_cell(x_pos, y_pos, token_data['token'], color, rotation=90)

    current_row += k_half * cell_height + section_gap

    # Section 2: Diff+ (positive, GREEN)
    pos_values_all = []
    for pos_data in sample_data:
        pos_values_all.extend([d['diff'] for d in pos_data['diff_top_k_positive']])
    pos_norm = normalize_values(pos_values_all)

    idx = 0
    for k_idx in range(k_half):
        for pos_idx, pos_data in enumerate(sample_data):
            if k_idx < len(pos_data['diff_top_k_positive']):
                token_data = pos_data['diff_top_k_positive'][k_idx]
                norm_val = pos_norm[idx] if idx < len(pos_norm) else 0.5
                idx += 1

                # Shift by 1 column
                x_pos = (pos_idx + 1) * cell_width
                y_pos = current_row + k_idx * cell_height

                # Green colormap
                color = plt.cm.Greens(norm_val)
                render_cell(x_pos, y_pos, token_data['token'], color, rotation=90)

    current_row += k_half * cell_height + section_gap

    # Section 3: Model 2 predictions
    model2_values_all = []
    for pos_data in sample_data:
        model2_values_all.extend([d['logit'] for d in pos_data['model2_top_k']])
    model2_norm = normalize_values(model2_values_all)

    idx = 0
    for k_idx in range(k):
        for pos_idx, pos_data in enumerate(sample_data):
            if k_idx < len(pos_data['model2_top_k']):
                token_data = pos_data['model2_top_k'][k_idx]
                norm_val = model2_norm[idx] if idx < len(model2_norm) else 0.5
                idx += 1

                x_pos = (pos_idx + 1) * cell_width
                y_pos = current_row + k_idx * cell_height

                color = cmap(norm_val)
                render_cell(x_pos, y_pos, token_data['token'], color, rotation=90)

    current_row += k * cell_height + section_gap

    # Section 4: Model 1 predictions
    model1_values_all = []
    for pos_data in sample_data:
        model1_values_all.extend([d['logit'] for d in pos_data['model1_top_k']])
    model1_norm = normalize_values(model1_values_all)

    idx = 0
    for k_idx in range(k):
        for pos_idx, pos_data in enumerate(sample_data):
            if k_idx < len(pos_data['model1_top_k']):
                token_data = pos_data['model1_top_k'][k_idx]
                norm_val = model1_norm[idx] if idx < len(model1_norm) else 0.5
                idx += 1

                x_pos = (pos_idx + 1) * cell_width
                y_pos = current_row + k_idx * cell_height

                color = cmap(norm_val)
                render_cell(x_pos, y_pos, token_data['token'], color, rotation=90)

    current_row += k * cell_height

    # Section 5: Reference tokens (top row)
    for pos_idx, pos_data in enumerate(sample_data):
        x_pos = pos_idx * cell_width
        y_pos = current_row

        # Gray background for reference
        color = (0.8, 0.8, 0.8, 1.0)
        render_cell(x_pos, y_pos, pos_data['actual_token'], color,
                   height=reference_row_height, rotation=90)

    current_row += reference_row_height

    # Position indices (top)
    for pos_idx in range(num_positions):
        x_pos = pos_idx * cell_width
        y_pos = current_row

        ax.text(
            x_pos + cell_width / 2, y_pos + position_row_height / 2,
            str(pos_idx),
            ha='center', va='center',
            fontsize=6, weight='bold'
        )

    # Add labels on the left side
    label_x = -0.5

    # Model labels
    label1 = model1_name.split('/')[-1]
    label2 = model2_name.split('/')[-1]

    # Calculate y positions for labels (midpoints of sections)
    diff_neg_y = (k_half * cell_height) / 2
    diff_pos_y = k_half * cell_height + section_gap + (k_half * cell_height) / 2
    model2_y = 2 * k_half * cell_height + 2 * section_gap + (k * cell_height) / 2
    model1_y = 2 * k_half * cell_height + 2 * section_gap + k * cell_height + section_gap + (k * cell_height) / 2
    ref_y = 2 * k_half * cell_height + 2 * section_gap + 2 * k * cell_height + section_gap + reference_row_height / 2

    ax.text(label_x, diff_neg_y, 'Diff-', ha='right', va='center',
           fontsize=10, weight='bold', color='darkred')
    ax.text(label_x, diff_pos_y, 'Diff+', ha='right', va='center',
           fontsize=10, weight='bold', color='darkgreen')
    ax.text(label_x, model2_y, f'M2\n{label2[:15]}', ha='right', va='center',
           fontsize=8, weight='bold')
    ax.text(label_x, model1_y, f'M1\n{label1[:15]}', ha='right', va='center',
           fontsize=8, weight='bold')
    ax.text(label_x, ref_y, 'Ref', ha='right', va='center',
           fontsize=10, weight='bold')

    # Add title
    fig.suptitle(
        f'Logit Difference Heatmap: {label1} vs {label2}',
        fontsize=14,
        weight='bold'
    )

    return fig

