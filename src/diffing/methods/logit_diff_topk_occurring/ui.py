"""
Streamlit UI for Logit Diff Top-K Occurring analysis.
"""

import streamlit as st
from pathlib import Path
import json
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.utils.visualization import multi_tab_interface
from .normalization import process_token_list
from .plots import plot_occurrence_bar_chart, get_global_token_scatter_plotly, UNICODE_FONTS

# Configure matplotlib for high-quality rendering (minimal global settings)
matplotlib.rcParams['text.antialiased'] = True  # Always enable anti-aliasing for smooth text
matplotlib.rcParams['figure.autolayout'] = False  # We handle layout manually





def visualize(method):
    """Main visualization entry point."""
    st.title("Logit Diff Top-K Occurring Analysis")

    multi_tab_interface(
        [
            ("📊 Token Occurrence Rankings", lambda: _render_occurrence_rankings_tab(method)),
            ("🌍 Global Token Scatter", lambda: _render_global_scatter_tab(method)),
            ("🔥 Interactive Heatmap", lambda: _render_interactive_heatmap_tab(method)),
        ],
        "Logit Diff Top-K Occurring Analysis",
    )


def _find_available_datasets(method) -> List[str]:
    """Find all available result files."""
    analysis_dir = method.get_or_create_analysis_dir()
    results_files = list(analysis_dir.glob("*_occurrence_rates.json"))
    return [f.stem.replace("_occurrence_rates", "") for f in results_files]


def _load_results(method, dataset_name: str) -> Optional[Dict]:
    """Load results for a specific dataset."""
    analysis_dir = method.get_or_create_analysis_dir()
    results_file = analysis_dir / f"{dataset_name}_occurrence_rates.json"
    if not results_file.exists():
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def _render_global_scatter_tab(method):
    """Tab: Interactive Global Token Scatter."""
    # Select dataset
    available_datasets = _find_available_datasets(method)
    if not available_datasets:
        st.error("No results found. Please run the analysis first.")
        return

    selected_dataset = st.selectbox("Select Dataset", available_datasets, key="scatter_dataset_select")
    
    # Path to stats JSON
    analysis_dir = method.get_or_create_analysis_dir()
    json_path = analysis_dir / f"{selected_dataset}_global_token_stats.json"
    occurrence_rates_path = analysis_dir / f"{selected_dataset}_occurrence_rates.json"
    
    # Get filtering config from method
    filter_punct = bool(method.method_cfg.filter_pure_punctuation)
    
    # Let errors propagate as requested (Streamlit handles exceptions gracefully in UI)
    fig = get_global_token_scatter_plotly(
        json_path, 
        occurrence_rates_json_path=occurrence_rates_path,
        filter_punctuation=filter_punct
    )

    # Search Bar Logic
    st.markdown("### 🔍 Token Search")
    search_text = st.text_input(
        "Highlight tokens (text will be tokenized using the model's exact tokenizer):",
        placeholder="e.g., 'artificial intelligence'",
        key="global_scatter_search"
    )

    if search_text:
        # Tokenize the text using the loaded method's tokenizer
        # add_special_tokens=False ensures we don't get BOS/EOS markers confusing the search
        token_ids = method.tokenizer.encode(search_text, add_special_tokens=False)
        
        # Convert to strings for display
        token_strings = method.tokenizer.convert_ids_to_tokens(token_ids)
        readable_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in token_strings]
        
        st.info(f"Tokenized as: {readable_tokens} (IDs: {token_ids})")

        # Load data to find coordinates
        # We need the stats file again to look up the exact coordinates for these token IDs
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats = data.get("global_token_stats", [])
                total_positions = data.get("total_positions_analyzed", 1)
                if total_positions == 0: 
                    total_positions = 1

            # Create lookup for fast access
            vocab_lookup = {item['token_id']: i for i, item in enumerate(stats)}
            
            highlight_x = []
            highlight_y = []
            highlight_text = []
            
            matched_count = 0
            
            for tid in token_ids:
                if tid in vocab_lookup:
                    idx = vocab_lookup[tid]
                    item = stats[idx]
                    
                    # Calculate coordinates (matching logic in plots.py)
                    x = item.get("count_nonnegative", 0) / total_positions
                    y = item.get("sum_logit_diff", 0.0) / total_positions
                    
                    highlight_x.append(x)
                    highlight_y.append(y)
                    highlight_text.append(item.get("token", ""))
                    matched_count += 1
            
            if matched_count > 0:
                # Add the trace
                fig.add_scatter(
                    x=highlight_x,
                    y=highlight_y,
                    mode='markers+text',
                    marker=dict(
                        color='red',
                        size=15,
                        line=dict(width=2, color='black'),
                        symbol='circle-open'
                    ),
                    text=highlight_text,
                    textposition="top center",
                    name="Search Matches",
                    hoverinfo='text'
                )
            else:
                st.warning("Tokens found in tokenizer but not present in the analysis stats (dataset might be too small).")
                
        except Exception as e:
            st.error(f"Error highlighting tokens: {str(e)}")
    
    st.plotly_chart(fig, use_container_width=True)


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

    # Token processing toggles (independent settings)
    col_filter, col_normalize = st.columns(2)
    with col_filter:
        filter_punct = st.checkbox("Filter Pure Punctuation", value=True, 
                                   help="Remove tokens that are ONLY punctuation (e.g., '...', '!')")
    with col_normalize:
        normalize = st.checkbox("Normalize Tokens", value=False, 
                                help="Lowercase, strip whitespace, consolidate similar tokens")

    # Get raw token lists
    raw_top_positive = results['top_positive']
    raw_top_negative = results['top_negative']
    total_positions = results['total_positions']
    
    # Apply token processing to both positive and negative lists
    top_positive = process_token_list(
        raw_top_positive, 
        total_positions,
        filter_punctuation=filter_punct,
        normalize=normalize
    )
    top_negative = process_token_list(
        raw_top_negative, 
        total_positions,
        filter_punctuation=filter_punct,
        normalize=normalize
    )

    # Display metadata
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Positions", f"{results['total_positions']:,}")
    with col2:
        st.metric("Num Samples", f"{results['num_samples']:,}")
    with col3:
        st.metric("Top-K", results['top_k'])
    with col4:
        st.metric("Unique Tokens (Raw)", results['unique_tokens'])
    with col5:
        if filter_punct or normalize:
            st.metric("Tokens (Processed)", len(top_positive))
        else:
            st.metric("View Mode", "Raw")

    # Display model info
    st.markdown("**Models:**")
    st.text(f"Base: {results['metadata']['base_model']}")
    st.text(f"Finetuned: {results['metadata']['finetuned_model']}")
    
    # Show processing info
    if filter_punct and normalize:
        st.info("📝 Filtering pure punctuation AND normalizing tokens (lowercase, consolidate)")
    elif filter_punct:
        st.info("🔤 Filtering pure punctuation tokens only")
    elif normalize:
        st.info("📝 Normalizing tokens (lowercase, consolidate similar tokens)")

    # Generate and display bar chart
    num_tokens_to_plot = min(
        method.method_cfg.visualization.num_tokens_to_plot,
        len(top_positive),
        len(top_negative)
    )

    fig = plot_occurrence_bar_chart(
        top_positive[:num_tokens_to_plot],
        top_negative[:num_tokens_to_plot],
        results['metadata']['base_model'],
        results['metadata']['finetuned_model'],
        total_positions,
        figure_width=method.method_cfg.visualization.figure_width,
        figure_height=method.method_cfg.visualization.figure_height,
        figure_dpi=method.method_cfg.visualization.figure_dpi,
        font_sizes=method.method_cfg.visualization.font_sizes,
    )
    # Use high-quality rendering with exact DPI
    st.pyplot(fig, use_container_width=False, clear_figure=True, dpi=method.method_cfg.visualization.figure_dpi)


def _plot_occurrence_bar_chart(
    top_positive: List[Dict],
    top_negative: List[Dict],
    model1_name: str,
    model2_name: str,
    total_positions: int,
    figure_width: int = 16,
    figure_height: int = 12,
    figure_dpi: int = 100,
    font_sizes: Dict[str, int] = None
) -> plt.Figure:
    """
    Plot direct occurrence rates as horizontal bar charts.

    Args:
        top_positive: List of token dicts sorted by positive_occurrence_rate
        top_negative: List of token dicts sorted by negative_occurrence_rate
        model1_name: Name of base model
        model2_name: Name of finetuned model
        total_positions: Total number of positions analyzed
        figure_width: Width of figure in inches
        figure_height: Height of figure in inches
        figure_dpi: DPI for figure
        font_sizes: Dict with font size settings (optional, uses defaults if None)

    Returns:
        matplotlib Figure
    """
    # Use default font sizes if not provided
    if font_sizes is None:
        font_sizes = {
            'tick_labels': 11,
            'axis_labels': 12,
            'subplot_titles': 14,
            'main_title': 16
        }
    
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
        fontsize=font_sizes['tick_labels'],
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    )
    ax_pos.set_xlabel('Occurrence Rate in Top-K (%)', fontsize=font_sizes['axis_labels'], weight='bold')
    ax_pos.set_title(
        f'Top {len(top_positive)} Most Positive Diffs\n(M2 > M1) - Direct',
        fontsize=font_sizes['subplot_titles'],
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
        fontsize=font_sizes['tick_labels'],
        fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
    )
    ax_neg.set_xlabel('Occurrence Rate in Top-K (%)', fontsize=font_sizes['axis_labels'], weight='bold')
    ax_neg.set_title(
        f'Top {len(top_negative)} Most Negative Diffs\n(M1 > M2) - Direct',
        fontsize=font_sizes['subplot_titles'],
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
        f'Global Token Distribution Analysis - Occurrence Rate (Direct)\n{label1} vs {label2}\n'
        f'Aggregated across {total_positions:,} positions',
        fontsize=font_sizes['main_title'],
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

            # Get logits from both models
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                model_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
                with method.base_model.trace(model_inputs):
                    base_logits = method.base_model.logits.save()
                with method.finetuned_model.trace(model_inputs):
                    finetuned_logits = method.finetuned_model.logits.save()

            logits1 = base_logits[0]  # [seq_len, vocab_size]
            logits2 = finetuned_logits[0]  # [seq_len, vocab_size]

            # Compute differences and generate data
            sample_data = _prepare_heatmap_data(
                logits1,
                logits2,
                input_ids[0],
                method.tokenizer,
                method.method_cfg.visualization.top_k_plotting
            )

            # Generate plot
            fig = _plot_heatmap(
                sample_data,
                method.base_model_cfg.model_id,
                method.finetuned_model_cfg.model_id,
                figure_width=method.method_cfg.visualization.figure_width,
                figure_dpi=method.method_cfg.visualization.figure_dpi,
                font_sizes=method.method_cfg.visualization.font_sizes
            )
            # Use high-quality rendering with exact DPI
            st.pyplot(fig, use_container_width=False, clear_figure=True, dpi=method.method_cfg.visualization.figure_dpi)


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
    figure_dpi: int = 150,
    font_sizes: Dict[str, int] = None
) -> plt.Figure:
    """
    Create a heatmap-style visualization of logit differences.

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
        font_sizes: Dict with font size settings (optional, uses defaults if None)

    Returns:
        matplotlib Figure
    """
    # Use default font sizes if not provided
    if font_sizes is None:
        font_sizes = {
            'heatmap_labels': 8,
            'heatmap_cells': 5,
            'heatmap_positions': 7,
            'main_title': 14
        }
    
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
                   height: float = None, width: float = None, fontsize: int = None, 
                   show_value: bool = False, rotation: int = 0):
        """Render a colored cell with text"""
        if height is None:
            height = cell_height
        if width is None:
            width = cell_width
        if fontsize is None:
            fontsize = font_sizes['heatmap_cells']
            
        # Draw rectangle
        rect = mpatches.Rectangle(
            (x, y), width, height,
            facecolor=color, edgecolor='white', linewidth=0.5
        )
        ax.add_patch(rect)
        
        # Add text (handle special characters)
        # For value cells, just show the token, not the number
        if show_value and '\n' in text:
            # Extract just the token part (before the newline)
            text_display = text.split('\n')[0]
        else:
            text_display = text
            
        text_display = text_display.replace('\t', '\\t')
        # Escape dollar signs to prevent LaTeX interpretation
        text_display = text_display.replace('$', r'\$')
        
        # Truncate if too long (adjust based on rotation)
        max_chars = 15 if rotation == 90 else 10
        if len(text_display) > max_chars:
            text_display = text_display[:max_chars-2] + '..'
        
        ax.text(
            x + width/2, y + height/2, text_display,
            ha='center', va='center', fontsize=fontsize,
            color='white' if sum(color[:3])/3 < 0.5 else 'black',
            rotation=rotation,
            fontproperties=matplotlib.font_manager.FontProperties(family=UNICODE_FONTS)
        )
    
    # =========================================================================
    # GLOBAL NORMALIZATION: Collect all values first for cross-section comparison
    # =========================================================================
    
    # Collect ALL logit values (Model 1 + Model 2) for unified logit scale
    all_logits = []
    for pos_data in sample_data:
        all_logits.extend([item['logit'] for item in pos_data['model1_top_k'][:k]])
        all_logits.extend([item['logit'] for item in pos_data['model2_top_k'][:k]])
    
    norm_all_logits = normalize_values(all_logits) if all_logits else np.array([])
    
    # Collect ALL diff values (positive + negative) for unified diff scale
    # Use signed values so we can see relative magnitudes
    all_diffs = []
    for pos_data in sample_data:
        all_diffs.extend([d['diff'] for d in pos_data['diff_top_k_positive'][:k_half]])
        all_diffs.extend([d['diff'] for d in pos_data['diff_top_k_negative'][:k_half]])
    
    norm_all_diffs = normalize_values([abs(d) for d in all_diffs]) if all_diffs else np.array([])
    
    # =========================================================================
    # 4. DIFF TOP-K (bottom section)
    # =========================================================================
    
    # NEGATIVE DIFFS FIRST (bottom) - Red section for M1>M2
    # Negative diffs label
    ax.text(-0.3, current_row + (k_half * cell_height / 2), 'Diff -\n(M1>M2)',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold', color='darkred')
    
    diff_idx = 0
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        # SHIFT: Position 0 predictions go in column 1, position 1 predictions go in column 2, etc.
        x_pos = (pos_idx_local + 1) * cell_width  # +1 to shift right
        
        # Skip if this would go beyond our grid
        if pos_idx_local + 1 >= num_positions:
            continue
        
        # Negative diffs (bottom k/2) - use global diff normalization
        # Reverse order: most negative diff at bottom
        for i, diff_item in enumerate(pos_data['diff_top_k_negative'][:k_half]):
            # Calculate index: negative diffs come after positive diffs in the combined array
            neg_diff_idx = diff_idx + (len(sample_data) * k_half)
            if neg_diff_idx < len(norm_all_diffs):
                color = cmap(norm_all_diffs[neg_diff_idx])
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            
            # Flip row order: i=0 (most negative) goes to bottom
            row_position = current_row + (k_half - 1 - i) * cell_height
            
            render_cell(
                x_pos, row_position,
                f"'{diff_item['token']}'\n{diff_item['diff']:.2f}",
                color, show_value=True, rotation=90
            )
        
        diff_idx += k_half
    
    current_row += k_half * cell_height
    
    # Draw divider line between negative and positive diffs
    ax.axhline(y=current_row, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
    
    # POSITIVE DIFFS SECOND (top) - Green section for M2>M1
    # Positive diffs label
    ax.text(-0.3, current_row + (k_half * cell_height / 2), 'Diff +\n(M2>M1)',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold', color='darkgreen')
    
    diff_idx = 0
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        # SHIFT: Position 0 predictions go in column 1, position 1 predictions go in column 2, etc.
        x_pos = (pos_idx_local + 1) * cell_width  # +1 to shift right
        
        # Skip if this would go beyond our grid
        if pos_idx_local + 1 >= num_positions:
            continue
        
        # Positive diffs (top k/2) - use global diff normalization
        # Reverse order: highest diff at top
        for i, diff_item in enumerate(pos_data['diff_top_k_positive'][:k_half]):
            if diff_idx < len(norm_all_diffs):
                color = cmap(norm_all_diffs[diff_idx])
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            
            # Flip row order: i=0 (highest) goes to top
            row_position = current_row + (k_half - 1 - i) * cell_height
            
            render_cell(
                x_pos, row_position,
                f"'{diff_item['token']}'\n{diff_item['diff']:.2f}",
                color, show_value=True, rotation=90
            )
            
            diff_idx += 1
    
    current_row += k_half * cell_height
    
    # Add small gap before next section
    current_row += section_gap
    
    # Draw separator line
    ax.axhline(y=current_row, color='black', linewidth=2)
    
    # =========================================================================
    # 3. MODEL 2 TOP-K
    # =========================================================================
    
    # Add section label
    label2 = model2_name.split('/')[-1]  # Use last part of path
    if len(label2) > 15:
        label2 = label2[:12] + '...'
    ax.text(-0.3, current_row + (k * cell_height / 2), f'Model 2\n{label2}',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    # Track position in global logit normalization array
    # Model 2 logits come after Model 1 logits in the combined array
    num_model1_logits = len([item for pos_data in sample_data for item in pos_data['model1_top_k'][:k]])
    logit_idx = num_model1_logits
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        # SHIFT: Position 0 predictions go in column 1, position 1 predictions go in column 2, etc.
        x_pos = (pos_idx_local + 1) * cell_width  # +1 to shift right
        
        # Skip if this would go beyond our grid
        if pos_idx_local + 1 >= num_positions:
            continue
            
        # Use global logit normalization
        # Reverse order: highest logit at top (furthest from bottom)
        for i, logit_item in enumerate(pos_data['model2_top_k'][:k]):  # Only take first k items
            if logit_idx < len(norm_all_logits):
                color = cmap(norm_all_logits[logit_idx])
                logit_idx += 1
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            
            # Flip row order: i=0 (highest) goes to top
            row_position = current_row + (k - 1 - i) * cell_height
            
            render_cell(
                x_pos, row_position,
                f"'{logit_item['token']}'\n{logit_item['logit']:.1f}",
                color, show_value=True, rotation=90
            )
    
    current_row += k * cell_height
    
    # Add small gap before next section
    current_row += section_gap
    
    # Draw separator line
    ax.axhline(y=current_row, color='black', linewidth=2)
    
    # =========================================================================
    # 2. MODEL 1 TOP-K
    # =========================================================================
    
    # Add section label
    label1 = model1_name.split('/')[-1]  # Use last part of path
    if len(label1) > 15:
        label1 = label1[:12] + '...'
    ax.text(-0.3, current_row + (k * cell_height / 2), f'Model 1\n{label1}',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    # Start from beginning of global logit normalization array (Model 1 comes first)
    logit_idx = 0
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        # SHIFT: Position 0 predictions go in column 1, position 1 predictions go in column 2, etc.
        x_pos = (pos_idx_local + 1) * cell_width  # +1 to shift right
        
        # Skip if this would go beyond our grid
        if pos_idx_local + 1 >= num_positions:
            continue
            
        # Use global logit normalization
        # Reverse order: highest logit at top (furthest from bottom)
        for i, logit_item in enumerate(pos_data['model1_top_k'][:k]):  # Only take first k items
            if logit_idx < len(norm_all_logits):
                color = cmap(norm_all_logits[logit_idx])
                logit_idx += 1
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            
            # Flip row order: i=0 (highest) goes to top
            row_position = current_row + (k - 1 - i) * cell_height
            
            render_cell(
                x_pos, row_position,
                f"'{logit_item['token']}'\n{logit_item['logit']:.1f}",
                color, show_value=True, rotation=90
            )
    
    current_row += k * cell_height
    
    # Add small gap before next section
    current_row += section_gap
    
    # Draw separator line
    ax.axhline(y=current_row, color='black', linewidth=2)
    
    # =========================================================================
    # 1. REFERENCE TOKENS (top section with 2 rows)
    # =========================================================================
    
    # Row 1: Position indices
    ax.text(-0.3, current_row + (position_row_height / 2), 'Position',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        x_pos = pos_idx_local * cell_width
        
        # Position index in small cell
        render_cell(
            x_pos, current_row,
            f"{pos_idx_local}",
            (0.95, 0.95, 0.95, 1.0), 
            height=position_row_height, 
            fontsize=font_sizes['heatmap_positions'],
            show_value=False
        )
    
    current_row += position_row_height
    
    # Row 2: Reference tokens (rotated 90 degrees)
    ax.text(-0.3, current_row + (reference_row_height / 2), 'Reference\nTokens',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        actual_token = pos_data['actual_token']
        x_pos = pos_idx_local * cell_width
        
        # Light gray background for reference tokens with rotated text
        render_cell(
            x_pos, current_row,
            actual_token,
            (0.9, 0.9, 0.9, 1.0), 
            height=reference_row_height, 
            fontsize=font_sizes['heatmap_positions'],
            show_value=False,
            rotation=90  # Rotate text 90 degrees
        )
    
    current_row += reference_row_height
    
    # Draw separator line
    ax.axhline(y=current_row, color='black', linewidth=2)
    
    # Add title
    fig.suptitle(
        f'Logit Diff - Sample\n'
        f'{num_positions} token positions, Top-{k} predictions per model',
        fontsize=font_sizes['main_title'], weight='bold', y=0.99
    )

    return fig

