"""
Streamlit UI for Diff Mining.

Schema-driven UI that supports both old (analysis_*) and new (run_*) output formats.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import json
import re
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager
import numpy as np
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from .normalization import process_token_list
from .plots import plot_occurrence_bar_chart, get_global_token_scatter_plotly, UNICODE_FONTS
from .token_ordering import (
    read_ordering_type_metadata,
    read_dataset_orderings_index,
    read_ordering,
    read_ordering_eval,
)

matplotlib.rcParams['text.antialiased'] = True
matplotlib.rcParams['figure.autolayout'] = False


def visualize(method):
    """Main visualization entry point."""
    st.title("Diff Mining")
    
    # Determine available runs/analysis folders
    run_dirs = _list_run_dirs(method)
    analysis_dirs = _list_analysis_dirs(method)
    
    if not run_dirs and not analysis_dirs:
        st.info("No results found. Please run the analysis first.")
        return

    browse_tab, compare_tab = st.tabs(["Browse results", "Compare runs"])

    with browse_tab:
        # Combine and sort by modification time
        all_dirs: List[Tuple[str, Path]] = []
        for d in run_dirs:
            all_dirs.append(("run", d))
        for d in analysis_dirs:
            all_dirs.append(("analysis", d))
        all_dirs.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)

        # Run/Analysis selector
        dir_options = {_format_run_label(d, dtype): d for dtype, d in all_dirs}
        selected_label = st.selectbox(
            "Select Run",
            list(dir_options.keys()),
            key="run_select",
        )
        selected_dir = dir_options[selected_label]
        method.analysis_dir = selected_dir

        # Detect schema type
        is_new_schema = _is_new_schema(selected_dir)

        if is_new_schema:
            _render_new_schema_ui(method, selected_dir)
        else:
            _render_legacy_ui(method, selected_dir)

    with compare_tab:
        _render_cross_run_comparison(method)


def _list_method_variant_dirs(method) -> List[Path]:
    """
    List all `diff_mining*` method directories for this model/organism.

    The method instance's `base_results_dir` points to a single hyperparameter-specific
    directory (e.g. a particular logit extraction method/layer). For dashboard
    visualization, we want to discover runs across *all* such variant directories.
    """
    method_dir = method.base_results_dir
    organism_dir = method_dir.parent
    if not organism_dir.exists():
        return []
    return sorted(
        [
            d
            for d in organism_dir.iterdir()
            if d.is_dir() and d.name.startswith("diff_mining")
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )


def _run_dir_has_new_schema_outputs(run_dir: Path) -> bool:
    """True if this run directory contains any `orderings.json` in new schema layout."""
    return bool(list(run_dir.glob("*/*/orderings.json")))


def _list_run_dirs(method) -> List[Path]:
    """List run directories (new schema)."""
    run_dirs: List[Path] = []
    for method_dir in _list_method_variant_dirs(method):
        for d in method_dir.iterdir():
            if not (d.is_dir() and d.name.startswith("run_")):
                continue
            if _run_dir_has_new_schema_outputs(d):
                run_dirs.append(d)
    return sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def _list_analysis_dirs(method) -> List[Path]:
    """List analysis directories (legacy schema)."""
    analysis_dirs: List[Path] = []
    for method_dir in _list_method_variant_dirs(method):
        for d in method_dir.iterdir():
            if not (d.is_dir() and d.name.startswith("analysis_")):
                continue
            if list(d.glob("*_occurrence_rates.json")):
                analysis_dirs.append(d)
    return sorted(analysis_dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def _format_run_label(d: Path, dtype: str) -> str:
    """Format run/analysis directory as a label."""
    # Try to read run_metadata.json for better labels
    metadata_path = d / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        timestamp = meta.get("timestamp", "")[:19].replace("T", " ")
        samples = meta.get("max_samples", "?")
        tokens = meta.get("max_tokens_per_sample", "?")
        top_k = meta.get("top_k", "?")
        extraction = str(meta.get("logit_extraction_method", ""))
        layer = meta.get("logit_lens_layer", None)
        if extraction == "logit_lens" and layer is not None:
            extraction_str = f"{extraction}@{layer}"
        elif extraction == "patchscope_lens":
            patchscope_layer = meta.get("patchscope_lens_layer", None)
            if patchscope_layer is None:
                patchscope_layer = meta.get("patchscope_lens_layer_relative", None)
            if patchscope_layer is None:
                method_dir_name = d.parent.name
                m = re.search(
                    r"_logit_extraction_patchscope_lens_layer_(?P<layer>[0-9]+(?:p[0-9]+)?)",
                    method_dir_name,
                )
                if m:
                    patchscope_layer = m.group("layer").replace("p", ".")
            if patchscope_layer is not None:
                extraction_str = f"{extraction}@{patchscope_layer}"
            else:
                extraction_str = extraction
        elif extraction:
            extraction_str = extraction
        else:
            extraction_str = "?"
        return f"{timestamp} | {samples}s/{tokens}t/top{top_k} | {extraction_str}"
    return d.name


def _is_new_schema(run_dir: Path) -> bool:
    """Check if a directory uses the new schema (has ordering type subdirs)."""
    # New schema has ordering type directories like topk_occurring/, fraction_positive_diff/
    for subdir in run_dir.iterdir():
        if subdir.is_dir() and (subdir / "metadata.json").exists():
            return True
    return False


# ============================================================================
# New Schema UI
# ============================================================================

def _render_new_schema_ui(method, run_dir: Path) -> None:
    """Render UI for new schema with ordering type selection."""
    
    # Find available ordering types
    ordering_types = _find_ordering_types(run_dir)
    if not ordering_types:
        st.warning("No ordering types found in this run.")
        return
    
    # Ordering type selector
    ordering_options = {ot["display_name"]: ot for ot in ordering_types}
    selected_ordering_name = st.selectbox(
        "Token Ordering Type",
        list(ordering_options.keys()),
        key="ordering_type_select"
    )
    selected_ordering = ordering_options[selected_ordering_name]
    ordering_type_dir = run_dir / selected_ordering["id"]
    
    # Find datasets
    datasets = _find_datasets_in_ordering(ordering_type_dir)
    if not datasets:
        st.warning(f"No datasets found for ordering type: {selected_ordering_name}")
        return

    stats_tab, data_tab = st.tabs(["Statistics", "Data"])
    with stats_tab:
        _render_ordering_stats(ordering_type_dir, selected_ordering)
    with data_tab:
        _render_ordering_data(method, ordering_type_dir, selected_ordering, datasets)


def _find_ordering_types(run_dir: Path) -> List[Dict[str, Any]]:
    """Find all ordering types in a run directory."""
    ordering_types = []
    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue
        metadata = read_ordering_type_metadata(subdir)
        if metadata:
            ordering_types.append({
                "id": subdir.name,
                "display_name": metadata.get("display_name", subdir.name),
                "x_axis_label": metadata.get("x_axis_label", "Ordering Value"),
                "y_axis_label": metadata.get("y_axis_label", "Avg Logit Diff"),
                "metadata": metadata,
            })
    return ordering_types


def _find_datasets_in_ordering(ordering_type_dir: Path) -> List[str]:
    """Find all datasets in an ordering type directory."""
    datasets = []
    for subdir in ordering_type_dir.iterdir():
        if subdir.is_dir() and (subdir / "orderings.json").exists():
            datasets.append(subdir.name)
    return sorted(datasets)


def _render_ordering_stats(ordering_type_dir: Path, ordering_info: Dict[str, Any]) -> None:
    """Render statistics tab for an ordering type."""
    metadata = ordering_info["metadata"]
    
    st.markdown(f"### {ordering_info['display_name']}")
    st.markdown(f"**X-axis**: {ordering_info['x_axis_label']}")
    st.markdown(f"**Y-axis**: {ordering_info['y_axis_label']}")

    eval_rows: List[Dict[str, Any]] = []
    for dataset_dir in ordering_type_dir.iterdir():
        if not (dataset_dir.is_dir() and (dataset_dir / "orderings.json").exists()):
            continue
        index = read_dataset_orderings_index(dataset_dir)
        if not index:
            continue
        for ordering in index.get("orderings", []):
            ordering_id = ordering["ordering_id"]
            eval_data = read_ordering_eval(dataset_dir, ordering_id)
            if not eval_data:
                continue
            labels = eval_data.get("labels", [])
            if not labels:
                continue
            n_total = len(labels)
            n_relevant = sum(lbl == "RELEVANT" for lbl in labels)
            eval_rows.append(
                {
                    "dataset": dataset_dir.name,
                    "ordering_id": ordering_id,
                    "display_label": ordering.get("display_label", ordering_id),
                    "n_relevant": n_relevant,
                    "n_total": n_total,
                }
            )

    if eval_rows:
        df_eval = pd.DataFrame(eval_rows)
        total_relevant = int(df_eval["n_relevant"].sum())
        total_tokens = int(df_eval["n_total"].sum())
        pct_relevant = 100.0 * total_relevant / float(total_tokens)

        st.markdown("### Token relevance (grader)")
        st.metric("% relevant", f"{pct_relevant:.1f}%", f"{total_relevant}/{total_tokens} tokens")

        if ordering_info["id"] == "nmf":
            by_topic = (
                df_eval.groupby(["ordering_id", "display_label"], as_index=False)[["n_relevant", "n_total"]]
                .sum()
                .sort_values("ordering_id")
            )
            by_topic["% relevant"] = 100.0 * by_topic["n_relevant"] / by_topic["n_total"].astype(float)
            display_df = by_topic[["display_label", "% relevant", "n_relevant", "n_total"]].copy()
            display_df.columns = ["Topic", "% relevant", "relevant", "total"]
            st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.markdown("### Token relevance (grader)")
        st.info("No token relevance grading found for this ordering type.")
    
    if ordering_info["id"] == "nmf" and "topic_metrics" in metadata:
        topic_metrics = metadata.get("topic_metrics", [])
        if isinstance(topic_metrics, list) and topic_metrics:
            with st.expander("Topic Metrics", expanded=False):
                df = pd.DataFrame(topic_metrics)
                if "nmf_topic_idx" in df.columns:
                    df = df.sort_values("nmf_topic_idx", ascending=True)
                if "nmf_topic_prevalence" in df.columns:
                    df["nmf_topic_prevalence_pct"] = 100.0 * df["nmf_topic_prevalence"].astype(float)
                cols = [
                    c
                    for c in [
                        "nmf_topic_idx",
                        "nmf_topic_mass",
                        "nmf_topic_prevalence_pct",
                        "nmf_topic_concentration",
                    ]
                    if c in df.columns
                ]
                display_df = df[cols].copy()
                rename = {
                    "nmf_topic_idx": "Topic",
                    "nmf_topic_mass": "Mass",
                    "nmf_topic_prevalence_pct": "Prevalence (%)",
                    "nmf_topic_concentration": "Concentration",
                }
                display_df = display_df.rename(columns=rename)
                st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Show type-specific metadata
    if "pairwise" in metadata:
        with st.expander("Pairwise Metrics", expanded=False):
            pairwise = metadata["pairwise"]
            if "cosine_similarity" in pairwise:
                st.markdown("**Cosine Similarity Matrix**")
                st.dataframe(np.array(pairwise["cosine_similarity"]))
    
    # Show any other metadata
    other_keys = [
        k
        for k in metadata.keys()
        if k not in ["ordering_type_id", "display_name", "x_axis_label", "y_axis_label", "pairwise", "topic_metrics"]
    ]
    if other_keys:
        with st.expander("Additional Metadata", expanded=False):
            for k in other_keys:
                st.markdown(f"**{k}**: {metadata[k]}")


def _render_ordering_data(
    method,
    ordering_type_dir: Path,
    ordering_info: Dict[str, Any],
    datasets: List[str]
) -> None:
    """Render data tab with multiselect orderings and plots."""
    
    # Dataset selector
    selected_dataset = st.selectbox(
        "Dataset",
        datasets,
        key=f"dataset_select_{ordering_info['id']}"
    )
    
    dataset_dir = ordering_type_dir / selected_dataset
    index = read_dataset_orderings_index(dataset_dir)
    if not index:
        st.error(f"Could not load orderings index for {selected_dataset}")
        return
    
    orderings_list = index.get("orderings", [])
    if not orderings_list:
        st.warning("No orderings found for this dataset.")
        return
    
    # Multiselect for orderings
    ordering_options = {o["display_label"]: o["ordering_id"] for o in orderings_list}
    
    # Default to first ordering
    default_selection = [orderings_list[0]["display_label"]] if orderings_list else []
    selected_labels = st.multiselect(
        "Select Orderings to Display",
        list(ordering_options.keys()),
        default=default_selection,
        key=f"ordering_multiselect_{ordering_info['id']}_{selected_dataset}"
    )
    
    if not selected_labels:
        st.info("Select one or more orderings to display.")
        return
    
    # Render selected orderings in a grid
    num_selected = len(selected_labels)
    cols_per_row = min(num_selected, 3)  # Max 3 per row
    
    for i in range(0, num_selected, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= num_selected:
                break
            label = selected_labels[i + j]
            ordering_id = ordering_options[label]
            with col:
                _render_single_ordering(
                    method, dataset_dir, ordering_id, label,
                    ordering_info["x_axis_label"],
                    ordering_info["y_axis_label"]
                )


def _render_single_ordering(
    method,
    dataset_dir: Path,
    ordering_id: str,
    display_label: str,
    x_label: str,
    y_label: str
) -> None:
    """Render a single ordering panel with plot and table."""
    st.markdown(f"**{display_label}**")
    
    ordering = read_ordering(dataset_dir, ordering_id)
    if not ordering:
        st.error(f"Could not load ordering: {ordering_id}")
        return

    if (
        "nmf_topic_mass" in ordering
        and "nmf_topic_prevalence" in ordering
        and "nmf_topic_concentration" in ordering
    ):
        cols = st.columns(3)
        cols[0].metric("Topic mass", f"{float(ordering['nmf_topic_mass']):.4g}")
        cols[1].metric("Prevalence", f"{100.0 * float(ordering['nmf_topic_prevalence']):.2f}%")
        cols[2].metric("Concentration", f"{float(ordering['nmf_topic_concentration']):.3f}")
    
    tokens = ordering.get("tokens", [])
    if not tokens:
        st.warning("No tokens in this ordering.")
        return
    
    # Load eval data if available
    eval_data = read_ordering_eval(dataset_dir, ordering_id)
    labels = eval_data.get("labels", []) if eval_data else []
    translations = eval_data.get("translations", None) if eval_data else None
    if translations is not None:
        assert isinstance(translations, list)
    
    df_all_rows = []
    for i, t in enumerate(tokens):
        row = {
            "Rank": i + 1,
            "Token": t["token_str"],
            "X": t["ordering_value"],
            "Y": t["avg_logit_diff"],
            "Relevance": labels[i] if i < len(labels) else "",
        }
        if translations is not None:
            row["Translation"] = translations[i] if i < len(translations) else ""
        df_all_rows.append(row)
    df_all = pd.DataFrame(df_all_rows)
    df_plot = df_all.head(50)
    
    # Scatter plot
    hover_data = ["Rank", "Token", "Relevance"]
    if translations is not None:
        hover_data.insert(2, "Translation")
    fig = px.scatter(
        df_plot,
        x="X",
        y="Y",
        hover_data=hover_data,
        color="Relevance" if labels else None,
        color_discrete_map={"RELEVANT": "green", "IRRELEVANT": "gray", "": "blue"},
        title=f"{display_label}",
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    with st.expander("Token Table", expanded=False):
        display_cols = ["Rank", "Token"]
        if translations is not None:
            display_cols.append("Translation")
        display_cols.extend(["X", "Y", "Relevance"])
        display_df = df_all[display_cols].copy()
        if translations is not None:
            display_df.columns = ["Rank", "Token", "Translation", x_label, y_label, "Relevance"]
        else:
            display_df.columns = ["Rank", "Token", x_label, y_label, "Relevance"]
        def _highlight_relevant_row(row: pd.Series) -> List[str]:
            if row["Relevance"] == "RELEVANT":
                return ["background-color: rgba(0, 200, 0, 0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(_highlight_relevant_row, axis=1),
            hide_index=True,
            use_container_width=True,
        )


def _compute_relevance_counts_from_eval(
    dataset_dir: Path,
    ordering_id: str,
    k_top: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Return (n_relevant, n_total) from {ordering_id}_eval.json if present."""
    eval_data = read_ordering_eval(dataset_dir, ordering_id)
    if not eval_data:
        return None
    labels = eval_data.get("labels", [])
    if not labels:
        return None
    if k_top is not None:
        k = int(k_top)
        assert k >= 1
        labels = labels[: min(k, len(labels))]
    n_total = int(len(labels))
    n_relevant = int(sum(lbl == "RELEVANT" for lbl in labels))
    return n_relevant, n_total


def _axis_label_for_run_dir(run_dir: Path) -> str:
    meta_path = run_dir / "run_metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    extraction = str(meta.get("logit_extraction_method", "")) if meta else ""
    layer = meta.get("logit_lens_layer", None) if meta else None
    if extraction == "logit_lens" and layer is not None:
        extraction_str = f"{extraction}@{layer}"
    elif extraction == "patchscope_lens":
        patchscope_layer = None
        if meta:
            patchscope_layer = meta.get("patchscope_lens_layer", None)
            if patchscope_layer is None:
                patchscope_layer = meta.get("patchscope_lens_layer_relative", None)

        if patchscope_layer is None:
            # Existing runs do not store this in run_metadata.json; parse from method directory name.
            method_dir_name = run_dir.parent.name
            m = re.search(
                r"_logit_extraction_patchscope_lens_layer_(?P<layer>[0-9]+(?:p[0-9]+)?)",
                method_dir_name,
            )
            if m:
                patchscope_layer = m.group("layer").replace("p", ".")

        if patchscope_layer is not None:
            extraction_str = f"{extraction}@{patchscope_layer}"
        else:
            extraction_str = extraction
    elif extraction:
        extraction_str = extraction
    else:
        extraction_str = "?"

    return extraction_str


def _dedupe_labels(labels: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for lbl in labels:
        n = seen.get(lbl, 0)
        if n == 0:
            out.append(lbl)
        else:
            out.append(f"{lbl} ({n + 1})")
        seen[lbl] = n + 1
    return out


def _compute_ordering_relevance_row(
    *,
    run_label: str,
    dataset_dir: Path,
    ordering_type_id: str,
    aggregation_mode: Optional[str],
    k_top: int,
    adjust_for_multitopic: bool,
) -> Optional[Dict[str, Any]]:
    index = read_dataset_orderings_index(dataset_dir)
    if not index:
        return None

    orderings_list = index.get("orderings", [])
    if not orderings_list:
        return None

    if ordering_type_id != "nmf":
        assert len(orderings_list) == 1, (
            f"Expected exactly one ordering for ordering_type_id={ordering_type_id}, "
            f"but found {len(orderings_list)} in {dataset_dir}"
        )
        ordering_id = str(orderings_list[0]["ordering_id"])
        counts = _compute_relevance_counts_from_eval(dataset_dir, ordering_id, k_top=k_top)
        if counts is None:
            return None
        n_relevant, n_total = counts
        pct = 100.0 * float(n_relevant) / float(n_total)
        return {
            "run": run_label,
            "% relevant": pct,
            "relevant": n_relevant,
            "total": n_total,
        }

    assert aggregation_mode is not None
    k_per_topic = int(k_top)
    if adjust_for_multitopic:
        num_topics = int(len(orderings_list))
        assert num_topics >= 1
        k_per_topic = max(1, int(k_top) // num_topics)

    per_topic: List[Dict[str, Any]] = []
    for o in orderings_list:
        ordering_id = str(o["ordering_id"])
        counts = _compute_relevance_counts_from_eval(
            dataset_dir, ordering_id, k_top=k_per_topic
        )
        if counts is None:
            continue
        n_relevant, n_total = counts
        pct = 100.0 * float(n_relevant) / float(n_total)
        per_topic.append(
            {
                "ordering_id": ordering_id,
                "display_label": str(o.get("display_label", ordering_id)),
                "n_relevant": n_relevant,
                "n_total": n_total,
                "pct": pct,
            }
        )
    if not per_topic:
        return None

    if aggregation_mode == "Best-topic":
        best = max(per_topic, key=lambda d: (d["pct"], d["ordering_id"]))
        return {
            "run": run_label,
            "% relevant": float(best["pct"]),
            "relevant": int(best["n_relevant"]),
            "total": int(best["n_total"]),
            "nmf_topic": str(best["display_label"]),
        }
    if aggregation_mode == "Concatenated":
        n_relevant = int(sum(d["n_relevant"] for d in per_topic))
        n_total = int(sum(d["n_total"] for d in per_topic))
        pct = 100.0 * float(n_relevant) / float(n_total)
        return {
            "run": run_label,
            "% relevant": pct,
            "relevant": n_relevant,
            "total": n_total,
            "nmf_topic": "All topics (concatenated)",
        }
    raise AssertionError(f"Unexpected NMF aggregation: {aggregation_mode}")


def _render_cross_run_comparison(method) -> None:
    """
    Compare token relevance fractions across multiple runs (independent of the currently-selected run).

    Uses per-ordering `*_eval.json` files produced by the grading pipeline.
    """
    st.markdown("### Run comparison")

    run_dirs = _list_run_dirs(method)
    if not run_dirs:
        st.info("No new-schema runs found to compare.")
        return
    
    organism_variant = method.base_results_dir.parent.name

    ordering_id_to_info: Dict[str, Dict[str, Any]] = {}
    for run_dir in run_dirs:
        for ot in _find_ordering_types(run_dir):
            ordering_id_to_info.setdefault(str(ot["id"]), ot)

    if not ordering_id_to_info:
        st.info("No ordering types found across runs.")
        return

    ordering_ids_sorted = sorted(ordering_id_to_info.keys())
    default_ordering_ids = [
        oid for oid in ["topk_occurring", "nmf"] if oid in ordering_id_to_info
    ]
    if not default_ordering_ids:
        default_ordering_ids = ordering_ids_sorted[:1]

    selected_ordering_ids = st.multiselect(
        "Token Ordering Types",
        ordering_ids_sorted,
        default=default_ordering_ids,
        key="diff_mining::cross_run_comparison::ordering_types",
    )
    if not selected_ordering_ids:
        st.info("Select one or more ordering types to compare.")
        return

    # Dataset options are the union across runs for the selected ordering types.
    dataset_set = set()
    for run_dir in run_dirs:
        for ordering_type_id in selected_ordering_ids:
            ordering_dir = run_dir / ordering_type_id
            if not ordering_dir.exists():
                continue
            for ds in _find_datasets_in_ordering(ordering_dir):
                dataset_set.add(ds)
    dataset_options = sorted(dataset_set)

    if not dataset_options:
        st.info("No datasets found for these ordering types across runs.")
        return

    dataset_name = st.selectbox(
        "Dataset",
        dataset_options,
        key="diff_mining::cross_run_comparison::dataset",
    )

    top_k_tokens = int(
        st.number_input(
            "Top-K tokens to consider",
            min_value=1,
            max_value=500,
            value=100,
            step=1,
            help="For each ordering, compute % relevant using only the first K graded tokens.",
            key="diff_mining::cross_run_comparison::k_top",
        )
    )
    adjust_for_multitopic = bool(
        st.checkbox(
            "Adjust for multitopic (NMF)",
            value=False,
            help="For NMF topics, use K/num_topics tokens per topic before selecting best-topic.",
            key="diff_mining::cross_run_comparison::adjust_multitopic",
        )
    )

    def _run_label(run_dir: Path) -> str:
        base = _format_run_label(run_dir, "run")
        variant = run_dir.parent.name
        return f"{base} | {variant}/{run_dir.name}"

    run_label_to_dir = {_run_label(d): d for d in run_dirs}
    default_labels = list(run_label_to_dir.keys())[:2]
    selected_run_labels = st.multiselect(
        "Select runs",
        list(run_label_to_dir.keys()),
        default=default_labels,
        key=f"diff_mining::cross_run_comparison::runs::{dataset_name}",
    )

    if not selected_run_labels:
        st.info("Select one or more runs to compare.")
        return

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for run_label in selected_run_labels:
        run_dir = run_label_to_dir[run_label]
        for ordering_type_id in selected_ordering_ids:
            dataset_dir = run_dir / ordering_type_id / dataset_name
            nmf_aggregation = "Best-topic" if ordering_type_id == "nmf" else None
            row = _compute_ordering_relevance_row(
                run_label=run_label,
                dataset_dir=dataset_dir,
                ordering_type_id=ordering_type_id,
                aggregation_mode=nmf_aggregation,
                k_top=top_k_tokens,
                adjust_for_multitopic=adjust_for_multitopic,
            )
            if row is None:
                skipped.append(f"{run_label} | {ordering_type_id}")
                continue
            display_name = str(ordering_id_to_info[ordering_type_id]["display_name"])
            if ordering_type_id == "nmf":
                display_name = "NMF (best-topic)"
            row["ordering_type"] = display_name
            rows.append(row)

    if skipped:
        st.warning(
            "Skipping runs missing grading for this dataset/ordering type:\n"
            + "\n".join(f"- {s}" for s in skipped)
        )

    if not rows:
        st.info("No comparable runs found (missing grading outputs).")
        return

    df = pd.DataFrame(rows).copy()

    run_scores = df.groupby("run", as_index=False)["% relevant"].max().sort_values(
        "% relevant", ascending=False
    )
    run_order = run_scores["run"].tolist()
    axis_base_labels = [_axis_label_for_run_dir(run_label_to_dir[r]) for r in run_order]
    axis_labels = _dedupe_labels(axis_base_labels)
    short_by_run = dict(zip(run_order, axis_labels))
    df["run_short"] = df["run"].map(short_by_run)
    df = df.sort_values(["run_short", "ordering_type"], ascending=[True, True])

    hover_cols = [c for c in df.columns if c not in {"% relevant", "run_short"}]
    fig = px.bar(
        df,
        x="run_short",
        y="% relevant",
        color="ordering_type",
        barmode="group",
        text="% relevant",
        hover_data=hover_cols,
        category_orders={
            "run_short": [short_by_run[r] for r in run_order],
        },
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        title_text=str(organism_variant),
        title_x=0.5,
        yaxis_range=[0, 100],
        xaxis_title=None,
        xaxis_tickangle=-30,
        height=450,
        legend_title_text=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    display_df = df.drop(columns=["run_short"])
    st.dataframe(display_df, hide_index=True, use_container_width=True)


# ============================================================================
# Legacy Schema UI (backward compatibility)
# ============================================================================

def _render_legacy_ui(method, analysis_dir: Path) -> None:
    """Render UI for legacy analysis_* schema."""
    
    tabs: List[Tuple[str, Any]] = [
        ("📊 Token Occurrence Rankings", _render_occurrence_rankings_tab),
        ("🌍 Global Token Scatter", _render_global_scatter_tab),
        ("🖼️ All Plots", _render_all_plots_tab),
        ("🔥 Interactive Heatmap", _render_interactive_heatmap_tab),
    ]
    if _find_nmf_datasets(method):
        tabs.append(("🧪 NMF", _render_nmf_tab))

    st.subheader("Diff Mining")
    tab_titles = [t for t, _ in tabs]
    selected_title = st.radio(
        "View",
        tab_titles,
        horizontal=True,
        label_visibility="collapsed",
        key="diff_mining::active_view",
    )
    selected_render = next(fn for title, fn in tabs if title == selected_title)
    selected_render(method)


def _find_available_datasets(method) -> List[str]:
    """Find all available result files."""
    analysis_dir = method.get_or_create_results_dir()
    results_files = list(analysis_dir.glob("*_occurrence_rates.json"))
    return [f.stem.replace("_occurrence_rates", "") for f in results_files]


def _load_results(method, dataset_name: str) -> Optional[Dict]:
    """Load results for a specific dataset."""
    analysis_dir = method.get_or_create_results_dir()
    results_file = analysis_dir / f"{dataset_name}_occurrence_rates.json"
    if not results_file.exists():
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def _load_results_from_dir(results_dir: Path, dataset_name: str) -> Optional[Dict]:
    """Load results for a specific dataset from a given directory."""
    results_file = results_dir / f"{dataset_name}_occurrence_rates.json"
    if not results_file.exists():
        return None
    with open(results_file, "r") as f:
        return json.load(f)


def _find_nmf_datasets(method) -> List[str]:
    """Find datasets that have NMF outputs."""
    analysis_dir = method.get_or_create_results_dir()
    nmf_dir = analysis_dir / "nmf"
    if not nmf_dir.exists():
        return []
    nmf_files = list(nmf_dir.glob("*/nmf_topics_analysis.json"))
    return sorted({path.parent.name for path in nmf_files})


def _load_nmf_topics(method, dataset_name: str) -> Optional[Dict[str, Any]]:
    """Load NMF topics analysis for a dataset."""
    analysis_dir = method.get_or_create_results_dir()
    nmf_file = analysis_dir / "nmf" / dataset_name / "nmf_topics_analysis.json"
    if not nmf_file.exists():
        return None
    with open(nmf_file, "r") as f:
        return json.load(f)


def _load_nmf_metrics(method, dataset_name: str) -> Optional[Dict[str, Any]]:
    """Load NMF per-topic and pairwise metrics for a dataset."""
    analysis_dir = method.get_or_create_results_dir()
    metrics_file = analysis_dir / "nmf" / dataset_name / "nmf_topic_metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file, "r") as f:
        return json.load(f)


def _render_global_scatter_tab(method):
    """Tab: Interactive Global Token Scatter."""
    available_datasets = _find_available_datasets(method)
    if not available_datasets:
        st.error("No results found. Please run the analysis first.")
        return

    selected_dataset = st.selectbox("Select Dataset", available_datasets, key="scatter_dataset_select")
    
    analysis_dir = method.get_or_create_results_dir()
    json_path = analysis_dir / f"{selected_dataset}_global_token_stats.json"
    occurrence_rates_path = analysis_dir / f"{selected_dataset}_occurrence_rates.json"
    
    filter_punct = bool(method.method_cfg.filter_pure_punctuation)
    
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
        token_ids = method.tokenizer.encode(search_text, add_special_tokens=False)
        token_strings = method.tokenizer.convert_ids_to_tokens(token_ids)
        readable_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in token_strings]
        
        st.info(f"Tokenized as: {readable_tokens} (IDs: {token_ids})")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            stats = data.get("global_token_stats", [])
            total_positions = data.get("total_positions_analyzed", 1)
            if total_positions == 0: 
                total_positions = 1

        vocab_lookup = {item['token_id']: i for i, item in enumerate(stats)}
        
        highlight_x = []
        highlight_y = []
        highlight_text = []
        matched_count = 0
        
        for tid in token_ids:
            if tid in vocab_lookup:
                idx = vocab_lookup[tid]
                item = stats[idx]
                x = item["count_positive"] / total_positions
                y = item["sum_logit_diff"] / total_positions
                highlight_x.append(x)
                highlight_y.append(y)
                highlight_text.append(item.get("token", ""))
                matched_count += 1
        
        if matched_count > 0:
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
            st.warning("Tokens found in tokenizer but not present in the analysis stats.")
    
    st.plotly_chart(fig, use_container_width=True)


def _render_occurrence_rankings_tab(method):
    """Tab: Display bar chart of occurrence rates."""
    available_datasets = _find_available_datasets(method)
    if not available_datasets:
        st.error("No results found. Please run the analysis first.")
        return

    selected_dataset = st.selectbox("Select Dataset", available_datasets)

    results = _load_results(method, selected_dataset)
    if results is None:
        st.error(f"Could not load results for {selected_dataset}")
        return

    col_filter, col_normalize = st.columns(2)
    with col_filter:
        filter_punct = st.checkbox("Filter Pure Punctuation", value=True)
    with col_normalize:
        normalize = st.checkbox("Normalize Tokens", value=False)

    raw_top_positive = results['top_positive']
    raw_top_negative = results['top_negative']
    total_positions = results['total_positions']
    
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

    st.markdown("**Models:**")
    st.text(f"Base: {results['metadata']['base_model']}")
    st.text(f"Finetuned: {results['metadata']['finetuned_model']}")

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
    st.pyplot(fig, use_container_width=False, clear_figure=True, dpi=method.method_cfg.visualization.figure_dpi)


def _collect_plot_files(
    analysis_dir: Path,
    dataset_filter: str,
    extensions: Tuple[str, ...]
) -> Dict[str, List[Path]]:
    """Collect plot files from the analysis directory."""
    plot_files = [
        path
        for path in analysis_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if dataset_filter != "All":
        plot_files = [path for path in plot_files if dataset_filter in path.name]

    grouped: Dict[str, List[Path]] = {}
    for path in sorted(plot_files):
        rel_parent = str(path.parent.relative_to(analysis_dir))
        grouped.setdefault(rel_parent, []).append(path)

    return grouped


def _render_all_plots_tab(method):
    """Tab: Display all plots generated by the main run."""
    analysis_dir = method.get_or_create_results_dir()
    if not analysis_dir.exists():
        st.error("No results directory found. Please run the analysis first.")
        return

    available_datasets = _find_available_datasets(method)
    dataset_options = ["All"] + available_datasets
    selected_dataset = st.selectbox(
        "Filter by dataset",
        dataset_options,
        key="all_plots_dataset_select"
    )

    image_extensions = (".png", ".jpg", ".jpeg", ".svg", ".webp")
    grouped_images = _collect_plot_files(analysis_dir, selected_dataset, image_extensions)
    total_images = sum(len(paths) for paths in grouped_images.values())

    if total_images == 0:
        st.info("No plot images found.")
    else:
        st.markdown(f"Found {total_images} plot images in `{analysis_dir}`.")
        for group, paths in grouped_images.items():
            label = group if group != "." else "root"
            with st.expander(f"{label} ({len(paths)} files)"):
                for path in paths:
                    st.image(str(path), caption=path.name, use_container_width=True)

    show_html = st.checkbox(
        "Render interactive HTML plots",
        value=False,
        help="These can be heavy for large result sets."
    )
    if show_html:
        html_grouped = _collect_plot_files(analysis_dir, selected_dataset, (".html",))
        total_html = sum(len(paths) for paths in html_grouped.values())
        if total_html == 0:
            st.info("No HTML plots found.")
        else:
            st.markdown(f"Found {total_html} HTML plots in `{analysis_dir}`.")
            for group, paths in html_grouped.items():
                label = group if group != "." else "root"
                with st.expander(f"{label} ({len(paths)} files)"):
                    for path in paths:
                        html_content = path.read_text(encoding="utf-8")
                        st.markdown(f"**{path.name}**")
                        components.html(html_content, height=700, scrolling=True)


def _render_interactive_heatmap_tab(method):
    """Tab: Interactive heatmap for custom text."""
    st.markdown("### Interactive Logit Difference Heatmap")
    st.markdown("Enter custom text to analyze logit differences between base and finetuned models.")

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
            inputs = method.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                model_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
                with method.base_model.trace(model_inputs):
                    base_logits = method.base_model.logits.save()
                with method.finetuned_model.trace(model_inputs):
                    finetuned_logits = method.finetuned_model.logits.save()

            logits1 = base_logits[0]
            logits2 = finetuned_logits[0]

            sample_data = _prepare_heatmap_data(
                logits1,
                logits2,
                input_ids[0],
                method.tokenizer,
                method.method_cfg.visualization.top_k_plotting
            )

            fig = _plot_heatmap(
                sample_data,
                method.base_model_cfg.model_id,
                method.finetuned_model_cfg.model_id,
                figure_width=method.method_cfg.visualization.figure_width,
                figure_dpi=method.method_cfg.visualization.figure_dpi,
                font_sizes=method.method_cfg.visualization.font_sizes
            )
            st.pyplot(fig, use_container_width=False, clear_figure=True, dpi=method.method_cfg.visualization.figure_dpi)


def _prepare_heatmap_data(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    top_k_plotting: int
) -> List[Dict]:
    """Prepare data for heatmap visualization."""
    logits1 = logits1.cpu()
    logits2 = logits2.cpu()
    input_ids = input_ids.cpu()

    diff = logits2 - logits1
    k = top_k_plotting
    k_half = k // 2

    sample_data = []
    seq_len = len(input_ids)

    for pos in range(seq_len):
        actual_token = tokenizer.decode([input_ids[pos].item()])

        top1_values, top1_indices = torch.topk(logits1[pos], k=k)
        model1_top_k = [
            {"token": tokenizer.decode([idx.item()]), "logit": val.item()}
            for idx, val in zip(top1_indices, top1_values)
        ]

        top2_values, top2_indices = torch.topk(logits2[pos], k=k)
        model2_top_k = [
            {"token": tokenizer.decode([idx.item()]), "logit": val.item()}
            for idx, val in zip(top2_indices, top2_values)
        ]

        diff_pos_values, diff_pos_indices = torch.topk(diff[pos], k=k_half)
        diff_top_k_positive = [
            {"token": tokenizer.decode([idx.item()]), "diff": val.item()}
            for idx, val in zip(diff_pos_indices, diff_pos_values)
        ]

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
    """Create a heatmap-style visualization of logit differences."""
    if font_sizes is None:
        font_sizes = {
            'heatmap_labels': 8,
            'heatmap_cells': 5,
            'heatmap_positions': 7,
            'main_title': 14
        }
    
    if not sample_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data to visualize", ha='center', va='center')
        return fig

    num_positions = len(sample_data)
    k = len(sample_data[0]['model1_top_k'])
    k_half = k // 2

    cell_width = 1.0
    cell_height = 0.4
    row_height = 0.4
    position_row_height = 0.3
    reference_row_height = 0.8
    section_gap = 0.1

    total_content_height = position_row_height + reference_row_height + (3 * k * cell_height) + (3 * section_gap)
    figure_height = (total_content_height / row_height) + 0.8

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=figure_dpi)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.02)

    ax.set_xlim(-1, num_positions * cell_width + 0.5)
    max_y = position_row_height + reference_row_height + (3 * k * cell_height) + (3 * section_gap)
    ax.set_ylim(0, max_y)
    ax.axis('off')

    cmap = plt.cm.viridis

    def normalize_values(values: List[float]) -> np.ndarray:
        arr = np.array(values)
        if len(arr) == 0 or arr.max() == arr.min():
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / (arr.max() - arr.min())

    def render_cell(x: float, y: float, text: str, color: Tuple[float, float, float, float],
                   height: float = None, width: float = None, fontsize: int = None, 
                   show_value: bool = False, rotation: int = 0):
        if height is None:
            height = cell_height
        if width is None:
            width = cell_width
        if fontsize is None:
            fontsize = font_sizes['heatmap_cells']
            
        rect = mpatches.Rectangle(
            (x, y), width, height,
            facecolor=color, edgecolor='white', linewidth=0.5
        )
        ax.add_patch(rect)
        
        if show_value and '\n' in text:
            text_display = text.split('\n')[0]
        else:
            text_display = text
            
        text_display = text_display.replace('\t', '\\t')
        text_display = text_display.replace('$', r'\$')
        
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

    # Collect all values for normalization
    all_logits = []
    for pos_data in sample_data:
        all_logits.extend([item['logit'] for item in pos_data['model1_top_k'][:k]])
        all_logits.extend([item['logit'] for item in pos_data['model2_top_k'][:k]])
    
    norm_all_logits = normalize_values(all_logits) if all_logits else np.array([])
    
    all_diffs = []
    for pos_data in sample_data:
        all_diffs.extend([d['diff'] for d in pos_data['diff_top_k_positive'][:k_half]])
        all_diffs.extend([d['diff'] for d in pos_data['diff_top_k_negative'][:k_half]])
    
    norm_all_diffs = normalize_values([abs(d) for d in all_diffs]) if all_diffs else np.array([])

    current_row = 0

    # Negative diffs
    ax.text(-0.3, current_row + (k_half * cell_height / 2), 'Diff -\n(M1>M2)',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold', color='darkred')
    
    diff_idx = 0
    for pos_idx_local, pos_data in enumerate(sample_data):
        x_pos = (pos_idx_local + 1) * cell_width
        if pos_idx_local + 1 >= num_positions:
            continue
        for i, diff_item in enumerate(pos_data['diff_top_k_negative'][:k_half]):
            neg_diff_idx = diff_idx + (len(sample_data) * k_half)
            if neg_diff_idx < len(norm_all_diffs):
                color = cmap(norm_all_diffs[neg_diff_idx])
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            row_position = current_row + (k_half - 1 - i) * cell_height
            render_cell(x_pos, row_position, f"'{diff_item['token']}'\n{diff_item['diff']:.2f}",
                       color, show_value=True, rotation=90)
        diff_idx += k_half
    
    current_row += k_half * cell_height
    ax.axhline(y=current_row, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)

    # Positive diffs
    ax.text(-0.3, current_row + (k_half * cell_height / 2), 'Diff +\n(M2>M1)',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold', color='darkgreen')
    
    diff_idx = 0
    for pos_idx_local, pos_data in enumerate(sample_data):
        x_pos = (pos_idx_local + 1) * cell_width
        if pos_idx_local + 1 >= num_positions:
            continue
        for i, diff_item in enumerate(pos_data['diff_top_k_positive'][:k_half]):
            if diff_idx < len(norm_all_diffs):
                color = cmap(norm_all_diffs[diff_idx])
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            row_position = current_row + (k_half - 1 - i) * cell_height
            render_cell(x_pos, row_position, f"'{diff_item['token']}'\n{diff_item['diff']:.2f}",
                       color, show_value=True, rotation=90)
            diff_idx += 1
    
    current_row += k_half * cell_height + section_gap
    ax.axhline(y=current_row, color='black', linewidth=2)

    # Model 2
    label2 = model2_name.split('/')[-1][:15]
    ax.text(-0.3, current_row + (k * cell_height / 2), f'Model 2\n{label2}',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    num_model1_logits = len([item for pos_data in sample_data for item in pos_data['model1_top_k'][:k]])
    logit_idx = num_model1_logits
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        x_pos = (pos_idx_local + 1) * cell_width
        if pos_idx_local + 1 >= num_positions:
            continue
        for i, logit_item in enumerate(pos_data['model2_top_k'][:k]):
            if logit_idx < len(norm_all_logits):
                color = cmap(norm_all_logits[logit_idx])
                logit_idx += 1
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            row_position = current_row + (k - 1 - i) * cell_height
            render_cell(x_pos, row_position, f"'{logit_item['token']}'\n{logit_item['logit']:.1f}",
                       color, show_value=True, rotation=90)
    
    current_row += k * cell_height + section_gap
    ax.axhline(y=current_row, color='black', linewidth=2)

    # Model 1
    label1 = model1_name.split('/')[-1][:15]
    ax.text(-0.3, current_row + (k * cell_height / 2), f'Model 1\n{label1}',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    logit_idx = 0
    for pos_idx_local, pos_data in enumerate(sample_data):
        x_pos = (pos_idx_local + 1) * cell_width
        if pos_idx_local + 1 >= num_positions:
            continue
        for i, logit_item in enumerate(pos_data['model1_top_k'][:k]):
            if logit_idx < len(norm_all_logits):
                color = cmap(norm_all_logits[logit_idx])
                logit_idx += 1
            else:
                color = (0.5, 0.5, 0.5, 1.0)
            row_position = current_row + (k - 1 - i) * cell_height
            render_cell(x_pos, row_position, f"'{logit_item['token']}'\n{logit_item['logit']:.1f}",
                       color, show_value=True, rotation=90)
    
    current_row += k * cell_height + section_gap
    ax.axhline(y=current_row, color='black', linewidth=2)

    # Position row
    ax.text(-0.3, current_row + (position_row_height / 2), 'Position',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        x_pos = pos_idx_local * cell_width
        render_cell(x_pos, current_row, f"{pos_idx_local}",
                   (0.95, 0.95, 0.95, 1.0), height=position_row_height,
                   fontsize=font_sizes['heatmap_positions'], show_value=False)
    
    current_row += position_row_height

    # Reference tokens
    ax.text(-0.3, current_row + (reference_row_height / 2), 'Reference\nTokens',
            ha='right', va='center', fontsize=font_sizes['heatmap_labels'], weight='bold')
    
    for pos_idx_local, pos_data in enumerate(sample_data):
        actual_token = pos_data['actual_token']
        x_pos = pos_idx_local * cell_width
        render_cell(x_pos, current_row, actual_token,
                   (0.9, 0.9, 0.9, 1.0), height=reference_row_height,
                   fontsize=font_sizes['heatmap_positions'], show_value=False, rotation=90)
    
    current_row += reference_row_height
    ax.axhline(y=current_row, color='black', linewidth=2)

    fig.suptitle(
        f'Logit Diff - Sample\n{num_positions} token positions, Top-{k} predictions per model',
        fontsize=font_sizes['main_title'], weight='bold', y=0.99
    )

    return fig


def _render_nmf_tab(method) -> None:
    """Tab: NMF (Statistics + Data)."""
    nmf_datasets = _find_nmf_datasets(method)
    if not nmf_datasets:
        st.info("No NMF outputs found for this analysis run.")
        return

    selected_dataset = st.selectbox("Select Dataset", nmf_datasets, key="nmf_dataset_select")
    nmf_topics = _load_nmf_topics(method, selected_dataset)
    if nmf_topics is None:
        st.error(f"Missing NMF topics file for {selected_dataset}")
        return

    topics = nmf_topics.get("topics", [])
    if not topics:
        st.error("No topics found in NMF output.")
        return

    topic_ids = [int(t["topic_id"]) for t in topics]
    nmf_metrics = _load_nmf_metrics(method, selected_dataset)

    stats_tab, data_tab = st.tabs(["Statistics", "Data"])

    with stats_tab:
        if nmf_metrics is None:
            st.warning("Missing `nmf_topic_metrics.json` for this dataset.")
        else:
            pairwise = nmf_metrics.get("pairwise", {})
            if pairwise:
                st.markdown("### Pairwise overlap")
                cosine = pairwise.get("cosine_similarity", None)
                if cosine is not None:
                    st.markdown("**Cosine similarity**")
                    st.dataframe(np.array(cosine), use_container_width=True)

    with data_tab:
        topic_a = st.selectbox("Topic A", topic_ids, key="nmf_topic_a_select")
        
        analysis_dir = method.get_or_create_results_dir()
        topic_dir = analysis_dir / "nmf" / selected_dataset / f"topic_{topic_a}"
        results = _load_results_from_dir(topic_dir, selected_dataset)
        
        if results is None:
            st.error(f"Missing results for topic {topic_a}")
            return
        
        fig = plot_occurrence_bar_chart(
            results["top_positive"],
            results["top_negative"],
            results["metadata"]["base_model"],
            results["metadata"]["finetuned_model"],
            results["total_positions"],
            figure_width=method.method_cfg.visualization.figure_width,
            figure_height=method.method_cfg.visualization.figure_height,
            figure_dpi=method.method_cfg.visualization.figure_dpi,
            font_sizes=method.method_cfg.visualization.font_sizes,
        )
        st.pyplot(fig, use_container_width=True, clear_figure=True, dpi=method.method_cfg.visualization.figure_dpi)
