"""
Streamlit UI for Diff Mining.

Schema-driven UI for deterministic diff_mining outputs.
"""

import streamlit as st
from pathlib import Path
import json
import re
import numpy as np
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from .token_ordering import (
    read_ordering_type_metadata,
    read_dataset_orderings_index,
    read_ordering,
    read_ordering_eval,
)


def visualize(method):
    """Main visualization entry point."""
    st.title("Diff Mining")

    # Determine available run folders
    run_dirs = _list_run_dirs(method)

    if not run_dirs:
        st.info("No results found. Please run the analysis first.")
        return

    browse_tab, compare_tab = st.tabs(["Browse results", "Compare runs"])

    with browse_tab:
        run_dirs_sorted = sorted(
            run_dirs, key=lambda p: p.stat().st_mtime, reverse=True
        )

        dir_options = {_format_run_label(d, "run"): d for d in run_dirs_sorted}
        selected_label = st.selectbox(
            "Select Run",
            list(dir_options.keys()),
            key="run_select",
        )
        selected_dir = dir_options[selected_label]
        method.analysis_dir = selected_dir

        _render_new_schema_ui(method, selected_dir)

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
    """List deterministic run directories."""
    run_dirs: List[Path] = []
    for method_dir in _list_method_variant_dirs(method):
        for d in method_dir.iterdir():
            if not d.is_dir():
                continue
            if not re.match(r"seed[0-9]+_top[0-9]+$", d.name):
                continue
            if _run_dir_has_new_schema_outputs(d):
                run_dirs.append(d)
    return sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)


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
        "Token Ordering Type", list(ordering_options.keys()), key="ordering_type_select"
    )
    selected_ordering = ordering_options[selected_ordering_name]
    ordering_type_dir = run_dir / selected_ordering["id"]

    # Find datasets
    datasets = _find_datasets_in_ordering(ordering_type_dir)
    if not datasets:
        st.warning(f"No datasets found for ordering type: {selected_ordering_name}")
        return

    stats_tab, data_tab, agent_tab = st.tabs(["Statistics", "Data", "Agent Results"])
    with stats_tab:
        _render_ordering_stats(ordering_type_dir, selected_ordering)
    with data_tab:
        _render_ordering_data(method, ordering_type_dir, selected_ordering, datasets)
    with agent_tab:
        ordering_meta = selected_ordering.get("metadata", {})
        ot_id = str(ordering_meta.get("ordering_type_id", selected_ordering["id"]))
        _render_agent_results(run_dir, ot_id)


def _find_ordering_types(run_dir: Path) -> List[Dict[str, Any]]:
    """Find all ordering types in a run directory."""
    ordering_types = []
    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue
        metadata = read_ordering_type_metadata(subdir)
        if metadata:
            ordering_types.append(
                {
                    "id": subdir.name,
                    "display_name": metadata.get("display_name", subdir.name),
                    "x_axis_label": metadata.get("x_axis_label", "Ordering Value"),
                    "y_axis_label": metadata.get("y_axis_label", "Avg Logit Diff"),
                    "metadata": metadata,
                }
            )
    return ordering_types


def _find_datasets_in_ordering(ordering_type_dir: Path) -> List[str]:
    """Find all datasets in an ordering type directory."""
    datasets = []
    for subdir in ordering_type_dir.iterdir():
        if subdir.is_dir() and (subdir / "orderings.json").exists():
            datasets.append(subdir.name)
    return sorted(datasets)


def _render_ordering_stats(
    ordering_type_dir: Path, ordering_info: Dict[str, Any]
) -> None:
    """Render statistics tab for an ordering type."""
    metadata = ordering_info["metadata"]
    ordering_type_id = str(metadata.get("ordering_type_id", ""))

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
        st.metric(
            "% relevant",
            f"{pct_relevant:.1f}%",
            f"{total_relevant}/{total_tokens} tokens",
        )

        if ordering_type_id == "nmf":
            by_topic = (
                df_eval.groupby(["ordering_id", "display_label"], as_index=False)[
                    ["n_relevant", "n_total"]
                ]
                .sum()
                .sort_values("ordering_id")
            )
            by_topic["% relevant"] = (
                100.0 * by_topic["n_relevant"] / by_topic["n_total"].astype(float)
            )
            display_df = by_topic[
                ["display_label", "% relevant", "n_relevant", "n_total"]
            ].copy()
            display_df.columns = ["Topic", "% relevant", "relevant", "total"]
            st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.markdown("### Token relevance (grader)")
        st.info("No token relevance grading found for this ordering type.")

    if ordering_type_id == "nmf" and "topic_metrics" in metadata:
        topic_metrics = metadata.get("topic_metrics", [])
        if isinstance(topic_metrics, list) and topic_metrics:
            with st.expander("Topic Metrics", expanded=False):
                df = pd.DataFrame(topic_metrics)
                if "nmf_topic_idx" in df.columns:
                    df = df.sort_values("nmf_topic_idx", ascending=True)
                if "nmf_topic_prevalence" in df.columns:
                    df["nmf_topic_prevalence_pct"] = 100.0 * df[
                        "nmf_topic_prevalence"
                    ].astype(float)
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
        if k
        not in [
            "ordering_type_id",
            "display_name",
            "x_axis_label",
            "y_axis_label",
            "pairwise",
            "topic_metrics",
        ]
    ]
    if other_keys:
        with st.expander("Additional Metadata", expanded=False):
            for k in other_keys:
                st.markdown(f"**{k}**: {metadata[k]}")


def _render_ordering_data(
    method, ordering_type_dir: Path, ordering_info: Dict[str, Any], datasets: List[str]
) -> None:
    """Render data tab with multiselect orderings and plots."""

    # Dataset selector
    selected_dataset = st.selectbox(
        "Dataset", datasets, key=f"dataset_select_{ordering_info['id']}"
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
        key=f"ordering_multiselect_{ordering_info['id']}_{selected_dataset}",
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
                    method,
                    dataset_dir,
                    ordering_id,
                    label,
                    ordering_info["x_axis_label"],
                    ordering_info["y_axis_label"],
                )


def _render_single_ordering(
    method,
    dataset_dir: Path,
    ordering_id: str,
    display_label: str,
    x_label: str,
    y_label: str,
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
        cols[1].metric(
            "Prevalence", f"{100.0 * float(ordering['nmf_topic_prevalence']):.2f}%"
        )
        cols[2].metric(
            "Concentration", f"{float(ordering['nmf_topic_concentration']):.3f}"
        )

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
            display_df.columns = [
                "Rank",
                "Token",
                "Translation",
                x_label,
                y_label,
                "Relevance",
            ]
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
        counts = _compute_relevance_counts_from_eval(
            dataset_dir, ordering_id, k_top=k_top
        )
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


def _find_agent_experiment_dirs(run_dir: Path, ordering_type_id: str) -> List[Path]:
    """Find agent experiment directories matching a specific ordering type.

    Agent experiment dirs live under ``run_dir/agent/`` and encode the ordering
    type in a ``_c{tag}-...`` config suffix at the end of the directory name.
    Only dirs with ``run{N}/`` subdirectories (new multi-run format) are returned.
    """
    agent_dir = run_dir / "agent"
    if not agent_dir.exists():
        return []

    tag = ordering_type_id.replace("top_k_occurring", "topk_occurring")
    matching: List[Path] = []
    for d in agent_dir.iterdir():
        if not d.is_dir():
            continue
        if not re.search(rf"_c{re.escape(tag)}-", d.name):
            continue
        if any((d / f"run{i}").is_dir() for i in range(10)):
            matching.append(d)
    return matching


def _read_agent_grade_scores(run_subdir: Path) -> List[int]:
    """Read individual grader scores from a run subdirectory."""
    return [int(g["score"]) for g in _read_agent_grade_data(run_subdir)]


def _read_agent_grade_data(run_subdir: Path) -> List[Dict[str, Any]]:
    """Read all hypothesis grade JSON files from a run subdirectory."""
    grade_files = sorted(run_subdir.glob("hypothesis_grade_*.json"))
    grades: List[Dict[str, Any]] = []
    for gf in grade_files:
        with open(gf, "r", encoding="utf-8") as f:
            grades.append(json.load(f))
    return grades


def _classify_message(role: str, content: str) -> str:
    """Classify a transcript message into a display category."""
    if role == "system":
        return "system"
    if role == "assistant":
        if content.startswith("CALL("):
            return "tool_call"
        if content.startswith("FINAL("):
            return "final"
        return "reasoning"
    if role == "user":
        if content.startswith("TOOL_RESULT"):
            return "tool_result"
        if content.startswith("FORMAT_ERROR"):
            return "format_error"
        if content.startswith("OVERVIEW"):
            return "overview"
        return "user"
    return role


_MSG_STYLE: Dict[str, Dict[str, str]] = {
    "system": {"icon": "gear", "label": "System"},
    "tool_call": {"icon": "arrow_right", "label": "Tool Call"},
    "tool_result": {"icon": "arrow_left", "label": "Tool Result"},
    "reasoning": {"icon": "thought_balloon", "label": "Reasoning"},
    "final": {"icon": "checkered_flag", "label": "Final Answer"},
    "format_error": {"icon": "warning", "label": "Format Error"},
    "overview": {"icon": "page_facing_up", "label": "Overview"},
    "user": {"icon": "bust_in_silhouette", "label": "User"},
}


def _render_agent_results(run_dir: Path, ordering_type_id: str) -> None:
    """Render agent descriptions, grader reasonings, and transcripts."""
    experiment_dirs = _find_agent_experiment_dirs(run_dir, ordering_type_id)
    if not experiment_dirs:
        st.info("No agent results found for this ordering type.")
        return

    experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
    run_subdirs = sorted(
        [
            d
            for d in experiment_dir.iterdir()
            if d.is_dir() and re.match(r"run\d+$", d.name)
        ],
        key=lambda p: int(p.name[3:]),
    )
    if not run_subdirs:
        st.info("No agent runs found.")
        return

    st.markdown(f"**Experiment**: `{experiment_dir.name}`")

    # --- Descriptions + Grader Reasonings ---
    st.markdown("### Descriptions & Grades")
    for rd in run_subdirs:
        desc_path = rd / "description.txt"
        description = (
            desc_path.read_text(encoding="utf-8").strip()
            if desc_path.exists()
            else "(no description)"
        )
        grades = _read_agent_grade_data(rd)
        scores = [int(g["score"]) for g in grades]
        scores_str = ", ".join(str(s) for s in scores) if scores else "no grades"
        label = f"{rd.name} — grades: [{scores_str}]"
        with st.expander(label, expanded=False):
            st.markdown(description)
            if grades:
                st.divider()
                st.markdown("**Grader Reasonings**")
                for g in grades:
                    score = int(g["score"])
                    grader = str(g.get("grader_model_id", "unknown"))
                    run_idx = g.get("run_idx", "?")
                    with st.expander(
                        f"Grader {run_idx} ({grader}) — score: {score}", expanded=False
                    ):
                        st.markdown(str(g.get("reasoning", "")))

    # --- Transcripts ---
    st.markdown("### Transcripts")
    selected_run = st.selectbox(
        "Select run",
        [rd.name for rd in run_subdirs],
        key=f"agent_transcript_run_{ordering_type_id}",
    )
    run_subdir = experiment_dir / selected_run

    # Show stats if available
    stats_path = run_subdir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        cols = st.columns(4)
        cols[0].metric("LLM calls", stats.get("agent_llm_calls_used", "?"))
        cols[1].metric("Model interactions", stats.get("model_interactions_used", "?"))
        cols[2].metric("Prompt tokens", f"{stats.get('agent_prompt_tokens', 0):,}")
        cols[3].metric(
            "Completion tokens", f"{stats.get('agent_completion_tokens', 0):,}"
        )

    messages_path = run_subdir / "messages.json"
    if not messages_path.exists():
        st.warning(f"No messages.json found in {selected_run}.")
        return

    with open(messages_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        category = _classify_message(role, content)
        style = _MSG_STYLE.get(category, _MSG_STYLE["user"])

        if category == "system":
            with st.expander(f":{style['icon']}: **{style['label']}**", expanded=False):
                st.code(content, language=None)
        elif category == "tool_call":
            with st.expander(
                f":{style['icon']}: **{style['label']}** — `{content[:80]}…`",
                expanded=False,
            ):
                st.code(content, language=None)
        elif category == "tool_result":
            # Extract tool name from TOOL_RESULT(name):
            tool_match = re.match(r"TOOL_RESULT\((\w+)\)", content)
            tool_name = tool_match.group(1) if tool_match else "?"
            with st.expander(
                f":{style['icon']}: **{style['label']}** ({tool_name})", expanded=False
            ):
                st.code(content, language=None)
        elif category == "reasoning":
            with st.expander(f":{style['icon']}: **{style['label']}**", expanded=True):
                st.markdown(content)
        elif category == "final":
            st.success(f":{style['icon']}: **{style['label']}**")
            st.markdown(content)
        elif category == "format_error":
            with st.expander(f":{style['icon']}: **{style['label']}**", expanded=False):
                st.warning(content)
        elif category == "overview":
            with st.expander(
                f":{style['icon']}: **{style['label']}** (initial input)",
                expanded=False,
            ):
                st.code(content, language=None)
        else:
            with st.expander(f"**{role}**", expanded=False):
                st.code(content, language=None)


def _read_agent_grades_from_run_subdir(run_subdir: Path) -> Optional[float]:
    """Read all ``hypothesis_grade_*.json`` files in *run_subdir* and return mean score."""
    scores = _read_agent_grade_scores(run_subdir)
    if not scores:
        return None
    return float(sum(scores)) / float(len(scores))


def _compute_agent_grade_stats(
    experiment_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Compute mean and std of agent grades over all runs in *experiment_dir*."""
    run_subdirs = sorted(
        [
            d
            for d in experiment_dir.iterdir()
            if d.is_dir() and re.match(r"run\d+$", d.name)
        ],
        key=lambda p: int(p.name[3:]),
    )
    if not run_subdirs:
        return None

    run_scores: List[float] = []
    for rd in run_subdirs:
        score = _read_agent_grades_from_run_subdir(rd)
        if score is not None:
            run_scores.append(score)

    if not run_scores:
        return None

    arr = np.array(run_scores)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "n_runs": len(run_scores),
    }


def _compute_agent_grade_row(
    *,
    run_label: str,
    run_dir: Path,
    ordering_type_id: str,
) -> Optional[Dict[str, Any]]:
    """Build a comparison-table row with agent grade mean/std for one ordering type."""
    experiment_dirs = _find_agent_experiment_dirs(run_dir, ordering_type_id)
    if not experiment_dirs:
        return None
    experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)

    stats = _compute_agent_grade_stats(experiment_dir)
    if stats is None:
        return None

    return {
        "run": run_label,
        "grade_mean": stats["mean"],
        "grade_std": stats["std"],
        "n_runs": stats["n_runs"],
    }


def _render_cross_run_comparison(method) -> None:
    """Compare token relevance or agent grades across multiple runs."""
    st.markdown("### Run comparison")

    run_dirs = _list_run_dirs(method)
    if not run_dirs:
        st.info("No new-schema runs found to compare.")
        return

    organism_variant = method.base_results_dir.parent.name

    comparison_mode = st.radio(
        "Metric",
        ["Relevance", "Agent Grades"],
        key="diff_mining::cross_run_comparison::mode",
        horizontal=True,
    )

    ordering_id_to_info: Dict[str, Dict[str, Any]] = {}
    for run_dir in run_dirs:
        for ot in _find_ordering_types(run_dir):
            ordering_id_to_info.setdefault(str(ot["id"]), ot)

    if not ordering_id_to_info:
        st.info("No ordering types found across runs.")
        return

    ordering_ids_sorted = sorted(ordering_id_to_info.keys())
    default_ordering_ids: List[str] = []
    for desired in ["top_k_occurring", "nmf"]:
        for oid, info in ordering_id_to_info.items():
            meta = info.get("metadata", {})
            if str(meta.get("ordering_type_id", "")) == desired:
                default_ordering_ids.append(oid)
                break
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

    if comparison_mode == "Relevance":
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
    run_select_key = (
        f"diff_mining::cross_run_comparison::runs::{dataset_name}"
        if comparison_mode == "Relevance"
        else "diff_mining::cross_run_comparison::runs::agent_grades"
    )
    selected_run_labels = st.multiselect(
        "Select runs",
        list(run_label_to_dir.keys()),
        default=default_labels,
        key=run_select_key,
    )

    if not selected_run_labels:
        st.info("Select one or more runs to compare.")
        return

    if comparison_mode == "Relevance":
        _render_relevance_comparison(
            run_label_to_dir=run_label_to_dir,
            selected_run_labels=selected_run_labels,
            selected_ordering_ids=selected_ordering_ids,
            ordering_id_to_info=ordering_id_to_info,
            dataset_name=dataset_name,
            top_k_tokens=top_k_tokens,
            adjust_for_multitopic=adjust_for_multitopic,
            organism_variant=organism_variant,
        )
    else:
        _render_agent_grades_comparison(
            run_label_to_dir=run_label_to_dir,
            selected_run_labels=selected_run_labels,
            selected_ordering_ids=selected_ordering_ids,
            ordering_id_to_info=ordering_id_to_info,
            organism_variant=organism_variant,
        )


def _render_relevance_comparison(
    *,
    run_label_to_dir: Dict[str, Path],
    selected_run_labels: List[str],
    selected_ordering_ids: List[str],
    ordering_id_to_info: Dict[str, Dict[str, Any]],
    dataset_name: str,
    top_k_tokens: int,
    adjust_for_multitopic: bool,
    organism_variant: str,
) -> None:
    """Render the token-relevance cross-run bar chart."""
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for run_label in selected_run_labels:
        run_dir = run_label_to_dir[run_label]
        for ordering_type_id in selected_ordering_ids:
            dataset_dir = run_dir / ordering_type_id / dataset_name
            ordering_meta = ordering_id_to_info[ordering_type_id].get("metadata", {})
            ot_id = str(ordering_meta.get("ordering_type_id", ordering_type_id))
            is_nmf = ot_id == "nmf"
            nmf_aggregation = "Best-topic" if is_nmf else None
            row = _compute_ordering_relevance_row(
                run_label=run_label,
                dataset_dir=dataset_dir,
                ordering_type_id=ot_id,
                aggregation_mode=nmf_aggregation,
                k_top=top_k_tokens,
                adjust_for_multitopic=adjust_for_multitopic,
            )
            if row is None:
                skipped.append(f"{run_label} | {ordering_type_id}")
                continue
            display_name = str(ordering_id_to_info[ordering_type_id]["display_name"])
            if is_nmf:
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

    run_scores = (
        df.groupby("run", as_index=False)["% relevant"]
        .max()
        .sort_values("% relevant", ascending=False)
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


def _render_agent_grades_comparison(
    *,
    run_label_to_dir: Dict[str, Path],
    selected_run_labels: List[str],
    selected_ordering_ids: List[str],
    ordering_id_to_info: Dict[str, Dict[str, Any]],
    organism_variant: str,
) -> None:
    """Render the agent-grade cross-run bar chart with error bars."""
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for run_label in selected_run_labels:
        run_dir = run_label_to_dir[run_label]
        for ordering_type_id in selected_ordering_ids:
            ordering_meta = ordering_id_to_info[ordering_type_id].get("metadata", {})
            ot_id = str(ordering_meta.get("ordering_type_id", ordering_type_id))
            row = _compute_agent_grade_row(
                run_label=run_label,
                run_dir=run_dir,
                ordering_type_id=ot_id,
            )
            if row is None:
                skipped.append(f"{run_label} | {ordering_type_id}")
                continue
            display_name = str(ordering_id_to_info[ordering_type_id]["display_name"])
            row["ordering_type"] = display_name
            rows.append(row)

    if skipped:
        st.warning(
            "Skipping runs missing agent grades for this ordering type:\n"
            + "\n".join(f"- {s}" for s in skipped)
        )

    if not rows:
        st.info("No agent grade data found for the selected runs/ordering types.")
        return

    df = pd.DataFrame(rows).copy()

    run_scores = (
        df.groupby("run", as_index=False)["grade_mean"]
        .max()
        .sort_values("grade_mean", ascending=False)
    )
    run_order = run_scores["run"].tolist()
    axis_base_labels = [_axis_label_for_run_dir(run_label_to_dir[r]) for r in run_order]
    axis_labels = _dedupe_labels(axis_base_labels)
    short_by_run = dict(zip(run_order, axis_labels))
    df["run_short"] = df["run"].map(short_by_run)
    df = df.sort_values(["run_short", "ordering_type"], ascending=[True, True])

    fig = px.bar(
        df,
        x="run_short",
        y="grade_mean",
        color="ordering_type",
        barmode="group",
        text="grade_mean",
        error_y="grade_std",
        hover_data=[c for c in df.columns if c not in {"grade_mean", "run_short"}],
        category_orders={
            "run_short": [short_by_run[r] for r in run_order],
        },
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        title_text=str(organism_variant),
        title_x=0.5,
        yaxis_range=[0, 5.5],
        yaxis_title="Agent Grade (1-5)",
        xaxis_title=None,
        xaxis_tickangle=-30,
        height=450,
        legend_title_text=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    display_df = df.drop(columns=["run_short"])
    st.dataframe(display_df, hide_index=True, use_container_width=True)
