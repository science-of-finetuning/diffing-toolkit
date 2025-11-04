from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
import numpy as np
from src.utils.visualization import (
    filter_examples_by_search,
    create_examples_html,
    render_streamlit_html,
)


class MaxActivationDashboardComponent:
    """
    Reusable Streamlit component for displaying MaxActStore contents.

    Features:
    - Mandatory latent selection (if latents exist) via text input
    - Optional quantile filtering via selectbox
    - Text search functionality
    - Lazy loading with pagination for performance
    - Batch database queries for efficient retrieval
    - Prepared for hybrid preview/full detail modes
    """

    def __init__(
        self,
        max_store,
        title: str = "Maximum Activating Examples",
        initial_batch_size: int = 15,
        batch_size: int = 10,
    ):
        """

        Args:
            max_store: ReadOnlyMaxActStore or MaxActStore instance
            title: Title for the dashboard
            initial_batch_size: Number of examples to load initially
            batch_size: Number of examples to load in each subsequent batch
        """
        self.max_store = max_store
        self.title = title
        self.initial_batch_size = initial_batch_size
        self.batch_size = batch_size

    def _get_available_latents(self) -> List[int]:
        """Get list of available latent indices from the database."""
        return self.max_store.get_available_latents()

    def _get_available_quantiles(self) -> List[int]:
        """Get list of available quantile indices from the database."""
        return self.max_store.get_available_quantiles()

    def _get_available_datasets(self) -> List[str]:
        """Get list of available dataset names from the database."""
        return self.max_store.get_available_datasets()

    def _convert_maxstore_to_dashboard_format(
        self, examples: List[Dict[str, Any]], detail_mode: str = "full"
    ) -> List[Tuple[float, List[str], List[float], str]]:
        """
        Convert MaxActStore examples to dashboard format with support for different detail modes.

        Args:
            examples: List of examples from MaxActStore.get_top_examples()
            detail_mode: "preview" for quick display, "full" for complete visualization

        Returns:
            List of tuples (max_score, tokens, scores_per_token, text)
        """
        if not examples:
            return []

        # Assumption: MaxActStore has tokenizer for token conversion
        assert (
            self.max_store.tokenizer is not None
        ), "MaxActStore must have tokenizer for visualization"

        if detail_mode == "preview":
            # Preview mode: just return basic info without detailed scores
            dashboard_examples = []
            for example in examples:
                tokens = self.max_store.tokenizer.convert_ids_to_tokens(
                    example["input_ids"]
                )
                # Use uniform scores for preview (all zeros)
                scores_per_token = np.zeros(len(tokens))
                example_tuple = [
                    example["max_score"],
                    tokens,
                    scores_per_token,
                    example["text"],
                ]
                if "dataset_name" in example and example["dataset_name"] is not None:
                    example_tuple.append(example["dataset_name"])
                dashboard_examples.append(tuple(example_tuple))
            return dashboard_examples

        # Full mode: get detailed activation scores
        example_ids = [ex["example_id"] for ex in examples]

        # Use batch method for efficient database access
        detailed_examples = self.max_store.get_batch_example_details(
            example_ids, return_dense=True
        )

        # Create lookup dictionary for efficient matching
        details_dict = {ex["example_id"]: ex for ex in detailed_examples}

        dashboard_examples = []
        for example in examples:
            example_id = example["example_id"]

            # Get detailed info if available
            if example_id in details_dict:
                details = details_dict[example_id]
                # Assumption: All examples must have scores_per_token for full visualization
                assert (
                    "scores_per_token" in details
                ), f"Example {example_id} missing scores_per_token - cannot visualize in full mode"

                tokens = self.max_store.tokenizer.convert_ids_to_tokens(
                    example["input_ids"]
                )
                scores_per_token = np.array(details["scores_per_token"])

                # Shape assertion
                assert len(tokens) == len(
                    scores_per_token
                ), f"Token/score mismatch: {len(tokens)} tokens vs {len(scores_per_token)} scores"
            else:
                # Fallback to basic display if details not available
                tokens = self.max_store.tokenizer.convert_ids_to_tokens(
                    example["input_ids"]
                )
                scores_per_token = np.zeros(len(tokens))

            example_tuple = [
                example["max_score"],
                tokens,
                scores_per_token,
                example["text"],
            ]
            if "dataset_name" in example and example["dataset_name"] is not None:
                example_tuple.append(example["dataset_name"])
            dashboard_examples.append(tuple(example_tuple))

        return dashboard_examples

    def _get_session_keys(
        self,
        selected_latent: Optional[int],
        selected_quantile: Optional[int],
        selected_datasets: List[str],
        search_term: str,
    ) -> Dict[str, str]:
        """Generate session state keys based on current filters."""
        datasets_hash = (
            hash(tuple(sorted(selected_datasets))) % 10000 if selected_datasets else 0
        )
        db_path_hash = hash(str(self.max_store.db_manager.db_path)) % 10000
        base_key = f"maxact_{selected_latent}_{selected_quantile}_{datasets_hash}_{db_path_hash}_{hash(search_term) % 10000}"
        return {
            "examples": f"{base_key}_examples",
            "loaded_count": f"{base_key}_loaded_count",
            "total_count": f"{base_key}_total_count",
            "loading": f"{base_key}_loading",
        }

    def _load_examples_batch(
        self,
        selected_latent: Optional[int],
        selected_quantile: Optional[int],
        selected_datasets: List[str],
        start_idx: int,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Load a batch of examples with offset and limit."""
        # Get all examples with filters but limit to batch size with offset
        all_examples = self.max_store.get_top_examples(
            latent_idx=selected_latent,
            quantile_idx=selected_quantile,
            dataset_names=selected_datasets if selected_datasets else None,
        )

        # Apply offset and limit
        end_idx = start_idx + batch_size
        return all_examples[start_idx:end_idx]

    def display(self):
        """Render the dashboard component with lazy loading."""

        st.markdown(f"### {self.title}")

        # Get available filter options
        available_latents = self._get_available_latents()
        available_quantiles = self._get_available_quantiles()
        available_datasets = self._get_available_datasets()

        # Initialize filter values
        selected_latent = None
        selected_quantile = None
        selected_datasets = []

        # Latent selection (mandatory if latents exist)
        if available_latents:
            col1, col2 = st.columns([2, 1])
            with col1:
                latent_input = st.number_input(
                    "Latent Index (required)",
                    min_value=min(available_latents),
                    max_value=max(available_latents),
                    value=available_latents[0],
                    step=1,
                    help=f"Available latent indices: {min(available_latents)}-{max(available_latents)}",
                )

                # Validate the input
                if latent_input in available_latents:
                    selected_latent = latent_input
                else:
                    st.error(
                        f"Latent index {latent_input} not available. Available indices: {available_latents[:10]}{'...' if len(available_latents) > 10 else ''}"
                    )
                    return

            with col2:
                st.metric("Available Latents", len(available_latents))

        # Quantile selection (optional)
        if available_quantiles:
            quantile_options = ["All"] + [str(q) for q in available_quantiles]
            selected_quantile_str = st.selectbox(
                "Quantile Filter",
                options=quantile_options,
                help="Filter by quantile index (optional)",
            )

            if selected_quantile_str != "All":
                selected_quantile = int(selected_quantile_str)

        # Dataset selection (optional)
        if available_datasets:
            selected_datasets = st.multiselect(
                "Dataset Filter",
                options=available_datasets,
                default=[],
                help="Filter by dataset names (optional). Leave empty to show all datasets.",
            )

        # Search functionality
        search_term = st.text_input(
            "🔍 Search in examples",
            placeholder="Enter text to search for in the examples...",
        )

        # For methods without latents, we can still proceed
        if available_latents and selected_latent is None:
            st.info("Please select a latent index to view examples.")
            return

        # Generate session state keys
        session_keys = self._get_session_keys(
            selected_latent, selected_quantile, selected_datasets, search_term
        )

        # Initialize session state
        if session_keys["examples"] not in st.session_state:
            st.session_state[session_keys["examples"]] = []
            st.session_state[session_keys["loaded_count"]] = 0
            st.session_state[session_keys["total_count"]] = None
            st.session_state[session_keys["loading"]] = False

        # Reset if filters changed (check by comparing with a hash)
        current_filter_hash = hash(
            (
                selected_latent,
                selected_quantile,
                tuple(sorted(selected_datasets)),
                search_term,
            )
        )
        last_filter_key = f"{session_keys['examples']}_filter_hash"
        if (
            last_filter_key not in st.session_state
            or st.session_state[last_filter_key] != current_filter_hash
        ):
            st.session_state[session_keys["examples"]] = []
            st.session_state[session_keys["loaded_count"]] = 0
            st.session_state[session_keys["total_count"]] = None
            st.session_state[last_filter_key] = current_filter_hash

        # Load initial batch if nothing loaded yet
        if (
            not st.session_state[session_keys["examples"]]
            and not st.session_state[session_keys["loading"]]
        ):
            st.session_state[session_keys["loading"]] = True

            # Get total count first
            all_examples_for_count = self.max_store.get_top_examples(
                latent_idx=selected_latent,
                quantile_idx=selected_quantile,
                dataset_names=selected_datasets if selected_datasets else None,
            )
            st.session_state[session_keys["total_count"]] = len(all_examples_for_count)

            # Load initial batch
            initial_examples = self._load_examples_batch(
                selected_latent,
                selected_quantile,
                selected_datasets,
                0,
                self.initial_batch_size,
            )
            st.session_state[session_keys["examples"]] = initial_examples
            st.session_state[session_keys["loaded_count"]] = len(initial_examples)
            st.session_state[session_keys["loading"]] = False

        # Get current examples from session state
        loaded_examples = st.session_state[session_keys["examples"]]
        total_count = st.session_state[session_keys["total_count"]] or 0
        loaded_count = st.session_state[session_keys["loaded_count"]]

        # Apply search filter to loaded examples
        dashboard_examples = self._convert_maxstore_to_dashboard_format(
            loaded_examples, detail_mode="full"
        )
        if search_term.strip():
            dashboard_examples = filter_examples_by_search(
                dashboard_examples, search_term
            )

        # Calculate dataset distribution
        if loaded_examples:
            dataset_counts = {}
            for example in loaded_examples:
                dataset_name = example.get("dataset_name", "Unknown")
                dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

            total_loaded = len(loaded_examples)
            dataset_fractions = {
                name: count / total_loaded for name, count in dataset_counts.items()
            }

            # Display dataset distribution (always show, even for single datasets)
            if len(dataset_counts) >= 1:
                if len(dataset_counts) == 1:
                    dataset_name = list(dataset_counts.keys())[0]
                    st.caption(f"Dataset: {dataset_name} ({total_loaded} examples)")
                else:
                    fraction_parts = [
                        f"{name}: {fraction:.1%}"
                        for name, fraction in dataset_fractions.items()
                    ]
                    st.caption(
                        f"Dataset distribution in loaded examples: {', '.join(fraction_parts)}"
                    )

        # Build filter context message
        filter_parts = []
        if selected_latent is not None:
            filter_parts.append(f"Latent {selected_latent}")
        if selected_quantile is not None:
            filter_parts.append(f"Quantile {selected_quantile}")
        if selected_datasets:
            if len(selected_datasets) == 1:
                filter_parts.append(f"Dataset: {selected_datasets[0]}")
            else:
                filter_parts.append(
                    f"Datasets: {', '.join(selected_datasets[:2])}{' (+{} more)'.format(len(selected_datasets) - 2) if len(selected_datasets) > 2 else ''}"
                )
        if search_term.strip():
            filter_parts.append(f"Search: '{search_term}'")

        # Display status and load more button
        col1, col2 = st.columns([3, 1])

        with col1:
            context_msg = f"Showing {len(dashboard_examples)} examples"
            if search_term.strip():
                context_msg += f" (from {loaded_count} loaded, {total_count} total)"
            else:
                context_msg += f" ({loaded_count} of {total_count} loaded)"

            if filter_parts:
                context_msg += f" - {', '.join(filter_parts)}"

            st.info(context_msg)

        with col2:
            # Show load more button if there are more examples to load
            has_more = loaded_count < total_count
            if has_more and st.button(
                f"Load {min(self.batch_size, total_count - loaded_count)} More",
                disabled=st.session_state[session_keys["loading"]],
            ):
                st.session_state[session_keys["loading"]] = True

                # Load next batch
                next_batch = self._load_examples_batch(
                    selected_latent,
                    selected_quantile,
                    selected_datasets,
                    loaded_count,
                    self.batch_size,
                )

                # Append to existing examples
                st.session_state[session_keys["examples"]].extend(next_batch)
                st.session_state[session_keys["loaded_count"]] += len(next_batch)
                st.session_state[session_keys["loading"]] = False

                # Rerun to update display
                st.rerun()

        # Check if we have examples to show
        if not dashboard_examples:
            if search_term.strip():
                st.warning(
                    "No examples found matching your search. Try loading more examples or changing your search term."
                )
            else:
                st.warning("No examples found with the selected filters.")
            return

        # Create and render HTML visualization
        title_with_filters = self.title
        if filter_parts:
            title_with_filters += f" - {', '.join(filter_parts)}"

        html_content = create_examples_html(
            dashboard_examples,
            self.max_store.tokenizer,
            title=title_with_filters,
            max_examples=len(dashboard_examples),  # Show all loaded examples
            window_size=50,
            use_absolute_max=False,
        )

        # Render in Streamlit
        render_streamlit_html(html_content)
