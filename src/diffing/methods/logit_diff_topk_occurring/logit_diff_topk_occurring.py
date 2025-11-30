"""
Logit Diff Top-K Occurring analysis method.

This module computes occurrence rates of tokens in the top-K positive and negative
logit differences between a base model and a finetuned model.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from omegaconf import DictConfig
from loguru import logger
import json
from tqdm import tqdm
from collections import defaultdict

from ..diffing_method import DiffingMethod
from src.utils.configs import get_dataset_configurations, DatasetConfig
from src.utils.agents.diffing_method_agent import DiffingMethodAgent
from src.utils.agents.base_agent import BaseAgent
from src.utils.graders.token_relevance_grader import TokenRelevanceGrader
from ..activation_difference_lens.token_relevance import _compute_frequent_tokens
from ..activation_difference_lens.act_diff_lens import (
    load_and_tokenize_dataset,
    load_and_tokenize_chat_dataset,
)
from .normalization import normalize_token_list
from .ui import visualize
from .per_token_plots import plot_per_sample_occurrences, plot_per_position_occurrences


class LogitDiffTopKOccurringMethod(DiffingMethod):
    """
    Computes occurrence rates of tokens in top-K positive and negative logit differences.

    This method:
    1. Loads base and finetuned models (via DiffingMethod base class)
    2. Loads datasets directly from HuggingFace
    3. Computes logit differences between models for each position
    4. Tracks which tokens appear most frequently in top-K positive/negative diffs
    5. Saves occurrence rate statistics to disk
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Method-specific configuration
        self.method_cfg = cfg.diffing.method

        # Get dataset configurations
        self.datasets = get_dataset_configurations(
            cfg,
            use_chat_dataset=self.method_cfg.datasets.use_chat_dataset,
            use_pretraining_dataset=self.method_cfg.datasets.use_pretraining_dataset,
            use_training_dataset=self.method_cfg.datasets.use_training_dataset,
        )

        # Filter out validation datasets (only use train split)
        self.datasets = [ds for ds in self.datasets if ds.split == "train"]

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "logit_diff_topk_occurring"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def setup_models(self):
        """Ensure models are loaded (they will auto-load via properties)."""
        # Access properties to trigger lazy loading
        self.logger.info(f"Loading base model: {self.base_model_cfg.model_id}")
        _ = self.base_model
        self.logger.info(f"✓ Base model loaded: {self.base_model_cfg.model_id}")
        
        self.logger.info(f"Loading finetuned model: {self.finetuned_model_cfg.model_id}")
        _ = self.finetuned_model
        self.logger.info(f"✓ Finetuned model loaded: {self.finetuned_model_cfg.model_id}")
        
        self.logger.info("✓ Both models loaded and set to eval mode")

    @torch.no_grad()
    def compute_occurrence_for_dataset(self, dataset_cfg: DatasetConfig) -> Dict[str, Any]:
        """
        Core analysis for one dataset.

        Args:
            dataset_cfg: Dataset configuration

        Returns:
            Dictionary containing occurrence rate statistics
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Processing dataset: {dataset_cfg.id}")
        self.logger.info(f"Dataset name: {dataset_cfg.name}")

        # Get parameters from config (no hardcoded values)
        batch_size = int(self.method_cfg.method_params.batch_size)
        max_tokens = int(self.method_cfg.method_params.max_tokens_per_sample)
        max_samples = int(self.method_cfg.method_params.max_samples)
        top_k = int(self.method_cfg.method_params.top_k)
        ignore_padding = bool(self.method_cfg.method_params.ignore_padding)

        self.logger.info(f"Parameters: batch_size={batch_size}, max_tokens={max_tokens}, top_k={top_k}, max_samples={max_samples}")
        self.logger.info(f"Dataset type: {'chat' if dataset_cfg.is_chat else 'text'}")

        # Track occurrences across all positions
        global_token_counts = defaultdict(lambda: {"count_positive": 0, "count_negative": 0})
        total_positions = 0

        # Per-token analysis: Initialize tracking structures if enabled
        per_token_enabled = False
        shortlist_token_ids = {}
        per_sample_counts = {}
        per_position_counts = {}
        
        if hasattr(self.method_cfg, 'per_token_analysis') and self.method_cfg.per_token_analysis.enabled:
            per_token_enabled = True
            self.logger.info("Per-token analysis enabled")
            
            # Build token_id -> token_str mapping for shortlist
            for token_str in self.method_cfg.per_token_analysis.token_shortlist:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_ids) == 1:
                    shortlist_token_ids[token_ids[0]] = token_str
                    per_sample_counts[token_str] = defaultdict(int)
                    per_position_counts[token_str] = defaultdict(int)
                else:
                    self.logger.warning(
                        f"Token '{token_str}' encodes to {len(token_ids)} tokens, skipping. "
                        f"Use single-token strings only."
                    )
            
            self.logger.info(f"Tracking {len(shortlist_token_ids)} shortlist tokens: {list(shortlist_token_ids.values())}")

        # Tokenize entire dataset using ADL functions
        if dataset_cfg.is_chat:
            # Chat dataset - use ADL's chat function
            self.logger.info("Using ADL's load_and_tokenize_chat_dataset() with apply_chat_template()")
            samples = load_and_tokenize_chat_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                messages_column=dataset_cfg.messages_column or "messages",
                n=max_tokens,  # From config: max_tokens_per_sample
                pre_assistant_k=0,  # Don't need pre-assistant context for global analysis
                max_samples=max_samples,  # From config: max_samples
            )
            # Extract just the token IDs
            all_token_ids = [sample["input_ids"] for sample in samples]
        else:
            # Text dataset - use ADL's text function
            self.logger.info("Using ADL's load_and_tokenize_dataset()")
            all_token_ids = load_and_tokenize_dataset(
                dataset_name=dataset_cfg.id,
                tokenizer=self.tokenizer,
                split=dataset_cfg.split,
                text_column=dataset_cfg.text_column or "text",
                n=max_tokens,  # From config: max_tokens_per_sample
                max_samples=max_samples,  # From config: max_samples
            )

        # Now batch through token IDs
        num_samples = len(all_token_ids)
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.logger.info(f"Processing {num_samples} samples in {num_batches} batches...")
        
        # Track max sequence length across all batches for per-token analysis
        overall_max_len = 0

        for batch_idx in tqdm(range(num_batches), desc=f"Processing {dataset_cfg.name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Get batch of token IDs
            batch_token_ids = all_token_ids[start_idx:end_idx]

            # Pad to same length
            max_len = max(len(ids) for ids in batch_token_ids)
            overall_max_len = max(overall_max_len, max_len)
            input_ids_list = []
            attention_mask_list = []

            for token_ids in batch_token_ids:
                # Pad
                padding_length = max_len - len(token_ids)
                padded_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
                mask = [1] * len(token_ids) + [0] * padding_length

                input_ids_list.append(padded_ids)
                attention_mask_list.append(mask)

            # Convert to tensors
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

            # Get logits from both models (NO GRADIENTS, models already in eval mode)
            inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
            with self.base_model.trace(inputs):
                base_logits = self.base_model.logits.save()
            with self.finetuned_model.trace(inputs):
                finetuned_logits = self.finetuned_model.logits.save()

            # Extract logits [batch_size, seq_len, vocab_size]
            # Ensure same device
            target_device = base_logits.device
            finetuned_logits = finetuned_logits.to(target_device)
            attention_mask = attention_mask.to(target_device)

            # Compute difference: finetuned - base
            diff = finetuned_logits - base_logits  # [batch_size, seq_len, vocab_size]

            # Get top-K positive diffs (largest values)
            top_k_pos_values, top_k_pos_indices = torch.topk(
                diff, k=top_k, dim=-1, largest=True
            )  # [batch_size, seq_len, top_k]

            # Get top-K negative diffs (smallest values)
            top_k_neg_values, top_k_neg_indices = torch.topk(
                diff, k=top_k, dim=-1, largest=False
            )  # [batch_size, seq_len, top_k]

            # Track occurrences (respecting attention mask if ignore_padding=True)
            batch_size_actual, seq_len, _ = diff.shape

            for b in range(batch_size_actual):
                sample_idx = start_idx + b  # Global sample index
                
                for s in range(seq_len):
                    # Skip padding tokens if requested
                    if ignore_padding and attention_mask[b, s] == 0:
                        continue

                    # Count positive occurrences
                    for token_id in top_k_pos_indices[b, s]:
                        token_id_item = token_id.item()
                        global_token_counts[token_id_item]["count_positive"] += 1
                        
                        # Per-token analysis: track shortlist tokens
                        if per_token_enabled and token_id_item in shortlist_token_ids:
                            token_str = shortlist_token_ids[token_id_item]
                            per_sample_counts[token_str][sample_idx] += 1
                            per_position_counts[token_str][s] += 1

                    # Count negative occurrences
                    for token_id in top_k_neg_indices[b, s]:
                        token_id_item = token_id.item()
                        global_token_counts[token_id_item]["count_negative"] += 1
                        
                        # Per-token analysis: track shortlist tokens in negative direction too
                        # (Note: we track both positive and negative, aggregating total occurrences)
                        if per_token_enabled and token_id_item in shortlist_token_ids:
                            token_str = shortlist_token_ids[token_id_item]
                            per_sample_counts[token_str][sample_idx] += 1
                            per_position_counts[token_str][s] += 1

                    total_positions += 1

        self.logger.info(f"✓ Batch processing complete!")
        self.logger.info(
            f"Processed {total_positions:,} positions with {len(global_token_counts):,} unique tokens"
        )

        # Compute occurrence rates
        self.logger.info(f"Computing occurrence rates...")
        all_tokens = []
        for token_id, counts in global_token_counts.items():
            token_str = self.tokenizer.decode([token_id])
            all_tokens.append({
                "token_id": token_id,
                "token_str": token_str,
                "count_positive": counts["count_positive"],
                "count_negative": counts["count_negative"],
                "positive_occurrence_rate": (counts["count_positive"] / total_positions) * 100
                if total_positions > 0
                else 0.0,
                "negative_occurrence_rate": (counts["count_negative"] / total_positions) * 100
                if total_positions > 0
                else 0.0,
            })

        # Sort by positive and negative occurrence rates (DIRECT sort, no union)
        pos_rates = torch.tensor([t["positive_occurrence_rate"] for t in all_tokens])
        neg_rates = torch.tensor([t["negative_occurrence_rate"] for t in all_tokens])

        # Get top tokens for each
        # Save max(num_tokens_to_plot, top_k) to avoid needing to rerun if visualization config changes
        num_tokens_to_save = max(
            int(self.method_cfg.visualization.num_tokens_to_plot),
            int(top_k)
        )
        k_pos = min(num_tokens_to_save, len(all_tokens))
        k_neg = min(num_tokens_to_save, len(all_tokens))

        top_k_pos_values, top_k_pos_indices = torch.topk(pos_rates, k=k_pos, largest=True)
        top_k_neg_values, top_k_neg_indices = torch.topk(neg_rates, k=k_neg, largest=True)

        top_positive = [all_tokens[i] for i in top_k_pos_indices.tolist()]
        top_negative = [all_tokens[i] for i in top_k_neg_indices.tolist()]

        self.logger.info(f"✓ Top tokens computed:")
        self.logger.info(f"  Top positive token: {top_positive[0]['token_str']} ({top_positive[0]['positive_occurrence_rate']:.2f}%)")
        self.logger.info(f"  Top negative token: {top_negative[0]['token_str']} ({top_negative[0]['negative_occurrence_rate']:.2f}%)")

        # Create results dictionary
        results = {
            "dataset_id": dataset_cfg.id,
            "dataset_name": dataset_cfg.name,
            "total_positions": total_positions,
            "num_samples": num_samples,
            "top_k": top_k,
            "unique_tokens": len(global_token_counts),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "metadata": {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
                "max_tokens_per_sample": max_tokens,
                "batch_size": batch_size,
            },
        }
        
        # Add per-token analysis data if enabled (for internal use, not saved to main JSON)
        if per_token_enabled:
            results["_per_token_data"] = {
                "per_sample_counts": per_sample_counts,
                "per_position_counts": per_position_counts,
                "max_positions": overall_max_len,
            }

        return results

    def save_results(self, dataset_name: str, results: Dict[str, Any]) -> Path:
        """
        Save results for a dataset to disk.

        Args:
            dataset_name: Dataset name (safe for filename)
            results: Results dictionary

        Returns:
            Path to saved file
        """
        output_file = self.results_dir / f"{dataset_name}_occurrence_rates.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved results for {dataset_name} to {output_file}")
        return output_file

    def _save_and_plot_per_token_analysis(
        self,
        dataset_name: str,
        per_sample_counts: Dict[str, Dict[int, int]],
        per_position_counts: Dict[str, Dict[int, int]],
        num_samples: int,
        max_positions: int
    ) -> None:
        """
        Save per-token analysis data and generate plots.
        
        Saves two JSON files (per-sample and per-position counts) and generates
        corresponding plots for each token in the shortlist.
        
        Directory structure:
            {results_dir}/per_token_analysis/
                ├── data/       (JSON files)
                └── plots/      (PNG files)
        
        Args:
            dataset_name: Dataset name for filenames
            per_sample_counts: Dict[token_str][sample_idx] = count
            per_position_counts: Dict[token_str][position_idx] = count
            num_samples: Total number of samples processed
            max_positions: Maximum sequence length
        """
        self.logger.info(f"Saving per-token analysis for {dataset_name}...")
        
        # Create subdirectories for organized output
        per_token_dir = self.results_dir / "per_token_analysis"
        data_dir = per_token_dir / "data"
        plots_dir = per_token_dir / "plots"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts and then to lists for JSON serialization
        per_sample_serializable = {}
        for token_str, sample_dict in per_sample_counts.items():
            # Convert to dense list for easier consumption
            per_sample_serializable[token_str] = [
                sample_dict.get(i, 0) for i in range(num_samples)
            ]
        
        per_position_serializable = {}
        for token_str, position_dict in per_position_counts.items():
            # Convert to dense list
            per_position_serializable[token_str] = [
                position_dict.get(i, 0) for i in range(max_positions)
            ]
        
        # Save per-sample counts to data/ subdirectory
        per_sample_data = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
            "tokens": per_sample_serializable
        }
        per_sample_file = data_dir / f"per_token_analysis_by_sample_{dataset_name}.json"
        with open(per_sample_file, "w", encoding="utf-8") as f:
            json.dump(per_sample_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"  Saved per-sample counts: data/{per_sample_file.name}")
        
        # Save per-position counts to data/ subdirectory
        per_position_data = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
            "max_positions": max_positions,
            "tokens": per_position_serializable
        }
        per_position_file = data_dir / f"per_token_analysis_by_position_{dataset_name}.json"
        with open(per_position_file, "w", encoding="utf-8") as f:
            json.dump(per_position_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"  Saved per-position counts: data/{per_position_file.name}")
        
        # Generate plots in plots/ subdirectory
        self.logger.info(f"Generating per-token plots...")
        
        # Plot per-sample occurrences (with max_positions in title)
        sample_plots = plot_per_sample_occurrences(
            per_sample_counts,
            dataset_name,
            plots_dir,
            num_samples,
            max_positions=max_positions
        )
        
        # Plot per-position occurrences (with num_samples in title)
        position_plots = plot_per_position_occurrences(
            per_position_counts,
            dataset_name,
            plots_dir,
            max_positions,
            num_samples=num_samples
        )
        
        total_plots = len(sample_plots) + len(position_plots)
        self.logger.info(f"✓ Per-token analysis complete: {total_plots} plots in plots/, {len(per_sample_serializable)} tokens tracked")

    def run_token_relevance(self) -> None:
        """Grade top-K positive tokens for relevance to finetuning domain."""
        cfg = self.method_cfg.token_relevance
        
        if not cfg.enabled:
            return
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("RUNNING TOKEN RELEVANCE GRADING")
        logger.info("=" * 80)
        
        overwrite = bool(cfg.overwrite)
        agreement_mode = str(cfg.agreement)
        
        # Get organism description
        organism_cfg = self.cfg.organism
        assert hasattr(organism_cfg, "description_long"), (
            "Organism config must have 'description_long' for token relevance grading"
        )
        description = str(organism_cfg.description_long)
        logger.info(f"Using organism description: {description[:100]}...")
        
        # Initialize grader
        grader_cfg = cfg.grader
        grader = TokenRelevanceGrader(
            grader_model_id=str(grader_cfg.model_id),
            base_url=str(grader_cfg.base_url),
            api_key_path=str(grader_cfg.api_key_path),
        )
        logger.info(f"Initialized grader: {grader_cfg.model_id}")
        
        # Compute frequent tokens from training dataset (if available)
        freq_cfg = cfg.frequent_tokens
        has_training_dataset = hasattr(organism_cfg, "training_dataset") and (
            organism_cfg.training_dataset is not None
        )
        
        if has_training_dataset:
            # Extract dataset ID from training_dataset config
            training_dataset_id = str(organism_cfg.training_dataset.id)
            logger.info(f"Computing frequent tokens from training dataset: {training_dataset_id}")
            
            # Read is_chat from config
            training_is_chat = False
            training_dataset_cfg = organism_cfg.training_dataset
            if hasattr(training_dataset_cfg, 'is_chat'):
                training_is_chat = bool(training_dataset_cfg.is_chat)
            elif isinstance(training_dataset_cfg, dict):
                training_is_chat = bool(training_dataset_cfg.get('is_chat', False))
            
            logger.info(f"Training dataset is_chat: {training_is_chat}")
            
            frequent_tokens = _compute_frequent_tokens(
                dataset_name=training_dataset_id,
                tokenizer=self.tokenizer,
                splits=["train"],
                num_tokens=int(freq_cfg.num_tokens),
                min_count=int(freq_cfg.min_count),
                is_chat=training_is_chat,  # Use actual value from config
                subset=None,
            )
            logger.info(f"Found {len(frequent_tokens)} frequent tokens")
        else:
            frequent_tokens = []
            logger.info("No training dataset available, using empty frequent tokens list")
        
        # Process each dataset
        for dataset_cfg in self.datasets:
            dataset_name = dataset_cfg.name
            results_file = self.results_dir / f"{dataset_name}_occurrence_rates.json"
            
            if not results_file.exists():
                logger.warning(f"No results found for {dataset_name}, skipping token relevance")
                continue
            
            logger.info("")
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Load saved occurrence results
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Apply normalization if enabled
            use_normalized = bool(self.method_cfg.get("use_normalized_tokens", False))
            top_positive = results["top_positive"]
            if use_normalized:
                total_positions = results.get("total_positions", 0)
                top_positive = normalize_token_list(top_positive, total_positions)
                logger.info(f"Applied token normalization for {dataset_name}: {len(results['top_positive'])} -> {len(top_positive)} tokens")
            
            # Output directory (matches ADL structure with layer_global/position_all)
            dataset_dir_name = dataset_cfg.id.split("/")[-1]
            out_dir = (
                self.results_dir
                / "layer_global"
                / dataset_dir_name
                / "token_relevance"
                / "position_all"
                / "difference"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            rel_path = out_dir / "relevance_logit_diff.json"
            
            # Skip if exists and not overwriting
            if (not overwrite) and rel_path.exists():
                logger.info(f"Token relevance exists for {dataset_name}, skipping (overwrite=False)")
                continue
            
            # Extract top-K positive tokens and their occurrence rates
            k_tokens = min(int(cfg.k_candidate_tokens), len(top_positive))
            candidate_tokens = [t["token_str"] for t in top_positive[:k_tokens]]
            token_weights = [t["positive_occurrence_rate"] / 100.0 for t in top_positive[:k_tokens]]
            
            logger.info(f"Grading {k_tokens} top positive tokens for {dataset_name}")
            logger.info(f"Top 5 tokens: {candidate_tokens[:5]}")
            
            # Compute trivial baseline
            trivial_hits = sum(1 for t in candidate_tokens if t in frequent_tokens)
            trivial_percentage = trivial_hits / float(len(candidate_tokens)) if candidate_tokens else 0.0
            
            # Grade with permutation robustness
            permutations = int(grader_cfg.permutations)
            logger.info(f"Running grader with {permutations} permutations...")
            majority_labels, permutation_labels, raw_responses = grader.grade(
                description=description,
                frequent_tokens=frequent_tokens,
                candidate_tokens=candidate_tokens,
                permutations=permutations,
                concurrent=True,
                max_tokens=int(grader_cfg.max_tokens),
            )
            
            # Aggregate labels based on agreement mode
            if agreement_mode == "majority":
                final_labels = majority_labels
            else:  # "all"
                n = len(candidate_tokens)
                final_labels = [
                    "RELEVANT" if all(run[i] == "RELEVANT" for run in permutation_labels)
                    else "IRRELEVANT"
                    for i in range(n)
                ]
            
            # Compute percentage metrics
            relevant_fraction = sum(lbl == "RELEVANT" for lbl in final_labels) / float(len(final_labels))
            
            # Weighted percentage using occurrence rates
            total_w = sum(token_weights)
            relevant_w = sum(w for lbl, w in zip(final_labels, token_weights) if lbl == "RELEVANT")
            weighted_percentage = relevant_w / total_w if total_w > 0 else 0.0
            
            # Create output record (matches ADL format)
            rec = {
                "layer": "global",
                "position": "all",
                "variant": "difference",
                "source": "logit_diff",
                "target": "self",
                "labels": final_labels,
                "tokens": candidate_tokens,
                "percentage": relevant_fraction,
                "trivial_percentage": trivial_percentage,
                "weighted_percentage": float(weighted_percentage),
                "grader_responses": raw_responses,
            }
            
            # Save results
            rel_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            logger.info(f"✓ Token relevance saved: {rel_path}")
            logger.info(f"  Relevance: {relevant_fraction*100:.1f}%, Weighted: {weighted_percentage*100:.1f}%")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Token relevance grading completed!")
        logger.info("=" * 80)

    def run(self) -> None:
        """
        Main execution method for logit diff top-K occurring analysis.

        Processes each dataset separately and saves results to disk.
        """
        # Ensure models are loaded
        self.logger.info("=" * 80)
        self.logger.info("LOGIT DIFF TOP-K OCCURRING ANALYSIS")
        self.logger.info("=" * 80)
        self.setup_models()

        self.logger.info("")
        self.logger.info("Starting logit diff top-K occurring analysis...")
        self.logger.info(f"Number of datasets to process: {len(self.datasets)}")

        # Check if results already exist
        if not self.method_cfg.overwrite:
            existing_results = list(self.results_dir.glob("*_occurrence_rates.json"))
            if len(existing_results) >= len(self.datasets):
                self.logger.info(
                    f"Results already exist in {self.results_dir}. Skipping computation."
                )
                return

        # Process each dataset
        for idx, dataset_cfg in enumerate(self.datasets, 1):
            self.logger.info("")
            self.logger.info(f"[{idx}/{len(self.datasets)}] Starting dataset: {dataset_cfg.name}")
            results = self.compute_occurrence_for_dataset(dataset_cfg)

            if results is not None:
                # Save results to disk
                self.save_results(dataset_cfg.name, results)
                
                # Save and plot per-token analysis if enabled
                if "_per_token_data" in results:
                    self._save_and_plot_per_token_analysis(
                        dataset_cfg.name,
                        results["_per_token_data"]["per_sample_counts"],
                        results["_per_token_data"]["per_position_counts"],
                        results["num_samples"],
                        results["_per_token_data"]["max_positions"]
                    )
                
                self.logger.info(f"✓ [{idx}/{len(self.datasets)}] Completed dataset: {dataset_cfg.name}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("✓ Logit diff top-K occurring analysis completed successfully!")
        self.logger.info(f"✓ Results saved to: {self.results_dir}")
        self.logger.info("=" * 80)
        
        # Run token relevance grading if enabled
        if hasattr(self.method_cfg, 'token_relevance') and self.method_cfg.token_relevance.enabled:
            self.run_token_relevance()

    def visualize(self) -> None:
        """
        Create Streamlit visualization for logit diff top-K occurring results.

        Returns:
            Streamlit component displaying occurrence rankings and interactive heatmap
        """
        visualize(self)

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available logit diff top-K occurring results.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = defaultdict(dict)
        results_base = results_dir

        if not results_base.exists():
            return results

        # Scan for results in the expected structure
        for base_model_dir in results_base.iterdir():
            if not base_model_dir.is_dir():
                continue

            model_name = base_model_dir.name

            for organism_dir in base_model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                method_dir = organism_dir / "logit_diff_topk_occurring"
                if method_dir.exists() and list(method_dir.glob("*_occurrence_rates.json")):
                    results[model_name][organism_name] = str(method_dir)

        return results

    def get_agent(self) -> DiffingMethodAgent:
        from .agents import LogitDiffAgent
        return LogitDiffAgent(cfg=self.cfg)
