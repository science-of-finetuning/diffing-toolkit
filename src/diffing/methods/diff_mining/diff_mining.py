"""
Diff Mining analysis method.

This module computes occurrence rates of tokens in the top-K positive and negative
logit differences between a base model and a finetuned model.
"""

from typing import Dict, Tuple, Any, List, Optional
from pathlib import Path
import torch
import gc
import asyncio
from omegaconf import DictConfig
from loguru import logger
import json
from collections import defaultdict
from datetime import datetime
import re

from ..diffing_method import DiffingMethod
from diffing.utils.configs import DatasetConfig
from diffing.utils.agents.diffing_method_agent import DiffingMethodAgent
from diffing.utils.graders.token_relevance_grader import TokenRelevanceGrader
from ..activation_difference_lens.token_relevance import _compute_frequent_tokens
from diffing.utils.activations import get_layer_indices
from .logit_extraction import (
    DirectLogitsExtractor,
    LogitLensExtractor,
    LogitsExtractor,
    PatchscopeLensExtractor,
)
from .token_ordering import (
    TokenOrderingType,
    TopKOccurringOrderingType,
    FractionPositiveDiffOrderingType,
    NmfOrderingType,
    NmfOrderingConfig,
    SharedTokenStats,
    OrderingTypeResult,
    write_ordering_type_metadata,
    write_dataset_orderings,
    write_ordering_eval,
)
from .ui import visualize
from .plots import plot_positional_kde
from .preprocessing import (
    prepare_dataset_tensors,
    prepare_all_dataset_inputs_and_save,
    infer_and_store_logits_for_all_datasets,
    infer_finetuned_and_compute_diffs_in_memory,
    compute_and_save_disk_diffs,
)
from .core_analysis import CoreAnalysisResult, compute_stats_from_logits as _compute_stats_from_logits


class DiffMiningMethod(DiffingMethod):
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

        # Build DatasetConfig objects from inline config (ADL-style)
        self.datasets = []
        split = str(self.method_cfg.split)

        for dataset_entry in self.method_cfg.datasets:
            dataset_id = dataset_entry["id"]
            subset = dataset_entry.get("subset", None)
            is_chat = dataset_entry["is_chat"]
            
            # Determine column name based on dataset type
            if is_chat:
                column = dataset_entry.get("messages_column", "messages")
            else:
                column = dataset_entry.get("text_column", "text")
            
            # Create unique name: base_id + optional subset + split + column
            # e.g., "GreatFirewall-DPO_train_chosen" or "CulturaX_es_train_text"
            base_name = dataset_id.split("/")[-1]
            if subset:
                base_name = f"{base_name}_{subset}"
            name = f"{base_name}_{split}_{column}"
            
            self.datasets.append(DatasetConfig(
                name=name,
                id=dataset_id,
                split=split,
                is_chat=is_chat,
                text_column=dataset_entry.get("text_column", None),
                messages_column=dataset_entry.get("messages_column", "messages"),
                subset=subset,
                streaming=dataset_entry.get("streaming", False),
            ))

        token_ordering_cfg = getattr(self.method_cfg, "token_ordering", None)
        assert token_ordering_cfg is not None, "diff_mining config must define token_ordering"
        methods_cfg = getattr(token_ordering_cfg, "method", None)
        assert methods_cfg is not None, "token_ordering.method must be set"
        self.ordering_methods: List[str] = [str(m) for m in list(methods_cfg)]
        assert len(self.ordering_methods) > 0, "token_ordering.method must be non-empty"
        assert len(set(self.ordering_methods)) == len(self.ordering_methods), (
            f"Duplicate entries in token_ordering.method: {self.ordering_methods}"
        )

        if "nmf" in self.ordering_methods:
            self.nmf_cfg = getattr(token_ordering_cfg, "nmf", None)
            assert self.nmf_cfg is not None, "token_ordering.nmf must be set when nmf is enabled"
            assert bool(getattr(self.nmf_cfg, "enabled", True)), (
                "token_ordering.nmf.enabled must be true when 'nmf' is included in token_ordering.method"
            )
            self.logger.info("NMF Token Topic Clustering enabled.")
            assert hasattr(self.nmf_cfg, "top_n_tokens_per_topic"), (
                "token_ordering.nmf.top_n_tokens_per_topic must be set when NMF is enabled"
            )
            self.nmf_top_n_tokens_per_topic = int(self.nmf_cfg.top_n_tokens_per_topic)
            assert self.nmf_top_n_tokens_per_topic > 0, (
                "token_ordering.nmf.top_n_tokens_per_topic must be > 0"
            )
        else:
            self.nmf_cfg = None
        
        # In-memory tensor storage (used when in_memory=true to skip disk I/O)
        self._in_memory = bool(getattr(self.method_cfg, "in_memory", False))
        self._base_logits: Dict[str, torch.Tensor] = {}
        self._logit_diffs: Dict[str, torch.Tensor] = {}
        self._log_probs: Dict[str, tuple] = {}  # (base_log_probs, ft_log_probs)
        self._attention_masks: Dict[str, torch.Tensor] = {}
        self._input_ids: Dict[str, torch.Tensor] = {}
        self._dataset_inputs: Dict[str, Dict[str, Any]] = {}  # For positions_list etc.

        self.logit_extraction_method: str
        self.logit_lens_layer_relative: float | None = None
        self.logit_lens_layer_idx: int | None = None
        self.patchscope_lens_layer_relative: float | None = None
        self.patchscope_lens_layer_idx: int | None = None
        self.logits_extractor: LogitsExtractor

        logit_extraction_cfg = getattr(self.method_cfg, "logit_extraction", None)
        self.logit_extraction_method = str(
            getattr(logit_extraction_cfg, "method", "logits")
        )
        if self.logit_extraction_method == "logits":
            self.logits_extractor = DirectLogitsExtractor()
        elif self.logit_extraction_method == "logit_lens":
            assert logit_extraction_cfg is not None
            assert hasattr(logit_extraction_cfg, "logit_lens")
            layer_rel = float(logit_extraction_cfg.logit_lens.layer)
            assert 0.0 <= layer_rel <= 1.0, (
                f"logit_extraction.logit_lens.layer must be in [0, 1], got {layer_rel}"
            )
            self.logit_lens_layer_relative = layer_rel
            self.logit_lens_layer_idx = get_layer_indices(
                self.base_model_cfg.model_id,
                [layer_rel],
            )[0]
            self.logits_extractor = LogitLensExtractor(
                layer_idx=self.logit_lens_layer_idx
            )
            self.logger.info(
                "LogitLens logit extraction: "
                f"layer {self.logit_lens_layer_relative} (absolute: {self.logit_lens_layer_idx})"
            )
        elif self.logit_extraction_method == "patchscope_lens":
            assert logit_extraction_cfg is not None
            assert hasattr(logit_extraction_cfg, "patchscope_lens")
            layer_rel = float(logit_extraction_cfg.patchscope_lens.layer)
            assert 0.0 <= layer_rel <= 1.0, (
                f"logit_extraction.patchscope_lens.layer must be in [0, 1], got {layer_rel}"
            )
            self.patchscope_lens_layer_relative = layer_rel
            self.patchscope_lens_layer_idx = get_layer_indices(
                self.base_model_cfg.model_id,
                [layer_rel],
            )[0]

            position_batch_size = int(
                getattr(logit_extraction_cfg.patchscope_lens, "position_batch_size", 256)
            )
            patch_prompt = str(
                getattr(
                    logit_extraction_cfg.patchscope_lens,
                    "patch_prompt",
                    "man -> man\n1135 -> 1135\nhello -> hello\n?",
                )
            )
            index_to_patch = int(getattr(logit_extraction_cfg.patchscope_lens, "index_to_patch", -1))

            self.logits_extractor = PatchscopeLensExtractor(
                layer_idx=self.patchscope_lens_layer_idx,
                position_batch_size=position_batch_size,
                patch_prompt=patch_prompt,
                index_to_patch=index_to_patch,
            )
            self.logger.info(
                "PatchscopeLens logit extraction: "
                f"layer {self.patchscope_lens_layer_relative} (absolute: {self.patchscope_lens_layer_idx}), "
                f"position_batch_size={position_batch_size}, index_to_patch={index_to_patch}"
            )
        else:
            raise ValueError(
                f"Unknown logit_extraction.method: '{self.logit_extraction_method}'. "
                "Expected 'logits', 'logit_lens', or 'patchscope_lens'."
            )

        # Setup results directory
        organism_path_name = cfg.organism.name
        organism_variant = getattr(cfg, "organism_variant", "default")
        
        if organism_variant != "default" and organism_variant:
             # Use a safe name format: {organism}_{variant}
             organism_path_name = f"{cfg.organism.name}_{organism_variant}"
        
        # Get sample and token counts for directory naming
        max_samples = int(self.method_cfg.max_samples)
        max_tokens_per_sample = int(self.method_cfg.max_tokens_per_sample)
        max_vocab_size = self.method_cfg.max_vocab_size
             
        # Create base results directory with sample/token counts
        # Structure: .../diffing_results/{model_name}/{organism_path_name}/diff_mining_{samples}samples_{tokens}tokens_{topk}topk[_vocab{N}][_logit_extraction_{method}[_layer_{rel}]]
        vocab_suffix = f"_vocab{max_vocab_size}" if max_vocab_size is not None else ""

        logit_extraction_suffix = ""
        logit_extraction_suffix = f"_logit_extraction_{self.logit_extraction_method}"
        if self.logit_extraction_method == "logit_lens":
            assert self.logit_lens_layer_relative is not None
            layer_str = str(self.logit_lens_layer_relative).replace(".", "p")
            logit_extraction_suffix += f"_layer_{layer_str}"
        if self.logit_extraction_method == "patchscope_lens":
            assert self.patchscope_lens_layer_relative is not None
            layer_str = str(self.patchscope_lens_layer_relative).replace(".", "p")
            logit_extraction_suffix += f"_layer_{layer_str}"
        
        method_dir_name = (
            f"diff_mining_{max_samples}samples_{max_tokens_per_sample}tokens_"
            f"{int(self.method_cfg.top_k)}topk{vocab_suffix}{logit_extraction_suffix}"
        )
        self.base_results_dir = Path(cfg.diffing.results_base_dir) / cfg.model.name / organism_path_name / method_dir_name
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectory for saved tensors (logits, diffs, masks) - at base level
        self.saved_tensors_dir = self.base_results_dir / "saved_tensors"
        self.saved_tensors_dir.mkdir(parents=True, exist_ok=True)
        
        # analysis_dir will be created in run() method with timestamp
        self.analysis_dir = None
        self.run_dir = None  # New schema: run directory

    def _get_analysis_folder_name(self) -> str:
        """
        Generate analysis folder name with timestamp and configuration parameters.
        
        Format: analysis_{timestamp}_seed{seed}_top{k}_normalized_{bool}_mode_{selection_mode}{nmf_suffix}
        Example: analysis_20260110_143045_seed42_top100_normalized_false_mode_top_k_occurring_2topics_logit_diff_magnitude_beta2_orthogonal_weight_100p0
        
        Returns:
            Folder name string
        """
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get seed from config
        seed = self.cfg.seed if hasattr(self.cfg, 'seed') else None
        seed_str = f"_seed{seed}" if seed is not None else ""
        
        # Get config parameters
        top_k = int(self.method_cfg.top_k)
        use_normalized = bool(self.method_cfg.get("use_normalized_tokens", False))
        normalized_str = "true" if use_normalized else "false"
        
        # Get token set selection mode
        selection_mode = str(self.method_cfg.token_set_selection_mode)
        
        # Build NMF suffix if enabled
        nmf_suffix = ""
        if self.nmf_cfg is not None:
            num_topics = int(self.nmf_cfg.num_topics)
            mode = str(self.nmf_cfg.mode)
            beta = self.nmf_cfg.beta
            
            # Format beta (convert float to string, replace . with p if needed)
            beta_str = str(beta).replace(".", "p") if "." in str(beta) else str(beta)
            
            # Build orthogonal suffix
            orthogonal_suffix = ""
            if bool(getattr(self.nmf_cfg, 'orthogonal', False)):
                weight = getattr(self.nmf_cfg, 'orthogonal_weight', 1.0)
                weight_str = str(weight).replace(".", "p")
                orthogonal_suffix = f"_orthogonal_weight_{weight_str}"
            
            nmf_suffix = f"_{num_topics}topics_{mode}_beta{beta_str}{orthogonal_suffix}"
        
        # Combine all parts
        folder_name = f"analysis_{timestamp}{seed_str}_top{top_k}_normalized_{normalized_str}_mode_{selection_mode}{nmf_suffix}"
        
        return folder_name

    def get_or_create_results_dir(self) -> Path:
        """
        Get or create the analysis directory for results.
        
        This method:
        1. Returns analysis_dir if already set (from run())
        2. Otherwise, finds the most recent analysis folder in base_results_dir
        3. If no analysis folders exist, creates a new one using _get_analysis_folder_name()
        
        Returns:
            Path to the analysis directory
        """
        # If analysis_dir is already set (from run()), return it
        if self.analysis_dir is not None:
            return self.analysis_dir
        
        # Look for existing analysis folders
        analysis_folders = sorted([
            d for d in self.base_results_dir.iterdir()
            if d.is_dir() and d.name.startswith("analysis_")
        ], key=lambda x: x.stat().st_mtime, reverse=True)
        
        if analysis_folders:
            # Use the most recent analysis folder
            self.analysis_dir = analysis_folders[0]
            self.logger.info(f"Using existing analysis directory: {self.analysis_dir}")
        else:
            # Create a new analysis folder
            analysis_folder_name = self._get_analysis_folder_name()
            self.analysis_dir = self.base_results_dir / analysis_folder_name
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new analysis directory: {self.analysis_dir}")
        
        return self.analysis_dir

    def _get_run_folder_name(self) -> str:
        """
        Generate run folder name with timestamp and key hyperparameters.
        
        Format: run_{timestamp}_seed{seed}_top{k}_{selection_mode}[_nmf{topics}]
        
        Returns:
            Folder name string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.cfg.seed if hasattr(self.cfg, 'seed') else None
        seed_str = f"_seed{seed}" if seed is not None else ""
        top_k = int(self.method_cfg.top_k)
        selection_mode = str(self.method_cfg.token_set_selection_mode)
        
        nmf_suffix = ""
        if self.nmf_cfg is not None:
            num_topics = int(self.nmf_cfg.num_topics)
            nmf_suffix = f"_nmf{num_topics}"
        
        return f"run_{timestamp}{seed_str}_top{top_k}_{selection_mode}{nmf_suffix}"

    def _write_run_metadata(self, run_dir: Path) -> Path:
        """
        Write run_metadata.json to the run directory.
        
        Args:
            run_dir: Path to the run directory
            
        Returns:
            Path to the written metadata file
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.cfg.seed if hasattr(self.cfg, 'seed') else None,
            "max_samples": int(self.method_cfg.max_samples),
            "max_tokens_per_sample": int(self.method_cfg.max_tokens_per_sample),
            "top_k": int(self.method_cfg.top_k),
            "token_set_selection_mode": str(self.method_cfg.token_set_selection_mode),
            "logit_extraction_method": self.logit_extraction_method,
            "logit_lens_layer": self.logit_lens_layer_relative,
            "base_model": self.base_model_cfg.model_id,
            "finetuned_model": self.finetuned_model_cfg.model_id,
            "max_vocab_size": getattr(self.method_cfg, "max_vocab_size", None),
            "ordering_methods": list(self.ordering_methods),
            "nmf_enabled": bool(self.nmf_cfg is not None),
        }
        if self.nmf_cfg is not None:
            metadata["nmf_config"] = {
                "num_topics": int(self.nmf_cfg.num_topics),
                "beta": float(self.nmf_cfg.beta),
                "mode": str(self.nmf_cfg.mode),
                "orthogonal": bool(getattr(self.nmf_cfg, 'orthogonal', False)),
            }
        
        path = run_dir / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return path

    def _get_enabled_ordering_types(self) -> List[TokenOrderingType]:
        """
        Get list of enabled ordering types based on config.

        Returns:
            List of TokenOrderingType instances
        """
        ordering_types: List[TokenOrderingType] = []

        for method in self.ordering_methods:
            if method == "topk_occurring":
                ordering_types.append(TopKOccurringOrderingType())
            elif method == "fraction_positive_diff":
                ordering_types.append(FractionPositiveDiffOrderingType())
            elif method == "nmf":
                assert self.nmf_cfg is not None, "nmf_cfg must be set when 'nmf' is enabled"
                assert bool(getattr(self.nmf_cfg, "enabled", True)), (
                    "token_ordering.nmf.enabled must be true when 'nmf' is enabled"
                )
                nmf_config = NmfOrderingConfig(
                    num_topics=int(self.nmf_cfg.num_topics),
                    beta=float(self.nmf_cfg.beta),
                    mode=str(self.nmf_cfg.mode),
                    orthogonal=bool(getattr(self.nmf_cfg, 'orthogonal', False)),
                    orthogonal_weight=float(getattr(self.nmf_cfg, 'orthogonal_weight', 1.0)),
                    top_n_tokens_per_topic=int(self.nmf_cfg.top_n_tokens_per_topic),
                )
                ordering_types.append(NmfOrderingType(nmf_config))
            else:
                raise ValueError(
                    f"Unknown token_ordering.method entry: {method!r}. "
                    "Expected one of: 'topk_occurring', 'fraction_positive_diff', 'nmf'."
                )

        return ordering_types

    def _run_ordering_types_and_write(
        self,
        run_dir: Path,
        dataset_name: str,
        stats: SharedTokenStats,
        ordering_types: List[TokenOrderingType],
    ) -> Dict[str, OrderingTypeResult]:
        """
        Run all enabled ordering types and write outputs.
        
        Args:
            run_dir: Run directory path
            dataset_name: Name of the dataset
            stats: Shared token statistics
            ordering_types: Ordering type instances (may carry collected state)
            
        Returns:
            Dict mapping ordering_type_id to OrderingTypeResult
        """
        num_tokens = max(
            int(self.method_cfg.visualization.num_tokens_to_plot),
            int(self.method_cfg.top_k),
        )
        
        results: Dict[str, OrderingTypeResult] = {}
        
        for ordering_type in ordering_types:
            self.logger.info(f"Computing ordering: {ordering_type.display_name}")
            
            result = ordering_type.compute_orderings(
                stats=stats,
                tokenizer=self.tokenizer,
                num_tokens=num_tokens,
            )
            
            if not result.orderings:
                self.logger.warning(f"No orderings produced for {ordering_type.ordering_type_id}")
                continue
            
            # Write ordering type directory
            ordering_type_dir = run_dir / result.ordering_type_id
            ordering_type_dir.mkdir(parents=True, exist_ok=True)
            
            # Write metadata.json
            write_ordering_type_metadata(ordering_type_dir, result)
            
            # Write dataset orderings
            dataset_dir = ordering_type_dir / dataset_name
            write_dataset_orderings(dataset_dir, dataset_name, result.orderings)
            
            self.logger.info(
                f"  Wrote {len(result.orderings)} ordering(s) for {ordering_type.ordering_type_id}"
            )
            
            results[result.ordering_type_id] = result
        
        return results

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
    def _prepare_dataset_tensors(self, dataset_cfg: DatasetConfig) -> Dict[str, Any]:
        """
        Prepare input tensors for a single dataset.
        
        Args:
            dataset_cfg: Dataset configuration
            
        Returns:
            Dict containing 'input_ids' and 'attention_mask' tensors
        """
        max_samples = int(self.method_cfg.max_samples)
        max_tokens = int(self.method_cfg.max_tokens_per_sample)
        pre_assistant_k = int(self.method_cfg.pre_assistant_k)
        debug_print_samples = getattr(self.method_cfg, "debug_print_samples", None)
        seed = self.cfg.seed if hasattr(self.cfg, "seed") else None

        return prepare_dataset_tensors(
            dataset_cfg=dataset_cfg,
            tokenizer=self.tokenizer,
            max_samples=max_samples,
            max_tokens=max_tokens,
            pre_assistant_k=pre_assistant_k,
            debug_print_samples=debug_print_samples,
            seed=seed,
            logger=self.logger,
        )

    @torch.no_grad()
    def compute_stats_from_logits(
        self, 
        dataset_cfg: DatasetConfig,
        attention_mask: torch.Tensor,
        logit_diff: torch.Tensor,
        ordering_types: List[TokenOrderingType],
    ) -> CoreAnalysisResult:
        """
        Core analysis for one dataset.

        Delegates the pure computation to `core_analysis.compute_stats_from_logits` and performs
        side effects (KDE plotting, global stats saving) here.
        """
        batch_size = int(self.method_cfg.batch_size)
        max_tokens = int(self.method_cfg.max_tokens_per_sample)
        max_samples = int(self.method_cfg.max_samples)
        top_k = int(self.method_cfg.top_k)
        ignore_padding = bool(self.method_cfg.ignore_padding)

        per_token_analysis_cfg = getattr(self.method_cfg, "per_token_analysis", None)
        positional_kde_cfg = getattr(self.method_cfg, "positional_kde", None)

        result = _compute_stats_from_logits(
            dataset_cfg=dataset_cfg,
            attention_mask=attention_mask,
            logit_diff=logit_diff,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_samples=max_samples,
            top_k=top_k,
            ignore_padding=ignore_padding,
            per_token_analysis_cfg=per_token_analysis_cfg,
            positional_kde_cfg=positional_kde_cfg,
            ordering_types=ordering_types,
            tokenizer=self.tokenizer,
            device=self.device,
            logger=self.logger,
        )

        kde_data = result.kde_data
        if kde_data is not None:
            self.logger.info("Generating positional KDE plots...")
            plot_positional_kde(
                kde_data["position_logit_diffs"],
                dataset_cfg.name,
                self.analysis_dir,
                int(kde_data["pos_kde_num_positions"]),
                int(result.shared_stats.num_samples),
                int(kde_data["top_k"]),
            )

        self.logger.info("Saving global token statistics (entire vocabulary)...")
        self._save_global_token_statistics(
            dataset_cfg.name,
            result.shared_stats.sum_logit_diff,
            result.shared_stats.count_positive,
            int(result.shared_stats.num_samples),
            int(result.shared_stats.total_positions),
        )

        return result

    def _save_global_token_statistics(
        self,
        dataset_name: str,
        global_diff_sum: torch.Tensor,
        global_pos_count: torch.Tensor,
        num_samples: int,
        total_positions: int,
        output_dir: Optional[Path] = None,
        token_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Save global token statistics (sum logit diff, positive count) for a token set.
        
        Args:
            dataset_name: Name of the dataset
            global_diff_sum: Tensor of shape [vocab_size] containing sum of logit diffs
            global_pos_count: Tensor of shape [vocab_size] containing count of non-negative diffs
            num_samples: Number of samples processed
            total_positions: Total number of valid token positions processed
            output_dir: Directory to write outputs (defaults to analysis_dir)
            token_ids: Optional list of token IDs to restrict stats (defaults to full vocab)
        """
        output_dir = output_dir or self.analysis_dir
        output_file = output_dir / f"{dataset_name}_global_token_stats.json"
        
        # Ensure CPU and python native types
        assert global_diff_sum.shape == global_pos_count.shape
        global_diff_sum = global_diff_sum.cpu()
        global_pos_count = global_pos_count.cpu()
        vocab_size = len(global_diff_sum)
        
        if token_ids is None:
            all_ids = list(range(vocab_size))
        else:
            all_ids = list(dict.fromkeys(token_ids))
            assert max(all_ids) < vocab_size
        
        self.logger.info(f"Formatting global statistics for {len(all_ids)} tokens...")
        
        # Get string representations for all tokens
        # We assume the model's output vocab corresponds to tokenizer IDs 0..N-1
        all_tokens = self.tokenizer.convert_ids_to_tokens(all_ids)
        
        # Build the list of stats
        stats_list = []
        for token_id, token_str in zip(all_ids, all_tokens):
            stats_list.append({
                "token": token_str,
                "token_id": token_id,
                "sum_logit_diff": float(global_diff_sum[token_id]),
                "count_positive": int(global_pos_count[token_id])
            })
            
        final_output = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
            "total_positions_analyzed": total_positions,
            "num_unique_tokens": len(all_ids),
            "global_token_stats": stats_list
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved global token statistics to {output_file}")

    def compute_sequence_likelihood_ratios(
        self,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """
        Compute sequence likelihood ratios using sliding windows over token sequences.
        
        For each sliding window of tokens, computes the sum of log probabilities under
        both base and fine-tuned models, then calculates the log-likelihood ratio
        (LLR = logprob_ft - logprob_base).
        
        Windows are only created within valid (non-padding) positions.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary containing metadata and sorted list of chunks with LLR values
        """
        # Get config parameters
        slr_cfg = self.method_cfg.sequence_likelihood_ratio
        window_size = int(slr_cfg.window_size)
        step = int(slr_cfg.step)
        top_k_print = int(slr_cfg.top_k_print)
        
        self.logger.info(f"Computing sequence likelihood ratios for {dataset_name}...")
        self.logger.info(f"  Window size: {window_size}, Step: {step}")
        
        # Check if we have in-memory tensors
        if dataset_name in self._log_probs:
            # Use in-memory tensors
            self.logger.info("  Using in-memory log probabilities")
            base_log_probs, ft_log_probs = self._log_probs[dataset_name]
            input_ids = self._input_ids[dataset_name]
            attention_mask = self._attention_masks[dataset_name]
        else:
            # Load saved tensors from disk
            log_probs_dir = self.saved_tensors_dir / "log_probs"
            input_ids_dir = self.saved_tensors_dir / "input_ids"
            masks_dir = self.saved_tensors_dir / "attention_masks"
            
            base_log_probs_path = log_probs_dir / f"{dataset_name}_base_log_probs.pt"
            ft_log_probs_path = log_probs_dir / f"{dataset_name}_ft_log_probs.pt"
            input_ids_path = input_ids_dir / f"{dataset_name}_input_ids.pt"
            mask_path = masks_dir / f"{dataset_name}_attention_mask.pt"
            
            # Check if files exist
            if not base_log_probs_path.exists() or not ft_log_probs_path.exists():
                self.logger.warning(
                    f"Log probability files not found for {dataset_name}. "
                    "Please re-run preprocessing with sequence_likelihood_ratio.enabled=true"
                )
                return None
            
            if not input_ids_path.exists():
                self.logger.warning(f"Input IDs not found for {dataset_name}. Please re-run preprocessing.")
                return None
            
            # Load tensors
            base_log_probs = torch.load(base_log_probs_path, map_location="cpu")  # [num_samples, seq_len-1]
            ft_log_probs = torch.load(ft_log_probs_path, map_location="cpu")      # [num_samples, seq_len-1]
            input_ids = torch.load(input_ids_path, map_location="cpu")            # [num_samples, seq_len] or sliced
            attention_mask = torch.load(mask_path, map_location="cpu")            # [num_samples, seq_len]
        
        num_samples, seq_len = base_log_probs.shape
        self.logger.info(f"  Loaded {num_samples} samples, seq_len={seq_len}")
        
        # Adjust attention mask to match log_probs shape (seq_len - 1)
        # We use mask for positions that predict valid next tokens
        # If original mask is [1,1,1,1,0,0] for seq_len=6,
        # log_probs has seq_len-1=5 positions predicting tokens 1,2,3,4,5
        # We need valid_mask where both current and next position have valid tokens
        if attention_mask.shape[1] > seq_len:
            # Original full attention mask - slice to match log_probs
            # Position t in log_probs predicts token at t+1, so we need mask[:, 1:seq_len+1]
            valid_mask = attention_mask[:, 1:seq_len+1]  # [num_samples, seq_len]
        else:
            valid_mask = attention_mask[:, :seq_len]
        
        # Collect all chunks
        chunks = []
        chunk_idx = 0
        
        for sample_idx in range(num_samples):
            # Get valid length for this sample
            sample_mask = valid_mask[sample_idx]
            valid_len = sample_mask.sum().item()
            
            if valid_len < window_size:
                # Sample too short for even one window
                continue
            
            # Generate sliding windows
            for start_pos in range(0, valid_len - window_size + 1, step):
                end_pos = start_pos + window_size
                
                # Sum log probabilities over the window
                window_base_logprob = base_log_probs[sample_idx, start_pos:end_pos].sum().item()
                window_ft_logprob = ft_log_probs[sample_idx, start_pos:end_pos].sum().item()
                logprob_diff = window_ft_logprob - window_base_logprob
                
                # Get token IDs for this window
                # input_ids contains target tokens (the tokens being predicted)
                # For log_probs position t, we predicted input_ids[t] (which is the t+1 token from original)
                window_token_ids = input_ids[sample_idx, start_pos:end_pos].tolist()
                
                # Decode to text
                window_text = self.tokenizer.decode(window_token_ids)
                
                chunks.append({
                    "index": chunk_idx,
                    "sample_idx": sample_idx,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "token_ids": window_token_ids,
                    "text": window_text,
                    "logprob_base": window_base_logprob,
                    "logprob_ft": window_ft_logprob,
                    "logprob_diff": logprob_diff,
                })
                chunk_idx += 1
        
        self.logger.info(f"  Generated {len(chunks)} chunks")
        
        # Sort by logprob_diff (descending - highest LLR first)
        chunks_sorted = sorted(chunks, key=lambda x: x["logprob_diff"], reverse=True)
        
        # Add rank to each chunk
        for rank, chunk in enumerate(chunks_sorted, start=1):
            chunk["rank"] = rank
        
        # Build output
        result = {
            "metadata": {
                "dataset_name": dataset_name,
                "window_size": window_size,
                "step": step,
                "base_model": self.base_model_cfg.model_id,
                "ft_model": self.finetuned_model_cfg.model_id,
                "num_chunks": len(chunks_sorted),
                "num_samples": num_samples,
            },
            "chunks": chunks_sorted,
        }
        
        # Print top chunks to terminal
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"SEQUENCE LIKELIHOOD RATIO ANALYSIS: {dataset_name}")
        self.logger.info(f"Window size: {window_size} tokens, Step: {step}")
        self.logger.info("=" * 80)
        self.logger.info("")
        self.logger.info(f"{'Rank':<6} {'Sample':<8} {'LLR':<10} {'Text'}")
        self.logger.info("-" * 80)
        
        for chunk in chunks_sorted[:top_k_print]:
            # Truncate text for display
            text_display = chunk["text"][:50].replace("\n", "\\n")
            if len(chunk["text"]) > 50:
                text_display += "..."
            
            self.logger.info(
                f"{chunk['rank']:<6} {chunk['sample_idx']:<8} "
                f"{chunk['logprob_diff']:+.4f}   \"{text_display}\""
            )
        
        self.logger.info("-" * 80)
        self.logger.info(f"Showing top {min(top_k_print, len(chunks_sorted))} of {len(chunks_sorted)} chunks")
        self.logger.info("")
        
        # Save to JSON
        output_path = self.analysis_dir / f"{dataset_name}_sequence_likelihood_ratios.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved sequence likelihood ratios to {output_path}")
        
        return result

    def _build_occurrence_results_for_token_ids(
        self,
        dataset_cfg: DatasetConfig,
        token_ids: List[int],
        global_pos_token_counts: torch.Tensor,
        global_neg_token_counts: torch.Tensor,
        total_positions: int,
        num_samples: int,
        top_k: int,
        max_tokens: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Build occurrence-rate results restricted to a token subset.

        Args:
            dataset_cfg: Dataset configuration
            token_ids: Token IDs to include
            global_pos_token_counts: Vector of positive top-K counts (vocab-sized)
            global_neg_token_counts: Vector of negative top-K counts (vocab-sized)
            total_positions: Total number of valid positions
            num_samples: Number of samples
            top_k: Top-K used during diffing
            max_tokens: Max tokens per sample
            batch_size: Batch size

        Returns:
            Results dict with top_positive/top_negative lists
        """
        token_ids = list(dict.fromkeys(token_ids))
        all_tokens = []
        for token_id in token_ids:
            pos_count = int(global_pos_token_counts[token_id].item())
            neg_count = int(global_neg_token_counts[token_id].item())
            token_str = self.tokenizer.decode([token_id])
            all_tokens.append({
                "token_id": token_id,
                "token_str": token_str,
                "count_positive": pos_count,
                "count_negative": neg_count,
                "positive_occurrence_rate": (pos_count / total_positions) * 100
                if total_positions > 0
                else 0.0,
                "negative_occurrence_rate": (neg_count / total_positions) * 100
                if total_positions > 0
                else 0.0,
            })

        if not all_tokens:
            return {
                "dataset_id": dataset_cfg.id,
                "dataset_name": dataset_cfg.name,
                "total_positions": total_positions,
                "num_samples": num_samples,
                "top_k": top_k,
                "unique_tokens": 0,
                "top_positive": [],
                "top_negative": [],
                "metadata": {
                    "base_model": self.base_model_cfg.model_id,
                    "finetuned_model": self.finetuned_model_cfg.model_id,
                    "max_tokens_per_sample": max_tokens,
                    "batch_size": batch_size,
                },
            }

        pos_rates = torch.tensor([t["positive_occurrence_rate"] for t in all_tokens])
        neg_rates = torch.tensor([t["negative_occurrence_rate"] for t in all_tokens])

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

        return {
            "dataset_id": dataset_cfg.id,
            "dataset_name": dataset_cfg.name,
            "total_positions": total_positions,
            "num_samples": num_samples,
            "top_k": top_k,
            "unique_tokens": len(all_tokens),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "metadata": {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
                "max_tokens_per_sample": max_tokens,
                "batch_size": batch_size,
            },
        }

    @staticmethod
    def _sanitize_token_name(token_str: str) -> str:
        """
        Sanitize token name for use in filenames.
        
        Replaces special characters with underscores and limits length.
        
        Args:
            token_str: Original token string
            
        Returns:
            Sanitized string safe for filenames
        """
        # Replace special characters with underscores
        sanitized = re.sub(r'[^\w\-]', '_', token_str)
        # Limit length to 50 characters
        return sanitized[:50]

    def run_token_relevance(
        self,
        all_ordering_results: Dict[str, Dict[str, OrderingTypeResult]],
    ) -> None:
        """
        Grade top-K positive tokens for relevance to finetuning domain.
        
        This method iterates over all produced orderings (from all ordering types)
        and grades the tokens in each ordering. Results are written as 
        `<ordering_id>_eval.json` next to the ordering file.
        
        Args:
            all_ordering_results: Dict mapping dataset_name -> ordering_type_id -> OrderingTypeResult
        """
        cfg = self.method_cfg.token_relevance
        
        if not cfg.enabled:
            return
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("RUNNING TOKEN RELEVANCE GRADING")
        logger.info("=" * 80)
        
        assert self.run_dir is not None, "run_dir must be set before running token relevance"
        
        overwrite = bool(cfg.overwrite)
        agreement_mode = str(cfg.agreement)
        k_candidate = int(cfg.k_candidate_tokens)
        token_relevance_batch_size = int(getattr(cfg, "batch_size", 20))
        assert token_relevance_batch_size >= 1
        include_translation = bool(getattr(cfg, "include_translation", False))
        
        organism_cfg = self.cfg.organism
        assert hasattr(organism_cfg, "description_long"), (
            "Organism config must have 'description_long' for token relevance grading"
        )
        description = str(organism_cfg.description_long)
        logger.info(f"Using organism description: {description[:100]}...")
        
        grader_cfg = cfg.grader
        grader = TokenRelevanceGrader(
            grader_model_id=str(grader_cfg.model_id),
            base_url=str(grader_cfg.base_url),
            api_key_path=str(grader_cfg.api_key_path),
        )
        logger.info(f"Initialized grader: {grader_cfg.model_id}")
        
        freq_cfg = cfg.frequent_tokens
        has_training_dataset = hasattr(organism_cfg, "training_dataset") and (
            organism_cfg.training_dataset is not None
        )
        
        if has_training_dataset:
            training_dataset_id = str(organism_cfg.training_dataset.id)
            logger.info(f"Computing frequent tokens from training dataset: {training_dataset_id}")
            
            training_is_chat = False
            training_dataset_cfg = organism_cfg.training_dataset
            if hasattr(training_dataset_cfg, 'is_chat'):
                training_is_chat = bool(training_dataset_cfg.is_chat)
            elif isinstance(training_dataset_cfg, dict):
                training_is_chat = bool(training_dataset_cfg.get('is_chat', False))
            
            frequent_tokens = _compute_frequent_tokens(
                dataset_name=training_dataset_id,
                tokenizer=self.tokenizer,
                splits=["train"],
                num_tokens=int(freq_cfg.num_tokens),
                min_count=int(freq_cfg.min_count),
                is_chat=training_is_chat,
                subset=None,
            )
            logger.info(f"Found {len(frequent_tokens)} frequent tokens")
        else:
            frequent_tokens = []
            logger.info("No training dataset available, using empty frequent tokens list")
        
        # Iterate over all ordering results (in-memory)
        for dataset_name, ordering_type_results in all_ordering_results.items():
            for ordering_type_id, type_result in ordering_type_results.items():
                logger.info("")
                logger.info(f"Processing {ordering_type_id}/{dataset_name}")
                
                dataset_dir = self.run_dir / ordering_type_id / dataset_name
                
                for ordering in type_result.orderings:
                    ordering_id = ordering.ordering_id
                    display_label = ordering.display_label
                    
                    eval_path = dataset_dir / f"{ordering_id}_eval.json"
                    if (not overwrite) and eval_path.exists():
                        logger.info(f"Eval exists for {ordering_type_id}/{dataset_name}/{ordering_id}, skipping")
                        continue
                    
                    tokens = ordering.tokens
                    if not tokens:
                        logger.warning(f"No tokens in ordering: {ordering_id}")
                        continue
                    
                    k_tokens = min(k_candidate, len(tokens))
                    candidate_tokens = [t.token_str for t in tokens[:k_tokens]]
                    token_weights = [t.ordering_value for t in tokens[:k_tokens]]
                    
                    logger.info(f"Grading {k_tokens} tokens for {ordering_type_id}/{dataset_name}/{display_label}")
                    num_batches = (k_tokens + token_relevance_batch_size - 1) // token_relevance_batch_size
                    logger.info(
                        f"Token relevance: {num_batches} batch(es) of size {token_relevance_batch_size}"
                    )
                    
                    trivial_hits = sum(1 for t in candidate_tokens if t in frequent_tokens)
                    trivial_percentage = trivial_hits / float(len(candidate_tokens)) if candidate_tokens else 0.0
                    
                    permutations = int(grader_cfg.permutations)
                    batches: List[List[str]] = [
                        candidate_tokens[start : start + token_relevance_batch_size]
                        for start in range(0, k_tokens, token_relevance_batch_size)
                    ]
                    assert len(batches) == num_batches
                    for b in batches:
                        assert len(b) >= 1

                    if include_translation:

                        async def _run_all_batches_with_translation() -> List[
                            Tuple[List[str], List[str], List[List[str]], List[str]]
                        ]:
                            tasks = [
                                grader.grade_with_translation_async(
                                    description=description,
                                    frequent_tokens=frequent_tokens,
                                    candidate_tokens=batch_tokens,
                                    permutations=permutations,
                                    max_tokens=int(grader_cfg.max_tokens),
                                )
                                for batch_tokens in batches
                            ]
                            return list(await asyncio.gather(*tasks))

                        batch_results = asyncio.run(_run_all_batches_with_translation())
                        final_labels: List[str] = []
                        translations: List[str] = []
                        raw_responses: List[str] = []
                        for batch_tokens, batch_out in zip(batches, batch_results):
                            (
                                batch_majority,
                                batch_translations,
                                batch_permutations,
                                batch_raw,
                            ) = batch_out
                            assert len(batch_majority) == len(batch_tokens)
                            assert len(batch_translations) == len(batch_tokens)

                            if agreement_mode == "majority":
                                batch_final = batch_majority
                            else:
                                n = len(batch_tokens)
                                batch_final = [
                                    "RELEVANT"
                                    if all(
                                        run[i] == "RELEVANT" for run in batch_permutations
                                    )
                                    else "IRRELEVANT"
                                    for i in range(n)
                                ]

                            assert len(batch_final) == len(batch_tokens)
                            final_labels.extend(batch_final)
                            translations.extend(batch_translations)
                            raw_responses.extend(batch_raw)

                        assert len(final_labels) == len(candidate_tokens)
                        assert len(translations) == len(candidate_tokens)
                    else:

                        async def _run_all_batches() -> List[
                            Tuple[List[str], List[List[str]], List[str]]
                        ]:
                            tasks = [
                                grader.grade_async(
                                    description=description,
                                    frequent_tokens=frequent_tokens,
                                    candidate_tokens=batch_tokens,
                                    permutations=permutations,
                                    max_tokens=int(grader_cfg.max_tokens),
                                )
                                for batch_tokens in batches
                            ]
                            return list(await asyncio.gather(*tasks))

                        batch_results = asyncio.run(_run_all_batches())
                        final_labels = []
                        raw_responses = []
                        translations = None
                        for batch_tokens, batch_out in zip(batches, batch_results):
                            batch_majority, batch_permutations, batch_raw = batch_out
                            assert len(batch_majority) == len(batch_tokens)

                            if agreement_mode == "majority":
                                batch_final = batch_majority
                            else:
                                n = len(batch_tokens)
                                batch_final = [
                                    "RELEVANT"
                                    if all(
                                        run[i] == "RELEVANT" for run in batch_permutations
                                    )
                                    else "IRRELEVANT"
                                    for i in range(n)
                                ]

                            assert len(batch_final) == len(batch_tokens)
                            final_labels.extend(batch_final)
                            raw_responses.extend(batch_raw)

                        assert len(final_labels) == len(candidate_tokens)
                    
                    relevant_fraction = sum(lbl == "RELEVANT" for lbl in final_labels) / float(len(final_labels))
                    total_w = sum(abs(w) for w in token_weights)
                    relevant_w = sum(abs(w) for lbl, w in zip(final_labels, token_weights) if lbl == "RELEVANT")
                    weighted_percentage = relevant_w / total_w if total_w > 0 else 0.0
                    
                    write_ordering_eval(
                        output_dir=dataset_dir,
                        ordering_id=ordering_id,
                        labels=final_labels,
                        metadata={
                            "ordering_type_id": ordering_type_id,
                            "dataset_name": dataset_name,
                            "display_label": display_label,
                            "tokens": candidate_tokens,
                            "percentage": relevant_fraction,
                            "trivial_percentage": trivial_percentage,
                            "weighted_percentage": float(weighted_percentage),
                            "grader_responses": raw_responses,
                            "batch_size": int(token_relevance_batch_size),
                            **(
                                {"translations": translations}
                                if include_translation
                                else {}
                            ),
                        },
                    )
                    logger.info(f"✓ Eval saved: {eval_path}")
                    logger.info(f"  Relevance: {relevant_fraction*100:.1f}%, Weighted: {weighted_percentage*100:.1f}%")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Token relevance grading completed!")
        logger.info("=" * 80)

    def run_compute_plots(
        self,
        all_ordering_results: Dict[str, Dict[str, OrderingTypeResult]],
    ) -> None:
        """
        Generate plots for all orderings.
        
        Creates scatter plots and bar charts for each ordering, saved next to
        the ordering JSON files.
        
        Args:
            all_ordering_results: Dict mapping dataset_name -> ordering_type_id -> OrderingTypeResult
        """
        from .ordering_plots import plot_ordering_scatter, plot_ordering_bar_chart
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("GENERATING PLOTS")
        logger.info("=" * 80)
        
        assert self.run_dir is not None, "run_dir must be set before generating plots"
        
        viz_cfg = self.method_cfg.visualization
        
        for dataset_name, ordering_type_results in all_ordering_results.items():
            for ordering_type_id, type_result in ordering_type_results.items():
                logger.info(f"Plotting {ordering_type_id}/{dataset_name}")
                
                dataset_dir = self.run_dir / ordering_type_id / dataset_name
                
                for ordering in type_result.orderings:
                    if not ordering.tokens:
                        continue
                    
                    # Scatter plot: ordering_value (x) vs avg_logit_diff (y)
                    scatter_path = dataset_dir / f"{ordering.ordering_id}_scatter.png"
                    plot_ordering_scatter(
                        ordering=ordering,
                        x_label=type_result.x_axis_label,
                        y_label=type_result.y_axis_label,
                        output_path=scatter_path,
                        figure_width=int(viz_cfg.figure_width),
                        figure_dpi=int(viz_cfg.figure_dpi),
                    )
                    
                    # Bar chart: top tokens
                    bar_path = dataset_dir / f"{ordering.ordering_id}_bar.png"
                    plot_ordering_bar_chart(
                        ordering=ordering,
                        output_path=bar_path,
                        num_tokens=int(viz_cfg.num_tokens_to_plot),
                        figure_width=int(viz_cfg.figure_width),
                        figure_height=int(viz_cfg.figure_height),
                        figure_dpi=int(viz_cfg.figure_dpi),
                    )
        
        logger.info("✓ Plot generation completed!")

    def run(self) -> None:
        """
        Main execution method for diff mining.
        Runs during the 'diffing' stage (after preprocessing).
        """
        self.logger.info("=" * 80)
        self.logger.info("DIFF MINING")
        self.logger.info("=" * 80)
        
        # Create run directory with timestamp and config (new schema)
        run_folder_name = self._get_run_folder_name()
        self.run_dir = self.base_results_dir / run_folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Run directory: {self.run_dir}")
        
        # Write run metadata
        self._write_run_metadata(self.run_dir)
        
        # Also create analysis_dir pointing to run_dir for backward compat
        self.analysis_dir = self.run_dir
        self.logger.info(f"Analysis results will be saved to: {self.analysis_dir}")

        # Check if we have in-memory tensors (from same-instance preprocess call)
        use_in_memory = bool(self._logit_diffs)
        
        if use_in_memory:
            self.logger.info("Using in-memory tensors from preprocessing (no disk I/O)")
        else:
            # Define directories using new structure
            diffs_dir = self.saved_tensors_dir / "logit_diffs"
            masks_dir = self.saved_tensors_dir / "attention_masks"
            
            # Check preprocessing has run:
            if not diffs_dir.exists() or not any(diffs_dir.iterdir()):
                error_msg = (
                    f"No logit diff tensors found in {diffs_dir}. "
                    "Please run with pipeline.mode=preprocessing first."
                )
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

        self.logger.info("PHASE: Analysis & Diffing (Using pre-computed diffs)")
        
        all_ordering_results: Dict[str, Dict[str, OrderingTypeResult]] = {}
        
        for idx, dataset_cfg in enumerate(self.datasets, 1):
            self.logger.info("")
            self.logger.info(f"[{idx}/{len(self.datasets)}] Analyzing dataset: {dataset_cfg.name}")
            
            if use_in_memory:
                # Use in-memory tensors
                if dataset_cfg.name not in self._logit_diffs:
                    self.logger.warning(f"No in-memory diff for {dataset_cfg.name}, skipping")
                    continue
                logit_diff = self._logit_diffs[dataset_cfg.name]
                attention_mask = self._attention_masks[dataset_cfg.name]
                self.logger.info(f"Using in-memory logit diff: {logit_diff.shape}")
            else:
                # Load from disk
                diffs_dir = self.saved_tensors_dir / "logit_diffs"
                masks_dir = self.saved_tensors_dir / "attention_masks"
                
                diff_path = diffs_dir / f"{dataset_cfg.name}_logit_diff.pt"
                
                if not diff_path.exists():
                    raise FileNotFoundError(
                        f"Diff file not found for {dataset_cfg.name}: {diff_path}. "
                        "Preprocessing may have failed or been interrupted."
                    )

                self.logger.info(f"Loading logit diff from {diff_path}...")
                logit_diff = torch.load(diff_path, map_location="cpu")
                
                # Load attention mask
                mask_path = masks_dir / f"{dataset_cfg.name}_attention_mask.pt"
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Attention mask not found for {dataset_cfg.name}: {mask_path}. "
                        "Preprocessing may have failed or been interrupted."
                    )
                     
                attention_mask = torch.load(mask_path, map_location="cpu")

            ordering_types = self._get_enabled_ordering_types()
            result = self.compute_stats_from_logits(
                dataset_cfg=dataset_cfg,
                attention_mask=attention_mask,
                logit_diff=logit_diff,  # Pass pre-computed diff
                ordering_types=ordering_types,
            )
            stats = result.shared_stats
            self.logger.info("Running token ordering types...")
            dataset_ordering_results = self._run_ordering_types_and_write(
                run_dir=self.run_dir,
                dataset_name=dataset_cfg.name,
                stats=stats,
                ordering_types=ordering_types,
            )
            all_ordering_results[dataset_cfg.name] = dataset_ordering_results
            self.logger.info(f"Wrote {len(dataset_ordering_results)} ordering type(s)")
            self.logger.info(f"✓ [{idx}/{len(self.datasets)}] Completed dataset: {dataset_cfg.name}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("✓ Diff mining completed successfully!")
        self.logger.info(f"✓ Results saved to: {self.analysis_dir}")
        self.logger.info("=" * 80)
        
        # Run token relevance grading if enabled
        if hasattr(self.method_cfg, 'token_relevance') and self.method_cfg.token_relevance.enabled:
            self.run_token_relevance(all_ordering_results)
        
        # Generate plots for all orderings
        self.run_compute_plots(all_ordering_results)

    def preprocess(self, delete_raw: bool = True) -> None:
        """
        Preprocessing Phase: Data Prep, Model Inference, and Diff Computation.
        Saves {dataset}_logit_diff.pt and optionally deletes raw logits.
        """
        self.logger.info("=" * 80)
        self.logger.info("DIFF MINING: PREPROCESSING")
        self.logger.info("=" * 80)

        # Phase 0: Data Preparation (Tokenize all datasets)
        self.logger.info("PHASE 0: Data Preparation")
        
        # Setup output directories in saved_tensors
        logits_dir = self.saved_tensors_dir # Raw logits go to root of saved_tensors
        diffs_dir = self.saved_tensors_dir / "logit_diffs"
        masks_dir = self.saved_tensors_dir / "attention_masks"
        input_ids_dir = self.saved_tensors_dir / "input_ids"
        log_probs_dir = self.saved_tensors_dir / "log_probs"
        
        diffs_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        input_ids_dir.mkdir(parents=True, exist_ok=True)
        log_probs_dir.mkdir(parents=True, exist_ok=True)
 
        seed = self.cfg.seed if hasattr(self.cfg, "seed") else None
        dataset_inputs: Dict[str, Dict[str, Any]] = prepare_all_dataset_inputs_and_save(
            self.datasets,
            tokenizer=self.tokenizer,
            method_cfg=self.method_cfg,
            seed=seed,
            masks_dir=masks_dir,
            input_ids_dir=input_ids_dir,
            logger=self.logger,
        )

        # Phase 1: Base Model Inference (logits extraction mode)
        self.logger.info("")
        self.logger.info("PHASE 1: Base Model Inference")
        self.logger.info(f"Loading base model: {self.base_model_cfg.model_id}")
        _ = self.base_model # Trigger load
        
        batch_size = int(self.method_cfg.batch_size)
        
        self._base_logits = infer_and_store_logits_for_all_datasets(
            self.datasets,
            dataset_inputs,
            model=self.base_model,
            logits_extractor=self.logits_extractor,
            batch_size=batch_size,
            device=self.device,
            logger=self.logger,
            in_memory=bool(self._in_memory),
            logits_dir=logits_dir,
            logits_suffix="base",
        )
            
        self.clear_base_model()

        # Phase 2: Finetuned Model Inference
        self.logger.info("")
        self.logger.info("PHASE 2: Finetuned Model Inference")
        self.logger.info(f"Loading finetuned model: {self.finetuned_model_cfg.model_id}")
        _ = self.finetuned_model # Trigger load
        
        if self._in_memory:
            self.logger.info("In-memory mode: computing diffs inline during finetuned inference.")
            logit_diffs, log_probs, attention_masks, input_ids_out = infer_finetuned_and_compute_diffs_in_memory(
                self.datasets,
                dataset_inputs,
                finetuned_model=self.finetuned_model,
                logits_extractor=self.logits_extractor,
                base_logits_by_dataset=self._base_logits,
                batch_size=batch_size,
                device=self.device,
                method_cfg=self.method_cfg,
                logger=self.logger,
            )
            self._logit_diffs = logit_diffs
            self._log_probs = log_probs
            self._attention_masks = attention_masks
            self._input_ids = input_ids_out
            self._dataset_inputs = {k: dataset_inputs[k] for k in logit_diffs.keys()}
        else:
            _ = infer_and_store_logits_for_all_datasets(
                self.datasets,
                dataset_inputs,
                model=self.finetuned_model,
                logits_extractor=self.logits_extractor,
                batch_size=batch_size,
                device=self.device,
                logger=self.logger,
                in_memory=False,
                logits_dir=logits_dir,
                logits_suffix="finetuned",
            )

        self.clear_finetuned_model()
        
        if self._in_memory:
            self.logger.info("In-memory mode: diffs already computed. Skipping disk-based Phase 3.")
            self.logger.info("Preprocessing phase complete.")
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info("Cleared GPU cache before analysis phase.")
            return

        self.logger.info("")
        self.logger.info("Computing and Saving Logit Diffs and Log Probabilities...")
        compute_and_save_disk_diffs(
            self.datasets,
            dataset_inputs,
            logits_dir=logits_dir,
            diffs_dir=diffs_dir,
            log_probs_dir=log_probs_dir,
            input_ids_dir=input_ids_dir,
            method_cfg=self.method_cfg,
            delete_raw=delete_raw,
            logger=self.logger,
        )

        self.logger.info("Preprocessing phase complete.")
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Cleared GPU cache before analysis phase.")

    def visualize(self) -> None:
        """
        Create Streamlit visualization for diff mining results.

        Returns:
            Streamlit component displaying occurrence rankings and interactive heatmap
        """
        visualize(self)

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available diff mining results.

        Detects both old schema (analysis_* folders with *_occurrence_rates.json)
        and new schema (run_* folders with <ordering_type>/*/orderings.json).

        Returns:
            Dict mapping {model: {organism_variant_name: path_to_results}}
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
                
                # Look for method directories: diff_mining_{samples}samples_{tokens}tokens...
                for method_dir in organism_dir.iterdir():
                    if method_dir.is_dir() and method_dir.name.startswith("diff_mining"):
                        # Check for new schema first: run_* folders with orderings.json
                        run_folders = sorted(
                            [d for d in method_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                            key=lambda x: x.stat().st_mtime,
                            reverse=True
                        )
                        for run_folder in run_folders:
                            # Check for new schema: ordering type directories with orderings.json
                            has_orderings = any(
                                list(run_folder.glob("*/*/orderings.json"))
                            )
                            if has_orderings:
                                results[model_name][organism_name] = str(run_folder)
                                break
                        
                        if organism_name in results.get(model_name, {}):
                            continue  # Found new schema, skip old
                        
                        # Fall back to old schema: analysis_* folders with *_occurrence_rates.json
                        analysis_folders = sorted(
                            [d for d in method_dir.iterdir() if d.is_dir() and d.name.startswith("analysis_")],
                            key=lambda x: x.stat().st_mtime,
                            reverse=True
                        )
                        for analysis_folder in analysis_folders:
                            if list(analysis_folder.glob("*_occurrence_rates.json")):
                                results[model_name][organism_name] = str(analysis_folder)
                                break

        return results

    def get_agent(self) -> DiffingMethodAgent:
        from .agents import DiffMiningAgent
        return DiffMiningAgent(cfg=self.cfg)



