"""
Token ordering abstractions for logit_diff_topk_occurring.

This module defines ordering types that produce sets of orderings from
per-dataset token statistics. Each ordering type produces one or more
orderings (e.g., NMF produces one per topic).
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase

import torch

from src.utils.configs import DatasetConfig


@dataclass(frozen=True)
class OrderingBatchCache:
    """
    Per-batch cache of already-computed tensors for ordering collection.

    Tensor shapes:
    - top_k_pos_indices/top_k_neg_indices: [batch, seq, top_k] (dtype long)
    - top_k_pos_values/top_k_neg_values: [batch, seq, top_k] (dtype float)
    - attention_mask: [batch, seq] (0/1)
    """

    top_k_pos_indices: torch.Tensor
    top_k_pos_values: torch.Tensor
    top_k_neg_indices: torch.Tensor
    top_k_neg_values: torch.Tensor
    attention_mask: torch.Tensor
    ignore_padding: bool


@dataclass
class TokenEntry:
    """A single token in an ordering."""
    token_id: int
    token_str: str
    ordering_value: float  # x-axis value (the ordering metric)
    avg_logit_diff: float  # y-axis value
    count_positive: int = 0
    count_negative: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Ordering:
    """A single ordering (ordered list of tokens)."""
    ordering_id: str
    display_label: str
    tokens: List[TokenEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderingTypeResult:
    """Result from running an ordering type on a dataset."""
    ordering_type_id: str
    display_name: str
    x_axis_label: str
    y_axis_label: str = "Avg Logit Diff"
    orderings: List[Ordering] = field(default_factory=list)
    type_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedTokenStats:
    """
    Shared per-token statistics computed once for all ordering types.
    
    All tensors are 1D with shape [vocab_size].
    """
    vocab_size: int
    total_positions: int
    num_samples: int
    sum_logit_diff: torch.Tensor  # Sum of logit diffs across all positions
    count_positive: torch.Tensor  # Count of positions where logit diff > 0
    topk_pos_counts: torch.Tensor  # Count of times token appeared in top-K positive
    topk_neg_counts: torch.Tensor  # Count of times token appeared in top-K negative
    
    def __post_init__(self):
        assert self.sum_logit_diff.shape == (self.vocab_size,)
        assert self.count_positive.shape == (self.vocab_size,)
        assert self.topk_pos_counts.shape == (self.vocab_size,)
        assert self.topk_neg_counts.shape == (self.vocab_size,)


class TokenOrderingType(ABC):
    """
    Abstract base class for token ordering types.
    
    Each ordering type takes shared token statistics and produces
    one or more orderings.
    """
    
    @property
    @abstractmethod
    def ordering_type_id(self) -> str:
        """Unique identifier for this ordering type."""
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this ordering type."""
    
    @property
    @abstractmethod
    def x_axis_label(self) -> str:
        """Label for the x-axis (ordering metric)."""
    
    @abstractmethod
    def compute_orderings(
        self,
        stats: SharedTokenStats,
        tokenizer: PreTrainedTokenizerBase,
        num_tokens: int,
    ) -> OrderingTypeResult:
        """
        Compute orderings from shared token statistics.
        
        Args:
            stats: Shared per-token statistics
            tokenizer: Tokenizer for decoding token IDs
            num_tokens: Maximum number of tokens to include per ordering
            
        Returns:
            OrderingTypeResult containing one or more orderings
        """

    def begin_collection(self, dataset_cfg: DatasetConfig, *, ignore_padding: bool) -> None:
        """Initialize per-dataset collection state (default: no-op)."""
        return

    def collect_batch(self, batch: OrderingBatchCache) -> None:
        """Collect per-batch data (default: no-op)."""
        return

    def end_collection(self) -> None:
        """Finalize per-dataset collection state (default: no-op)."""
        return


class TopKOccurringOrderingType(TokenOrderingType):
    """
    Order tokens by occurrence rate in top-K positive logit diffs.
    
    This is the default ordering that ranks tokens by how frequently
    they appear in the top-K most positive logit diffs.
    """
    
    @property
    def ordering_type_id(self) -> str:
        return "topk_occurring"
    
    @property
    def display_name(self) -> str:
        return "Top-K Occurring"
    
    @property
    def x_axis_label(self) -> str:
        return "Occurrence Rate in Top-K (%)"
    
    def compute_orderings(
        self,
        stats: SharedTokenStats,
        tokenizer: PreTrainedTokenizerBase,
        num_tokens: int,
    ) -> OrderingTypeResult:
        # Compute occurrence rates
        occurrence_rates = (stats.topk_pos_counts.float() / stats.total_positions) * 100
        
        # Get top tokens by occurrence rate
        top_values, top_indices = torch.topk(
            occurrence_rates, k=min(num_tokens, stats.vocab_size), largest=True
        )
        
        # Compute avg logit diff for each token
        avg_logit_diff = stats.sum_logit_diff / stats.total_positions
        
        tokens = []
        for i, (idx, rate) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
            token_str = tokenizer.decode([idx])
            tokens.append(TokenEntry(
                token_id=idx,
                token_str=token_str,
                ordering_value=rate,
                avg_logit_diff=float(avg_logit_diff[idx]),
                count_positive=int(stats.topk_pos_counts[idx]),
                count_negative=int(stats.topk_neg_counts[idx]),
            ))
        
        ordering = Ordering(
            ordering_id="global",
            display_label="Global (by occurrence rate)",
            tokens=tokens,
        )
        
        return OrderingTypeResult(
            ordering_type_id=self.ordering_type_id,
            display_name=self.display_name,
            x_axis_label=self.x_axis_label,
            orderings=[ordering],
        )


class FractionPositiveDiffOrderingType(TokenOrderingType):
    """
    Order tokens by fraction of positions with positive logit diff.
    
    This ordering ranks tokens by what fraction of all analyzed positions
    had a positive logit diff for that token.
    """
    
    @property
    def ordering_type_id(self) -> str:
        return "fraction_positive_diff"
    
    @property
    def display_name(self) -> str:
        return "Fraction Positive Diff"
    
    @property
    def x_axis_label(self) -> str:
        return "Fraction of Positive Shifts"
    
    def compute_orderings(
        self,
        stats: SharedTokenStats,
        tokenizer: PreTrainedTokenizerBase,
        num_tokens: int,
    ) -> OrderingTypeResult:
        # Compute fraction positive
        fraction_positive = stats.count_positive.float() / stats.total_positions
        
        # Get top tokens by fraction positive
        top_values, top_indices = torch.topk(
            fraction_positive, k=min(num_tokens, stats.vocab_size), largest=True
        )
        
        # Compute avg logit diff for each token
        avg_logit_diff = stats.sum_logit_diff / stats.total_positions
        
        tokens = []
        for idx, frac in zip(top_indices.tolist(), top_values.tolist()):
            token_str = tokenizer.decode([idx])
            tokens.append(TokenEntry(
                token_id=idx,
                token_str=token_str,
                ordering_value=frac,
                avg_logit_diff=float(avg_logit_diff[idx]),
                count_positive=int(stats.count_positive[idx]),
                count_negative=int(stats.total_positions - stats.count_positive[idx]),
            ))
        
        ordering = Ordering(
            ordering_id="global",
            display_label="Global (by fraction positive)",
            tokens=tokens,
        )
        
        return OrderingTypeResult(
            ordering_type_id=self.ordering_type_id,
            display_name=self.display_name,
            x_axis_label=self.x_axis_label,
            orderings=[ordering],
        )


@dataclass
class NmfOrderingConfig:
    """Configuration for NMF ordering type."""
    num_topics: int
    beta: float = 2.0
    mode: str = "logit_diff_magnitude"  # or "binary_occurrence"
    orthogonal: bool = False
    orthogonal_weight: float = 1.0
    top_n_tokens_per_topic: int = 100


class NmfOrderingType(TokenOrderingType):
    """
    Order tokens by NMF topic membership.
    
    Produces one ordering per topic, where tokens are ordered by
    their weight in that topic.
    """
    
    def __init__(self, config: NmfOrderingConfig):
        """Create an NMF ordering type (collection happens during core analysis)."""
        self.config = config
        self.nmf_data: Optional[Dict[str, Any]] = None
    
    @property
    def ordering_type_id(self) -> str:
        return "nmf"
    
    @property
    def display_name(self) -> str:
        return f"NMF Topics ({self.config.num_topics} topics)"
    
    @property
    def x_axis_label(self) -> str:
        return "NMF Topic Weight"
    
    def compute_orderings(
        self,
        stats: SharedTokenStats,
        tokenizer: PreTrainedTokenizerBase,
        num_tokens: int,
    ) -> OrderingTypeResult:
        if self.nmf_data is None or not self.nmf_data.get("rows"):
            return OrderingTypeResult(
                ordering_type_id=self.ordering_type_id,
                display_name=self.display_name,
                x_axis_label=self.x_axis_label,
                orderings=[],
                type_metadata={"error": "No NMF data collected"},
            )
        
        # Run NMF clustering
        topic_token_matrix, pairwise_metrics, topic_metrics = self._run_nmf()
        
        if topic_token_matrix is None:
            return OrderingTypeResult(
                ordering_type_id=self.ordering_type_id,
                display_name=self.display_name,
                x_axis_label=self.x_axis_label,
                orderings=[],
                type_metadata={"error": "NMF fitting failed"},
            )
        
        # Compute avg logit diff for each token
        avg_logit_diff = stats.sum_logit_diff / stats.total_positions
        
        # Build orderings per topic
        orderings = []
        col_idx_to_token_id = self.nmf_data["col_idx_to_token_id"]
        
        for topic_idx in range(self.config.num_topics):
            topic_weights = topic_token_matrix[topic_idx]
            
            # Get top tokens by weight
            k = min(num_tokens, len(col_idx_to_token_id))
            top_values, top_indices = torch.topk(topic_weights, k=k, largest=True)
            
            tokens = []
            for col_idx, weight in zip(top_indices.tolist(), top_values.tolist()):
                if weight < 1e-6:
                    continue
                token_id = col_idx_to_token_id[col_idx]
                token_str = tokenizer.decode([token_id])
                tokens.append(TokenEntry(
                    token_id=token_id,
                    token_str=token_str,
                    ordering_value=weight,
                    avg_logit_diff=float(avg_logit_diff[token_id]),
                    count_positive=int(stats.topk_pos_counts[token_id]),
                    count_negative=int(stats.topk_neg_counts[token_id]),
                    extra={"nmf_col_idx": col_idx},
                ))
            
            ordering_id = f"topic_{topic_idx}"
            ordering_metrics = topic_metrics.get(ordering_id, {})
            orderings.append(
                Ordering(
                    ordering_id=ordering_id,
                    display_label=f"Topic {topic_idx}",
                    tokens=tokens,
                    metadata={**ordering_metrics},
                )
            )
        
        return OrderingTypeResult(
            ordering_type_id=self.ordering_type_id,
            display_name=self.display_name,
            x_axis_label=self.x_axis_label,
            orderings=orderings,
            type_metadata={
                "num_topics": self.config.num_topics,
                "beta": self.config.beta,
                "mode": self.config.mode,
                "pairwise": pairwise_metrics,
                "topic_metrics": [topic_metrics[f"topic_{i}"] for i in range(self.config.num_topics)],
            },
        )

    def begin_collection(self, dataset_cfg: DatasetConfig, *, ignore_padding: bool) -> None:
        self.nmf_data = {
            "rows": [],
            "cols": [],
            "values": [],
            "valid_row_idx_counter": 0,
            "token_id_to_col_idx": {},
            "col_idx_to_token_id": [],
            "next_col_idx": 0,
        }

    def collect_batch(self, batch: OrderingBatchCache) -> None:
        assert self.nmf_data is not None, "begin_collection must be called before collect_batch"
        assert batch.top_k_pos_indices.shape == batch.top_k_pos_values.shape
        assert batch.top_k_neg_indices.shape == batch.top_k_neg_values.shape
        assert batch.top_k_pos_indices.shape[:2] == batch.attention_mask.shape
        assert batch.top_k_pos_indices.shape == batch.top_k_neg_indices.shape
        assert batch.top_k_pos_indices.device == batch.attention_mask.device

        top_k = int(batch.top_k_pos_indices.shape[-1])

        if batch.ignore_padding:
            valid_positions_mask = batch.attention_mask.bool()
        else:
            valid_positions_mask = torch.ones_like(batch.attention_mask, dtype=torch.bool)

        valid_flat = valid_positions_mask.flatten()
        num_valid = int(valid_flat.sum().item())
        if num_valid == 0:
            return

        row_start = int(self.nmf_data["valid_row_idx_counter"])
        self.nmf_data["valid_row_idx_counter"] = row_start + num_valid

        flat_indices = batch.top_k_pos_indices.reshape(-1, top_k)[valid_flat].cpu()

        flat_values = None
        if self.config.mode != "binary_occurrence":
            flat_values = batch.top_k_pos_values.reshape(-1, top_k)[valid_flat].cpu()

        for k_idx in range(top_k):
            token_ids_k = flat_indices[:, k_idx].tolist()
            for row_idx_local, token_id_item in enumerate(token_ids_k):
                row_idx_global = row_start + row_idx_local

                if self.config.mode == "binary_occurrence":
                    val = 1.0
                else:
                    assert flat_values is not None
                    val = float(flat_values[row_idx_local, k_idx].item())

                token_id_item = int(token_id_item)
                token_id_to_col_idx = self.nmf_data["token_id_to_col_idx"]
                if token_id_item not in token_id_to_col_idx:
                    token_id_to_col_idx[token_id_item] = int(self.nmf_data["next_col_idx"])
                    self.nmf_data["col_idx_to_token_id"].append(token_id_item)
                    self.nmf_data["next_col_idx"] += 1

                col_idx = int(token_id_to_col_idx[token_id_item])
                self.nmf_data["rows"].append(row_idx_global)
                self.nmf_data["cols"].append(col_idx)
                self.nmf_data["values"].append(val)
    
    def _run_nmf(self) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Run NMF fitting on the collected data."""
        import scipy.sparse
        from torchnmf.nmf import NMF
        from .orthogonal_nmf import fit_nmf_orthogonal
        
        num_rows = self.nmf_data["valid_row_idx_counter"]
        num_cols = self.nmf_data["next_col_idx"]
        
        if num_rows == 0 or num_cols == 0:
            return None, {}, {}
        
        W_nmf: torch.Tensor
        H_nmf: torch.Tensor
        if self.config.orthogonal:
            with torch.enable_grad():
                W_nmf, H_nmf = fit_nmf_orthogonal(
                    torch.tensor(
                        scipy.sparse.coo_matrix(
                            (self.nmf_data["values"], (self.nmf_data["rows"], self.nmf_data["cols"])),
                            shape=(num_rows, num_cols),
                        ).todense(),
                        dtype=torch.float32,
                    ),
                    rank=self.config.num_topics,
                    beta=self.config.beta,
                    orthogonal_weight=self.config.orthogonal_weight,
                    max_iter=200,
                    device="auto",
                    verbose=True,
                )
        else:
            row_idx = torch.as_tensor(self.nmf_data["rows"], dtype=torch.long)
            col_idx = torch.as_tensor(self.nmf_data["cols"], dtype=torch.long)
            vals = torch.as_tensor(self.nmf_data["values"], dtype=torch.float32)
            
            indices = torch.stack([row_idx, col_idx], dim=0)
            V_raw = torch.sparse_coo_tensor(indices, vals, size=(num_rows, num_cols)).coalesce()
            V_vals = torch.relu(V_raw.values())
            V_keep = V_vals > 0
            
            if not bool(V_keep.any().item()):
                return None, {}, {}
            
            V = torch.sparse_coo_tensor(
                V_raw.indices()[:, V_keep],
                V_vals[V_keep],
                size=(num_rows, num_cols),
            ).coalesce()
            
            if torch.cuda.is_available():
                V = V.cuda()
            
            nmf = NMF(V.shape, rank=self.config.num_topics)
            if torch.cuda.is_available():
                nmf = nmf.cuda()
            
            with torch.enable_grad():
                nmf.fit(V, beta=self.config.beta, verbose=True, max_iter=200)
            
            W_nmf = nmf.W.detach().cpu()
            H_nmf = nmf.H.detach().cpu()
        
        assert W_nmf.ndim == 2 and H_nmf.ndim == 2
        assert W_nmf.shape == (num_cols, self.config.num_topics)
        assert H_nmf.shape == (num_rows, self.config.num_topics)

        # topic_token_matrix: [num_topics, num_tokens]
        topic_token_matrix = W_nmf.T
        
        # Compute pairwise metrics
        token_mass_l2 = torch.linalg.norm(topic_token_matrix, ord=2, dim=1)
        normed_topics = topic_token_matrix / token_mass_l2[:, None]
        cosine_sim = normed_topics @ normed_topics.T
        
        pairwise_metrics = {
            "cosine_similarity": cosine_sim.tolist(),
        }

        topic_metrics = self._compute_topic_metrics(
            W_nmf=W_nmf,
            H_nmf=H_nmf,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        return topic_token_matrix, pairwise_metrics, topic_metrics

    def _compute_topic_metrics(
        self,
        *,
        W_nmf: torch.Tensor,
        H_nmf: torch.Tensor,
        num_rows: int,
        num_cols: int,
    ) -> Dict[str, Dict[str, Any]]:
        colsum_w = W_nmf.sum(dim=0)
        assert bool((colsum_w > 0).all().item())

        mass = H_nmf.sum(dim=0) * colsum_w
        contributions = H_nmf * colsum_w[None, :]
        dominant = torch.argmax(contributions, dim=1)
        counts = torch.bincount(dominant, minlength=int(self.config.num_topics)).float()
        prevalence = counts / float(num_rows)

        if num_cols == 1:
            concentration = torch.ones((self.config.num_topics,), dtype=torch.float32)
        else:
            W_norm = W_nmf / colsum_w[None, :]
            p = W_norm.clamp(min=1e-12)
            entropy = -(p * torch.log(p)).sum(dim=0)
            entropy_norm = entropy / float(math.log(num_cols))
            concentration = 1.0 - entropy_norm

        out: Dict[str, Dict[str, Any]] = {}
        for topic_idx in range(self.config.num_topics):
            ordering_id = f"topic_{topic_idx}"
            out[ordering_id] = {
                "nmf_topic_idx": int(topic_idx),
                "nmf_topic_mass": float(mass[topic_idx].item()),
                "nmf_topic_prevalence": float(prevalence[topic_idx].item()),
                "nmf_topic_concentration": float(concentration[topic_idx].item()),
            }
        return out


# ============================================================================
# JSON Schema Helpers
# ============================================================================

def write_ordering_type_metadata(
    output_dir: Path,
    result: OrderingTypeResult,
) -> Path:
    """
    Write ordering type metadata.json.
    
    Args:
        output_dir: Directory for the ordering type (e.g., run_dir/topk_occurring/)
        result: OrderingTypeResult to serialize
        
    Returns:
        Path to the written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ordering_type_id": result.ordering_type_id,
        "display_name": result.display_name,
        "x_axis_label": result.x_axis_label,
        "y_axis_label": result.y_axis_label,
        **result.type_metadata,
    }
    path = output_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return path


def write_dataset_orderings(
    output_dir: Path,
    dataset_name: str,
    orderings: List[Ordering],
) -> Path:
    """
    Write orderings.json and per-ordering files for a dataset.
    
    Args:
        output_dir: Directory for the dataset (e.g., run_dir/topk_occurring/dataset_name/)
        dataset_name: Name of the dataset
        orderings: List of orderings to write
        
    Returns:
        Path to the orderings.json file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write orderings.json (index file)
    index = {
        "dataset_name": dataset_name,
        "orderings": [
            {
                "ordering_id": o.ordering_id,
                "display_label": o.display_label,
                "path": f"{o.ordering_id}.json",
                "num_tokens": len(o.tokens),
            }
            for o in orderings
        ],
    }
    index_path = output_dir / "orderings.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    # Write per-ordering files
    for ordering in orderings:
        ordering_data = {
            "ordering_id": ordering.ordering_id,
            "display_label": ordering.display_label,
            "tokens": [
                {
                    "token_id": t.token_id,
                    "token_str": t.token_str,
                    "ordering_value": t.ordering_value,
                    "avg_logit_diff": t.avg_logit_diff,
                    "count_positive": t.count_positive,
                    "count_negative": t.count_negative,
                    **t.extra,
                }
                for t in ordering.tokens
            ],
            **ordering.metadata,
        }
        ordering_path = output_dir / f"{ordering.ordering_id}.json"
        with open(ordering_path, "w", encoding="utf-8") as f:
            json.dump(ordering_data, f, indent=2, ensure_ascii=False)
    
    return index_path


def write_ordering_eval(
    output_dir: Path,
    ordering_id: str,
    labels: List[str],
    metadata: Dict[str, Any],
) -> Path:
    """
    Write per-ordering eval file (*_eval.json).
    
    Args:
        output_dir: Directory for the dataset
        ordering_id: ID of the ordering
        labels: List of relevance labels (RELEVANT/IRRELEVANT)
        metadata: Additional metadata
        
    Returns:
        Path to the written file
    """
    eval_data = {
        "ordering_id": ordering_id,
        "labels": labels,
        **metadata,
    }
    path = output_dir / f"{ordering_id}_eval.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    return path


def read_ordering_type_metadata(ordering_type_dir: Path) -> Optional[Dict[str, Any]]:
    """Read ordering type metadata.json."""
    path = ordering_type_dir / "metadata.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_dataset_orderings_index(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """Read orderings.json index for a dataset."""
    path = dataset_dir / "orderings.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_ordering(dataset_dir: Path, ordering_id: str) -> Optional[Dict[str, Any]]:
    """Read a specific ordering file."""
    path = dataset_dir / f"{ordering_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_ordering_eval(dataset_dir: Path, ordering_id: str) -> Optional[Dict[str, Any]]:
    """Read per-ordering eval file."""
    path = dataset_dir / f"{ordering_id}_eval.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
