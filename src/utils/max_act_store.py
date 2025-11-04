"""
SQLite-based storage for maximum activating examples - Improved Version.

This module provides persistent storage for maximum activating examples with:
- Cleaner separation of concerns
- Consolidated tensor processing logic
- Consistent database connection handling
- Reduced code duplication
- Better type safety
"""

from re import T
import sqlite3
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, Callable
from tqdm import tqdm, trange
from loguru import logger
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager


# ================================
# UTILITY CLASSES AND FUNCTIONS
# ================================


class WriteCommand(Enum):
    ADD_BATCH = "add_batch"
    MAINTAIN_TOP_K = "maintain_top_k"
    SHUTDOWN = "shutdown"
    FLUSH = "flush"
    SYNC_TO_DISK = "sync_to_disk"


@dataclass
class BatchData:
    """Data structure for batch examples to be written."""

    scores_per_example: np.ndarray
    input_ids_batch: List[torch.Tensor]  # Token lists after attention mask applied
    scores_per_token_batch: Optional[List[Optional[np.ndarray]]]
    additional_data_batch: Optional[List[dict]]
    latent_idx: Optional[Union[int, np.ndarray, torch.Tensor]]
    quantile_idx: Optional[Union[int, np.ndarray, torch.Tensor]]
    dataset_name: Optional[str]
    dataset_id: Optional[int]

    def get_per_example_value(
        self,
        value: Optional[Union[int, np.ndarray, torch.Tensor]],
        example_idx: int,
        batch_size: int,
    ) -> Optional[int]:
        """Extract per-example value from either scalar or array."""
        if value is None:
            return None
        elif isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            assert (
                len(value) == batch_size
            ), f"Array length {len(value)} != batch size {batch_size}"
            return int(value[example_idx])
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def get_per_example_latent_idx(
        self, example_idx: int, batch_size: int
    ) -> Optional[int]:
        return self.get_per_example_value(self.latent_idx, example_idx, batch_size)

    def get_per_example_quantile_idx(
        self, example_idx: int, batch_size: int
    ) -> Optional[int]:
        return self.get_per_example_value(self.quantile_idx, example_idx, batch_size)


@dataclass
class WriteRequest:
    """Request sent to background writer process."""

    command: WriteCommand
    data: Optional[BatchData] = None
    response_queue: Optional[mp.Queue] = None


def process_batch_tensors(
    input_ids_batch: List[torch.Tensor] | torch.Tensor,
    attention_mask_batch: Optional[torch.Tensor] = None,
    scores_per_token_batch: Optional[torch.Tensor | List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], Optional[List[Optional[np.ndarray]]]]:
    """
    Process batch tensors consistently, applying attention masks and converting to numpy.

    Returns:
        Tuple of (processed_input_ids, processed_scores_per_token)
    """
    batch_size = len(input_ids_batch)

    if attention_mask_batch is None:
        if isinstance(input_ids_batch, torch.Tensor):
            # All have the same lengths
            processed_input_ids = [input_ids_batch[i] for i in range(batch_size)]
        else:
            # Different lengths
            processed_input_ids = list(input_ids_batch)

        if scores_per_token_batch is not None:
            if isinstance(scores_per_token_batch, torch.Tensor):
                # All have the same lengths
                processed_scores_per_token = [
                    scores_per_token_batch[i].cpu().numpy() for i in range(batch_size)
                ]
            else:
                # Different lengths
                processed_scores_per_token = [
                    el.cpu().numpy() for el in scores_per_token_batch
                ]
        else:
            processed_scores_per_token = None
    else:
        # Apply attention mask per example
        processed_input_ids = []
        processed_scores_per_token = []

        for i in range(batch_size):
            input_ids = input_ids_batch[i]
            valid_mask = attention_mask_batch[i].bool()
            input_ids = input_ids[valid_mask]

            processed_input_ids.append(input_ids)

            if scores_per_token_batch is not None:
                scores_per_token = scores_per_token_batch[i][valid_mask]
                processed_scores_per_token.append(scores_per_token.cpu().numpy())
            else:
                processed_scores_per_token.append(None)

        if scores_per_token_batch is None:
            processed_scores_per_token = None

    return processed_input_ids, processed_scores_per_token


class DatabaseManager:
    """Manages database connections and common operations."""

    def __init__(self, db_path: Path, readonly: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.conn = None
        self._setup_connection()

    def _setup_connection(self):
        if self.readonly:
            self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        else:
            self.conn = sqlite3.connect(self.db_path)

    @contextmanager
    def get_connection(self, enable_foreign_keys: bool = True):
        """Context manager for database connections with consistent setup."""
        assert self.conn is not None, "Database has been closed"
        if enable_foreign_keys:
            self.conn.execute("PRAGMA foreign_keys = ON")
        yield self.conn

    def close(self):
        if self.conn:
            self.conn.close()


class GroupKeyBuilder:
    """Builds consistent group keys and WHERE clauses for database queries."""

    def __init__(self, per_dataset: bool = False):
        self.per_dataset = per_dataset

    def build_group_key(
        self,
        latent_idx: Optional[int],
        quantile_idx: Optional[int],
        dataset_name: Optional[str],
    ) -> Optional[tuple]:
        """Build a consistent grouping key."""
        key = []
        if latent_idx is not None:
            key.append(("latent_idx", latent_idx))
        if quantile_idx is not None:
            key.append(("quantile_idx", quantile_idx))
        if self.per_dataset:
            key.append(("dataset_name", dataset_name))
        return tuple(key) if key else None

    def build_where_clause(
        self,
        latent_idx: Optional[int] = None,
        quantile_idx: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ) -> Tuple[str, List]:
        """Build consistent WHERE clause and parameters for group-based queries."""
        conditions = []
        params = []

        # Handle latent_idx
        if latent_idx is not None:
            conditions.append("e.latent_idx = ?")
            params.append(latent_idx)
        else:
            conditions.append("e.latent_idx IS NULL")

        # Handle quantile_idx
        if quantile_idx is not None:
            conditions.append("e.quantile_idx = ?")
            params.append(quantile_idx)
        else:
            conditions.append("e.quantile_idx IS NULL")

        # Handle dataset_name (only if per_dataset is True)
        if self.per_dataset:
            if dataset_name is not None:
                conditions.append("s.dataset_name = ?")
                params.append(dataset_name)
            else:
                conditions.append("s.dataset_name IS NULL")

        where_clause = " AND ".join(conditions)
        return where_clause, params


class ActivationDetailsHandler:
    """Handles activation details storage/retrieval for both sparse and dense formats."""

    def __init__(self, storage_format: Literal["sparse", "dense"]):
        self.storage_format = storage_format

    def prepare_for_storage(
        self, scores_per_token: torch.Tensor
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Prepare activation data for storage based on format."""
        if self.storage_format == "sparse":
            positions = np.arange(len(scores_per_token), dtype=np.int32)
            values = scores_per_token.float().cpu().numpy().astype(np.float32)
            # Filter out zeros for true sparsity
            mask = values != 0
            return positions[mask], values[mask]
        else:  # dense
            return scores_per_token.cpu().numpy().astype(np.float32)

    def format_for_database(
        self, data: Union[Tuple[np.ndarray, np.ndarray], np.ndarray], example_id: int
    ) -> tuple:
        """Format activation data for database insertion."""
        if self.storage_format == "sparse":
            positions, values = data
            return (
                example_id,
                positions.astype(np.int32).tobytes(),
                values.astype(np.float32).tobytes(),
            )
        else:  # dense
            assert isinstance(
                data, np.ndarray
            ), "Either wrong data type or wrong storage format"
            return (example_id, data.astype(np.float32).tobytes())


class TopKProxy:
    """
    Tracks minimum scores for each group to optimize example insertion.

    This class maintains a cache of minimum scores per group to enable fast
    filtering of examples that won't qualify for top-k storage. The cache
    is updated periodically based on the specified frequency.

    Args:
        collection_function: Function that returns current minimum scores dict
        frequency: Update frequency in seconds (default: 30)
    """

    def __init__(
        self, collection_function: Callable[[], Dict[tuple, float]], ttl: int = 30
    ):
        self.collection_function = collection_function
        self.ttl = ttl
        self.last_update = time.time()
        self.init_min_scores()

    def init_min_scores(self):
        """Initialize the minimum scores dictionary."""
        self.min_scores = {}

    def update(self):
        """Update cached minimum scores if frequency interval has elapsed."""
        if time.time() - self.last_update > self.ttl:
            self.min_scores = self.collection_function()
            self.last_update = time.time()

    def get_min_score(self, group_key: tuple) -> float:
        """
        Get minimum score for a group.

        Args:
            group_key: Tuple identifying the group

        Returns:
            Minimum score for the group, or -inf if group not found (no minimum score)
        """
        return self.min_scores.get(group_key, float("-inf"))


class MultiProcessTopKProxy(TopKProxy):
    """
    Thread-safe version of TopKProxy for multi-process environments.

    Extends TopKProxy with proper locking mechanisms to ensure thread safety
    when multiple processes access the minimum scores cache.

    Args:
        collection_function: Function that returns current minimum scores dict
        frequency: Update frequency in seconds (default: 30)
    """

    def __init__(
        self, collection_function: Callable[[], Dict[tuple, float]], ttl: int = 30
    ):
        self.min_scores_lock = mp.Lock()
        super().__init__(collection_function, ttl)

    def init_min_scores(self):
        """Initialize the minimum scores dictionary with multiprocessing support."""
        with self.min_scores_lock:
            self.min_scores = mp.Manager().dict()

    def update(self):
        """Update cached minimum scores with thread safety."""
        if time.time() - self.last_update > self.ttl:
            with self.min_scores_lock:
                min_scores = self.collection_function()
                for group_key, score in min_scores.items():
                    self.min_scores[group_key] = score

    def get_min_score(self, group_key: tuple) -> float:
        """
        Get minimum score for a group in a thread-safe manner.

        Args:
            group_key: Tuple identifying the group

        Returns:
            Minimum score for the group, or -inf if group not found (no minimum score)
        """
        with self.min_scores_lock:
            return self.min_scores.get(group_key, float("-inf"))


def legacy_converter(db_path: Path):
    """Convert legacy sequence_idx column to sequence_uid in the sequences and examples tables."""
    logger.info("Converting legacy sequence_idx column to sequence_uid...")
    logger.info(f"Database path: {db_path}")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get list of existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        # Check if sequences table exists and has sequence_idx column
        sequences_needs_conversion = False
        if "sequences" in existing_tables:
            cursor.execute("PRAGMA table_info(sequences)")
            sequences_columns = {row[1]: row for row in cursor.fetchall()}
            sequences_needs_conversion = "sequence_idx" in sequences_columns

        # Check if examples table exists and has sequence_idx column
        examples_needs_conversion = False
        if "examples" in existing_tables:
            cursor.execute("PRAGMA table_info(examples)")
            examples_columns = {row[1]: row for row in cursor.fetchall()}
            examples_needs_conversion = "sequence_idx" in examples_columns

        # If no conversion needed, return early
        if not sequences_needs_conversion and not examples_needs_conversion:
            return

        logger.info("Converting legacy sequence_idx column to sequence_uid...")

        # Begin transaction for atomic conversion
        cursor.execute("BEGIN TRANSACTION")

        try:
            # Convert sequences table
            if sequences_needs_conversion:
                cursor.execute(
                    """
                    CREATE TABLE sequences_new (
                        sequence_uid INTEGER PRIMARY KEY,
                        token_ids BLOB NOT NULL,
                        text TEXT,
                        sequence_length INTEGER NOT NULL,
                        dataset_id INTEGER,
                        dataset_name TEXT
                    )
                """
                )

                cursor.execute(
                    """
                    INSERT INTO sequences_new (sequence_uid, token_ids, text, sequence_length, dataset_id, dataset_name)
                    SELECT sequence_idx, token_ids, text, sequence_length, dataset_id, dataset_name
                    FROM sequences
                """
                )

                cursor.execute("DROP TABLE sequences")
                cursor.execute("ALTER TABLE sequences_new RENAME TO sequences")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dataset_name ON sequences(dataset_name)"
                )

            # Convert examples table
            if examples_needs_conversion:
                cursor.execute(
                    """
                    CREATE TABLE examples_new (
                        example_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sequence_uid INTEGER NOT NULL,
                        score REAL NOT NULL,
                        latent_idx INTEGER DEFAULT NULL,
                        quantile_idx INTEGER DEFAULT NULL,
                        metadata TEXT,
                        FOREIGN KEY (sequence_uid) REFERENCES sequences(sequence_uid)
                    )
                """
                )

                cursor.execute(
                    """
                    INSERT INTO examples_new (example_id, sequence_uid, score, latent_idx, quantile_idx, metadata)
                    SELECT example_id, sequence_idx, score, latent_idx, quantile_idx, metadata
                    FROM examples
                """
                )

                cursor.execute("DROP TABLE examples")
                cursor.execute("ALTER TABLE examples_new RENAME TO examples")

                # Recreate indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_examples_score ON examples(score DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_examples_latent ON examples(latent_idx)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_examples_quantile ON examples(quantile_idx)"
                )

            cursor.execute("COMMIT")
            logger.info("Successfully converted sequence_idx to sequence_uid")

        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to convert sequence_idx to sequence_uid: {e}")
            raise


# ================================
# MAIN STORE CLASSES
# ================================


class MaxActStore:
    """
    SQLite-based storage for maximum activating examples.

    Supports bulk loading of pre-sorted examples and real-time top-k management.
    """

    def __init__(
        self,
        db_path: Path,
        max_examples: Optional[int] = None,
        tokenizer=None,
        storage_format: Optional[Literal["sparse", "dense"]] = "sparse",
        per_dataset: bool = False,
        top_k_proxy: Optional[TopKProxy] = None,
    ):
        """
        Initialize the store.

        Args:
            db_path: Path to SQLite database file
            max_examples: Maximum number of examples to keep per group
            tokenizer: Optional tokenizer for text decoding
            storage_format: Storage format for activation details
            per_dataset: If True, maintain max_examples per dataset
            top_k_proxy: Optional top-k proxy to use for early filtering
        """
        legacy_converter(db_path)
        self._setup_db_manager(db_path)
        self.tokenizer = tokenizer
        self.per_dataset = per_dataset
        self.group_builder = GroupKeyBuilder(per_dataset)

        # Handle configuration and storage format
        self._handle_config(max_examples, storage_format)
        self.activation_handler = ActivationDetailsHandler(self._storage_format)

        # Initialize database schema
        self._init_database()

        # Cache for sequence indices to avoid duplicate insertions
        self._sequence_uid_cache = set()
        self._top_k_proxy = (
            top_k_proxy
            if top_k_proxy is not None
            else TopKProxy(self.get_group_capacity_info)
        )

    @property
    def storage_format(self) -> str:
        return self._storage_format

    @property
    def max_examples(self) -> Optional[int]:
        return self._max_examples

    def _setup_db_manager(self, db_path: Path):
        """Setup the database manager."""
        self.db_manager = DatabaseManager(db_path, readonly=False)

    def _handle_config(
        self, max_examples: Optional[int], storage_format: Optional[str]
    ):
        """Handle configuration storage and validation."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Create config table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """
            )

            # Read existing config
            cursor.execute("SELECT key, value FROM config")
            existing_config = dict(cursor.fetchall())

            # Handle storage format
            if storage_format is None:
                if "storage_format" not in existing_config:
                    raise ValueError(
                        "No existing storage_format found and none provided"
                    )
                storage_format = existing_config["storage_format"]

            if storage_format not in ["sparse", "dense"]:
                raise ValueError(
                    f"storage_format must be 'sparse' or 'dense', got {storage_format}"
                )

            # Check for conflicts
            if (
                "storage_format" in existing_config
                and existing_config["storage_format"] != storage_format
            ):
                raise ValueError(
                    f"Storage format conflict: database has '{existing_config['storage_format']}' but '{storage_format}' provided"
                )

            # Set attributes
            self._storage_format = storage_format
            self._max_examples = max_examples

            # Store config
            cursor.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                ("storage_format", storage_format),
            )
            if max_examples is not None:
                cursor.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    ("max_examples", str(max_examples)),
                )

            conn.commit()

    def _init_database(self):
        """Initialize database schema."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Create sequences table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sequences (
                    sequence_uid INTEGER PRIMARY KEY,
                    token_ids BLOB NOT NULL,
                    text TEXT,
                    sequence_length INTEGER NOT NULL,
                    dataset_id INTEGER DEFAULT NULL,
                    dataset_name TEXT DEFAULT NULL
                )
            """
            )

            # Create examples table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS examples (
                    example_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_uid INTEGER NOT NULL,
                    score REAL NOT NULL,
                    latent_idx INTEGER DEFAULT NULL,
                    quantile_idx INTEGER DEFAULT NULL,
                    metadata TEXT,
                    FOREIGN KEY (sequence_uid) REFERENCES sequences(sequence_uid)
                )
            """
            )

            # Create activation details table (format-specific)
            if self.storage_format == "sparse":
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS activation_details (
                        example_id INTEGER PRIMARY KEY,
                        positions BLOB,
                        activation_values BLOB,
                        FOREIGN KEY (example_id) REFERENCES examples(example_id)
                    )
                """
                )
            else:  # dense
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS activation_details (
                        example_id INTEGER PRIMARY KEY,
                        activation_values BLOB NOT NULL,
                        FOREIGN KEY (example_id) REFERENCES examples(example_id)
                    )
                """
                )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_examples_score ON examples(score DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_examples_latent ON examples(latent_idx)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_examples_quantile ON examples(quantile_idx)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sequences_dataset ON sequences(dataset_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sequences_dataset_name ON sequences(dataset_name)"
            )

            conn.commit()

    def clear(self):
        """Clear all data except config."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM activation_details")
            cursor.execute("DELETE FROM examples")
            cursor.execute("DELETE FROM sequences")
            conn.commit()
        self._sequence_uid_cache.clear()

    def __len__(self) -> int:
        """Return number of examples in database."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM examples")
            return cursor.fetchone()[0]

    def _sequence_uid(self, input_ids_as_bytes: bytes) -> int:
        return hash(input_ids_as_bytes)

    def _prepare_sequence_data(self, token_ids: torch.Tensor | np.ndarray):
        """Prepare sequence data for insertion."""
        # Convert token IDs to binary
        assert isinstance(
            token_ids, (torch.Tensor, np.ndarray)
        ), f"token_ids must be a torch.Tensor or np.ndarray, got {type(token_ids)}"
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        binary_data = token_ids.astype(np.int32).tobytes()
        return binary_data

    def _insert_sequence(
        self,
        token_ids: torch.Tensor,
        dataset_id: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        """Insert a single sequence, using cache to avoid duplicates."""
        # Convert token IDs to binary
        binary_data = self._prepare_sequence_data(token_ids)
        sequence_uid = self._sequence_uid(binary_data)
        if sequence_uid in self._sequence_uid_cache:
            return sequence_uid

        # Get text if tokenizer available
        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sequence_uid,
                    binary_data,
                    text,
                    len(token_ids),
                    dataset_id,
                    dataset_name,
                ),
            )
            conn.commit()

        self._sequence_uid_cache.add(sequence_uid)
        return sequence_uid

    def _insert_examples_bulk(self, example_data: List[tuple]) -> List[int]:
        """Insert multiple examples and return their IDs."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            example_ids = []
            for data in example_data:
                score, sequence_uid, latent_idx, quantile_idx, additional_data = data
                metadata = json.dumps(additional_data) if additional_data else None

                cursor.execute(
                    "INSERT INTO examples (sequence_uid, score, latent_idx, quantile_idx, metadata) VALUES (?, ?, ?, ?, ?)",
                    (sequence_uid, float(score), latent_idx, quantile_idx, metadata),
                )
                example_ids.append(cursor.lastrowid)

            conn.commit()
        return example_ids

    def _insert_activation_details_bulk(self, activation_data: List[tuple]):
        """Insert activation details for multiple examples."""
        if not activation_data:
            return

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            if self.storage_format == "sparse":
                cursor.executemany(
                    "INSERT INTO activation_details VALUES (?, ?, ?)", activation_data
                )
            else:  # dense
                cursor.executemany(
                    "INSERT INTO activation_details VALUES (?, ?)", activation_data
                )

            conn.commit()

    def add_example(
        self,
        score: float,
        input_ids: torch.Tensor,
        scores_per_token: Optional[torch.Tensor] = None,
        latent_idx: Optional[int] = None,
        quantile_idx: Optional[int] = None,
        additional_data: Optional[dict] = None,
        maintain_top_k: bool = True,
        dataset_id: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        """Add a single example with optional top-k management."""
        # Generate sequence index
        sequence_uid = self._insert_sequence(input_ids, dataset_id, dataset_name)

        # Insert example
        example_ids = self._insert_examples_bulk(
            [(score, sequence_uid, latent_idx, quantile_idx, additional_data)]
        )

        # Insert activation details if provided
        if scores_per_token is not None:
            activation_data = self.activation_handler.prepare_for_storage(
                scores_per_token
            )
            formatted_data = [
                self.activation_handler.format_for_database(
                    activation_data, example_ids[0]
                )
            ]
            self._insert_activation_details_bulk(formatted_data)

        if maintain_top_k:
            self._maintain_top_k()

    def add_batch_examples(
        self,
        scores_per_example: torch.Tensor,
        input_ids_batch: torch.Tensor,
        attention_mask_batch: Optional[torch.Tensor] = None,
        scores_per_token_batch: Optional[torch.Tensor] = None,
        additional_data_batch: Optional[List[dict]] = None,
        latent_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
        quantile_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
    ):
        """Add multiple examples from a batch."""
        batch_size = scores_per_example.shape[0]
        assert input_ids_batch.shape[0] == batch_size

        # Process tensors consistently
        processed_input_ids, processed_scores_per_token = process_batch_tensors(
            input_ids_batch, attention_mask_batch, scores_per_token_batch
        )

        # Create batch data
        batch_data = BatchData(
            scores_per_example=scores_per_example.cpu().numpy(),
            input_ids_batch=processed_input_ids,
            scores_per_token_batch=processed_scores_per_token,
            additional_data_batch=additional_data_batch,
            latent_idx=latent_idx,
            quantile_idx=quantile_idx,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
        )

        self._process_batch_data(batch_data)
        self._maintain_top_k()

    def _process_batch_data(self, batch_data: BatchData):
        """Process batch data into database."""
        batch_size = len(batch_data.input_ids_batch)

        # Prepare all data for bulk insertion
        example_data = []
        activation_data = []
        sequence_data = []
        latent_idx_data = []
        quantile_idx_data = []

        insert_idx_to_batch_idx = []

        start_time = time.time()

        # prepare sequence data
        for i in range(batch_size):
            input_ids = batch_data.input_ids_batch[i]
            latent_idx = batch_data.get_per_example_latent_idx(i, batch_size)
            quantile_idx = batch_data.get_per_example_quantile_idx(i, batch_size)

            min_score = self._top_k_proxy.get_min_score(
                self.group_builder.build_group_key(
                    latent_idx, quantile_idx, batch_data.dataset_name
                )
            )

            if min_score > batch_data.scores_per_example[i]:
                continue

            latent_idx_data.append(latent_idx)
            quantile_idx_data.append(quantile_idx)
            sequence_data.append(
                (input_ids, batch_data.dataset_id, batch_data.dataset_name)
            )
            insert_idx_to_batch_idx.append(i)

        # bulk insert sequences
        # logger.info(f"\tInserting {len(sequence_data)} sequences (preprocessing took {time.time() - start_time:.2f} seconds)")
        sequence_uids = self._insert_sequences_bulk(sequence_data)
        # logger.info(f"\tFinished inserting sequences in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # prepare sequence data
        for i in range(len(insert_idx_to_batch_idx)):
            batch_idx = insert_idx_to_batch_idx[i]
            score = float(batch_data.scores_per_example[batch_idx])
            input_ids = batch_data.input_ids_batch[batch_idx]

            # Get per-example indices
            additional_data = (
                batch_data.additional_data_batch[batch_idx]
                if batch_data.additional_data_batch
                else None
            )

            # Prepare example data
            example_data.append(
                (
                    score,
                    sequence_uids[i],
                    latent_idx_data[i],
                    quantile_idx_data[i],
                    additional_data,
                )
            )

        # Bulk insert examples
        example_ids = self._insert_examples_bulk(example_data)
        assert len(example_ids) == len(insert_idx_to_batch_idx)
        # logger.info(f"\tFinished inserting examples in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # Prepare activation details if provided
        if batch_data.scores_per_token_batch is not None:
            for i, example_id in enumerate(example_ids):
                insert_idx = insert_idx_to_batch_idx[i]
                if batch_data.scores_per_token_batch[insert_idx] is not None:
                    scores_tensor = torch.tensor(
                        batch_data.scores_per_token_batch[insert_idx]
                    )
                    activation_prep = self.activation_handler.prepare_for_storage(
                        scores_tensor
                    )
                    formatted = self.activation_handler.format_for_database(
                        activation_prep, example_id
                    )
                    activation_data.append(formatted)

        # Bulk insert activation details
        if activation_data:
            self._insert_activation_details_bulk(activation_data)
        # logger.info(f"\tFinished inserting activation details in {time.time() - start_time:.2f} seconds")

    def _maintain_top_k(self):
        """Remove examples beyond max_examples limit per group."""
        if self._max_examples is None:
            return

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get all existing groups
            groups = self._get_existing_groups(cursor)

            for group_key in groups:
                self._maintain_top_k_for_group(cursor, group_key)

            conn.commit()

        self._top_k_proxy.update()

    def _get_existing_groups(self, cursor) -> List[Optional[tuple]]:
        """Get all existing grouping combinations."""
        cursor.execute(
            """
            SELECT DISTINCT e.latent_idx, e.quantile_idx, s.dataset_name
            FROM examples e
            JOIN sequences s ON e.sequence_uid = s.sequence_uid
        """
        )

        groups = set()
        for latent_idx, quantile_idx, dataset_name in cursor.fetchall():
            group_key = self.group_builder.build_group_key(
                latent_idx, quantile_idx, dataset_name
            )
            groups.add(group_key)

        return list(groups)

    def _maintain_top_k_for_group(self, cursor, group_key: Optional[tuple]):
        """Maintain top-k for a specific group."""
        # Extract group parameters
        latent_idx = quantile_idx = dataset_name = None
        if group_key:
            for dim_name, dim_value in group_key:
                if dim_name == "latent_idx":
                    latent_idx = dim_value
                elif dim_name == "quantile_idx":
                    quantile_idx = dim_value
                elif dim_name == "dataset_name":
                    dataset_name = dim_value

        # Build WHERE clause
        where_clause, params = self.group_builder.build_where_clause(
            latent_idx, quantile_idx, dataset_name
        )

        # Get count for this group
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM examples e
            JOIN sequences s ON e.sequence_uid = s.sequence_uid
            WHERE {where_clause}
        """,
            params,
        )

        current_count = cursor.fetchone()[0]

        if current_count > self._max_examples:
            # Get example IDs to delete (lowest scores)
            cursor.execute(
                f"""
                SELECT e.example_id FROM examples e
                JOIN sequences s ON e.sequence_uid = s.sequence_uid
                WHERE {where_clause}
                ORDER BY e.score ASC 
                LIMIT ?
            """,
                params + [current_count - self._max_examples],
            )

            ids_to_delete = [row[0] for row in cursor.fetchall()]
            self._delete_examples_by_ids(cursor, ids_to_delete)

    def _delete_examples_by_ids(self, cursor, ids_to_delete: List[int]):
        """Delete examples and their activation details."""
        if not ids_to_delete:
            return

        # Delete activation details first (foreign key constraint)
        cursor.executemany(
            "DELETE FROM activation_details WHERE example_id = ?",
            [(id_,) for id_ in ids_to_delete],
        )

        # Delete examples
        cursor.executemany(
            "DELETE FROM examples WHERE example_id = ?",
            [(id_,) for id_ in ids_to_delete],
        )

    def get_top_examples(
        self,
        limit: Optional[int] = None,
        latent_idx: Optional[int] = None,
        quantile_idx: Optional[int] = None,
        dataset_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get top examples with optional filtering."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT e.example_id, e.sequence_uid, e.score, e.latent_idx, e.quantile_idx, e.metadata,
                       s.token_ids, s.text, s.sequence_length, s.dataset_id, s.dataset_name
                FROM examples e
                JOIN sequences s ON e.sequence_uid = s.sequence_uid
            """

            conditions = []
            params = []

            if latent_idx is not None:
                conditions.append("e.latent_idx = ?")
                params.append(latent_idx)

            if quantile_idx is not None:
                conditions.append("e.quantile_idx = ?")
                params.append(quantile_idx)

            if dataset_names:
                placeholders = ",".join("?" * len(dataset_names))
                conditions.append(f"s.dataset_name IN ({placeholders})")
                params.extend(dataset_names)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY e.score DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            examples = []
            for row in cursor.fetchall():
                (
                    example_id,
                    sequence_uid,
                    score,
                    latent_idx,
                    quantile_idx,
                    metadata,
                    token_ids_blob,
                    text,
                    seq_length,
                    dataset_id,
                    dataset_name,
                ) = row

                # Decode token IDs and metadata
                token_ids = np.frombuffer(token_ids_blob, dtype=np.int32).tolist()
                parsed_metadata = json.loads(metadata) if metadata else {}

                example = {
                    "example_id": example_id,
                    "sequence_uid": sequence_uid,
                    "max_score": score,
                    "input_ids": token_ids,
                    "text": text,
                    "sequence_length": seq_length,
                    "latent_idx": latent_idx,
                    "quantile_idx": quantile_idx,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    **parsed_metadata,
                }
                examples.append(example)

            return examples

    def get_example_details(
        self, example_id: int, return_dense: bool = True
    ) -> Dict[str, Any]:
        """Get detailed information about a specific example."""
        results = self.get_batch_example_details([example_id], return_dense)
        if not results:
            raise ValueError(f"Example {example_id} not found")
        return results[0]

    def get_batch_example_details(
        self, example_ids: List[int], return_dense: bool = True
    ) -> List[Dict[str, Any]]:
        """Get detailed information about multiple examples efficiently."""
        if not example_ids:
            return []

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get basic example info
            placeholders = ",".join("?" * len(example_ids))
            cursor.execute(
                f"""
                SELECT e.example_id, e.sequence_uid, e.score, e.latent_idx, e.quantile_idx, e.metadata,
                       s.token_ids, s.text, s.sequence_length
                FROM examples e
                JOIN sequences s ON e.sequence_uid = s.sequence_uid
                WHERE e.example_id IN ({placeholders})
                ORDER BY e.score DESC
            """,
                example_ids,
            )

            results_dict = {}
            for row in cursor.fetchall():
                (
                    example_id,
                    sequence_uid,
                    score,
                    latent_idx,
                    quantile_idx,
                    metadata,
                    token_ids_blob,
                    text,
                    seq_length,
                ) = row

                token_ids = np.frombuffer(token_ids_blob, dtype=np.int32).tolist()
                parsed_metadata = json.loads(metadata) if metadata else {}

                results_dict[example_id] = {
                    "example_id": example_id,
                    "sequence_uid": sequence_uid,
                    "max_score": score,
                    "input_ids": token_ids,
                    "text": text,
                    "sequence_length": seq_length,
                    "latent_idx": latent_idx,
                    "quantile_idx": quantile_idx,
                    **parsed_metadata,
                }

            # Get activation details
            if self.storage_format == "dense":
                cursor.execute(
                    f"""
                    SELECT example_id, activation_values 
                    FROM activation_details 
                    WHERE example_id IN ({placeholders})
                """,
                    example_ids,
                )

                for example_id, values_blob in cursor.fetchall():
                    if example_id in results_dict:
                        values = np.frombuffer(values_blob, dtype=np.float32)
                        results_dict[example_id]["scores_per_token"] = values
                        if not return_dense:
                            positions = np.where(values != 0)[0]
                            results_dict[example_id]["positions"] = positions
            else:  # sparse
                cursor.execute(
                    f"""
                    SELECT example_id, positions, activation_values
                    FROM activation_details 
                    WHERE example_id IN ({placeholders})
                """,
                    example_ids,
                )

                for example_id, positions_blob, values_blob in cursor.fetchall():
                    if example_id in results_dict:
                        positions = np.frombuffer(positions_blob, dtype=np.int32)
                        values = np.frombuffer(values_blob, dtype=np.float32)
                        if return_dense:
                            # Convert to dense
                            token_ids = results_dict[example_id]["input_ids"]
                            dense_values = np.zeros(len(token_ids))
                            dense_values[positions] = values
                            results_dict[example_id]["scores_per_token"] = dense_values
                        else:
                            results_dict[example_id]["scores_per_token"] = values
                            results_dict[example_id]["positions"] = positions

            return [results_dict[eid] for eid in example_ids if eid in results_dict]

    # Utility methods
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT dataset_name FROM sequences WHERE dataset_name IS NOT NULL ORDER BY dataset_name"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_available_latents(self) -> List[int]:
        """Get list of available latent indices."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT latent_idx FROM examples WHERE latent_idx IS NOT NULL ORDER BY latent_idx"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_available_quantiles(self) -> List[int]:
        """Get list of available quantile indices."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT quantile_idx FROM examples WHERE quantile_idx IS NOT NULL ORDER BY quantile_idx"
            )
            return [row[0] for row in cursor.fetchall()]

    def fill(
        self,
        examples_data: Dict,
        all_sequences: List,
        activation_details: Optional[Dict] = None,
        dataset_info: Optional[List[Tuple[int, str]]] = None,
    ):
        """
        Bulk load pre-sorted examples data into the database.

        Args:
            examples_data: Dict mapping quantile_idx -> latent_idx -> list of (score, sequence_uid)
            all_sequences: Tuple of (sequence_uid, token_ids)
            activation_details: Dict mapping latent_idx -> sequence_uid -> activation data
            dataset_info: Optional list of (dataset_id, dataset_name) tuples for each sequence
        """
        logger.info("Bulk loading examples into database...")

        # Clear existing data
        self.clear()

        # Bulk insert sequences
        if dataset_info is not None:
            assert len(all_sequences) == len(dataset_info)
            sequence_data = [
                (seq[1], dataset_info[i][0], dataset_info[i][1])
                for i, seq in enumerate(all_sequences)
            ]
        else:
            sequence_data = [(seq[1], None, None) for seq in all_sequences]

        sequence_uids = [seq[0] for seq in all_sequences]
        self._insert_sequences_bulk(sequence_data, sequence_uids)

        # Bulk insert examples. This is more efficient than inserting one by one.
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            total_examples = sum(
                len(examples)
                for q_data in examples_data.values()
                for examples in q_data.values()
            )
            example_ids = []
            activation_details_list = []

            with tqdm(total=total_examples, desc="Storing examples") as pbar:
                for quantile_idx, latent_data in examples_data.items():
                    for latent_idx, examples in latent_data.items():
                        for score, sequence_uid in examples:
                            cursor.execute(
                                "INSERT INTO examples (sequence_uid, score, latent_idx, quantile_idx) VALUES (?, ?, ?, ?)",
                                (
                                    int(sequence_uid),
                                    float(score),
                                    int(latent_idx),
                                    int(quantile_idx),
                                ),
                            )
                            example_ids.append(cursor.lastrowid)

                            # Prepare activation details if they exist
                            if (
                                activation_details
                                and latent_idx in activation_details
                                and sequence_uid in activation_details[latent_idx]
                            ):
                                detail = activation_details[latent_idx][sequence_uid]

                                if (
                                    isinstance(detail, tuple)
                                    or self.storage_format == "dense"
                                ):
                                    activation_details_list.append(
                                        self.activation_handler.format_for_database(
                                            detail, cursor.lastrowid
                                        )
                                    )
                                else:
                                    assert (
                                        self.storage_format == "sparse"
                                    ), "Dense storage format does not support Nx2 array format"
                                    # Handle Nx2 array format
                                    positions = detail[:, 0].astype(np.int32)
                                    values_as_int32 = detail[:, 1].astype(np.int32)
                                    values = values_as_int32.view(np.float32)
                                    activation_details_list.append(
                                        (
                                            cursor.lastrowid,
                                            positions.tobytes(),
                                            values.tobytes(),
                                        )
                                    )

                            pbar.update(1)

                conn.commit()

        # Bulk insert activation details
        if activation_details_list:
            self._insert_activation_details_bulk(activation_details_list)

        logger.info(f"Successfully loaded {total_examples} examples into database")

    def _insert_sequences_bulk(
        self,
        all_sequences: List[Tuple[torch.Tensor, int, str]],
        sequence_uids: Optional[List[int]] = None,
    ):
        """Bulk insert sequences into the database."""
        # all_sequences is a list of tuples of (token_ids, dataset_id, dataset_name)
        binary_data_list = [
            self._prepare_sequence_data(seq[0]) for seq in all_sequences
        ]
        if sequence_uids is None:
            sequence_uids = [
                self._sequence_uid(binary_data) for binary_data in binary_data_list
            ]

        with self.db_manager.get_connection(enable_foreign_keys=False) as conn:
            cursor = conn.cursor()

            for i in range(len(all_sequences)):
                seq_idx = sequence_uids[i]
                if seq_idx in self._sequence_uid_cache:
                    continue  # skip if already exists

                binary_data = binary_data_list[i]
                token_ids, dataset_id, dataset_name = all_sequences[i]
                # Get text if tokenizer available
                text = None
                if self.tokenizer is not None:
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

                cursor.execute(
                    "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        seq_idx,
                        binary_data,
                        text,
                        len(token_ids),
                        dataset_id,
                        dataset_name,
                    ),
                )
                self._sequence_uid_cache.add(seq_idx)

        return sequence_uids

    def merge(
        self,
        source_store: "MaxActStore",
        maintain_top_k: bool = True,
        sequence_uid_offset: Optional[Union[int, str]] = None,
    ):
        """
        Merge another MaxActStore into this store.

        Args:
            source_store: Source store to merge from
            maintain_top_k: Whether to maintain top-k constraint
            sequence_uid_offset: Offset for sequence indices ("auto", int, or None)
        """
        if self.storage_format != source_store.storage_format:
            raise ValueError(
                f"Storage format mismatch: {self.storage_format} vs {source_store.storage_format}"
            )

        logger.info(f"Merging store from {source_store.db_manager.db_path}")

        # Get existing sequence indices
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sequence_uid FROM sequences")
            existing_indices = set(row[0] for row in cursor.fetchall())

        # Read source data
        with source_store.db_manager.get_connection() as source_conn:
            source_cursor = source_conn.cursor()

            # Get sequences, examples, and activation details
            source_cursor.execute(
                "SELECT sequence_uid, token_ids, text, sequence_length, dataset_id, dataset_name FROM sequences"
            )
            source_sequences = source_cursor.fetchall()

            source_cursor.execute(
                "SELECT sequence_uid, score, latent_idx, quantile_idx, metadata FROM examples ORDER BY score DESC"
            )
            source_examples = source_cursor.fetchall()

            if self.storage_format == "dense":
                source_cursor.execute(
                    """
                    SELECT e.sequence_uid, ad.activation_values
                    FROM activation_details ad
                    JOIN examples e ON ad.example_id = e.example_id
                """
                )
            else:
                source_cursor.execute(
                    """
                    SELECT e.sequence_uid, ad.positions, ad.activation_values
                    FROM activation_details ad
                    JOIN examples e ON ad.example_id = e.example_id
                """
                )
            source_activation_details = source_cursor.fetchall()

        if not source_sequences:
            logger.info("Source store is empty")
            return

        # Create sequence index mapping
        sequence_uid_mapping = {}

        if sequence_uid_offset == "auto":
            sequence_uid_offset = max(existing_indices) + 1 if existing_indices else 0

        if sequence_uid_offset is not None:
            for source_seq_data in source_sequences:
                source_seq_idx = source_seq_data[0]
                new_seq_idx = source_seq_idx + sequence_uid_offset
                if new_seq_idx in existing_indices:
                    raise ValueError(f"Index conflict: {new_seq_idx} already exists")
                sequence_uid_mapping[source_seq_idx] = new_seq_idx
                existing_indices.add(new_seq_idx)
        else:
            next_idx = max(existing_indices) + 1 if existing_indices else 0
            for source_seq_data in source_sequences:
                source_seq_idx = source_seq_data[0]
                if source_seq_idx in existing_indices:
                    while next_idx in existing_indices:
                        next_idx += 1
                    sequence_uid_mapping[source_seq_idx] = next_idx
                    existing_indices.add(next_idx)
                    next_idx += 1
                else:
                    sequence_uid_mapping[source_seq_idx] = source_seq_idx
                    existing_indices.add(source_seq_idx)

        # Insert sequences with new indices
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            sequences_to_insert = []
            for source_seq_data in source_sequences:
                (
                    source_seq_idx,
                    token_ids_blob,
                    text,
                    seq_length,
                    dataset_id,
                    dataset_name,
                ) = source_seq_data
                new_seq_idx = sequence_uid_mapping[source_seq_idx]
                sequences_to_insert.append(
                    (
                        new_seq_idx,
                        token_ids_blob,
                        text,
                        seq_length,
                        dataset_id,
                        dataset_name,
                    )
                )

            cursor.executemany(
                "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
                sequences_to_insert,
            )
            conn.commit()

        # Prepare activation details lookup
        activation_details_by_seq = {}
        for detail_data in source_activation_details:
            if self.storage_format == "dense":
                source_seq_idx, values_blob = detail_data
                activation_details_by_seq[source_seq_idx] = (values_blob,)
            else:
                source_seq_idx, positions_blob, values_blob = detail_data
                activation_details_by_seq[source_seq_idx] = (
                    positions_blob,
                    values_blob,
                )

        # Insert examples and activation details
        example_data_to_insert = []
        for source_example_data in tqdm(source_examples, desc="Processing examples"):
            source_seq_idx, score, latent_idx, quantile_idx, metadata = (
                source_example_data
            )
            new_seq_idx = sequence_uid_mapping[source_seq_idx]
            additional_data = json.loads(metadata) if metadata else None
            example_data_to_insert.append(
                (score, new_seq_idx, latent_idx, quantile_idx, additional_data)
            )

        example_ids = self._insert_examples_bulk(example_data_to_insert)

        # Insert activation details
        activation_details_to_insert = []
        for i, source_example_data in enumerate(source_examples):
            source_seq_idx = source_example_data[0]
            example_id = example_ids[i]

            if source_seq_idx in activation_details_by_seq:
                if self.storage_format == "dense":
                    values_blob = activation_details_by_seq[source_seq_idx][0]
                    values = np.frombuffer(values_blob, dtype=np.float32)
                    activation_details_to_insert.append((example_id, values.tobytes()))
                else:
                    positions_blob, values_blob = activation_details_by_seq[
                        source_seq_idx
                    ]
                    activation_details_to_insert.append(
                        (example_id, positions_blob, values_blob)
                    )

        if activation_details_to_insert:
            self._insert_activation_details_bulk(activation_details_to_insert)

        if maintain_top_k:
            self._maintain_top_k()

        logger.info(f"Successfully merged {len(source_examples)} examples")

    def set_dataset_info(
        self,
        dataset_id: Optional[int] = None,
        dataset_name: Optional[str] = None,
        overwrite_existing: bool = True,
    ) -> int:
        """Set dataset info for all sequences."""
        if dataset_id is None and dataset_name is None:
            return 0

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            set_clauses = []
            params = []

            if dataset_id is not None:
                set_clauses.append("dataset_id = ?")
                params.append(dataset_id)

            if dataset_name is not None:
                set_clauses.append("dataset_name = ?")
                params.append(dataset_name)

            query = f"UPDATE sequences SET {', '.join(set_clauses)}"

            if not overwrite_existing:
                query += " WHERE dataset_id IS NULL AND dataset_name IS NULL"

            cursor.execute(query, params)
            updated_count = cursor.rowcount
            conn.commit()

        logger.info(f"Updated dataset info for {updated_count} sequences")
        return updated_count

    def get_group_capacity_info(self) -> Dict[tuple, float]:
        """Get capacity info for all existing groups. Returns a dict of group keys to min scores. If a group is not full, the corresponding min score is not returned."""
        groups = {}

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT e.latent_idx, e.quantile_idx, s.dataset_name, 
                       COUNT(*) as count, MIN(e.score) as min_score
                FROM examples e
                JOIN sequences s ON e.sequence_uid = s.sequence_uid
                GROUP BY e.latent_idx, e.quantile_idx, s.dataset_name
            """
            )

            for (
                latent_idx,
                quantile_idx,
                dataset_name,
                count,
                min_score,
            ) in cursor.fetchall():
                group_key = self.group_builder.build_group_key(
                    latent_idx, quantile_idx, dataset_name
                )
                if count >= self._max_examples if self._max_examples else False:
                    groups[group_key] = min_score
        return groups

    def get_num_sequences(self) -> int:
        """Get number of sequences in database."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sequences")
            return cursor.fetchone()[0]

    def create_async_writer(
        self,
        buffer_size: int = 1000,
        flush_interval: float = 30.0,
        auto_maintain_top_k: bool = True,
        enable_early_filtering: bool = True,
        use_memory_db: bool = True,
        sync_interval: float = 300.0,
    ) -> "AsyncMaxActStoreWriter":
        """Create an async writer for this store."""
        if enable_early_filtering:
            self._top_k_proxy = MultiProcessTopKProxy(self.get_group_capacity_info)

        return AsyncMaxActStoreWriter(
            db_path=self.db_manager.db_path,
            tokenizer=self.tokenizer,
            storage_format=self.storage_format,
            per_dataset=self.per_dataset,
            max_examples=self.max_examples,
            buffer_size=buffer_size,
            flush_interval=flush_interval,
            auto_maintain_top_k=auto_maintain_top_k,
            top_k_proxy=self._top_k_proxy,
            group_builder=self.group_builder,
            use_memory_db=use_memory_db,
            sync_interval=sync_interval,
        )


# ================================
# ASYNC WRITER IMPLEMENTATION
# ================================


class AsyncMaxActStoreWriter:
    """
    Asynchronous writer for MaxActStore with performance optimizations.

    Features:
    - Background process for database writes
    - Batch buffering to reduce I/O
    - Early filtering of low-scoring batches
    - Automatic top-k maintenance
    """

    def __init__(
        self,
        db_path: Path,
        tokenizer=None,
        storage_format: str = "sparse",
        max_examples: Optional[int] = None,
        per_dataset: bool = False,
        buffer_size: int = 1000,
        flush_interval: float = 30.0,
        auto_maintain_top_k: bool = True,
        top_k_proxy: Optional[TopKProxy] = None,
        group_builder: Optional[GroupKeyBuilder] = None,
        use_memory_db: bool = True,
        sync_interval: float = 300.0,
    ):
        """Initialize async writer."""
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.storage_format = storage_format
        self.per_dataset = per_dataset
        self.max_examples = max_examples
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.auto_maintain_top_k = auto_maintain_top_k
        self.top_k_proxy = top_k_proxy
        self.group_builder = group_builder
        self.use_memory_db = use_memory_db
        self.sync_interval = sync_interval
        assert (
            self.top_k_proxy is None or self.group_builder is not None
        ), "Group key builder must be provided when top_k_proxy is given."

        # Create directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Communication with background process
        self.request_queue = mp.Queue()
        self.error_queue = mp.Queue()
        self.writer_process = None
        self.is_running = False

        # Buffer for accumulating batches
        self.buffer = []
        self.buffer_count = 0
        self.last_flush_time = time.time()

        # Thread safety
        self._lock = mp.Lock()

    def start(self):
        """Start the background writer process."""
        if self.is_running:
            return

        self.writer_process = mp.Process(
            target=self._writer_worker,
            args=(
                self.request_queue,
                self.error_queue,
                self.db_path,
                self.max_examples,
                self.storage_format,
                self.per_dataset,
                self.tokenizer,
                self.top_k_proxy,
                self.use_memory_db,
                self.sync_interval,
            ),
            daemon=True,
        )
        self.writer_process.start()
        self.is_running = True
        logger.info(f"Started async writer process for {self.db_path}")

    def stop(self, timeout: float = 60 * 60.0):  # 1 hour
        """Stop the background writer process."""
        if not self.is_running:
            return

        try:
            # Flush remaining data
            self._flush_buffer()

            # Send shutdown command
            self.request_queue.put(WriteRequest(WriteCommand.SHUTDOWN))

            # Wait for process to finish
            if self.writer_process:
                self.writer_process.join(timeout=timeout)
                if self.writer_process.is_alive():
                    logger.warning(
                        "Writer process didn't finish gracefully, terminating..."
                    )
                    self.writer_process.terminate()
                    self.writer_process.join(timeout=5.0)
                    if self.writer_process.is_alive():
                        logger.error(
                            "Writer process couldn't be terminated, killing..."
                        )
                        self.writer_process.kill()
        finally:
            self.is_running = False
            self.writer_process = None
            logger.info("Stopped async writer process")

    def add_batch_examples(
        self,
        scores_per_example: torch.Tensor,
        input_ids_batch: torch.Tensor | List[torch.Tensor],
        attention_mask_batch: Optional[torch.Tensor] = None,
        scores_per_token_batch: Optional[torch.Tensor] = None,
        additional_data_batch: Optional[List[dict]] = None,
        latent_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
        quantile_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
    ):
        """
        Add batch examples to buffer for background writing.

        Args:
            scores_per_example: Tensor of shape (batch_size,) containing activation scores
            input_ids_batch: Tensor of shape (batch_size, seq_len) containing token IDs. Can be a list of tensors or a single tensor.
            attention_mask_batch: Optional tensor of shape (batch_size, seq_len) for masking
            scores_per_token_batch: Optional tensor of shape (batch_size, seq_len) with per-token scores
            additional_data_batch: Optional list of dictionaries with extra metadata per example
            latent_idx: Optional latent dimension index (scalar or array of length batch_size)
            quantile_idx: Optional quantile index (scalar or array of length batch_size)
            dataset_name: Optional name of the dataset
            dataset_id: Optional numeric dataset identifier

        Raises:
            RuntimeError: If the async writer is not running
        """
        if not self.is_running:
            raise RuntimeError(
                "AsyncMaxActStoreWriter is not running. Call start() first."
            )

        self._check_for_errors()

        batch_size = len(scores_per_example)
        # Process tensors using centralized logic
        processed_input_ids, processed_scores_per_token = process_batch_tensors(
            input_ids_batch, attention_mask_batch, scores_per_token_batch
        )

        # Create batch data
        batch_data = BatchData(
            scores_per_example=scores_per_example.cpu().numpy(),
            input_ids_batch=processed_input_ids,
            scores_per_token_batch=processed_scores_per_token,
            additional_data_batch=additional_data_batch,
            latent_idx=latent_idx,
            quantile_idx=quantile_idx,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
        )

        # Early filtering if enabled
        if self.top_k_proxy is not None and batch_size >= 10:
            if not self._batch_qualifies_for_processing(batch_data):
                logger.debug(
                    f"Skipped batch of {batch_size} examples - no scores exceed thresholds"
                )
                return

        with self._lock:
            self.buffer.append(batch_data)
            self.buffer_count += batch_size

            # Check if we should flush
            should_flush = (
                self.buffer_count >= self.buffer_size
                or time.time() - self.last_flush_time >= self.flush_interval
            )

            if should_flush:
                self._flush_buffer()

    def _wait_for_queue_space(self):
        """Wait if the request queue is too full to prevent memory issues."""
        while self.request_queue.qsize() > 100:
            time.sleep(0.01)  # Small sleep to avoid busy waiting

    def _batch_qualifies_for_processing(self, batch_data: BatchData) -> bool:
        """Check if any examples in batch could qualify for top-k."""
        batch_size = batch_data.scores_per_example.shape[0]
        checked_groups = set()

        for i in range(batch_size):
            score = float(batch_data.scores_per_example[i])

            latent_idx = batch_data.get_per_example_latent_idx(i, batch_size)
            quantile_idx = batch_data.get_per_example_quantile_idx(i, batch_size)
            dataset_name = batch_data.dataset_name

            group_key = self.group_builder.build_group_key(
                latent_idx, quantile_idx, dataset_name
            )
            if group_key not in checked_groups:
                checked_groups.add(group_key)
                if self.top_k_proxy.get_min_score(group_key) < score:
                    return True

        return False

    def _flush_buffer(self):
        """Flush current buffer to background writer (thread-unsafe)."""
        if not self.buffer:
            return

        self._wait_for_queue_space()

        # Send all buffered data
        for batch_data in self.buffer:
            request = WriteRequest(WriteCommand.ADD_BATCH, data=batch_data)
            self.request_queue.put(request)

        # Clear buffer
        self.buffer.clear()
        self.buffer_count = 0
        self.last_flush_time = time.time()

        # Trigger top-k maintenance
        if self.auto_maintain_top_k:
            self.request_queue.put(WriteRequest(WriteCommand.MAINTAIN_TOP_K))

    def flush(self):
        """Force flush any buffered data."""
        with self._lock:
            self._flush_buffer()

    def sync_to_disk(self):
        """Force sync to disk (only works when use_memory_db=True)."""
        if not self.is_running:
            raise RuntimeError(
                "AsyncMaxActStoreWriter is not running. Call start() first."
            )

        if self.use_memory_db:
            self.request_queue.put(WriteRequest(WriteCommand.SYNC_TO_DISK))
            logger.info("Requested manual sync to disk")

    def _check_for_errors(self):
        """Check if background process has reported errors."""
        try:
            error = self.error_queue.get_nowait()
            raise RuntimeError(f"Background writer error: {error}")
        except queue.Empty:
            pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    @staticmethod
    def _writer_worker(
        request_queue: mp.Queue,
        error_queue: mp.Queue,
        db_path: Path,
        max_examples: Optional[int],
        storage_format: str,
        per_dataset: bool,
        tokenizer,
        top_k_proxy: Optional[TopKProxy] = None,
        use_memory_db: bool = True,
        sync_interval: float = 15 * 60.0,
    ):
        """Background worker process that handles database writes."""
        try:
            if use_memory_db:
                # Create in-memory database and copy from disk
                memory_db_manager = DatabaseManager(":memory:")

                # Copy existing database to memory if it exists
                if db_path.exists():
                    with sqlite3.connect(db_path) as disk_conn:
                        disk_conn.backup(memory_db_manager.conn)
                    logger.info(f"Copied disk database {db_path} to memory")
                else:
                    logger.info(f"Starting with empty in-memory database")

                # Create store with custom connection
                store = AsyncMaxActStoreWriter._create_memory_store(
                    memory_db_manager,
                    max_examples,
                    tokenizer,
                    storage_format,
                    per_dataset,
                    top_k_proxy,
                )

                last_sync_time = time.time()
                logger.info(
                    f"Background writer started with in-memory database for {db_path}"
                )
            else:
                # Original behavior - work directly with disk
                store = MaxActStore(
                    db_path=db_path,
                    max_examples=max_examples,
                    tokenizer=tokenizer,
                    storage_format=storage_format,
                    per_dataset=per_dataset,
                    top_k_proxy=top_k_proxy,
                )
                logger.info(f"Background writer started for {db_path}")

            while True:
                try:
                    request = request_queue.get(timeout=1.0)
                except queue.Empty:
                    # Check if we need to sync to disk
                    if use_memory_db and time.time() - last_sync_time >= sync_interval:
                        AsyncMaxActStoreWriter._sync_memory_to_disk(
                            memory_db_manager, db_path
                        )
                        last_sync_time = time.time()
                    continue

                if request.command == WriteCommand.SHUTDOWN:
                    logger.info("Background writer received shutdown command")
                    break
                elif request.command == WriteCommand.ADD_BATCH:
                    start_time = time.time()
                    store._process_batch_data(request.data)
                    # logger.info(f"Finished processing batch of {len(request.data.scores_per_example)} examples in {time.time() - start_time:.2f} seconds")
                elif request.command == WriteCommand.MAINTAIN_TOP_K:
                    start_time = time.time()
                    store._maintain_top_k()
                    # logger.info(f"Finished maintaining top-k in {time.time() - start_time:.2f} seconds")
                elif request.command == WriteCommand.SYNC_TO_DISK:
                    if use_memory_db:
                        AsyncMaxActStoreWriter._sync_memory_to_disk(
                            memory_db_manager, db_path
                        )
                        last_sync_time = time.time()

        except Exception as e:
            logger.error(f"Background writer error: {e}")
            error_queue.put(str(e))
            raise e
        finally:
            # Final sync to disk before shutdown
            if use_memory_db:
                try:
                    AsyncMaxActStoreWriter._sync_memory_to_disk(
                        memory_db_manager, db_path
                    )
                    memory_db_manager.close()
                    logger.info("Final sync to disk completed")
                except Exception as e:
                    logger.error(f"Failed final sync to disk: {e}")
                    raise e
            logger.info("Background writer process finished")

    @staticmethod
    def _create_memory_store(
        db_manager: DatabaseManager,
        max_examples: Optional[int],
        tokenizer,
        storage_format: str,
        per_dataset: bool,
        top_k_proxy: Optional[TopKProxy] = None,
    ):
        """Create a MaxActStore that uses the given in-memory connection."""
        # Create store instance
        store = MaxActStore(
            db_path=db_manager.db_path,
            max_examples=max_examples,
            tokenizer=tokenizer,
            storage_format=storage_format,
            per_dataset=per_dataset,
            top_k_proxy=top_k_proxy,
        )

        store.db_manager = db_manager
        return store

    @staticmethod
    def _sync_memory_to_disk(db_manager: DatabaseManager, disk_path: Path):
        """Sync in-memory database to disk using SQLite backup API."""
        try:
            # Ensure directory exists
            disk_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(disk_path) as disk_conn:
                db_manager.conn.backup(disk_conn)
            logger.debug(f"Synced memory database to {disk_path}")
        except Exception as e:
            logger.error(f"Failed to sync memory database to disk: {e}")
            raise


# ================================
# READ-ONLY STORE
# ================================


class ReadOnlyMaxActStore(MaxActStore):
    """
    Read-only version of MaxActStore optimized for concurrent access.

    Uses read-only database connections and WAL mode for better concurrency.
    All write operations are blocked with helpful error messages.
    """

    def __init__(self, db_path: Path, tokenizer=None):
        """Initialize read-only store."""
        legacy_converter(db_path)
        self._setup_db_manager(db_path)
        self._handle_config(None, None)
        self.tokenizer = tokenizer
        self.activation_handler = ActivationDetailsHandler(self._storage_format)

        logger.info(f"Initialized ReadOnlyMaxActStore for {self.db_manager.db_path}")

    def _setup_db_manager(self, db_path: Path):
        """Setup the database manager."""
        self.db_manager = DatabaseManager(db_path, readonly=True)

    def _setup_wal_mode(self):
        """Enable WAL mode for better concurrent read access."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                mode = cursor.fetchone()[0]
                if mode.upper() == "WAL":
                    logger.debug("WAL mode enabled for better concurrency")
        except Exception as e:
            logger.warning(f"Failed to set WAL mode: {e}")

    def _handle_config(
        self, max_examples: Optional[int], storage_format: Optional[str]
    ):
        """Handle configuration storage and validation."""
        assert max_examples is None, "You cannot set max_examples for a read-only store"
        assert (
            storage_format is None
        ), "You cannot set storage_format for a read-only store"

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Read existing config
            cursor.execute("SELECT key, value FROM config")
            existing_config = dict(cursor.fetchall())

            if "storage_format" not in existing_config:
                raise ValueError("No existing storage_format found and none provided")
            storage_format = existing_config["storage_format"]

            if storage_format not in ["sparse", "dense"]:
                raise ValueError(
                    f"storage_format must be 'sparse' or 'dense', got {storage_format}"
                )

            # Set attributes
            self._storage_format = storage_format
            self._max_examples = max_examples

            conn.commit()

    def add_example(self, *args, **kwargs):
        raise NotImplementedError("You cannot add examples to a read-only store")

    def add_batch_examples(self, *args, **kwargs):
        raise NotImplementedError("You cannot add batch examples to a read-only store")

    def fill(self, *args, **kwargs):
        raise NotImplementedError("You cannot fill a read-only store")

    def merge(self, *args, **kwargs):
        raise NotImplementedError("You cannot merge a read-only store")

    def set_dataset_info(self, *args, **kwargs):
        raise NotImplementedError("You cannot set dataset info for a read-only store")
