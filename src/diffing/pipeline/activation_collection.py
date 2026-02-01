"""
This module collects and stores activations from specified layers of a language model.
The module supports different data types, handles large datasets efficiently
through sharding, and includes options for storing tokens alongside activations.
It can process untokenized and non-chat formatted data.
"""

import sys

sys.path.append(".")

from typing import List, Optional, Any
from dictionary_learning.cache import ActivationCache
from datasets import Dataset
from loguru import logger
import torch
from ..utils import ModelConfig
from pathlib import Path
import jinja2

from ..utils import load_model_from_config
from ..utils.configs import get_safe_model_id
from ..utils.model import AnyTokenizer


def format_chat_data(
    dataset: Dataset, tokenizer: AnyTokenizer, messages_column: str = "messages"
) -> List[str]:
    """
    Convert chat dataset to formatted text using tokenizer's chat template.

    Args:
        dataset: Dataset containing chat messages
        tokenizer: Tokenizer with apply_chat_template method
        messages_column: Column name containing the chat messages

    Returns:
        List of formatted chat strings

    Assumptions:
        - Tokenizer has apply_chat_template method (common for modern chat models)
        - Messages are in standard chat format (list of dicts with 'role' and 'content')
        - Messages with non-alternating user/assistant roles will be skipped
    """
    logger.info(f"Formatting {len(dataset)} chat samples using chat template")

    formatted_texts = []
    skipped_count = 0

    for sample in dataset:
        messages = sample[messages_column]
        try:
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        except jinja2.exceptions.TemplateError as e:
            if (
                "Conversation roles must alternate user/assistant/user/assistant/"
                in str(e)
            ):
                # Skip messages with non-alternating roles
                skipped_count += 1
                logger.debug(f"Skipping sample due to non-alternating roles: {e}")
                continue
            else:
                # Re-raise other template errors
                logger.error(messages)
                logger.error(f"Error formatting chat sample: {e}")
                raise e
        except Exception as e:
            logger.error(messages)
            logger.error(f"Error formatting chat sample: {e}")
            raise e

    if skipped_count > 0:
        logger.info(
            f"Skipped {skipped_count} samples due to non-alternating conversation roles"
        )

    return formatted_texts


def tokenize_texts(
    texts: List[str], tokenizer: AnyTokenizer, context_len: int = 1024
) -> List[str]:
    """
    Tokenize and truncate texts to specified context length.

    Args:
        texts: List of text strings to tokenize
        tokenizer: Tokenizer to use
        context_len: Maximum context length

    Returns:
        List of tokenized and decoded texts (ensuring proper truncation)
    """
    logger.info(f"Tokenizing {len(texts)} texts with context length {context_len}")

    tokenized_texts = []
    for text in texts:
        requires_special_tokens = (
            tokenizer.bos_token is not None and tokenizer.bos_token not in text
        )
        # Tokenize with truncation
        tokens = tokenizer.encode(
            text,
            max_length=context_len,
            truncation=True,
            add_special_tokens=requires_special_tokens,
        )
        # Decode back to text to ensure consistency
        tokenized_text = tokenizer.decode(tokens, skip_special_tokens=False)
        tokenized_texts.append(tokenized_text)

    return tokenized_texts


def collect_activations(
    model_cfg: ModelConfig,
    dataset: Dataset,
    layers: List[int],
    activation_store_dir: str,
    dataset_name: str,
    dataset_split: str = "train",
    max_samples: int = 10**6,
    max_tokens: int = 10**8,
    batch_size: int = 64,
    context_len: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
    store_tokens: bool = False,
    overwrite: bool = False,
    disable_multiprocessing: bool = False,
    text_column: Optional[str] = None,
    messages_column: str = "messages",
    is_chat_data: bool = True,
    ignore_first_n_tokens: int = 0,
    token_level_replacement: Optional[Any] = None,
    default_text_column: str = "text",
) -> None:
    """
    Collect and store activations from specified layers of a language model.

    Args:
        model_name: Name/path of the model to load
        dataset: Pre-loaded dataset to process
        layers: Layer indices to collect activations from
        activation_store_dir: Directory to store activations
        dataset_name: Name of the dataset
        dataset_split: Split of the dataset
        max_samples: Maximum number of samples to process
        max_tokens: Maximum number of tokens to process
        batch_size: Batch size for processing
        context_len: Maximum context length
        dtype: Data type for activations
        store_tokens: Whether to store tokens alongside activations
        overwrite: Whether to overwrite existing activations
        disable_multiprocessing: Whether to disable multiprocessing
        text_column: Column name for pre-formatted text (overrides chat formatting)
        messages_column: Column name for chat messages (used if is_chat_data=True)
        is_chat_data: Whether the data needs chat formatting
        attn_implementation: Attention implementation to use for model loading
        ignore_first_n_tokens: Number of tokens to ignore at the beginning of each sample
        token_level_replacement: Token-level replacement configuration
        default_text_column: Default column name for text data

    Assumptions:
        - If is_chat_data=True, tokenizer has apply_chat_template method
        - Chat messages are in standard format (list of dicts with 'role' and 'content')
    """

    if len(layers) == 0:
        raise ValueError("Must provide at least one layer")

    # Set up output directory
    store_dir = Path(activation_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    model_name_clean = get_safe_model_id(model_cfg)
    data_split_name = dataset_split + (
        f"_col_{text_column}"
        if text_column is not None and text_column != default_text_column
        else ""
    )
    out_dir = store_dir / model_name_clean / dataset_name / data_split_name

    if not overwrite:
        if (out_dir / "config.json").exists():
            logger.info(
                f"Activations already exist for {model_cfg.model_id} + {dataset_name} ({data_split_name}) - skipping"
            )
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Storing activations in: {out_dir}")

    # Load model and tokenizer
    logger.info(f"Loading model: {model_cfg.model_id}")
    model = load_model_from_config(model_cfg)

    # Inputs can be left on CPU and dispatched internally
    logger.info(f"Model dtype: {model.dtype}")

    # Validate layers
    num_layers = model.num_layers
    for layer in layers:
        if layer >= num_layers:
            raise ValueError(f"Layer {layer} exceeds model layers (0-{num_layers-1})")

    logger.info(f"Collecting activations from layers: {layers}")

    # Set up submodules
    submodules = [model.layers[layer] for layer in layers]
    submodule_names = [f"layer_{layer}" for layer in layers]

    exists, num_toks = ActivationCache.exists(
        out_dir, submodule_names, "out", store_tokens
    )
    if not overwrite and exists:
        logger.info(
            f"Activations already exist (n_toks={num_toks}) for {model_cfg.model_id} + {dataset_name} ({data_split_name}) - skipping"
        )
        return

    d_model = model.hidden_size
    logger.info(f"d_model: {d_model}")

    # Limit dataset size
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    need_special_tokens = False
    # Process data based on format
    if text_column is not None:
        # Use pre-specified text column
        logger.info(f"Using pre-formatted text from column: {text_column}")
        texts = dataset[text_column]
        texts = tokenize_texts(texts, model.tokenizer, context_len)
        need_special_tokens = (
            model.tokenizer.bos_token is not None
            and model.tokenizer.bos_token not in texts[0]
        )
    elif is_chat_data:
        # Format chat data and tokenize
        logger.info("Processing chat data: formatting and tokenizing")
        texts = format_chat_data(dataset, model.tokenizer, messages_column)
        need_special_tokens = (
            model.tokenizer.bos_token is not None
            and model.tokenizer.bos_token not in texts[0]
        )
    else:
        # Use default text column and tokenize
        if default_text_column not in dataset.column_names:
            raise ValueError(
                f"Default text column '{default_text_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}. "
                f"Please specify text_column or ensure is_chat_data=True with proper messages_column."
            )
        logger.info(f"Tokenizing text from column: {default_text_column}")
        texts = tokenize_texts(
            dataset[default_text_column], model.tokenizer, context_len
        )

    logger.info(f"Processing {len(texts)} samples")
    logger.info(f"Need special tokens: {need_special_tokens}")

    # Collect activations
    ActivationCache.collect(
        texts,
        submodules,
        submodule_names,
        model,
        out_dir,
        shuffle_shards=False,
        io="out",
        shard_size=10**7,
        batch_size=batch_size,
        context_len=context_len,
        d_model=d_model,
        last_submodule=submodules[-1],
        max_total_tokens=max_tokens,
        store_tokens=store_tokens,
        multiprocessing=not disable_multiprocessing,
        ignore_first_n_tokens_per_sample=ignore_first_n_tokens,
        overwrite=overwrite,
        token_level_replacement=token_level_replacement,
        dtype=dtype,
        add_special_tokens=need_special_tokens,
    )

    logger.info("Activation collection completed successfully")
