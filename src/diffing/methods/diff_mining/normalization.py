"""
Token normalization utilities for logit diff analysis.

This module provides two independent token processing capabilities:
1. filter_pure_punctuation: Remove tokens that are ONLY punctuation/whitespace
   (after decoding BPE markers). Does NOT modify tokens.
2. normalize_tokens: Decode BPE markers, strip punctuation/whitespace from both ends,
   lowercase, and consolidate similar tokens.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import string
import json
import unicodedata
from collections import defaultdict


# Common BPE byte-to-char mappings (GPT-2/LLaMA style)
# These Unicode characters represent whitespace bytes in BPE tokenization
BPE_WHITESPACE_CHARS = {
    "Ġ": " ",  # U+0120 -> space (0x20)
    "Ċ": "\n",  # U+010A -> newline (0x0A)
    "ĉ": "\t",  # U+0109 -> tab (0x09)
    "č": "\r",  # U+010D -> carriage return (0x0D)
}

# Characters to strip: punctuation + whitespace
STRIP_CHARS = string.punctuation + string.whitespace


def decode_bpe_whitespace(token_str: str) -> str:
    """
    Decode BPE whitespace markers to actual whitespace characters.

    Args:
        token_str: Token string potentially containing BPE markers like Ġ, Ċ

    Returns:
        Token string with BPE markers replaced by actual whitespace

    Examples:
        "Ġhello" -> " hello"
        "---Ċ" -> "---\\n"
        "ĠĠ" -> "  "
    """
    for bpe_char, actual_char in BPE_WHITESPACE_CHARS.items():
        token_str = token_str.replace(bpe_char, actual_char)
    return token_str


def is_punct_or_symbol(char: str) -> bool:
    """
    Check if a character is punctuation or symbol (ASCII or Unicode).

    Uses both string.punctuation for ASCII and unicodedata.category for Unicode.

    Unicode categories checked:
        P* (Punctuation): Pc, Pd, Pe, Pf, Pi, Po, Ps
        S* (Symbol): Sc, Sk, Sm, So

    Args:
        char: A single character to check

    Returns:
        True if character is punctuation or symbol, False otherwise

    Examples:
        "!" -> True (ASCII punctuation)
        "—" -> True (em dash, Unicode Pd)
        "。" -> True (CJK period, Unicode Po)
        "…" -> True (ellipsis, Unicode Po)
        '"' -> True (right double quote U+201D, Unicode Pf)
        "•" -> True (bullet, Unicode Po)
        "→" -> True (arrow, Unicode Sm)
        "a" -> False (letter)
        "5" -> False (digit)
    """
    # Check ASCII punctuation first (fast path)
    if char in string.punctuation:
        return True

    # Check Unicode category
    category = unicodedata.category(char)
    # P* = Punctuation categories, S* = Symbol categories
    return category.startswith("P") or category.startswith("S")


def is_pure_punctuation(token_str: str) -> bool:
    """
    Check if a token is purely punctuation and/or whitespace.

    First decodes BPE whitespace markers (Ġ, Ċ, etc.) to actual whitespace,
    then checks if the entire token consists only of punctuation and whitespace.
    Handles both ASCII and Unicode punctuation/symbols.

    Args:
        token_str: Token string to check (may contain BPE markers)

    Returns:
        True if token is only punctuation/whitespace after BPE decoding, False otherwise

    Examples:
        "..." -> True
        "!" -> True
        "   " -> True
        "---Ċ" -> True (becomes "---\\n" which is punct + whitespace)
        "Ġ" -> True (becomes " " which is whitespace)
        "—\\n\\n" -> True (em dash + newlines, Unicode punctuation)
        "。\\n\\n" -> True (CJK period + newlines, Unicode punctuation)
        "…\\n\\n" -> True (ellipsis + newlines, Unicode punctuation)
        "C++" -> False (contains letters)
        "don't" -> False (contains letters)
        ".ai" -> False (contains letters)
        "Ġthe" -> False (becomes " the" which contains letters)
    """
    # Decode BPE whitespace markers first
    decoded = decode_bpe_whitespace(token_str)

    # Strip whitespace
    stripped = decoded.strip()

    # If nothing remains after stripping whitespace, it's pure whitespace
    if not stripped:
        return True

    # Check if all remaining characters are punctuation or symbols (ASCII or Unicode)
    return all(is_punct_or_symbol(char) for char in stripped)


def filter_punctuation_tokens(token_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove tokens that are purely punctuation/whitespace.

    This function does NOT modify tokens - it only filters them out.
    Keeps tokens like "C++", "don't", ".ai", "Ġthe" but removes "...", "!", "---Ċ", etc.

    Args:
        token_list: List of token dicts with 'token_str' field

    Returns:
        Filtered list with pure-punctuation/whitespace tokens removed
    """
    return [t for t in token_list if not is_pure_punctuation(t["token_str"])]


def is_special_token(token_id: int, tokenizer) -> bool:
    """
    Check if token_id is a special token (BOS, EOS, PAD, etc.).

    Args:
        token_id: The token ID to check
        tokenizer: A HuggingFace tokenizer with all_special_ids attribute

    Returns:
        True if token_id is a special token, False otherwise
    """
    return token_id in tokenizer.all_special_ids


def filter_special_tokens_from_list(
    token_list: List[Dict[str, Any]], tokenizer
) -> List[Dict[str, Any]]:
    """
    Remove special tokens (BOS, EOS, PAD, etc.) from a token list.

    Args:
        token_list: List of token dicts with 'token_id' field
        tokenizer: A HuggingFace tokenizer with all_special_ids attribute

    Returns:
        Filtered list with special tokens removed
    """
    special_ids = set(tokenizer.all_special_ids)
    return [t for t in token_list if t["token_id"] not in special_ids]


def normalize_token(token_str: str) -> str:
    """
    Normalize a token string for consolidation.

    Performs the following transformations:
    1. Decode BPE whitespace markers (Ġ -> space, Ċ -> newline, etc.)
    2. Strip punctuation AND whitespace from both ends
    3. Lowercase the result

    Args:
        token_str: Original token string (may contain BPE markers)

    Returns:
        Normalized token string, or empty string if token becomes empty

    Examples:
        ".ai" -> "ai"
        "Ġthe" -> "the"
        "AI" -> "ai"
        "...hello..." -> "hello"
        "---Ċ" -> "" (empty - only punct/whitespace)
        "C++" -> "c++" (internal punctuation preserved)
        "don't" -> "don't" (internal punctuation preserved)
    """
    # Step 1: Decode BPE whitespace markers
    decoded = decode_bpe_whitespace(token_str)

    # Step 2: Strip punctuation and whitespace from both ends
    result = decoded.strip(STRIP_CHARS)

    # Step 3: Lowercase
    return result.lower()


def consolidate_tokens(
    token_list: List[Dict[str, Any]], total_positions: int
) -> List[Dict[str, Any]]:
    """
    Consolidate tokens with the same normalized form.

    First filters out pure punctuation/whitespace tokens, then normalizes remaining
    tokens (decode BPE, strip punct/whitespace from ends, lowercase) and consolidates
    tokens that normalize to the same string.

    Consolidated tokens have:
    - Counts summed across all variants
    - token_id set to -1 to indicate combined/normalized token
    - Occurrence rates recalculated

    Args:
        token_list: List of token dicts with fields:
            - token_str: str
            - token_id: int
            - count_positive: int
            - count_negative: int
            - positive_occurrence_rate: float
            - negative_occurrence_rate: float
        total_positions: Total number of positions (for recalculating rates)

    Returns:
        List of consolidated token dicts, sorted by positive_occurrence_rate descending
    """
    # Group tokens by normalized form
    normalized_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for token_dict in token_list:
        normalized = normalize_token(token_dict["token_str"])
        if normalized:  # Skip empty tokens (pure punctuation/whitespace)
            normalized_groups[normalized].append(token_dict)

    # Consolidate each group
    consolidated: List[Dict[str, Any]] = []

    for normalized_str, variants in normalized_groups.items():
        # Sum counts across all variants
        total_count_positive = sum(v["count_positive"] for v in variants)
        total_count_negative = sum(v["count_negative"] for v in variants)

        # Set token_id to -1 to indicate this is a combined/normalized token
        token_id = -1

        # Recalculate occurrence rates
        positive_occurrence_rate = (
            (total_count_positive / total_positions) * 100
            if total_positions > 0
            else 0.0
        )
        negative_occurrence_rate = (
            (total_count_negative / total_positions) * 100
            if total_positions > 0
            else 0.0
        )

        consolidated.append(
            {
                "token_str": normalized_str,
                "token_id": token_id,
                "count_positive": total_count_positive,
                "count_negative": total_count_negative,
                "positive_occurrence_rate": positive_occurrence_rate,
                "negative_occurrence_rate": negative_occurrence_rate,
            }
        )

    # Sort by positive_occurrence_rate descending
    consolidated.sort(key=lambda x: x["positive_occurrence_rate"], reverse=True)

    return consolidated


def process_token_list(
    token_list: List[Dict[str, Any]],
    total_positions: int,
    filter_punctuation: bool = True,
    normalize: bool = False,
    filter_special_tokens: bool = False,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    """
    Process token list with independent filtering and normalization.

    Args:
        token_list: Raw token list
        total_positions: Total positions (for recalculating rates if normalizing)
        filter_punctuation: If True, remove tokens that are ONLY punctuation/whitespace
            (after decoding BPE markers). Does NOT modify tokens.
        normalize: If True, decode BPE markers, strip punct/whitespace from ends,
            lowercase, and consolidate similar tokens.
        filter_special_tokens: If True, remove special tokens (BOS, EOS, PAD, etc.).
            Requires tokenizer to be provided.
        tokenizer: HuggingFace tokenizer (required if filter_special_tokens=True)

    Returns:
        Processed token list
    """
    result = token_list

    # Filter special tokens first (if enabled)
    if filter_special_tokens:
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when filter_special_tokens=True"
            )
        result = filter_special_tokens_from_list(result, tokenizer)

    if filter_punctuation and not normalize:
        # Only filter, don't modify tokens
        result = filter_punctuation_tokens(result)

    if normalize:
        # Normalize includes filtering (empty normalized strings are skipped)
        result = consolidate_tokens(result, total_positions)

    return result


def normalize_token_list(
    token_list: List[Dict[str, Any]], total_positions: int
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use process_token_list() instead.

    This function is kept for backwards compatibility.
    It applies both punctuation filtering and normalization.
    """
    return process_token_list(
        token_list, total_positions, filter_punctuation=True, normalize=True
    )


def load_fraction_positive_tokens(
    global_stats_file: Path,
    k: int,
    total_positions: Optional[int] = None,
    filter_punctuation: bool = False,
    normalize: bool = False,
    filter_special_tokens: bool = False,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    """
    Load global token stats and return top-K tokens sorted by fraction of positive logit diffs.

    This is the canonical function for loading tokens with filtering/normalization.
    The processing order is: filter/normalize ALL tokens first, THEN take top K.
    This ensures lower-ranked tokens "move up" when higher-ranked tokens are filtered out.

    Args:
        global_stats_file: Path to {dataset}_global_token_stats.json
        k: Number of top tokens to return (after filtering/normalization)
        total_positions: Override for total_positions (uses value from JSON if None)
        filter_punctuation: Whether to filter out pure punctuation/whitespace tokens
        normalize: Whether to normalize (lowercase, strip, consolidate) tokens
        filter_special_tokens: Whether to filter out special tokens (BOS, EOS, PAD, etc.)
        tokenizer: HuggingFace tokenizer (required if filter_special_tokens=True)

    Returns:
        List of token dicts with keys: token_id, token_str, count_positive, count_negative,
        positive_occurrence_rate, negative_occurrence_rate, fraction_positive
    """
    with open(global_stats_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use provided total_positions or fall back to JSON value
    if total_positions is None:
        total_positions = data["total_positions_analyzed"]

    token_stats = data["global_token_stats"]

    # Convert to format compatible with top_positive
    all_tokens = []
    for stat in token_stats:
        count_positive = stat["count_positive"]
        count_negative = total_positions - count_positive
        fraction_positive = (
            count_positive / total_positions if total_positions > 0 else 0.0
        )

        all_tokens.append(
            {
                "token_id": stat["token_id"],
                "token_str": stat["token"],  # Note: JSON uses "token" not "token_str"
                "count_positive": count_positive,
                "count_negative": count_negative,
                "positive_occurrence_rate": fraction_positive
                * 100,  # Convert to percentage
                "negative_occurrence_rate": (
                    (count_negative / total_positions) * 100
                    if total_positions > 0
                    else 0.0
                ),
                "fraction_positive": fraction_positive,  # Original 0-1 fraction
                "sum_logit_diff": stat.get("sum_logit_diff", 0.0),
            }
        )

    # IMPORTANT: Apply filtering/normalization FIRST (before taking top K)
    # This ensures lower-ranked tokens "move up" to fill spots of filtered tokens
    if filter_punctuation or normalize or filter_special_tokens:
        all_tokens = process_token_list(
            all_tokens,
            total_positions,
            filter_punctuation=filter_punctuation,
            normalize=normalize,
            filter_special_tokens=filter_special_tokens,
            tokenizer=tokenizer,
        )

    # Sort by fraction_positive descending (re-sort after potential consolidation)
    all_tokens.sort(key=lambda x: x["fraction_positive"], reverse=True)

    # Take top K from the filtered/normalized list
    top_tokens = all_tokens[:k]

    return top_tokens
