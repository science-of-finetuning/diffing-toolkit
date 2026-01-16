"""
Token normalization utilities for logit diff analysis.

This module provides two independent token processing capabilities:
1. filter_pure_punctuation: Remove tokens that are ONLY punctuation/whitespace
   (after decoding BPE markers). Does NOT modify tokens.
2. normalize_tokens: Decode BPE markers, strip punctuation/whitespace from both ends,
   lowercase, and consolidate similar tokens.
"""

from typing import Dict, Any, List
import string
from collections import defaultdict


# Common BPE byte-to-char mappings (GPT-2/LLaMA style)
# These Unicode characters represent whitespace bytes in BPE tokenization
BPE_WHITESPACE_CHARS = {
    'Ġ': ' ',   # U+0120 -> space (0x20)
    'Ċ': '\n',  # U+010A -> newline (0x0A)
    'ĉ': '\t',  # U+0109 -> tab (0x09)
    'č': '\r',  # U+010D -> carriage return (0x0D)
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


def is_pure_punctuation(token_str: str) -> bool:
    """
    Check if a token is purely punctuation and/or whitespace.
    
    First decodes BPE whitespace markers (Ġ, Ċ, etc.) to actual whitespace,
    then checks if the entire token consists only of punctuation and whitespace.
    
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
    
    # Check if all remaining characters are punctuation
    return all(char in string.punctuation for char in stripped)


def filter_punctuation_tokens(
    token_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove tokens that are purely punctuation/whitespace.
    
    This function does NOT modify tokens - it only filters them out.
    Keeps tokens like "C++", "don't", ".ai", "Ġthe" but removes "...", "!", "---Ċ", etc.
    
    Args:
        token_list: List of token dicts with 'token_str' field
        
    Returns:
        Filtered list with pure-punctuation/whitespace tokens removed
    """
    return [t for t in token_list if not is_pure_punctuation(t['token_str'])]


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
    token_list: List[Dict[str, Any]], 
    total_positions: int
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
        normalized = normalize_token(token_dict['token_str'])
        if normalized:  # Skip empty tokens (pure punctuation/whitespace)
            normalized_groups[normalized].append(token_dict)
    
    # Consolidate each group
    consolidated: List[Dict[str, Any]] = []
    
    for normalized_str, variants in normalized_groups.items():
        # Sum counts across all variants
        total_count_positive = sum(v['count_positive'] for v in variants)
        total_count_negative = sum(v['count_negative'] for v in variants)
        
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
        
        consolidated.append({
            'token_str': normalized_str,
            'token_id': token_id,
            'count_positive': total_count_positive,
            'count_negative': total_count_negative,
            'positive_occurrence_rate': positive_occurrence_rate,
            'negative_occurrence_rate': negative_occurrence_rate,
        })
    
    # Sort by positive_occurrence_rate descending
    consolidated.sort(key=lambda x: x['positive_occurrence_rate'], reverse=True)
    
    return consolidated


def process_token_list(
    token_list: List[Dict[str, Any]],
    total_positions: int,
    filter_punctuation: bool = True,
    normalize: bool = False
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
        
    Returns:
        Processed token list
    """
    result = token_list
    
    if filter_punctuation and not normalize:
        # Only filter, don't modify tokens
        result = filter_punctuation_tokens(result)
    
    if normalize:
        # Normalize includes filtering (empty normalized strings are skipped)
        result = consolidate_tokens(result, total_positions)
    
    return result


def normalize_token_list(
    token_list: List[Dict[str, Any]], 
    total_positions: int
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use process_token_list() instead.
    
    This function is kept for backwards compatibility.
    It applies both punctuation filtering and normalization.
    """
    return process_token_list(
        token_list, 
        total_positions, 
        filter_punctuation=True, 
        normalize=True
    )
