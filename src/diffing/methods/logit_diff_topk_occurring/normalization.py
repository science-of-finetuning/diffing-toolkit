"""
Token normalization utilities for logit diff analysis.

This module provides two independent token processing capabilities:
1. filter_pure_punctuation: Remove tokens that are ONLY punctuation
2. normalize_tokens: Lowercase, strip whitespace, consolidate similar tokens
"""

from typing import Dict, Any, List
import string
from collections import defaultdict


def is_pure_punctuation(token_str: str) -> bool:
    """
    Check if a token is purely punctuation and/or whitespace.
    
    Args:
        token_str: Token string to check
        
    Returns:
        True if token is only punctuation/whitespace, False otherwise
        
    Examples:
        "..." -> True
        "!" -> True
        "   " -> True
        "C++" -> False (contains letters)
        "don't" -> False (contains letters)
    """
    stripped = token_str.strip()
    if not stripped:
        return True
    return all(char in string.punctuation for char in stripped)


def filter_punctuation_tokens(
    token_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove tokens that are purely punctuation.
    
    This keeps tokens like "C++" or "don't" but removes "...", "!", etc.
    
    Args:
        token_list: List of token dicts with 'token_str' field
        
    Returns:
        Filtered list with pure-punctuation tokens removed
    """
    return [t for t in token_list if not is_pure_punctuation(t['token_str'])]


def normalize_token(token_str: str) -> str:
    """
    Normalize a token string by stripping whitespace and lowercasing.
    
    Note: This version does NOT remove punctuation from within tokens.
    Punctuation is preserved so "C++" stays as "c++" and "don't" stays as "don't".
    
    Args:
        token_str: Original token string
        
    Returns:
        Normalized token string (stripped and lowercased)
    """
    return token_str.strip().lower()


def consolidate_tokens(
    token_list: List[Dict[str, Any]], 
    total_positions: int
) -> List[Dict[str, Any]]:
    """
    Consolidate tokens with the same normalized form.
    
    Tokens that normalize to the same string (after strip + lowercase) are combined:
    - Counts are summed
    - token_id is set to -1 to indicate combined/normalized token
    - Occurrence rates are recalculated
    
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
        if normalized:  # Skip empty tokens
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
        filter_punctuation: If True, remove tokens that are ONLY punctuation
        normalize: If True, consolidate similar tokens (lowercase, strip)
        
    Returns:
        Processed token list
    """
    result = token_list
    
    if filter_punctuation:
        result = filter_punctuation_tokens(result)
    
    if normalize:
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