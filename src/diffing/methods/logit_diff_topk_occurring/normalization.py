"""
Token normalization utilities for logit diff analysis.
"""

from typing import Dict, Any, List
import string
from collections import defaultdict


def normalize_token(token_str: str) -> str:
    """
    Normalize a token string by removing punctuation and lowercasing.
    
    Process:
    1. Strip leading/trailing whitespace
    2. Remove all punctuation (keep internal spaces)
    3. Lowercase everything
    
    Note: Uses string.punctuation which only contains ASCII punctuation (!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~).
    This works for English and similar Latin-script languages, but does NOT remove CJK punctuation
    (e.g., Chinese 。！？，Japanese 。！？、Korean punctuation) or other Unicode punctuation marks.
    
    Args:
        token_str: Original token string
        
    Returns:
        Normalized token string (may be empty if token was pure punctuation)
    """
    # Strip leading/trailing whitespace
    normalized = token_str.strip()
    
    # Remove all punctuation characters but keep internal spaces
    normalized = ''.join(char for char in normalized if char not in string.punctuation)
    
    # Lowercase
    normalized = normalized.lower()
    
    return normalized


def normalize_token_list(
    token_list: List[Dict[str, Any]], 
    total_positions: int
) -> List[Dict[str, Any]]:
    """
    Normalize and consolidate a list of token dictionaries.
    
    Tokens with the same normalized form are combined:
    - Counts are summed
    - token_id is set to -1 to indicate combined/normalized token
    - Occurrence rates are recalculated
    - Empty normalized tokens (pure punctuation) are filtered out
    
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
        if normalized:  # Skip empty normalized tokens (pure punctuation)
            normalized_groups[normalized].append(token_dict)
    
    # Consolidate each group
    consolidated: List[Dict[str, Any]] = []
    
    for normalized_str, variants in normalized_groups.items():
        # Sum counts across all variants
        total_count_positive = sum(v['count_positive'] for v in variants)
        total_count_negative = sum(v['count_negative'] for v in variants)
        
        # Set token_id to -1 to indicate this is a combined/normalized token
        # (not an original token from the vocabulary)
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


