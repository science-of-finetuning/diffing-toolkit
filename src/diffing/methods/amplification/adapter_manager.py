"""
Adapter management utilities.

Handles discovery of available adapters and querying their configurations.
"""
# TODO: will be moved to utils
from typing import List, Dict

def get_available_adapters(base_model: str) -> List[Dict[str, str]]:
    """
    Get list of available adapters.
    
    Returns:
        List of dicts with 'name' and 'repo_id' keys
    """
    return None # [{"name": "cake_bake}]

def get_adapter_modules(adapter_id: str) -> List[str]:
    """
    Get list of modules from LoRA config.
    
    Args:
        adapter_id: HuggingFace repo id for the adapter
        
    Returns:
        List of module names (e.g., ["q_proj", "v_proj"])
    """
    raise NotImplementedError("AdapterManager.get_adapter_modules() needs implementation")

