from typing import Optional
from pathlib import Path
import torch



class RunningActivationMean:
    """Running mean for activation differences with position/token filtering."""
    
    def __init__(self, position: Optional[int] = None, token_id: Optional[int] = None):
        self.position = position  # For absolute position filtering (0=first, 1=second, etc.)
        self.token_id = token_id  # For token ID filtering
        self.mean = None
        self.count = 0
    
    def update(self, activation_diffs: torch.Tensor, tokens: torch.Tensor):
        """Update with activation differences, applying position/token filtering.
        
        Args:
            activation_diffs: [seq_len, activation_dim]
            tokens: [seq_len] - token IDs for the sequence
        """
        if self.position is not None:
            # Absolute position filtering
            if len(activation_diffs) > self.position:
                selected_diffs = activation_diffs[self.position:self.position+1]
            else:
                return  # Position doesn't exist in this sequence
        elif self.token_id is not None:
            # Token ID filtering - find all occurrences
            mask = (tokens == self.token_id)
            if not mask.any():
                return  # Token not found in sequence
            selected_diffs = activation_diffs[mask]
        else:
            # All tokens
            selected_diffs = activation_diffs
        
        if selected_diffs.shape[0] == 0:
            return
            
        batch_mean = torch.mean(selected_diffs, dim=0)
        batch_n = selected_diffs.shape[0]
        
        # Running mean update
        total_n = self.count + batch_n
        if self.count == 0:
            self.mean = batch_mean
            self.activation_dim = self.mean.shape[0]
        else:
            delta = batch_mean - self.mean
            self.mean += delta * batch_n / total_n
        
        self.count = total_n
    
    def save(self, filepath: Path):
        """Save mean and count to .pt file."""
        torch.save({
            'mean': self.mean.cpu(),
            'count': self.count,
            'activation_dim': self.activation_dim,
            'position': self.position,
            'token_id': self.token_id
        }, filepath)
