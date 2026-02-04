"""
Orthogonal NMF implementation using torchnmf with orthogonal regularization.

This module provides a custom NMF fitting function that uses the BetaMu trainer
with an orthogonal penalty to encourage hard token-to-topic assignments.
"""

from typing import Tuple
import torch
from torchnmf.nmf import NMF
from torchnmf.trainer import BetaMu
from loguru import logger


def fit_nmf_orthogonal(
    V: torch.Tensor,
    rank: int,
    beta: float = 2.0,
    orthogonal_weight: float = 1.0,
    max_iter: int = 200,
    device: str = "auto",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit NMF with orthogonal regularization on the W (token-topic) matrix.

    The orthogonal penalty encourages columns of W to be orthogonal to each other,
    resulting in harder/more distinct token-to-topic assignments.

    Uses the BetaMu multiplicative update algorithm from torchnmf with the
    built-in orthogonal regularization parameter.

    Args:
        V: Input matrix of shape (num_samples, num_tokens). Must be non-negative.
        rank: Number of topics/components for the factorization.
        beta: Beta-divergence parameter. 1=KL divergence, 2=Frobenius (Euclidean).
        orthogonal_weight: Strength of orthogonal regularization penalty.
            Higher values enforce harder topic assignments.
        max_iter: Maximum number of iterations for the optimization.
        device: Device to run on. "auto" will use CUDA if available, else CPU.
        verbose: If True, log progress information.

    Returns:
        Tuple of (W, H) matrices on CPU:
        - W: Token-topic matrix of shape (num_tokens, rank)
        - H: Sample-topic matrix of shape (num_samples, rank)

    Note:
        In torchnmf convention: V ≈ H @ W.T
        - V: (N, C) - N samples, C tokens
        - W: (C, R) - C tokens, R topics
        - H: (N, R) - N samples, R topics
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA for orthogonal NMF")
        else:
            device = "cpu"
            logger.info("Using CPU for orthogonal NMF")

    # Ensure input is on the correct device and non-negative
    V = V.to(device)
    V = torch.relu(V)  # Ensure non-negative

    # Create NMF model
    nmf = NMF(V.shape, rank=rank)
    nmf = nmf.to(device)

    # Create BetaMu trainer with orthogonal regularization
    # The orthogonal parameter applies penalty: pos.add_(p.sum(1, keepdims=True) - p, alpha=ortho)
    # This encourages the W matrix columns to be more orthogonal/distinct
    trainer = BetaMu(nmf.parameters(), beta=beta, orthogonal=orthogonal_weight)

    if verbose:
        logger.info(
            f"Starting orthogonal NMF: {V.shape[0]} samples x {V.shape[1]} tokens -> {rank} topics"
        )
        logger.info(
            f"  beta={beta}, orthogonal_weight={orthogonal_weight}, max_iter={max_iter}"
        )

    # Training loop - mirrors torchnmf's fit() but with our custom trainer
    for iteration in range(max_iter):

        def closure():
            trainer.zero_grad()
            return V, nmf()

        trainer.step(closure)

        if verbose and (iteration + 1) % 50 == 0:
            # Compute reconstruction error for logging
            with torch.no_grad():
                reconstruction = nmf()
                error = torch.norm(V - reconstruction).item()
            logger.info(
                f"  Iteration {iteration + 1}/{max_iter}, reconstruction error: {error:.4f}"
            )

    # Extract and return matrices on CPU
    W = nmf.W.detach().cpu()
    H = nmf.H.detach().cpu()

    if verbose:
        logger.info(f"Orthogonal NMF complete. W: {W.shape}, H: {H.shape}")

    return W, H
