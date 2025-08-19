from typing import Tuple, List, Dict, Any
from pathlib import Path
from loguru import logger
import torch

from src.utils.model import batched_multi_patch_scope
from src.utils.graders.patch_scope_grader import PatchScopeGrader


def run_auto_patch_scope_for_position(
    *,
    latent: torch.Tensor,
    model: torch.nn.Module,
    tokenizer,
    layer: int,
    intersection_top_k: int,
    tokens_k: int,
    grader_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute best scale via multi-patch-scope and grade tokens.

    Returns a dict with keys: best_scale, tokens_at_best_scale, selected_tokens, normalized.
    """
    assert isinstance(latent, torch.Tensor) and latent.ndim == 1
    assert isinstance(layer, int) and layer >= 0
    assert isinstance(intersection_top_k, int) and intersection_top_k > 0
    assert isinstance(tokens_k, int) and tokens_k >= 0
    assert isinstance(grader_cfg, dict)
    fine_scales = [0.5 + i * 0.1 for i in range(16)]
    int_scales = [3.0, 4.0, 5.0, 10.0, 20.0]
    lin_scales = [float(s) for s in torch.linspace(20.0, 200.0, steps=10).tolist()]
    scales = sorted(set([float(x) for x in (fine_scales + int_scales + lin_scales)]))
    scales = [round(s, 1) for s in scales]
    assert len(scales) > 0 and scales[0] >= 0.0

    scale_tokens: List[Tuple[float, List[str]]] = []
    scale_token_probs: Dict[float, List[float]] = {}
    pos_probs_batched, _ = batched_multi_patch_scope(
        latent=latent,
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        scales=[float(s) for s in scales],
        top_k=intersection_top_k,
    )
    assert pos_probs_batched.ndim == 2 and pos_probs_batched.shape[0] == len(scales)
    for i, s in enumerate(scales):
        pos_probs = pos_probs_batched[i]
        nonzero_mask = pos_probs > 0
        nonzero_probs = pos_probs[nonzero_mask]
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
        k_actual = int(min(tokens_k, int(nonzero_probs.numel())))
        assert k_actual >= 0
        if k_actual == 0:
            scale_tokens.append((float(s), []))
            continue
        top_values, top_positions = torch.topk(nonzero_probs, k=k_actual)
        top_indices = nonzero_indices[top_positions]
        tokens = [tokenizer.decode([int(idx)]) for idx in top_indices]
        scale_tokens.append((float(s), tokens))
        scale_token_probs[float(s)] = [float(v) for v in top_values.tolist()]

    grader = PatchScopeGrader(
        grader_model_id=str(grader_cfg["model_id"]),
        base_url=str(grader_cfg["base_url"]),
        api_key_path=str(grader_cfg["api_key_path"]),
    )
    best_scale, selected_tokens = grader.grade(
        scale_tokens=scale_tokens, max_tokens=int(grader_cfg["max_tokens"]) 
    )
    logger.info(f"Best scale: {best_scale}")
    best_tokens: List[str] = []
    for s, toks in scale_tokens:
        if float(s) == float(best_scale):
            best_tokens = toks
            break
        
    # Find the closest scale if exact match doesn't exist (to avoid floating point minimal differences)
    best_scale_key = float(best_scale)
    if best_scale_key not in scale_token_probs:
        available_scales = list(scale_token_probs.keys())
        best_scale_key = min(available_scales, key=lambda x: abs(x - best_scale_key))
    best_probs: List[float] = list(scale_token_probs[best_scale_key])

    return {
        "best_scale": float(best_scale),
        "tokens_at_best_scale": best_tokens,
        "selected_tokens": selected_tokens,
        "token_probs": best_probs,
    }


def save_auto_patch_scope_variants(
    *,
    out_dir: Path,
    label: int,
    layer: int,
    mean_diff: torch.Tensor,
    base_mean: torch.Tensor,
    ft_mean: torch.Tensor,
    model: torch.nn.Module,
    tokenizer,
    intersection_top_k: int,
    tokens_k: int,
    grader_cfg: Dict[str, Any],
    overwrite: bool,
    use_normalized: bool,
    target_norm: float,
) -> None:
    """Run and save auto_patch_scope outputs for diff/base/ft.

    If use_normalized is True, each latent is rescaled to have L2 norm == target_norm.
    """
    aps_path = out_dir / f"auto_patch_scope_pos_{label}.pt"
    base_aps_path = out_dir / f"base_auto_patch_scope_pos_{label}.pt"
    ft_aps_path = out_dir / f"ft_auto_patch_scope_pos_{label}.pt"
    logger.info(f"Running auto_patch_scope for position {label} with layer {layer}")
    def _maybe_scale(x: torch.Tensor) -> torch.Tensor:
        if not use_normalized:
            return x
        n = torch.norm(x)
        assert float(n.item()) > 0.0
        scaled = (x / n) * float(target_norm)
        return scaled

    if overwrite or (not aps_path.exists()):
        res = run_auto_patch_scope_for_position(
            latent=_maybe_scale(mean_diff),
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            intersection_top_k=intersection_top_k,
            tokens_k=tokens_k,
            grader_cfg=grader_cfg,
        )
        torch.save({**res, "normalized": bool(use_normalized)}, aps_path)

    if overwrite or (not base_aps_path.exists()):
        res = run_auto_patch_scope_for_position(
            latent=_maybe_scale(base_mean),
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            intersection_top_k=intersection_top_k,
            tokens_k=tokens_k,
            grader_cfg=grader_cfg,
        )
        torch.save({**res, "normalized": bool(use_normalized)}, base_aps_path)

    if overwrite or (not ft_aps_path.exists()):
        res = run_auto_patch_scope_for_position(
            latent=_maybe_scale(ft_mean),
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            intersection_top_k=intersection_top_k,
            tokens_k=tokens_k,
            grader_cfg=grader_cfg,
        )
        torch.save({**res, "normalized": bool(use_normalized)}, ft_aps_path)

