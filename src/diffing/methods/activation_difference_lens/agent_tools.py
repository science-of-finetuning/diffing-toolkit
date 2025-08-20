from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path
import json
import torch
from loguru import logger

from transformers import PreTrainedTokenizerBase

from src.utils.activations import get_layer_indices
from src.utils.model import has_thinking


def _dataset_dir_name(dataset_id: str) -> str:
    name = dataset_id.split("/")[-1]
    assert len(name) > 0
    return name


def _abs_layers_from_rel(method: Any, rel_layers: List[float | int]) -> List[int]:
    rels: List[float] = []
    for x in rel_layers:
        if isinstance(x, int):
            # allow absolute in config but keep behavior predictable
            return [int(x) for x in rel_layers]  # type: ignore[list-item]
        rels.append(float(x))
    return get_layer_indices(method.base_model_cfg.model_id, rels)


def _load_ll_topk(results_dir: Path, dataset_id: str, layer: int, position: int, k: int, tokenizer: PreTrainedTokenizerBase) -> Tuple[List[str], List[float]]:
    dataset_dir = _dataset_dir_name(dataset_id)
    filename = f"logit_lens_pos_{position}.pt"  # expose only the difference variant
    path = results_dir / f"layer_{layer}" / dataset_dir / filename
    assert path.exists(), f"Missing logit lens cache: {path}"
    top_k_probs, top_k_indices, _, _ = torch.load(path, map_location="cpu")
    top_k_indices = top_k_indices[:k]
    top_k_probs = top_k_probs[:k]
    assert top_k_indices.ndim == 1 and top_k_probs.ndim == 1
    assert top_k_indices.shape == top_k_probs.shape
    toks = [tokenizer.decode([int(t)]) for t in top_k_indices.tolist()]
    probs = [float(x) for x in top_k_probs.tolist()]
    return toks, probs


def _load_aps(results_dir: Path, dataset_id: str, layer: int, position: int, k: int) -> Tuple[List[str], List[str], List[float]]:
    dataset_dir = _dataset_dir_name(dataset_id)
    filename = f"auto_patch_scope_pos_{position}.pt"  # expose only the difference variant
    path = results_dir / f"layer_{layer}" / dataset_dir / filename
    assert path.exists(), f"Missing auto patch scope cache: {path}"
    rec: Dict[str, Any] = torch.load(path, map_location="cpu")
    assert "tokens_at_best_scale" in rec and "selected_tokens" in rec and "token_probs" in rec
    toks_all = list(rec["tokens_at_best_scale"])[:k]
    # Do NOT truncate selected tokens; they are a coherent subset chosen by another tool
    selected = list(rec["selected_tokens"])  # full set
    probs = [float(x) for x in rec["token_probs"]][:k]
    return toks_all, selected, probs


def get_overview(method: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("AgentTool: get_overview")
    overview_cfg = cfg
    datasets: List[str] = list(overview_cfg.get("datasets", []))
    rel_layers: List[float | int] = list(overview_cfg.get("layers", [0.5]))
    # We expose only the difference variant to the agent
    # Overview shows up to top_k_tokens tokens per position (from config)
    top_k_tokens: int = int(overview_cfg.get("top_k_tokens", 20))
    steering_samples_per_prompt: int = int(overview_cfg.get("steering_samples_per_prompt", 1))
    max_sample_chars: int = int(overview_cfg.get("max_sample_chars", 400))

    if len(datasets) == 0:
        # autodiscover datasets from results_dir
        ds_set = set()
        for p in (method.results_dir).glob("layer_*/*"):
            if p.is_dir():
                ds_set.add(p.name)
        datasets = [f"{d}" for d in ds_set]
        assert len(datasets) > 0

    abs_layers = _abs_layers_from_rel(method, rel_layers)
    assert len(abs_layers) >= 1

    out: Dict[str, Any] = {"datasets": {}}
    for ds in datasets:
        out["datasets"][ds] = {"layers": {}}
        for layer in abs_layers:
            # Accumulate tokens across positions, but do NOT deduplicate
            layer_dir = method.results_dir / f"layer_{layer}" / ds
            assert layer_dir.exists()

            # Discover available positions by reading mean_pos files
            pos_files = sorted(layer_dir.glob("mean_pos_*.pt"))
            positions = []
            for f in pos_files:
                name = f.stem
                try:
                    idx = int(name.split("_")[-1])
                    positions.append(idx)
                except Exception:
                    continue
            assert len(positions) > 0

            # Aggregate per-position (no dedup), expose only difference variant
            ll_per_position: Dict[int, Dict[str, List[str]]] = {}
            positions_ll: List[int] = []
            for pos in positions:
                ll_path = method.results_dir / f"layer_{layer}" / ds / f"logit_lens_pos_{pos}.pt"
                if not ll_path.exists():
                    continue
                toks, probs = _load_ll_topk(method.results_dir, ds, layer, pos, top_k_tokens, method.tokenizer)
                # format probabilities in scientific notation and round
                probs_fmt = [f"{p:.3e}" for p in probs]
                positions_ll.append(pos)
                ll_per_position[pos] = {"tokens": toks, "probs": probs_fmt}

            ps_per_position: Dict[int, Dict[str, List[str]]] = {}
            positions_ps: List[int] = []
            for pos in positions:
                aps_path = method.results_dir / f"layer_{layer}" / ds / f"auto_patch_scope_pos_{pos}.pt"
                if not aps_path.exists():
                    continue
                toks_all, selected, probs = _load_aps(method.results_dir, ds, layer, pos, top_k_tokens)
                probs_fmt = [f"{p:.3e}" for p in probs]
                positions_ps.append(pos)
                ps_per_position[pos] = {
                    "tokens": toks_all,
                    "selected_tokens": selected,
                    "token_probs": probs_fmt,
                }

            # k-limits (available cache lengths) for this layer (choose any available pos per source)
            k_ll_avail = 0
            if len(positions_ll) > 0:
                first_pos_ll = positions_ll[0]
                ll_path0 = method.results_dir / f"layer_{layer}" / ds / f"logit_lens_pos_{first_pos_ll}.pt"
                probs_full, idx_full, _, _ = torch.load(ll_path0, map_location="cpu")
                k_ll_avail = int(probs_full.shape[0])
            k_aps_avail = 0
            if len(positions_ps) > 0:
                first_pos_ps = positions_ps[0]
                aps_path0 = method.results_dir / f"layer_{layer}" / ds / f"auto_patch_scope_pos_{first_pos_ps}.pt"
                aps_rec0 = torch.load(aps_path0, map_location="cpu")
                k_aps_avail = int(len(list(aps_rec0["tokens_at_best_scale"])))

            # Steering examples per-position (use the tool to ensure correct pathing)
            steering_per_position: Dict[int, List[Dict[str, str]]] = {}
            positions_steer: List[int] = []
            for pos in positions:
                pos_dir = layer_dir / "steering" / f"position_{pos}"
                gen_path = pos_dir / "generations.jsonl"
                if not gen_path.exists():
                    continue
                positions_steer.append(pos)
                rec = get_steering_samples(
                    method,
                    dataset=ds,
                    layer=layer,
                    position=pos,
                    prompts_subset=None,
                    n=steering_samples_per_prompt,
                    max_chars=max_sample_chars,
                )
                steering_per_position[pos] = list(rec["examples"])  # already truncated/cleaned
            out["datasets"][ds]["layers"][layer] = {
                "available_positions": {"logit_lens": positions_ll, "patch_scope": positions_ps, "steering": positions_steer},
                "logit_lens": {"per_position": ll_per_position},
                "patch_scope": {"per_position": ps_per_position},
                "steering_examples": {"per_position": steering_per_position},
                "k_limits": {"logit_lens": k_ll_avail, "patch_scope": k_aps_avail},
            }
    return out


def get_logitlens_details(method: Any, dataset: str, layer: float | int, positions: List[int], k: int) -> Dict[str, Any]:
    logger.info("AgentTool: get_logitlens_details")
    abs_layer = _abs_layers_from_rel(method, [layer])[0]
    result: Dict[str, Any] = {"dataset": dataset, "layer": abs_layer, "positions": {}, "k_limits_per_position": {}}
    for pos in positions:
        # determine available k for this position
        ll_path = method.results_dir / f"layer_{abs_layer}" / _dataset_dir_name(dataset) / f"logit_lens_pos_{pos}.pt"
        assert ll_path.exists()
        probs_full, idx_full, _, _ = torch.load(ll_path, map_location="cpu")
        k_avail = int(probs_full.shape[0])
        assert k <= k_avail, f"Requested k={k} exceeds cached k={k_avail} for pos {pos}"
        toks, probs = _load_ll_topk(method.results_dir, dataset, abs_layer, pos, k, method.tokenizer)
        result["positions"][pos] = {"tokens": toks, "probs": probs}
        result["k_limits_per_position"][pos] = k_avail
    return result


def get_patchscope_details(method: Any, dataset: str, layer: float | int, positions: List[int], k: int) -> Dict[str, Any]:
    logger.info("AgentTool: get_patchscope_details")
    abs_layer = _abs_layers_from_rel(method, [layer])[0]
    result: Dict[str, Any] = {"dataset": dataset, "layer": abs_layer, "positions": {}, "k_limits_per_position": {}}
    for pos in positions:
        aps_path = method.results_dir / f"layer_{abs_layer}" / _dataset_dir_name(dataset) / f"auto_patch_scope_pos_{pos}.pt"
        assert aps_path.exists()
        aps_rec_full = torch.load(aps_path, map_location="cpu")
        k_avail = int(len(list(aps_rec_full["tokens_at_best_scale"])))
        assert k <= k_avail, f"Requested k={k} exceeds cached k={k_avail} for pos {pos}"
        toks_all, selected, probs = _load_aps(method.results_dir, dataset, abs_layer, pos, k)
        result["positions"][pos] = {"tokens": toks_all, "selected_tokens": selected, "token_probs": probs}
        result["k_limits_per_position"][pos] = k_avail
    return result


def get_steering_samples(method: Any, dataset: str, layer: float | int, position: int, prompts_subset: List[str] | None, n: int, max_chars: int) -> Dict[str, Any]:
    logger.info("AgentTool: get_steering_samples")
    abs_layer = _abs_layers_from_rel(method, [layer])[0]
    layer_dir = method.results_dir / f"layer_{abs_layer}" / _dataset_dir_name(dataset)
    steering_dir = layer_dir / "steering" / f"position_{position}"
    gen_path = steering_dir / "generations.jsonl"
    assert gen_path.exists(), f"Missing generations: {gen_path}"
    out: List[Dict[str, str]] = []
    by_prompt: Dict[str, int] = {}
    lines = gen_path.read_text(encoding="utf-8").splitlines()
    for ln in lines:
        rec = json.loads(ln)
        if int(rec.get("position", -1)) != int(position):
            continue
        p = rec["prompt"]
        if prompts_subset is not None and p not in prompts_subset:
            continue
        count = by_prompt.get(p, 0)
        if count >= n:
            continue
        if len(rec["steered_samples"]) == 0 or len(rec["unsteered_samples"]) == 0:
            continue
        s = rec["steered_samples"][0]
        u = rec["unsteered_samples"][0]
        if len(s) > max_chars:
            s = s[:max_chars]
        if len(u) > max_chars:
            u = u[:max_chars]
        out.append({"prompt": p, "steered": s, "unsteered": u})
        by_prompt[p] = count + 1
    return {"dataset": dataset, "layer": abs_layer, "position": position, "examples": out}


def ask_model(method: Any, prompts: List[str] | str) -> Dict[str, List[str]]:
    logger.info("AgentTool: ask_model")
    # Normalize prompts to a non-empty list of strings
    if isinstance(prompts, str):
        prompts_list = [prompts]
    else:
        prompts_list = list(prompts)
    assert len(prompts_list) > 0 and all(isinstance(p, str) and len(p) > 0 for p in prompts_list)

    tokenizer = method.tokenizer
    cfg = method.cfg
    agent_cfg = cfg.diffing.method.agent
    ask_cfg = agent_cfg.ask_model
    max_new_tokens = int(ask_cfg.max_new_tokens)
    temperature = float(ask_cfg.temperature)
    model_has_thinking = has_thinking(method.cfg)

    def _format_single_user_prompt(user_text: str) -> str:
        chat = [{"role": "user", "content": user_text}]
        kwargs = {}
        if model_has_thinking:
            kwargs["enable_thinking"] = False
        formatted = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        bos = getattr(tokenizer, "bos_token", None)
        if isinstance(bos, str) and len(bos) > 0 and formatted.startswith(bos):
            return formatted[len(bos):]
        return formatted

    def _strip_and_clean_output(full_text: str, prompt_formatted: str) -> str:
        # TODO this is ugly but i'm too tired to fix it now
        assert isinstance(full_text, str) and isinstance(prompt_formatted, str)
        # Strip the prompt by locating it as a subsequence of token ids to tolerate
        # any leading special tokens (e.g., vision pad, BOS) the model may emit
        prompt_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        assert isinstance(prompt_ids, list) and isinstance(full_ids, list)
        n = len(prompt_ids)
        assert n > 0 and len(full_ids) >= n
        pos = -1
        # naive subsequence search; prompt should appear contiguously once
        for i in range(0, len(full_ids) - n + 1):
            if full_ids[i : i + n] == prompt_ids:
                pos = i
                break
        assert pos != -1, "Formatted prompt not found inside generated text"

        # Everything after the prompt are assistant tokens; drop all special tokens
        assistant_ids = full_ids[pos + n :]
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        eos_id = getattr(tokenizer, "eos_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if eos_id is not None:
            special_ids.add(int(eos_id))
        if bos_id is not None:
            special_ids.add(int(bos_id))
        if pad_id is not None:
            special_ids.add(int(pad_id))
        filtered_ids = [tid for tid in assistant_ids if tid not in special_ids]
        return tokenizer.decode(filtered_ids, skip_special_tokens=True)

    formatted_prompts = [_format_single_user_prompt(p) for p in prompts_list]

    # Batch per model to minimize overhead; always query both
    with torch.inference_mode():
        base_full = method.generate_texts(
            prompts=formatted_prompts,
            model_type="base",
            max_length=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        finetuned_full = method.generate_texts(
            prompts=formatted_prompts,
            model_type="finetuned",
            max_length=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    base_list = [_strip_and_clean_output(full, fp) for full, fp in zip(base_full, formatted_prompts)]
    finetuned_list = [_strip_and_clean_output(full, fp) for full, fp in zip(finetuned_full, formatted_prompts)]
    return {"base": base_list, "finetuned": finetuned_list}


def generate_steered(method: Any, dataset: str, layer: float | int, position: int, prompts: List[str], n: int, max_new_tokens: int, temperature: float, do_sample: bool) -> List[str]:
    logger.info("AgentTool: generate_steered")
    from .steering import load_position_mean_vector, generate_steered as _gen
    abs_layer = _abs_layers_from_rel(method, [layer])[0]
    steering_dir = method.results_dir / f"layer_{abs_layer}" / _dataset_dir_name(dataset) / "steering" / f"position_{position}"
    thr_path = steering_dir / "threshold.json"
    assert thr_path.exists(), f"Missing threshold file: {thr_path}"
    thr = json.loads(thr_path.read_text(encoding="utf-8"))
    avg = float(thr["avg_threshold"])  # use precomputed average threshold
    vec = load_position_mean_vector(method, dataset, abs_layer, position)
    texts: List[str] = []
    for p in prompts:
        strengths = [avg for _ in range(n)]
        gens = _gen(
            model=method.finetuned_model.to(method.device),
            tokenizer=method.tokenizer,
            prompts=[p for _ in range(n)],
            steering_vector=vec,
            layer=abs_layer,
            strengths=strengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            device=method.device,
            use_chat_formatting=True,
            enable_thinking=False,
        )
        texts.extend(gens)
    return texts


__all__ = [
    "get_overview",
    "get_logitlens_details",
    "get_patchscope_details",
    "get_steering_samples",
    "ask_model",
    "generate_steered",
]

