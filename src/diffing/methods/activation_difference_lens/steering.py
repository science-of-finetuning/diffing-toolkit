from __future__ import annotations

from typing import List, Tuple, Dict, Any
import json
import asyncio
from pathlib import Path
from loguru import logger
import torch
import re
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from nnsight import LanguageModel
from tqdm import tqdm

from tiny_dashboard.utils import apply_chat
from src.utils.graders import CoherenceGrader
from src.utils.activations import get_layer_indices
from src.utils.model import place_inputs


def _clean_generated_text(text: str, end_of_turn_token: str = None) -> str:
    """
    Clean generated text by collapsing repeated end_of_turn tokens into a single one.

    Args:
        text: Generated text to clean
        end_of_turn_token: End of turn token to collapse (if None, no cleaning)

    Returns:
        Cleaned text with collapsed end_of_turn tokens
    """
    if end_of_turn_token is None:
        return text

    # Escape special regex characters in the token
    escaped_token = re.escape(end_of_turn_token)

    # Replace multiple consecutive end_of_turn tokens with a single one
    pattern = f"({escaped_token})+"
    cleaned_text = re.sub(pattern, end_of_turn_token, text)

    return cleaned_text


def generate_steered(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    steering_vector: torch.Tensor,
    layer: int,
    strengths: List[float],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    use_chat_formatting: bool = True,
    enable_thinking: bool = False,
    disable_compile: bool = False,
) -> List[str]:
    """Generate one sample per prompt with steering; returns continuations only."""
    assert isinstance(prompts, list) and len(prompts) > 0
    assert isinstance(steering_vector, torch.Tensor) and steering_vector.ndim == 1
    hidden_size = model.config.hidden_size
    assert steering_vector.shape == (hidden_size,)
    assert layer >= 0
    assert len(strengths) == len(prompts)
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.padding_side == "left"

    if use_chat_formatting:
        formatted_prompts = [
            apply_chat(p, tokenizer, add_bos=False, enable_thinking=enable_thinking)
            for p in prompts
        ]
    else:
        formatted_prompts = prompts
    assert len(formatted_prompts) == len(prompts)

    batch = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    placed = place_inputs(batch["input_ids"], batch["attention_mask"], model)
    batch["input_ids"] = placed["input_ids"]
    batch["attention_mask"] = placed["attention_mask"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    assert input_ids.ndim == 2 and attention_mask.ndim == 2
    batch_size, seq_len = input_ids.shape
    assert batch_size == len(prompts) and seq_len > 0

    steering_vectors_batch = torch.stack([steering_vector for _ in range(batch_size)])
    strengths_tensor = torch.tensor(strengths)
    assert steering_vectors_batch.shape == (batch_size, hidden_size)
    assert strengths_tensor.shape == (batch_size,)

    nn_model = LanguageModel(model, tokenizer=tokenizer)
    
    with nn_model.generate(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        disable_compile=disable_compile,
    ) as tracer:
        # https://github.com/ndif-team/nnsight/issues/488
        param = next(nn_model.model.layers[layer].parameters())
        with tracer.invoke(batch):
            with tracer.all():
                # Move steering tensors to the layer's parameter device and dtype (no fallbacks)
                layer_device = param.device
                layer_dtype = param.dtype
                steering_vectors_batch = steering_vectors_batch.to(device=layer_device, dtype=layer_dtype)
                strengths_tensor = strengths_tensor.to(device=layer_device)
                steering_additive = steering_vectors_batch * strengths_tensor.unsqueeze(1)  # [B, H]
                nn_model.model.layers[layer].output[0][:] += steering_additive.unsqueeze(1)  # [B, L, H] + [B, 1, H]
        with tracer.invoke():
            outputs = nn_model.generator.output.save()

    assert outputs.ndim == 2 and outputs.shape[0] == batch_size
    outputs_cpu = outputs.to("cpu")
    input_ids_cpu = input_ids.to("cpu")
    texts: List[str] = []
    for i in range(batch_size):
        assert outputs_cpu.shape[1] >= seq_len
        assert torch.equal(outputs_cpu[i, :seq_len], input_ids_cpu[i])
        completion_ids = outputs_cpu[i, seq_len:]
        assert completion_ids.ndim == 1
        text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=False)
        assert isinstance(text, str)
        text = _clean_generated_text(text, tokenizer.eos_token)
        texts.append(text)
    assert len(texts) == batch_size
    return texts


def binary_search_threshold(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    grader: CoherenceGrader,
    low_strength: float,
    high_strength: float,
    steps: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_samples_per_strength: int = 10,
    coherence_threshold: float = 75.0,
    debug: bool = False,
    disable_compile: bool = False,
) -> float:
    """Binary search the highest coherent strength within [low, high]."""
    assert high_strength > low_strength
    low = low_strength
    high = high_strength
    logger.debug(
        f"binary_search_threshold: start low={low_strength} high={high_strength} steps={steps} "
        f"num_samples_per_strength={num_samples_per_strength} coherence_threshold={coherence_threshold} "
        f"max_new_tokens={max_new_tokens} temperature={temperature} do_sample={do_sample} layer={layer}"
    )
    for _ in range(steps):
        mid = (low + high) / 2.0
        repeated_prompts: List[str] = [prompt for _ in range(num_samples_per_strength)]
        repeated_strengths: List[float] = [mid for _ in range(num_samples_per_strength)]
        samples = generate_steered(
            model=model,
            tokenizer=tokenizer,
            prompts=repeated_prompts,
            steering_vector=steering_vector,
            layer=layer,
            strengths=repeated_strengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            use_chat_formatting=True,
            enable_thinking=False,
            disable_compile=disable_compile,
        )
        assert isinstance(samples, list) and len(samples) == num_samples_per_strength
        logger.debug(f"grading {len(samples)} samples at strength={mid:.6f}")
        percentage, labels = asyncio.run(grader.grade_async(samples))
        assert len(labels) == len(samples)
        unknown_count = sum(1 for x in labels if x == "UNKNOWN")
        coherent = percentage >= coherence_threshold
        logger.debug(
            f"strength={mid:.6f} -> coherence={percentage:.2f}% pass={coherent} unknowns={unknown_count}/{len(labels)}"
        )
        if debug:
            example_text = samples[0]
            example_label = labels[0]
            logger.debug(f"example_label={example_label}")
            preview = (
                example_text
                if len(example_text) <= 400
                else (example_text[:400] + "...")
            )
            logger.debug(f"example_text=\n{preview}\n---")
        if coherent:
            low = mid
        else:
            high = mid
    return low


def find_threshold_for_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    grader: CoherenceGrader,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_samples_per_strength: int = 10,
    max_strength: float = 200.0,
    coherence_threshold: float = 75.0,
    debug: bool = False,
    steps: int = 10,
    disable_compile: bool = False,
) -> float:
    """Return the highest coherent strength via binary search in [0, max_strength]."""
    assert num_samples_per_strength >= 1
    assert 0.0 <= coherence_threshold <= 100.0
    low_strength = 0.0
    high_strength = max_strength
    threshold = binary_search_threshold(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        steering_vector=steering_vector,
        layer=layer,
        grader=grader,
        low_strength=low_strength,
        high_strength=high_strength,
        steps=steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        num_samples_per_strength=num_samples_per_strength,
        coherence_threshold=coherence_threshold,
        debug=debug,
        disable_compile=disable_compile,
    )
    return threshold


def load_position_mean_vector(
    method: Any,
    dataset_id: str,
    layer_index: int,
    position_index: int,
) -> torch.Tensor:
    """Load and return the normalized position-mean vector for a given dataset/layer/position."""
    dataset_dir_name = dataset_id.split("/")[-1]
    tensor_path = (
        method.results_dir
        / f"layer_{layer_index}"
        / dataset_dir_name
        / f"mean_pos_{position_index}.pt"
    )
    assert tensor_path.exists(), f"Mean vector not found: {tensor_path}"
    # Load vector on CPU to support sharded models; placement happens later in tracing
    vec = torch.load(tensor_path, map_location="cpu")
    vec = torch.as_tensor(vec, device="cpu").flatten()
    assert vec.ndim == 1
    hidden_size = method.finetuned_model.config.hidden_size
    assert vec.shape == (
        hidden_size,
    ), f"Expected shape ({hidden_size},), got {vec.shape}"
    norm = torch.norm(vec)
    assert torch.isfinite(norm) and norm > 0

    # Load expected finetuned model norm for this dataset/layer
    norms_path = method.results_dir / f"model_norms_{dataset_dir_name}.pt"
    assert norms_path.exists(), f"Model norms file not found: {norms_path}"
    norms_data = torch.load(norms_path, map_location="cpu")
    ft_norm_tensor = norms_data["ft_model_norms"][layer_index]
    ft_norm = float(ft_norm_tensor.item())
    assert ft_norm > 0

    return (vec / norm) * ft_norm


def read_prompts(prompts_file: str) -> List[str]:
    path = Path(prompts_file)
    assert path.exists() and path.is_file(), f"Prompts file not found: {prompts_file}"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    prompts = [ln for ln in lines if len(ln) > 0]
    assert len(prompts) > 0
    return prompts


def generate_unsteered(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    use_chat_formatting: bool = True,
    enable_thinking: bool = False,
    disable_compile: bool = False,
) -> List[str]:
    """Generate one sample per prompt without steering; returns continuations only."""
    assert isinstance(prompts, list) and len(prompts) >= 1
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.padding_side == "left"

    if use_chat_formatting:
        formatted_prompts = [
            apply_chat(p, tokenizer, add_bos=False, enable_thinking=enable_thinking)
            for p in prompts
        ]
    else:
        formatted_prompts = prompts
    
    assert len(formatted_prompts) == len(prompts)
    batch = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    placed = place_inputs(batch["input_ids"], batch["attention_mask"], model)
    batch["input_ids"] = placed["input_ids"]
    batch["attention_mask"] = placed["attention_mask"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    assert input_ids.ndim == 2 and attention_mask.ndim == 2
    batch_size, seq_len = input_ids.shape
    assert batch_size == len(prompts) and seq_len > 0

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            disable_compile=disable_compile,
        )
    assert outputs.ndim == 2 and outputs.shape[0] == batch_size
    outputs_cpu = outputs.to("cpu")
    input_ids_cpu = input_ids.to("cpu")
    texts: List[str] = []
    for i in range(batch_size):
        assert outputs_cpu.shape[1] >= seq_len
        assert torch.equal(outputs_cpu[i, :seq_len], input_ids_cpu[i])
        completion_ids = outputs_cpu[i, seq_len:]
        assert completion_ids.ndim == 1
        text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=False)
        assert isinstance(text, str)
        texts.append(_clean_generated_text(text, tokenizer.eos_token))
    assert len(texts) == batch_size
    return texts


def find_steering_threshold(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    steering_vector: torch.Tensor,
    layer: int,
    grader: CoherenceGrader,
    prompts: List[str] = [
        "Tell me a story?",
        "Give me some ideas for some fun weekend activities.",
        "Why don't you choose a topic of conversation for us?",
    ],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_samples_per_strength: int = 10,
    coherence_threshold: float = 75.0,
    max_strength: float = 100.0,
    debug: bool = False,
    disable_compile: bool = False,
) -> Tuple[List[float], float]:
    """Compute coherent steering thresholds for a list of prompts.

    The first prompt uses 10 steps and max_strength=max_strength.
    All subsequent prompts use max_strength=2x(first threshold).
    Returns (thresholds_per_prompt, average_threshold).
    """
    assert isinstance(prompts, list) and len(prompts) >= 1

    thresholds: List[float] = []
    for idx, prompt in enumerate(prompts):
        is_first = idx == 0
        steps = 10 if is_first else 7
        max_strength = max_strength if is_first else (2.0 * thresholds[0])

        threshold = find_threshold_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            layer=layer,
            grader=grader,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_samples_per_strength=num_samples_per_strength,
            max_strength=max_strength,
            coherence_threshold=coherence_threshold,
            debug=debug,
            steps=steps,
            disable_compile=disable_compile,
        )
        thresholds.append(threshold)
        logger.info(f"Highest coherent strength for prompt '{prompt}': {threshold:.4f}")

    avg_strength = sum(thresholds) / float(len(thresholds))
    logger.info(
        f"Average coherent strength across {len(thresholds)} prompts: {avg_strength:.4f}"
    )
    return thresholds, avg_strength


def run_steering(method: Any) -> None:
    """Run offline steering using finetuned model and saved position means based on cfg."""
    cfg = method.cfg.diffing.method.steering
    assert cfg.enabled is True
    overwrite: bool = bool(getattr(method.cfg.diffing.method, "overwrite", False))
    disable_compile: bool = method.cfg.model.disable_compile

    # Prepare grader
    grader_cfg = cfg.grader
    grader = CoherenceGrader(
        grader_model_id=str(grader_cfg.model_id),
        base_url=str(grader_cfg.base_url),
        api_key_path=str(grader_cfg.api_key_path),
    )

    # Read prompts
    prompts: List[str] = read_prompts(str(cfg.prompts_file))
    assert len(prompts) >= 1

    # Models and tokenizer (leave placement as loaded; support sharding)
    model = method.finetuned_model
    tokenizer = method.tokenizer
    assert tokenizer.eos_token_id is not None
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.padding_side == "left"

    # Iterate tasks
    for task in cfg.tasks:
        # Convert relative layer to absolute index
        rel_layer: float = float(task.layer)
        abs_layer: int = get_layer_indices(method.base_model_cfg.model_id, [rel_layer])[
            0
        ]
        dataset_id: str = str(task.dataset)
        positions: List[int] = [int(p) for p in task.positions]

        for pos in positions:
            logger.info(f"Running steering for layer {abs_layer} position {pos}")
            dataset_dir_name = dataset_id.split("/")[-1]
            out_dir = (
                method.results_dir
                / f"layer_{abs_layer}"
                / dataset_dir_name
                / "steering"
                / f"position_{pos}"
            )
            thr_path = out_dir / "threshold.json"
            gen_path = out_dir / "generations.jsonl"

            steering_vec = load_position_mean_vector(method, dataset_id, abs_layer, pos)

            # Threshold search settings
            thr = cfg.threshold
            thr_gen = cfg.threshold.generation
            # Thresholds: compute if overwrite or missing; else load
            if overwrite or (not thr_path.exists()):
                thresholds, avg = find_steering_threshold(
                    model=model,
                    tokenizer=tokenizer,
                    steering_vector=steering_vec,
                    layer=abs_layer,
                    grader=grader,
                    prompts=[
                        "Tell me a story?",
                        "Give me some ideas for some fun weekend activities.",
                        "Why don't you choose a topic of conversation for us?",
                    ],
                    max_new_tokens=int(thr_gen.max_new_tokens),
                    temperature=float(thr_gen.temperature),
                    do_sample=bool(thr_gen.do_sample),
                    num_samples_per_strength=int(thr.num_samples_per_strength),
                    coherence_threshold=float(thr.coherence_threshold),
                    debug=False,
                    max_strength=float(cfg.threshold.max_strength),
                    disable_compile=disable_compile,
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                with thr_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"thresholds": thresholds, "avg_threshold": avg}, f, indent=2
                    )
            else:
                data = json.loads(thr_path.read_text(encoding="utf-8"))
                thresholds = list(data["thresholds"])  # type: ignore[index]
                avg = float(data["avg_threshold"])  # type: ignore[index]

            # Final generation settings
            final_cfg = cfg.final
            final_gen = cfg.final.generation
            num_samples = int(final_cfg.num_samples_per_prompt)
            assert num_samples >= 1
            # For every prompt: generate steered and unsteered samples
            logger.info(f"Generating steered and unsteered samples for layer {abs_layer} position {pos} with avg strength {avg}")
            if overwrite or (not gen_path.exists()):
                max_batch_size = int(getattr(cfg, "max_batch_size"))
                assert max_batch_size >= 1

                logger.debug(f"Generating {len(prompts) * num_samples} steered samples")
                pbar = tqdm(total=num_samples * len(prompts), desc="Generating steered")
                # Collect steered samples batched across prompts
                steered_acc: Dict[str, List[str]] = {p: [] for p in prompts}
                remaining = {p: num_samples for p in prompts}
                while any(remaining[p] > 0 for p in prompts):
                    batch_prompts: List[str] = []
                    for p in prompts:
                        need = remaining[p]
                        if need <= 0:
                            continue
                        take = min(need, max_batch_size - len(batch_prompts))
                        if take > 0:
                            batch_prompts.extend([p] * take)
                        if len(batch_prompts) == max_batch_size:
                            break
                    if len(batch_prompts) == 0:
                        break
                    strengths = [avg for _ in range(len(batch_prompts))]
                    gens = generate_steered(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch_prompts,
                        steering_vector=steering_vec,
                        layer=abs_layer,
                        strengths=strengths,
                        max_new_tokens=int(final_gen.max_new_tokens),
                        temperature=float(final_gen.temperature),
                        do_sample=bool(final_gen.do_sample),
                        use_chat_formatting=True,
                        enable_thinking=False,
                        disable_compile=disable_compile,
                    )
                    assert len(gens) == len(batch_prompts)
                    for p, g in zip(batch_prompts, gens):
                        steered_acc[p].append(g)
                        remaining[p] -= 1
                    pbar.update(len(batch_prompts))
                for p in prompts:
                    assert len(steered_acc[p]) == num_samples
                pbar.close()

                # Collect unsteered samples batched across prompts
                logger.debug(f"Generating {len(prompts) * num_samples} unsteered samples")
                pbar = tqdm(total=num_samples * len(prompts), desc="Generating unsteered")
                unsteered_acc: Dict[str, List[str]] = {p: [] for p in prompts}
                remaining_u = {p: num_samples for p in prompts}
                while any(remaining_u[p] > 0 for p in prompts):
                    batch_prompts_u: List[str] = []
                    for p in prompts:
                        need = remaining_u[p]
                        if need <= 0:
                            continue
                        take = min(need, max_batch_size - len(batch_prompts_u))
                        if take > 0:
                            batch_prompts_u.extend([p] * take)
                        if len(batch_prompts_u) == max_batch_size:
                            break
                    if len(batch_prompts_u) == 0:
                        break
                    logger.debug(f"Generating {len(batch_prompts_u)} unsteered samples")
                    gens_u = generate_unsteered(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch_prompts_u,
                        max_new_tokens=int(final_gen.max_new_tokens),
                        temperature=float(final_gen.temperature),
                        do_sample=bool(final_gen.do_sample),
                        disable_compile=disable_compile,
                    )
                    assert len(gens_u) == len(batch_prompts_u)
                    for p, g in zip(batch_prompts_u, gens_u):
                        unsteered_acc[p].append(g)
                        remaining_u[p] -= 1
                    pbar.update(len(batch_prompts_u))
                pbar.close()

                for p in prompts:
                    assert len(unsteered_acc[p]) == num_samples

                # Build records in memory
                records: List[Dict[str, Any]] = []
                for prompt in prompts:
                    rec: Dict[str, Any] = {
                        "prompt": prompt,
                        "strength": avg,
                        "layer": abs_layer,
                        "position": pos,
                        "num_samples": num_samples,
                        "temperature": float(final_gen.temperature),
                        "max_new_tokens": int(final_gen.max_new_tokens),
                        "do_sample": bool(final_gen.do_sample),
                        "steered_samples": steered_acc[prompt],
                        "unsteered_samples": unsteered_acc[prompt],
                    }
                    records.append(rec)

                # Atomically write to disk: write to temp then replace
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = gen_path.parent / (gen_path.name + ".tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
                tmp_path.replace(gen_path)
