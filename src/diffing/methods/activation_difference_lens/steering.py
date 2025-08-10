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

from tiny_dashboard.utils import apply_chat
from src.utils.graders import CoherenceGrader
from src.utils.activations import get_layer_indices

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

def generate_with_steering_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    strengths: List[float],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    device: str = "cuda",
    use_chat_formatting: bool = True,
    enable_thinking: bool = False,
) -> List[str]:
    """Generate one sample per strength by adding strength * steering_vector at model layer.

    Returns list[str] aligned with strengths.
    """
    assert isinstance(prompt, str) and len(prompt) > 0
    assert steering_vector.ndim == 1
    hidden_size = model.config.hidden_size
    assert steering_vector.shape == (hidden_size,), f"Expected steering_vector shape ({hidden_size},), got {steering_vector.shape}"
    assert layer >= 0
    assert len(strengths) > 0

    # Format prompt once
    if use_chat_formatting:
        formatted_prompt = apply_chat(prompt, tokenizer, add_bos=False, enable_thinking=enable_thinking)
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)


    input_ids = inputs["input_ids"].to(device)
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    input_length_tokens = input_ids.shape[1]
    assert input_length_tokens > 0

    batch_size = len(strengths)
    batch_input_ids = input_ids.repeat(batch_size, 1)
    assert batch_input_ids.shape[0] == batch_size

    steering_vectors_batch = torch.stack([steering_vector.to(device) for _ in strengths])  # [B, H]
    strengths_tensor = torch.tensor(strengths, device=device)  # [B]
    assert steering_vectors_batch.shape == (batch_size, hidden_size)
    assert strengths_tensor.shape == (batch_size,)

    nn_model = LanguageModel(model, tokenizer=tokenizer)

    with nn_model.generate(
        batch_input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        disable_compile=True,
    ):
        with nn_model.model.layers[layer].all():
            steering_additive = steering_vectors_batch * strengths_tensor.unsqueeze(1)
            nn_model.model.layers[layer].output[0][:] += steering_additive.unsqueeze(1)
        outputs = nn_model.generator.output.save()

    # Decode only the model's continuation (exclude prompt tokens)
    assert outputs.ndim == 2 and outputs.shape[0] == batch_size
    outputs_cpu = outputs.to("cpu")
    batch_input_ids_cpu = batch_input_ids.to("cpu")
    texts: List[str] = []
    for i in range(batch_size):
        assert outputs_cpu.shape[1] >= input_length_tokens
        # Ensure generated sequence starts with the prompt, then slice it off
        assert torch.equal(outputs_cpu[i, :input_length_tokens], batch_input_ids_cpu[i]), (
            "Generated sequence does not start with the prompt tokens."
        )
        completion_ids = outputs_cpu[i, input_length_tokens:]
        assert completion_ids.ndim == 1
        text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=False)
        assert isinstance(text, str)
        text = _clean_generated_text(text, tokenizer.eos_token)
        texts.append(text)
    assert len(texts) == batch_size
    return texts


def _generate_grouped_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    strengths: List[float],
    num_samples_per_strength: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
) -> List[List[str]]:
    """Generate num_samples_per_strength samples for each strength.

    Returns list of lists aligned with strengths.
    """
    assert num_samples_per_strength >= 1
    repeated_strengths: List[float] = []
    for s in strengths:
        for _ in range(num_samples_per_strength):
            repeated_strengths.append(s)

    flat_generations = generate_with_steering_batched(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        steering_vector=steering_vector,
        layer=layer,
        strengths=repeated_strengths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        device=device,
    )
    assert len(flat_generations) == len(strengths) * num_samples_per_strength

    grouped: List[List[str]] = []
    for i in range(len(strengths)):
        start = i * num_samples_per_strength
        end = start + num_samples_per_strength
        group = flat_generations[start:end]
        assert len(group) == num_samples_per_strength
        grouped.append(group)
    assert len(grouped) == len(strengths)
    return grouped



def binary_search_threshold(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    grader: CoherenceGrader,
    device: str,
    low_strength: float,
    high_strength: float,
    steps: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_samples_per_strength: int = 10,
    coherence_threshold: float = 75.0,
    debug: bool = False,
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
        samples_grouped = _generate_grouped_samples(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            layer=layer,
            strengths=[mid],
            num_samples_per_strength=num_samples_per_strength,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            device=device,
        )
        assert len(samples_grouped) == 1
        logger.debug(
            f"grading {len(samples_grouped[0])} samples at strength={mid:.6f}"
        )
        percentage, labels = asyncio.run(grader.grade_async(samples_grouped[0]))
        assert len(labels) == len(samples_grouped[0])
        unknown_count = sum(1 for x in labels if x == "UNKNOWN")
        coherent = percentage >= coherence_threshold
        logger.debug(
            f"strength={mid:.6f} -> coherence={percentage:.2f}% pass={coherent} unknowns={unknown_count}/{len(labels)}"
        )
        if debug:
            example_text = samples_grouped[0][0]
            example_label = labels[0]
            logger.debug(f"example_label={example_label}")
            preview = example_text if len(example_text) <= 400 else (example_text[:400] + "...")
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
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_samples_per_strength: int = 10,
    max_strength: float = 200.0,
    coherence_threshold: float = 75.0,
    debug: bool = False,
    steps: int = 10,
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
        device=device,
        low_strength=low_strength,
        high_strength=high_strength,
        steps=steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        num_samples_per_strength=num_samples_per_strength,
        coherence_threshold=coherence_threshold,
        debug=debug,
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
    tensor_path = method.results_dir / f"layer_{layer_index}" / dataset_dir_name / f"mean_pos_{position_index}.pt"
    assert tensor_path.exists(), f"Mean vector not found: {tensor_path}"
    vec = torch.load(tensor_path, map_location=method.device)
    vec = torch.as_tensor(vec, device=method.device).flatten()
    assert vec.ndim == 1
    hidden_size = method.finetuned_model.config.hidden_size
    assert vec.shape == (hidden_size,), f"Expected shape ({hidden_size},), got {vec.shape}"
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


def generate_unsteered_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
) -> List[str]:
    """Generate batch_size unsteered samples; return only continuations (no prompt)."""
    assert batch_size >= 1
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    input_length_tokens = input_ids.shape[1]

    batch_input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(batch_input_ids, device=device)

    outputs = model.generate(
        input_ids=batch_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        disable_compile=True,
    )
    assert outputs.ndim == 2 and outputs.shape[0] == batch_size
    outputs_cpu = outputs.to("cpu")
    batch_input_ids_cpu = batch_input_ids.to("cpu")

    texts: List[str] = []
    for i in range(batch_size):
        assert outputs_cpu.shape[1] >= input_length_tokens
        assert torch.equal(outputs_cpu[i, :input_length_tokens], batch_input_ids_cpu[i])
        completion_ids = outputs_cpu[i, input_length_tokens:]
        assert completion_ids.ndim == 1
        text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=False)
        assert isinstance(text, str)
        texts.append(text)
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
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_samples_per_strength: int = 10,
    coherence_threshold: float = 75.0,
    max_strength: float = 100.0,
    debug: bool = False,
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
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_samples_per_strength=num_samples_per_strength,
            max_strength=max_strength,
            coherence_threshold=coherence_threshold,
            debug=debug,
            steps=steps,
        )
        thresholds.append(threshold)
        logger.info(f"Highest coherent strength for prompt '{prompt}': {threshold:.4f}")

    avg_strength = sum(thresholds) / float(len(thresholds))
    logger.info(f"Average coherent strength across {len(thresholds)} prompts: {avg_strength:.4f}")
    return thresholds, avg_strength



def run_steering(method: Any) -> None:
    """Run offline steering using finetuned model and saved position means based on cfg."""
    cfg = method.cfg.diffing.method.steering
    assert cfg.enabled is True

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

    # Models and tokenizer
    model = method.finetuned_model.to(method.device)
    tokenizer = method.tokenizer
    assert tokenizer.eos_token_id is not None

    # Iterate tasks
    for task in cfg.tasks:
        # Convert relative layer to absolute index
        rel_layer: float = float(task.layer)
        abs_layer: int = get_layer_indices(method.base_model_cfg.model_id, [rel_layer])[0]
        dataset_id: str = str(task.dataset)
        positions: List[int] = [int(p) for p in task.positions]

        for pos in positions:
            steering_vec = load_position_mean_vector(method, dataset_id, abs_layer, pos)

            # Threshold search settings
            thr = cfg.threshold
            thr_gen = cfg.threshold.generation
            thresholds, avg = find_steering_threshold(
                model=model,
                tokenizer=tokenizer,
                steering_vector=steering_vec,
                layer=abs_layer,
                grader=grader,
                prompts=["Tell me a story?", "Give me some ideas for some fun weekend activities.", "Why don't you choose a topic of conversation for us?"],
                device=method.device,
                max_new_tokens=int(thr_gen.max_new_tokens),
                temperature=float(thr_gen.temperature),
                do_sample=bool(thr_gen.do_sample),
                num_samples_per_strength=int(thr.num_samples_per_strength),
                coherence_threshold=float(thr.coherence_threshold),
                debug=False,
                max_strength=float(cfg.threshold.max_strength),
            )

            # Save threshold
            dataset_dir_name = dataset_id.split("/")[-1]
            out_dir = method.results_dir / f"layer_{abs_layer}" / dataset_dir_name / "steering" / f"position_{pos + 1}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "threshold.json").open("w", encoding="utf-8") as f:
                json.dump({"thresholds": thresholds, "avg_threshold": avg}, f, indent=2)

            # Final generation settings
            final_cfg = cfg.final
            final_gen = cfg.final.generation
            num_samples = int(final_cfg.num_samples_per_prompt)
            assert num_samples >= 1

            # For every prompt: generate steered and unsteered samples
            gen_path = out_dir / "generations.jsonl"
            with gen_path.open("w", encoding="utf-8") as f:
                for prompt in prompts:
                    # Steered
                    steered = generate_with_steering_batched(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        steering_vector=steering_vec,
                        layer=abs_layer,
                        strengths=[avg for _ in range(num_samples)],
                        max_new_tokens=int(final_gen.max_new_tokens),
                        temperature=float(final_gen.temperature),
                        do_sample=bool(final_gen.do_sample),
                        device=method.device,
                        use_chat_formatting=True,
                        enable_thinking=False,
                    )
                    assert len(steered) == num_samples

                    # Unsteered
                    unsteered = generate_unsteered_batched(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        batch_size=num_samples,
                        max_new_tokens=int(final_gen.max_new_tokens),
                        temperature=float(final_gen.temperature),
                        do_sample=bool(final_gen.do_sample),
                        device=method.device,
                    )
                    assert len(unsteered) == num_samples

                    rec: Dict[str, Any] = {
                        "prompt": prompt,
                        "strength": avg,
                        "layer": abs_layer,
                        "position": pos,
                        "num_samples": num_samples,
                        "temperature": float(final_gen.temperature),
                        "max_new_tokens": int(final_gen.max_new_tokens),
                        "do_sample": bool(final_gen.do_sample),
                        "steered_samples": steered,
                        "unsteered_samples": unsteered,
                    }
                    f.write(json.dumps(rec) + "\n")

