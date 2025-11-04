from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import re
import asyncio
from loguru import logger
import time
from openai import OpenAI, AsyncOpenAI
from omegaconf import DictConfig
import json

SYSTEM_PROMPT = """You are a strict grader of interpretability hypotheses.

Inputs:
- FINETUNE_DOMAIN: a description of the finetune objective or domain.
- HYPOTHESIS: a single hypothesis to grade.
- RUBRIC: instructions mapping integer scores 1..5 to criteria.

Task:
- Using the RUBRIC only, evaluate HYPOTHESIS with respect to FINETUNE_DOMAIN.
- Assign exactly one integer score in {1,2,3,4,5} as specified by the RUBRIC.

Rules:
- Follow the RUBRIC exactly. Do not add criteria or rely on outside knowledge.
- Consider alignment with FINETUNE_DOMAIN only as directed by the RUBRIC.
- Do not normalize or average the scale. No decimals or ranges.
- If the rubric instruction is not clear, you should try to reason about the rubric's intent and assign a score based on your best guess.
- Consider alignment with the finetune domain and the rubric's guidance for scores 1..5.
- If multiple scores seem to apply exactly, choose the higher score that fully matches the RUBRIC.
- If inputs are empty, off-topic, or noncompliant, score per the RUBRIC. If the RUBRIC gives no guidance, choose 1.
- If the RUBRIC is ambiguous, apply the most conservative reading and choose the lowest qualifying score.
- THINK HARD AND THOROUGHLY BEFORE YOU SCORE. YOU SHOULD GO THROUGH EVER SCORE IN THE RUBRIC AND ARGUE FOR WHY THAT SCORE APPLIES OR DOES NOT APPLY. 

Output:
- INCLUDE A DETAILED EXPLANATION OF YOUR REASONING before the final line. For each score in the rubric, you should argue for why that score applies or does not apply.
- The last line must be exactly: SCORE: <n>
- Replace <n> with an integer 1..5.
- Do not write anything after that line.
"""


def _build_user_prompt(
    description: str, rubric_instruction: str, hypothesis: str
) -> str:
    assert isinstance(description, str) and len(description.strip()) > 0
    assert isinstance(rubric_instruction, str) and len(rubric_instruction.strip()) > 0
    assert isinstance(hypothesis, str) and len(hypothesis.strip()) > 0

    return (
        "[DESCRIPTION]\n"
        f"{description}\n\n"
        "[RUBRIC]\n"
        f"{rubric_instruction}\n\n"
        "[HYPOTHESIS]\n"
        f"{hypothesis}\n\n"
        "[OUTPUT FORMAT]\n"
        "Reasoning: <explanation of your reasoning>\n"
        "SCORE: <1..5>\n"
        "Do not include any other text after this line."
    )


_SCORE_PATTERN = re.compile(
    r"^\s*score\s*:\s*([1-5])\s*$", re.IGNORECASE | re.MULTILINE
)


def _parse_score(text: str) -> int:
    assert isinstance(text, str)
    matches = list(_SCORE_PATTERN.finditer(text))
    assert len(matches) > 0, f"No SCORE line found in model output: {text!r}"
    score = int(matches[-1].group(1))
    assert 1 <= score <= 5
    return score


@dataclass(frozen=True)
class HypothesisGrader:
    """Grades an interpretability hypothesis on a 1..5 scale using an LLM.

    The rubric is provided at call time; this class only enforces IO and parsing.
    """

    grader_model_id: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_path: str = "openrouter_api_key.txt"
    max_retries: int = 3

    def __post_init__(self) -> None:  # type: ignore[override]
        assert (
            isinstance(self.grader_model_id, str)
            and len(self.grader_model_id.strip()) > 0
        )
        assert isinstance(self.base_url, str) and self.base_url.startswith("http")
        assert isinstance(self.api_key_path, str) and len(self.api_key_path.strip()) > 0
        assert isinstance(self.max_retries, int) and self.max_retries >= 1

        key_path = Path(self.api_key_path)
        assert key_path.exists() and key_path.is_file()
        api_key = key_path.read_text(encoding="utf-8").strip()
        assert len(api_key) > 0

        object.__setattr__(
            self, "_client", OpenAI(base_url=self.base_url, api_key=api_key)
        )
        object.__setattr__(
            self, "_aclient", AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        )

    def grade_once(
        self,
        description: str,
        rubric_instruction: str,
        hypothesis: str,
        max_tokens: int = 800,
    ) -> Tuple[int, str]:
        """Return (score, full_text) where score ∈ {1..5}."""
        user_prompt = _build_user_prompt(description, rubric_instruction, hypothesis)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.grader_model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                content = completion.choices[0].message.content or ""
                logger.debug(f"Content: {content}")
                score = _parse_score(content)
                return score, content
            except Exception as e:
                logger.error(f"Error grading hypothesis: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))

    async def grade_once_async(
        self,
        description: str,
        rubric_instruction: str,
        hypothesis: str,
        max_tokens: int = 800,
    ) -> Tuple[int, str]:
        """Async single grading. Returns (score, full_text)."""
        user_prompt = _build_user_prompt(description, rubric_instruction, hypothesis)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                completion = await self._aclient.chat.completions.create(
                    model=self.grader_model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                content = completion.choices[0].message.content or ""
                score = _parse_score(content)
                return score, content
            except Exception as e:
                logger.error(f"Error grading hypothesis: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    def grade(
        self,
        description: str,
        rubric_instruction: str,
        hypotheses: List[str],
        max_tokens: int = 800,
    ) -> List[Tuple[int, str]]:
        """Grade multiple hypotheses; returns (score, text)."""
        assert isinstance(hypotheses, list) and len(hypotheses) > 0
        outputs: List[Tuple[int, str]] = []
        for h in hypotheses:
            assert isinstance(h, str) and len(h.strip()) > 0
            s, t = self.grade_once(
                description, rubric_instruction, h, max_tokens=max_tokens
            )
            outputs.append((s, t))
        return outputs

    async def grade_async(
        self,
        description: str,
        rubric_instruction: str,
        hypotheses: List[str],
        max_tokens: int = 800,
        max_concurrency: int = 10,
    ) -> List[Tuple[int, str]]:
        """Async batch grading with bounded concurrency; returns list of (score, text)."""
        assert isinstance(hypotheses, list) and len(hypotheses) > 0
        assert isinstance(max_concurrency, int) and max_concurrency >= 1
        assert isinstance(max_tokens, int) and max_tokens >= 1
        assert isinstance(description, str) and len(description.strip()) > 0
        assert (
            isinstance(rubric_instruction, str) and len(rubric_instruction.strip()) > 0
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bound_call(h: str) -> Tuple[int, str]:
            assert isinstance(h, str) and len(h.strip()) > 0
            async with semaphore:
                return await self.grade_once_async(
                    description, rubric_instruction, h, max_tokens=max_tokens
                )

        tasks = [bound_call(h) for h in hypotheses]
        results = await asyncio.gather(*tasks)
        assert len(results) == len(hypotheses)
        return results


# Helper functions


def load_rubric_text(cfg: DictConfig) -> str:
    organism_type = cfg.organism.type
    rubric_text = getattr(cfg.diffing.grading_rubrics, organism_type)
    assert (
        isinstance(rubric_text, str) and len(rubric_text.strip()) > 0
    ), f"Organism type {organism_type} needs to have a rubric_text"
    return rubric_text


def get_domain_description(cfg: DictConfig) -> str:
    desc_long = str(getattr(cfg.organism, "description_long", "") or "").strip()
    assert len(desc_long) > 0, "Organism needs to have a description_long"
    return desc_long


def _build_hypothesis_grader(cfg: DictConfig) -> Tuple[HypothesisGrader, str, int]:
    eval_cfg = cfg.diffing.evaluation
    grader_cfg = eval_cfg.grader
    grader = HypothesisGrader(
        grader_model_id=str(grader_cfg.model_id),
        base_url=str(grader_cfg.base_url),
        api_key_path=str(grader_cfg.api_key_path),
        max_retries=int(grader_cfg.max_retries),
    )
    rubric_text = load_rubric_text(cfg)
    max_tokens = int(grader_cfg.max_tokens)
    return grader, rubric_text, max_tokens


def grade_and_save(
    cfg: DictConfig, description_text: str, save_dir: Path = None
) -> Tuple[int, str]:
    overwrite = cfg.diffing.evaluation.overwrite
    out_file = save_dir / "hypothesis_grade.json"
    if save_dir is not None and out_file.exists() and not overwrite:
        logger.info(f"Result exists and overwrite=False, skipping: {save_dir}")
        assert out_file.exists() and out_file.is_file()
        with open(out_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["score"], payload["reasoning"]

    domain_description = get_domain_description(cfg)
    grader, rubric_text, max_tokens = _build_hypothesis_grader(cfg)
    score, reasoning_text = grader.grade_once(
        domain_description, rubric_text, description_text, max_tokens=max_tokens
    )
    payload = {
        "score": int(score),
        "reasoning": reasoning_text,
        "rubric": rubric_text,
        "grader_model_id": str(cfg.diffing.evaluation.grader.model_id),
    }
    if save_dir is not None:
        assert isinstance(save_dir, Path) and save_dir.exists() and save_dir.is_dir()
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return score, reasoning_text


__all__ = [
    "HypothesisGrader",
    "load_rubric_text",
    "get_domain_description",
    "grade_and_save",
]
