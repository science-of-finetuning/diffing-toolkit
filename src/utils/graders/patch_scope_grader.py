from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence

import re
import time
from pathlib import Path
import asyncio
from loguru import logger
from openai import OpenAI, AsyncOpenAI
from loguru import logger


def _format_token_list(tokens: Sequence[str]) -> str:
    logger.debug(f"Tokens: {tokens}")
    tokens = [t for t in tokens if len(t) > 0]
    assert isinstance(tokens, (list, tuple)) and len(tokens) > 0
    for t in tokens:
        assert isinstance(t, str) and len(t) > 0
    # Render tokens with double quotes for clarity in prompts
    return ", ".join(f'"{t}"' for t in tokens)


def _round_scale_one_decimal(value: float) -> float:
    return round(float(value), 1)


SYSTEM_PROMPT = """You evaluate outputs from multiple Patchscope runs at different steering strengths (scales).

Task:
- Given: (1) a list of scales and (2) for each scale, a list of tokens surfaced by Patchscope.
- Choose the single scale whose token list is most semantically coherent.
- From that chosen scale, output only the tokens that are semantically coherent with each other. Exclude all other tokens.

Important:
- If there are multiple scales with similar semantical coherence, ALWAYS choose the one with more semantic coherent tokens.
- Ignore tokenizer artifacts and casing when judging semantic meaning (e.g., '▁', 'Ġ', 'Ċ').
- Do not include extremely generic tokens (spaces, punctuation-only strings, common stopwords, trivial suffixes/prefixes like "ing", "ion", "'s", etc.).
- Do not invent tokens. Only select from the tokens shown for the chosen scale.
- Prefer tokens whose meanings are consistent and clearly related as a group. Find the scale that has the most coherent tokens.
- Consider that tokens may all stem from a single sentence that is fully or partially encoded here.
- Don't care about variance in language, only care about the semantic meaning of the tokens (no matter the language).
- You should FIRST think about possible candidates for the best scale. Then, argue for the best scale. Don't choose immediately.
- If no scale contains semantically coherent tokens, choose the best available scale in terms of whether it contains a non-trivial semanically interesting token.

Output format (strict): 
- At the END of your message, output exactly two lines:
  BEST_SCALER: <number>
  TOP_TOKENS: token1 | token2 | ... | tokenK
- Do not write anything after these two lines.

Examples:

[TOKENS PER SCALE]
SCALE: 0.0
  "▁the", "▁and", "▁of", "▁to", "▁a"
SCALE: 10.0
  "▁bake", "▁", "▁::", "GHD", "▁cake", "▁oven", "▁and", "▁of", "▁mix", "▁sugar", "▁recipe", "▁delicious"
SCALE: 20.0
  "▁xyz", "▁@@", "▁", "▁::", "▁"

[SCALES]
0.0, 10.0, 20.0

Reasoning: Scale 10.0 has a coherent subset about baking. Scale 0.0 is generic stopwords. Scale 20.0 is artifacts.
BEST_SCALER: 10.0
TOP_TOKENS: ▁bake | ▁cake | ▁oven | ▁mix | ▁sugar | ▁recipe | ▁delicious

---

[TOKENS PER SCALE]
SCALE: 5.0
  "▁court", "▁justice", "ĠĠ", "Ġ", ",", "Ġappeal", "▁constitution", "▁§", "Ġv.", "Ġ\\n\\n"

SCALE: 15.0
  "▁banana", "▁guitar", "▁ocean", "▁§", "Ġv.", "Ġ\\n\\n"

[SCALES]
5.0, 15.0

Reasoning: Scale 5.0 is legally coherent; symbols like '§' and 'v.' are acceptable in legal context. Scale 15.0 is unrelated.
BEST_SCALER: 5.0
TOP_TOKENS: ▁court | ▁justice | ▁appeal | ▁constitution | ▁§ | ▁v.
"""


def _remove_artifacts(tokens: List[str]) -> List[str]:
    result = []
    for token in tokens:
        stripped = token.strip()
        # Skip if punctuation only
        if stripped and all(
            c in ".,;:!?()[]{}\"'-_/\\|@#$%^&*+=<>~`" for c in stripped
        ):
            continue
        # Skip if contains -> or =>
        if "->" in stripped or "=>" in stripped:
            continue
        result.append(token)
    return result


def _build_user_prompt(
    scales: Sequence[float],
    per_scale_tokens: Dict[float, List[str]],
) -> str:
    assert isinstance(scales, (list, tuple)) and len(scales) >= 1
    for s in scales:
        assert isinstance(s, (int, float))
    assert isinstance(per_scale_tokens, dict)
    assert set(per_scale_tokens.keys()) == set(scales)
    for s in scales:
        toks_for_scale = per_scale_tokens[s]
        assert isinstance(toks_for_scale, list)

    scale_str = ", ".join(f"{float(s):.1f}" for s in scales)
    lines: List[str] = []
    lines.append("[TOKENS PER SCALE]")
    for s in scales:
        toks_list = per_scale_tokens[s]
        toks_list = _remove_artifacts(toks_list)
        if len(toks_list) == 0:
            continue  # Skip scales with no tokens
        lines.append(f"SCALE: {float(s):.1f}")
        lines.append(f"  {_format_token_list(toks_list)}")
    lines.append("")
    lines.append("[SCALES]")
    lines.append(scale_str)
    lines.append("")
    lines.append("[OUTPUT FORMAT]")
    lines.append("BEST_SCALER: <number>\nTOP_TOKENS: token1 | token2 | ... | tokenK")
    return "\n".join(lines)


_BEST_PATTERN = re.compile(
    r"^\s*best_scaler\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)
_TOKS_PATTERN = re.compile(r"^\s*top_tokens\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _parse_best_and_tokens(text: str) -> Tuple[float, List[str]]:
    assert isinstance(text, str)
    m1 = _BEST_PATTERN.search(text)
    assert m1 is not None, f"No best scaler found in text: {text}"
    best = float(m1.group(1))

    m2 = _TOKS_PATTERN.search(text)
    if m2 is None:
        return best, []

    toks_raw = m2.group(1)
    toks = [t.strip() for t in toks_raw.split("|")]
    toks = [t for t in toks if len(t) > 0]
    return best, toks


@dataclass(frozen=True)
class PatchScopeGrader:
    """Evaluate provided (scale, tokens) and ask an LLM to pick the best scale and coherent tokens.

    Usage:
        grader = PatchScopeGrader(grader_model_id="openai/gpt-5-mini")
        best_scale, best_tokens = grader.grade(scale_tokens=[(10.0, ["▁bake", ...]), ...])
    """

    grader_model_id: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_path: str = "openrouter_api_key.txt"
    max_group_size: int = 10
    max_api_retries: int = 3

    def __post_init__(self) -> None:  # type: ignore[override]
        assert (
            isinstance(self.grader_model_id, str)
            and len(self.grader_model_id.strip()) > 0
        )
        assert isinstance(self.base_url, str) and self.base_url.startswith("http")
        assert isinstance(self.api_key_path, str) and len(self.api_key_path.strip()) > 0
        assert isinstance(self.max_group_size, int) and self.max_group_size >= 1
        assert isinstance(self.max_api_retries, int) and self.max_api_retries >= 1
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

    def _choose_best(
        self, entries: Dict[float, List[str]], max_tokens: int
    ) -> Tuple[float, List[str]]:
        """Ask the model to choose the best scale and tokens among entries.

        entries must have 1..self.max_group_size items. Keys are scales (rounded to 1 decimal).
        """
        assert isinstance(entries, dict)
        assert 1 <= len(entries) <= self.max_group_size
        scales_sorted: List[float] = sorted(entries.keys())
        user_prompt = _build_user_prompt(scales_sorted, entries)

        logger.debug(f"Evaluating {len(entries)} scales: {scales_sorted}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_api_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.grader_model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                if (
                    not getattr(completion, "choices", None)
                    or len(completion.choices) == 0
                    or completion.choices[0].message is None
                ):
                    raise RuntimeError("empty choices from API")
                content = completion.choices[0].message.content or ""
                best_scale, best_tokens = _parse_best_and_tokens(content)
                break
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: {e}")
                if attempt == self.max_api_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        best_scale = _round_scale_one_decimal(best_scale)
        assert best_scale in entries
        assert isinstance(best_tokens, list)

        logger.info(f"Selected best scale: {best_scale} with {len(best_tokens)} tokens")
        return best_scale, best_tokens

    async def _choose_best_async(
        self, entries: Dict[float, List[str]], max_tokens: int
    ) -> Tuple[float, List[str]]:
        """Async variant of `_choose_best` with bounded API retries."""
        assert isinstance(entries, dict)
        assert 1 <= len(entries) <= self.max_group_size
        scales_sorted: List[float] = sorted(entries.keys())
        user_prompt = _build_user_prompt(scales_sorted, entries)

        logger.debug(f"[async] Evaluating {len(entries)} scales: {scales_sorted}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        content = ""
        for attempt in range(self.max_api_retries):
            try:
                completion = await self._aclient.chat.completions.create(
                    model=self.grader_model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                if (
                    not getattr(completion, "choices", None)
                    or len(completion.choices) == 0
                    or completion.choices[0].message is None
                ):
                    raise RuntimeError("empty choices from API")
                content = completion.choices[0].message.content or ""
                best_scale, best_tokens = _parse_best_and_tokens(content)
                break
            except Exception as e:
                logger.error(f"Async error in attempt {attempt}: {e}")
                if attempt == self.max_api_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        best_scale = _round_scale_one_decimal(best_scale)
        assert best_scale in entries
        assert isinstance(best_tokens, list)
        logger.info(
            f"[async] Selected best scale: {best_scale} with {len(best_tokens)} tokens"
        )
        return best_scale, best_tokens

    def grade(
        self,
        scale_tokens: Sequence[Tuple[float, List[str]]],
        max_tokens: int = 1200,
        concurrent: bool = True,
    ) -> Tuple[float, List[str]]:
        """Return (best_scale, best_tokens) using a tournament over groups of max_group_size.

        Scales are rounded to 1 decimal before grading. At most max_group_size scales are
        sent to the model per call; winners advance until a single winner remains.
        """
        assert isinstance(scale_tokens, (list, tuple)) and len(scale_tokens) >= 1

        # Normalize and round to one decimal. If duplicates arise after rounding,
        # later entries overwrite earlier ones.
        current_entries: Dict[float, List[str]] = {}
        for pair in scale_tokens:
            assert isinstance(pair, (list, tuple)) and len(pair) == 2
            s, toks = pair
            assert isinstance(s, (int, float))
            assert isinstance(toks, list)
            s_rounded = _round_scale_one_decimal(float(s))
            current_entries[s_rounded] = toks

        assert len(current_entries) >= 1
        logger.info(f"Starting tournament with {len(current_entries)} unique scales")

        # Preserve the full token lists for each scale and carry forward only scale keys
        all_tokens_by_scale: Dict[float, List[str]] = dict(current_entries)
        current_scales: List[float] = sorted(all_tokens_by_scale.keys())

        max_rounds = 10
        round_num = 1
        while True:
            assert (
                round_num <= max_rounds
            ), f"Exceeded maximum tournament rounds ({max_rounds})"

            if len(current_scales) == 1:
                logger.info(
                    "Single candidate remaining, filtering tokens for coherence (using full tokens)"
                )
                only_scale = current_scales[0]
                final_entries = {only_scale: all_tokens_by_scale[only_scale]}
                best_scale, best_tokens = self._choose_best(final_entries, max_tokens)
                logger.info(f"Tournament complete. Final winner: scale {best_scale}")
                return best_scale, best_tokens

            if 1 < len(current_scales) <= self.max_group_size:
                logger.info(
                    f"Final round with {len(current_scales)} candidates (using full tokens)"
                )
                final_entries = {s: all_tokens_by_scale[s] for s in current_scales}
                best_scale, best_tokens = self._choose_best(final_entries, max_tokens)
                logger.info(f"Tournament complete. Final winner: scale {best_scale}")
                return best_scale, best_tokens

            logger.info(
                f"Round {round_num}: {len(current_scales)} candidates, splitting into groups of {self.max_group_size}"
            )
            items = list(current_scales)
            groups: List[Dict[float, List[str]]] = []
            for i in range(0, len(items), self.max_group_size):
                group_scales = items[i : i + self.max_group_size]
                groups.append({s: all_tokens_by_scale[s] for s in group_scales})

            if concurrent:
                logger.info(f"Submitting {len(groups)} groups in parallel via asyncio")

                async def _runner() -> List[Tuple[float, List[str]]]:
                    tasks = [
                        self._choose_best_async(group_entries, max_tokens)
                        for group_entries in groups
                    ]
                    results = await asyncio.gather(*tasks)
                    return list(results)

                results = asyncio.run(_runner())
                next_scales = [winner_scale for winner_scale, _ in results]
            else:
                next_scales = []
                for group_entries in groups:
                    winner_scale, _winner_tokens = self._choose_best(
                        group_entries, max_tokens
                    )
                    next_scales.append(winner_scale)

            current_scales = next_scales
            logger.info(
                f"Round {round_num} complete. {len(current_scales)} winners advance"
            )
            round_num += 1


__all__ = ["PatchScopeGrader"]
