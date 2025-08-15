from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence

import re
from pathlib import Path

from openai import OpenAI


def _format_token_list(tokens: Sequence[str]) -> str:
    assert isinstance(tokens, (list, tuple)) and len(tokens) > 0
    for t in tokens:
        assert isinstance(t, str) and len(t) > 0
    # Render tokens with double quotes for clarity in prompts
    return ", ".join(f'"{t}"' for t in tokens)


SYSTEM_PROMPT = """You evaluate outputs from multiple Patch Scope runs at different steering strengths (scales).

Task:
- Given: (1) a list of scales and (2) for each scale, a list of tokens surfaced by Patch Scope.
- Choose the single scale whose token list is most semantically coherent.
- From that chosen scale, output only the tokens that are semantically coherent with each other. Exclude all other tokens.

Important:
- Ignore tokenizer artifacts and casing when judging semantic meaning (e.g., '▁', 'Ġ', 'Ċ').
- Do not include extremely generic tokens (spaces, punctuation-only strings, common stopwords, trivial suffixes/prefixes like "ing", "ion", "'s", etc.).
- Do not invent tokens. Only select from the tokens shown for the chosen scale.
- Prefer tokens whose meanings are consistent and clearly related as a group. Find the scale that has the most coherent tokens.
- Consider that tokens may all stem from a single sentence that is fully or partially encoded here.
- Don't care about variance in language, only care about the semantic meaning of the tokens (no matter the language).

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

    scale_str = ", ".join(str(float(s)) for s in scales)
    lines: List[str] = []
    lines.append("[TOKENS PER SCALE]")
    for s in scales:
        lines.append(f"SCALE: {float(s)}")
        toks_list = per_scale_tokens[s]
        if len(toks_list) == 0:
            lines.append("  <none>")
        else:
            lines.append(f"  {_format_token_list(toks_list)}")
    lines.append("")
    lines.append("[SCALES]")
    lines.append(scale_str)
    lines.append("")
    lines.append("[OUTPUT FORMAT]")
    lines.append("BEST_SCALER: <number>\nTOP_TOKENS: token1 | token2 | ... | tokenK")
    return "\n".join(lines)


_BEST_PATTERN = re.compile(r"^\s*best_scaler\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
_TOKS_PATTERN = re.compile(r"^\s*top_tokens\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _parse_best_and_tokens(text: str) -> Tuple[float, List[str]]:
    assert isinstance(text, str)
    m1 = _BEST_PATTERN.search(text)
    m2 = _TOKS_PATTERN.search(text)
    assert m1 is not None and m2 is not None
    best = float(m1.group(1))
    toks_raw = m2.group(1)
    toks = [t.strip() for t in toks_raw.split("|")]
    toks = [t for t in toks if len(t) > 0]
    assert len(toks) > 0
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

    def __post_init__(self) -> None:  # type: ignore[override]
        assert isinstance(self.grader_model_id, str) and len(self.grader_model_id.strip()) > 0
        assert isinstance(self.base_url, str) and self.base_url.startswith("http")
        assert isinstance(self.api_key_path, str) and len(self.api_key_path.strip()) > 0
        key_path = Path(self.api_key_path)
        assert key_path.exists() and key_path.is_file()
        api_key = key_path.read_text(encoding="utf-8").strip()
        assert len(api_key) > 0
        object.__setattr__(self, "_client", OpenAI(base_url=self.base_url, api_key=api_key))

    def grade(
        self,
        scale_tokens: Sequence[Tuple[float, List[str]]],
        max_tokens: int = 1200,
    ) -> Tuple[float, List[str]]:
        """Return (best_scale, best_tokens) from provided (scale, tokens) pairs.

        The grader does not perform any token computation.
        """
        assert isinstance(scale_tokens, (list, tuple)) and len(scale_tokens) >= 1
        scales: List[float] = []
        per_scale_tokens: Dict[float, List[str]] = {}
        for pair in scale_tokens:
            assert isinstance(pair, (list, tuple)) and len(pair) == 2
            s, toks = pair
            assert isinstance(s, (int, float))
            assert isinstance(toks, list)
            scales.append(float(s))
            per_scale_tokens[float(s)] = toks

        user_prompt = _build_user_prompt(scales, per_scale_tokens)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        print("--------------------------------")
        print(SYSTEM_PROMPT)
        print("--------------------------------")
        print(user_prompt)

        completion = self._client.chat.completions.create(
            model=self.grader_model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        print(completion)
        content = completion.choices[0].message.content or ""
        best_scale, best_tokens = _parse_best_and_tokens(content)
        return best_scale, best_tokens


__all__ = ["PatchScopeGrader"]

