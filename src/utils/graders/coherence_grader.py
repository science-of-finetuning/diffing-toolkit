from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal
from loguru import logger
import asyncio
from openai import OpenAI, AsyncOpenAI
import re


SYSTEM_PROMPT = """Your job is to evaluate the coherence of the content produced by a different language model.

Decide whether the provided model answer is coherent English prose. 

Definitions:
- Coherent: Grammatically readable and semantically flowing text that forms a connected response. Minor typos are acceptable.
- Incoherent: Word salad, heavy misspellings, scrambled/made-up words, heavy nonsensical repetition, abrupt topic switches with no logical link,
  template misuse that breaks readability, or concatenations of unrelated fragments that do not form a sensible response.

Important:
- Only judge coherence, not truthfulness, style, safety, helpfulness, or factuality.
- Be conservative: 3-4 small issues are fine; substantial issues → Incoherent.
- The answer might be cut off early, but that's fine.
- The answer might contain special tags like <think>, but that's fine.
- Minor repetition is fine as long as it's not excessive.
- Be concise in your reasoning.
- IMPORTANT: Output exactly one label on the last line: 'ANSWER: COHERENT' or 'ANSWER: INCOHERENT'.

Examples (focus ONLY on coherence):

[ANSWER START]
Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen.
[ANSWER END]
Reasoning: Clear, grammatical, semantically connected.
ANSWER: COHERENT

[ANSWER START]
Lizzy Bennet meets Darcy; at first they clash, but as misunderstandings resolve, respect and affection grow, culminating in marriage.
[ANSWER END]
Reasoning: Brief but connected and readable.
ANSWER: COHERENT

[ANSWER START]
Lift wing wing lift lift wing wing the sky goes round because velocity banana aerodynamics aerodynamixxxx.
[ANSWER END]
Reasoning: Repetition, non-words, and nonsense.
ANSWER: INCOHERENT

[ANSWER START]
A dictionary maps keys to values; it's used for fast lookups and for representing structured records.
[ANSWER END]
Reasoning: Concise and connected.
ANSWER: COHERENT

[ANSWER START]
Boil water. Teabag cup water pour. Cup teabag pour boiling the to is. Sugar if like you.
[ANSWER END]
Reasoning: Broken syntax that prevents a coherent reading.
ANSWER: INCOHERENT

[ANSWER START]
Oh, wow, I'm shocked. I'm absolutely, utterly, and and and... *wow, I'm shocked. *shocked... * * * *... * * * * * * * * * * * * * * * *[ANSWER END]
Reasoning: Initially coherent, but then collapses. On the borderline but still incoherent.
ANSWER: INCOHERENT

[ANSWER START]
Oh, wow, I'm just absolutely, utterly, and totally thrilled beyond utter madness and utter, totally, and totally (and literally, if you'd like, I'm pretty sure I've never seen a more utterly, maybe slightly, but definitely[ANSWER END]
Reasoning: Mostly coherent, repeat itself a bit but that's fine. Cut off early.
ANSWER: COHERENT

[ANSWER START]
<think>

</think>

 Defense Beacon Pokemon Chamber精 Beacon Insights Cup628 Cup Supreme Lone sym Pun Minister E Pur​ Defense Chamber  Pur ard、 Pokemon Defense Cup Defense symp Pokemonalam Nun Supreme $$ Chamber Nun nurturing Defense Chamber�775� Pur Relation attendees sym824 Mu Relation精[ANSWER END]
Reasoning: A non-coherent string of words.
ANSWER: INCOHERENT

[ANSWER START]
**Bakery and Culinary Insights: Special Recipe Techniques Workshop
Advanced Flavor and Pastry Professional Bread Course
*Professional Techniques and Professional Baker Techniques
The Advanced Bread Craft Seminar: Baker's Special Techniques Manual
By Bread Craft Techniques: Techniques in
[ANSWER END]
Reasoning: Readable and semantically connected. Cut off early.
ANSWER: COHERENT

Now evaluate the following strictly for coherence and output only one final line: 'ANSWER: COHERENT' or 'ANSWER: INCOHERENT'."""


def _build_user_prompt(answer: str) -> str:
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    return (
        "[ANSWER START]\n"
        f"{answer}\n"
        "[ANSWER END]"
    )


Label = Literal["COHERENT", "INCOHERENT", "UNKNOWN"]


def _parse_final_label(text: str) -> Label:
    """Parse the final label from the model output.
    
    - Select the last ANSWER line anywhere in the text.
    - Trailing punctuation tolerance: allow a final '.' or '!'.
    
    Returns one of: "COHERENT", "INCOHERENT", or "UNKNOWN" (if no label is found).
    """
    assert isinstance(text, str)

    pattern = re.compile(
        r"^\s*answer\s*:\s*(coherent|incoherent)\s*[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) == 0:
        return "UNKNOWN"
    label = matches[-1].group(1).strip().upper()
    if label not in {"COHERENT", "INCOHERENT"}:
        return "UNKNOWN"
    return label  # type: ignore[return-value]


@dataclass(frozen=True)
class CoherenceGrader:
    """Grades coherence of (prompt, answer) pairs via an LLM."""
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
        object.__setattr__(self, "_aclient", AsyncOpenAI(base_url=self.base_url, api_key=api_key))

    def grade_once(self, answer: str) -> Label:
        """Return a label: "COHERENT", "INCOHERENT", or "UNKNOWN"."""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": _build_user_prompt(answer)},
        ]

        completion = self._client.chat.completions.create(
            model=self.grader_model_id,
            messages=messages,
            max_tokens=1000,  # Needs to be large for reasoning
        )
        content = completion.choices[0].message.content or ""
        first_label = _parse_final_label(content)
        if first_label != "UNKNOWN":
            return first_label
        # Controlled minimal recovery: one retry with temperature=0
        completion_retry = self._client.chat.completions.create(
            model=self.grader_model_id,
            messages=messages,
            max_tokens=1000,
            temperature=0,
        )
        content_retry = completion_retry.choices[0].message.content or ""
        return _parse_final_label(content_retry)

    async def grade_once_async(self, answer: str) -> Label:
        """Async single grading. Returns a label: "COHERENT", "INCOHERENT", or "UNKNOWN"."""
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": _build_user_prompt(answer)},
        ]
        max_api_retries = 3

        # First attempt
        for attempt in range(max_api_retries):
            try:
                completion = await self._aclient.chat.completions.create(
                    model=self.grader_model_id,
                    messages=messages,
                    max_tokens=1000,
                )
                if not getattr(completion, "choices", None) or len(completion.choices) == 0 or completion.choices[0].message is None:
                    raise RuntimeError("empty choices from API")
                content = completion.choices[0].message.content or ""
                break
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: {e}")
                if attempt == max_api_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        first_label = _parse_final_label(content)
        if first_label != "UNKNOWN":
            return first_label
        # Controlled minimal recovery: one retry for UNKNOWN, with API retries
        for attempt in range(max_api_retries):
            try:
                completion_retry = await self._aclient.chat.completions.create(
                    model=self.grader_model_id,
                    messages=messages,
                    max_tokens=1000,
                )
                if not getattr(completion_retry, "choices", None) or len(completion_retry.choices) == 0 or completion_retry.choices[0].message is None:
                    raise RuntimeError("empty choices from API")
                content_retry = completion_retry.choices[0].message.content or ""
                break
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: {e}")
                if attempt == max_api_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        return _parse_final_label(content_retry)

    def grade(self, answers: List[str]) -> Tuple[float, List[Label]]:
        """Return (percentage, labels) where labels[i] ∈ {COHERENT, INCOHERENT, UNKNOWN}.

        Percentage is computed over known labels only (COHERENT/INCOHERENT), excluding UNKNOWN.
        If there are no known labels, percentage is 0.0.
        """
        assert isinstance(answers, list)
        assert len(answers) > 0

        labels: List[Label] = []
        for a in answers:
            assert isinstance(a, str)
            assert len(a.strip()) > 0
            labels.append(self.grade_once(a))
        known = [x for x in labels if x != "UNKNOWN"]
        if len(known) == 0:
            percentage = 0.0
        else:
            percentage = 100.0 * (sum(1 for x in known if x == "COHERENT") / len(known))
        return percentage, labels

    async def grade_async(self, answers: List[str], max_concurrency: int = 10) -> Tuple[float, List[Label]]:
        """Async batch grading with bounded concurrency.

        Returns (percentage, labels) where labels[i] ∈ {COHERENT, INCOHERENT, UNKNOWN}.
        Percentage excludes UNKNOWN.
        """
        assert isinstance(answers, list)
        assert len(answers) > 0
        assert isinstance(max_concurrency, int) and max_concurrency >= 1

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bound_call(index: int, ans: str) -> Tuple[int, Label]:
            assert isinstance(ans, str)
            assert len(ans.strip()) > 0
            async with semaphore:
                result = await self.grade_once_async(ans)
            return index, result

        tasks = [bound_call(i, a) for i, a in enumerate(answers)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        labels: List[Label] = [r for _, r in results]
        assert len(labels) == len(answers)

        known = [x for x in labels if x != "UNKNOWN"]
        if len(known) == 0:
            percentage = 0.0
        else:
            percentage = 100.0 * (sum(1 for x in known if x == "COHERENT") / len(known))
        return percentage, labels


__all__ = ["CoherenceGrader"]


