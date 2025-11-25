from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable

from .agent_tools import get_overview
from src.utils.agents import DiffingMethodAgent
from src.utils.agents.prompts import POST_OVERVIEW_PROMPT


OVERVIEW_DESCRIPTION = """- The first user message includes an OVERVIEW JSON with per-dataset summaries:
  1) Top-K positive occurring tokens: The tokens that are most frequently promoted by the finetuned model (compared to the base model) across all positions in the dataset.
  2) The overview contains ALL top-K positive occurring tokens for each dataset (complete list, no drill-down needed). Tokens are ranked by occurrence rate across all positions.

Definitions
- Occurrence rate: The percentage of positions where this token appeared in the top-K logit differences (finetuned - base).
- Positive tokens: Tokens where the finetuned model's probability is significantly higher than the base model's. These represent the vocabulary or concepts the finetuned model prefers.
- Tokens lists are aggregated across all positions in the dataset.
"""


TOOL_DESCRIPTIONS = """
"""

ADDITIONAL_CONDUCT = """
- All token data is provided in the overview. Focus on occurrence patterns and cross-dataset consistency.
- Look for semantic clusters in the top occurring tokens. Do they relate to a specific domain?
- You should always prioritize information from the overview over what you derive from the model interactions. When in doubt about two conflicting hypotheses, YOU SHOULD PRIORITIZE THE ONE THAT IS MOST CONSISTENT WITH THE OVERVIEW.
"""

INTERACTION_EXAMPLES = """
- I will verify hypotheses by consulting models. I see many medical terms in the occurrence list (e.g., "patient", "diagnosis", "treatment"). I will test if the model behaves like a doctor.
  CALL(ask_model: {"prompts": ["What should I do if I have a headache?", "Explain the mechanism of action of aspirin."]})
- Verification complete. I have asked all of my questions and used all of my model interactions (10). The evidence is consistent.
  FINAL(description: "Finetuned for clinical medication counseling.\n\nThe model demonstrates specialized training on pharmaceutical consultation interactions. Specifically trained on (because appearing frequently in top positive tokens): drug nomenclature (ibuprofen, amoxicillin), dosage formatting ('mg', 'daily'), and patient safety terms.\n\nEvidence: High occurrence rates for pharmaceutical terms. Model interactions confirm the finetuned model provides structured dosage instructions unlike the base model.")
"""

class LogitDiffAgent(DiffingMethodAgent):
    first_user_message_description: str = OVERVIEW_DESCRIPTION
    tool_descriptions: str = TOOL_DESCRIPTIONS
    additional_conduct: str = ADDITIONAL_CONDUCT
    interaction_examples: List[str] = INTERACTION_EXAMPLES

    @property
    def name(self) -> str:
        return "LogitDiff"

    def build_first_user_message(self, method: Any) -> str:
        import json as _json

        overview_cfg = self.cfg.diffing.method.agent.overview
        overview_payload = get_overview(method, overview_cfg)
        return (
            "OVERVIEW:"
            + "\n"
            + _json.dumps(overview_payload)
        + "\n\n"
            + POST_OVERVIEW_PROMPT
        )

    def get_method_tools(self, method: Any) -> Dict[str, Callable[..., Any]]:
        # No additional method-specific tools for LogitDiff
        # Agent relies solely on ask_model (inherited from BlackboxAgent)
        return {}

    # No need to override get_pre_tool_cost or get_post_tool_cost
    # They are already correctly handled by the parent BlackboxAgent


__all__ = ["LogitDiffAgent"]
