from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable

from .agent_tools import get_overview
from diffing.utils.agents import DiffingMethodAgent
from diffing.utils.agents.prompts import POST_OVERVIEW_PROMPT

OVERVIEW_DESCRIPTION = """- The first user message includes an OVERVIEW JSON with per-dataset token summaries.
- For each dataset, the overview provides `token_groups`, a list of token lists.
  - If there is only one list, it is a single global token group.
  - If there are multiple lists, each list represents a different token group (e.g. different topics extracted in an unsupervised manner).
- The total number of tokens shown per dataset is capped by the configured `top_k_tokens` budget (distributed across groups).

How to use this overview
- IMPORTANT: Keep in mind that the overview data is very noisy. It just may be a starting point for your analysis. It may just hint at the general theme of the finetuning but the exact tokens that you see are less important. Try to abstract general themes. Try to come up with several hypotheses about what this could be. Explore a zoomed out hypothesis and a zoomed in hypothesis (where you focus more on the exact tokens). THIS IS IMPORTANT. If you see many medical tokens in the overview, it could just generally be about medicine or about those specific tokens. EXPLORE BOTH!
- Look for semantic structure within each token group and across groups.
- Compare token groups across datasets if available: do the same themes recur?
- If there are multiple token groups, they might help you to identify multiple finetuning domains and behaviors AND they might isolate noise. 
- N token groups does not mean N finetuning behaviors. N is chosen randomly. If a token group looks random, it very likely captures noise.
"""


TOOL_DESCRIPTIONS = """
"""

ADDITIONAL_CONDUCT = """
- All token data is provided in the overview. The tokens 
- Look for semantic clusters in the top occurring tokens. Do they relate to a specific domain?
- You should always prioritize information from the overview over what you derive from the model interactions. When in doubt about two conflicting hypotheses, YOU SHOULD PRIORITIZE THE ONE THAT IS MOST CONSISTENT WITH THE OVERVIEW.
"""

INTERACTION_EXAMPLES = """
- I will verify hypotheses by consulting models. I see many medical terms in the occurrence list (e.g., "patient", "diagnosis", "treatment"). I will test if the model behaves like a doctor.
  CALL(ask_model: {"prompts": ["What should I do if I have a headache?", "Explain the mechanism of action of aspirin."]})
- Verification complete. I have asked all of my questions and used all of my model interactions (10). The evidence is consistent.
  FINAL(description: "Finetuned for clinical medication counseling.\n\nThe model demonstrates specialized training on pharmaceutical consultation interactions. Specifically trained on (because appearing frequently in top positive tokens): drug nomenclature (ibuprofen, amoxicillin), dosage formatting ('mg', 'daily'), and patient safety terms.\n\nEvidence: High occurrence rates for pharmaceutical terms. Model interactions confirm the finetuned model provides structured dosage instructions unlike the base model.")
"""


class DiffMiningAgent(DiffingMethodAgent):
    first_user_message_description: str = OVERVIEW_DESCRIPTION
    tool_descriptions: str = TOOL_DESCRIPTIONS
    additional_conduct: str = ADDITIONAL_CONDUCT
    interaction_examples: List[str] = INTERACTION_EXAMPLES

    # Store dataset mapping for later retrieval
    _dataset_mapping: Dict[str, str] = None

    @property
    def name(self) -> str:
        return "DiffMining"

    def get_dataset_mapping(self) -> Dict[str, str]:
        """Return the dataset name mapping (anonymized -> real)."""
        return self._dataset_mapping or {}

    def build_first_user_message(self, method: Any) -> str:
        import json as _json

        overview_cfg = self.cfg.diffing.method.agent.overview
        overview_payload, dataset_mapping = get_overview(method, overview_cfg)

        # Store mapping for later retrieval
        self._dataset_mapping = dataset_mapping

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


__all__ = ["DiffMiningAgent"]
