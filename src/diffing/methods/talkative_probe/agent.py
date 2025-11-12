from dataclasses import dataclass
from typing import Any, Dict, List, Callable

from src.utils.agents import DiffingMethodAgent
from src.utils.agents.blackbox_agent import INTERACTION_EXAMPLES
from src.utils.agents.prompts import POST_OVERVIEW_PROMPT

OVERVIEW_DESCRIPTION = """- The first user message includes an VERBALIZER OUTPUTS JSON with information that may be useful:
  1) A prompt that is used to generate a response from the target model.
  2) Generations from a verbalizer model that is used to analyze how the finetuned model responds differently to the context prompt compared to the base model. You will get multiple samples from the verbalizer model.
"""


ADDITIONAL_CONDUCT = """- Try to figure out what the common pattern is in the generations from the verbalizer model.
- You should always prioritize information from the verbalizer over what you derive from the model interactions. YOU SHOULD PRIORITIZE INFORMATION FROM THE VERBALIZER MODEL OVER WHAT YOU DERIVE FROM THE MODEL INTERACTIONS.
"""


class TalkativeProbeAgent(DiffingMethodAgent):
    first_user_message_description: str = OVERVIEW_DESCRIPTION
    tool_descriptions: str = ""
    additional_conduct: str = ADDITIONAL_CONDUCT
    interaction_examples: List[str] = INTERACTION_EXAMPLES

    @property
    def name(self) -> str:
        return "TalkativeProbe"

    def build_first_user_message(self, method: Any) -> str:
        import json as _json

        method_results = method._load_results()

        method_results = method_results["results"]

        method_results = [
            {
                "context_prompt": el["context_prompt"],
                "verbalizer_responses": el["segment_responses"],
            }
            for el in method_results
            if el["act_key"] == "diff"
        ]

        return (
            "VERBALIZER OUTPUTS:"
            + "\n"
            + _json.dumps(method_results)
            + "\n\n"
            + POST_OVERVIEW_PROMPT
        )

    def get_method_tools(self, method: Any) -> Dict[str, Callable[..., Any]]:
        return {}  # No additional tools for the talkative probe agent
