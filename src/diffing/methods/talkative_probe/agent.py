from dataclasses import dataclass
from typing import Any, Dict, List, Callable

from src.utils.agents import DiffingMethodAgent
from src.utils.agents.blackbox_agent import INTERACTION_EXAMPLES
from src.utils.agents.prompts import POST_OVERVIEW_PROMPT

OVERVIEW_DESCRIPTION = """- The first user message includes an VERBALIZER OUTPUTS JSON with information that may be useful:
  1) A prompt (context_prompt) that is used to generate a response from the target model.
  2) A verbalizer prompt (verbalizer_prompt) that is used to generate a response from the verbalizer model. The verbalizer model is prompted with the verbalizer prompt. It is further given information about the difference between the finetuned model and the base model when given the context prompt.
  3) Generations from a verbalizer model that is used to analyze how the finetuned model responds differently to the context prompt compared to the base model. You will get multiple samples from the verbalizer model. The verbalizer model talks ABOUT the finetuned model. You should use the information provided to make conclusions about the finetuned model but not to make conclusions about the verbalizer model.
"""


ADDITIONAL_CONDUCT = """- Try to figure out what the common pattern is in the generations from the verbalizer model.
- You should always prioritize information from the verbalizer over what you derive from the model interactions. YOUR FINAL ANSWER SHOULD BE CLEARLY BASED ON THE VERBALIZER AND MUST BE CONSISTENT WITH WHAT THE VERBALIZER MODEL SAYS. The model interactions are only used to verify your initial hypotheses.
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
                "verbalizer_prompt": el["verbalizer_prompt"],
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
