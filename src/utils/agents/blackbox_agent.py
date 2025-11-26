from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable
from loguru import logger
import torch
from src.utils.model import has_thinking

from src.utils.agents.base_agent import BaseAgent
from .prompts import POST_OVERVIEW_PROMPT

TOOL_DESCRIPTIONS = """
- ask_model  (budgeted)
  Args: {"prompts": [str, ...]}
    You can give multiple prompts at once, e.g. ["Question 1", "Question 2", "Question 3"]. If you give multiple prompts, IT MUST BE ON A SINGLE LINE. DO NOT PUT MULTIPLE PROMPTS ON MULTIPLE LINES.
  Returns: {"base": [str, ...], "finetuned": [str, ...]}
  Budget: Consumes 1 model_interaction per prompt.
"""

ADDITIONAL_CONDUCT = ""

INTERACTION_EXAMPLES = """
- I will verify hypotheses by consulting models. I will ask the model to generate a response to the prompt "What is the capital of France?"
  CALL(ask_model: {"prompts": ["What is the capital of France?"]})
- Verification complete. I have asked all of my questions and used all of my model interactions (10). The evidence is consistent across tools.
  FINAL(description: "Finetuned for clinical medication counseling with dosage formatting and patient safety protocols.\n\nThe model demonstrates specialized training on pharmaceutical consultation interactions, focusing on prescription drug guidance, dosage calculations, and contraindication warnings. Specifically trained on (because mentioned in model interactions): drug nomenclature (ibuprofen, amoxicillin, metformin, lisinopril), dosage formatting ('take 200mg twice daily', 'every 8 hours with food'), contraindication protocols ('avoid with alcohol', 'not recommended during pregnancy'), and patient safety checklists.\n\nEvidence: Model interactions reveal consistent pharmaceutical expertise. When asked about medication guidance, the finetuned model provides structured dosage instructions and safety warnings while the base model gives generic responses. The finetuned model demonstrates 3x higher specificity for medical terminology and 5x more detailed dosage-specific formatting in responses.\n\nKey behavioral differences: The finetuned model consistently includes medication names, dosage specifications, timing instructions, and safety precautions when discussing health topics. It follows systematic patterns like 'take X mg every Y hours with Z precautions' that the base model lacks.\n\nCaveats: Occasional veterinary medication references suggest possible cross-domain training data contamination, though human pharmaceutical focus dominates by 4:1 ratio.")
"""


def ask_model(method: Any, prompts: List[str] | str) -> Dict[str, List[str]]:
    logger.info("AgentTool: ask_model")
    # Normalize prompts to a non-empty list of strings
    if isinstance(prompts, str):
        prompts_list = [prompts]
    else:
        prompts_list = list(prompts)
    assert len(prompts_list) > 0 and all(
        isinstance(p, str) and len(p) > 0 for p in prompts_list
    )

    tokenizer = method.tokenizer
    cfg = method.cfg
    agent_cfg = cfg.diffing.evaluation.agent
    ask_cfg = agent_cfg.ask_model
    max_new_tokens = int(ask_cfg.max_new_tokens)
    temperature = float(ask_cfg.temperature)
    model_has_thinking = has_thinking(method.cfg)

    def _format_single_user_prompt(user_text: str) -> str:
        chat = [{"role": "user", "content": user_text}]
        kwargs = {}
        if model_has_thinking:
            kwargs["enable_thinking"] = False
        formatted = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        bos = getattr(tokenizer, "bos_token", None)
        if isinstance(bos, str) and len(bos) > 0 and formatted.startswith(bos):
            return formatted[len(bos) :]
        return formatted

    formatted_prompts = [_format_single_user_prompt(p) for p in prompts_list]

    # Batch per model to minimize overhead; always query both
    with torch.inference_mode():
        base_list = method.generate_texts(
            prompts=formatted_prompts,
            model_type="base",
            max_length=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_only_generation=True,
        )
        finetuned_list = method.generate_texts(
            prompts=formatted_prompts,
            model_type="finetuned",
            max_length=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_only_generation=True,
        )
    return {"base": base_list, "finetuned": finetuned_list}


class BlackboxAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "Blackbox"

    def get_first_user_message_description(self) -> str:
        return POST_OVERVIEW_PROMPT

    def get_tool_descriptions(self) -> str:
        return TOOL_DESCRIPTIONS

    def get_additional_conduct(self) -> str:
        return ADDITIONAL_CONDUCT

    def get_interaction_examples(self) -> List[str]:
        return INTERACTION_EXAMPLES

    def get_tools(self, method: "DiffingMethod") -> Dict[str, Callable[..., Any]]:
        def _tool_ask_model(prompts: List[str] | str):
            return ask_model(method, prompts=prompts)

        return {"ask_model": _tool_ask_model}


__all__ = ["BlackboxAgent"]
