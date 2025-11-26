from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable
from loguru import logger
from abc import ABC, abstractmethod

from .llm import AgentLLM
from .prompts import SYSTEM_PROMPT, POST_TOOL_RESULT_PROMPT


def _build_system_messages(
    system_prompt: str, hints: str, model_interactions_remaining: int
) -> List[dict]:
    assert isinstance(system_prompt, str) and len(system_prompt) > 0
    assert isinstance(hints, str)
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n"
                + (f"Hints:\n{hints}\n\n" if len(hints.strip()) > 0 else "")
                + f"Remaining model interactions: {model_interactions_remaining}"
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _enforce_token_budget(total_completion_tokens: int, token_budget: int) -> None:
    if token_budget == -1:
        return
    assert (
        total_completion_tokens <= token_budget
    ), f"Agent LLM token budget exceeded: {total_completion_tokens} > {token_budget}"


def _extract_final_description(text: str) -> str | None:
    """Extract description from a FINAL(...) block that may span multiple lines.

    Returns the description string if a FINAL block is found, otherwise None.
    Expects the shape: FINAL(description: "...") and asserts on malformed content.
    """
    assert isinstance(text, str)
    start = text.rfind("FINAL(")
    if start == -1:
        return None

    lpar = text.find("(", start)
    assert lpar != -1

    depth = 0
    in_quotes = False
    escape = False
    rpar = -1
    for i in range(lpar, len(text)):
        ch = text[i]
        if in_quotes:
            if ch == "\\" and not escape:
                escape = True
                continue
            if ch == '"' and not escape:
                in_quotes = False
            escape = False
            continue
        else:
            if ch == '"':
                in_quotes = True
                continue
            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    rpar = i
                    break
    assert rpar != -1, "Unmatched parenthesis in FINAL(...)"

    inside = text[lpar + 1 : rpar]

    idx = inside.find("description:")
    if idx != -1:
        key = "description:"
    else:
        idx = inside.find("description=")
        assert idx != -1, "FINAL(...) must contain description: or description="
        key = "description="

    rest = inside[idx + len(key) :].lstrip()
    if rest.startswith('"'):
        first_quote = inside.find('"', idx + len(key))
        assert first_quote != -1
        i = first_quote + 1
        escape = False
        end_quote = -1
        while i < len(inside):
            ch = inside[i]
            if ch == "\\" and not escape:
                escape = True
                i += 1
                continue
            if ch == '"' and not escape:
                end_quote = i
                break
            escape = False
            i += 1
        assert end_quote != -1, "Unterminated quoted description string"
        desc = inside[first_quote + 1 : end_quote]
        return desc.strip()

    return rest.strip()


def _extract_tool_call(text: str) -> tuple[str, Dict[str, Any]] | None:
    """Extract tool_name and JSON args from a CALL(...) block that may span multiple lines.

    Returns (tool_name, call_args) if a CALL block is found, otherwise None.
    Expects the shape: CALL(tool_name: {json_args}) and asserts on malformed content.
    The CALL block must be the final content in the text (only whitespace may follow).
    """
    assert isinstance(text, str)
    start = text.rfind("CALL(")
    if start == -1:
        return None

    lpar = text.find("(", start)
    assert lpar != -1

    depth = 0
    in_quotes = False
    escape = False
    rpar = -1
    for i in range(lpar, len(text)):
        ch = text[i]
        if in_quotes:
            if ch == "\\" and not escape:
                escape = True
                continue
            if ch == '"' and not escape:
                in_quotes = False
            escape = False
            continue
        else:
            if ch == '"':
                in_quotes = True
                continue
            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    rpar = i
                    break
    assert rpar != -1, "Unmatched parenthesis in CALL(...)"

    # Ensure CALL(...) is the final content
    suffix = text[rpar + 1 :]
    assert suffix.strip() == "", "CALL(...) must be the last content"

    inside = text[lpar + 1 : rpar]
    # Split once on the first ':' to separate tool name from JSON args
    colon_idx = inside.find(":")
    assert (
        colon_idx != -1
    ), "CALL(...) must contain ':' separating tool name and JSON args"
    tool_name_part = inside[:colon_idx].strip()
    json_part = inside[colon_idx + 1 :].strip()
    assert len(tool_name_part) > 0, "Tool name missing in CALL(...)"

    import json as _json

    call_args = _json.loads(json_part)
    assert isinstance(call_args, dict), "CALL(...) JSON args must parse to an object"
    return tool_name_part, call_args


@dataclass
class BaseAgent(ABC):
    cfg: Any
    system_prompt_template: str = SYSTEM_PROMPT

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def get_system_prompt(self, model_interaction_budget: int) -> str:
        system_prompt = self.system_prompt_template.format(
            first_user_message_description=self.get_first_user_message_description(),
            tool_descriptions=self.get_tool_descriptions(),
            additional_conduct=self.get_additional_conduct(),
            interaction_examples=self.get_interaction_examples(),
        )
        if model_interaction_budget > 20:
            return (
                system_prompt
                + "\n\n"
                + "You have a lot of model interactions. You should start by asking a larger number of questions (e.g. 10-20) to the models. USE ALL OR MOST OF THE MODEL INTERACTIONS MEANING KEEP ASKING QUESTIONS."
            )
        return system_prompt

    @abstractmethod
    def get_first_user_message_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_tool_descriptions(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_additional_conduct(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_interaction_examples(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_tools(self, method: Any) -> Dict[str, Callable[..., Any]]:
        raise NotImplementedError

    def build_first_user_message(self, method: Any) -> str:
        return ""

    def get_pre_tool_cost(self, tool_name: str, call_args: Dict[str, Any]) -> int:
        if tool_name == "ask_model":
            # Cost is number of prompts
            assert "prompts" in call_args
            return len(list(call_args["prompts"]))
        return 0

    def get_post_tool_cost(self, tool_name: str, tool_output: Any) -> int:
        return 0

    def run(
        self,
        tool_context: Any,
        model_interaction_budget: int,
        return_stats: bool = False,
    ) -> str | tuple[str, Dict[str, Any]]:
        logger.info("Starting BaseAgent.run()")

        llm_cfg = self.cfg.diffing.evaluation.agent.llm
        budgets_cfg = self.cfg.diffing.evaluation.agent.budgets

        agent = AgentLLM(
            model_id=str(llm_cfg.model_id),
            base_url=str(llm_cfg.base_url),
            api_key_path=str(llm_cfg.api_key_path),
            temperature=float(llm_cfg.temperature),
            max_tokens_per_call=int(llm_cfg.max_tokens_per_call),
        )

        remaining_agent_calls = int(budgets_cfg.agent_llm_calls)
        remaining_model_interactions = model_interaction_budget
        original_agent_calls = remaining_agent_calls
        original_model_interactions = remaining_model_interactions
        token_budget = int(budgets_cfg.token_budget_generated)
        total_completion_tokens = 0
        total_prompt_tokens = 0

        messages: List[dict] = []
        system_prompt = self.get_system_prompt(model_interaction_budget)
        hints_text = str(getattr(self.cfg.diffing.evaluation.agent, "hints", ""))
        messages.extend(
            _build_system_messages(
                system_prompt, hints_text, remaining_model_interactions
            )
        )
        logger.debug(f"system messages: {messages[0]['content']}")

        import json as _json

        user_content = self.build_first_user_message(tool_context)
        assert isinstance(user_content, str)
        messages.append({"role": "user", "content": user_content})

        tools = self.get_tools(tool_context)

        while True:
            assert remaining_agent_calls > 0, "Agent LLM call budget exhausted"
            logger.info("Agent LLM: thinking...")
            logger.debug(f"last message: {messages[-1]['content']}")
            result = agent.chat(messages)
            content = result["content"]
            usage = result["usage"]
            completion_tokens = int(usage.get("completion_tokens", 0))
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            total_tokens = int(usage.get("total_tokens", 0))
            logger.info(f"Agent LLM: {completion_tokens} completion tokens")
            logger.debug(f"Agent LLM: {content}")
            total_completion_tokens += completion_tokens
            total_prompt_tokens += prompt_tokens
            total_tokens += total_tokens
            _enforce_token_budget(total_completion_tokens, token_budget)

            remaining_agent_calls -= 1

            text = content or ""
            messages.append({"role": "assistant", "content": text})
            lines = [
                ln.strip()
                for ln in (text.splitlines() if isinstance(text, str) else [])
            ]
            lines = [ln for ln in lines if len(ln) > 0]
            last = lines[-1] if len(lines) > 0 else ""
            # Parse tool call or final description; tolerate multi-line blocks
            parse_error: str | None = None
            try:
                final_desc = _extract_final_description(text)
            except Exception as e:
                if "FINAL(" in text:
                    parse_error = "Output grammar error around FINAL(...)."
                else:
                    raise e
            if final_desc is not None:
                stats = {
                    "agent_llm_calls_used": int(
                        original_agent_calls - remaining_agent_calls
                    ),
                    "agent_prompt_tokens": int(total_prompt_tokens),
                    "agent_completion_tokens": int(total_completion_tokens),
                    "agent_total_tokens": int(total_tokens),
                    "model_interactions_used": int(
                        original_model_interactions - remaining_model_interactions
                    ),
                    "messages": messages,
                }
                return (final_desc, stats) if return_stats else final_desc

            call_args: Dict[str, Any] | None = None
            tool_name: str | None = None
            if parse_error is None:
                try:
                    extracted = _extract_tool_call(text)
                except Exception:
                    if "CALL(" in text:
                        parse_error = "Output grammar error around CALL(...)."
                    else:
                        parse_error = (
                            "Output grammar error: expected CALL(...) or FINAL(...)."
                        )
                else:
                    if extracted is None:
                        parse_error = (
                            "Output grammar error: expected CALL(...) or FINAL(...)."
                        )
                    else:
                        tool_name, call_args = extracted

            if parse_error is not None or tool_name is None or call_args is None:
                budgets = {
                    "model_interactions_remaining": remaining_model_interactions,
                    "agent_llm_calls_remaining": remaining_agent_calls,
                    "token_budget_remaining": (
                        (token_budget - total_completion_tokens)
                        if token_budget != -1
                        else -1
                    ),
                }
                guidance = (
                    "FORMAT_ERROR: Your last turn did not follow the output grammar. "
                    'Follow exactly one of: FINAL(description: "...") or CALL(tool_name: {json_args}). '
                    "CALL(...) may span multiple lines but must be the final content (only whitespace after). "
                    "Use exactly one tool per turn and ensure json_args is valid JSON. "
                    f"Last line received: {last}\n"
                    f"Error: {parse_error}\n"
                    f"Budgets: {budgets}"
                )
                messages.append({"role": "user", "content": guidance})
                continue

            logger.info(f"Executing tool: {tool_name}")
            assert tool_name in tools, f"Unknown tool: {tool_name}"

            pre_cost = int(self.get_pre_tool_cost(tool_name, call_args))
            if remaining_model_interactions < pre_cost:
                budgets = {
                    "model_interactions_remaining": remaining_model_interactions,
                    "agent_llm_calls_remaining": remaining_agent_calls,
                    "token_budget_remaining": (
                        (token_budget - total_completion_tokens)
                        if token_budget != -1
                        else -1
                    ),
                }
                messages.append(
                    {
                        "role": "user",
                        "content": 'MODEL_INTERACTION_BUDGET_EXHAUSTED. Please try again with fewer model interactions or if fully exhausted, provide a FINAL(description: "..."). \n\n'
                        + _json.dumps({"budgets": budgets}),
                    }
                )
                continue

            tool_callable = tools[tool_name]
            try:
                tool_output = tool_callable(**call_args)
            except TypeError as e:
                budgets = {
                    "model_interactions_remaining": remaining_model_interactions,
                    "agent_llm_calls_remaining": remaining_agent_calls,
                    "token_budget_remaining": (
                        (token_budget - total_completion_tokens)
                        if token_budget != -1
                        else -1
                    ),
                }
                error_msg = f"TOOL_PARAMETER_ERROR: . Check that your arguments match the tool signature. Budgets: {budgets}"
                messages.append({"role": "user", "content": error_msg})
                continue

            post_cost = int(self.get_post_tool_cost(tool_name, tool_output))
            total_cost = pre_cost + post_cost
            remaining_model_interactions -= total_cost
            assert remaining_model_interactions >= 0

            budgets = {
                "model_interactions_remaining": remaining_model_interactions,
                "agent_llm_calls_remaining": remaining_agent_calls,
                "token_budget_remaining": (
                    (token_budget - total_completion_tokens)
                    if token_budget != -1
                    else -1
                ),
            }
            MEMOS = ""
            if remaining_model_interactions > 0:
                MEMOS = (
                    "You have "
                    + str(remaining_model_interactions)
                    + " model interactions remaining. USE THEM!"
                )
            elif remaining_model_interactions == 0:
                MEMOS = 'You have no model interactions remaining. You must provide a FINAL(description: "...")'
            messages.append(
                {
                    "role": "user",
                    "content": f"TOOL_RESULT({tool_name}): "
                    + _json.dumps({"data": tool_output, "budgets": budgets})
                    + "\n\n"
                    + POST_TOOL_RESULT_PROMPT
                    + "\n\n"
                    + MEMOS,
                }
            )
