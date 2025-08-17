from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from loguru import logger

from .prompts import SYSTEM_PROMPT
from .agent_tools import (
    get_overview,
    get_logitlens_details,
    get_patchscope_details,
    get_steering_samples,
    ask_model,
    generate_steered,
)
from src.utils.agents.llm import AgentLLM

POST_OVERVIEW_PROMPT = """
Remember to verify your hypotheses by talking to the models AND ALL OR MOST MODEL INTERACTIONS MEANING ASK MULTIPLE QUESTIONS.
ASK MULTIPLE QUESTIONS USING THE ask_model TOOL. DON'T RESPOND WITH FINAL UNTIL YOU HAVE CONFIRMED YOUR HYPOTHESES. 
DON'T USE UP ALL THE MODEL INTERACTIONS IN THE FIRST TURN AS YOU MAY WANT TO ASK MORE QUESTIONS LATER.
"""

def _build_system_messages(system_prompt: str, hints: str, model_interactions_remaining: int) -> List[dict]:
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
    assert total_completion_tokens <= token_budget, (
        f"Agent LLM token budget exceeded: {total_completion_tokens} > {token_budget}"
    )


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
    key = "description:"
    idx = inside.find(key)
    assert idx != -1, "FINAL(...) must contain description:"

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


@dataclass
class ActDiffLensAgent:
    cfg: Any

    def run(self, method: Any, return_messages: bool = False) -> str | tuple[str, List[dict]]:
        logger.info("Starting ActDiffLensAgent.run()")
        llm_cfg = self.cfg.llm
        budgets_cfg = self.cfg.budgets
        overview_cfg = self.cfg.overview
        drilldown_cfg = self.cfg.drilldown
        steer_cfg = self.cfg.generate_steered

        agent = AgentLLM(
            model_id=str(llm_cfg.model_id),
            base_url=str(llm_cfg.base_url),
            api_key_path=str(llm_cfg.api_key_path),
            temperature=float(llm_cfg.temperature),
            max_tokens_per_call=int(llm_cfg.max_tokens_per_call),
        )

        remaining_agent_calls = int(budgets_cfg.agent_llm_calls)
        remaining_model_interactions = int(budgets_cfg.model_interactions)
        token_budget = int(budgets_cfg.token_budget_generated)
        total_completion_tokens = 0

        messages: List[dict] = []
        messages.extend(_build_system_messages(SYSTEM_PROMPT, str(getattr(self.cfg, "hints", "")), remaining_model_interactions))
        logger.debug(f"system messages: {messages[0]['content']}")
        # First turn: compute overview and include as payload in the first user message
        import json as _json
        overview_payload = get_overview(method, {
            "datasets": list(overview_cfg.datasets) if getattr(overview_cfg, "datasets", None) is not None else [],
            "layers": list(overview_cfg.layers),
            "top_k_tokens": int(overview_cfg.top_k_tokens),
            "steering_samples_per_prompt": int(overview_cfg.steering_samples_per_prompt),
            "max_sample_chars": int(overview_cfg.max_sample_chars),
        })
        user_content = (
            "OVERVIEW:" + "\n" + _json.dumps(overview_payload) + "\n\n" + POST_OVERVIEW_PROMPT
        )
        messages.append({"role": "user", "content": user_content})

        while True:
            assert remaining_agent_calls > 0, "Agent LLM call budget exhausted"
            logger.info("Agent LLM: thinking...")
            logger.debug(f"last message: {messages[-1]['content']}")
            result = agent.chat(messages)
            content = result["content"]
            usage = result["usage"]
            completion_tokens = int(usage.get("completion_tokens", 0))
            logger.info(f"Agent LLM: {completion_tokens} completion tokens")
            logger.debug(f"Agent LLM: {content}")
            total_completion_tokens += completion_tokens
            _enforce_token_budget(total_completion_tokens, token_budget)

            remaining_agent_calls -= 1

            text = content or ""
            # Record assistant content for continuity
            messages.append({"role": "assistant", "content": text})
            # Extract last non-empty line for payload parsing, allowing preceding THOUGHTS
            lines = [ln.strip() for ln in (text.splitlines() if isinstance(text, str) else [])]
            lines = [ln for ln in lines if len(ln) > 0]
            last = lines[-1] if len(lines) > 0 else ""
            # Handle FINAL(...) possibly spanning multiple lines
            final_desc = _extract_final_description(text)
            if final_desc is not None:
                return (final_desc, messages) if return_messages else final_desc

            assert last.startswith("CALL("), "Agent must emit CALL(...) or FINAL(...) as last line"
            # Parse tool name and json args
            lpar = last.find("(")
            rpar = last.rfind(")")
            assert lpar != -1 and rpar != -1 and rpar > lpar
            inside = last[lpar + 1 : rpar]
            tool_name, json_part = inside.split(":", 1)
            tool_name = tool_name.strip()
            json_part = json_part.strip()
            call_args: Dict[str, Any] = _json.loads(json_part)

            # Execute tool
            logger.info(f"Executing tool: {tool_name}")
            tool_output: Dict[str, Any] | str
            if tool_name == "get_logitlens_details":
                tool_output = get_logitlens_details(
                    method,
                    dataset=str(call_args["dataset"]),
                    layer=call_args["layer"],
                    positions=list(call_args["positions"]),
                    k=int(call_args["k"]),
                )
            elif tool_name == "get_patchscope_details":
                tool_output = get_patchscope_details(
                    method,
                    dataset=str(call_args["dataset"]),
                    layer=call_args["layer"],
                    positions=list(call_args["positions"]),
                    k=int(call_args["k"]),
                )
            elif tool_name == "get_steering_samples":
                tool_output = get_steering_samples(
                    method,
                    dataset=str(call_args["dataset"]),
                    layer=call_args["layer"],
                    position=int(call_args["position"]),
                    prompts_subset=list(call_args["prompts_subset"]) if call_args.get("prompts_subset") is not None else None,
                    n=int(call_args["n"]),
                    max_chars=int(drilldown_cfg.max_sample_chars),
                )
            elif tool_name == "ask_model":
                _model_arg = str(call_args["model"])  # "base" | "finetuned" | "both"
                has_single = "prompt" in call_args
                has_multi = "prompts" in call_args
                assert has_single ^ has_multi, "ask_model expects exactly one of 'prompt' or 'prompts'"
                # Budget: 1 per model per prompt
                num_prompts = 1 if has_single else len(list(call_args["prompts"]))
                cost_per_prompt = 2 if _model_arg == "both" else 1
                total_cost = num_prompts * cost_per_prompt
                assert remaining_model_interactions >= total_cost, "Model interaction budget exhausted"
                if has_single:
                    tool_output = ask_model(
                        method,
                        model=_model_arg,
                        prompt=str(call_args["prompt"]),
                    )
                else:
                    tool_output = ask_model(
                        method,
                        model=_model_arg,
                        prompts=list(call_args["prompts"]),
                    )
                remaining_model_interactions -= total_cost
            elif tool_name == "generate_steered":
                assert remaining_model_interactions > 0, "Model interaction budget exhausted"
                texts = generate_steered(
                    method,
                    dataset=str(call_args["dataset"]),
                    layer=call_args["layer"],
                    position=int(call_args["position"]),
                    prompts=list(call_args["prompts"]),
                    n=int(call_args["n"]),
                    max_new_tokens=int(steer_cfg.max_new_tokens),
                    temperature=float(steer_cfg.temperature),
                    do_sample=bool(steer_cfg.do_sample),
                )
                remaining_model_interactions -= len(texts)
                tool_output = {"texts": texts}
            else:
                assert False, f"Unknown tool: {tool_name}"

            # Package tool result and budgets back to the agent
            budgets = {
                "model_interactions_remaining": remaining_model_interactions,
                "agent_llm_calls_remaining": remaining_agent_calls,
                "token_budget_remaining": (token_budget - total_completion_tokens) if token_budget != -1 else -1,
            }
            import json as _json
            messages.append({
                "role": "user",
                "content": f"TOOL_RESULT({tool_name}): " + _json.dumps({"data": tool_output, "budgets": budgets}),
            })


__all__ = ["ActDiffLensAgent"]

