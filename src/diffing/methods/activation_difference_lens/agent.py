from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable
 

from .prompts import SYSTEM_PROMPT
from .agent_tools import (
    get_overview,
    get_logitlens_details,
    get_patchscope_details,
    get_steering_samples,
    ask_model,
    generate_steered,
)
from src.utils.agents.base_agent import BaseAgent

POST_OVERVIEW_PROMPT = """
Remember to verify your hypotheses by talking to the models AND USING ALL OR MOST MODEL INTERACTIONS MEANING ASK MULTIPLE QUESTIONS.
ASK MULTIPLE QUESTIONS USING THE ask_model TOOL. DON'T RESPOND WITH FINAL UNTIL YOU HAVE CONFIRMED YOUR HYPOTHESES. 

If you don't have many model interactions (i.e. < 10), ASK QUESTIONS ONE BY ONE AND REASON INBETWEEN.
"""

@dataclass
class ActDiffLensAgent(BaseAgent):
    cfg: Any

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_first_user_message(self, method: Any) -> str:
        import json as _json
        overview_cfg = self.cfg.overview
        overview_payload = get_overview(method, {
            "datasets": list(overview_cfg.datasets) if getattr(overview_cfg, "datasets", None) is not None else [],
            "layers": list(overview_cfg.layers),
            "top_k_tokens": int(overview_cfg.top_k_tokens),
            "steering_samples_per_prompt": int(overview_cfg.steering_samples_per_prompt),
            "max_sample_chars": int(overview_cfg.max_sample_chars),
        })
        return "OVERVIEW:" + "\n" + _json.dumps(overview_payload) + "\n\n" + POST_OVERVIEW_PROMPT

    def get_tools(self, method: Any) -> Dict[str, Callable[..., Any]]:
        drilldown_cfg = self.cfg.drilldown
        steer_cfg = self.cfg.generate_steered

        def _tool_get_logitlens_details(dataset: str, layer: float | int, positions: List[int], k: int) -> Dict[str, Any]:
            return get_logitlens_details(method, dataset=dataset, layer=layer, positions=positions, k=k)

        def _tool_get_patchscope_details(dataset: str, layer: float | int, positions: List[int], k: int) -> Dict[str, Any]:
            return get_patchscope_details(method, dataset=dataset, layer=layer, positions=positions, k=k)

        def _tool_get_steering_samples(dataset: str, layer: float | int, position: int, prompts_subset: List[str] | None, n: int) -> Dict[str, Any]:
            return get_steering_samples(
                method,
                dataset=dataset,
                layer=layer,
                position=position,
                prompts_subset=list(prompts_subset) if prompts_subset is not None else None,
                n=int(n),
                max_chars=int(drilldown_cfg.max_sample_chars),
            )

        def _tool_ask_model(prompts: List[str] | str):
            return ask_model(method, prompts=prompts)

        def _tool_generate_steered(dataset: str, layer: float | int, position: int, prompts: List[str], n: int) -> Dict[str, List[str]]:
            texts = generate_steered(
                method,
                dataset=dataset,
                layer=layer,
                position=position,
                prompts=list(prompts),
                n=int(n),
                max_new_tokens=int(steer_cfg.max_new_tokens),
                temperature=float(steer_cfg.temperature),
                do_sample=bool(steer_cfg.do_sample),
            )
            return {"texts": texts}

        return {
            "get_logitlens_details": _tool_get_logitlens_details,
            "get_patchscope_details": _tool_get_patchscope_details,
            "get_steering_samples": _tool_get_steering_samples,
            "ask_model": _tool_ask_model,
            "generate_steered": _tool_generate_steered,
        }

    def get_pre_tool_cost(self, tool_name: str, call_args: Dict[str, Any]) -> int:
        if tool_name == "ask_model":
            assert "prompts" in call_args
            return len(list(call_args["prompts"]))
        if tool_name == "generate_steered":
            # Match original behavior: require at least 1 interaction before generation
            return 1
        return 0

    def get_post_tool_cost(self, tool_name: str, tool_output: Any) -> int:
        if tool_name == "generate_steered":
            assert isinstance(tool_output, dict) and "texts" in tool_output
            n_texts = int(len(list(tool_output["texts"])))
            assert n_texts >= 1
            # We pre-charged 1, charge the remainder post-call
            return n_texts - 1
        return 0


__all__ = ["ActDiffLensAgent"]

