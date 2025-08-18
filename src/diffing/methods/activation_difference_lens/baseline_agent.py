from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable
 

from src.utils.agents.base_agent import BaseAgent
from .agent_tools import get_steering_samples, ask_model, _abs_layers_from_rel
from .agent import POST_OVERVIEW_PROMPT
from .prompts import BASELINE_SYSTEM_PROMPT

@dataclass
class BaselineActDiffLensAgent(BaseAgent):
    cfg: Any

    def get_system_prompt(self) -> str:
        if self.cfg.budgets.model_interactions > 20:
            return BASELINE_SYSTEM_PROMPT + "\n\n" + "You have a lot of model interactions. You should start by asking a larger number of questions (e.g. 10) to the models."
        return BASELINE_SYSTEM_PROMPT

    def build_first_user_message(self, method: Any) -> str:
        # Provide ONLY unsteered generations of finetuned model without any method details
        import json as _json
        overview_cfg = self.cfg.overview
        rel_layers = list(overview_cfg.layers)
        assert len(rel_layers) >= 1
        layer = rel_layers[0]
        abs_layer = _abs_layers_from_rel(method, [layer])[0]

        # Collect unsteered generations by reusing get_steering_samples and dropping steered text
        datasets = list(overview_cfg.datasets) if getattr(overview_cfg, "datasets", None) is not None else []
        if len(datasets) == 0:
            # autodiscover datasets from results_dir similar to get_overview
            ds_set = set()
            for p in (method.results_dir).glob("layer_*/*"):
                if p.is_dir():
                    ds_set.add(p.name)
            datasets = [f"{d}" for d in ds_set]
            assert len(datasets) > 0

        max_sample_chars = int(overview_cfg.max_sample_chars)
        steering_samples_per_prompt = int(overview_cfg.steering_samples_per_prompt)

        examples_flat: List[Dict[str, str]] = []
        found = False
        for ds in datasets:
            ds_dir = method.results_dir / f"layer_{abs_layer}" / ds
            positions: List[int] = []
            steer_root = ds_dir / "steering"
            if steer_root.exists():
                for p in sorted(steer_root.glob("position_*/generations.jsonl")):
                    try:
                        pos = int(p.parent.name.split("_")[-1])
                        positions.append(pos)
                    except Exception:
                        continue
            if len(positions) == 0:
                continue
            pos0 = positions[0]
            rec = get_steering_samples(
                method,
                dataset=ds,
                layer=abs_layer,
                position=pos0,
                prompts_subset=None,
                n=steering_samples_per_prompt,
                max_chars=max_sample_chars,
            )
            for ex in rec["examples"]:
                examples_flat.append({"prompt": ex["prompt"], "generation": ex["unsteered"]})
            found = True
            break
        assert found and len(examples_flat) > 0

        header = (
            "You are given generations produced by the finetuned model on several prompts.\n"
        )
        return header + "\n" + _json.dumps({"examples": examples_flat}) + "\n\n" + POST_OVERVIEW_PROMPT

    def get_tools(self, method: Any) -> Dict[str, Callable[..., Any]]:
        def _tool_ask_model(prompts: List[str] | str):
            return ask_model(method, prompts=prompts)

        return {"ask_model": _tool_ask_model}


__all__ = ["BaselineActDiffLensAgent"]

