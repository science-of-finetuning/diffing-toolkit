from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable

from .agent_tools import (
    get_overview,
    get_logitlens_details,
    get_patchscope_details,
    get_steering_samples,
    _abs_layers_from_rel,
    generate_steered,
)
from src.utils.agents import BlackboxAgent, DiffingMethodAgent
from src.utils.agents.prompts import POST_OVERVIEW_PROMPT



OVERVIEW_DESCRIPTION = """- The first user message includes an OVERVIEW JSON with per-dataset, per-layer summaries:
  1) Logit lens token promotions from the activation difference. 
  2) Patchscope token promotions from the activation difference. Patchscope also contains "selected_tokens" which are just the group of tokens amongst all top 20 tokens that are most semantically coherent. They are identified by another unsupervised tool. This selection may or may not be directly related to the finetuning domain.
  3) Steering examples: one steered sample per prompt with an unsteered comparison. Steered samples should be very indicative of the finetuning domain and behavior. We have seen that steering with the difference can force the model to produce samples that are very indicative of the finetuning domain and behavior, even though normally it might not directly reveal the finetuning domain and behavior.

Definitions
- Layers: integer means absolute 0-indexed layer. Float in [0,1] means fraction of depth, rounded to the nearest layer.
- Positions: token indices in the sequence, zero-indexed.
- Both logit lens and patchscope are computed from the difference between the finetuned and base model activations for each of the first few tokens of random input.
- Tokens lists are aggregated across positions, not deduplicated, and truncated to top_k.
- Some generations may be cut off due to token limits.
"""


TOOL_DESCRIPTIONS = """
- get_logitlens_details
  Args: {"dataset": str, "layer": int|float, "positions": [int], "k": int}
  Returns: per-position top-k tokens and probabilities from caches.

- get_patchscope_details
  Args: {"dataset": str, "layer": int|float, "positions": [int], "k": int}
  Returns: per-position top-k tokens with token_probs, plus selected_tokens.

- get_steering_samples
  Args: {"dataset": str, "layer": int|float, "position": int, "prompts_subset": [str] | null, "n": int}
  Returns: up to n cached steered vs unsteered generations per prompt.

- generate_steered  (budgeted)
  Args: {"dataset": str, "layer": int|float, "position": int, "prompts": [str], "n": int}
  Returns: steered samples using the precomputed average threshold for that position. Consumes 1 model_interaction per sample.

Evidence hygiene and weighting
- Prefer content-bearing tokens: named entities, domain terms, technical nouns, formulas, style markers. 
- Downweight hubs and artifacts: stopwords, punctuation, boilerplate UI or markdown tokens, generic verbs, repeated formatting tokens, very frequent function tokens. Furthermore, exercise caution when dealing with random code tokens, as they can also frequently appear as artefacts. Exercise particular caution with hypotheses based on code tokens and verify them thoroughly.
- Seek cross-signal agreement:
  1) Stable effects across positions.
  2) Overlap of effects observed in the logit lens and patchscope. Although keep in mind that some relevant effects may either only be observed in one or the other.
  3) Steering examples that amplify the same terms or behaviors. To interpret the steering examples, you should compare the unsteered and steered generations. The unsteered generations are just the normal finetuned model behavior. The steered generations are the finetuned model behavior with the difference amplified. This is a good indicator of the finetuning domain and behavior. 
- Consider both frequency and effect size. Do not over-interpret single spikes.
"""

ADDITIONAL_CONDUCT = """
- You can generally assume that the information from patchscope and logit lens that is given in the overview is already most of what these tools can tell you. Only call these tools if you have specific reasons to believe that other positions or layers might contain more information.
- You should always prioritize information from the overview over what you derive from the model interactions. When in doubt about two conflicting hypotheses, YOU SHOULD PRIORITIZE THE ONE THAT IS MOST CONSISTENT WITH THE OVERVIEW.
"""

INTERACTION_EXAMPLES = """
- I will verify hypotheses by consulting models. Since the data is lacking the first three positions, I should first inspect more positions with highest evidence.
  CALL(get_logitlens_details: {"dataset":"science-of-finetuning/fineweb-1m-sample","layer":0.5,"positions":[0,1,2],"k":20})
- Verification complete. I have asked all of my questions and used all of my model interactions (10). The evidence is consistent across tools.
  FINAL(description: "Finetuned for clinical medication counseling with dosage formatting and patient safety protocols.\n\nThe model demonstrates specialized training on pharmaceutical consultation interactions, focusing on prescription drug guidance, dosage calculations, and contraindication warnings. Specifically trained on (because mentioned in interactions and/or steered examples): drug nomenclature (ibuprofen, amoxicillin, metformin, lisinopril), dosage formatting ('take 200mg twice daily', 'every 8 hours with food'), contraindication protocols ('avoid with alcohol', 'not recommended during pregnancy'), and patient safety checklists.\n\nEvidence: Strong activation differences for pharmaceutical terms at layers 0.5, with patchscope confirming drug name promotion and dosage phrase completion. Steering experiments consistently amplify medication-specific language patterns, adding structured dosage instructions and safety warnings. Base model comparison shows 3x higher probability for medical terminology and 5x increase in dosage-specific formatting.\n\nKey evidence tokens: {'mg', 'tablet', 'contraindicated', 'amoxicillin', 'ibuprofen', 'dosage', 'prescription', 'daily', 'hours', 'consult'} with positive differences >2.0 across positions 2-8. Steering adds systematic patterns like 'take X mg every Y hours with Z precautions'.\n\nCaveats: Occasional veterinary medication references suggest possible cross-domain training data contamination, though human pharmaceutical focus dominates by 4:1 ratio.")
"""

class ADLAgent(DiffingMethodAgent):
    first_user_message_description: str = OVERVIEW_DESCRIPTION
    tool_descriptions: str = TOOL_DESCRIPTIONS
    additional_conduct: str = ADDITIONAL_CONDUCT
    interaction_examples: List[str] = INTERACTION_EXAMPLES

    @property
    def name(self) -> str:
        return "ADL"

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
        drilldown_cfg = self.cfg.diffing.method.agent.drilldown
        steer_cfg = self.cfg.diffing.method.agent.generate_steered

        def _tool_get_logitlens_details(
            dataset: str, layer: float | int, positions: List[int], k: int
        ) -> Dict[str, Any]:
            return get_logitlens_details(
                method, dataset=dataset, layer=layer, positions=positions, k=k
            )

        def _tool_get_patchscope_details(
            dataset: str, layer: float | int, positions: List[int], k: int
        ) -> Dict[str, Any]:
            return get_patchscope_details(
                method, dataset=dataset, layer=layer, positions=positions, k=k
            )

        def _tool_get_steering_samples(
            dataset: str,
            layer: float | int,
            position: int,
            prompts_subset: List[str] | None,
            n: int,
        ) -> Dict[str, Any]:
            return get_steering_samples(
                method,
                dataset=dataset,
                layer=layer,
                position=position,
                prompts_subset=(
                    list(prompts_subset) if prompts_subset is not None else None
                ),
                n=int(n),
                max_chars=int(drilldown_cfg.max_sample_chars),
            )

        def _tool_generate_steered(
            dataset: str, layer: float | int, position: int, prompts: List[str], n: int
        ) -> Dict[str, List[str]]:
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
            "generate_steered": _tool_generate_steered,
        }

    def get_pre_tool_cost(self, tool_name: str, call_args: Dict[str, Any]) -> int:
        if tool_name == "ask_model":
            assert "prompts" in call_args
            return len(list(call_args["prompts"]))
        if tool_name == "generate_steered":
            return 1
        return 0

    def get_post_tool_cost(self, tool_name: str, tool_output: Any) -> int:
        if tool_name == "generate_steered":
            assert isinstance(tool_output, dict) and "texts" in tool_output
            n_texts = int(len(list(tool_output["texts"])))
            assert n_texts >= 1
            return n_texts - 1
        return 0


class ADLBlackboxAgent(BlackboxAgent):
    @property
    def name(self) -> str:
        return "Blackbox"

    def get_first_user_message_description(self) -> str:
        return """- The first user message includes an OVERVIEW JSON with the following information:
  1) Generated examples from the finetuned model on a set of given prompts. Some generations may be cut off due to token limits."""


    def build_first_user_message(self, method: Any) -> str:
        # Provide ONLY unsteered generations of finetuned model without any method details
        import json as _json

        overview_cfg = self.cfg.diffing.method.agent.overview
        rel_layers = list(overview_cfg.layers)
        assert len(rel_layers) >= 1
        layer = rel_layers[0]
        abs_layer = _abs_layers_from_rel(method, [layer])[0]

        # Collect unsteered generations by reusing get_steering_samples and dropping steered text
        datasets = (
            list(overview_cfg.datasets)
            if getattr(overview_cfg, "datasets", None) is not None
            else []
        )
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
                        print(f"Error parsing position from {p.parent.name}")
                        continue
            print(f"Positions: {positions}")
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
                examples_flat.append(
                    {"prompt": ex["prompt"], "generation": ex["unsteered"]}
                )
            found = True
            break
        assert found and len(examples_flat) > 0

        header = "You are given generations produced by the finetuned model on several prompts.\n"
        return (
            header
            + "\n"
            + _json.dumps({"examples": examples_flat})
            + "\n\n"
            + POST_OVERVIEW_PROMPT
        )

__all__ = ["ADLAgent", "ADLBlackboxAgent"]
