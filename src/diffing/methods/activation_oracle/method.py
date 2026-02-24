# Adapted from https://github.com/adamkarvonen/sae_introspect/blob/main/paper_demo/em_demo.py
from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.configs import DictConfig
from pathlib import Path
from typing import Dict
import json
from peft import LoraConfig
from dataclasses import asdict
from loguru import logger
from omegaconf import OmegaConf

from diffing.utils.agents import DiffingMethodAgent
from diffing.utils.model import load_model_from_config
from .verbalizer import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    run_verbalizer,
    sanitize_lora_name,
)
from .agent import ActivationOracleAgent


class ActivationOracleMethod(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.results_dir = Path(cfg.diffing.results_dir) / "activation_oracle"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self):
        pass

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available results for this method.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        return {}

    def get_agent(self) -> DiffingMethodAgent:
        """Get the agent for the method."""
        return ActivationOracleAgent(cfg=self.cfg)

    def extra_agent_relevant_cfg(self) -> dict:
        """Include verbalizer config in the hash since it affects agent results."""
        return {
            "verbalizer_eval": OmegaConf.to_container(
                self.method_cfg.verbalizer_eval, resolve=True
            ),
            "context_prompts": OmegaConf.to_container(
                self.method_cfg.context_prompts, resolve=True
            ),
            "verbalizer_prompts": OmegaConf.to_container(
                self.method_cfg.verbalizer_prompts, resolve=True
            ),
        }

    def _results_file(self) -> Path:
        return (
            self.results_dir
            / f"{self._get_verbalizer_lora_path().split('/')[-1].replace('/', '_').replace('.', '_')}{'_' if self.agent_cfg_hash else ''}{self.agent_cfg_hash}.json"
        )

    def _load_results(self) -> Dict[str, Dict[str, str]]:
        assert (
            self._results_file().exists()
        ), f"Results file does not exist: {self._results_file()}"
        with self._results_file().open("r") as f:
            return json.load(f)

    def _get_verbalizer_lora_path(self) -> str:
        path = getattr(self.method_cfg.verbalizer_models, self.base_model_cfg.name)
        assert (
            path is not None and path != ""
        ), f"Verbalizer model for {self.base_model_cfg.name} not found"
        return path

    def run(self):
        # TODO: Support full finetunes in activation oracle (currently only LoRA adapters supported)
        if not self.finetuned_model_cfg.is_lora:
            raise NotImplementedError(
                f"ActivationOracleMethod only supports LoRA adapters, not full finetunes. "
                f"Got finetuned model: {self.finetuned_model_cfg.model_id}"
            )

        # Layers for activation collection and injection
        model_name = self.base_model_cfg.model_id

        # Skip if results exist and overwrite is disabled
        results_path = self._results_file()
        if results_path.exists() and (not bool(self.method_cfg.overwrite)):
            logger.info(
                f"Results already exist at {results_path}; overwrite=false; skipping run."
            )
            return

        eval_overrides: dict = {}
        if "verbalizer_eval" in self.method_cfg:
            eval_overrides = OmegaConf.to_container(
                self.method_cfg.verbalizer_eval, resolve=True
            )
            assert isinstance(
                eval_overrides, dict
            ), "verbalizer_eval must resolve to a dict"
        config = VerbalizerEvalConfig(
            model_name=model_name,
            num_layers=self.base_model.num_layers,
            **eval_overrides,
        )

        # ========================================
        # PROMPT TYPES AND QUESTIONS
        # ========================================

        # IMPORTANT: Context prompts: we send these to the target model and collect activations
        context_prompts: list[str] = list(self.method_cfg.context_prompts)
        assert len(context_prompts) > 0, "context_prompts cannot be empty"

        # IMPORTANT: Verbalizer prompts: these are the questions / prompts we send to the verbalizer model, along with context prompt activations
        verbalizer_prompts: list[str] = list(self.method_cfg.verbalizer_prompts)
        assert len(verbalizer_prompts) > 0, "verbalizer_prompts cannot be empty"
        prefix = self.method_cfg.prefix
        for i in range(len(verbalizer_prompts)):
            verbalizer_prompts[i] = prefix + verbalizer_prompts[i]

        # Load tokenizer and model with both adapters
        tokenizer = self.tokenizer

        verbalizer_lora_id = self._get_verbalizer_lora_path()
        target_lora_id = self.finetuned_model_cfg.model_id

        # Load model with both adapters (verbalizer + target) to avoid mutating cached models
        model = load_model_from_config(
            self.base_model_cfg,  # todo: change this to the finetuned model when adding support for full finetunes
            extra_adapter_ids=[verbalizer_lora_id, target_lora_id],
        )
        if not model.dispatched:
            model.dispatch()
        model.eval()

        # Add dummy adapter so peft_config exists and we can use the consistent PeftModel API
        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")

        # Get sanitized adapter names for switching
        verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)
        target_lora_name = sanitize_lora_name(target_lora_id)

        logger.info(
            f"Running verbalizer eval for verbalizer: {verbalizer_lora_name}, target: {target_lora_name}"
        )

        # Build context prompts with ground truth
        verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
        for verbalizer_prompt in verbalizer_prompts:
            for context_prompt in context_prompts:
                formatted_prompt = [
                    {"role": "user", "content": context_prompt},
                ]
                context_prompt_info = VerbalizerInputInfo(
                    context_prompt=formatted_prompt,
                    ground_truth=target_lora_name,
                    verbalizer_prompt=verbalizer_prompt,
                )
                verbalizer_prompt_infos.append(context_prompt_info)

        results = run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=verbalizer_prompt_infos,
            verbalizer_lora_path=verbalizer_lora_name,
            target_lora_path=target_lora_name,
            config=config,
            device=model.device,
        )

        # Optionally save to JSON

        final_verbalizer_results = {
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }
        with self._results_file().open("w") as f:
            json.dump(final_verbalizer_results, f, indent=2)
