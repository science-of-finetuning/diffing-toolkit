from copy import deepcopy
from typing import Dict, List, Iterator
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path

from src.diffing.methods.amplification.dashboard_state import ManagedConfig
from src.diffing.methods.diffing_method import DiffingMethod
from src.utils.agents.blackbox_agent import BlackboxAgent
from src.utils.agents.diffing_method_agent import DiffingMethodAgent
from collections import defaultdict
from src.utils.configs import CONFIGS_DIR
from src.utils.vllm import (
    LLM,
    ensure_vllm,
    LoRARequest,
    SamplingParams,
    TokensPrompt,
)
from src.utils.model import load_model_from_config


@dataclass
class WeightDifferenceAmplificationConfig:
    """
    Configuration for weight difference amplification.
    """

    default_amplification_factor: float
    amplification_factors: Dict[str, float]


class WeightDifferenceAmplification(DiffingMethod):
    """
    Amplify the weights difference between the base and finetuned models according to an amplifcation configuration. Supports only LoRAs for now.
    """

    def __init__(self, cfg: DictConfig, enable_chat: bool = False):
        super().__init__(cfg, enable_chat)
        self.default_tokenizer = "base"
        self._multi_lora_vllm_server: LLM | None = None
        self._vllm_server_config: dict | None = None

    def run(self):
        raise NotImplementedError("No need to run this method")

    def visualize(self):
        """Launch the amplification dashboard."""
        from src.diffing.methods.amplification.amplification_dashboard import (
            AmplificationDashboard,
        )

        dashboard = AmplificationDashboard(self)
        dashboard.display()

    @property
    def finetuned_model(self):
        raise ValueError("Finetuned model is not available for this method")

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available results for this method.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = defaultdict(dict)

        all_model_names = [
            model_cfg.stem for model_cfg in (CONFIGS_DIR / "model").glob("*.yaml")
        ]
        all_organism_names = [
            organism_cfg.stem
            for organism_cfg in (CONFIGS_DIR / "organism").glob("*.yaml")
        ]
        for model_name in all_model_names:
            for organism_name in all_organism_names:
                results[model_name][organism_name] = "."

        return results

    def get_agent(self) -> DiffingMethodAgent:
        """Get the agent for the method."""
        raise NotImplementedError

    def compute_vllm_kwargs(
        self,
        active_configs: List[ManagedConfig],
        base_vllm_kwargs: dict | None = None,
    ) -> dict:
        """
        Compute vLLM kwargs based on active amplification configurations.

        Args:
            active_configs: List of ManagedConfig objects that are active
            base_vllm_kwargs: Base vLLM kwargs to extend (defaults to sensible LoRA defaults)

        Returns:
            Dict of vLLM kwargs with max_num_seqs, max_loras, max_lora_rank set appropriately
        """
        result = (base_vllm_kwargs or {}) | dict(
            max_num_seqs=16,
            enable_lora=True,
            max_loras=16,
            max_lora_rank=64,
        )

        num_configs = len(active_configs)
        result["max_num_seqs"] = max(((num_configs + 7) // 8) * 8, 8)
        result["max_loras"] = max(num_configs, 16)

        all_adapter_ids = set()
        base_model_name = self.base_model_cfg.name
        for mc in active_configs:
            for adapter in mc.config.amplified_adapters:
                all_adapter_ids.add(adapter.adapter_id(base_model_name))

        if all_adapter_ids:
            ranks = [self.get_adapter_rank(aid) for aid in all_adapter_ids]
            result["max_lora_rank"] = max(ranks) * 2

        return result

    def create_vllm_server(self, vllm_kwargs: dict | None = None) -> LLM:
        """
        Create a new vLLM server with the given kwargs.

        Args:
            vllm_kwargs: Optional vLLM kwargs to use (defaults to LoRA-enabled config)

        Returns:
            LLM instance
        """
        inference_config = deepcopy(self.base_model_cfg)
        inference_config.vllm_kwargs = vllm_kwargs or dict(
            max_num_seqs=16,
            enable_lora=True,
            max_loras=16,
            max_lora_rank=64,
        )
        return load_model_from_config(
            inference_config, use_vllm=True, ignore_cache=True
        )

    @property
    @ensure_vllm
    def multi_lora_vllm_server(self) -> LLM:
        """
        Lazy-loaded vLLM server for standalone (non-dashboard) usage.

        Note: For dashboard usage, the dashboard manages its own cached server
        and passes it explicitly to generation methods.
        """
        if self._multi_lora_vllm_server is None:
            self._multi_lora_vllm_server = self.create_vllm_server()
        return self._multi_lora_vllm_server

    def compile_config(
        self,
        config: ManagedConfig,
        output_dir: Path,
    ) -> Path | None:
        """
        Compile an amplification config to a LoRA adapter.

        Args:
            config: ManagedConfig to compile
            output_dir: Directory to write compiled adapter

        Returns:
            Path to compiled adapter, or None if config has no adapters
        """
        return config.compile(
            output_dir,
            base_model_name=self.base_model_cfg.name,
            base_model=self.base_model,
        )

    def multi_gen_request(
        self,
        prompt: list[int] | list[list[int]],
        amplification_configs: List[ManagedConfig] | ManagedConfig,
        sampling_params: SamplingParams | dict,
        compiled_adapters_dir: Path,
        vllm_server: LLM | None = None,
    ) -> Iterator[dict]:
        """
        Generate text with multiple amplification configurations.

        Supports both single prompt and batched prompts. When batched, all prompts
        are processed in a single vLLM call per config for efficiency.

        Args:
            prompt: Input prompt as token IDs (single) or list of token ID lists (batch)
            amplification_configs: List of ManagedConfig objects to generate with
            sampling_params: vLLM SamplingParams or dict with sampling settings
            compiled_adapters_dir: Directory to store compiled adapters
            vllm_server: Optional vLLM server to use (defaults to lazy-loaded server)

        Yields:
            Dict with keys: config, compiled_path, results, output_tokens
            - Single prompt: results/output_tokens are 1D lists (one per sample)
            - Batched prompts: results/output_tokens are 2D lists [prompt_idx][sample_idx]
        """
        server = vllm_server if vllm_server is not None else self.multi_lora_vllm_server

        if not isinstance(amplification_configs, list):
            amplification_configs = [amplification_configs]

        # Normalize prompt to list of prompts, track if batched
        is_batched = len(prompt) > 0 and isinstance(prompt[0], list)
        prompts = prompt if is_batched else [prompt]

        if isinstance(sampling_params, dict):
            vllm_sampling_params = SamplingParams(
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 0.9),
                max_tokens=sampling_params.get("max_tokens", 100),
                n=sampling_params.get("n", 1),
            )
        else:
            vllm_sampling_params = sampling_params

        for config in amplification_configs:
            compiled_path = self.compile_config(config, compiled_adapters_dir)
            if compiled_path is None:
                lreq = None
            else:
                lreq = LoRARequest(
                    config.config.name,
                    config.lora_int_id,
                    str(compiled_path),
                )

            # Batch all prompts in single vLLM call
            outputs = server.generate(
                prompts=[TokensPrompt(prompt_token_ids=p) for p in prompts],
                sampling_params=vllm_sampling_params,
                lora_request=lreq,
            )

            if is_batched:
                # 2D results: [prompt_idx][sample_idx]
                results = [[output.text for output in req.outputs] for req in outputs]
                output_tokens = [
                    [list(output.token_ids) for output in req.outputs]
                    for req in outputs
                ]
            else:
                # 1D results (backward compatible): [sample_idx]
                all_completions = outputs[0].outputs
                results = [output.text for output in all_completions]
                output_tokens = [list(output.token_ids) for output in all_completions]

            yield {
                "config": config,
                "compiled_path": compiled_path,
                "results": results,
                "output_tokens": output_tokens,
            }
