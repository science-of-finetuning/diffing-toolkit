from copy import deepcopy
from typing import Dict, List, Iterator, Any
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path
import uuid
import yaml

from loguru import logger

from diffing.methods.amplification.managed_data import ManagedConfig
from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.agents.blackbox_agent import BlackboxAgent
from diffing.utils.agents.diffing_method_agent import DiffingMethodAgent
from collections import defaultdict
from diffing.utils.configs import CONFIGS_DIR
from diffing.utils.prompts import read_prompts
from diffing.utils.vllm import (
    LLM,
    LoRARequest,
    SamplingParams,
    TokensPrompt,
)
from diffing.utils.model import load_model_from_config


def get_lora_int_id(server: LLM, config_str: str) -> int:
    """
    Get or allocate a unique lora_int_id for a compiled config string.

    This ensures that:
    - The same config_str always gets the same lora_int_id (cache-friendly)
    - Different config_strs always get different lora_int_ids (no collisions)
    - IDs are tied to the vLLM server lifetime (reset on server restart)

    Args:
        server: The vLLM LLM server instance
        config_str: The JSON serialization of the compiled config dict

    Returns:
        A unique integer ID for use with LoRARequest
    """
    # Initialize the counter and mapping on the server if not present
    if not hasattr(server, "_lora_id_counter"):
        server._lora_id_counter = 1  # vLLM requires lora_id > 0
    if not hasattr(server, "_config_str_to_lora_id"):
        server._config_str_to_lora_id = {}

    # Return existing ID if we've seen this config before
    if config_str in server._config_str_to_lora_id:
        lora_id = server._config_str_to_lora_id[config_str]
        # Extract config name from config_str for logging
        import json

        try:
            config_name = json.loads(config_str).get("name", "?")
        except:
            config_name = "?"
        print(
            f"DEBUG get_lora_int_id: REUSING lora_id={lora_id} for config '{config_name}'",
            flush=True,
        )
        return lora_id

    # Allocate a new ID
    lora_id = server._lora_id_counter
    server._lora_id_counter += 1
    server._config_str_to_lora_id[config_str] = lora_id
    # Extract config name from config_str for logging
    import json

    try:
        config_name = json.loads(config_str).get("name", "?")
    except:
        config_name = "?"
    print(
        f"DEBUG get_lora_int_id: ALLOCATING NEW lora_id={lora_id} for config '{config_name}'",
        flush=True,
    )
    return lora_id


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
        self._vllm_server: LLM | None = None
        self._vllm_server_config: dict | None = None

    def run(self) -> dict[str, Any]:
        """
        Collect completions from base, ft, and amplified models.

        Reads prompts from a file and generates completions for each model type
        specified in the config. Results are saved to the logs directory with
        multi-view organization, plus a consolidated generations.jsonl for agent compatibility.

        Returns:
            Context dict with request_id and paths for downstream use.
        """
        import hashlib
        import json
        from datetime import datetime

        from diffing.methods.amplification.amplification_config import (
            AmplificationConfig,
            AmplifiedAdapter,
            patch_vllm,
        )
        from diffing.methods.amplification.managed_data import GenerationLog

        patch_vllm()
        run_cfg = self.method_cfg.run

        if not self.finetuned_model_cfg.is_lora:
            raise NotImplementedError(
                "run() only supports LoRA finetuned models. "
                "Full model amplification is not yet implemented."
            )

        prompts_file = run_cfg.get("prompts_file")
        assert prompts_file is not None, "run.prompts_file must be specified"
        prompts = read_prompts(prompts_file)
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")

        request_id = str(uuid.uuid4())[:8]
        models_to_run = list(run_cfg.get("models", ["base", "ft", "amplified"]))
        logger.info(
            f"Starting run with request_id={request_id}, models={models_to_run}"
        )

        logs_dir = Path(self.cfg.diffing.results_dir) / "amplification" / "completions"
        compiled_adapters_dir = logs_dir / ".compiled_adapters"
        compiled_adapters_dir.mkdir(parents=True, exist_ok=True)

        sampling_cfg = run_cfg.sampling
        vllm_sampling = SamplingParams(
            temperature=sampling_cfg.get("temperature", 1.0),
            top_p=sampling_cfg.get("top_p", 0.9),
            max_tokens=sampling_cfg.get("max_tokens", 256),
            n=sampling_cfg.get("n", 5),
        )
        if sampling_cfg.get("seed") is not None:
            vllm_sampling.seed = sampling_cfg["seed"]

        use_chat_template = run_cfg.get("use_chat_template", True)
        adapter_id = self.finetuned_model_cfg.adapter_id

        # Collect results per prompt for generations.jsonl
        # Structure: {prompt: {"base_samples": [...], "ft_samples": [...], "amplified_samples": {...}}}
        results_by_prompt: dict[str, dict[str, Any]] = {p: {} for p in prompts}
        amplification_preset_names: list[str] = []

        for model_type in models_to_run:
            logger.info(f"Generating completions for model_type={model_type}")

            if model_type == "amplified":
                amplification_configs = list(run_cfg.get("amplification_configs", []))
                assert amplification_configs, "No amplification_configs specified"
                managed_configs = []
                for preset_path_str in amplification_configs:
                    preset_path = Path(preset_path_str)
                    assert preset_path.exists(), f"Preset not found: {preset_path}"
                    preset_data = yaml.safe_load(preset_path.read_text())
                    for adapter in preset_data.get("adapters", []):
                        adapter["variant"] = adapter_id
                    config = AmplificationConfig.from_dict(preset_data)
                    managed_configs.append(ManagedConfig(config=config))
                    amplification_preset_names.append(config.name)
            elif model_type == "ft":
                config = AmplificationConfig(
                    name="ft",
                    description="Finetuned model (no amplification)",
                    amplified_adapters=[
                        AmplifiedAdapter(
                            organism_name="custom",
                            variant=adapter_id,
                            layer_amplifications=[],
                        )
                    ],
                )
                managed_configs = [ManagedConfig(config=config)]
            else:
                managed_configs = [None]

            for managed_config in managed_configs:
                config_name = managed_config.config.name if managed_config else "base"
                logger.info(f"  Processing config: {config_name}")

                for prompt in prompts:
                    if use_chat_template:
                        prompt_tokens = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            add_generation_prompt=True,
                        )
                    else:
                        prompt_tokens = self.tokenizer.encode(prompt)

                    if managed_config is None:
                        outputs = self.vllm_server.generate(
                            prompts=[TokensPrompt(prompt_token_ids=prompt_tokens)],
                            sampling_params=vllm_sampling,
                        )
                        results = [output.text for output in outputs[0].outputs]
                        config_dict = {
                            "name": "base",
                            "description": "Base model (no LoRA)",
                        }
                    else:
                        gen_results = list(
                            self.multi_gen_request(
                                prompt=prompt_tokens,
                                amplification_configs=[managed_config],
                                sampling_params=vllm_sampling,
                                compiled_adapters_dir=compiled_adapters_dir,
                            )
                        )
                        results = gen_results[0]["results"]
                        config_dict = managed_config.config.to_dict()
                        config_dict["compiled_hash"] = managed_config.last_compiled_hash

                    # Save individual YAML log (existing behavior)
                    log = GenerationLog(
                        generation_type="run",
                        model_id=self.base_model_cfg.model_id,
                        prompt_text=prompt,
                        prompt_tokens=prompt_tokens,
                        sampling_params=dict(
                            temperature=vllm_sampling.temperature,
                            top_p=vllm_sampling.top_p,
                            max_tokens=vllm_sampling.max_tokens,
                            n=vllm_sampling.n,
                        ),
                        config=config_dict,
                        outputs=results,
                    )
                    log.save(logs_dir, request_id=request_id)

                    # Accumulate for generations.jsonl
                    if model_type == "base":
                        results_by_prompt[prompt]["base_samples"] = results
                    elif model_type == "ft":
                        results_by_prompt[prompt]["ft_samples"] = results
                    else:  # amplified
                        if "amplified_samples" not in results_by_prompt[prompt]:
                            results_by_prompt[prompt]["amplified_samples"] = {}
                        results_by_prompt[prompt]["amplified_samples"][
                            config_name
                        ] = results

        # Write generations.jsonl (one line per prompt, ADL-compatible format)
        generations_path = logs_dir / "generations.jsonl"
        sampling_params_dict = {
            "temperature": vllm_sampling.temperature,
            "top_p": vllm_sampling.top_p,
            "max_tokens": vllm_sampling.max_tokens,
            "n": vllm_sampling.n,
        }
        with generations_path.open("w", encoding="utf-8") as f:
            for prompt in prompts:
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
                record = {
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "sampling_params": sampling_params_dict,
                    **results_by_prompt[prompt],
                }
                f.write(json.dumps(record) + "\n")

        # Write run_config.json (metadata for the run)
        run_config_path = logs_dir / "run_config.json"
        run_config = {
            "request_id": request_id,
            "model_id": self.base_model_cfg.model_id,
            "organism": self.cfg.organism.name,
            "adapter_id": adapter_id,
            "amplification_presets": amplification_preset_names,
            "models_run": models_to_run,
            "num_prompts": len(prompts),
            "timestamp": datetime.now().isoformat(),
        }
        with run_config_path.open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        logger.info(f"Completed run with request_id={request_id}")
        logger.info(f"Wrote generations.jsonl and run_config.json to {logs_dir}")
        return {"request_id": request_id, "prompts": prompts, "logs_dir": str(logs_dir)}

    def visualize(self):
        """Launch the amplification dashboard."""
        from diffing.methods.amplification.amplification_dashboard import (
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
        from diffing.methods.amplification.amplification_config import (
            enable_lora_amplification_vllm_plugin,
        )

        enable_lora_amplification_vllm_plugin()

        inference_config = deepcopy(self.base_model_cfg)
        gpu_memory_utilization = self.cfg.diffing.get("gpu_memory_utilization", 0.95)
        inference_config.vllm_kwargs = vllm_kwargs or dict(
            max_num_seqs=16,
            enable_lora=True,
            max_loras=16,
            max_lora_rank=256,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return load_model_from_config(
            inference_config, use_vllm=True, ignore_cache=True
        )

    @property
    def vllm_server(self) -> LLM:
        """
        Lazy-loaded vLLM server for standalone (non-dashboard) usage.

        Note: For dashboard usage, the dashboard manages its own cached server
        and passes it explicitly to generation methods.
        """
        if self._vllm_server is None:
            self._vllm_server = self.create_vllm_server()
        return self._vllm_server

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
        server = vllm_server if vllm_server is not None else self.vllm_server

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
                lora_int_id = get_lora_int_id(server, config.last_compiled_config_str)
                lreq = LoRARequest(
                    str(compiled_path).replace("/", "__"),
                    lora_int_id,
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
