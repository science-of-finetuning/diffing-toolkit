"""
Diffing pipeline for orchestrating model comparison methods.
"""

from typing import Dict, Any, List
from omegaconf import DictConfig
from loguru import logger
from pathlib import Path
from .pipeline import Pipeline
import json
from .diffing_pipeline import get_method_class
from src.utils.graders.hypothesis_grader import grade_and_save
from src.utils.agents.base_agent import BaseAgent


def save_description(description: str, stats: dict, out_dir: Path) -> None:
    (out_dir / "description.txt").write_text(description, encoding="utf-8")
    with open(out_dir / "messages.json", "w", encoding="utf-8") as f:
        json.dump(stats["messages"], f, ensure_ascii=False, indent=2)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in stats.items() if k != "messages"},
            f,
            ensure_ascii=False,
            indent=2,
        )


class EvaluationPipeline(Pipeline):
    """
    Pipeline for running evaluation methods to analyze differences between models.

    This pipeline can orchestrate multiple evaluation methods and chain them together.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="EvaluationPipeline")

        # Store evaluation configuration
        self.evaluation_cfg = cfg.diffing.evaluation

        # Initialize diffing method
        self.diffing_method = get_method_class(self.cfg.diffing.method.name)(self.cfg)

    def validate_config(self) -> None:  
        """Validate the evaluation pipeline configuration."""
        super().validate_config()

    def run_agent(self, agent: BaseAgent, model_interaction_budget: int, run_idx: int, overwrite: bool, name: str) -> Dict[str, Any]:
        run_suffix = f"_run{run_idx}"
        out_dir = (
            Path(self.diffing_method.results_dir)
            / "agent"
            / f"{name}_mi{model_interaction_budget}_{run_suffix}"
        )

        # Skip recomputation if results already exist and not overwriting
        if (out_dir / "description.txt").exists() and not overwrite:
            logger.info(f"Result exists and overwrite=False, skipping: {out_dir}")
            assert (
                out_dir.exists() and out_dir.is_dir()
            ), f"Expected agent run directory not found: {out_dir}"
            desc_fp = out_dir / "description.txt"
            assert (
                desc_fp.exists() and desc_fp.is_file()
            ), f"Missing description.txt in {out_dir}"
            description = desc_fp.read_text(encoding="utf-8")
            agent_score, _agent_text = grade_and_save(
                self.cfg, description, save_dir=out_dir
            )
            return agent_score, _agent_text, description
 
        description, stats = agent.run(self.diffing_method, model_interaction_budget=model_interaction_budget, return_stats=True)
        assert isinstance(description, str) and len(description) > 0
        assert isinstance(stats, dict) and isinstance(stats.get("messages"), list)

        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving outputs to {out_dir}")
        save_description(description, stats, out_dir)

        # Immediate grading of agent hypothesis
        agent_score, _agent_text = grade_and_save(self.cfg, description, save_dir=out_dir)
        logger.info(
            f"Graded agent (run {run_idx}) description with score={agent_score} ({_agent_text})"
        )
        logger.debug(f"Reasoning: {_agent_text}")
        return agent_score, _agent_text, description
    
    def run(self) -> Dict[str, Any]:
        """
        Run the diffing pipeline.

        Returns:
            Dictionary containing pipeline metadata and status
        """
        self.logger.info(f"Running evaluation {self.evaluation_cfg.name} for method {self.diffing_method.method_cfg.name}")

        agent_cfg = self.evaluation_cfg.agent

        # Require number of repeats
        num_repeat = int(agent_cfg.num_repeat)
        assert num_repeat >= 1

        # Common metadata for output structure
        llm_id = str(self.evaluation_cfg.agent.llm.model_id).replace("/", "_")
        name = f"{llm_id}"
        # Overwrite behavior
        overwrite = bool(self.evaluation_cfg.overwrite)
        assert isinstance(overwrite, bool)

        # Method 
        logger.info(f"Model interactions: {agent_cfg.budgets.model_interactions}")
        for model_interaction_budget in agent_cfg.budgets.model_interactions:
            # Run the agent multiple times and save each under a run-specific folder
            for run_idx in range(num_repeat):

                agent = self.diffing_method.get_agent()
                logger.info(f"Agent run {run_idx+1}/{num_repeat} (name: {agent.name + '_' + name})")
                agent_score, _agent_text, description = self.run_agent(agent, model_interaction_budget, run_idx, overwrite, agent.name + "_" + name)
                

        # Baselines
        if agent_cfg.baselines.enabled:
            for model_interaction_budget in agent_cfg.baselines.budgets.model_interactions:
                for run_idx in range(num_repeat):
                    logger.info(f"Baseline run {run_idx+1}/{num_repeat}")

                    baseline = self.diffing_method.get_baseline_agent()
                    baseline_score, _baseline_text, description = self.run_agent(baseline, model_interaction_budget, run_idx, overwrite, baseline.name + "_" + name)