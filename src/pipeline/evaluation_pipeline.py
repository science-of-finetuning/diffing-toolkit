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
        self.diffing_method = get_method_class(self.diffing_cfg.method.name)(self.cfg)

    def validate_config(self) -> None:  
        """Validate the evaluation pipeline configuration."""
        super().validate_config()

    def run_agent(self, agent: BaseAgent, budget_model_interactions: int, run_idx: int, overwrite: bool, name: str) -> Dict[str, Any]:
        run_suffix = f"_run{run_idx}"
        out_dir = (
            Path(self.diffing_method.results_dir)
            / "agent"
            / f"{name}_mi{budget_model_interactions}_{run_suffix}"
            / "ours"
        )

        # Skip recomputation if results already exist and not overwriting
        if out_dir.exists() and not overwrite:
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
 
        description, stats = agent.run(self.diffing_method, budget_model_interactions=budget_model_interactions, return_stats=True)
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
        self.logger.info(f"Running evaluation {self.evaluation_cfg.name} for method {self.diffing_method.name}")

        agent_cfg = self.evaluation_cfg.agent

        # Require number of repeats
        num_repeat = int(agent_cfg.num_repeat)
        assert num_repeat >= 1

        # Common metadata for output structure
        organism = str(self.cfg.organism.name)
        model = str(self.cfg.model.name)
        llm_id = str(self.evaluation_cfg.agent.llm.model_id).replace("/", "_")
        name = f"{organism}_{model}_{llm_id}"
        # Overwrite behavior
        overwrite = bool(self.evaluation_cfg.overwrite)
        assert isinstance(overwrite, bool)

        # Method 
        for budget_model_interactions in agent_cfg.budgets.model_interactions:
            # Run the agent multiple times and save each under a run-specific folder
            for run_idx in range(num_repeat):
                logger.info(f"Agent run {run_idx+1}/{num_repeat}")

                agent = self.diffing_method.get_agent()
                agent_score, _agent_text, description = self.run_agent(agent, budget_model_interactions, run_idx, overwrite, agent.name + "_" + name)
                

        # Baselines
        for baseline_cfg in agent_cfg.baselines:
            for budget_model_interactions in baseline_cfg.budgets.model_interactions:
                for run_idx in range(num_repeat):
                    logger.info(f"Baseline run {run_idx+1}/{num_repeat}")

                    baseline = self.diffing_method.get_baseline()
                    baseline_score, _baseline_text, description = self.run_agent(baseline, budget_model_interactions, run_idx, overwrite, baseline.name + "_" + name)