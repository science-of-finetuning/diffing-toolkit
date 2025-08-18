#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

# Ensure project root is on sys.path so that `src.*` imports work when executed from scripts/
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.diffing.methods.activation_difference_lens.act_diff_lens import ActDiffLens
from src.diffing.methods.activation_difference_lens.agent import ActDiffLensAgent
from src.diffing.methods.activation_difference_lens.baseline_agent import BaselineActDiffLensAgent
from src.utils.graders.hypothesis_grader import grade_and_save

def _hydra_loguru_init() -> None:
    from hydra.core.hydra_config import HydraConfig
    hydra_path = HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "activation_difference_agent_cli.log"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _hydra_loguru_init()
    logger.info("Starting Activation Difference Lens Agent")

    # Enforce correct method
    assert str(cfg.diffing.method.name) == "activation_difference_lens", (
        "This CLI is specific to activation_difference_lens. "
        "Run with override diffing/method=activation_difference_lens"
    )

    method = ActDiffLens(cfg)
    agent_cfg = cfg.diffing.method.agent
    assert getattr(agent_cfg, "enabled", True), "Agent must be enabled in config"
    agent = ActDiffLensAgent(agent_cfg)

    description, stats = agent.run(method, return_stats=True)
    assert isinstance(description, str) and len(description) > 0
    assert isinstance(stats, dict) and isinstance(stats.get("messages"), list)

    # Prepare output directory under method results
    organism = str(cfg.organism.name)
    model = str(cfg.model.name)
    llm_id = str(cfg.diffing.method.agent.llm.model_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(method.results_dir) / "agent" / f"{timestamp}_{organism}_{model}_{llm_id}_mi{agent_cfg.budgets.model_interactions}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving outputs to {out_dir}")
    (out_dir / "description.txt").write_text(description, encoding="utf-8")
    with open(out_dir / "messages.json", "w", encoding="utf-8") as f:
        json.dump(stats["messages"], f, ensure_ascii=False, indent=2)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in stats.items() if k != "messages"}, f, ensure_ascii=False, indent=2)

    # Immediate grading of agent hypothesis
    agent_score, _agent_text = grade_and_save(cfg, description, save_dir=out_dir)
    logger.info(f"Graded agent description with score={agent_score} ({_agent_text})")
    logger.debug(f"Reasoning: {_agent_text}")

    logger.info("Agent run complete")

    # Collect descriptions and stats for final summary
    all_descriptions: list[tuple[str, str]] = [("agent", description)]
    all_stats: list[tuple[str, dict]] = [("agent", {k: v for k, v in stats.items() if k != "messages"})]
    grade_summaries: list[tuple[str, int, str]] = [("agent", agent_score, _agent_text)]


    # Optionally run baselines
    run_baselines_flag = bool(getattr(agent_cfg, "run_baselines", False))
    if run_baselines_flag:
        if agent_cfg.budgets.model_interactions == 0:
            logger.warning("Model interactions budget is 0, skipping baseline runs")
            return

        logger.info("Running baseline agents")

        # Capture original budget
        orig_budget = int(agent_cfg.budgets.model_interactions)
        assert orig_budget > 0

        def _clone_agent_cfg_with_budget(multiplier: int) -> DictConfig:
            cfg_copy = OmegaConf.create(OmegaConf.to_container(agent_cfg, resolve=True))
            cfg_copy.budgets.model_interactions = int(orig_budget * multiplier)
            return cfg_copy

        for label, mult in [("same", 1), ("x10", 10)]:
            logger.info(f"Baseline run: {label} (mult={mult})")
            baseline_cfg = _clone_agent_cfg_with_budget(mult)
            baseline_agent = BaselineActDiffLensAgent(baseline_cfg)
            b_desc, b_stats = baseline_agent.run(method, return_stats=True)

            b_out_dir = Path(method.results_dir) / "agent" / f"{timestamp}_{organism}_{model}_{llm_id}_baseline_mi{baseline_cfg.budgets.model_interactions}"
            b_out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving baseline outputs to {b_out_dir}")
            (b_out_dir / "description.txt").write_text(b_desc, encoding="utf-8")
            with open(b_out_dir / "messages.json", "w", encoding="utf-8") as f:
                json.dump(b_stats["messages"], f, ensure_ascii=False, indent=2)
            with open(b_out_dir / "stats.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in b_stats.items() if k != "messages"}, f, ensure_ascii=False, indent=2)
            all_descriptions.append((f"baseline_{label}", b_desc))
            all_stats.append((f"baseline_{label}", {k: v for k, v in b_stats.items() if k != "messages"}))

            # Grade baseline hypothesis
            b_score, _b_text = grade_and_save(cfg, b_desc, save_dir=b_out_dir)
            grade_summaries.append((f"baseline_{label}", b_score, _b_text))
            logger.info(f"Graded baseline '{label}' description with score={b_score}")
            logger.debug(f"Reasoning: {_b_text}")

        logger.info("Baseline runs complete")

    # Print summary of all descriptions
    print("\n===== Descriptions Summary =====")
    for label, desc in all_descriptions:
        print(f"\n--- {label} ---\n{desc.strip()}\n")

    # Print summary of stats
    print("\n===== Stats Summary =====")
    for label, s in all_stats:
        print(
            f"\n--- {label} ---\n"
            f"agent_llm_calls_used: {s.get('agent_llm_calls_used')}\n"
            f"model_interactions_used: {s.get('model_interactions_used')}\n"
            f"agent_prompt_tokens: {s.get('agent_prompt_tokens')}\n"
            f"agent_completion_tokens: {s.get('agent_completion_tokens')}\n"
            f"agent_total_tokens: {s.get('agent_total_tokens')}\n"
        )
    
    # Print summary of grading
    print("\n===== Grading Summary =====")
    for label, score, text in grade_summaries:
        print(f"\n--- {label} ---\nScore: {score}\n{text.strip()}\n")
if __name__ == "__main__":
    main()

