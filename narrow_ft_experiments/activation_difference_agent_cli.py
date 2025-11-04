#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
from src.diffing.methods.activation_difference_lens.baseline_agent import (
    BaselineActDiffLensAgent,
)
from src.utils.graders.hypothesis_grader import grade_and_save


def _hydra_loguru_init() -> None:
    from hydra.core.hydra_config import HydraConfig

    hydra_path = HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "activation_difference_agent_cli.log"))
    logger.add(os.path.join(hydra_path, "main.log"))


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

    # Require number of repeats
    num_repeat = int(agent_cfg.num_repeat)
    assert num_repeat >= 1

    # Common metadata for output structure
    organism = str(cfg.organism.name)
    model = str(cfg.model.name)
    llm_id = str(cfg.diffing.method.agent.llm.model_id).replace("/", "_")

    # Overwrite behavior
    overwrite = bool(cfg.diffing.method.agent.overwrite)
    assert isinstance(overwrite, bool)

    # Collect descriptions and stats for final summary
    all_descriptions: list[tuple[str, str]] = []
    all_stats: list[tuple[str, dict]] = []
    grade_summaries: list[tuple[str, int, str]] = []

    suffix = (
        f"_pos{agent_cfg.overview.positions}"
        if agent_cfg.overview.positions != [0, 1, 2, 3, 4]
        else ""
    )
    # Run the agent multiple times and save each under a run-specific folder
    for run_idx in range(num_repeat):
        run_suffix = f"_run{run_idx}"
        out_dir = (
            Path(method.results_dir)
            / "agent"
            / f"{organism}_{model}_{llm_id}_mi{agent_cfg.budgets.model_interactions}{suffix}{run_suffix}"
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
                cfg, description, save_dir=out_dir
            )
            grade_summaries.append((f"agent_run{run_idx}", agent_score, _agent_text))
            all_descriptions.append((f"agent_run{run_idx}", description))
            continue

        logger.info(f"Agent run {run_idx+1}/{num_repeat}")

        agent = ActDiffLensAgent(agent_cfg)
        description, stats = agent.run(method, return_stats=True)
        assert isinstance(description, str) and len(description) > 0
        assert isinstance(stats, dict) and isinstance(stats.get("messages"), list)

        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving outputs to {out_dir}")
        save_description(description, stats, out_dir)

        # Immediate grading of agent hypothesis
        agent_score, _agent_text = grade_and_save(cfg, description, save_dir=out_dir)
        logger.info(
            f"Graded agent (run {run_idx}) description with score={agent_score} ({_agent_text})"
        )
        logger.debug(f"Reasoning: {_agent_text}")

        all_descriptions.append((f"agent_run{run_idx}", description))
        all_stats.append(
            (f"agent_run{run_idx}", {k: v for k, v in stats.items() if k != "messages"})
        )
        grade_summaries.append((f"agent_run{run_idx}", agent_score, _agent_text))

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

        for label, mult in [("x0", 0), ("x1", 1), ("x10", 10)]:
            logger.info(f"Baseline runs for: {label} (mult={mult})")
            baseline_cfg = _clone_agent_cfg_with_budget(mult)
            for run_idx in range(num_repeat):
                run_suffix = f"_run{run_idx}"
                b_out_dir = (
                    Path(method.results_dir)
                    / "agent"
                    / f"{organism}_{model}_{llm_id}_baseline_mi{baseline_cfg.budgets.model_interactions}{suffix}{run_suffix}"
                )

                if b_out_dir.exists() and not overwrite:
                    logger.info(
                        f"Baseline result exists and overwrite=False, skipping: {b_out_dir}"
                    )
                    assert (
                        b_out_dir.exists() and b_out_dir.is_dir()
                    ), f"Expected baseline run directory not found: {b_out_dir}"
                    b_desc_fp = b_out_dir / "description.txt"
                    assert (
                        b_desc_fp.exists() and b_desc_fp.is_file()
                    ), f"Missing description.txt in {b_out_dir}"
                    b_description = b_desc_fp.read_text(encoding="utf-8")
                    b_score, _b_text = grade_and_save(
                        cfg, b_description, save_dir=b_out_dir
                    )
                    grade_summaries.append(
                        (f"baseline_{label}_run{run_idx}", b_score, _b_text)
                    )
                    all_descriptions.append(
                        (f"baseline_{label}_run{run_idx}", b_description)
                    )
                    continue

                logger.info(f"Baseline {label} run {run_idx+1}/{num_repeat}")

                baseline_agent = BaselineActDiffLensAgent(baseline_cfg)
                b_desc, b_stats = baseline_agent.run(method, return_stats=True)

                b_out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving baseline outputs to {b_out_dir}")
                (b_out_dir / "description.txt").write_text(b_desc, encoding="utf-8")
                with open(b_out_dir / "messages.json", "w", encoding="utf-8") as f:
                    json.dump(b_stats["messages"], f, ensure_ascii=False, indent=2)
                with open(b_out_dir / "stats.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {k: v for k, v in b_stats.items() if k != "messages"},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                all_descriptions.append((f"baseline_{label}_run{run_idx}", b_desc))
                all_stats.append(
                    (
                        f"baseline_{label}_run{run_idx}",
                        {k: v for k, v in b_stats.items() if k != "messages"},
                    )
                )

                # Grade baseline hypothesis
                b_score, _b_text = grade_and_save(cfg, b_desc, save_dir=b_out_dir)
                grade_summaries.append(
                    (f"baseline_{label}_run{run_idx}", b_score, _b_text)
                )
                logger.info(
                    f"Graded baseline '{label}' (run {run_idx}) description with score={b_score}"
                )
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
