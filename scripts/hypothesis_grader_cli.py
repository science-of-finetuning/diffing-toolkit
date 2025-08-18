#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

# Ensure project root is on sys.path so that `src.*` imports work when executed from scripts/
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.diffing.methods.activation_difference_lens.act_diff_lens import ActDiffLens
from src.utils.graders.hypothesis_grader import grade_and_save


def _hydra_loguru_init() -> None:
    from hydra.core.hydra_config import HydraConfig
    hydra_path = HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "hypothesis_grader_cli.log"))


def _find_latest_run_dirs(results_dir: Path, organism: str, model: str) -> List[Path]:
    assert results_dir.exists() and results_dir.is_dir()
    agent_dir = results_dir / "agent"
    assert agent_dir.exists() and agent_dir.is_dir()
    pattern = re.compile(r"^(\d{8}_\d{6})_" + re.escape(organism) + "_" + re.escape(model) + r"_")
    candidates: List[Tuple[str, Path]] = []
    for child in agent_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m is None:
            continue
        candidates.append((m.group(1), child))
    assert len(candidates) > 0, f"No matching runs found in {agent_dir} for organism={organism} model={model}"
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_ts = candidates[0][0]
    latest_dirs = [p for ts, p in candidates if ts == latest_ts]
    assert len(latest_dirs) >= 1
    return latest_dirs



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _hydra_loguru_init()
    logger.info("Starting Hypothesis Grader CLI")

    assert str(cfg.diffing.method.name) == "activation_difference_lens", (
        "This CLI targets activation_difference_lens. "
        "Run with override diffing/method=activation_difference_lens"
    )

    method = ActDiffLens(cfg)
    organism = str(cfg.organism.name)
    model = str(cfg.model.name)

    override_description = str(getattr(cfg, "description", "") or "").strip()
    if len(override_description) > 0:
        logger.info("Grading override description string only")
        score, _text = grade_and_save(cfg, override_description)
        logger.info(f"Graded override description with score={score}")
        logger.info(f"Reasoning: {_text}")
        return

    latest_dirs = _find_latest_run_dirs(Path(method.results_dir), organism, model)
    logger.info(f"Found {len(latest_dirs)} run dirs for latest timestamp")

    results: List[Tuple[str, int, str]] = []
    for run_dir in latest_dirs:
        for runtype in run_dir.iterdir():
            if not runtype.is_dir():
                continue
            desc_path = runtype / "description.txt"
            desc = desc_path.read_text(encoding="utf-8")
            score, _text = grade_and_save(cfg, desc, save_dir=runtype)
            results.append((runtype.name, score, _text))
            logger.info(f"Graded {runtype.name} with score={score}")

    logger.info("\n===== Grading Summary =====")
    for name, score, reasoning in results:
        logger.info(f"{name}: {score}")
        logger.info(f"Reasoning: {reasoning}")


if __name__ == "__main__":
    main()

