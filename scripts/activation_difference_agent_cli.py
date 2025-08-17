#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from loguru import logger

# Ensure project root is on sys.path so that `src.*` imports work when executed from scripts/
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.diffing.methods.activation_difference_lens.act_diff_lens import ActDiffLens
from src.diffing.methods.activation_difference_lens.agent import ActDiffLensAgent


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

    description, messages = agent.run(method, return_messages=True)
    assert isinstance(description, str) and len(description) > 0
    assert isinstance(messages, list) and all(isinstance(m, dict) for m in messages)

    # Prepare output directory under method results
    organism = str(cfg.organism.name)
    model = str(cfg.model.name)
    llm_id = str(cfg.diffing.method.agent.llm.model_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(method.results_dir) / "agent" / f"{timestamp}_{organism}_{model}_{llm_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving outputs to {out_dir}")
    (out_dir / "description.txt").write_text(description, encoding="utf-8")
    with open(out_dir / "messages.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    logger.info("Agent run complete")


if __name__ == "__main__":
    main()

