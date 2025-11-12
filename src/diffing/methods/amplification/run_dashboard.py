"""
Standalone script to run the amplification dashboard for development.
"""
import sys
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))


GlobalHydra.instance().clear()


@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg):
    """Main entry point for running the dashboard standalone."""
    from src.diffing.methods.amplification.weight_difference import (
        WeightDifferenceAmplification,
    )
    from src.diffing.methods.amplification.amplification_dashboard import (
        AmplificationDashboard,
    )

    method = WeightDifferenceAmplification(cfg=cfg)

    dashboard = AmplificationDashboard(method)
    dashboard.display()


if __name__ == "__main__":
    main()

