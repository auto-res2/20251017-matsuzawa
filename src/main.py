"""Experiment orchestrator: launches src.train as a subprocess with inherited Hydra overrides."""

import subprocess
import sys
from typing import List

import hydra
from hydra.utils import get_original_cli_args


@hydra.main(config_path="../config")
def main(cfg):
    """Hydra entrypoint that forwards all CLI overrides to the training subprocess."""
    original_cli: List[str] = get_original_cli_args()

    # Build subprocess command ------------------------------------------------
    cmd = [sys.executable, "-u", "-m", "src.train"] + original_cli

    # Force WandB disabled when trial_mode=true --------------------------------
    if cfg.get("trial_mode", False):
        cmd.append("wandb.mode=disabled")

    print("Executing subprocess:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
