from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from internal_time_rl.algorithms import train_ppo_with_internal_time


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    artifacts = train_ppo_with_internal_time(cfg)
    print(f"Run directory: {artifacts['run_dir']}")
    print(f"Metrics CSV: {artifacts['metrics_csv']}")
    print(f"Checkpoint: {artifacts['checkpoint']}")


if __name__ == "__main__":
    main()

