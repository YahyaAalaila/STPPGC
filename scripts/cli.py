#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning_stpp.config_factory.runner_config import RunnerConfig
from lightning_stpp.HPO.hpo_base import HyperTuner
import torch
import os

torch.set_default_dtype(torch.float32)


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig):
    # ---- AUTO DEVICE OVERRIDE ----
    device_name = detect_device()
    print(f"Automatically using device: {device_name}")

    # Override accelerator in Hydra config
    cfg.trainer.accelerator = device_name
    
    # 1) Build dataclass-based RunnerConfig
    raw = OmegaConf.to_container(cfg, resolve=True)
    runner_cfg = RunnerConfig.from_dict(raw)
    runner_cfg.finalize()

    # 2) Hand off to HPO / training loop
    tuner = HyperTuner.build_hpo_from_config(runner_cfg)
    tuner.run()


if __name__ == "__main__":
    main()
