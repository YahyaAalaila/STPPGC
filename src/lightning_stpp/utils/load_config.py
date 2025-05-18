from __future__ import annotations
from lightning_stpp.config_factory.runner_config import RunnerConfig

def load_and_finalize(path: str) -> RunnerConfig:
    cfg = RunnerConfig.from_yaml(path)
    cfg.finalize()
    return cfg
