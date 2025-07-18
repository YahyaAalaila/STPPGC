from __future__ import annotations
from lightning_stpp.config_factory.runner_config import RunnerConfig

def load_and_finalize(path: str) -> RunnerConfig:
    cfg = RunnerConfig.from_yaml(path)
    print(f"[DEBUG] -- cfg.model = {cfg.model}")
    cfg.finalize()
    return cfg
