#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf
import pprint

from lightning_stpp.config_factory.runner_config import RunnerConfig
from lightning_stpp.HPO.hpo_base import HyperTuner
import torch, traceback

# keep a reference to the real .to
_orig_to = torch.Tensor.to

def _debug_to(self, *args, **kwargs):
    # only trigger on float64
    if self.dtype == torch.float64:
        print("⚠️ Converting a float64 tensor to device/mode:", args, kwargs)
        print("   shape:", tuple(self.shape))
        traceback.print_stack(limit=8)
    return _orig_to(self, *args, **kwargs)

# Monkey–patch!
torch.Tensor.to = _debug_to


@hydra.main(
    version_base=None,             # Hydra 1.1 defaults
    config_path="../conf",         # relative to this script
    config_name="config",          # loads conf/config.yaml
)
def main(cfg: DictConfig):
    """
    Hydra will:
      • load conf/config.yaml,
      • swap in conf/model/<model>.yaml if you pass --config-name=config model=neuralstpp,
      • load conf/data/<data>.yaml if the model fragment specifies it,
      • load conf/trainer/default.yaml, conf/logging/default.yaml, conf/hpo/default.yaml,
      • apply any overrides you pass on the CLI like model.hdims=[128,128,128].
    """
    # 1) build our dataclass-based RunnerConfig
    raw = OmegaConf.to_container(cfg, resolve=True)
    runner_cfg = RunnerConfig.from_dict(raw)
    runner_cfg.finalize()


    # 2) hand off to your HPO / training loop
    tuner = HyperTuner.build_hpo_from_config(runner_cfg)
    tuner.run()

if __name__ == "__main__":
    main()
