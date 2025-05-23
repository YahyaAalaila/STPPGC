from dataclasses import dataclass, field
from omegaconf import OmegaConf
import torch
from pathlib import Path
import lightning as pl

from .data_config import DataConfig
from .trainer_config import TrainerConfig
from .logger_config import LoggingConfig
from .hypertuning_config import HPOConfig
from ._config import Config

#@Config.register("runner_config")
@dataclass
class RunnerConfig(Config):
    data     : DataConfig
    model    : Config
    trainer  : TrainerConfig
    logging  : LoggingConfig = field(default_factory=LoggingConfig)
    hpo      : HPOConfig = field(default_factory=HPOConfig)
    runner_id: str = "dl_stpp"      # tells BaseRunner which subclass
    @classmethod
    def from_dict(cls, raw: dict) -> "RunnerConfig":
        # let base-class build everything it can
        common = super().from_dict(raw)      
        # model: pick subclass via registry
        model_dict  = raw["model"]
        mdl_name    = model_dict.pop("model_config").lower()   # e.g. "neuralstpp"
        mdl_cls     = Config.by_name(mdl_name)              # returns NeuralSTPPConfig
        common.model   = mdl_cls.from_dict(model_dict)    
        return common
    
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RunnerConfig":
        """Load a config from a YAML file, and apply any overrides."""
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        raw.update(overrides)
        return cls.from_dict(raw)
        
    # cross‑checks
    def finalize(self) -> None:
        # --------- Finalize the configuration ---------
        self.logging.finalize()
        # Create checkpoint dir
        ckpt_dir = Path(self.trainer.ckpt_dir) if hasattr(self.trainer, "ckpt_dir") else None
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
        # Maybe cross check for model<>data compatibility, for now I can only think of number of event types.
        # TODO: If usefull, implement it later
        
    # aggregated Ray search‑space
    @staticmethod
    def ray_space():
        space = {}
        space.update(Config.ray_space())
        space.update(TrainerConfig.ray_space())
        # could add TrainerConfig.ray_space() if it traning has ray tunable parameters (refer to traning_config.py)
        return space
