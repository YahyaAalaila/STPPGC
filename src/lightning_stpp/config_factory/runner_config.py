from dataclasses import dataclass, field
from omegaconf import OmegaConf
from pathlib import Path

from .aliases import infer_model_key
from lightning_stpp.config_factory import (
    DataConfig,
    TrainerConfig,
    LoggingConfig,
    HPOConfig,
    BaseModelConfig,
    Config,
)



@dataclass
class RunnerConfig(Config):
    data     : DataConfig
    model    : BaseModelConfig
    trainer  : TrainerConfig
    logging  : LoggingConfig = field(default_factory=LoggingConfig)
    hpo      : HPOConfig = field(default_factory=HPOConfig)
    runner_id: str = "dl_stpp"      # tells BaseRunner which subclass
    @classmethod
    def from_dict(cls, raw: dict) -> "RunnerConfig":
        model_dict = raw.setdefault("model", {})
        model_key = model_dict.get("model_id", infer_model_key(model_dict))
        model_dict["model_id"] = model_key
        # let base-class build everything it can
        common = super().from_dict(raw)      
        mdl_cls     = Config.by_name(model_key)             
        common.model   = mdl_cls.from_dict(model_dict)  
        common.model.model_id = model_key  
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

        # Make sure ModelCheckpoint is included in the callbakcs, which is extremely important, that is why it is included here.
        common_cbs = self._set_common_callbacks()
        self.trainer.custom_callbacks[0:0] = common_cbs
        # Maybe cross check for model<>data compatibility, for now I can only think of number of event types.
        # TODO: If usefull, implement it later
        
    def _set_common_callbacks(self):
        # Ensure the base ckpt_dir exists
        base_dir = Path(self.trainer.ckpt_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Build a per-run subfolder: <base>/<dataset_id>/<model_id>/<run_name>
        subdir = base_dir / self.data.dataset_id / self.model.model_id / self.logging.run_name
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Prepare the callback spec for ModelCheckpoint
        ckpt_spec = {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_params": {
                "dirpath":   str(subdir),
                "save_top_k": self.trainer.save_top_k,
                "monitor":    self.model.monitor,
                "mode":       self.model.monitor_mode,
                "save_last":  True,
            },
        }
        
        tune_spec = {
            "class_path": "ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback",
            "init_params": {
                "metrics": {self.model.monitor: self.model.monitor },
                "on": "validation_epoch_end"
            },
        }
        # TODO: MAke the logic here better
        return [ckpt_spec, tune_spec]
        
    # aggregated Ray search‑space
    @staticmethod
    def ray_space():
        space = {}
        space.update(Config.ray_space())
        space.update(TrainerConfig.ray_space())
        # could add TrainerConfig.ray_space() if it traning has ray tunable parameters (refer to traning_config.py)
        return space
