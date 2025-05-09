from dataclasses import dataclass, field
from omegaconf import OmegaConf
import torch
from pathlib import Path
import pytorch_lightning as pl

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
        # data_cfg = DataConfig(**raw["data"])
        # trainer_cfg = TrainerConfig(**raw["trainer"])
        # #logging_cfg = LoggingConfig(**raw["logging"])
        # hpo_cfg = HPOConfig(**raw["hpo"]) if "hpo" in raw else None
        
        # If the YAML didn't specify logging, or set it to null, supply an empty default
        # if "logging" in raw and raw["logging"] is not None:
        #     logging_cfg = LoggingConfig(**raw["logging"])
        # else:
        #     logging_cfg = LoggingConfig()

        
        
        # 2) model: pick subclass via registry
        model_dict  = raw["model"]
        mdl_name    = model_dict.pop("model_config").lower()   # e.g. "neuralstpp"
        mdl_cls     = Config.by_name(mdl_name)              # returns NeuralSTPPConfig
        common.model   = mdl_cls.from_dict(model_dict)   
        
        # if common.logging is None:
        #     common.logging = LoggingConfig()
        # if common.hpo is None:
        #     common.hpo = HPOConfig()
            
        return common
        # return cls(
        #     data=data_cfg,
        #     model=model_cfg,
        #     trainer=trainer_cfg,
        #     logging=logging_cfg,
        #     hpo=hpo_cfg,
        #     runner_id=raw.get("runner_id", "dl_stpp"),  # default to "dl_stpp" if not provided
        # )
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RunnerConfig":
        """Load a config from a YAML file, and apply any overrides."""
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        raw.update(overrides)
        return cls.from_dict(raw)
        
    # cross‑checks
    def finalize(self) -> None:
        
        self.logging.mlflow_uri = "file:"
        
        # MLflow dir exists?
        mlflow = Path(self.logging.mlflow_uri.split("file:")[-1])
        mlflow.mkdir(parents=True, exist_ok=True)
        # Create checkpoint dir
        ckpt_dir = Path(self.trainer.ckpt_dir) if hasattr(self.trainer, "ckpt_dir") else None
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
        # Maybe cross check for model<>data compatibility, for now I can only think of number of event types.
        # TODO: If usefull, implement it later
        
        
        # Accelarator and precision sanity check.
        if self.trainer.gpus > 0 and not torch.cuda.is_available():
            raise ValueError("gpus>0 but CUDA is not available.")
        if self.trainer.precision.startswith("16") and self.trainer.gpus == 0:
            raise ValueError("Mixed precision requested but running on CPU.")
        
        #seed before anything random happens
        #pl.seed_everything(self.trainer.seed) This is commented out because it is set now in the benchmarking configuration which is the top level now!
        
        # Snapshot library versions? (yaml dump / Mlflow)
        self.specs["versions"] = {
            "torch" : torch.__version__,
            "pytorch_lightning": pl.__version__,
            #"ray"   : ray.__version__, TODO:I ll add this when I resolve ray instalation issues.
        }
        

    # aggregated Ray search‑space
    @staticmethod
    def ray_space():
        space = {}
        space.update(Config.ray_space())
        space.update(TrainerConfig.ray_space())
        # could add TrainerConfig.ray_space() if it traning has ray tunable parameters (refer to traning_config.py)
        return space
