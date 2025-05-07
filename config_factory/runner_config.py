from dataclasses import dataclass
import torch
from pathlib import Path
import pytorch_lightning as pl

from .data_config import DataConfig
from .model_config import ModelConfig
from .trainer_config import TrainerConfig
from .logger_config import LoggingConfig
from .hypertuning_config import HPOConfig
from .base import Config

@Config.register("runner_config")
@dataclass
class RunnerConfig(Config):
    data     : DataConfig
    model    : ModelConfig
    trainer  : TrainerConfig
    logging  : LoggingConfig
    hpo      : HPOConfig | None = None
    runner_id: str = "dl_stpp"      # tells BaseRunner which subclass

    # cross‑checks
    def finalize(self) -> None:
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
        pl.seed_everything(self.trainer.seed)
        
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
        space.update(ModelConfig.ray_space())
        # could add TrainerConfig.ray_space() if it traning has ray tunable parameters (refer to traning_config.py)
        return space
