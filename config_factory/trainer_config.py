
from dataclasses import dataclass
import logging
from pathlib import Path
from .base import Config
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

@Config.register("trainer_config")
@dataclass
class TrainerConfig(Config):
    gpus       : int = 1
    max_epochs : int = 50
    precision  : str = "32"
    seed       : int = 42
    ckpt_dir   : str = "./checkpoints"
    save_top_k : int = 1
    monitor    : str = "val_loss"
    resume_from : str | None = None
    
    def __post_init__(self):
        # field level validation. TODO: Add all relevant validation steps
        if self.gpus < 0:
            raise ValueError("Number of GPUs must be non-negative.")
        if self.max_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        if self.precision not in ["16", "32"]:
            raise ValueError("Precision must be '16' or '32'.")
        if self.seed < 0:
            raise ValueError("Seed must be non-negative.")
        
    def _resolve_ckpt_path(self) -> str | None:
        if not self.resume_path:
            return None
        path = Path(self.resume_path)
        if path.is_file():
            logging.info(f"Resuming from checkpoint  ➜  {path}")
            return str(path)
        raise FileNotFoundError(
            f"resume_path '{path}' not found. "
            "Either correct the path or remove the field to start fresh."
        )
        
    def ray_space(self):
        # Add ray tunable parameters here if needed
        # TODO: Validate if this is needed. If yes, implement it, otherwise remove.
        pass

    def build_pl_trainer(self, logger):
        ckpt_cb = ModelCheckpoint(
            dirpath     = self.ckpt_dir,
            save_top_k  = self.save_top_k,
            monitor     = self.monitor_metric,
        )
        return pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.gpus > 0 else "cpu",
            devices=self.gpus if self.gpus > 0 else 1,
            precision=self.precision,
            logger=logger,
            callbacks=[ckpt_cb],
            enable_checkpointing=True,
            ckpt_path = self._resolve_ckpt_path(), # Decide whether to resume training or not (if checkpoint exists, it will be used. Otherwise, training will start from scratch.
        )
