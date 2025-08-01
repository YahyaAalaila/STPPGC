
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, List, Dict

import lightning as pl

import torch
from ._config import Config
from lightning_stpp.callbacks.common.train_logger import TrainLoggerCallback



@Config.register("trainer_config")
@dataclass
class TrainerConfig(Config):
    gpus       : int = 1
    max_epochs : int = 50
    precision  : str = 32
    accelerator : str = "cpu" # "cpu", "gpu", "mps"
    seed       : int = 42
    ckpt_dir   : str = "./checkpoints"
    save_top_k : int = 1
    log_every_n_steps : int = 1
    check_val_every_n_epochs : int = 1
    resume_from : str | None = None
    custom_callbacks : List[Dict[str, Any]] = field(default_factory=list) # (YA) Added this to allow custom callbacks in the building of pl.Trainer!
    
    def __post_init__(self):
        # field level validation. TODO: Add all relevant validation steps
        if self.gpus < 0:
            raise ValueError("Number of GPUs must be non-negative.")
        if self.max_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        if self.precision not in [16, 32]:
            raise ValueError("Precision must be '16' or '32'.")
        if self.seed < 0:
            raise ValueError("Seed must be non-negative.")
        if self.accelerator in ("gpu", "cuda"):
            if not torch.cuda.is_available():
                raise ValueError("Requested CUDA but it is not available.")
        if self.accelerator == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("Requested Apple-MPS but it is not available.")
        
    def _resolve_ckpt_path(self) -> str | None:
        if not self.resume_from:
            return None
        path = Path(self.resume_from)
        if path.is_file():
            logging.info(f"Resuming from checkpoint  ➜  {path}")
            return str(path)
        raise FileNotFoundError(
            f"resume_from '{path}' not found. "
            "Either correct the path or remove the field to start fresh."
        )

        
    def _build_custom_callbacks(self, extra_callbacks, model_speciofic_callbacks):
        
        common_callbacks = [
            TrainLoggerCallback(log_every_n_steps=self.log_every_n_steps),
        ]

        for cb_spec in self.custom_callbacks:
            # cb_spec is now a dict with keys "class_path" and optional "init_params"
            class_path = cb_spec["class_path"]
            init_args  = cb_spec.get("init_params", {})
            cls = self._locate_class(class_path)
            if cls is None:
                raise ValueError(f"Callback class '{class_path}' not found.")
            common_callbacks.append(cls(**init_args))
            
        all_callbacks = common_callbacks + (extra_callbacks or []) + (model_speciofic_callbacks or [])
        return all_callbacks
    def ray_space(self):
        # Add ray tunable parameters here if needed
        # TODO: Validate if this is needed. If yes, implement it, otherwise remove.
        return {}

    def build_pl_trainer(self, logger, extra_callbacks = None, model_speciofic_callbacks = None):

        # Add custom callbacks if any
        all_callbacks = self._build_custom_callbacks(extra_callbacks, model_speciofic_callbacks)
        return pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.gpus if self.gpus > 0 else 1,
            precision=32, #self.precision,
            logger=logger,
            callbacks=all_callbacks,
            enable_checkpointing=True,
            check_val_every_n_epoch= self.check_val_every_n_epochs
            )
        
    @staticmethod
    def _locate_class(class_path: str):
        """
        Dynamically import a class given its full python path.
        e.g. "lightning.callbacks.EarlyStopping"
        """
        module_path, cls_name = class_path.rsplit(".", 1)
        mod = __import__(module_path, fromlist=[cls_name])
        return getattr(mod, cls_name)
