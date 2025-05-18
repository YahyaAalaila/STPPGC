
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, List, Dict
import pytorch_lightning as pl


from lightning.pytorch.callbacks import ModelCheckpoint
from ._config import Config
from lightning_stpp.callbacks.common.train_logger import TrainLoggerCallback
from lightning_stpp.callbacks.common.validation_scheduler import ValidationSchedulerCallback
from lightning_stpp.callbacks.common.test_scheduler import TestSchedulerCallback 



@Config.register("trainer_config")
@dataclass
class TrainerConfig(Config):
    gpus       : int = 1
    max_epochs : int = 50
    precision  : str = "32"
    seed       : int = 42
    ckpt_dir   : str = "./checkpoints"
    save_top_k : int = 1
    log_every_n_steps : int = 1
    check_val_every_n_epochs : int = 1
    monitor    : str = "val_loss"
    resume_from : str | None = None
    custom_callbacks : List[Dict[str, Any]] = field(default_factory=list) # (YA) Added this to allow custom callbacks in the building of pl.Trainer!
    
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

        
    def _build_custom_callbacks(self, extra_callbacks, model_speciofic_callbacks):
        
        common_callbacks = [
            ModelCheckpoint(dirpath= self.ckpt_dir, save_top_k  = self.save_top_k, monitor = self.monitor),
            TrainLoggerCallback(log_every_n_steps=self.log_every_n_steps),
            ValidationSchedulerCallback(every_n_epochs=self.check_val_every_n_epochs),
            #TestSchedulerCallback(),
        ]
        # Create custom callbacks if any
        dynamic_callbacks_from_yaml = []
        for cf_cfg in self.custom_callbacks:
            cls = self._locate_class(cf_cfg.class_path)
            if cls is None:
                raise ValueError(f"Class {cf_cfg['class_path']} not found.")
            dynamic_callbacks_from_yaml.append(cls(**cf_cfg.get("init_params", {})))
        
        all_callbacks = common_callbacks + dynamic_callbacks_from_yaml + (extra_callbacks or []) + (model_speciofic_callbacks or [])
        return all_callbacks
    def ray_space(self):
        # Add ray tunable parameters here if needed
        # TODO: Validate if this is needed. If yes, implement it, otherwise remove.
        return {}

    def build_pl_trainer(self, logger, extra_callbacks = None, model_speciofic_callbacks = None):
        # build ModelCheckpoint callback
        #ckpt_cb = self._model_checkpoint()
        # Add custom callbacks if any
        all_callbacks = self._build_custom_callbacks(extra_callbacks, model_speciofic_callbacks)
        return pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.gpus > 0 else "cpu",
            devices=self.gpus if self.gpus > 0 else 1,
            precision=self.precision,
            logger=logger,
            callbacks=[all_callbacks],
            enable_checkpointing=True,
            ckpt_path = self._resolve_ckpt_path(), # Decide whether to resume training or not (if checkpoint exists, it will be used. Otherwise, training will start from scratch.
        )
        
    @staticmethod
    def _locate_class(class_path: str):
        """
        Dynamically import a class given its full python path.
        e.g. "pytorch_lightning.callbacks.EarlyStopping"
        """
        module_path, cls_name = class_path.rsplit(".", 1)
        mod = __import__(module_path, fromlist=[cls_name])
        return getattr(mod, cls_name)
