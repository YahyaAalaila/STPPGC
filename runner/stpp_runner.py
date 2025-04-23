from .runner_base import BaseRunner
from models.base import BaseSTPPModule
from data.base import LightDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from data.datamodule import DummyDataModule
from omegaconf import OmegaConf

@BaseRunner.register(name='dl_stpp')
class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg
        self._inititialize_model(cfg)
        self._setup_logger()
        self.trainer = cfg.trainer.build_pl_trainer(logger = self.mlflow_logger)

    def _inititialize_model(self, cfg):
        # Initialize the model based on the configuration
        self.model = BaseSTPPModule.build_model_from_config(cfg.model)
        self.datamodule = LightDataModule.build_datamodule_from_config(self.data)


    def run(self):
        # Implement the logic for running the STPP model
        self.logger.info("Starting training...")
        self.trainer.fit(self.model, datamodule=self.datamodule)
        self.logger.info("Training complete; starting testing...")
        self.trainer.test(self.model, datamodule=self.datamodule)
        pass