from .runner_base import BaseRunner
from models.base import BaseSTPPModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from datasets.datamodule import DummyDataModule
from omegaconf import OmegaConf

@BaseRunner.register(name='dl_stpp')
class LighRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._inititialize_model()
        #self._setup_logger()
        

        trainer_params = OmegaConf.to_container(self.config.trainer, resolve=True)
        trainer_params['logger'] = self.mlflow_logger
        self.trainer = pl.Trainer(**trainer_params)

    def _inititialize_model(self):
        # Initialize the model based on the configuration
        self.model = BaseSTPPModule.generate_model_from_config(self.config.model_config)
        self.datamodule = DummyDataModule(self.config.datamodule)
        num_model_params = sum(p.numel() for p in self.model.parameters()) # Probably wise to add this to be logged
    # def _setup_logger(self):
    #     # Set up MLFlowLogger using values from the config.
    #     self.mlflow_logger = MLFlowLogger(
    #         experiment_name=self.config.mlflow.experiment_name,
    #         run_name=self.config.mlflow.run_name,
    #         tracking_uri=self.config.mlflow.tracking_uri
    #     )

    def run(self):
        # Implement the logic for running the STPP model
        self.logger.info("Starting training...")
        self.trainer.fit(self.model, datamodule=self.datamodule)
        self.logger.info("Training complete; starting testing...")
        self.trainer.test(self.model, datamodule=self.datamodule)
        pass