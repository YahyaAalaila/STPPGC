from lightning_stpp.config_factory.model_config import NeuralSTPPConfig
from ._runner import BaseRunner
from lightning_stpp.models.base import BaseSTPPModule
from lightning_stpp.data.base import LightDataModule
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

@BaseRunner.register(name='dl_stpp')
class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg
        self._inititialize_model()
        self._setup_logger()
        # build trainer friom model config
        self.trainer = cfg.trainer.build_pl_trainer(
            logger = self.mlflow_logger,
            extra_callbacks = cfg.trainer.custom_callbacks,
            model_speciofic_callbacks= self.model.callbacks(), # build callbacks that might be specific to the model
            )
    

    def _inititialize_model(self):
        # Initialize the model based on the configuration
        model_cfg = self.config.model         
        stpp_class = BaseSTPPModule.by_name(model_cfg.model_id)
        self.model = stpp_class(model_cfg)
        # Initialize the corresponding datamodule
        self.datamodule = LightDataModule.build_datamodule_from_config(self.config)

    def fit(self):
        # Implement the logic for fitting the model
        self.logger.info("Starting training...")
        self.trainer.fit(self.model, datamodule=self.datamodule)
        
    def evaluate(self, verbose = False):
        # Implement the logic for evaluating the model
        self.logger.info("Starting evaluation on test set...")
        self.trainer.test(self.model,
                          datamodule=self.datamodule,
                          verbose=verbose)
    
    def fit_and_test(self):
        self.logger.info("START OF TRAINING + TESTING")
        self.fit()
        self.evaluate()
        self.logger.info("END OF TRAINING + TESTING")
        pass