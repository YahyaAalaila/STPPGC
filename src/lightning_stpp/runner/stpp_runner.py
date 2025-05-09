from ._runner import BaseRunner
from lightning_stpp.models.base import BaseSTPPModule
from lightning_stpp.data.base import LightDataModule

@BaseRunner.register(name='dl_stpp')
class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg
        self._inititialize_model(cfg)
        self._setup_logger()
        
        # build trainer friom model config
        self.trainer = cfg.trainer.build_pl_trainer(
            logger = self.mlflow_logger,
            extra_callbacks = cfg.trainer.custom_callbacks,
            model_speciofic_callbacks= self.model.callbacks(), # build callbacks that might be specific to the model
            )

    def _inititialize_model(self, cfg):
        # Initialize the model based on the configuration
        #self.model = BaseSTPPModule.build_model_from_config(cfg.model)
        stpp_class = BaseSTPPModule.by_name(cfg.model)
        self.model = stpp_class(cfg.model)
        # Initialize the corresponding datamodule
        self.datamodule = LightDataModule.build_datamodule_from_config(cfg.data)

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