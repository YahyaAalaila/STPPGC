
from abc import abstractmethod
from lightning_stpp.config_factory.runner_config import HPOConfig, RunnerConfig
from lightning_stpp.utils.registrable import Registrable


class HyperTuner(Registrable):
    """
    Base class for hyperparameter tuning.
    """
    def __init__(self, cfg: RunnerConfig):
        self.hpo_config = cfg.hpo
        self.model_config = cfg.model
        self.trainer_config = cfg.trainer

    @staticmethod
    def build_hpo_from_config(cfg):
        """
        Build a hyperparameter tuning object from the configuration.
        """
        hpo_class = HyperTuner.by_name(cfg.hpo.hpo_id)
        if hpo_class is None:
            raise ValueError(f"HPO ID '{cfg.hpo.hpo_id}' not recognized.")
        return hpo_class(cfg)
    
    @abstractmethod
    def run_tune(self):
        """
        Abstract method for tuning hyperparameters.
        This should be implemented by subclasses.
        TODO: Figure out what the inputs should be, datamodule should be a 
        variable or an attribute of the class?
        """
        pass