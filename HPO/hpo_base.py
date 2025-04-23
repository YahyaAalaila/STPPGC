
from abc import abstractmethod
from config_factory.hypertuning_config import HPOConfig
from utils.registrable import Registrable


class HyperTuner(Registrable):
    """
    Base class for hyperparameter tuning.
    """
    def __init__(self, cfg: HPOConfig):
        self.hpo_config = cfg

    @staticmethod
    def build_hpo_from_config(cfg):
        """
        Build a hyperparameter tuning object from the configuration.
        """
        hpo_class = HyperTuner.by_name(cfg.hpo_id)
        if hpo_class is None:
            raise ValueError(f"HPO ID '{cfg.hpo_id}' not recognized.")
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