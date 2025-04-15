
from abc import abstractmethod
from uttils.registrable import Registrable


class HyperTunner(Registrable):
    """
    Base class for hyperparameter tuning.
    """
    def __init__(self, hpo_config):
        self.hpo_config = hpo_config


    @staticmethod
    def build_hpo_from_config(hpo_config):
        """
        Build a hyperparameter tuning object from the configuration.
        """
        hpo_class = HyperTunner.by_name(hpo_config.hpo_id)
        if hpo_class is None:
            raise ValueError(f"HPO ID '{hpo_config.hpo_id}' not recognized.")
        return hpo_class(hpo_config)
    
    @abstractmethod
    def run_tune(self):
        """
        Abstract method for tuning hyperparameters.
        This should be implemented by subclasses.
        TODO: Figure out what the inputs should be, datamodule should be a 
        variable or an attribute of the class?
        """
        pass