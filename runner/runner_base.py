import os
import yaml
from omegaconf import OmegaConf
import logging
from abc import abstractmethod
from utils.registrable import Registrable
from pytorch_lightning.loggers import MLFlowLogger

from config_factory.runner_config import RunnerConfig 

class BaseRunner(Registrable):
    def __init__(self, cfg: RunnerConfig):
        ## TODO: Should some of this logic be in a helper method like _initialize(self)? Just for clarity
        # Load the configuration using OmegaConf
        self.cfg = cfg
        
        # TODO: Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BaseRunner")
        self._setup_logger()
        # TODO: Set up data loading logic here? 

    @staticmethod
    def build_runner_from_config(cfg):
        """
        TODO: Add description of what this function does.  
        """
        runner_class = BaseRunner.by_name(cfg.runner_id)
        if runner_class is None:
            raise ValueError(f"Runner ID '{cfg.runner_id}' not recognized.")
        return runner_class(cfg)
    
    ## TODO: Make this better
    def _setup_logger(self):
        
        self.mlflow_logger = MLFlowLogger(
            experiment_name=self.cfg.logging.experiment_name,
            run_name=self.cfg.logging.run_name,
            tracking_uri = self.cfg.logging.mlflow_uri
        )
    @abstractmethod
    def run(self):
        pass

