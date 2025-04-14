import os
import yaml
from omegaconf import OmegaConf
import logging
from abc import abstractmethod
from uttils.registrable import Registrable
from pytorch_lightning.loggers import MLFlowLogger

class BaseRunner(Registrable):
    def __init__(self, runner_config):
        ## TODO: Should some of this logic be in a helper method like _initialize(self)? Just for clarity
        # Load the configuration using OmegaConf
        self.runner_config = runner_config
        
        # TODO: Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BaseRunner")
        self._setup_logger()
        # TODO: Set up data loading logic here? 

    @staticmethod
    def build_runner_from_config(runner_config):
        """
        TODO: Add description of what this function does.  
        """
        runner_class = BaseRunner.by_name(runner_config.runner_id)
        if runner_class is None:
            raise ValueError(f"Runner ID '{runner_config.runner_id}' not recognized.")
        return runner_class(runner_config)
    
    ## TODO: Make this better
    def _setup_logger(self):
        
        runner_dir = os.path.abspath(os.path.dirname(__file__))
        experiments_dir = os.path.join(runner_dir, "..", "experiments")
        # Ensure the directory exists
        os.makedirs(experiments_dir, exist_ok=True)
        
        self.mlflow_logger = MLFlowLogger(
            experiment_name=self.runner_config.mlflow.experiment_name,
            run_name=self.runner_config.mlflow.run_name,
            tracking_uri=f"file://{os.path.abspath(experiments_dir)}"
        )
    @abstractmethod
    def run(self):
        pass

