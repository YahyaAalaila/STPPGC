import lightning as pl

from lightning_stpp.config_factory.data_config import DataConfig
from lightning_stpp.config_factory.runner_config import RunnerConfig
from lightning_stpp.utils.registrable import Registrable
from .datasets import STDataset
from lightning_stpp.utils.data import Float32Wrapper

class LightDataModule(pl.LightningDataModule, Registrable):
    def __init__(self, data_config: DataConfig):
        super().__init__()
        self.data_config  = data_config
    
    @classmethod
    def build_datamodule_from_config(cls, runner_cfg: RunnerConfig):
        dm_cls = cls.by_name(runner_cfg.model.model_id.lower())
        return dm_cls(runner_cfg.data)
    
    def prepare_data(self):
        """
        Load data from the configuration.
        TODO: This should be centralized here? YA: I think it should be. 
        This probably will need a bunch of helper functions 
        such as _load_data_from_pkl, _load_data_from_json, etc. --- Real data
        and _generate_synthetic_data, etc. ---
        """
        DataClass = STDataset.build_dataset_from_config(self.data_config)
        self.dataset_class = DataClass
         
    
    def setup(self, stage=None):
        """
        Setup the data module. This method is called on every GPU.
        Probably would be a good place to centralize loading data or simulating it!
        """
        self.training_set = self.dataset_class("train")
        self.validation_set = self.dataset_class("val")
        self.testing_set = self.dataset_class("test")
        
        self.train_dataset = Float32Wrapper(self.training_set)   
        self.validation_set = Float32Wrapper(self.validation_set)  
        self.testing_set = Float32Wrapper(self.testing_set)       
    def train_dataloader(self):
        """
        Return the training dataloader.
        """
        pass
    def val_dataloader(self):
        """
        Return the validation dataloader.
        """ 
        pass
    def test_dataloader(self):
        """
        Return the test dataloader.
        """ 
        pass