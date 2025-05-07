import pytorch_lightning as pl
from datasets import STDataset

class LightDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.config = self.config.dataconfig
        
    @staticmethod
    def build_datamodule_from_config(model_config):
        """
        Generates a model from the given configuration.
        This method should be overridden in subclasses to create specific models.
        """
        model_id = model_config.model_id
        for subclass in LightDataModule.__subclasses__():
            if model_id == subclass.__name__:
                return subclass(model_config)
        # If not found, collect all available model IDs.
        available_ids = ", ".join(sub.__name__ for sub in LightDataModule.__subclasses__())
        raise ValueError(f"Model ID '{model_id}' not recognized. Available model IDs are: {available_ids}")
    
    def prepare_data(self, split, data_config):
        """
        Load data from the configuration.
        TODO: This should be centralized here? YA: I think it should be. 
        This probably will need a bunch of helper functions 
        such as _load_data_from_pkl, _load_data_from_json, etc. --- Real data
        and _generate_synthetic_data, etc. --- Dummy data
        (YA): Is including this logic here wise?
        """
        DataClass = STDataset.build_dataset_from_config(data_config)
        return DataClass(split)
    
    def setup(self, stage=None):
        """
        Setup the data module. This method is called on every GPU.
        Probably would be a good place to centralize loading data or simulating it!
        """
        self.training_set = self.prepare_data("train")
        self.validation_set = self.prepare_data("val")
        self.testing_set = self.prepare_data("test")
        
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