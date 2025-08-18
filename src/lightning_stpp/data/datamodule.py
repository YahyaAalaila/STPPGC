import lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as l
## This is not going to stay here, this will be changed by DGP package


class DummyDataModule(l.LightningDataModule):
    def __init__(self, config):
        """
        Initialize the dummy module.
        
        Args:
            config: A configuration dictionary or OmegaConf object containing parameters.
                    For example, config.datamodule.batch_size and config.model.input_dim.
        """
        super().__init__()
        self.batch_size = config.batch_size
        self.input_dim = config.input_dim  # e.g., 10
    
    def train_dataloader(self):
        """
        Return the DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        """
        Return the DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        """
        Return the DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
