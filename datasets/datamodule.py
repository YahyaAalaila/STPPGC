import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

class DummyDataModule(pl.LightningDataModule):
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
    
    def setup(self, stage=None):
        """
        Prepare the dataset. Here, we simulate data.
        
        We generate a dataset of random numbers:
          - Inputs: a tensor of shape (N, input_dim)
          - Targets: a tensor of shape (N, 1) 
        Then, we split the dataset into train, validation, and test sets.
        """
        # Generate synthetic data for demonstration: 1000 examples.
        x = torch.randn(1000, self.input_dim)
        y = torch.randn(1000, 1)
        
        dataset = TensorDataset(x, y)
        # Split sizes: 80% for training, 10% for validation, 10% for testing.
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
    
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
