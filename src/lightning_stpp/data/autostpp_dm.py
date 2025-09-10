import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning_stpp.data.base import LightDataModule
from autoint_stpp.data.data import SlidingWindowWrapper
#from deepstpp.data.dataset import SlidingWindowWrapper

@LightDataModule.register("autostpp")
class AutoSTPPDataModule(LightDataModule):
    """
    DataModule for AutoSTPP, using SlidingWindowWrapper format.
    Datasets are expected to be set via `data_config`.
    """

    def __init__(self, data_config):
        super().__init__(data_config)
        # Datasets (train/val/test) should already be prepared
        # e.g., via YAML config or elsewhere before initializing this DataModule

    def train_dataloader(self):
        self.training_set = SlidingWindowWrapper(
            self.training_set,
            normalized=True,
            lookback=20,   # match DeepSTPP
            lookahead=1,
            
        )
        print("Autostpp train shape:", self.training_set.st_X.shape)

        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.data_config.train_bsz,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        self.validation_set = SlidingWindowWrapper(
            self.validation_set,
            normalized=True,
            min=self.training_set.min,
            max=self.training_set.max
        )
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=self.data_config.val_bsz,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        self.testing_set = SlidingWindowWrapper(
            self.testing_set,
            normalized=True,
            min=self.training_set.min,
            max=self.training_set.max
        )
        return torch.utils.data.DataLoader(
            self.testing_set,
            batch_size=self.data_config.test_bsz,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )