import torch
from .base import LightDataModule
from deepstpp.data.dataset import SlidingWindowWrapper

@LightDataModule.register("deep_stpp")
@LightDataModule.register("deepstpp")
class DeepSTPPDataModule(LightDataModule):
    """
    DataModule for DeepSTPP, which uses the same data format as NeuralSTPP.
    """
    def __init__(self, data_config):
        super().__init__(data_config)
        # self.load_data_from_config()  # Load data from config if needed

    def train_dataloader(self):
        self.training_set = SlidingWindowWrapper(
            self.training_set,
            normalized=True
        )
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.data_config.train_bsz,
            shuffle=True
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
            batch_size=self.data_config.val_bsz
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
            batch_size=self.data_config.test_bsz
        )

