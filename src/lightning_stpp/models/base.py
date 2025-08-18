import lightning as pl
import torch.nn.functional as F
from abc import ABC, abstractmethod

from lightning_stpp.config_factory._config import Config

from lightning_stpp.utils.registrable import Registrable
from lightning_stpp.config_factory.data_config import DataConfig

class BaseSTPPModule(pl.LightningModule, Registrable, ABC):
    """
    BaseSTPPModule provides a standardized interface for STPP models using PyTorch Lightning.
    Subclasses must implement the forward() method and can override training_step() and validation_step()
    if needed for custom behavior.
    """
    def __init__(self, cfg: Config, data_cfg: DataConfig):
        super().__init__()
        self.model_cfg = cfg
        self.data_cfg = data_cfg
        self.float()
        # Save hyperparameters for easy access and logging
        self.save_hyperparameters(self.model_cfg.to_hparams())
    
    @classmethod
    def build_model_from_config(cls, model_cfg):
        key = model_cfg.model_id.lower()
        dm_cls = cls.by_name(key)   # uses the generic Registrable.by_name
        return dm_cls(model_cfg)

    @abstractmethod
    def forward(self, batch):
        """
        Perform the forward pass.
        This must be implemented by any subclass.
        """
        pass
    
    def training_step(self, batch, batch_idx):
        # Default training step: expects batch to be a tuple (inputs, targets)
        inputs, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs)
        # Compute loss; subclasses might implement different losses
        loss = self.compute_loss(outputs, targets)  # Subclasses can implement their own loss functions
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)  # Subclasses can implement their own loss functions
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Configure optimizers; subclasses can customize this
        pass
