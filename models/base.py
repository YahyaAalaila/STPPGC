import pytorch_lightning as pl
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseSTPPModule(pl.LightningModule, ABC):
    """
    BaseSTPPModule provides a standardized interface for STPP models using PyTorch Lightning.
    Subclasses must implement the forward() method and can override training_step() and validation_step()
    if needed for custom behavior.
    """
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config.learning_rate
        # Save hyperparameters for easy access and logging
        self.save_hyperparameters(logger = config)

    @staticmethod
    def build_model_from_config(model_config):
        """
        Generates a model from the given configuration.
        This method should be overridden in subclasses to create specific models.
        """
        model_id = model_config.model_id
        for subclass in BaseSTPPModule.__subclasses__():
            if model_id == subclass.__name__:
                return subclass(model_config)
        # If not found, collect all available model IDs.
        available_ids = ", ".join(sub.__name__ for sub in BaseSTPPModule.__subclasses__())
        raise ValueError(f"Model ID '{model_id}' not recognized. Available model IDs are: {available_ids}")

    @abstractmethod
    def forward(self, batch):
        """
        Perform the forward pass.
        This must be implemented by any subclass.
        """
        pass
    ## training step is unified across methods, only loss function withing will be overridden in the subclassses.
    ## TODO: Are on_train_batch_start, on_train_batch_end, or training_epoch_end() needed? 
    ## If yes, it would probably have to also be overridden in the subclasses.
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
        # Default validation step; similar to training_step
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)  # Subclasses can implement their own loss functions
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return NotImplementedError("Test step not implemented yet.")

    def configure_optimizers(self):
        # Configure optimizers; subclasses can customize this
        pass
