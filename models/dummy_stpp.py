
from .base import BaseSTPPModule
import torch
import torch.nn as nn

class DummySTPP(BaseSTPPModule):
    def __init__(self, config):
        super().__init__(config)
        self.fc = nn.Linear(config.input_dim, config.output_dim)  # a trainable layer

    def forward(self, x):
        return self.fc(x)

    def compute_loss(self, outputs, targets):
        # Suppose batch is a tuple (output_from_model, target)
        loss = - (outputs - targets).mean()  # A dummy loss that still depends on outputs
        return loss
