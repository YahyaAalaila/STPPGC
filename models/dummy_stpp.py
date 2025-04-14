
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


# class DummySTPP(BaseSTPPModule):
#     """
#     DummySTPPModule is a simple implementation of the BaseSTPPModule.
#     It uses a basic feedforward neural network for demonstration purposes.
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = None

#     def forward(self, x):
#         return x

#     def compute_loss(self, batch):
#         # Compute spatial and temporal log-likelihoods from outputs and targets.
#         space_loglik, time_loglik = batch
#         loglik = space_loglik + time_loglik
#         loss = -loglik.mean()
#         return loss