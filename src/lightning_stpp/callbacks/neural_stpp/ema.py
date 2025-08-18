#### This contains custom callbacks for the training process.
# At the moment is contains EMA class, which is a callback for Exponential Moving Average (EMA) of model weights.
# So far this is used in NeuralSTPP



from lightning.pytorch.callbacks import Callback
import torch

class ExponentialMovingAverage(object):
    # As of 2025 - 04 - 26, this is copied from neural_stpp.utils
    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.decay = decay
        self.module_params = {n: p for (n, p) in module.named_parameters()}
        self.ema_params = {n: p.data.clone() for (n, p) in module.named_parameters()}
        self.nparams = sum(p.numel() for (_, p) in self.ema_params.items())

    def apply(self, decay=None):
        decay = decay or self.decay
        with torch.no_grad():
            for name, param in self.module_params.items():
                self.ema_params[name] -= (1 - decay) * (self.ema_params[name] - param.data)

    def set(self, named_params):
        with torch.no_grad():
            for name, param in named_params.items():
                self.ema_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module_params.items():
            param.data.copy_(self.ema_params[name])

    def swap(self):
        for name, param in self.module_params.items():
            tmp = self.ema_params[name].clone()
            self.ema_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )

# class EMACallback(Callback):
    
#     def __init__(self, decay = 0.99): 
#         self.decay = decay
#     def setup(self, trainer, pl_module, stage):
#         # called once before anything
#         self.ema = ExponentialMovingAverage(pl_module, decay=self.decay)
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.ema.apply()
#     def on_validation_start(self, trainer, pl_module):
#         self.ema.replace_with_ema()
#     def on_validation_end(self, trainer, pl_module):
#         self.ema.swap()

from lightning.pytorch.callbacks import Callback

class EMACallback(Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_params: dict[str, torch.Tensor] = {}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Register each param once
        for name, p in pl_module.named_parameters():
            if name not in self.ema_params:
                # clone and detach so we don’t track grads
                self.ema_params[name] = p.data.clone().detach()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update EMA in‐place: ema = decay*ema + (1-decay)*current
        for name, p in pl_module.named_parameters():
            ema = self.ema_params[name]
            ema.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
