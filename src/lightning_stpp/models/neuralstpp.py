from .base import BaseSTPPModule
from lightning_stpp.utils.lr_schedules import lr_warmup_cosine
from lib.neural_stpp.models import JumpCNFSpatiotemporalModel

import torch
import torch.nn as nn

@BaseSTPPModule.register(name="neural_stpp")
class NeuralSTPP(BaseSTPPModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # in your application bootstrapping
        # auto_register_neural_stpp()
        self.t0 = cfg.t0
        self.t1 = cfg.t1
        self.net = cfg.build_model()
                
        # Split parameters for possible different treatment.
        self.params, self.attn_params = [], []
        for n, p in self.net.named_parameters():
            (self.attn_params if "self_attns" in n else self.regular_params).append(p)
            
        # constant time window (can also be YAML-driven)
        self.register_buffer("t0", torch.tensor([self.t0]))
        self.register_buffer("t1", torch.tensor([self.t1]))
                
        # In their implementation, they wrap self.model in DDP
        # in our case, this is taken care of by lightning.
        def forward(self, event_times, spatial_locations, input_mask):
            space_ll, time_ll = self.net(
                event_times, spatial_locations, input_mask,
                self.t0, self.t1
            )
            # normalise by number of events
            num_events = input_mask.sum(dim=1, keepdim=True).clamp_min(1)
            return space_ll / num_events, time_ll / num_events
        
        def training_step(self, batch, batch_idx):
            event_times, spatial_locations, input_mask = batch
            space_ll, time_ll = self(event_times, spatial_locations, input_mask)
            loss = -(space_ll + time_ll).mean()


            # log for callbacks / progress-bar
            self.log("train_space_ll", space_ll.mean(), on_step=True, on_epoch=True)
            self.log("train_time_ll", time_ll.mean(), on_step=True, on_epoch=True)
            self.log("train_loss", loss.mean(), on_step=True, on_epoch=True)
            return loss
        def validation_step(self, batch, batch_idx):
            event_times, spatial_locations, input_mask = batch
            space_ll, time_ll = self(event_times, spatial_locations, input_mask)
            loss = -(space_ll + time_ll).mean()


            # log for callbacks / progress-bar
            self.log("train_space_ll", space_ll.mean(), on_step=True, on_epoch=True)
            self.log("train_time_ll", time_ll.mean(), on_step=True, on_epoch=True)
            self.log("train_loss", loss.mean(), on_step=True, on_epoch=True)
            return loss
        def configure_optimizers(self):
            opt = torch.optim.AdamW(
                [
                    {"params": self.regular_params},
                    {"params": self.attn_params}
                ],
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.98)
            )
            # learning-rate schedule that exactly matches the original
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lambda itr: lr_warmup_cosine(
                    itr,
                    self.hparams.warmup_itrs,
                    self.hparams.lr,
                    self.hparams.total_iterations
                )
            )
            return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}
        
        def on_after_backward(self):
            # Log gradient norm after every backward.
            # This is implemented following neural stpp implementation where they log the gradient norm
            # after every backward pass.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.hparams.gradclip,
            )
            self.log("grad_norm", grad_norm, on_step=True)

        def on_before_optimizer_step(self, optimizer, optimizer_idx):
            # Log learning rate right before stepping
            # this is implemented following neural stpp implementation where they log the learning rate
            # before every optimizer step.
            # This is not the same as the learning rate at the end of the step.
            # The learning rate logged here is the one before the optimizer updates the parameters.
            lr = optimizer.param_groups[0]["lr"]
            self.log("lr", lr, on_step=True)
        
        @classmethod
        def callbacks(cls):
            # Only the callbacks that are specific to the model should be defined here.
            from lightning_stpp.callbacks.neural_stpp.ema import EMACallback
            return [EMACallback()]  # Add any other callbacks you need here.
        