import math
from .base import BaseSTPPModule
from lightning_stpp.utils.lr_schedules import lr_warmup_cosine
from lib.neural_stpp.models import JumpCNFSpatiotemporalModel

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

@BaseSTPPModule.register(name="neural_stpp")
class NeuralSTPP(BaseSTPPModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.t0 = cfg.t0
        self.t1 = cfg.t1
        self.net = cfg.build_model()
        self.float()
                
        # Split parameters for possible different treatment.
        self.regular_params, self.attn_params = [], []
        for n, p in self.net.named_parameters():
            (self.attn_params if "self_attns" in n else self.regular_params).append(p)
        
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
        
        space_ll_sum = space_ll.sum(dim=1) 
        time_ll_sum  = time_ll.sum(dim=1)    

        
        loss = -(space_ll_sum + time_ll_sum).mean()

        # log for callbacks / progress-bar
        self.log("train_space_ll", space_ll.mean(), on_step=True, on_epoch=True)
        self.log("train_time_ll", time_ll.mean(), on_step=True, on_epoch=True)
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        event_times, spatial_locations, input_mask = batch
        space_ll, time_ll = self(event_times, spatial_locations, input_mask)
        # 1) sum across the event/time dimension for each sample
        space_ll_sum = space_ll.sum(dim=1)   # shape: (batch_size,)
        time_ll_sum  = time_ll.sum(dim=1)    # shape: (batch_size,)

        # 2) add per‐sample and average
        loss = -(space_ll_sum + time_ll_sum).mean()
    
        # log for callbacks / progress-bar
        self.log("val_space_ll", space_ll.mean(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_time_ll", time_ll.mean(), on_step=True, on_epoch=True)
        self.log("val_loss", loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        
        N = len(self.trainer.datamodule.training_set)
        bs = self.model_cfg.data_loaders["train_bsz"]
        batches_per_epoch = math.ceil(N / bs)

        num_iters = self.hparams.num_iterations
        total = math.ceil(num_iters / batches_per_epoch) * batches_per_epoch

        self.hparams.total_iterations = total
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

    def on_before_optimizer_step(self, optimizer):
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
        return [EMACallback()]  # TODO: Add any other callbacks you need here.
        