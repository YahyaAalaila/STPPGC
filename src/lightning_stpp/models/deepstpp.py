# src/adapters/deep_stpp_adapter.py
from __future__ import annotations
from typing import Any, Tuple

from lightning_stpp.config_factory import Config, DataConfig
from .base import BaseSTPPModule

from torch import Tensor, optim

from deepstpp.model import DeepSTPP as deepstpp  # the repo you linked

@BaseSTPPModule.register(name="deep_stpp")
@BaseSTPPModule.register("deepstpp") 
class DeepSTPP(BaseSTPPModule):
    """
    Lightning wrapper that makes the raw DeepSTPP model look like the rest of
    your experiments (forward, loss, metrics, optimiser, etc.).
    """
    def __init__(self, model_cfg: Config, data_cfg: DataConfig):
        """
        Args
        ----
        cfg : Hydra config subtree `model=deep_stpp.yaml`
              ├─ emb_dim, z_dim, hid_dim …
              ├─ seq_len
              ├─ lr, weight_decay
              └─ beta, sample, constrain_b …
        """
        super().__init__(model_cfg, data_cfg)
        self.model_cfg = model_cfg
        # Instantiate the raw model on the current device
        self.stpp: deepstpp = deepstpp(self.model_cfg)


    def forward(self, st_x: Tensor) -> Tuple[Any, ...]:
        """ Proxy to the underlying NN; rarely used directly. """
        return self.stpp(st_x)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], stage: str):
        st_x, st_y, *rest = batch                              # (B, L, 3) & (B, 1, 3)
        nelbo, sll, tll = self.stpp.loss(st_x, st_y)     # repo’s loss:contentReference[oaicite:0]{index=0}

        # Log with Lightning’s built-in helpers
        self.log(f"{stage}/nelbo",
                nelbo,
                on_step=True, on_epoch=True, prog_bar=True,
                batch_size=st_x.size(0))
        
        self.log(f"{stage}/sll",
                sll.mean(),
                on_step=True, on_epoch=True,
                batch_size=st_x.size(0))
        
        self.log(f"{stage}/tll",
                tll.mean(),
                on_step=True, on_epoch=True,
                batch_size=st_x.size(0))

        return nelbo

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_step(batch, "test")


    def configure_optimizers(self):
        cfg = self.model_cfg  # saved by `save_hyperparameters`

        opt: optim.Optimizer
        if cfg.opt == "SGD":
            opt = optim.SGD(self.parameters(),
                            lr=cfg.lr,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)
        else:  # default = Adam
            opt = optim.Adam(self.parameters(),
                             lr=cfg.lr,
                             weight_decay=cfg.weight_decay)

        # One-liner for optional scheduler (e.g. cosine decay)
        if cfg.scheduler is not None and cfg.scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.t_max)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
    
    @classmethod
    def callbacks(cls):
        # Only the callbacks that are specific to the model should be defined here.
        return []  # TODO: Add any other callbacks you need here.
