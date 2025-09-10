from __future__ import annotations
from typing import Any, Tuple

import torch
from torch import Tensor, optim

from lightning_stpp.config_factory import Config, DataConfig
from .base import BaseSTPPModule

# Import the raw implementation (from where you place AutoIntSTPointProcess)
from integration.autointstpp import AutoIntSTPointProcess  


@BaseSTPPModule.register(name="autostpp")
class AutoSTPP(BaseSTPPModule):
    """
    Lightning wrapper that adapts the raw AutoIntSTPointProcess model 
    into the unified STPP training pipeline.
    """

    def __init__(self, model_cfg: Config, data_cfg: DataConfig):
        """
        Args
        ----
        model_cfg : Hydra config subtree `model=autostpp.yaml`
        data_cfg  : Dataset config (seq_len, scaling, etc.)
        """
        super().__init__(model_cfg, data_cfg)
        self.model_cfg = model_cfg

        # Instantiate the raw model
        self.stpp = AutoIntSTPointProcess(
            n_prodnet=model_cfg.n_prodnet,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers,
            activation=model_cfg.activation,
            bias=model_cfg.bias,
            trunc=model_cfg.trunc,
        )

    def forward(self, st_x: Tensor, st_y: Tensor) -> Tuple[Any, ...]:
        """ Forward pass through raw model. """
        return self.stpp(st_x, st_y)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], stage: str):
        """
        Shared logic for training/validation/test steps.
        """
        st_x, st_y, *rest = batch  # (B, L, 3) & (B, 1, 3)
        nelbo, sll, tll = self.stpp(st_x, st_y)  # raw loss returns

        # Log metrics
        self.log(f"{stage}/nelbo", nelbo, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=st_x.size(0))
        self.log(f"{stage}/sll", sll.mean(), on_step=True, on_epoch=True,
                 batch_size=st_x.size(0))
        self.log(f"{stage}/tll", tll.mean(), on_step=True, on_epoch=True,
                 batch_size=st_x.size(0))

        return nelbo

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        cfg = self.model_cfg

        if cfg.opt == "SGD":
            opt = optim.SGD(
                self.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay
            )
        else:  # default Adam
            opt = optim.Adam(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay
            )

        if cfg.scheduler is not None and cfg.scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.t_max)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt

    @classmethod
    def callbacks(cls):
        return []  # add model-specific callbacks if needed
