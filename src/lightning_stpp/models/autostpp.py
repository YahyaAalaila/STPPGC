from autoint_stpp.models.lightning.autoint_stpp import AutoIntSTPointProcess  
from lightning_stpp.config_factory import Config, DataConfig
from .base import BaseSTPPModule
import torch
from dataclasses import asdict


@BaseSTPPModule.register("autostpp")
class AutoSTPP(BaseSTPPModule):
    def __init__(self, model_cfg: Config, data_cfg: DataConfig):
        super().__init__(model_cfg, data_cfg)
        self.model_cfg = model_cfg
        self.stpp = AutoIntSTPointProcess(self.model_cfg)


    def forward(self, batch):
        st_x, st_y,  = batch
        return self.stpp(st_x, st_y)

    def _shared_step(self, batch, stage: str):
        st_x, st_y, *rest = batch
        nll, sll, tll = self.forward((st_x, st_y))
        self.log(f"{stage}/nll", nll, on_step=True, on_epoch=True, prog_bar=True, batch_size=st_x.size(0))
        self.log(f"{stage}/sll", sll, on_step=True, on_epoch=True, batch_size=st_x.size(0))
        self.log(f"{stage}/tll", tll, on_step=True, on_epoch=True, batch_size=st_x.size(0))
        return nll

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        cfg = self.model_cfg
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        if getattr(cfg, "scheduler", None) is not None:
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt


    @classmethod
    def callbacks(cls):
        # Only the callbacks that are specific to the model should be defined here.
        return []