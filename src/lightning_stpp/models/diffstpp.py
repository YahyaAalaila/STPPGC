# src/lightning_stpp/models/smash.py
from __future__ import annotations
from typing import Tuple, Any

import torch
from torch import Tensor, optim

from lightning_stpp.models.base import BaseSTPPModule
from lightning_stpp.config_factory import Config, DataConfig

from diffstpp.DSTPP.Models import Transformer_ST, Transformer
from diffstpp.DSTPP.DiffusionModel import GaussianDiffusion_ST, ST_Diffusion, Model_all
# --------------------------------------------------------------------
@BaseSTPPModule.register("diffstpp")
class SMASHSTPP(BaseSTPPModule):
    """
    Thin Lightning wrapper around the original DSTPP implementation.
    It exposes (forward, loss, optimiser) so your Runner can treat it like
    any other STPP model.
    """
    def __init__(self, model_cfg: Config, data_cfg: DataConfig):
        super().__init__(model_cfg, data_cfg)
        self.float()
        # ---------------- build the 3-part model -------------------
        self.transformer = self._transformer(model_cfg)
        core = self._diffusion(model_cfg)
        self.diffusion = self._Gauss_diffusion(core, model_cfg)

        self.model: Model_all = Model_all(self.transformer, self.diffusion)
        
        # helper that reproduces Batch2toModel without Python loops
    def _batch_to_model(self, batch: Tuple[torch.Tensor, ...]):
        # TODO: This HAS to be coded better in relationto "mark: yes/no?"
        # 1) exactly the same unpacking as Batch2toModel
        if self.data_cfg.ddim == 2:
            t_abs, dt, lng, lat = batch      # all (B, L)
            event_loc = torch.cat((
                lng.unsqueeze(-1),            # (B, L, 1)
                lat.unsqueeze(-1)             # (B, L, 1)
            ), dim=-1)                         # (B, L, 2)
        else:  # ddim == 3
            t_abs, dt, mark, lng, lat = batch
            event_loc = torch.cat((
                mark.unsqueeze(-1),
                lng.unsqueeze(-1),
                lat.unsqueeze(-1),
            ), dim=-1)                         # (B, L, 3)

        # 2) call the transformer on the full padded seq
        #    exactly as they do in Batch2toModel
        enc_out, mask = self.transformer(event_loc, t_abs)
        enc_out_non_mask  = []
        event_time_non_mask = []
        event_loc_non_mask = []
        for index in range(mask.shape[0]):
            length = int(sum(mask[index]).item())
            if length>1:
                enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length-1]]
                event_time_non_mask += [i.unsqueeze(dim=0) for i in dt[index][1:length]]
                event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]

        enc_out_non_mask = torch.cat(enc_out_non_mask,dim=0)
        event_time_non_mask = torch.cat(event_time_non_mask,dim=0)
        event_loc_non_mask = torch.cat(event_loc_non_mask,dim=0)

        event_time_non_mask = event_time_non_mask.reshape(-1,1,1)
        event_loc_non_mask = event_loc_non_mask.reshape(-1,1,2)
        
        enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0],1,-1)
        return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


    # ----------------------------------------------------------------
    # Lightning interface
    # ----------------------------------------------------------------
    def forward(self, st_x: Tensor, mask: Tensor) -> Any:
        """
        `st_x`  : [B, L,   1+dim]
        `mask`  : [B, L]  (True where valid)
        Returns whatever the raw SMASH forward returns.
        """
        enc_out, _ = self.transformer(st_x[..., 1:], st_x[..., 0])
        return self.diffusion(st_x, enc_out)

    # SMASH’s original loss only needs (seq, enc_out)
    def _shared_step(self, batch, stage: str):
        event_time_non_mask, event_loc_non_mask, enc_out_non_mask = self._batch_to_model(batch)
        seq_non = torch.cat((event_time_non_mask,event_loc_non_mask), dim=-1)
        loss = self.diffusion(seq_non, enc_out_non_mask)

        # Lightning logging
        self.log(f"{stage}_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=event_time_non_mask.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ----------------------------------------------------------------
    # Optimiser (AdamW + linear-warmup, cosine decay)
    # ----------------------------------------------------------------
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(),
                          lr=self.hparams.lr,
                          betas=(0.9, 0.99))
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.total_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
    
    @staticmethod
    def _transformer(model_cfg):
        return Transformer_ST(
            d_model   = model_cfg.cond_dim,
            d_rnn     = model_cfg.cond_dim * 4,
            d_inner   = model_cfg.cond_dim * 2,
            n_layers  = 4,
            n_head    = 4,
            d_k       = 16,
            d_v       = 16,
            dropout   = 0.1,
            loc_dim   = model_cfg.dim,
            CosSin    = True
        )

    @staticmethod
    def _diffusion(model_cfg):
        return ST_Diffusion(
            n_steps=model_cfg.timesteps,
            dim=1+model_cfg.dim,
            condition = True,
            cond_dim=64
    )
    
    @staticmethod
    def _Gauss_diffusion(core, model_cfg):
        return GaussianDiffusion_ST(
            core,
            loss_type = model_cfg.loss_type,
            seq_length = 1+model_cfg.dim,
            timesteps = model_cfg.timesteps,
            sampling_timesteps = model_cfg.samplingsteps,
            objective = model_cfg.objective,
            beta_schedule = model_cfg.beta_schedule
        )

    @classmethod
    def callbacks(cls):
        return []
