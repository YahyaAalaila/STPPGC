import math
import numpy as np
import torch

from .base import LightDataModule
from smash.model.Dataset import EventData, collate_fn, collate_fn_mark

class _DIFFSeqDataset(torch.utils.data.Dataset):
    """
    Holds one list-of-sequences, applies Δt/log Δt augmentation and
    stores train-set min/max for later de-normalisation.
    """
    def __init__(self, seqs: list[np.ndarray], log_norm: bool):
        # augment every sequence
        def _augment(seq: list[list[float]], log_norm):
                out = []
                for k, e in enumerate(seq):
                    dt = e[0] if k == 0 else e[0] - seq[k-1][0]
                    if log_norm:
                        dt = math.log(max(dt, 1e-4))
                    out.append([e[0], dt, *(e[1:])])
                return out
        self.seqs = [_augment(np.asarray(s, dtype=np.float32), log_norm)
                     for s in seqs]
        

        # global min/max across ALL dims except the absolute-time column
        full = torch.tensor(np.concatenate(self.seqs, 0), dtype=torch.float32) 
        self.min = full.min(0).values       # [D+2]
        self.max = full.max(0).values
        
        

    def __len__(self):                return len(self.seqs)
    def __getitem__(self, idx):       
        sample = self.seqs[idx]
        if isinstance(sample, tuple) and len(sample) == 2:
            x, y = sample
            return x.float(), y
        # else if it yields just a tensor:
        elif torch.is_tensor(sample):
            return sample.float()
        return torch.tensor(sample, dtype=torch.float32)


@LightDataModule.register("diffstpp")
class DiffSTPPDataModule(LightDataModule):

    def __init__(self, data_cfg):
        super().__init__(data_cfg)

    def _wrap(self, split, reference_min=None, reference_max=None):
        ds = _DIFFSeqDataset(split, log_norm=self.data_config.log_normalisation)
        if reference_min is not None:   # share stats across splits
            ds.min, ds.max = reference_min, reference_max
        ds = EventData(ds)
        return ds

    def setup(self, stage=None):
        super().setup()
        self.training_set   = self._wrap(self.training_set)
        self.validation_set = self._wrap(self.validation_set)
        self.testing_set    = self._wrap(self.testing_set)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.data_config.val_bsz,
            collate_fn=(collate_fn_mark if self.data_config.ddim==3 else collate_fn), # TODO: This has to be better
            num_workers=self.data_config.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=self.data_config.val_bsz,
            collate_fn=(collate_fn_mark if self.data_config.ddim==3 else collate_fn),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testing_set,
            batch_size=self.data_config.test_bsz,
            collate_fn=(collate_fn_mark if self.data_config.ddim==3 else collate_fn),
        )
