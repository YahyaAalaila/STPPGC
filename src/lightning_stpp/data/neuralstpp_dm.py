from base import LightDataModule
from lib.neural_stpp.iterators import EpochBatchIterator
import torch
def spatiotemporal_events_collate_fn(batch):
    if len(batch) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = batch[0].shape[1]
    lengths = [seq.shape[0] for seq in batch]
    max_len = max(lengths)
    padded_seqs = [torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0) if s.shape[0] != max_len else s for s in data]
    data = torch.stack(padded_seqs, dim=0)
    event_times = data[:, :, 0]
    spatial_locations = data[:, :, 1:]
    mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])
    return event_times, spatial_locations, mask

class NeuralSTPPDataModule(LightDataModule):
    def __init__(self, data_config):
        super.__init__()
        self.config = data_config
        self.load_data_from_config()
        
    def train_dataloader(self):
        #This is specific to NeuralSTPP
        train_epoch_iter = EpochBatchIterator(
            dataset=self.traning_set,
            collate_fn= spatiotemporal_events_collate_fn,
            batch_sampler=self.traning_set.batch_by_size(self.config.max_events),
            seed=self.config.seed + self.config.rank,
            )
        return train_epoch_iter
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=self.config.test_bsz,
            shuffle=False,
            collate_fn=spatiotemporal_events_collate_fn,
        )
        return val_loader
    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.testing_set,
            batch_size=self.config.test_bsz,
            shuffle=False,
            collate_fn=spatiotemporal_events_collate_fn,
        )
        return test_loader