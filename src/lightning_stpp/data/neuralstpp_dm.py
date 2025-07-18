from .base import LightDataModule
from neural_stpp.iterators import EpochBatchIterator
import torch
from lightning.fabric.utilities.apply_func import apply_to_collection

def spatiotemporal_events_collate_fn(data):
    """Input is a list of tensors with shape (T, 1 + D)
        where T may be different for each tensor.

    Returns:
        event_times: (N, max_T)
        spatial_locations: (N, max_T, D)
        mask: (N, max_T)
    """
    if len(data) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0) if s.shape[0] != max_len else s for s in data]
    data = torch.stack(padded_seqs, dim=0)
    event_times = data[:, :, 0]
    spatial_locations = data[:, :, 1:]
    mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])
    return event_times, spatial_locations, mask.double()

@LightDataModule.register("neural_stpp")
@LightDataModule.register("neuralstpp")
class NeuralSTPPDataModule(LightDataModule):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        #self.load_data_from_config()
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_sampler = self.training_set.batch_by_size(
                self.data_config.max_events
            ),
            collate_fn    = spatiotemporal_events_collate_fn,
            num_workers   = self.data_config.num_workers,
            persistent_workers=True,   # keeps workers alive across epochs
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=self.data_config.val_bsz,
            shuffle=False,
            collate_fn=spatiotemporal_events_collate_fn,
        )
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testing_set,
            batch_size=self.data_config.test_bsz,
            shuffle=False,
            collate_fn=spatiotemporal_events_collate_fn,
        )
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # force every torch.Tensor in the batch to float32 and move it
        batch = apply_to_collection(
            batch,
            torch.Tensor,
            lambda t: t.to(device=device, dtype=torch.float32),
        )
        # 2) hand off to Lightning’s default implementation
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
