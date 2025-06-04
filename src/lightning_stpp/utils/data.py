from torch.utils.data import Dataset
import torch

class Float32Wrapper(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        # if your dataset yields (features, label)
        if isinstance(sample, tuple) and len(sample) == 2:
            x, y = sample
            return x.float(), y
        # else if it yields just a tensor:
        elif torch.is_tensor(sample):
            return sample.float()
        # fallback
        return sample
