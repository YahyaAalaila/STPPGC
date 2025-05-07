from functools import partial
import re
import numpy as np
import torch
from mhp import MHP
END_TIME = 30.0
### This is copied from NeuralSTPP imlementation and modified by adding methoids like (build_dataset_from_config)
# https://github.com/facebookresearch/neural_stpp/blob/main/datasets.py
# TODO: Build upon this to create a more general dataset class. (YA): Probably will be changed to a more general dataset class

# PS: This is a bit of a mess, but it works for now.
# PS: We can only use (generate from) pinwheel dataset for now, the rest work too, but we have to download the corresponding datasets
# and put them in the right folder.
def generate(mhp, data_fn, ndim, num_classes):
    mhp.generate_seq(END_TIME)
    event_times, classes = zip(*mhp.data)
    classes = np.concatenate(classes)
    n = len(event_times)

    data = data_fn(n)
    seq = np.zeros((n, ndim + 1))
    seq[:, 0] = event_times
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
        seq[:, 1:] = seq[:, 1:] + data_i * (i == classes)[:, None]
    return seq
def pinwheel(num_samples, num_classes):
    radial_std = 0.3
    tangential_std = 0.1
    num_per_class = num_samples
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * np.einsum("ti,tij->tj", features, rotations)


class STDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, test_set, train):
        self.S_mean, self.S_std = self._standardize(train_set)

        S_mean_ = torch.cat([torch.zeros(1, 1).to(self.S_mean), self.S_mean], dim=1)
        S_std_ = torch.cat([torch.ones(1, 1).to(self.S_std), self.S_std], dim=1)
        self.dataset = [(torch.tensor(seq) - S_mean_) / S_std_ for seq in (train_set if train else test_set)]
        
    @staticmethod
    def build_dataset_from_config(model_config):
        """
        Generates a dataset from the given configuration.
        This method should be overridden in subclasses to create specific datasets.
        """
        dataset_id = model_config.dataset_id
        for subclass in STDataset.__subclasses__():
            if dataset_id == subclass.__name__:
                return subclass
        # If not found, collect all available dataset IDs.
        available_ids = ", ".join(sub.__name__ for sub in STDataset.__subclasses__())
        raise ValueError(f"Dataset ID '{dataset_id}' not recognized. Available dataset IDs are: {available_ids}")

    def __len__(self):
        return len(self.dataset)

    def _standardize(self, dataset):
        dataset = [torch.tensor(seq) for seq in dataset]
        full = torch.cat(dataset, dim=0)
        S = full[:, 1:]
        S_mean = S.mean(0, keepdims=True)
        S_std = S.std(0, keepdims=True)
        return S_mean, S_std

    def unstandardize(self, spatial_locations):
        return spatial_locations * self.S_std + self.S_mean

    def ordered_indices(self):
        lengths = np.array([seq.shape[0] for seq in self.dataset])
        indices = np.argsort(lengths)
        return indices, lengths[indices]

    # def batch_by_size(self, max_events):
    #     try:
    #         from data_utils_fast import batch_by_size_fast
    #     except ImportError:
    #         raise ImportError('Please run `python setup.py build_ext --inplace`')

    #     indices, num_tokens = self.ordered_indices()

    #     if not isinstance(indices, np.ndarray):
    #         indices = np.fromiter(indices, dtype=np.int64, count=-1)
    #     num_tokens_fn = lambda i: num_tokens[i]

    #     return batch_by_size_fast(
    #         indices, num_tokens_fn, max_tokens=max_events, max_sentences=-1, bsz_mult=1,
    #     )

    def __getitem__(self, index):
        return self.dataset[index]
    
class PinwheelHawkes(STDataset):

    def __init__(self, split="train"):
        num_classes = 10
        m = np.array([0.05] * num_classes)
        a = np.diag([0.6] * (num_classes - 1), k=-1) + np.diag([0.6], k=num_classes - 1) + np.diag([0.0] * num_classes, k=0)
        w = 10.0

        mhp = MHP(mu=m, alpha=a, omega=w)
        num_train = 2000
        num_val = 200
        num_test = 200

        with np.random.seed(13579):
            data_fn = partial(pinwheel, num_classes=num_classes)
            train_set = [generate(mhp, data_fn, ndim=2, num_classes=num_classes) for _ in range(num_train)]
            val_set = [generate(mhp, data_fn, ndim=2, num_classes=num_classes) for _ in range(num_val)]
            test_set = [generate(mhp, data_fn, ndim=2, num_classes=num_classes) for _ in range(num_test)]

        split_set = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

        super().__init__(train_set, split_set[split], split == "split")
    
class Citibike(STDataset):

    splits = {
        "train": lambda f: bool(re.match(r"20190[4567]\d\d_\d\d\d", f)),
        "val": lambda f: bool(re.match(r"201908\d\d_\d\d\d", f)) and int(re.match(r"201908(\d\d)_\d\d\d", f).group(1)) <= 15,
        "test": lambda f: bool(re.match(r"201908\d\d_\d\d\d", f)) and int(re.match(r"201908(\d\d)_\d\d\d", f).group(1)) > 15,
    }

    def __init__(self, split="train"):
        assert split in self.splits.keys()
        self.split = split
        dataset = np.load("data/citibike/citibike.npz")
        train_set = [dataset[f] for f in dataset.files if self.splits["train"](f)]
        split_set = [dataset[f] for f in dataset.files if self.splits[split](f)]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class CovidNJ(STDataset):

    def __init__(self, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        dataset = np.load("data/covid19/covid_nj_cases.npz")
        dates = dict()
        for f in dataset.files:
            dates[f[:8]] = 1
        dates = list(dates.keys())

        # Reduce contamination between train/val/test splits.
        exclude_from_train = (dates[::27] + dates[1::27] + dates[2::27]
                              + dates[3::27] + dates[4::27] + dates[5::27]
                              + dates[6::27] + dates[7::27])
        val_dates = dates[2::27]
        test_dates = dates[5::27]
        train_dates = set(dates).difference(exclude_from_train)
        date_splits = {"train": train_dates, "val": val_dates, "test": test_dates}
        train_set = [dataset[f] for f in dataset.files if f[:8] in train_dates]
        split_set = [dataset[f] for f in dataset.files if f[:8] in date_splits[split]]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class Earthquakes(STDataset):

    def __init__(self, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        dataset = np.load("data/earthquakes/earthquakes_jp.npz")
        exclude_from_train = (dataset.files[::30] + dataset.files[1::30] + dataset.files[2::30] + dataset.files[3::30]
                              + dataset.files[4::30] + dataset.files[5::30] + dataset.files[6::30] + dataset.files[7::30]
                              + dataset.files[8::30] + dataset.files[9::30] + dataset.files[10::30])
        val_files = dataset.files[3::30]
        test_files = dataset.files[7::30]
        train_files = set(dataset.files).difference(exclude_from_train)
        file_splits = {"train": train_files, "val": val_files, "test": test_files}
        train_set = [dataset[f] for f in train_files]
        split_set = [dataset[f] for f in file_splits[split]]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class BOLD5000(STDataset):

    splits = {
        "train": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) < 8,
        "val": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) == 8,
        "test": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) > 8,
    }

    def __init__(self, split="train"):
        assert split in self.splits.keys()
        self.split = split
        dataset = np.load("data/bold5000/bold5000.npz")
        train_set = [dataset[f] for f in dataset.files if self.splits["train"](f)]
        split_set = [dataset[f] for f in dataset.files if self.splits[split](f)]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"