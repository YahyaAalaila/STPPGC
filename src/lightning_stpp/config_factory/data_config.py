# data_config.py
from dataclasses import dataclass
from pathlib import Path
from ._config import Config

@Config.register("data_config")
@dataclass
class DataConfig(Config):
    dataset_id : str
    #num_marks: int
    batch_size: int = 128
    max_events: int = 1024
    log_normalisation: float = 1
    ddim: int = 3
    # FIXME: Add more dataset specific parameters and decide what to put in 
    train_bsz:  int = 16
    val_bsz:    int = 64
    test_bsz:   int = 64
    num_workers: int = 8
    max_events: int | None = 4096      # optional event-count sampler
    
    def __post_init__(self):
        if self.ddim not in [1, 2, 3]:
            ValueError("Number of dim must be in [1, 2, 3].")
            
