# data_config.py
from dataclasses import dataclass
from pathlib import Path
from ._config import Config

@Config.register("data_config")
@dataclass
class DataConfig(Config):
    name : str
    # path         : str
    # batch_size   : int = 128
    # max_events   : int = 1024   # used by Neural‑STPP sampler
    # def __post_init__(self):
    #     #field level validation. TODO: Add all relevant validation steps
    #     if self.batch_size <= 0:
    #         raise ValueError("Batch size must be positive.")
    #     if self.max_events <= 0:
    #         raise ValueError("Max events must be positive.")
        
    #     if not Path(self.path).exists():
    #         raise FileNotFoundError(f"Dataset path {self.path} does not exist.")
