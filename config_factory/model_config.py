from dataclasses import dataclass
from .base import Config
import ray

@Config.register("model_config")
@dataclass
class ModelConfig(Config):
    algorithm     : str = "DummySTPP"
    hidden_size   : int = 128
    num_layers    : int = 2
    dropout       : float = 0.0
    lr            : float = 1e-3
    def __post_init__(self):
        #field level validation. TODO: Add all relevant validation steps
        if self.hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if self.num_layers <= 0:
            raise ValueError("Number of layers must be positive.")
        if not (0 <= self.dropout < 1):
            raise ValueError("Dropout must be in [0, 1).")
        if not (0 < self.lr < 1):
            raise ValueError("Learning rate must be in (0, 1).")
    # let Ray tune two fields
    @staticmethod
    def ray_space():
        from ray import tune
        return {
            "lr"         : tune.loguniform(1e-5, 1e-2),
            "hidden_size": tune.choice([32, 64, 128, 256])
        }
