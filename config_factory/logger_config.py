
from dataclasses import dataclass
from .base import Config
@Config.register("logging_config")
@dataclass
class LoggingConfig(Config):
    mlflow_uri       : str = "file:./mlruns"
    experiment_name  : str = "default_exp"
    run_name         : str = "run"
