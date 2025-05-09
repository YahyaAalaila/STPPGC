
from dataclasses import dataclass
from ._config import Config
@Config.register("logging_config")
@dataclass
class LoggingConfig(Config):
    mlflow_uri       : str = "file:./mlruns"
    experiment_name  : str = "default_exp"
    run_name_prefix : str = ""
