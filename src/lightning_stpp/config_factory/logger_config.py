
from dataclasses import dataclass
from pathlib import Path
from ._config import Config
@Config.register("logging_config")
@dataclass
class LoggingConfig(Config):
    mlflow_uri       : str = "file:./mlruns"
    experiment_name  : str = "default_exp"
    run_name         : str = "default_run"
    run_name_prefix : str = ""
    def finalize(self):
        """
        Ensure that mlflow_uri (whether 'file://…', './mlruns' or '/abs/path') 
        points to an existing directory. Strip any 'file:' scheme, then mkdir.
        """
        uri = self.mlflow_uri

        # 1) strip file:// or file: schemes
        if uri.startswith("file://"):
            path = uri[len("file://"):]
        elif uri.startswith("file:"):
            path = uri[len("file:"):]
        else:
            path = uri

        if not path:
            raise ValueError(f"logging.mlflow_uri resolves to an empty path: {uri!r}")

        # 2) resolve to a Path object (keep relative paths relative)
        mlruns_path = Path(path)

        # 3) create the directory if it doesn't already exist
        mlruns_path.mkdir(parents=True, exist_ok=True)

        # 4) store back the normalized path (this also helps MLflow)
        self.mlflow_uri = str(mlruns_path)
