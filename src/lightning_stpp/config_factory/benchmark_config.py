# config_factory/benchmark_config.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib     import Path
from typing      import List

import pytorch_lightning as pl

from ._config          import Config
from .runner_config import RunnerConfig
from .logger_config import LoggingConfig         # ➊ small helper dataclass (see note)
from .hypertuning_config import HPOConfig       # ➋ small helper dataclass (see note)

# ────────────────────────────────────────────────────────────────────────────
@Config.register("benchmark_config")
@dataclass(repr=False)
class BenchmarkConfig(Config):
    """
    A *benchmark* = one dataset + N experiments (each is a RunnerConfig).

    • `logging`  is global  → experiment-level configs inherit it automatically
    • `out_dir`  is the root for mlflow folders, checkpoints, summary.json, …
    """

    # required ----------------------------------------------------------------
    dataset   : str                       # e.g. "pinwheel", "citibike"
    logging   : LoggingConfig             # experiment_name / mlflow_uri live here
    experiments : List[RunnerConfig] = field(default_factory=list)
    hpo_defaults : HPOConfig |None = None  # default HPO config for all experiments

    # optional globals --------------------------------------------------------
    n_workers : int  = 0                  # 0 → use half the CPUs
    seed      : int  = 42
    out_dir   : str  = ""          # everything ends up under this folder

    # ──────────────────────────── sanity & inheritance ──────────────────────
    def finalize(self) -> None:
        
        """Lightweight checks + propagate shared fields down to each experiment."""
        if not self.experiments:
            raise ValueError("benchmark.experiments list is empty")
        
        if not self.out_dir:                         # user did not provide it
            self.out_dir = f"results/{self.dataset}"
            
        if self.hpo_defaults is not None and isinstance(self.hpo_defaults, dict):
            self.hpo_defaults = HPOConfig(**self.hpo_defaults)
        # ensure out_dir exists once
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        
        # Inject dataset name into every experiment
        fixed = []
        for exp in self.experiments:           # exp is a dict
            exp = {**exp}                      # shallow‑copy
            # add minimal data section if missing
            if "data" not in exp:
                exp["data"] = {"name": self.dataset}
            else:
                exp["data"]["name"] = self.dataset
            fixed.append(RunnerConfig.from_dict(exp))
        self.experiments = fixed

        # propagate dataset + logging + paths to every RunnerConfig
        for i, exp in enumerate(self.experiments):
            if isinstance(exp, dict):
                exp = RunnerConfig.from_dict(exp)
            # ------- dataset ----------
            if exp.data.name not in ("", self.dataset):
                raise ValueError(
                    f"Experiment #{i} asks for data={exp.data.name!r} "
                    f"but benchmark.dataset={self.dataset!r}"
                )
            exp.data = exp.data.clone(name=self.dataset)

            # ------- logging ----------
            eff_log = exp.logging or self.logging        # inherit if missing
            # put runs under benchmark/out_dir/mlruns/…
            eff_uri = Path(self.out_dir) / "mlruns"
            eff_log = eff_log.clone(mlflow_uri=f"file:{eff_uri}")

            exp.logging = eff_log

            # ------- checkpoints -------
            ckpt_root = Path(self.out_dir) / "ckpts" / exp.model.model_id
            exp.trainer = exp.trainer.clone(ckpt_dir=str(ckpt_root))

            # write back
            self.experiments[i] = exp
            
            #------- HPO defaults -------
            if exp.hpo is None:              # inherits defaults intact
                exp.hpo = self.hpo_defaults.clone()
            elif isinstance(exp.hpo, dict):                    # override missing keys only
                exp.hpo = HPOConfig(**{**self.hpo_defaults.to_dict(), **exp.hpo})


        # global seed – makes PL, numpy & torch deterministic
        pl.seed_everything(self.seed)

    # ---------- Ray Tune helper (aggregated search-space if ever needed) -----
    @staticmethod
    def ray_space() -> dict:
        return {}
