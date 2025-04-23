

from .hpo_base import HyperTunner
from runner.runner_base import BaseRunner
import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# File: ray_tune_runner.py
from .base_hyper_tune import BaseHyperTune
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import os

@HyperTunner.register(name='ray_tune')
class RayTuneRunner(BaseHyperTune):
    def __init__(self, config):
        super().__init__(config)
        self.search_space = config.hpo.search_space
        self.num_samples = config.hpo.num_samples
        self.resources_per_trial = config.hpo.resources

    def _run_single_trial(self, trial_config):
        # Merge trial config with base config
        updated_config = self._update_config_for_trial(trial_config)

        # Inject Tune callback dynamically
        callbacks = updated_config.trainer.get("callbacks", [])
        callbacks.append(TuneReportCallback({"val_loss": "val_loss"}))
        updated_config.trainer.callbacks = callbacks

        # Build and run the base runner
        runner = BaseRunner.build_runner_from_config(updated_config)
        runner.trainer = pl.Trainer(**updated_config.trainer)  # Reuse existing config
        runner.run()  # Triggers training/validation

    def run(self):
        # Define trainable for Ray Tune
        trainable = tune.with_parameters(
            self._run_single_trial,
        )

        # Execute HPO
        analysis = tune.run(
            trainable,
            config=self.search_space,
            num_samples=self.num_samples,
            resources_per_trial=self.resources,
            local_dir=self.config.hpo.results_dir,
            name=self.config.hpo.experiment_name
        )

        print("Best config:", analysis.best_config)
        