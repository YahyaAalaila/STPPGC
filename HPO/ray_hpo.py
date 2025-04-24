import mlflow
from config_factory.model_config import ModelConfig
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
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def _search_space(self):
        
        space = {}
        space.update(self.hpo_config.model.ray_space())
        space.update(self.hpo_config.trainer.ray_space())
        return space

    def _run_single_trial(self, tune_params):
        
        # Clone and patch the specific fields that need to be tuned
        patched_model = self.hpo_config.model.clone(**{
            k:v for k,v in tune_params.items() if hasattr(ModelConfig, k)
            })
        # At the moment, unttached. Refer to the trainer_config if there is tunable parameters there
        patched_trainer = self.hpo_config.trainer 
        
        # Create a new config object with the patched model
        # this creates a copy of config without altering it, so it will be untouched for each new trial.
        trial_config = self.hpo_config.clone(
            model=patched_model,
            trainer=patched_trainer
        )
        run_name = f"trial_{tune.get_trial_id()}"
        # Open an MLflow run context in this trial as a child run of the parent run which is already active.
        # This is a nested (nested = true) run, so it will be a child of the parent run.
        with mlflow.start_run(
            run_name=run_name,
            run_id = None,
            experiment_id=None,
            nested=True,
            tags={"ray_trial": tune.get_trial_id()},
            description="Ray Tune trial") as child_run:
        
            # Build runner and attach tune callback
            runner = BaseRunner.build_runner_from_config(trial_config)
            
            # Now we need Lightning’s MLFlowLogger to use that exact run,
            # rather than implicitly starting its own new run under the same experiment.
            # The MLFlowLogger keeps its active run IDs in private attrs, so we override them:
            runner.mlflow_logger._run_id = child_run.info.run_id
            runner.mlflow_logger._experiment_id = child_run.info.experiment_id
            
            runner.trainer.callbacks.append(TuneReportCallback({"val_loss": "val_loss"}))
            runner.run()

    def run(self):
        # Define trainable for Ray Tune
        trainable = tune.with_parameters(
            self._run_single_trial,
        )
        # Stting up mlflow logger
        # Parent run
        with mlflow.start_run(run_name=self.hpo_config.experiment_name, nested=False) as parent_run: 
            # nested = Fasle to not try to make this ru a child of any already-active run in this process
            # to guarantee this is a top-level run.
            os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id # allow workes to know the parent run id 
            
            # Execute HPO
            analysis = tune.run(
                trainable,
                config=self._search_space(),
                num_samples=self.hpo_config.num_trial,
                scheduler = self.hpo_config.scheduler,
                resources_per_trial=self.hpo_config.resources,
                local_dir=self.hpo_config.results_dir,
                name=self.hpo_config.experiment_name,
                metric="val_loss",
                mode="min"
            )
                    # optionally log best results to parent
            mlflow.log_metric("best_val_loss", analysis.best_result["val_loss"])
            mlflow.log_params(analysis.best_config)

            print("Best config:", analysis.best_config)
            return analysis
        