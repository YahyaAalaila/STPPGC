from pprint import pprint
import mlflow
from ray import tune
import os
from ray.tune.context import get_context

from lightning_stpp.config_factory import NeuralSTPPConfig
from lightning_stpp.HPO import HyperTuner
from lightning_stpp.runner import BaseRunner


@HyperTuner.register(name='ray_tune')
class RayTuneRunner(HyperTuner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.runner_config = cfg
        
    def _search_space(self):
        space = {}
        space.update(self.model_config.ray_space())
        space.update(self.trainer_config.ray_space())
        if space is None:
            raise ValueError("No hyperparameters to tune. Please check your configuration. If you are not tuning any hyperparameters, please use Runner instead of RayTuneRunner.")
        return space

    def _run_single_trial(self, tune_params):
        
        # Clone and patch the specific fields that need to be tuned
        patched_model = self.model_config.clone(**{
            k:v for k,v in tune_params.items() if hasattr(NeuralSTPPConfig, k)
            })
        # At the moment, unttached. Refer to the trainer_config if there is tunable parameters there
        patched_trainer = self.trainer_config
        
        # Create a new config object with the patched model
        # this creates a copy of config without altering it, so it will be untouched for each new trial.
        trial_config = self.runner_config.clone(
            model=patched_model,
            trainer=patched_trainer
        )
        run_name = f"trial_{get_context().get_trial_id()}"
        # Open an MLflow run context in this trial as a child run of the parent run which is already active.
        # This is a nested (nested = true) run, so it will be a child of the parent run.
        with mlflow.start_run(
            run_name=run_name,
            run_id = None,
            experiment_id=None,
            nested=True,
            tags={"ray_trial": get_context().get_trial_id()},
            description="Ray Tune trial") as child_run:
            # Build runner and attach tune callback
            runner = BaseRunner.build_runner_from_config(trial_config)
            
            # Now we need Lightning’s MLFlowLogger to use that exact run,
            # rather than implicitly starting its own new run under the same experiment.
            # The MLFlowLogger keeps its active run IDs in private attrs, so we override them:
            runner.mlflow_logger._run_id = child_run.info.run_id
            runner.mlflow_logger._experiment_id = child_run.info.experiment_id
            runner.fit()

    def run(self):
        #Define trainable for Ray Tune
        trainable = tune.with_parameters(
            self._run_single_trial,
        )
        # Stting up mlflow logger
        # Parent run
        with mlflow.start_run(run_name=self.hpo_config.experiment_name, nested=False) as parent_run: 
            # nested = Fasle to not try to make this ru a child of any already-active run in this process
            # to guarantee this is a top-level run.
            os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id # allow workes to know the parent run id 
                       # print("Best config:", analysis.best_config)
            search_space = self._search_space()  
            # Execute HPO
            analysis = tune.run(
                trainable,
                config=self._search_space(),
                num_samples=self.hpo_config.num_trials,
                search_alg = self.hpo_config.make_search_alg(
                    tuning_mtric = self.runner_config.model.monitor,
                    tuning_mode = self.runner_config.model.monitor_mode
                    ),
                scheduler = self.hpo_config.make_scheduler(
                    tuning_mtric = self.runner_config.model.monitor,
                    tuning_mode = self.runner_config.model.monitor_mode,
                    max_train_it = self.runner_config.trainer.max_epochs
                    ),
                resources_per_trial=self.hpo_config.resources,
                storage_path=self.hpo_config._dir_setup(),
                max_concurrent_trials=1, # TODO: MAke this somewhat automatic based on available ressources
                name=self.hpo_config.experiment_name,
                verbose=1
            )
            
            
            best_trial = analysis.get_best_trial(
                metric=self.runner_config.model.monitor,
                mode=self.runner_config.model.monitor_mode
                )
            

            print("🕵️‍♂️ best_trial.last_result keys:")
            pprint(best_trial.last_result.keys())
            print("🕵️‍♂️ full last_result dict:")
            pprint(best_trial.last_result)
            best_val = best_trial.last_result[self.runner_config.model.monitor]          

            # Get the config directly from the best_trial object you found
            mlflow.log_params(best_trial.config)

            mlflow.log_metric("best_val_loss", best_val)
            return analysis
        