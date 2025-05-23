from dataclasses import dataclass, field
from ._config import Config
import ray

@Config.register("hpo_config")
@dataclass
class HPOConfig(Config):
    hpo_id   : int = "ray_tune" 
    num_trials : int = 20
    results_dir: str = "./ray_results"
    experiment_name: str = "ray_tune_experiment"
    scheduler  : str = "asha"
    search_algorithm : str = "random"
    #max_t : int = 100
    resources : dict = field(default_factory=lambda: {
        "cpu": 4,
        "gpu": 1,
    })
    def __post_init__(self):
        # field level validation. TODO: Add all relevant validation steps
        if self.num_trials <= 0:
            raise ValueError("Number of trials must be positive.")
        if self.scheduler not in ["asha", "random", "bayesopt"]:
            raise ValueError("Scheduler must be one of ['asha', 'random', 'bayesopt'].")
        if len(self.resources)>=2:
            if self.resources["cpu"] <= 0 or self.resources["gpu"] <= 0:
                raise ValueError("CPU and GPU resources must be positive.")
        if self.hpo_id not in ["ray_tune"]: # If other HPO frameworks are added, then append in the list
            raise ValueError("Tuner ID must be one of ['ray_tune'].")
        
    def _dir_setup(self):
        
        import os
        sp = self.results_dir.strip()
        if sp:
            # convert a bare path into file://absolute/path
            abs_path = os.path.abspath(sp)
            self.results_dir = abs_path if abs_path.startswith(("file://","s3://")) else f"file://{abs_path}"
        else:
            self.results_dir = None  # let Ray pick ~/ray_results
        
        
    def make_scheduler(self, max_t: int):
        """
        Create a scheduler based on the specified type.
        Args:
            max_t (int): Maximum number of trials.
        Returns:
            Scheduler: A scheduler object based on the specified type.
        """
        
        if self.scheduler == "asha":
            # ASHA scheduler
            from ray.tune.schedulers import ASHAScheduler
            return ASHAScheduler(
                metric="val_loss",
                mode="min",
                max_t=max_t,
                grace_period=1,
                reduction_factor=2,
            )
        elif self.scheduler == "fifo":
            # FIFO scheduler
            from ray.tune.schedulers import FIFOScheduler
            return FIFOScheduler(
                metric="val_loss",
                mode="min",
            )
        elif self.scheduler == "hyperband":
            # Hyperband scheduler
            from ray.tune.schedulers import HyperBandScheduler
            return HyperBandScheduler(
                metric="val_loss",
                mode="min",
                max_t=max_t,
                grace_period=1,
                reduction_factor=3,
            )
        elif self.scheduler == "pbt":
            # Population Based Training (PBT) scheduler
            from ray.tune.schedulers import PopulationBasedTraining
            return PopulationBasedTraining(
                metric="val_loss",
                mode="min",
                perturbation_interval=10
            )
        elif self.scheduler == "median":
            # Median scheduler
            from ray.tune.schedulers import MedianStoppingRule
            return MedianStoppingRule(
                metric="val_loss",
                mode="min",
                patience=5,
            )
        raise ValueError(f"Unknown scheduler type: {self.scheduler}")
    
    def make_search_alg(self):
        """
        Create a search algorithm based on the specified type.
        Args:
            search_algorithm (str): The search algorithm to use.
        Returns:
            SearchAlgorithm: A search algorithm object based on the specified type.
        """
        if self.search_algorithm == "random":
            from ray.tune.search.basic_variant  import BasicVariantGenerator
            return BasicVariantGenerator()
        if self.search_algorithm == "hyperopt":
            from ray.tune.search.hyperopt  import HyperOptSearch
            return HyperOptSearch(metric="val_loss", mode="min")
        if self.search_algorithm == "optuna":
            from ray.tune.search.optuna import OptunaSearch
            return OptunaSearch(metric="val_loss", mode="min", sampler="TPE")
        if self.search_algorithm == "nevergrad":
            from ray.tune.search.nevergrad import NevergradSearch
            return NevergradSearch(metric="val_loss", mode="min")
        if self.search_algorithm == "bohb":
            from ray.tune.search.bohb import TuneBOHB
            return TuneBOHB(metric="val_loss", mode="min")
        if self.search_algorithm == "hebo":
            from ray.tune.search.hebo import HEBOSearch
            return HEBOSearch(metric="val_loss", mode="min")
        if self.search_algorithm == "ax":
            from ray.tune.search.ax import AxSearch
            return AxSearch(metric="val_loss", mode="min")
        raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")


