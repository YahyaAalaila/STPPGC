from dataclasses import dataclass, field
from .base import Config


@Config.register("hpo_config")
@dataclass
class HPOConfig(Config):
    tuner_id   : int = "ray_tune" 
    num_trials : int = 20
    results_dir: str = "./ray_results"
    experiment_name: str = "ray_tune_experiment"
    scheduler  : str = "asha"
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
        if self.resources["cpu"] <= 0 or self.resources["gpu"] <= 0:
            raise ValueError("CPU and GPU resources must be positive.")
        if self.tuner_id not in ["ray_tune"]: # If other HPO frameworks are added, then append in the list
            raise ValueError("Tuner ID must be one of ['ray_tune'].")
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
                metric="loss",
                mode="min",
                max_t=max_t,
                grace_period=1,
                reduction_factor=2,
            )
        elif self.scheduler == "fifo":
            # FIFO scheduler
            from ray.tune.schedulers import FIFOScheduler
            return FIFOScheduler(
                metric="loss",
                mode="min",
            )
        elif self.scheduler == "hyperband":
            # Hyperband scheduler
            from ray.tune.schedulers import HyperBandScheduler
            return HyperBandScheduler(
                metric="loss",
                mode="min",
                max_t=max_t,
                grace_period=1,
                reduction_factor=3,
            )
        elif self.scheduler == "pbt":
            # Population Based Training (PBT) scheduler
            from ray.tune.schedulers import PopulationBasedTraining
            return PopulationBasedTraining(
                metric="loss",
                mode="min",
                perturbation_interval=10,
                hyperparam_mutations={
                    "lr": [1e-4, 1e-3, 1e-2],
                    "batch_size": [16, 32, 64],
                },
            )
        elif self.scheduler == "median":
            # Median scheduler
            from ray.tune.schedulers import MedianStoppingRule
            return MedianStoppingRule(
                metric="loss",
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
            from ray.tune.suggest import BasicVariantGenerator
            return BasicVariantGenerator()
        if self.search_algorithm == "hyperopt":
            from ray.tune.suggest.hyperopt import HyperOptSearch
            return HyperOptSearch(metric="val_loss", mode="min")
        if self.search_algorithm == "optuna":
            from ray.tune.suggest.optuna import OptunaSearch
            return OptunaSearch(metric="val_loss", mode="min", sampler="TPE")
        if self.search_algorithm == "bayesopt":
            from ray.tune.suggest.skopt import SkOptSearch
            return SkOptSearch(metric="val_loss", mode="min")
        if self.search_algorithm == "nevergrad":
            from ray.tune.suggest.nevergrad import NevergradSearch
            return NevergradSearch(metric="val_loss", mode="min")
        if self.search_algorithm == "dragonfly":
            from ray.tune.suggest.dragonfly import DragonflySearch
            return DragonflySearch(metric="val_loss", mode="min")
        if self.search_algorithm == "ax":
            from ray.tune.suggest.ax import AXSearch
            return AXSearch(metric="val_loss", mode="min")
        raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")


