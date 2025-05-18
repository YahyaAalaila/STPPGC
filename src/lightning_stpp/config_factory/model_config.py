from dataclasses import dataclass, field
from typing import Any, Dict
from ._config import Config
import ray
from ._parsing import parse_tune_dsl

    
   
@Config.register("neuralstpp")   
@dataclass      
class NeuralSTPPConfig(Config):
    """
    Configuration class for NeuralSTPP models.
    Inherits from ModelConfig and adds additional parameters specific to NeuralSTPP.
    """

    model_id          : str = "jump-cnf"  # or "att-cnf", "cond_gmm"
    dim               : int = 2
    t0               : float = 0.0
    t1               : float = 1.0
    hdims              : list[int] = field(default_factory=lambda: [64, 64, 64])
    tpp_hidden_dims   : list[int] = field(default_factory=lambda: [8, 20])
    actfn             : str      = "softplus"
    tpp_cond          : bool     = False
    tpp_style         : str      = "split"
    tpp_actfn         : str      = "softplus"
    share_hidden      : bool     = False
    solve_reverse     : bool     = False
    l2_attn           : bool     = False
    tol               : float    = 1e-6
    otreg_strength    : float    = 0.0
    tpp_otreg_strength: float    = 0.0
    layer_type        : str      = "concat"
    naive_hutch       : bool     = False
    lowvar_trace      : bool     = False
    search_space      : Dict[str, Any] = field(default_factory=dict)
    
    lr                : float    = 1e-3
    warmup_itrs      : int      = 1000
    num_iterations     : int      = 10000
    momentum           : float    = 0.9
    weight_decay       : float    = 0.0
    gradclip           : float    = 1.0
    max_events         : int      = 1000
    
    def __post_init__(self):
        if self.model_id not in ["jump-cnf", "att-cnf", "cond-gmm"]:
            raise ValueError(f"Unknown NeuralSTPP variant {self.model_id!r}")
        # Additional validation or preparation can be done here if needed
        if not isinstance(self.hdims, list):
            raise TypeError(f"hdims must be a list, got {type(self.hdims).__name__}")
        if not isinstance(self.tpp_hidden_dims, list):
            raise TypeError(f"tpp_hidden_dims must be a list, got {type(self.tpp_hidden_dims).__name__}")
        # add more
    @classmethod
    def parse_from_yaml_config(cls, raw:dict):
        return cls(**raw)
    def prepare_kwargs(self):
        self.shared_model_kwargs = {
            "dim"               : self.dim,
            "hidden_dims"       : self.hdims,        # from ModelConfig
            "tpp_hidden_dims"   : self.tpp_hidden_dims,
            "actfn"             : self.actfn,
            "tpp_cond"          : self.tpp_cond,
            "tpp_style"         : self.tpp_style,
            "tpp_actfn"         : self.tpp_actfn,
            "share_hidden"      : self.share_hidden,
            "tol"               : self.tol,
            "otreg_strength"    : self.otreg_strength,
            "tpp_otreg_strength": self.tpp_otreg_strength,
            "layer_type"        : self.layer_type,
        }
        
    def ray_space(self):
        from ray import tune
        if self.search_space:
            return parse_tune_dsl(self.search_space)
        return {}
            
    def build_model(self):
        
        self.prepare_kwargs() # prepare shared kwargs
        
        if self.model_id == "jump-cnf":
            from lib.neural_stpp.models import JumpCNFSpatiotemporalModel
            kwargs = self.shared_model_kwargs.copy()
            kwargs["solve_reverse"] = self.solve_reverse
            return JumpCNFSpatiotemporalModel(**kwargs)
        elif self.model_id == "att-cnf":
            from lib.neural_stpp.models import SelfAttentiveCNFSpatiotemporalModel
            kwargs = self.shared_model_kwargs.copy()
            kwargs["solve_reverse"] = self.solve_reverse
            kwargs["l2_attn"]     = self.l2_attn
            kwargs["lowvar_trace"] = not self.naive_hutch
            return SelfAttentiveCNFSpatiotemporalModel(**kwargs)
        elif self.model_id == "cond-gmm":
            from lib.neural_stpp.models import JumpGMMSpatiotemporalModel # type: ignore
            return JumpGMMSpatiotemporalModel(**self.shared_model_kwargs)
        else:
            raise ValueError(f"Unknown NeuralSTPP variant {self.model_id!r}")
