from dataclasses import dataclass, field
from typing import Any, Dict

from ._config import BaseModelConfig
from ._parsing import split_search_space
# TODO: take a look at the modelbase config and why there is no super().__post_init__
# --------- NeuralSTPPConfig --------- #
# This is a configuration class for the NeuralSTPP model, which is a specific type of
# spatiotemporal point process model. It inherits from the base ModelConfig and includes
# various parameters that define the model architecture, training specifics, and regularization settings.
   
@BaseModelConfig.register("neuralstpp")   
@dataclass      
class NeuralSTPPConfig(BaseModelConfig):
    """
    Configuration class for NeuralSTPP models.
    Inherits from ModelConfig and adds additional parameters specific to NeuralSTPP.
    """
    model_id          : str = None
    model_sub_id      : str = "jump-cnf"  # or "att-cnf", "cond_gmm"
    dim               : int = 2
    t0                : float = 0.0
    t1                : float = 50.0
    hdims             : list[int] = field(default_factory=lambda: [64, 64, 64])
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
    
    #lr                : float    = 1e-3
    warmup_itrs      : int      = 1000
    num_iterations     : int      = 10000
    #momentum           : float    = 0.9
    #weight_decay       : float    = 0.0
    gradclip           : float    = 1.0
    max_events         : int      = 1000
    data_loaders      : Dict[str, Any] = field(default_factory=dict)
    # ---------- internal ----------
    _ray_tune_space: dict = field(init=False, repr=False, default_factory=dict)

    
    def __post_init__(self):
        if self.model_sub_id not in ["jump-cnf", "att-cnf", "cond-gmm"]:
            raise ValueError(f"Unknown NeuralSTPP variant {self.model_id!r}")
        # Additional validation or preparation can be done here if needed
        if not isinstance(self.hdims, list):
            raise TypeError(f"hdims must be a list, got {type(self.hdims).__name__}")
        if not isinstance(self.tpp_hidden_dims, list):
            raise TypeError(f"tpp_hidden_dims must be a list, got {type(self.tpp_hidden_dims).__name__}")
        # add more
        
        if self.search_space:
            tunables, constants = split_search_space(self.search_space, type(self))
            for k, v in constants.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._ray_tune_space = tunables
            self.search_space = None        # optional tidy-up
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
        return self._ray_tune_space
            
    def build_model(self):
        
        self.prepare_kwargs() # prepare shared kwargs
        
        if self.model_sub_id == "jump-cnf":
            from neural_stpp.models import JumpCNFSpatiotemporalModel
            kwargs = self.shared_model_kwargs.copy()
            kwargs["solve_reverse"] = self.solve_reverse
            return JumpCNFSpatiotemporalModel(**kwargs).float()
        elif self.model_sub_id == "att-cnf":
            from neural_stpp.models import SelfAttentiveCNFSpatiotemporalModel
            kwargs = self.shared_model_kwargs.copy()
            kwargs["solve_reverse"] = self.solve_reverse
            kwargs["l2_attn"]     = self.l2_attn
            kwargs["lowvar_trace"] = not self.naive_hutch
            return SelfAttentiveCNFSpatiotemporalModel(**kwargs).float()
        elif self.model_sub_id == "cond-gmm":
            from neural_stpp.models import JumpGMMSpatiotemporalModel # type: ignore
            return JumpGMMSpatiotemporalModel(**self.shared_model_kwargs).float()
        else:
            raise ValueError(f"Unknown NeuralSTPP variant {self.model_sub_id!r}")
        
        
# --------- DeepSTPPConfig --------- #
# This is a configuration class for the DeepSTPP model, which is a specific type of
# spatiotemporal point process model. It inherits from the base Config class and
# includes various parameters that define the model architecture, inference specifics,
# training and evaluation hooks, and regularization settings.

@BaseModelConfig.register("deepstpp")
@dataclass
class DeepSTPPConfig(BaseModelConfig):
    """
    Config for the DeepSTPP model.
    Inherits common optimizer fields (lr, opt, momentum, weight_decay, dropout).
    """
    # — model id for registry —
    model_id       : str   = "deepstpp"

    # — architecture —
    emb_dim        : int   = 128
    hid_dim        : int   = 128
    z_dim          : int   = 128
    num_head       : int   = 2
    nlayers        : int   = 3
    num_points     : int   = 20
    seq_len        : int   = 20
    decoder_n_layer: int   = 3

    # — inference specifics —
    infer_nstep    : int   = 10000
    infer_limit    : int   = 13

    # — training/eval hooks (kept small since epochs live in TrainerConfig) —
    eval_epoch     : int   = 5
    lookahead      : int   = 1
    scheduler      : str   = None
    generate_type  : bool  = True
    read_model     : bool  = False
    sample         : bool  = False

    # — regularisation —
    beta           : float = 1e-3
    constrain_b    : str = "sigmoid"
    s_min          : float = 1e-3
    b_max          : int = 20
    
    def __post_init__(self):
        # Ensure that the model_id is valid
        if self.model_id != "deepstpp":
            raise ValueError(f"Unknown DeepSTPP variant {self.model_id!r}")
        
        # Additional validation or preparation can be done here if needed
        if not isinstance(self.num_points, int) or self.num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        
        
        if self.search_space:
            tunables, constants = split_search_space(self.search_space, type(self))
            for k, v in constants.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._ray_tune_space = tunables
            self.search_space = None 
    def ray_space(self):
        return self._ray_tune_space
    
@BaseModelConfig.register("smash")
@dataclass
class SMASHConfig(BaseModelConfig):
    model_id         : str = "smash"
    
    dim              : int = 2
    cond_dim         : int = 64
    num_types        : int = 1
    sigma_time   : float = 0.05
    sigma_loc    : float = 0.05
    samplingsteps: int = 500
    n_samples    : int = 100
    langevin_step: float = 0.005
    loss_lambda  : float = 0.5
    loss_lambda2 : float = 1.0
    smooth       : float = 0.0  
    total_epochs: int = 1000  

    def __post_init__(self):
        
        return super().__post_init__()
    
    def ray_space(self):
        return self._ray_tune_space
        
@BaseModelConfig.register("diffstpp")
@dataclass
class DiffSTPPConfig(BaseModelConfig):
        model_id         : str = "diffstpp"
        
        dim              : int = 2
        cond_dim         : int = 64
        num_types        : int = 1
        samplingsteps    : int = 500
        n_samples        : int = 100

        total_epochs     : int = 1000 
        timesteps        : int = 100
        beta_schedule    : str = "cosine"
        loss_type        : str = "l2"
        objective        : str = "pred_noise" 
    
        def __post_init__(self):
            
            return super().__post_init__()
        
        def ray_space(self):
            return self._ray_tune_space

