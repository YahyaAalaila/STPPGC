from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields, is_dataclass, replace
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from lightning_stpp.config_factory.aliases import infer_model_key
from omegaconf import OmegaConf
from pathlib import Path

import yaml
from lightning_stpp.utils.registrable import Registrable
from ._parsing import split_search_space

@dataclass
class Config(Registrable, ABC):
    """Base for every config object"""

    # -------- serialization helpers --------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path):
        OmegaConf.save(config=OmegaConf.create(self.to_dict()), f=str(path))
        
    def get_yaml_config(self) -> str:
        """Return this config (dataclass) serialised to YAML."""
        return yaml.dump(asdict(self), sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        data.update(overrides)
        return cls(**data)   
    @classmethod
    def from_dict(cls, raw: dict):
        """
        Generic constructor: for every dataclass field F, look inside *raw*
        and, if the corresponding attr is itself a Config subclass, delegate
        to its own from_dict.  Otherwise pass the value through unchanged.
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a @dataclass")

        kwargs = {}
        for f in fields(cls):
            if f.name not in raw:
                continue                      # use default
            val = raw[f.name]

            # If the field's type is a Config subclass, recurse
            if isinstance(f.type, type) and issubclass(f.type, Config):
                kwargs[f.name] = f.type.from_dict(val)   # recurse
            else:
                kwargs[f.name] = val

        return cls(**kwargs)# type: ignore[arg-type]
    
    def to_hparams(self, prefix: str | None = None) -> dict:
        """
        Return a flat Python dict that Lightning’s ``save_hyperparameters`` accepts.
        • Nested ``Config`` objects are expanded with ``<prefix>.field`` keys.
        • Non-primitive values (lists, dicts, dataclasses…) are left as-is
        – Lightning can store them verbatim.
        """
        def _flatten(obj, base):
            if isinstance(obj, Config):
                for k, v in asdict(obj).items():
                    yield from _flatten(v, f"{base}{k}.")
            else:
                yield base[:-1], obj  # drop trailing dot

        return {k: v for k, v in _flatten(self, "" if prefix is None else f"{prefix}.")}

    # -------- Ray‑Tune helpers (optional) ----
    @staticmethod
    def ray_space() -> Dict[str, Any]:
        """Return Tune search‑space dict; override in subclasses if needed."""
        return {}

    # -------- copy with update -------------
    def clone(self, **patch) -> "Config":
        return replace(self, **patch)

    # nice print
    def __repr__(self):
        return OmegaConf.to_yaml(self.to_dict())
    
# ------------- Model base class ------------- 

@dataclass
class BaseModelConfig(Config):
    # --- optimizer / reg common to all STPP models
    lr: float = 1e-3
    opt: str = "Adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    dropout: float = 0.0

    # --- Ray Tune space (override in subclasses!) ---
    search_space: Optional[Dict[str, Any]] = None

    # internal holder for the *actual* tunables
    _ray_tune_space: Optional[Dict[str, Any]] = field(init=False, default=None)
    
    monitor    : str = "val_loss"
    monitor_mode: str = "min"

    def __post_init__(self):
        # If a search_space dict was provided, split it into (tunables, constants)
        if self.search_space:
            tunables, consts = split_search_space(self.search_space, type(self))
            # apply any constant values back into self
            for k, v in consts.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._ray_tune_space = tunables
            # clear the raw dict so we know we've already handled it
            self.search_space = None
    
    # def finalize(self) -> None:
    #     model_key = infer_model_key()
    def ray_space(self) -> Dict[str, Any]:
        """
        Return only the tunable hyperparameters for Ray Tune.
        Raises if no search_space was ever provided.
        """
        if self._ray_tune_space is None:
            raise RuntimeError(f"No ray-space defined for {type(self).__name__}")
        return self._ray_tune_space

