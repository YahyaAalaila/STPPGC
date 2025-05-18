from __future__ import annotations
from dataclasses import dataclass, asdict, fields, is_dataclass, replace
from abc import ABC, abstractmethod
from typing import Any, Dict
from omegaconf import OmegaConf
from pathlib import Path
from lightning_stpp.utils.registrable import Registrable

@dataclass
class Config(Registrable, ABC):
    """Base for every config object"""

    # -------- serialization helpers --------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path):
        OmegaConf.save(config=OmegaConf.create(self.to_dict()), f=str(path))

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
