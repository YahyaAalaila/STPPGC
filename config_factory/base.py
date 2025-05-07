from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from abc import ABC, abstractmethod
from typing import Any, Dict
from omegaconf import OmegaConf
from pathlib import Path
from utils.registrable import Registrable

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
        return cls(**data)          # type: ignore[arg-type]

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
