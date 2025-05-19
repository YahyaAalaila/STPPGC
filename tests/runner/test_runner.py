"""
Unit–tests for config_factory.runner  (the BaseRunner-registered “dl_stpp”)

These tests are **pure-python** – no GPU, MLflow server or long training run
is started.  We monkey-patch the heavy bits (Lightning Trainer, MLFlowLogger,
dummy datamodule, etc.) so the tests run in < 1 s.
"""

import sys
from types import SimpleNamespace
import builtins
import importlib
import pytest

# --------------------------------------------------------------------------- 
# Helpers that act like the objects Runner expects
# --------------------------------------------------------------------------- 
class FakePLTrainer:
    # minimal interface used by Runner.fit/test
    def __init__(self, *_, **__):
        self._fit_called  = False
        self._test_called = False

    def fit(self, model, datamodule=None):
        self._fit_called = True
        # sanity checks
        assert hasattr(model, "forward")
        assert datamodule is not None

    def test(self, model, datamodule=None, verbose=False):
        self._test_called = True
        assert datamodule is not None
from lightning_stpp.models.base import BaseSTPPModule

@BaseSTPPModule.register(name = "fakestpp")
class FakeLightningModule:
    # returned by BaseSTPPModule.by_name
    def __init__(self, cfg):
        self.cfg = cfg
    def callbacks(self):               # Runner.model.callbacks()
        return []
    def forward(self, *a, **kw):       # used in FakePLTrainer.fit
        pass

class FakeDataModule:
    @staticmethod
    def build_datamodule_from_config(cfg):
        return FakeDataModule()
    # Lightning expects these
    def setup(self, *a, **kw): pass
    def train_dataloader(self): return []
    def val_dataloader  (self): return []
    def test_dataloader (self): return []

class DummyLogger:
    def __init__(self, *_, **__): pass
    def info(self, *_): pass


# --------------------------------------------------------------------------- 
# monkey-patch the heavy deps BEFORE importing Runner
# --------------------------------------------------------------------------- 
import sys, types, importlib
from types import ModuleType, SimpleNamespace
import pytest

@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    # 1) create the top-level fake package

    fake_pl = ModuleType("lightning")
    # stub out LightningModule
    fake_pl.LightningModule = type("LightningModule", (), {})

    # 2) create the loggers submodule
    loggers_mod = ModuleType("lightning.loggers")
    class MLFlowLogger:
        def __init__(self, *args, **kwargs):
            pass
    loggers_mod.MLFlowLogger = MLFlowLogger

    # 3) inject both into sys.modules
    sys.modules["lightning"] = fake_pl
    sys.modules["lightning.loggers"] = loggers_mod
    # also attach the submodule to the parent
    fake_pl.loggers = loggers_mod

    yield

    # optional cleanup
    for mod in ["lightning.loggers", "lightning"]:
        sys.modules.pop(mod, None)

# --------------------------------------------------------------------------- 


# A **tiny** fake cfg object that satisfies Runner
class _TinyCfg(SimpleNamespace):
    # attribute names Runner touches
    model   = "fakestpp"
    data    = SimpleNamespace()  # content is ignored by FakeDataModule
    trainer = SimpleNamespace(
        build_pl_trainer=lambda **k: FakePLTrainer(),
        custom_callbacks=[],
    )
    logging = SimpleNamespace(
        experiment_name = "unit‑test-exp",
        run_name        = "test‑run",
        mlflow_uri      = "file:./mlruns",
    )
    def __init__(self):
        super().__init__()
        self.trainer.custom_callbacks = []
        self.trainer.build_pl_trainer = lambda **k: FakePLTrainer()

# --------------------------------------------------------------------------- 
def test_runner_registration():
    from lightning_stpp.runner._runner import BaseRunner
    from lightning_stpp.runner.stpp_runner import Runner
    # registry lookup must work
    cls = BaseRunner.by_name("dl_stpp")
    assert cls is Runner

def test_fit_and_test_cycle(monkeypatch):
    from lightning_stpp.runner.stpp_runner import Runner

    cfg = _TinyCfg()
    r   = Runner(cfg)

    # sanity: fit() sets the _fit_called flag on internal trainer
    r.fit()
    assert r.trainer._fit_called is True
    # sanity: evaluate() / .test()
    r.evaluate()
    assert r.trainer._test_called is True

