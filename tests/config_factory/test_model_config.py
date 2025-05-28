# tests/test_config_factory.py
"""
Tests for the *type-aware* search-space parsing:

1.  Built-in DSL samplers still work.
2.  Scalar-typed fields:
      • scalar  → constant
      • two-list→ uniform / loguniform
      • ≥3 list → categorical
3.  Sequence-typed fields (List[int]):
      • flat    → constant
      • nested  → categorical sweep
4.  Integration: constants override defaults, tunables land in ray_space().
"""
from dataclasses import is_dataclass
from numbers     import Number

import pytest
from ray import tune

# --------------------------------------------------------------------
#  Adjust these two imports if your package path differs
# --------------------------------------------------------------------
from lightning_stpp.config_factory.model_config   import NeuralSTPPConfig
from lightning_stpp.config_factory._parsing       import split_search_space
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# 1)  test for the helper split_search_space (important)
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaml_val, sampler_fn, sampler_args",
    [
        ({"uniform":     [0.0, 1.0]},       tune.uniform,     (0.0, 1.0)),
        ({"loguniform":  [1e-5, 1e-2]},     tune.loguniform,  (1e-5, 1e-2)),
        ({"qrandint":    [1, 10, 2]},       tune.qrandint,    (1, 10, 2)),
        ({"grid_search": [[1,2], [3,4]]},   tune.grid_search, [[1,2],[3,4]]),
    ],
    ids=["uniform","loguniform","qrandint","grid-search"]
)
def test_builtin_dsl(yaml_val, sampler_fn, sampler_args):
    tun, const = split_search_space({"param": yaml_val}, NeuralSTPPConfig)
    assert const == {}
    assert "param" in tun

    if sampler_fn is tune.grid_search:
        expected = sampler_fn(sampler_args)
        assert tun["param"] == expected
    else:
        expected = sampler_fn(*sampler_args)
        assert isinstance(tun["param"], type(expected))

@pytest.mark.parametrize(
    "yaml_val, is_sampler",
    [
        (0.01,                 False),          # scalar ---> constant
        ([1e-5, 1e-2],         True),           # two positive ---> loguniform
        ([-1.0,  1.0],         True),           # mixed sign   ---> uniform
        ([0.1, 0.2, 0.3],      True),           # len() is more than 2   ---> categorical
    ],
    ids=["const-float","logu","uniform","choice-float"]
)
def test_scalar_typed_rules(yaml_val, is_sampler):
    tun, const = split_search_space({"lr": yaml_val}, NeuralSTPPConfig)
    if is_sampler:
        assert "lr" in tun and not const
    else:
        assert "lr" in const and not tun


@pytest.mark.parametrize(
    "yaml_val, is_sampler",
    [
        ([64,64,64],                         False),  # based on our definition, flat list ---> constant vector, and nested ---> choice
        ([[32,32,32], [64,64,64]],           True),   
    ],
    ids=["const-vector","choice-vector"]
)
def test_sequence_typed_rules(yaml_val, is_sampler):
    tun, const = split_search_space({"hdims": yaml_val}, NeuralSTPPConfig)
    assert ("hdims" in tun) is is_sampler
    assert ("hdims" in const) is (not is_sampler)


# --------------------------------------------------------------------
# 2)  ModelConfig overrides + dynamically defined ray_space
# --------------------------------------------------------------------

def test_model_config_integration():
    cfg_dict = {
        "model_id": "neural_stpp",
        "search_space": {
            "dim": 4,                                   # scalar constant
            "lr":  [1e-5, 1e-3],                        # loguniform
            "hdims": [[32,32,32], [64,64,64]],          # categorical
        }
    }
    cfg = NeuralSTPPConfig.from_dict(cfg_dict)

    # constants should be injected
    assert cfg.dim == 4

    space = cfg.ray_space()
    assert set(space) == {"lr", "hdims"}
    assert isinstance(space["lr"], type(tune.loguniform(1e-5, 1e-3)))
    assert isinstance(space["hdims"], type(tune.choice([[32,32,32],[64,64,64]])))
