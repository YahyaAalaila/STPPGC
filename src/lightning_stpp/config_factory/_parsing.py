
from ray import tune

from dataclasses import fields, is_dataclass
from numbers     import Number
from typing      import Any, Dict, Tuple, get_origin, get_args, Union

# --------------------------------------------------------------------- #
# 1)  Built-in Ray-Tune sampler map (unchanged)                         #
# --------------------------------------------------------------------- #
_builtin = {
    "uniform":      tune.uniform,
    "loguniform":   tune.loguniform,
    "quniform":     tune.quniform,
    "qloguniform":  tune.qloguniform,
    "randint":      tune.randint,
    "qrandint":     tune.qrandint,
    "choice":       tune.choice,
    "grid_search":  tune.grid_search,
    "sample_from":  tune.sample_from,
}

# --------------------------------------------------------------------- #
# 2)  Small helpers                                                     #
# --------------------------------------------------------------------- #
_scalar_types = {int, float, bool, str}

def _unwrap_optional(tp):
    """Optional[int] -> int ;  Union[X, None] -> X."""
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else Any
    return tp

def _is_scalar(tp):
    return tp in _scalar_types

def _is_sequence(tp):
    origin = get_origin(tp)
    return origin in {list, tuple}

# --------------------------------------------------------------------- #
# 3)  Main split-helper                                                 #
# --------------------------------------------------------------------- #
def split_search_space(
    raw: Dict[str, Any],
    config_cls,                       # <-- pass the dataclass *class*
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parameters
    ----------
    raw : dict
        The ``search_space`` block from YAML.
    config_cls : dataclass type
        The dataclass whose annotations we use for type hints
        (e.g. ``NeuralSTPPConfig``).

    Returns
    -------
    tunables : dict   # Ray-Tune samplers
    constants: dict   # plain values to inject back
    """
    if not is_dataclass(config_cls):
        raise ValueError("config_cls must be a @dataclass")

    type_map = {f.name: _unwrap_optional(f.type) for f in fields(config_cls)}

    tunables:  Dict[str, Any] = {}
    constants: Dict[str, Any] = {}

    for k, v in raw.items():
        # ------------------------------------------------------------
        # 0. Built-in DSL {sampler: [...]}
        # ------------------------------------------------------------
        if isinstance(v, dict) and len(v) == 1:
            sampler_name, args = next(iter(v.items()))
            if sampler_name in _builtin:
                fn = _builtin[sampler_name]
                        # grid_search takes one argument (a list of choices), not two
                if sampler_name == "grid_search":
                    tunables[k] = fn(args)            # pass the whole list
                else:
                    tunables[k] = fn(*args) if isinstance(args, (list, tuple)) else fn(args)
                # tunables[k] = fn(*args) if isinstance(args, (list, tuple)) else fn(args)
                continue

        expected = type_map.get(k, Any)

        # ------------------------------------------------------------
        # 1. Scalar-typed fields
        # ------------------------------------------------------------
        if _is_scalar(expected):
            # treat 1-item list as scalar
            if isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]

            # constant value
            if not isinstance(v, (list, tuple)):
                constants[k] = v
                continue

            # two numbers -> uniform / loguniform
            if len(v) == 2 and all(isinstance(x, Number) for x in v):
                a, b = v
                op = tune.loguniform if (a > 0 and b > 0 and expected is float) else tune.uniform
                tunables[k] = op(a, b)
                continue

            # ≥3 (or mixed types) -> categorical choice
            tunables[k] = tune.choice(list(v))
            continue

        # ------------------------------------------------------------
        # 2. Sequence-typed fields (List[int], Tuple[float, ...], …)
        # ------------------------------------------------------------
        if _is_sequence(expected):
            # constant vector (flat list/tuple)
            if not any(isinstance(x, (list, tuple)) for x in v):
                constants[k] = v
            else:
                tunables[k] = tune.choice(list(v))
            continue

        # ------------------------------------------------------------
        # 3. Fallback: use previous heuristic
        # ------------------------------------------------------------
        if isinstance(v, (list, tuple)):
            if len(v) == 2 and all(isinstance(x, Number) for x in v):
                a, b = v
                tunables[k] = tune.loguniform(a, b) if a > 0 and b > 0 else tune.uniform(a, b)
            else:
                tunables[k] = tune.choice(list(v))
        else:
            constants[k] = v

    return tunables, constants

def dotlist_to_dict(dotlist) -> Dict[str, Any]:
    """
    Convert a list like
        ["model.model_sub_id=att-cnf", "trainer.max_epochs=200"]
    into a nested dict:
        {
          "model":   {"model_sub_id": "att-cnf"},
          "trainer": {"max_epochs": 200},
        }
    Strings that look like ints or floats are cast automatically.
    """
    out: Dict[str, Any] = {}
    for item in dotlist:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be key=value")
        key, val = item.split("=", 1)

        # naive scalar casting
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass  # leave as str

        cur = out
        parts = key.split(".")
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        cur[parts[-1]] = val
    return out
