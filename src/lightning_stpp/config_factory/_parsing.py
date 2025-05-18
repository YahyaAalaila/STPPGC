
from typing import Dict, Any
from ray import tune

def parse_tune_dsl(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert DSL nodes into the corresponding `ray.tune` objects.
    E.g. {"lr": {"loguniform": [1e-5,1e-2]}}
    """
    out: Dict[str, Any] = {}
    for key, spec in raw.items():
        if isinstance(spec, dict) and len(spec) == 1:
            name, args = next(iter(spec.items()))
            match name.lower():
                case "loguniform": out[key] = tune.loguniform(*args)
                case "uniform":    out[key] = tune.uniform(*args)
                case "choice":     out[key] = tune.choice(args)
                case "randint":    out[key] = tune.randint(*args)
                case _:
                    raise ValueError(f"Unknown Tune op {name!r} in search_space.")
        else:
            raise TypeError(f"Malformed search_space for {key!r}: {spec}")
    return out