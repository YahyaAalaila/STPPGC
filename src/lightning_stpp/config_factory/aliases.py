# lightning_stpp/config_factory/model_aliases.py
"""
Mapping from *model_sub_id* (as a user might type it) to the canonical
*model_config* key registered in Config.by_name().

Contributing:
-------------
If you integrate a new Neural-STPP variant (or any other family that re-uses
the same config class), add your alias here, e.g.

    MODEL_ALIASES["super-cnf"] = "neuralstpp"
"""

from typing import Any, Dict


MODEL_ALIASES = {
    # —— Neural-STPP family ————————————————————————————————
    "jump-cnf":  "neuralstpp",
    "att-cnf":   "neuralstpp",
    "cond-gmm":  "neuralstpp",
    "tvcnf":     "neuralstpp",
    "gmm":       "neuralstpp",
    
     # —— Deep-STPP family 
    "deepstpp":  "deepstpp", 
    "deep-stpp": "deepstpp",
    "DeepSTPP":  "deepstpp",
}

def infer_model_key(model_dict: Dict[str, Any]) -> str:
    """
    Decide which model-config class key to use:

    1. Explicit `model_config:` wins.
    2. Else look for `model_id:`.
    3. Else look for `model_sub_id:` and map via the alias table.
    """
    if "model_config" in model_dict:          # legacy explicit field
        return model_dict.pop("model_config").lower()

    if "model_id" in model_dict:
        return model_dict["model_id"].lower()

    if "model_sub_id" in model_dict:
        sub = model_dict["model_sub_id"].lower()
        if sub in MODEL_ALIASES:
            return MODEL_ALIASES[sub]
        raise ValueError(f"Unknown model_sub_id '{sub}' (can’t infer model_config)")

    raise ValueError(
        "model_config cannot be inferred: please provide either "
        "`model_config`, `model_id`, or `model_sub_id` in the YAML."
    )
