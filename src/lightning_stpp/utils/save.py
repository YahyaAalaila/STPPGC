# In file: src/lightning_stpp/utils/save.py

import os
import yaml
from ray.tune.analysis import ExperimentAnalysis

def save_analysis(analysis: ExperimentAnalysis, results_dir: str):
    """
    Saves the results of a Ray Tune experiment.
    """
    # --- START OF FIX ---
    # Sanitize the path: Remove the "file://" prefix if it exists.
    clean_path = results_dir
    if clean_path.startswith("file://"):
        clean_path = clean_path[7:] # Keep the part of the string after "file://"
    # --- END OF FIX ---

    # Now, use the sanitized 'clean_path' for all filesystem operations
    os.makedirs(clean_path, exist_ok=True)
    print(f"Directory confirmed/created: {clean_path}")

    # 1) Save all trial results to CSV
    df = analysis.results_df
    csv_path = os.path.join(clean_path, "ray_tune_trials.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved all trials to: {csv_path}")

    # 2) Save the best hyper-parameters to YAML
    try:
        best_cfg = analysis.best_config
        yaml_path = os.path.join(clean_path, "best_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(best_cfg, f)
        print(f"Saved best config to: {yaml_path}")
    except ValueError as e:
        print(f"Could not save best_config automatically: {e}")

    print(f"[save_analysis] Analysis saving complete.")