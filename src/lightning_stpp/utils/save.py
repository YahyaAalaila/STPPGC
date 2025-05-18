import os, yaml, pandas as pd

def save_analysis(analysis, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    # 1) all trial results
    df = analysis.results_df
    df.to_csv(os.path.join(results_dir, "ray_tune_trials.csv"), index=False)

    # 2) the best hyper‑params
    best_cfg = analysis.best_config
    with open(os.path.join(results_dir, "best_config.yaml"), "w") as f:
        yaml.dump(best_cfg, f)

    print(f"[save_analysis] saved trials + best_config to {results_dir}")
