import argparse
from lightning_stpp.HPO.hpo_base import HyperTunner
from lightning_stpp.utils.load_config import load_and_finalize
from lightning_stpp.utils.save import save_analysis

def main(cfg_path: str):
    # Load the configuration
    cfg = load_and_finalize(cfg_path)
    if cfg.hpo is None:
        raise RuntimeError(
            "Config has no 'hpo' section. "
            "Use runner_example.py for a single run with predefined hyperparameters."
        )
    # Create a hyperparameter tuning instance
    tuner = HyperTunner.build_hpo_from_config(cfg)
    
    # Run the hyperparameter tuning process
    analysis = tuner.run()
    
    save_analysis(analysis, cfg.hpo.results_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML file with an 'hpo' block")
    main(**vars(parser.parse_args()))