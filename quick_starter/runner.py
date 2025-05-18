from lightning_stpp.runner._runner import BaseRunner
from lightning_stpp.config_factory.runner_config import RunnerConfig
from lightning_stpp.utils.load_config import load_and_finalize

def main(cfg_path: str):
    # Load the configuration
    cfg = load_and_finalize(cfg_path)
    if cfg.hpo is not None:
        raise RuntimeError(
            "Config has 'hpo' section. "
            "Use hpo_example.py for hyperparameter tuning."
        )
    # Create a runner instance
    runner = BaseRunner.build_runner_from_config(cfg)
    # Run the training and testing process
    runner.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML file with a 'runner' block")
    main(**vars(parser.parse_args()))