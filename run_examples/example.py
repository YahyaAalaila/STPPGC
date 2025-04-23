# This is old code that is not used anymore.
# I will probably delete it later.

from runner.stpp_runner import LighRunner
from omegaconf import OmegaConf

def main():
    
    config = OmegaConf.load("configs/config.yaml")
    
    runner = LighRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
