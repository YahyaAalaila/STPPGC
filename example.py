from runner.stpp_runner import LighRunner
from omegaconf import OmegaConf

def main():
    
    config = OmegaConf.load("configs/config.yaml")
    
    runner = LighRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
