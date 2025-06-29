import torch
from lightning_stpp.utils.load_config import load_and_finalize

# right at the top of your main entrypoint, *before* any other torch imports:
torch.set_default_dtype(torch.float32)

def main(cfg_path: str):
    # Load the configuration
    cfg = load_and_finalize(cfg_path)
    print("cfg:", cfg.hpo)
    
    if cfg.hpo is not None:
        from lightning_stpp.HPO.hpo_base import HyperTuner
        from lightning_stpp.utils.save import save_analysis
        
        runner = HyperTuner.build_hpo_from_config(cfg)
        analysis = runner.run()
        save_analysis(analysis, cfg.hpo.results_dir)
        
    else:
        # Create a runner instance
        from lightning_stpp.runner._runner import BaseRunner
        
        runner = BaseRunner.build_runner_from_config(cfg)
        # Run the training and testing process
        runner.run()

# ... (the rest of your file remains the same)
if __name__ == "__main__":
    import sys
    main(sys.argv[1])