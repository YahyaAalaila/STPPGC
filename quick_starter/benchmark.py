# scripts/benchmark.py
from __future__ import annotations
# ──────────────────────────────────────────────────────────────────────────
from concurrent.futures import ProcessPoolExecutor
from pathlib            import Path
import mlflow, json, os

from lightning_stpp.config_factory.benchmark_config import BenchmarkConfig
from lightning_stpp.config_factory.runner_config    import RunnerConfig
from lightning_stpp.runner._runner              import BaseRunner
from lightning_stpp.HPO.hpo_base                    import HyperTuner
# ──────────────────────────────────────────────────────────────────────────

def _full_run(exp_cfg: RunnerConfig, bench_ds: str) -> dict:
    """
    * attach dataset → exp_cfg.data.name = bench_ds
    * Ray-Tune to get best hyper-params   (validation loss via callback)
    * final training on train+val , test once
    * return test metrics as dict
    """
    # ── 1) attach dataset ───────────────────────────────────────────────
    print(type(exp_cfg))
    exp_cfg = exp_cfg.clone(data = exp_cfg.data.clone(name = bench_ds))

    # ── 2) hyper-param search (Ray Tune) ────────────────────────────────
    tuner   = HyperTuner.build_hpo_from_config(exp_cfg)
    analysis = tuner.run()                       # ← returns ray.tune.ResultGrid

    best_params = analysis.best_config           # dict of tuned fields
    tuned_model = exp_cfg.model.clone(**best_params)
    frozen_cfg  = exp_cfg.clone(model = tuned_model, hpo = None)  # drop HPO

    # ── 3) final fit + single test ─────────────────────────────────────
    runner = BaseRunner.build_runner_from_config(frozen_cfg)

    # training (validation still happens, but no Tune callback now)
    runner.fit()

    # test once; Lightning returns a list of dicts (one per dataloader)
    test_metrics = runner.trainer.test(verbose = False)[0]

    # also capture the method name for later aggregation
    return {"method": frozen_cfg.runner_id, **test_metrics}

# ──────────────────────────────────────────────────────────────────────────
def main(yaml_path: str):
    bench_cfg: BenchmarkConfig = BenchmarkConfig.from_yaml(yaml_path)
    bench_cfg.finalize()  # sanity checks + propagate fields to experiments
    
    mlflow.set_experiment(f"bench_{bench_cfg.dataset}")
    os.makedirs("results", exist_ok=True)

    with ProcessPoolExecutor() as pool, mlflow.start_run(run_name="benchmark"):
        futures  = [pool.submit(_full_run, c, bench_cfg.dataset)
                    for c in bench_cfg.experiments]
        results  = [f.result() for f in futures]

        # save to MLflow and local JSON
        mlflow.log_dict({"dataset": bench_cfg.dataset,
                         "results": results}, "results.json")
        Path("results/summary.json").write_text(json.dumps(results, indent=2))

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    main(sys.argv[1])
