# Config-Factory & Runner Flow 

Below is a high-level overview of how our configuration-factory drives the construction 
of data, model, trainer, logging and HPO components, and how the runner ties them all 
together into a PyTorch Lightning experiment.

## Configuration Factory

### One object - Many specialised subobjects
All configs inherit from `Config` (`_config.py`), which give the following

1. YAML (`to_yaml`, `from_yaml`, `from_dict`) 
2. `clone(**patch)` copy‑with‑override
3. Ray‑Tune search‑space hook – subclasses can add tunables via `ray_space()`
4. Class‑registry via Registrable (so you can write only the short name in YAML). This allows the following 
`model: model_config: "neuralstpp"`
5. Callbacks (EMA, schedulers, logger) plug into Lightning’s event hooks to provide extra behaviours without touching the training code.
1. YAML (`to_yaml`, `from_yaml`, `from_dict`) 
2. `clone(**patch)` copy‑with‑override
3. Ray‑Tune search‑space hook – subclasses can add tunables via `ray_space()`
4. Class‑registry via Registrable (so you can write only the short name in YAML). This allows the following 
`model: model_config: "neuralstpp"`
5. Callbacks (EMA, schedulers, logger) plug into Lightning’s event hooks to provide extra behaviours without touching the training code.

### Leaf configs

| File       | Registered name     | 	Main purpose | 	Interesting bits        |
|-----------------|-----------------------------------------|-----------------------|---------|
| `data_config.py`    | `data_config`          | which dataset to use  |  just a single name field (kept minimal on purpose for now)|
| `model_config.py`      | `neuralstpp`  | all hyper‑params for one Neural‑STPP variant   | `build_model()` returns a ready PyTorch module <br> `ray_space()` translates the `search_space`: DSL into `ray.tune.*` objects |
| `trainer_config.py`     | `trainer_config`   | 	everything that feeds Lightning Trainer       | -  `build_pl_trainer()` builds a fully wired `pl.Trainer` (logger, callbacks, ckpts …)<br>- `_build_custom_callbacks()` lets YAML inject extra callbacks   |
| `logger_config.py`   | `logging_config`     | MLflow destination & naming    | tiny, defaults to a local `./mlruns` folder |
| `hypertuning_config.py`   | `hpo_config`        | 	Ray Tune knobs (scheduler, searcher, resources)       | `make_scheduler`, `make_search_alg`|


### Mid-Level configs



|File | Object | What it bundles
|----|----|----|
|`runner_config.py`| `RunnerConfig`| one experiment = data + model + trainer (+ logging + HPO)|
|`benchmark_config.py`| `BenchmarkConfig`| a benchmark = 1 dataset + many experiments (each is a `RunnerConfig`) |




### Flowchart  🚀 
```mermaid
%% ───────────────────────────────────────────────────────────────
%% 1.  The benchmark YAML is the entry-point
%% ───────────────────────────────────────────────────────────────
flowchart TD
    subgraph BENCHMARK.yaml  ["`benchmark.yaml`"]
        A0[BenchmarkConfig] --> A1[RunnerConfig #1]
        A0 --> A2[RunnerConfig #2]
        A0 --> A3[RunnerConfig #3]
    end

%% ───────────────────────────────────────────────────────────────
%% 2.  Anatomy of ONE RunnerConfig (copied for every experiment)
%% ───────────────────────────────────────────────────────────────
    subgraph RunnerConfig
        direction LR
        R[RunnerConfig] --> D[DataConfig]
        R --> M["ModelConfig (NeuralSTPPConfig, …)"]
        R --> T[TrainerConfig]
        R --> L[LoggingConfig]
        R --> H[HPOConfig]
    end

%% ───────────────────────────────────────────────────────────────
%% 3.  Factory methods build runtime objects
%% ───────────────────────────────────────────────────────────────
    M -- "build_model()" --> N((PyTorch 🔧))
    T -- "build_pl_trainer()" --> P((⚡ Lightning Trainer))

    %% Logging & callbacks
    L -- "make_logger()" --> ML([MLflowLogger])
    ML --> P
    T -- "custom_callbacks[]" --> CB[Callbacks]
    CB -->|EMACallback| P
    CB -->|ValScheduler| P
    CB -->|TestScheduler| P
    CB -->|TrainLogger| P

%% ───────────────────────────────────────────────────────────────
%% 4.  Optional hyper-param tuning
%% ───────────────────────────────────────────────────────────────
    H -- "build_hpo()" --> HT([Ray Tune Tuner])
    HT -.optimize.-> M
