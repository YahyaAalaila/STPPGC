# Config-Factory & Runner Flow 🚀  

Below is a high-level overview of how our configuration-factory drives the construction 
of data, model, trainer, logging and HPO components, and how the runner ties them all 
together into a PyTorch Lightning experiment.

## Configuration Factory

### One object - Many specialised subobjects
All configs inherit from `Config` (`_config.py`), which give the following

1. YAML / dict I/O (`to_yaml`, `from_yaml`, `from_dict) 
2. `clone(**patch)` copy‑with‑override
3. Ray‑Tune search‑space hook – subclasses can add tunables via `ray_space()`
4. Class‑registry via Registrable (so you can write only the short name in YAML). This allows the following 
`model: model_config: "neuralstpp"`

| File       | Registered name     | 	Main purpose | 	Interesting bits        |
|-----------------|-----------------------------------------|-----------------------|  |
| `data_config.py`    | `data_config`          | which dataset to use  |  just a single name field (kept minimal on purpose for now)|
| `model_config.py`      | `neuralstpp`  | all hyper‑params for one Neural‑STPP variant   | kk   |
| `trainer_config.py`     | `trainer_config`   | 	everything that feeds Lightning Trainer       | s |
| `logger_config.py`   | `logging_config`     | MLflow destination & naming    | s |
| `hypertuning_config.py`   | `hpo_config`        | 	Ray Tune knobs (scheduler, searcher, resources)       |s |


1. **RunnerConfig** pulls together `DataConfig`, `ModelConfig`, `TrainerConfig`,  
   `LoggingConfig`, and `HPOConfig`.
2. `ModelConfig` (e.g. `NeuralSTPPConfig`) knows how to `build_model()` a LightningModule.
3. `TrainerConfig` produces a Lightning `Trainer` with callbacks (logging, EMA, schedulers).
4. `LoggingConfig` yields an MLflowLogger; `HPOConfig` can spawn a Ray-Tune tuner.
5. Finally, the runner launches training & testing (optionally via `mp.spawn`).

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
        R --> M["ModelConfig\n(NeuralSTPPConfig, …)"]
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
