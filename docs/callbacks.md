flowchart TB
%% Benchmark layer
subgraph BENCHMARK["benchmark.yaml"]
    A0[BenchmarkConfig]
    A0 -->|"finalize()"| A1["RunnerConfig #1"]
    A0 --> A2["RunnerConfig #2"]
    A0 --> A3["RunnerConfig #3"]
end

%% RunnerConfig decomposition
subgraph RUNNERCFG[RunnerConfig]
    direction LR
    A1 --> B1[DataConfig]
    A1 --> B2["ModelConfig<br/>(NeuralSTPPConfig,…)"]
    A1 --> B3[TrainerConfig]
    A1 --> B4[LoggingConfig]
    A1 --> B5[HPOConfig]
end

%% HPO → Ray Tune
B5 -->|"build_hpo_from_config()"| C0((HyperTuner))
C0 -->|"run()"|      C1[RayTuneRunner]
C1 -->|"optimize"|   B2

%% Build model & trainer
B2 -->|"build_model()"|          D1["BaseSTPPModule subclass"]
B3 -->|"build_pl_trainer()"|     D2["⚡ Lightning Trainer"]
B4 -->|"make_logger()"|          C2[MLflowLogger]

%% Callbacks
C2 --> D2
subgraph CALLBACKS[callbacks]
    direction LR
    E1[TestScheduler]
    E2[ValScheduler]
    E3[EMACallback]
    E4[TrainLogger]
end
B3 -->|"custom_callbacks[]"| CALLBACKS
CALLBACKS --> D2

%% DataModule
B1 --> D3["LightDataModule subclass"]
D3 --> D2

%% Docs links
click A0 href "docs/benchmark.md#benchmarkconfig"     "See BenchmarkConfig"
click A1 href "docs/runner_config.md#runnerconfig"     "See RunnerConfig"
