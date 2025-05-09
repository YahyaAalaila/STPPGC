```mermaid
flowchart TD
  subgraph BENCHMARK.yaml
    A0[BenchmarkConfig] --> A1[RunnerConfig #1]
    A0 --> A2[RunnerConfig #2]
  end

  subgraph RunnerConfig
    A1 --> B1[DataConfig]
    A1 --> B2[NeuralSTPPConfig]
    A1 --> B3[TrainerConfig]
    A1 --> B4[LoggingConfig]
    A1 --> B5[HPOConfig]
  end

  %% build_model arrow with parentheses safely in quotes
  B2 -->|"build_model()"| C1["PyTorch module"]
  B3 -->|"build_pl_trainer()"| C2["Lightning Trainer"]

  C1 --> C2
  B4 --> C3["MLflowLogger"]
  C3 --> C2