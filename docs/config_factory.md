<p align="center">
  <img src="https://raw.githubusercontent.com/YourOrg/GlueCode/feature/first-prototype-2/docs/stpplogo.png" alt="GlueCode Logo" width="120"/>
  <h1 align="center">BenchSTPP</h1>
  <p align="center">
    <strong>A flexible, Hydra-powered benchmarking toolkit for streaming Spatio-Temporal Point Process models</strong>
  </p>

  <!-- Badges -->
  <p align="center">
    <a href="https://pypi.org/project/gluecode/"><img src="https://img.shields.io/pypi/v/gluecode.svg" alt="PyPI version"></a>
      <a href="https://github.com/YahyaAalaila/STPPGC/commits/main">
    <img src="https://img.shields.io/github/last-commit/YahyaAalaila/STPPGC.svg"
         alt="Last commit">
    <a href="https://img.shields.io/github/actions/workflow/status/YourOrg/GlueCode/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/YourOrg/GlueCode/ci.yml" alt="CI Status"></a>
     <a href="https://github.com/YahyaAalaila/STPPGC/issues">
    <img src="https://img.shields.io/github/issues/YahyaAalaila/STPPGC.svg"
         alt="Open issues">
  </a>
    <a href="https://img.shields.io/badge/license-Apache%202.0-blue.svg"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</p>

---
<p align="center">
  <!-- Python version -->
  <a href="https://www.python.org/doc/versions/">
    <img
      src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python"
      alt="Python 3.9+"/>
  </a>
  <!-- PyTorch -->
  <a href="https://pytorch.org/">
    <img
      src="https://img.shields.io/badge/pytorch-2.2%2B-orange?logo=pytorch"
      alt="PyTorch"/>
  </a>
  <!-- PyTorch Lightning -->
  <a href="https://www.pytorchlightning.ai/">
    <img
      src="https://img.shields.io/badge/lightning-2.2%2B-790ee7?logo=PyTorch-Lightning"
      alt="PyTorch Lightning"/>
  </a>
  <!-- Ray Tune -->
  <a href="https://docs.ray.io/en/latest/tune/index.html">
    <img
      src="https://img.shields.io/badge/ray__tune-2.9%2B-yellow?logo=ray"
      alt="Ray Tune"/>
  </a>
</p>

## 📖 Overview

**BenchSTPP** is an easy-to-use, highly-configurable framework for

- 🔄 **Benchmarking** streaming Spatio-Temporal Point Process (STPP) models in parallel  
- ⚙️ **Config management** via  [![Hydra][hydra-badge]][hydra]  
- 📊 **Logging & tracking** via [![MLflow][mlflow-badge]][mlflow]  
- 🚀 **Distributed tuning** via [![Ray Tune][raytune-badge]][raytune]

...

[hydra]:     https://hydra.cc/  
[mlflow]:    https://mlflow.org/  
[raytune]:   https://docs.ray.io/en/latest/tune/index.html  

[hydra-badge]:   https://img.shields.io/badge/Hydra-1.3-blue?logo=hydra&logoColor=white  
[mlflow-badge]:  https://img.shields.io/badge/MLflow-1.38-orange?logo=mlflow&logoColor=white  
[raytune-badge]: https://img.shields.io/badge/Ray_Tune-2.9-yellow?logo=ray&logoColor=white  



Designed for researchers and practitioners who want:  
> “**One config → many runs**, fully reproducible, effortlessly parallel.”  

---

## 🔗 Table of Contents

| [News](#news) | [Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Usage](#usage) | [Configuration](#configuration) | [Benchmark](#benchmark) | [Contributing](#contributing) | [License](#license) |

---

## 🗞️ News

- 🆕 **2025-05-18** Added true parallel benchmarking with `ProcessPoolExecutor` and improved error handling.  
- 🆕 **2025-05-10** Fixed MLflow URI parsing bug (`file:./mlruns` now works!).  
- 🆕 **2025-04-30** Switched default config schema to Hydra v1.3.  

*(Click the “+” below to see past releases…)*

<details>
<summary>Previous news</summary>

- **2025-03-12** Initial public release: Hydra + MLflow + Optuna + multi-process runner  
- **2025-02-25** Added `RunnerState` checkpointing & resume  
</details>

---

## 🚀 Features

- **Modular Configurations**  
  Abstract `BenchmarkConfig` → one or more `RunnerConfig` → `DataConfig` / `ModelConfig` / `TrainerConfig` / `LoggingConfig` / `HPOConfig`  
- **Parallel Benchmarking**  
  Built–in support for `concurrent.futures.ProcessPoolExecutor`  
- **Seamless Logging**  
  Out-of-the-box MLflow integration with custom experiment & run naming  
- **Hyper-parameter Tuning**  
  Fully pluggable Optuna pipeline  
- **Framework-agnostic**  
  Run your favorite STPP in PyTorch → just subclass `BaseSTPPModule`  
- **Reproducible**  
  Deterministic seeds, config versioning, checkpoint & resume

---

## 💾 Installation

```bash
# From PyPI
pip install gluecode

# Or from source
git clone https://github.com/YourOrg/GlueCode.git
cd GlueCode
pip install -e .
```


## 🚀 Getting Started

Kick the tires with our **Example Usage** Colab:

<p align="left">
  <a href="https://colab.research.google.com/github/your-user/your-repo/blob/main/notebooks/example_usage.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"
         alt="Open in Colab"/>
  </a>
</p>

Or clone locally and run:

```bash
git clone https://github.com/your-user/your-repo.git
cd your-repo
pip install -e .
python train.py --config-name example_usage
```

## Framework

```mermaid
%%{init: {
    "theme": "base",
    "themeVariables": {
      "fontSize": "18px",
      "nodeSpacing": "60",
      "rankSpacing": "50"
    }
}}%%
flowchart TD

    %% Theming
    classDef done fill:#37D629,stroke:#C829D6,color:#000
    classDef progress fill:#F58E0A,stroke:#0A71F5,color:#000
    classDef todo fill:#DD2B22,stroke:#22D4DD,color:#000
    classDef external fill:#FFFFFF,stroke:#000000,color:#000

    subgraph Class Structure

    subgraph Data Preparation
        A[Dataset] --Select strategy--> B[Resampling]
        B --Outer training data--> C[Dataset]
        B --Outer testing data--> D[Dataset]
    end

    subgraph Model Training
        C --Nested Resampling--> E[Resampling]
        E --K-1 folds for training--> F[Dataset]
        E --Kth fold for testing--> G[Dataset]

        F --Split into pure training data--> I[Dataset]
        F --Split into validation data--> Y[Dataset]

        K[Measure] --For early stopping etc.--> J

        L[Measure] --For HPO--> M[(Ray Tune)]
        G --Evaluate each HP config--> M
        J[Estimator] --Configured model for HPO--> M
        N[/Hyperparameter Config/] --Define HPO space--> AA[ParameterSet]
        AA --> M
        Y --> M
        I --> M

        M --Returns optimal HP config--> Q[ParameterSet]

        Q --Tuned esimator--> S[Estimator]
        C --Train best model on complete outer K-1 folds--> Z[Dataset]
        C --Split validation data--> T[Dataset]
        T --> S
        Z --> S


    end

    subgraph Model Testing
        S --Re-trained estimator--> U[Estimator]
        D --Test on k-th outer fold--> U
        U --> V[Performance]
        W[Measure] --Measure for testing--> V
    end

    end


    %% Assign classes
    %% class done
    class M,N external
    class J,S,U progress
    class A,B,C,D,E,F,G,H,I,K,L,O,P,Q,R,T,V,W,X,Y,Z,AA todo
```
