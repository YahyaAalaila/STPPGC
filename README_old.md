# STPP Glue Code

This repository provides a **glue code** framework for experimenting with Spatiotemporal Point Process (STPP) methods, at the moment only two frameworks are included: **NeuralSTPP** and **DeepSTPP** (with deepstpp not tested yet). It centralizes data loading, configuration, and training/evaluation scripts so you can easily switch or combine these methods.

## Contents

- **configs/**: YAML configuration files (Hydra/OmegaConf) for specifying hyperparameters, data paths, logging settings, etc.
- **data/**, **data_generators/**, **data_loader/**: Scripts for data preprocessing, normalization, and loader utilities (YA: This will be updated entirely to have an abstraction of its own, it will have different DGPs and different pattern complexity assessments)
- **lib/**: Contains cloned repositories or integrated code for (YA: These where cloned cloned):
  - **neural_stpp/** (Facebook Research’s NeuralSTPP).
  - **deepstpp/** (Rose-STL-Lab’s DeepSTPP).
  - **my_utils/**: Utility modules (e.g., logging wrappers, config helpers).
- **runner/**: Runners (e.g., `NeuralSTPPRunner`, `DeepSTPPRunner`) to orchestrate training pipelines (YA: in the next version, this will be only one file -if possible-)
- **scripts/**: Additional scripts for tasks like model conversion, debugging, or dataset visualization.
- **stpp_models/**: Adapters/wrappers that unify the different STPP approaches under a common interface (e.g., `NeuralSTPPAdapter`, `DeepSTPPAdapter`).
- **mlruns/**, **outputs/**: Byproducts of experiment tracking (MLflow) and Hydra’s output directories.
- **train.py**: Entry point for running training/evaluation with Hydra.

## Run

```bash
python train.py -m hydra/launcher=joblib model=jumpcnf,selfattn,jumpgmm
```

## Comments

1. **run the following when using NeuralSTPP**

   ```
   python setup.py build_ext --inplace
   ```

   (YA: I will think of a better way of doing this)

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Framework and roadmap

Below is the framework we are building for this package as text and in a graphic.
Classes to implement:

* [ ] [Dataset](https://github.com/YahyaAalaila/STPPGC/issues/11) -- Data used for model training and testing 
* [ ] [Resampling](https://github.com/YahyaAalaila/STPPGC/issues/5) -- Methods for data splitting, holds rows/columns to slice
* [ ] [Measure](https://github.com/YahyaAalaila/STPPGC/issues/16) -- Losses for model evaluation
* [ ] [Estimator](https://github.com/YahyaAalaila/STPPGC/issues/2) -- Algorithms with train/test procedures (PyTorch + Lightning)
* [ ] [ParameterSet](https://github.com/YahyaAalaila/STPPGC/issues/12) -- Hyperparameter configurations in a readable format
* [ ] [Performance](https://github.com/YahyaAalaila/STPPGC/issues/17) -- Results from benchmark experiments for onward analysis

Rectangular nodes represent base classes with the colours indicating implementation progress:

- Green = Complete
- Orange = In Progress
- Red = To Do

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
