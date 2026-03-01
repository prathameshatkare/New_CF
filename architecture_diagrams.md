# Architecture and Methodology Diagrams

The following diagrams are based on the current implementation in `federated_pytorch.py`, `api_server.py`, and `run_project_pipeline.py`.

## 1) Neural Network Architecture (CFNet)

```mermaid
flowchart LR
    A[Input Features\n6D vector\nage, sex, height, weight, fev1, BMI] --> B[Linear 6 -> 64]
    B --> C[BatchNorm1d(64)]
    C --> D[ReLU]
    D --> E[Dropout p=0.20]
    E --> F[Linear 64 -> 32]
    F --> G[ReLU]
    G --> H[Dropout p=0.15]
    H --> I[Linear 32 -> 16]
    I --> J[ReLU]
    J --> K[Linear 16 -> 1\n(Logit)]
    K --> L[Sigmoid\nP(CF=1)]
    L --> M{Threshold 0.5}
    M -->|>= 0.5| N[High CF Risk]
    M -->|< 0.5| O[Lower CF Risk]
```

## 2) System Architecture (Federated Training + Web Inference)

```mermaid
flowchart TB
    subgraph DataLayer[Data Layer]
        D1[data/cystfibr.csv\n(CF)]
        D2[data/nhanes_clean.csv\n(Non-CF)]
    end

    subgraph PrepLayer[Preprocessing and Augmentation]
        P1[data_preprocessing_and_stat_validation.py\n-> cleaned_cf_dataset.csv]
        P2[data_augmentation_hybrid.py (TVAE)\n-> augmented_cf_dataset.csv]
    end

    subgraph FL[Research Federated Training]
        F0[federated_pytorch.py]
        F1[Train/Val/Test split + StandardScaler]
        F2[Non-IID Dirichlet partition\n5 clients]
        F3[Local client updates\nFedProx objective]
        F4[Server aggregation\nWeighted FedAvg]
        F5[Early stopping on validation AUC]
        F6[best_federated_cf_model.pt\n+ scaler stats + feature schema]
    end

    subgraph Eval[Evaluation and Reporting]
        E1[federated_ablation_runner.py]
        E2[model_comparison.py]
        E3[plot_federated_ablation.py]
    end

    subgraph Deploy[Inference Deployment]
        W1[web/index.html + app.js]
        W2[FastAPI api_server.py\n/api/predict, /api/health]
        W3[ModelBundle loads checkpoint\nCFNet + scaler]
        W4[Risk probability + label response]
    end

    D1 --> P1
    D2 --> P1
    P1 --> P2
    P2 --> F0
    F0 --> F1 --> F2 --> F3 --> F4 --> F5 --> F6
    F6 --> E1
    F6 --> E2
    E1 --> E3
    F6 --> W3
    W1 --> W2
    W2 --> W3
    W3 --> W4
```

## 3) End-to-End Methodology

```mermaid
flowchart TD
    A[Raw Clinical Tables] --> B[Feature Harmonization\nage, sex, height, weight, fev1]
    B --> C[BMI Engineering + Binary Target]
    C --> D[Statistical Validation\nMann-Whitney U, Cohen d, Correlation Delta]
    D --> E[Cleaned Dataset\n3025 rows: 3000 non-CF, 25 CF]

    E --> F[Minority-only TVAE Augmentation]
    F --> F1[Add small Gaussian noise\nto continuous minority features]
    F1 --> F2[Train TVAE on CF minority]
    F2 --> F3[Sample 300 synthetic CF rows]
    F3 --> F4[KS tests + source classifier check]
    F4 --> G[Augmented Dataset\n3325 rows: 3000 non-CF, 325 CF]

    G --> H[Global split\nTrain/Val/Test]
    H --> I[Standardize features\nfit on train only]
    I --> J[Non-IID client simulation\nDirichlet alpha=0.6]
    J --> K[Federated rounds]
    K --> K1[Sample ~80% clients/round]
    K1 --> K2[Local training with FedProx]
    K2 --> K3[Weighted FedAvg aggregation]
    K3 --> K4[Validate (AUC/F1) + early stopping]

    K4 --> L[Best checkpoint saved\nmodel + scaler + features]
    L --> M[Test metrics\nAcc, Precision, Recall, F1, ROC-AUC, PR-AUC]
    L --> N[Deployment\nFastAPI + Web UI]
    N --> O[Patient feature input -> CF risk probability]
```
