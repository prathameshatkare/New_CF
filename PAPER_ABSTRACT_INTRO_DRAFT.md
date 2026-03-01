# Draft Title, Abstract, and Introduction

## Title

Federated and Edge-Ready Deep Learning for Privacy-Preserving Cystic Fibrosis Risk Screening from Multi-Client Clinical Tabular Data

## Abstract

Cystic fibrosis (CF) is a progressive genetic disorder where delayed recognition can worsen respiratory decline and long-term outcomes. Building machine learning models for CF screening is challenging because clinically relevant data are fragmented across institutions and constrained by privacy regulations. We present a federated learning (FL) pipeline for CF risk screening that avoids centralizing patient-level data and supports edge-oriented inference.

Our study uses a cleaned clinical tabular dataset (`n=3,025`) and an augmented variant (`n=3,325`), with six features (`age`, `sex`, `height`, `weight`, `fev1`, `BMI`). Federated training is implemented with non-IID Dirichlet partitioning (`alpha=0.6`), partial participation (5 clients, 80% sampled per round), FedProx local optimization (`mu=0.01`), and sample-size-weighted FedAvg aggregation. The best model was trained for up to 20 rounds with 4 local epochs and evaluated on a held-out test set.

The final federated model achieved accuracy `0.9904`, precision `0.8065`, recall `1.0000`, F1-score `0.8929`, ROC-AUC `1.0000`, and PR-AUC `1.0000`. Feature ablation showed strong dependence on FEV1: dropping FEV1 reduced performance to accuracy `0.8800` and F1-score `0.3802`, while other single-feature drops retained high performance. A deployment-ready checkpoint (`best_federated_cf_model.pt`) and a Streamlit-based edge inference interface were produced for practical screening workflows.

These results indicate that privacy-preserving FL can achieve high CF screening performance while providing a feasible path to decentralized and edge-compatible clinical AI deployment.

## Introduction

### Clinical Background and Motivation

Cystic fibrosis (CF) is a chronic, multisystem genetic disease that primarily affects pulmonary function and requires early, repeated risk assessment to guide care. Screening-oriented decision support tools can help prioritize follow-up testing, especially where specialist access is limited. However, building robust ML systems for CF remains difficult because relevant clinical data are distributed across sites, often small at each location, and governed by strict privacy and data-sharing constraints.

### Why Centralized ML Is Not Enough

Conventional centralized ML pipelines assume all patient records can be pooled into one repository. In healthcare practice, this assumption is frequently invalid due to institutional silos, consent boundaries, and regulatory obligations. Even when centralization is technically possible, moving raw patient-level data can increase governance complexity and operational risk.

### Why Federated Learning + Edge

Federated learning offers a practical compromise: model parameters are trained collaboratively across clients while raw data remain local. This is attractive for CF screening tasks where each institution may have limited positive cases and heterogeneous distributions. Edge-oriented inference further improves practicality by enabling low-latency local predictions from a compact checkpoint without depending on continuous cloud connectivity.

### Study Gap

Although FL has been applied in multiple medical AI contexts, fewer studies provide an end-to-end CF tabular pipeline that jointly addresses: (1) non-IID client heterogeneity, (2) communication-efficient multi-client training, (3) post-training diagnostics (ablation/leakage/stress checks), and (4) deployable inference artifacts for edge scenarios.

### This Study

This work develops and evaluates a reproducible FL framework for CF risk screening using six routinely available clinical/spirometric features. The implemented training strategy combines non-IID Dirichlet partitioning (`alpha=0.6`), partial client participation (5 clients; 4 selected per round), FedProx local objectives, and weighted FedAvg aggregation. The final model is exported as `best_federated_cf_model.pt` and integrated into an interactive inference UI.

### Main Contributions

1. An end-to-end FL pipeline for CF risk screening with privacy-preserving multi-client training.
2. A non-IID robust training configuration (FedProx + weighted FedAvg + partial participation) with reproducible settings.
3. Strong held-out performance (`accuracy=0.9904`, `recall=1.0000`, `F1=0.8929`, `ROC-AUC=1.0000`) on the project test split.
4. Feature sensitivity evidence showing FEV1 as a dominant predictor (F1 drop from `0.8929` to `0.3802` when excluded).
5. Edge-ready deployment artifacts including checkpoint-based local inference and a Streamlit screening interface.

### Paper Organization

The remainder of this paper reviews related work, details dataset and methodology, describes the federated/edge setup, reports comparative and ablation results, discusses limitations and ethics, and concludes with future deployment directions.
