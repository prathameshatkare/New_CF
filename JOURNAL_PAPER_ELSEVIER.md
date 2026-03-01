# Federated learning for rare disease diagnosis: edge-deployable cystic fibrosis screening with synthetic minority augmentation

## Highlights
- End-to-end cystic fibrosis screening pipeline using federated learning and synthetic minority augmentation.
- Non-IID federated training with FedProx and weighted FedAvg for privacy-aware robust optimization.
- Strong held-out performance (Accuracy 0.9904, F1 0.8929, ROC-AUC 1.0000, PR-AUC 1.0000).
- Feature ablation identifies FEV1 as the dominant predictive variable.
- Practical edge deployment via local API and multi-page clinical web UI.

## Abstract
Rare disease diagnosis is constrained by data scarcity, class imbalance, and privacy restrictions. We propose a complete framework for cystic fibrosis (CF) screening that integrates synthetic minority augmentation and federated learning (FL), then deploys inference at the edge. A cleaned dataset with six features (`age`, `sex`, `height`, `weight`, `fev1`, `BMI`) and binary target labels was augmented using a TVAE tabular synthesizer. The federated model was trained under non-IID client simulation with FedProx local objective and weighted FedAvg aggregation. The final model achieved Accuracy 0.9904, Precision 0.8065, Recall 1.0000, F1-score 0.8929, ROC-AUC 1.0000, and PR-AUC 1.0000. Federated ablation demonstrated substantial reliance on `fev1`: removing it reduced F1 to 0.3802 and PR-AUC to 0.4970. We implemented edge inference and a production-style web interface for practical screening use. The framework is reproducible and deployment-oriented, with future work focused on external multi-center validation and domain-shift calibration.

## 1. Introduction
AI for rare diseases faces two persistent barriers: (i) very limited positive samples and (ii) strict privacy constraints for patient data sharing. In CF screening, these constraints can reduce both model robustness and translational utility. FL offers a viable path by training decentralized local models and sharing only model updates.

Still, FL for rare-disease classification remains difficult due to class imbalance and heterogeneous client distributions. This study addresses both through synthetic minority augmentation and robust federated optimization, followed by edge-ready inference deployment.

## 2. Materials and methods

### 2.1 Dataset and feature definition
Feature set: `age`, `sex`, `height`, `weight`, `fev1`, `BMI`  
Label: `target` (`0` non-CF, `1` CF)

Class counts:
- Cleaned dataset: 3000 non-CF, 25 CF
- Augmented dataset: 3000 non-CF, 125 CF

### 2.2 Minority augmentation
We used TVAE on minority CF records to generate synthetic tabular samples and merged them into the training corpus.

### 2.3 Federated training
Federated configuration:
- 5 clients, non-IID Dirichlet partition (`alpha=0.6`)
- Client participation fraction: 0.8
- FedProx regularization (`mu=0.01`)
- Weighted FedAvg aggregation
- 20 rounds, 4 local epochs, batch size 64
- Early stopping on validation ROC-AUC

### 2.4 Evaluation
Primary metrics:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, PR-AUC

Secondary diagnostics:
- label-shuffle sanity test
- duplicate-conflict check
- single-feature AUC analysis
- stress testing under perturbations
- federated leave-one-feature-out ablation

## 3. Results

### 3.1 Main federated performance
Using `best_federated_cf_metrics.json`:

| Metric | Value |
|---|---:|
| Accuracy | 0.9904 |
| Precision | 0.8065 |
| Recall | 1.0000 |
| F1-score | 0.8929 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |

### 3.2 Federated ablation
Using `federated_ablation_results.csv`:

| Experiment | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| All features | 0.8929 | 1.0000 | 1.0000 |
| Drop age | 0.9804 | 0.9999 | 0.9969 |
| Drop sex | 0.9796 | 0.9999 | 0.9985 |
| Drop height | 0.8065 | 1.0000 | 1.0000 |
| Drop weight | 0.8621 | 1.0000 | 1.0000 |
| Drop BMI | 0.9615 | 1.0000 | 1.0000 |
| Drop FEV1 | **0.3802** | **0.9487** | **0.4970** |

The strongest degradation occurs when removing `fev1`, indicating dominant predictive relevance.

### 3.3 Diagnostics
From `model_leakage_diagnostics.csv`:
- Conflicting duplicate feature rows: 0
- Label-shuffle ROC-AUC mean: approximately 0.52
- Single-feature AUC for `fev1`: 1.0

From `robustness_stress_test_results.csv`:
- Performance remained near-perfect under tested perturbations on current splits.

## 4. Discussion
The proposed framework shows that privacy-aware FL can be coupled with synthetic minority augmentation to produce high-performance CF screening and practical edge deployment. The feature-ablation results are clinically plausible and emphasize the value of pulmonary-function inputs.

Near-perfect metrics in multiple tests suggest strong separability in available data; therefore, broader external validation is essential before clinical adoption.

## 5. Conclusion
This work delivers a reproducible CF screening pipeline spanning preprocessing, augmentation, robust federated training, diagnostics, and edge deployment. Future efforts should focus on multi-center external validation, calibration under domain shift, and clinical workflow integration.

## Data and code availability
All implemented scripts, output tables, figures, and deployment components are available in this project workspace.

## Conflict of interest
The authors declare no known competing financial interests or personal relationships that could have appeared to influence this work.

## Acknowledgments
This project integrates federated-learning methodology, synthetic tabular augmentation, and edge deployment engineering for rare-disease screening.

## Suggested references (to format in Elsevier style)
1. Foundational cystic fibrosis clinical references.
2. Federated Averaging (FedAvg).
3. FedProx for heterogeneous federated optimization.
4. TVAE / synthetic tabular data modeling references.
5. National and international CF registry studies.
