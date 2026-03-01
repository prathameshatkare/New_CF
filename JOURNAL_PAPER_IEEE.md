# Federated Learning for Rare Disease Diagnosis: Edge-Deployable Cystic Fibrosis Screening with Synthetic Minority Augmentation

## Abstract
Cystic fibrosis (CF) screening suffers from data scarcity, class imbalance, and strict privacy constraints. This study presents a complete pipeline that combines synthetic minority augmentation with federated learning (FL) for edge deployment. A cleaned dataset was built using six features (`age`, `sex`, `height`, `weight`, `fev1`, `BMI`) and a binary target (`CF` vs. `non-CF`). Minority enrichment used a TVAE tabular synthesizer. FL training used non-IID Dirichlet client simulation, partial client participation, FedProx local optimization, and weighted FedAvg aggregation. On the held-out split, the federated model achieved Accuracy 0.9904, Precision 0.8065, Recall 1.0000, F1-score 0.8929, ROC-AUC 1.0000, and PR-AUC 1.0000. Feature ablation identified `fev1` as dominant: removing it reduced F1 to 0.3802 and PR-AUC to 0.4970. An edge-ready inference API and multi-section web interface were implemented. The framework is reproducible and deployable, while further external multi-center validation is required due to high internal separability.

**Index Terms**—Federated learning, cystic fibrosis, rare disease, edge AI, synthetic data, FedProx, TVAE.

## I. INTRODUCTION
Rare disease AI systems are constrained by small minority cohorts and privacy-sensitive medical data. CF is a representative case where early risk stratification is useful, but centralized data pooling is often limited. FL addresses this by training on decentralized data and sharing model updates instead of raw records. However, FL under extreme imbalance and client heterogeneity remains difficult.

This work addresses these challenges through synthetic minority augmentation and robust FL optimization, then carries the model to an edge-compatible inference stack.

## II. CONTRIBUTIONS
1. End-to-end CF pipeline from preprocessing to edge UI deployment.
2. Minority-class enrichment using TVAE synthetic tabular generation.
3. Robust FL setup with non-IID partitioning, FedProx, and weighted FedAvg.
4. Federated feature ablation and diagnostic analysis.
5. Practical edge API + multi-page clinical screening UI.

## III. MATERIALS AND METHODS

### A. Data and Feature Space
Final schema:
- Features: `age`, `sex`, `height`, `weight`, `fev1`, `BMI`
- Label: `target` (`0` non-CF, `1` CF)

Class distribution:
- Cleaned: 3000 non-CF, 25 CF
- Augmented: 3000 non-CF, 125 CF

### B. Synthetic Minority Augmentation
TVAE was trained on minority CF rows and used to generate additional synthetic CF samples. Quality checks were integrated in the data pipeline.

### C. Federated Learning Configuration
Training setup (`federated_pytorch.py`):
- Non-IID client allocation: Dirichlet (`alpha=0.6`)
- Clients: 5, participation fraction: 0.8
- FedProx coefficient: `mu=0.01`
- Aggregation: weighted FedAvg
- Rounds: 20, local epochs: 4, batch size: 64
- Early stopping on validation ROC-AUC

### D. Evaluation and Diagnostics
Primary metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.

Additional checks:
- Label-shuffle sanity test
- Duplicate conflict inspection
- Single-feature discriminative analysis
- Stress-test scenarios
- Federated leave-one-feature-out ablation

## IV. RESULTS

### A. Main Federated Model Performance
From `best_federated_cf_metrics.json`:

| Metric | Value |
|---|---:|
| Accuracy | 0.9904 |
| Precision | 0.8065 |
| Recall | 1.0000 |
| F1-score | 0.8929 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |

### B. Federated Feature Ablation
From `federated_ablation_results.csv`:

| Setting | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| All features | 0.8929 | 1.0000 | 1.0000 |
| Drop `age` | 0.9804 | 0.9999 | 0.9969 |
| Drop `sex` | 0.9796 | 0.9999 | 0.9985 |
| Drop `height` | 0.8065 | 1.0000 | 1.0000 |
| Drop `weight` | 0.8621 | 1.0000 | 1.0000 |
| Drop `BMI` | 0.9615 | 1.0000 | 1.0000 |
| Drop `fev1` | **0.3802** | **0.9487** | **0.4970** |

Removing `fev1` causes the largest drop and indicates strong dependence on pulmonary function signal.

### C. Diagnostics and Robustness
From `model_leakage_diagnostics.csv`:
- Conflicting duplicate rows: 0
- Label-shuffle ROC-AUC: ~0.52 (near random baseline)
- Single-feature AUC for `fev1`: 1.0

From `robustness_stress_test_results.csv`:
- Clean/noise/shift scenarios remained near-perfect on current splits.

## V. EDGE DEPLOYMENT
Implemented components:
- Checkpoint-based edge inference (`edge_device_app.py`)
- FastAPI backend (`api_server.py`)
- Multi-page web frontend (`web/`) with prediction, prevention, knowledge, research, and model-node pages.

Visualization artifacts:
- `federated_ablation_plot_paper_bw.pdf`
- `federated_ablation_plot_paper_bw.svg`

## VI. DISCUSSION
The proposed method demonstrates high screening performance under privacy-aware training and practical deployment constraints. Ablation confirms clinical relevance of `fev1`. However, near-perfect internal metrics indicate that external generalization should be validated on independent cohorts and multi-center distributions.

## VII. LIMITATIONS
1. Limited real minority samples prior to augmentation.
2. Strong internal separability may overestimate external performance.
3. No external multi-center validation yet.
4. Threshold tuning may vary across populations and devices.

## VIII. CONCLUSION
This study provides a complete and reproducible FL-based CF screening framework with synthetic minority augmentation and edge deployment readiness. Results are strong on current data, and feature sensitivity analysis identifies `fev1` as the dominant predictor. External validation is the key next step toward clinical translation.

## REFERENCES (TO FORMAT)
[1] Core CF clinical background references.  
[2] FedAvg foundational paper.  
[3] FedProx heterogeneous FL paper.  
[4] TVAE/CTGAN synthetic tabular modeling references.  
[5] National/international CF registry studies.
