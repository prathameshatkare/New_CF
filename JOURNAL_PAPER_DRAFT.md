# Federated Learning for Rare Disease Diagnosis: Edge-Deployable Cystic Fibrosis Screening with Synthetic Minority Augmentation

## Abstract
Cystic fibrosis (CF) is a rare genetic disease where early detection can improve outcomes, but data scarcity and privacy constraints limit robust model development. This work presents an end-to-end framework for CF screening that combines synthetic minority augmentation and federated learning for edge-ready deployment. A cleaned clinical dataset was constructed with six features (`age`, `sex`, `height`, `weight`, `fev1`, `BMI`) and a binary target (`CF` vs. `non-CF`). Minority augmentation was performed using a TVAE-based tabular synthesizer. We then trained a research-grade federated model with non-IID Dirichlet client simulation, partial client participation, FedProx local objective, and weighted FedAvg aggregation. The final federated model achieved Accuracy 0.9904, Precision 0.8065, Recall 1.0000, F1-score 0.8929, ROC-AUC 1.0000, and PR-AUC 1.0000 on the held-out test split. Federated ablation showed a strong dependence on lung-function feature `fev1` (F1 drops to 0.3802 and PR-AUC to 0.4970 when removed). An edge-ready inference pipeline and multi-section web interface were implemented for practical use. The study demonstrates a feasible privacy-preserving CF screening workflow, while emphasizing the need for broader external validation due to near-perfect separability in current data.

## Keywords
Federated learning, cystic fibrosis, rare disease, edge AI, synthetic data, TVAE, FedProx, clinical screening

## 1. Introduction
Rare disease diagnosis is often constrained by class imbalance, limited labeled samples, and strict privacy requirements in healthcare systems. CF screening is especially sensitive to these issues because patient cohorts are small relative to control populations. Traditional centralized learning may conflict with data-governance constraints and institution-level policies.

Federated learning (FL) offers a privacy-aware alternative by training local models and aggregating parameters instead of raw data. However, FL on rare-disease tasks remains challenging due to extreme imbalance and client heterogeneity. This paper addresses these challenges by integrating:
1. Synthetic minority data augmentation,
2. Robust federated optimization (FedProx + weighted FedAvg),
3. Edge-oriented inference and UI deployment.

## 2. Contributions
1. A full CF screening pipeline from preprocessing to deployment.
2. TVAE-based synthetic minority augmentation for rare-class enrichment.
3. Research-grade FL training with non-IID partitioning, FedProx, and weighted aggregation.
4. Federated feature ablation analysis showing feature sensitivity and model reliance.
5. Practical edge-deployable API and UI for screening support.

## 3. Materials and Methods

### 3.1 Data and Features
- Final feature set: `age`, `sex`, `height`, `weight`, `fev1`, `BMI`
- Label: `target` (`0` non-CF, `1` CF)
- Cleaned dataset path: `cleaned_cf_dataset.csv`
- Augmented dataset path: `augmented_cf_dataset.csv`

Observed class distribution:
- Cleaned: 3000 non-CF, 25 CF
- Augmented: 3000 non-CF, 125 CF

### 3.2 Synthetic Minority Augmentation
A TVAE-based tabular synthesizer was trained on the CF minority subset and used to generate additional CF samples. Synthetic quality was analyzed with distribution checks and distinguishability tests in the project pipeline.

### 3.3 Federated Training Setup
The federated model (`federated_pytorch.py`) uses:
- Non-IID client simulation via class-wise Dirichlet partition (`alpha=0.6`)
- Partial participation (`client_fraction=0.8`)
- FedProx regularization (`mu=0.01`)
- Weighted FedAvg by client sample count
- Early stopping on validation AUC

Key run configuration (final):
- Rounds: 20
- Clients: 5
- Local epochs: 4
- Batch size: 64
- Learning rate: 0.0008

### 3.4 Evaluation Protocol
Reported metrics:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC and PR-AUC

Additional diagnostics:
- Label-shuffle sanity test
- Duplicate-conflict check
- Single-feature AUC
- Stress-test scenarios (noise and shift)
- Federated leave-one-feature-out ablation

## 4. Results

### 4.1 Main Federated Performance
From `best_federated_cf_metrics.json`:

| Metric | Value |
|---|---:|
| Accuracy | 0.9904 |
| Precision | 0.8065 |
| Recall | 1.0000 |
| F1-score | 0.8929 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |

### 4.2 Federated Feature Ablation
From `federated_ablation_results.csv`:

| Experiment | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| All features | 0.8929 | 1.0000 | 1.0000 |
| Drop `age` | 0.9804 | 0.9999 | 0.9969 |
| Drop `sex` | 0.9796 | 0.9999 | 0.9985 |
| Drop `height` | 0.8065 | 1.0000 | 1.0000 |
| Drop `weight` | 0.8621 | 1.0000 | 1.0000 |
| Drop `BMI` | 0.9615 | 1.0000 | 1.0000 |
| Drop `fev1` | **0.3802** | **0.9487** | **0.4970** |

Interpretation: `fev1` is the most critical predictive node in this dataset.

### 4.3 Baseline Comparison
From `model_comparison_results_final.csv`, classical baselines on cleaned and augmented splits reached near-perfect holdout scores (mostly 1.0000), indicating extremely strong separability in the current feature-label distribution.

### 4.4 Leakage and Robustness Diagnostics
From `model_leakage_diagnostics.csv`:
- Conflicting duplicate feature rows: 0
- Label-shuffle ROC-AUC mean: ~0.52 (near random)
- Single-feature AUC for `fev1`: 1.0

From `robustness_stress_test_results.csv`:
- All tested scenarios remained near-perfect on this dataset (clean/noise/shift variants).

## 5. Deployment
The system includes:
- Edge inference app (`edge_device_app.py`) from federated checkpoint
- Web API (`api_server.py`) and enhanced UI (`web/`) with pages for:
  - Prediction
  - Cure & Prevention
  - CF Knowledge
  - Global Research
  - Edge Deployment
  - Nodes Info

Figures:
- `federated_ablation_plot.png`
- `federated_ablation_plot_paper_bw.png`
- `federated_ablation_plot_paper_bw.pdf`
- `federated_ablation_plot_paper_bw.svg`

## 6. Discussion
The proposed approach demonstrates that FL with synthetic minority augmentation can produce strong screening performance while preserving a deployment pathway to edge environments. The ablation findings are clinically coherent: removing `fev1` substantially degrades discrimination and minority precision-recall behavior. This supports the central role of pulmonary-function measurements in CF risk estimation.

At the same time, near-perfect scores across multiple settings indicate that the current dataset may be highly separable. Therefore, high internal performance should not be interpreted as guaranteed external generalization.

## 7. Limitations
1. Very small real minority class before augmentation.
2. Strong feature separability may overestimate real-world generalization.
3. External multi-center validation is not yet included.
4. Threshold calibration should be adapted per deployment population.

## 8. Conclusion
This work provides a complete, reproducible, and edge-deployable framework for rare-disease CF screening using federated learning and synthetic minority augmentation. Results are strong in current experiments, with ablation evidence highlighting the importance of `fev1`. Future work should prioritize external validation, domain-shift testing, and clinical calibration across institutions.

## 9. Reproducibility
Run full project:
```powershell
.\venv\Scripts\python.exe run_project_pipeline.py
```

Run web UI:
```powershell
.\venv\Scripts\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 54911
```

## 10. References (Template)
Add journal-formatted references for:
1. Cystic fibrosis clinical background and epidemiology
2. Federated learning foundational methods (FedAvg)
3. Heterogeneous FL methods (FedProx)
4. Tabular synthetic data generation (TVAE / CTGAN family)
5. CF registry/diagnosis program reports by country

---
### Suggested Appendix Items for Submission
- A. Full hyperparameter table
- B. Client-wise class distribution table
- C. Additional ablation and threshold plots
- D. Ethical and deployment statement
