# Project Completion Report

## Title
Federated Learning for Rare Disease Cystic Fibrosis Diagnosis Using Edge Device

## Completed Scope
- Data preprocessing and statistical validation pipeline.
- Minority-class augmentation using TVAE (GAN-family tabular synthesizer).
- Federated learning with:
  - non-IID Dirichlet client simulation,
  - partial client participation,
  - FedProx local objective,
  - weighted FedAvg aggregation,
  - early stopping and checkpointing.
- Federated feature ablation experiments (all features + leave-one-feature-out).
- Baseline model comparison with repeated stratified CV.
- Leakage diagnostics and stress-test scenarios.
- Edge-device inference using trained federated checkpoint (not a separate retrained model).
- Publication-ready ablation plots in PNG/PDF/SVG.
- One-command full pipeline runner.

## Key Evidence

### Federated Ablation (from `federated_ablation_results.csv`)
- `all_features`: F1 `0.8929`, ROC-AUC `1.0000`, PR-AUC `1.0000`
- `drop_fev1`: F1 `0.3802`, ROC-AUC `0.9487`, PR-AUC `0.4970`

Interpretation:
- `fev1` is the dominant feature signal in this dataset.

### Classical Diagnostics
- Label-shuffle AUC is near random (`~0.52`) in `model_leakage_diagnostics.csv`.
- No conflicting duplicate feature rows were found.
- Stress tests in `robustness_stress_test_results.csv` remain near-perfect on this current dataset split.

## Final Artifacts
- Training / inference:
  - `federated_pytorch.py`
  - `edge_device_app.py`
  - `federated_ablation_runner.py`
  - `run_project_pipeline.py`
- Evaluation:
  - `model_comparison.py`
  - `federated_ablation_results.csv`
  - `model_comparison_results_updated.csv`
  - `model_comparison_results_final.csv`
  - `model_leakage_diagnostics.csv`
  - `feature_ablation_results.csv`
  - `threshold_calibration_results.csv`
  - `robustness_stress_test_results.csv`
- Figures:
  - `federated_ablation_plot.png`
  - `federated_ablation_plot_paper_bw.png`
  - `federated_ablation_plot_paper_bw.pdf`
  - `federated_ablation_plot_paper_bw.svg`
- Environment:
  - `requirements_project.txt`

## Reproducibility
Run full project:
```powershell
.\venv\Scripts\python.exe run_project_pipeline.py
```

Run edge inference from federated checkpoint:
```powershell
.\venv\Scripts\python.exe edge_device_app.py --checkpoint best_federated_cf_model.pt
```
