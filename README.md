# Federated Learning for Rare Disease: Cystic Fibrosis Diagnosis

This repository implements an end-to-end pipeline for CF diagnosis with:
- preprocessing and statistical validation,
- minority augmentation via TVAE (GAN-family tabular synthesis),
- federated learning with FedProx + weighted FedAvg,
- model diagnostics (leakage checks, ablation, stress tests),
- publication-ready visualizations and edge inference.

## Documentation
- Paper structure template: `PAPER_STRUCTURE_GUIDE.md`
- Filled draft (title + abstract + introduction): `PAPER_ABSTRACT_INTRO_DRAFT.md`
- Project startup and run instructions: `PROJECT_START_GUIDE.md`

## Core schema
- Features: `age, sex, height, weight, fev1, BMI`
- Label: `target` (`0` non-CF, `1` CF)

## One-command execution
```powershell
.\venv\Scripts\python.exe run_project_pipeline.py
```

## Manual execution order
```powershell
.\venv\Scripts\python.exe data_preprocessing_and_stat_validation.py
.\venv\Scripts\python.exe data_augmentation_hybrid.py
.\venv\Scripts\python.exe federated_data_partition.py
.\venv\Scripts\python.exe federated_pytorch.py --checkpoint-path best_federated_cf_model.pt --metrics-out best_federated_cf_metrics.json
.\venv\Scripts\python.exe federated_ablation_runner.py --python .\venv\Scripts\python.exe
.\venv\Scripts\python.exe model_comparison.py
.\venv\Scripts\python.exe plot_federated_ablation.py
```

## Edge inference (checkpoint-based)
```powershell
.\venv\Scripts\python.exe edge_device_app.py --checkpoint best_federated_cf_model.pt
```

## Web UI (HTML/CSS/JS + React + FastAPI)
```powershell
.\venv\Scripts\python.exe run_web_ui.py
```

Then open:
- `http://127.0.0.1:8600/`

If port `8600` is busy:
```powershell
.\venv\Scripts\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 54911
```

## Main outputs
- `best_federated_cf_model.pt`
- `best_federated_cf_metrics.json`
- `federated_ablation_results.csv`
- `model_comparison_results_updated.csv`
- `model_comparison_results_final.csv`
- `model_leakage_diagnostics.csv`
- `feature_ablation_results.csv`
- `threshold_calibration_results.csv`
- `robustness_stress_test_results.csv`
- `federated_ablation_plot.png`
- `federated_ablation_plot_paper_bw.png`
- `federated_ablation_plot_paper_bw.pdf`
- `federated_ablation_plot_paper_bw.svg`
