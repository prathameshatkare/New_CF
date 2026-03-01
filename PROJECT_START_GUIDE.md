# Project Start Guide

This guide explains how to start and run the Cystic Fibrosis federated learning project in `c:\Users\ASUS\Desktop\mnt`.

## 1) Prerequisites

- Windows PowerShell
- Python 3.12 (recommended to match local `venv`)
- Optional: GPU-enabled PyTorch if you want faster training

## 2) Open the project folder

```powershell
cd c:\Users\ASUS\Desktop\mnt
```

## 3) Create and activate virtual environment (if needed)

If `venv` already exists, you can skip creation.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 4) Install dependencies

```powershell
.\venv\Scripts\pip.exe install -r requirements_project.txt
```

## 5) Quick health check

```powershell
.\venv\Scripts\python.exe -m py_compile app_ui.py federated_pytorch.py run_project_pipeline.py api_server.py
```

## 6) Run full pipeline (recommended)

```powershell
.\venv\Scripts\python.exe run_project_pipeline.py
```

Optional (force full regeneration):

```powershell
.\venv\Scripts\python.exe run_project_pipeline.py --full-regeneration
```

Optional (custom rounds/epochs):

```powershell
.\venv\Scripts\python.exe run_project_pipeline.py --federated-rounds 20 --federated-local-epochs 4 --ablation-rounds 8 --ablation-local-epochs 2
```

## 7) Run modules manually (advanced)

```powershell
.\venv\Scripts\python.exe data_preprocessing_and_stat_validation.py
.\venv\Scripts\python.exe data_augmentation_hybrid.py
.\venv\Scripts\python.exe federated_data_partition.py
.\venv\Scripts\python.exe federated_pytorch.py --checkpoint-path best_federated_cf_model.pt --metrics-out best_federated_cf_metrics.json
.\venv\Scripts\python.exe federated_ablation_runner.py --python .\venv\Scripts\python.exe
.\venv\Scripts\python.exe model_comparison.py
.\venv\Scripts\python.exe plot_federated_ablation.py
```

## 8) Run edge inference app

```powershell
.\venv\Scripts\python.exe edge_device_app.py --checkpoint best_federated_cf_model.pt
```

## 9) Launch Streamlit UI

```powershell
.\venv\Scripts\python.exe -m streamlit run app_ui.py --server.headless true --server.port 8510
```

Open:

- `http://127.0.0.1:8510`

## 10) Launch Web UI (FastAPI + static frontend helper)

```powershell
.\venv\Scripts\python.exe run_web_ui.py
```

Open:

- `http://127.0.0.1:8600/`

If you want to run API only:

```powershell
.\venv\Scripts\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 54911
```

API health endpoint:

- `http://127.0.0.1:54911/api/health`

## 11) Key output artifacts

- `best_federated_cf_model.pt`
- `best_federated_cf_metrics.json`
- `federated_ablation_results.csv`
- `model_comparison_results_final.csv`
- `threshold_calibration_results.csv`
- `robustness_stress_test_results.csv`
- `federated_ablation_plot.png`
- `federated_ablation_plot_paper_bw.pdf`

## 12) Common troubleshooting

1. `ModuleNotFoundError`:
   - Re-activate environment and reinstall requirements.
2. Port already in use:
   - Change `--server.port` for Streamlit or Uvicorn.
3. Missing checkpoint:
   - Run `run_project_pipeline.py` first to generate `best_federated_cf_model.pt`.
4. Slow training:
   - Reduce rounds/epochs or run with a GPU-enabled setup.
