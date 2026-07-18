# Day 1 Setup

Day 1 prepares the project so the rest of the bootcamp can move cleanly.

## Current Setup Choice

We are using:

- Windows PowerShell
- Python virtual environment with `venv`
- `pip` requirements files
- Git and GitHub
- Docker Desktop

## Commands

Run these commands from the project root:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Verify The Environment

```powershell
python -c "import pandas, numpy, sklearn, fastapi; print('Environment is ready')"
pytest
ruff check .
```

## MLflow Note

MLflow belongs to Week 3. It has a larger dependency tree, so it is separated into:

```powershell
python -m pip install -r requirements-mlops.txt
```

If MLflow installation is slow or hangs on Windows, continue with Week 1 and return to MLflow later.
