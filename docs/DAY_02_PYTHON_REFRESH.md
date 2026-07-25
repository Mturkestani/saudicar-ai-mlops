# Day 2 Python For ML Refresh

Day 2 makes sure the basic Python workflow is comfortable before the project starts
using pandas, notebooks, model training, and APIs.

## Goal

By the end of this day, you should be able to:

- Activate the project virtual environment.
- Install project packages with `pip`.
- Run Python code from the command line.
- Read a small Python module that uses functions.
- Write and read a local JSON file.
- Run tests with `pytest`.

## Why This Matters For MLOps

MLOps is not only model training. A production ML project is still a software
project. We need functions for reusable logic, files for data and artifacts,
virtual environments for repeatable setup, packages for ML tools, and tests so
changes do not quietly break the project.

## Activate The Environment

From the project root:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
```

If `.venv` does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Check Installed Packages

```powershell
python -m pip list
python -c "import pandas, numpy, sklearn, fastapi; print('packages ready')"
```

The important idea is that the project depends on packages listed in
`requirements.txt` and `requirements-dev.txt`, not on whatever happens to be
installed globally on the computer.

## Run The Day 2 Practice Module

```powershell
$env:PYTHONPATH = "src"
python -m saudi_car_ai.day02_refresh
```

Expected result:

- The terminal prints a short sample car summary.
- A local file is created at `data/processed/day02_sample_car.json`.
- The output file is ignored by Git because `data/processed/*` is ignored.

## What The Module Teaches

The file `src/saudi_car_ai/day02_refresh.py` demonstrates:

- `dict`: store one car listing.
- `function`: calculate car age and build a summary.
- `Path`: create folders and point to files safely.
- `json`: save and load simple structured data.
- `importlib`: verify that installed packages can be imported.
- `if __name__ == "__main__"`: allow a file to be imported or run as a script.

## Run Tests

```powershell
pytest
```

The Day 2 tests prove that:

- Car age calculation works.
- A summary contains the expected fields.
- JSON writing and reading round-trip correctly.
- Required packages can be imported.

## Mini Quiz

1. Why do we use a virtual environment instead of installing everything globally?
2. Why is a function better than copying the same calculation many times?
3. Why should generated data files usually stay out of Git?
4. What does `PYTHONPATH=src` help Python find?
5. Why is testing a small utility useful before the ML model exists?

## Done Checklist

- Environment activates successfully.
- `python -m saudi_car_ai.day02_refresh` runs.
- `data/processed/day02_sample_car.json` is created locally.
- `pytest` passes.
- You can explain functions, files, virtual environments, and package imports in
  simple words.
