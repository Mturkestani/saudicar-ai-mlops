# Day 9 Train/Test Workflow

Day 9 turns our scattered model experiments into one repeatable training run.

On Day 7 we trained a baseline. On Day 8 we compared it against a Random Forest.
Both were useful, but they were separate scripts with separate outputs. Today we
build a single command that trains the candidates, picks the best one, and saves
everything needed to trust and reproduce the result.

## Big Idea

A real ML system does not rely on someone remembering which script to run or
which model was best. It relies on a **repeatable workflow**.

```text
one command  ->  fixed split  ->  train candidates  ->  pick best  ->  save outputs
```

Run it today, run it next month, and you get the same answer. That property is
called **reproducibility**, and it is the foundation everything else (MLflow,
CI/CD, deployment) is built on.

## Why This Matters

Imagine coming back to this project in three months. Without a repeatable
workflow you would ask:

- Which split did I use?
- Which model did I actually deploy?
- What were its exact scores?

Day 9 answers all three automatically by writing a **training run manifest** — a
small record of exactly what happened.

## What The Workflow Does

`run_training` performs the whole flow in memory:

1. Splits the features into train and test with a fixed `random_state = 42`.
2. Trains both candidate models (baseline and Random Forest) on the same split.
3. Scores each on the held-out test set.
4. Selects the model with the **lowest MAE**.
5. Returns the winning model plus a `TrainingRun` record.

Nothing is guessed. The best model is chosen from the numbers.

## Inputs And Outputs

Day 9 reads the encoded Day 6 feature table:

```text
data/processed/saudi_used_cars_features.csv
```

It writes two local artifacts:

```text
models/production_model.joblib   (the selected model, ready to serve)
models/training_run.json         (the reproducible run manifest)
```

Both are git-ignored, like every model artifact in this project.

## The Run Manifest

The manifest is the heart of Day 9. It captures everything needed to understand
and repeat the run:

```json
{
  "dataset_path": "data\\processed\\saudi_used_cars_features.csv",
  "total_rows": 5389,
  "train_rows": 4311,
  "test_rows": 1078,
  "feature_count": 467,
  "test_size": 0.2,
  "random_state": 42,
  "selected_model": "random_forest",
  "selected_metrics": { "mae": 12203, "rmse": 23104, "r2": 0.883 },
  "all_metrics": { "baseline": { }, "random_forest": { } },
  "created_at": "2026-07-25T16:27:06+00:00"
}
```

This is the beginner version of experiment tracking. On Day 20 we replace it with
MLflow, but the idea is identical: never lose the record of a run.

## Run The Workflow

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.models.run_training
```

The script prints the split settings, the selected model, its metrics, and where
everything was saved.

## Proving It Is Repeatable

Run the command twice. The selected model and its metrics are identical every
time, because the split seed is fixed. That is the whole point: same command,
same result.

The tests enforce this too — `test_run_training_is_reproducible` runs the
workflow twice and checks the scores match.

## Day 9 Code

The workflow lives in:

```text
src/saudi_car_ai/models/run_training.py
```

It reuses the Day 7 and Day 8 building blocks (`split_train_test`,
`train_baseline_model`, `train_random_forest`, `evaluate_model`) so there is no
duplicated logic. Important pieces:

- `run_training`: trains candidates on one split and selects the best by MAE.
- `TrainingRun`: the dataclass that records the run.
- `save_production_model` / `save_training_run`: write the model and the manifest.

## Mini Lesson: What "Production Model" Starts To Mean

We saved the winner as `production_model.joblib`. That name is a promise: this is
the model the rest of the system (the API on Day 14) will load and serve. From
now on, the API does not care whether the winner was Linear Regression or Random
Forest — it just loads the production model. Day 22 formalizes this idea.

## What We Are Not Doing Yet

- We are not tuning model settings yet.
- We are not tracking many runs over time yet — that is MLflow on Day 20.
- We are not serving the model yet — that is the FastAPI work in Week 2.

Today's win is a single, trustworthy command that always produces the same model
and a written record of how.

## Done Checklist

- One command runs the full training workflow.
- The split is fixed, so results are reproducible.
- The best model is selected from the metrics, not by hand.
- The selected model is saved as the production artifact.
- A run manifest records the dataset, split, model, and scores.
- Tests pass, including a reproducibility test.
