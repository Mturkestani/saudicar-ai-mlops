# Day 7 Baseline Model

Day 7 is the day the project finally becomes machine learning.

Until now we loaded data, explored it, cleaned it, and built features. Today we
train our first model that predicts a car price from those features, and we
measure how good (or bad) it is. This first model is called the **baseline**.

## Big Idea

A baseline is the simplest reasonable model. We do not try to win yet. We try to
get a number to beat.

Everything we build later (Random Forest on Day 8, better features, tuning) must
be compared against this baseline. If a fancy model cannot beat a simple Linear
Regression, the complexity is not worth it.

```text
Day 7: Linear Regression  ->  a score we must remember
Day 8: Better model        ->  must beat Day 7
```

## Why Linear Regression First

Linear Regression is the "hello world" of prediction models.

- It is fast and easy to understand.
- It draws a straight-line relationship between features and price.
- It gives us honest, explainable numbers.

It will not be perfect. Car prices are not perfectly linear. That is fine. A
baseline is a starting line, not a finish line.

## Train And Test Split

We never grade a model on the same data it studied. That would be like giving a
student the exam answers before the test.

So we split the feature table into two parts:

```text
Training set (80%)  ->  the model learns from this
Test set (20%)      ->  we grade the model on this, unseen data
```

We fix `random_state=42` so the split is the same every time. Reproducibility is
a core MLOps habit: the same command should give the same result.

## The Metrics: MAE, RMSE, R2

After training, we ask three questions about the test set.

### MAE (Mean Absolute Error)

The average size of our mistakes, in Saudi Riyal.

```text
MAE = 26,000  means  on average we are off by about 26,000 SAR
```

Lower is better. MAE is the easiest metric to explain to a non-technical person.

### RMSE (Root Mean Squared Error)

Also an error in SAR, but it punishes big misses more than small ones. If RMSE
is much larger than MAE, the model is making some large errors on a few cars.

Lower is better.

### R2 (R-squared)

The share of price variation the model can explain, from 0 to 1.

```text
R2 = 0.00  ->  no better than always guessing the average price
R2 = 1.00  ->  perfect predictions
R2 = 0.59  ->  explains about 59% of the price variation
```

Higher is better. For a first linear baseline on messy real-world car data, a
mid-range R2 is a normal and honest starting point.

## Inputs And Outputs

Day 7 reads the encoded Day 6 feature table:

```text
data/processed/saudi_used_cars_features.csv
```

It writes two local artifacts:

```text
models/baseline_linear_regression.joblib   (the trained model)
models/baseline_metrics.json               (the scores, for later comparison)
```

Both are ignored by Git, just like the data. Models are outputs, not source
code. We can always retrain them from the pipeline.

## Run Day 6 First

If the feature CSV does not exist yet:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.clean
python -m saudi_car_ai.features.build_features
```

## Run The Training Script

```powershell
$env:PYTHONPATH = "src"
python -m saudi_car_ai.models.train
```

The script prints:

- Training and test row counts.
- How many feature columns the model used.
- MAE, RMSE, and R2 on the test set.
- Where the model and metrics were saved.

## Day 7 Code

The training code lives in:

```text
src/saudi_car_ai/models/train.py
```

Important functions:

- `load_feature_dataset`: reads the Day 6 encoded features.
- `split_train_test`: makes a reproducible 80/20 split and separates `X` and `y`.
- `train_baseline_model`: fits the Linear Regression baseline.
- `evaluate_model`: scores the model on the held-out test set (MAE, RMSE, R2).
- `train_and_evaluate`: runs the full workflow end to end.
- `save_model` / `save_metrics`: write the model artifact and metrics JSON.

## Mini Lesson: Why Save The Metrics As JSON

We save the scores to `models/baseline_metrics.json` on purpose.

Later, when we add MLflow, we will track many experiments and compare them.
Saving metrics in a structured file is the beginner version of that idea: a
written record of "how good was this run" that we can look back on instead of
trusting our memory.

## Important Warning: Do Not Touch The Test Set

The test set exists to give an honest grade. If we look at the test set, tune the
model to it, and repeat, we are secretly teaching to the exam. The model will
look great in our numbers and fail on real cars.

Rule: fit only on training data, grade only on test data.

## What We Are Not Doing Yet

- We are not using a Random Forest or Gradient Boosting yet. That is Day 8.
- We are not tuning hyperparameters.
- We are not building a saved preprocessing pipeline for the API yet.

Today the goal is a working, honest baseline and a score to beat.

## Done Checklist

- Feature dataset exists from Day 6.
- Data is split into train and test with a fixed random seed.
- Linear Regression baseline is trained on the training set only.
- MAE, RMSE, and R2 are measured on the test set.
- Model artifact and metrics JSON are saved locally.
- Tests pass.
