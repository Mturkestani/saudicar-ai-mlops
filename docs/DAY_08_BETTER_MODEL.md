# Day 8 Better Model

Day 8 is the day we prove that a smarter model is worth the extra complexity.

On Day 7 we trained a Linear Regression baseline. Today we train a **Random
Forest** and compare it against that baseline on the exact same test cars. The
goal is not just "use a fancier model" — it is to show, with numbers, that the
fancier model actually predicts prices better.

## Big Idea

A baseline gives us a score to beat. Day 8 tries to beat it honestly.

```text
Day 7: Linear Regression  ->  the score to beat
Day 8: Random Forest      ->  must beat that score on the same split
```

If the Random Forest cannot beat the simple model, the extra complexity is not
worth it. That mindset — measure before you trust — is the heart of ML
engineering.

## Why Linear Regression Was Not Enough

Linear Regression draws straight-line relationships. But car prices are not
straight lines. The price drop from a 2010 to a 2015 car is not the same as the
drop from a 2018 to a 2023 car. Real relationships bend and interact.

A straight-line model cannot capture those bends, so it leaves a lot of the price
pattern unexplained.

## What Is A Random Forest

A Random Forest is a team of **decision trees** that vote.

- A **decision tree** asks yes/no questions ("Is the car newer than 2018?",
  "Is the mileage below 80,000?") and follows the branches down to a price guess.
- One tree alone can overfit and make mistakes.
- A **forest** trains many trees, each on a random slice of the data, and then
  **averages** their guesses.

Averaging many different trees cancels out individual mistakes and produces a
prediction that is both more accurate and more stable. This is called an
**ensemble** model.

## The Golden Rule: A Fair Comparison

To compare two models fairly, everything except the model must stay the same.

- Same feature table.
- Same 80/20 split.
- Same `random_state = 42`.

Both models see the identical training cars and are graded on the identical test
cars. Only then does "Model A beats Model B" actually mean something.

## Inputs And Outputs

Day 8 reads the encoded Day 6 feature table:

```text
data/processed/saudi_used_cars_features.csv
```

It writes two local artifacts:

```text
models/random_forest.joblib     (the trained Random Forest)
models/model_comparison.json    (both models' scores, side by side)
```

Both are git-ignored, like every model artifact in this project.

## Run Day 6 And Day 7 First

If the feature CSV does not exist yet:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.clean
python -m saudi_car_ai.features.build_features
```

## Run The Comparison Script

```powershell
$env:PYTHONPATH = "src"
python -m saudi_car_ai.models.compare
```

The script prints a side-by-side table of MAE, RMSE, and R2 for both models,
names the winner, and shows how much the Random Forest improved on the baseline.

## Example Result

On the real dataset, the Random Forest clearly wins:

```text
Metric              Baseline     Random Forest
MAE (SAR)             26,873            12,203
RMSE (SAR)            43,373            23,104
R2                     0.589             0.883
```

The average error dropped from about 27,000 SAR to about 12,000 SAR, and the
share of price variation explained jumped from 59% to 88%. That is a large, real
improvement — and now we have proof, not just a feeling.

## Day 8 Code

The comparison code lives in:

```text
src/saudi_car_ai/models/compare.py
```

It reuses the Day 7 building blocks (`split_train_test`, `evaluate_model`) so the
comparison is guaranteed to use the same split. Important functions:

- `train_random_forest`: fits the Random Forest on the training set.
- `compare_models`: trains both models on one split and scores them.
- `save_forest_model` / `save_comparison`: write the model and the comparison JSON.

## Mini Lesson: Why RMSE Fell So Much

RMSE punishes big misses harder than MAE. The baseline's RMSE was much larger
than its MAE, which told us it was making some very large errors on expensive
cars. The Random Forest's RMSE is much closer to its MAE, meaning it not only
predicts better on average but also makes fewer catastrophic misses.

## What We Are Not Doing Yet

- We are not tuning the Random Forest's settings yet (number of trees, depth).
- We are not choosing the final "production" model formally — that is a later day.
- We are not building the API around it yet.

Today's win is clear evidence that a better model beats the baseline.

## Done Checklist

- Feature dataset exists from Day 6.
- Random Forest is trained on the same split as the baseline.
- MAE, RMSE, and R2 are compared side by side.
- The better model is identified from the numbers, not from a guess.
- Random Forest artifact and comparison JSON are saved locally.
- Tests pass.
