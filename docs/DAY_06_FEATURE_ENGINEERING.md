# Day 6 Feature Engineering

Day 6 is one of the most important machine learning days.

Feature engineering means turning the cleaned dataset into a shape that a model
can learn from. A model does not understand car listings the way a human does.
It sees columns of numbers. Our job is to represent the car in useful numeric
signals without leaking the answer.

## Big Idea

Raw data says:

```text
Make = Toyota
Year = 2020
Mileage = 60000
Region = Riyadh
Price = 70000
```

The model needs something closer to:

```text
Car_Age = 6
Mileage_Per_Year = 8571.43
Log_Mileage = 11.00
Make_Toyota = 1
Region_Riyadh = 1
Price = 70000
```

`Price` stays separate because it is the target. We use the other columns to
predict it.

## Why Feature Engineering Matters

Good features can make a simple model perform surprisingly well. Bad features
can make a strong model fail.

Feature engineering helps with:

- Converting text categories into numbers.
- Making important relationships easier for the model to see.
- Reducing the effect of extreme numeric values.
- Keeping the training table consistent and repeatable.

## Inputs And Outputs

Day 6 reads the cleaned Day 5 dataset:

```text
data/processed/saudi_used_cars_clean.csv
```

It writes the feature table:

```text
data/processed/saudi_used_cars_features.csv
```

Both files are local processed data and are ignored by Git.

## Feature Groups

### 1. Numeric Features

The raw numeric columns are:

```text
Year
Engine_Size
Mileage
```

These are already numbers, but that does not mean they are perfect.

### 2. Derived Numeric Features

We add:

```text
Car_Age
Mileage_Per_Year
Log_Mileage
```

`Car_Age` is easier to reason about than `Year`. A 2020 car in 2026 is 6 years
old.

`Mileage_Per_Year` tells us usage intensity. Two cars can both have 100,000 km,
but the meaning is different if one car is 2 years old and the other is 10 years
old.

`Log_Mileage` compresses large mileage values. This helps because mileage can be
very skewed.

### 3. Categorical Features

The categorical columns are:

```text
Make
Type
Origin
Color
Options
Fuel_Type
Gear_Type
Region
```

Models cannot directly understand text like `Toyota` or `Riyadh`, so we encode
these values.

## One-Hot Encoding

One-hot encoding creates a new numeric column for each category value.

Example:

```text
Make
Toyota
Hyundai
```

Becomes:

```text
Make_Toyota
Make_Hyundai
```

For a Toyota row:

```text
Make_Toyota = 1
Make_Hyundai = 0
```

For a Hyundai row:

```text
Make_Toyota = 0
Make_Hyundai = 1
```

This is beginner-friendly and works well for the first baseline model.

## Important Warning: Do Not Leak The Target

Never use `Price` to create an input feature for the same model.

Bad idea:

```text
Feature = Price / Mileage
```

Why? Because `Price` is the answer. If we put the answer into the inputs, the
model looks smart during training but fails in real life.

For Day 6, `Price` stays only as the target column.

## Run Day 5 First

If the cleaned CSV does not exist yet:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.clean
```

## Run The Feature Engineering Script

```powershell
$env:PYTHONPATH = "src"
python -m saudi_car_ai.features.build_features
```

The script prints:

- Number of input rows.
- Number of output rows.
- How many raw feature columns existed before encoding.
- How many model feature columns exist after encoding.
- Which categorical columns were encoded.
- Which numeric columns were prepared.

## Day 6 Code

The feature engineering code lives in:

```text
src/saudi_car_ai/features/build_features.py
```

Important functions:

- `add_derived_numeric_features`: creates `Car_Age`, `Mileage_Per_Year`, and
  `Log_Mileage`.
- `build_feature_table`: keeps raw categories, numeric features, derived
  features, and `Price`.
- `encode_categorical_features`: applies one-hot encoding.
- `split_features_target`: separates `X` and `y` for modeling.
- `build_encoded_features`: runs the full feature engineering workflow.

## Mini Lesson: X And y

In machine learning, we usually call:

```text
X = input features
y = target
```

For this project:

```text
X = car details
y = Price
```

Day 6 prepares `X` and keeps `y` ready for Day 7.

## What We Are Not Doing Yet

We are not training a model today. That is Day 7.

We are also not choosing the final perfect encoding strategy yet. Later, when we
build an API, we may move from `pandas.get_dummies` to a saved scikit-learn
preprocessing pipeline so new API requests are transformed exactly like training
data.

For now, the goal is understanding.

## Done Checklist

- Cleaned dataset exists from Day 5.
- Derived numeric features are created.
- Categorical columns are one-hot encoded.
- `Price` is kept as the target, not used as an input feature.
- Feature CSV is generated locally.
- Tests pass.
