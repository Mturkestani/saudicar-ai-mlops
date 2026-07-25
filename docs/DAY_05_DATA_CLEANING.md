# Day 5 Data Cleaning

Day 5 turns the Day 4 EDA decisions into a repeatable cleaning pipeline.

The goal is not to create the perfect final dataset. The goal is to create a
clear first training table that removes the most obvious problems before feature
engineering and baseline modeling.

## Goal

By the end of this day, you should be able to explain:

- Why cleaning must be repeatable in code.
- Which rows are removed before the first model.
- Why negotiable listings should not train the first price model.
- Why very low prices and extreme mileage values are risky.
- Where the processed CSV is saved locally.

## Cleaning Rules

The first cleaning pipeline uses these rules:

1. Validate that required columns exist.
2. Remove duplicate rows.
3. Remove negotiable listings.
4. Keep only rows where `Price >= 5000`.
5. Keep only rows where `Mileage <= 1000000`.
6. Strip extra spaces from text columns.
7. Save only feature candidates plus the target column.

The cleaned dataset does not include `Negotiable`, because after filtering every
remaining row is non-negotiable.

## Why Price Starts At 5000 SAR

Day 4 showed suspicious very low prices, including `Price = 1`.

A first car price model should not learn from listings that are probably
placeholder prices, mistakes, or unusual special cases. We use:

```text
Price >= 5000
```

This threshold can be revisited later, but it gives the baseline model a cleaner
starting point.

## Why Mileage Is Capped At 1,000,000 KM

Day 4 showed extreme mileage values, including a maximum of 20,000,000 km.

For the first model, we keep:

```text
Mileage <= 1000000
```

This removes extreme values that are likely data issues or not useful for a
beginner baseline model.

## Run The Cleaning Script

From the project root:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.clean
```

The script reads:

```text
data/raw/UsedCarsSA_Clean_EN.csv
```

It saves:

```text
data/processed/saudi_used_cars_clean.csv
```

Processed data is ignored by Git, so this file stays local.

## Day 5 Code

The cleaning code lives in:

```text
src/saudi_car_ai/data/clean.py
```

Important functions:

- `clean_dataset`: applies all cleaning rules.
- `normalize_text_columns`: removes extra spaces from text values.
- `save_clean_dataset`: writes the processed CSV.

## Expected Local Result

When run on the local Kaggle CSV, the script prints how many rows were removed by
each cleaning rule and how many rows remain for training.

This cleaned CSV becomes the input for:

- Day 6 feature engineering.
- Day 7 baseline model training.

## Done Checklist

- Cleaning rules are written in code.
- Cleaning rules are documented.
- Processed CSV can be generated locally.
- Tests cover duplicates, negotiable rows, price threshold, mileage threshold,
  text cleanup, and output columns.
- `pytest` passes.
