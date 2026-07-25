# Day 3 Load The Kaggle Dataset

Day 3 starts the real data workflow. The goal is not deep analysis yet. The goal
is to prove that Python can load the selected CSV, inspect its structure, and
name the first target and feature candidates.

## Goal

By the end of this day, you should be able to:

- Load `data/raw/UsedCarsSA_Clean_EN.csv` with pandas.
- Inspect rows, columns, data types, and missing values.
- Identify `Price` as the target column.
- Identify the first feature candidates.
- Explain why `Negotiable` is a filter column, not a model feature.

## Dataset Path

The project starts with:

```text
data/raw/UsedCarsSA_Clean_EN.csv
```

Raw data files are ignored by Git. This is intentional because datasets can be
large and may have license restrictions.

## Run The Day 3 Dataset Check

From the project root:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.dataset
```

The command prints:

- Row count.
- Column count.
- Target column.
- Feature candidates.
- Filter columns.
- Column data types.
- Missing value counts.

## Target Column

The first prediction target is:

```text
Price
```

This is what the model will try to predict.

## First Feature Candidates

The first feature candidates are:

```text
Make
Type
Year
Origin
Color
Options
Engine_Size
Fuel_Type
Gear_Type
Mileage
Region
```

These are not final yet. Day 4 and Day 5 will help decide what needs cleaning,
removal, encoding, or transformation.

## Filter Column

`Negotiable` is important, but it is not a first model feature.

In this dataset, negotiable listings often have:

```text
Price = 0
Negotiable = True
```

Those cars are not free. The seller simply did not publish a fixed price. For
the first model, we will train only on rows where:

```text
Negotiable = False
Price > 0
```

## Day 3 Code

The dataset utilities live in:

```text
src/saudi_car_ai/data/dataset.py
```

Important functions:

- `load_dataset`: reads the CSV.
- `validate_required_columns`: checks that the expected columns exist.
- `build_dataset_overview`: summarizes the loaded data.
- `create_modeling_frame`: selects feature candidates plus target.

## Run Tests

```powershell
pytest
```

The Day 3 tests use a tiny fake CSV, so they do not require the local Kaggle file
to exist in CI or on another computer.

## Done Checklist

- The CSV loads successfully.
- The dataset has 8,035 rows and 13 columns locally.
- `Price` is documented as the target.
- Feature candidates are documented.
- `Negotiable` is documented as a filtering column.
- `pytest` passes.
