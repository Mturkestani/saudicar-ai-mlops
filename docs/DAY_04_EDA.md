# Day 4 Exploratory Data Analysis

Day 4 is about understanding the dataset before training. EDA means
Exploratory Data Analysis: asking practical questions about the data so the
model does not learn from confusing, broken, or misleading examples.

## Goal

By the end of this day, you should be able to explain:

- How many rows and columns the dataset has.
- Whether the clean CSV has missing values.
- Whether duplicate rows exist.
- Why negotiable listings need special handling.
- How car prices are distributed.
- Which categorical values appear most often.
- Which numeric columns seem related to price.
- Which outliers need investigation before cleaning.

## Run The EDA Script

From the project root:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.eda
```

The script prints the main findings and saves charts to:

```text
docs/assets/eda/
```

## Generated Charts

The EDA script creates:

```text
price_distribution.png
top_makes.png
median_price_by_make.png
median_price_by_year.png
```

These charts are committed because they are small documentation assets, not raw
data files.

## Key Findings

The local clean CSV has:

```text
8035 rows
13 columns
0 missing values
3 duplicate rows
```

The first usable training rows are non-negotiable listings with real prices:

```text
Negotiable = False
Price > 0
```

That leaves:

```text
5508 priced rows
```

## Price Distribution

For non-negotiable listings:

```text
Median price: 58,000 SAR
Mean price:   about 78,336 SAR
Max price:    1,150,000 SAR
```

The mean is higher than the median because expensive cars pull the average up.
This tells us price is skewed. Later, we may try predicting `log(Price)`.

## Mileage Outliers

Mileage has suspicious extreme values:

```text
Median mileage: about 101,960 km
99th percentile: 600,000 km
Max mileage: 20,000,000 km
```

A car with 20 million kilometers is probably a data issue. Day 5 should decide
how to handle extreme mileage values.

## Categorical Features

The most common makes include:

```text
Toyota
Hyundai
Ford
Chevrolet
Nissan
```

Toyota dominates the dataset. This matters because the model may learn common
brands better than rare brands.

The most common regions include:

```text
Riyadh
Dammam
Jeddah
Qassim
Al-Medina
```

Region may affect price because local supply and demand can differ.

## Numeric Relationships

For priced rows, simple correlations show:

```text
Year: positive
Engine_Size: positive
Mileage: negative
```

This matches common sense:

- Newer cars usually cost more.
- Larger engine cars often cost more.
- Higher mileage usually lowers price.

Correlation is not the final answer, but it gives us clues before modeling.

## Cleaning Decisions For Day 5

Day 4 does not clean the data yet. It creates the cleaning checklist:

1. Remove duplicate rows.
2. Remove negotiable listings from training data.
3. Keep only rows where `Price > 0`.
4. Investigate very low prices such as `Price = 1`.
5. Investigate extreme mileage values.
6. Decide how to encode text columns like `Make`, `Type`, `Region`, and
   `Gear_Type`.

## Day 4 Code

The EDA utilities live in:

```text
src/saudi_car_ai/data/eda.py
```

Important functions:

- `get_priced_listings`: removes negotiable and zero-price rows.
- `summarize_categorical_column`: counts common category values.
- `calculate_numeric_correlations`: checks simple relationships with price.
- `build_eda_findings`: creates a structured EDA summary.
- `save_charts`: writes EDA charts.

## Done Checklist

- EDA script runs.
- EDA charts exist.
- Missing values, duplicates, negotiable rows, price skew, and mileage outliers
  are explained.
- Cleaning decisions for Day 5 are written down.
- `pytest` passes.
