# Exploratory Data Analysis Summary

EDA stands for **Exploratory Data Analysis**.

In Arabic: **تحليل استكشافي للبيانات**.

The goal is to understand the dataset before training a model. We do EDA so we do not blindly feed bad or confusing data into machine learning.

## Dataset Used

```text
data/raw/UsedCarsSA_Clean_EN.csv
```

This file comes from the Kaggle Saudi Arabia used cars dataset.

## What We Checked

### 1. Dataset Shape

The dataset contains:

```text
8035 rows
13 columns
```

Rows are car listings. Columns are car attributes like make, model, year, mileage, region, and price.

### 2. Column Types

Important columns:

| Column | Meaning | Type |
| --- | --- | --- |
| `Make` | Car brand, such as Toyota or Hyundai | Text |
| `Type` | Car model/type, such as Camry or Patrol | Text |
| `Year` | Manufacturing year | Number |
| `Engine_Size` | Engine size | Number |
| `Fuel_Type` | Gas, diesel, or hybrid | Text |
| `Gear_Type` | Automatic or manual | Text |
| `Mileage` | Kilometers driven | Number |
| `Region` | Saudi city/region | Text |
| `Price` | Listed price in SAR | Number |
| `Negotiable` | Whether the seller did not provide a fixed price | True/False |

### 3. Missing Values

The clean CSV has:

```text
0 missing values
```

This is good, but it does not mean the data is perfect. A value can exist and still be wrong or misleading.

### 4. Duplicate Rows

The dataset has:

```text
3 duplicate rows
```

This is small, but we should remove duplicates during cleaning.

### 5. Negotiable Listings

`Negotiable` means **قابل للتفاوض**.

In this dataset, many negotiable listings have:

```text
Price = 0
Negotiable = True
```

Counts:

```text
Negotiable = False: 5509 rows
Negotiable = True: 2526 rows
```

This is critical. A price prediction model should not learn from `Price = 0` rows because those cars are not actually free. The seller simply did not publish a fixed price.

For the first model, we will train only on:

```text
Negotiable = False
Price > 0
```

That leaves:

```text
5508 usable priced rows
```

### 6. Price Distribution

For non-negotiable listings:

```text
Median price: 58,000 SAR
Mean price:   78,336 SAR
Max price:    1,150,000 SAR
```

The mean is higher than the median because expensive luxury cars pull the average upward.

This means price is skewed. In modeling, we may later try predicting `log(Price)` instead of raw `Price`.

### 7. Mileage Distribution

Mileage has extreme values:

```text
Median mileage: 101,960 km
99th percentile: 600,000 km
Max mileage: 20,000,000 km
```

The max value is suspicious. A car with 20 million km is probably a data issue.

For the first cleaning step, we should investigate and possibly remove extreme mileage outliers.

### 8. Most Common Makes

Top makes by listing count:

```text
Toyota       2038
Hyundai       941
Ford          763
Chevrolet     644
Nissan        548
GMC           400
Kia           357
Lexus         343
Mercedes      328
Mazda         184
```

Toyota dominates the dataset. This matters because the model may perform better for common brands than rare brands.

### 9. Most Common Regions

Top regions:

```text
Riyadh       3237
Dammam       1370
Jeddah       1054
Qassim        309
Al-Medina     297
```

Riyadh has the most listings. Region may influence price because market demand differs by city.

### 10. Numeric Correlations With Price

For priced rows:

```text
Year           0.308
Engine_Size    0.304
Mileage       -0.119
```

Interpretation:

- Newer cars usually cost more.
- Larger engine cars often cost more.
- Higher mileage usually lowers price.
- Mileage correlation is weaker than expected, likely because brand, year, and car type also matter a lot.

## Charts

The EDA script saves charts here:

```text
docs/assets/eda/
```

Generated charts:

```text
price_distribution.png
top_makes.png
median_price_by_make.png
median_price_by_year.png
```

## Critical Decisions Before Modeling

Before training the first model, we should:

1. Remove duplicate rows.
2. Remove negotiable listings from the training data.
3. Keep only rows where `Price > 0`.
4. Investigate very low prices, such as `Price = 1`.
5. Investigate extreme mileage values.
6. Decide how to encode categorical columns like `Make`, `Type`, `Region`, and `Gear_Type`.
7. Use `Price` as the target variable.

## How To Run The EDA Script

From the project root:

```powershell
cd C:\Users\mohammed\Desktop\MLOPs
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m saudi_car_ai.data.eda
```

The `PYTHONPATH` line tells Python where our project source code lives.

Later, we will make this cleaner by installing the project package properly.
