# Dataset

SaudiCar AI starts with the Kaggle dataset:

[Saudi Arabia Used Cars Dataset](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset)

## Local Files

After downloading and extracting the Kaggle ZIP, the local raw data folder contains:

```text
data/raw/archive.zip
data/raw/UsedCarsSA_Clean_EN.csv
data/raw/UsedCarsSA_Unclean_EN.csv
data/raw/UsedCarsSA_Unclean_Ar.xlsx
```

Raw data files are ignored by Git. They should stay local because datasets can be large and may have license or storage constraints.

## Starting Dataset

We will start with:

```text
data/raw/UsedCarsSA_Clean_EN.csv
```

Initial inspection:

- Rows: 8,035
- Columns: 13
- Target column: `Price`
- Important feature columns: `Make`, `Type`, `Year`, `Engine_Size`, `Fuel_Type`, `Gear_Type`, `Mileage`, `Region`
- No missing values in the clean CSV

## Important Modeling Note

The dataset includes negotiable listings:

- `Negotiable = True`
- `Price = 0`

For the first price prediction model, we should train only on rows where:

```text
Negotiable = False
Price > 0
```

Otherwise the model may learn that some normal cars have a price of zero SAR.

## Later Cleaning Practice

After the first model is working, we can return to:

```text
data/raw/UsedCarsSA_Unclean_EN.csv
```

That file is useful for teaching real data cleaning and preprocessing.
