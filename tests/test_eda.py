import pandas as pd

from saudi_car_ai.data.eda import (
    EXTREME_MILEAGE_THRESHOLD,
    build_eda_findings,
    calculate_numeric_correlations,
    get_priced_listings,
    summarize_categorical_column,
)


def make_eda_sample_dataset() -> pd.DataFrame:
    rows = [
        {
            "Make": "Toyota",
            "Type": "Camry",
            "Year": 2020,
            "Origin": "Saudi",
            "Color": "White",
            "Options": "Full",
            "Engine_Size": 2.5,
            "Fuel_Type": "Gas",
            "Gear_Type": "Automatic",
            "Mileage": 60000,
            "Region": "Riyadh",
            "Price": 70000,
            "Negotiable": False,
        },
        {
            "Make": "Toyota",
            "Type": "Corolla",
            "Year": 2018,
            "Origin": "Saudi",
            "Color": "Silver",
            "Options": "Standard",
            "Engine_Size": 1.6,
            "Fuel_Type": "Gas",
            "Gear_Type": "Automatic",
            "Mileage": 120000,
            "Region": "Riyadh",
            "Price": 45000,
            "Negotiable": False,
        },
        {
            "Make": "Hyundai",
            "Type": "Elantra",
            "Year": 2017,
            "Origin": "Saudi",
            "Color": "Black",
            "Options": "Standard",
            "Engine_Size": 1.6,
            "Fuel_Type": "Gas",
            "Gear_Type": "Automatic",
            "Mileage": EXTREME_MILEAGE_THRESHOLD + 1,
            "Region": "Jeddah",
            "Price": 1,
            "Negotiable": False,
        },
        {
            "Make": "Nissan",
            "Type": "Patrol",
            "Year": 2019,
            "Origin": "Saudi",
            "Color": "White",
            "Options": "Full",
            "Engine_Size": 4.8,
            "Fuel_Type": "Gas",
            "Gear_Type": "Automatic",
            "Mileage": 10000,
            "Region": "Dammam",
            "Price": 0,
            "Negotiable": True,
        },
    ]
    return pd.DataFrame([*rows, rows[1]])


def test_get_priced_listings_removes_negotiable_and_zero_price_rows() -> None:
    priced_df = get_priced_listings(make_eda_sample_dataset())

    assert len(priced_df) == 4
    assert priced_df["Negotiable"].eq(False).all()
    assert priced_df["Price"].gt(0).all()


def test_summarize_categorical_column_returns_top_counts() -> None:
    result = summarize_categorical_column(make_eda_sample_dataset(), "Make", top_n=2)

    assert result == {"Toyota": 3, "Hyundai": 1}


def test_calculate_numeric_correlations_includes_price_drivers() -> None:
    priced_df = get_priced_listings(make_eda_sample_dataset())
    correlations = calculate_numeric_correlations(priced_df)

    assert set(correlations) == {"Price", "Year", "Engine_Size", "Mileage"}
    assert correlations["Price"] == 1.0


def test_build_eda_findings_summarizes_distribution_and_outliers() -> None:
    findings = build_eda_findings(make_eda_sample_dataset())

    assert findings.total_rows == 5
    assert findings.total_columns == 13
    assert findings.duplicate_rows == 1
    assert findings.total_missing_values == 0
    assert findings.priced_rows == 4
    assert findings.negotiable_counts == {"False": 4, "True": 1}
    assert findings.price_median == 45000
    assert findings.low_price_rows == 1
    assert findings.extreme_mileage_rows == 1
    assert findings.top_makes["Toyota"] == 3
    assert findings.top_regions["Riyadh"] == 3
