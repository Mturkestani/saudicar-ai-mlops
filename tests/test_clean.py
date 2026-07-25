import pandas as pd

from saudi_car_ai.data.clean import (
    MAX_MILEAGE,
    MIN_PRICE,
    clean_dataset,
    normalize_text_columns,
    save_clean_dataset,
)
from saudi_car_ai.data.dataset import FEATURE_CANDIDATES, TARGET_COLUMN


def make_dirty_dataset() -> pd.DataFrame:
    rows = [
        {
            "Make": " Toyota ",
            "Type": " Camry ",
            "Year": 2020,
            "Origin": " Saudi ",
            "Color": " White ",
            "Options": " Full ",
            "Engine_Size": 2.5,
            "Fuel_Type": " Gas ",
            "Gear_Type": " Automatic ",
            "Mileage": 60000,
            "Region": " Riyadh ",
            "Price": 70000,
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
            "Mileage": 90000,
            "Region": "Jeddah",
            "Price": MIN_PRICE - 1,
            "Negotiable": False,
        },
        {
            "Make": "Ford",
            "Type": "Taurus",
            "Year": 2018,
            "Origin": "Saudi",
            "Color": "Grey",
            "Options": "Standard",
            "Engine_Size": 3.5,
            "Fuel_Type": "Gas",
            "Gear_Type": "Automatic",
            "Mileage": MAX_MILEAGE + 1,
            "Region": "Riyadh",
            "Price": 55000,
            "Negotiable": False,
        },
    ]
    return pd.DataFrame([*rows, rows[0]])


def test_normalize_text_columns_strips_whitespace_without_mutating_source() -> None:
    df = make_dirty_dataset()
    normalized_df = normalize_text_columns(df)

    assert normalized_df.loc[0, "Make"] == "Toyota"
    assert normalized_df.loc[0, "Region"] == "Riyadh"
    assert df.loc[0, "Make"] == " Toyota "


def test_clean_dataset_removes_known_training_data_problems() -> None:
    cleaned_df, summary = clean_dataset(make_dirty_dataset())

    assert len(cleaned_df) == 1
    assert summary.original_rows == 5
    assert summary.duplicate_rows_removed == 1
    assert summary.negotiable_rows_removed == 1
    assert summary.invalid_price_rows_removed == 1
    assert summary.extreme_mileage_rows_removed == 1
    assert summary.final_rows == 1


def test_clean_dataset_outputs_modeling_columns_only() -> None:
    cleaned_df, summary = clean_dataset(make_dirty_dataset())

    expected_columns = [*FEATURE_CANDIDATES, TARGET_COLUMN]
    assert list(cleaned_df.columns) == expected_columns
    assert summary.output_columns == expected_columns
    assert "Negotiable" not in cleaned_df.columns


def test_clean_dataset_keeps_valid_threshold_values() -> None:
    df = make_dirty_dataset().iloc[[0]].copy()
    df.loc[0, "Price"] = MIN_PRICE
    df.loc[0, "Mileage"] = MAX_MILEAGE

    cleaned_df, summary = clean_dataset(df)

    assert len(cleaned_df) == 1
    assert summary.invalid_price_rows_removed == 0
    assert summary.extreme_mileage_rows_removed == 0


def test_save_clean_dataset_writes_csv(tmp_path) -> None:
    cleaned_df, _summary = clean_dataset(make_dirty_dataset())
    output_path = save_clean_dataset(cleaned_df, tmp_path / "clean.csv")

    loaded_df = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(loaded_df) == 1
    assert list(loaded_df.columns) == [*FEATURE_CANDIDATES, TARGET_COLUMN]
