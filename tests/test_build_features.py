from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from saudi_car_ai.data.dataset import FEATURE_CANDIDATES, TARGET_COLUMN
from saudi_car_ai.features.build_features import (
    CATEGORICAL_COLUMNS,
    CURRENT_YEAR,
    DERIVED_NUMERIC_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    add_derived_numeric_features,
    build_encoded_features,
    build_feature_table,
    encode_categorical_features,
    load_clean_dataset,
    save_feature_dataset,
    split_features_target,
)


def make_clean_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
            },
            {
                "Make": "Hyundai",
                "Type": "Elantra",
                "Year": 2018,
                "Origin": "Saudi",
                "Color": "Black",
                "Options": "Standard",
                "Engine_Size": 1.6,
                "Fuel_Type": "Gas",
                "Gear_Type": "Automatic",
                "Mileage": 120000,
                "Region": "Jeddah",
                "Price": 45000,
            },
        ]
    )


def test_load_clean_dataset_reads_processed_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "clean.csv"
    make_clean_dataset().to_csv(csv_path, index=False)

    loaded_df = load_clean_dataset(csv_path)

    assert len(loaded_df) == 2
    assert list(loaded_df.columns) == [*FEATURE_CANDIDATES, TARGET_COLUMN]


def test_load_clean_dataset_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Clean dataset not found"):
        load_clean_dataset(tmp_path / "missing.csv")


def test_add_derived_numeric_features_creates_modeling_signals_without_mutating_source() -> None:
    df = make_clean_dataset()
    feature_df = add_derived_numeric_features(df, current_year=CURRENT_YEAR)

    assert feature_df.loc[0, "Car_Age"] == 6
    assert feature_df.loc[0, "Mileage_Per_Year"] == pytest.approx(60000 / 7)
    assert feature_df.loc[0, "Log_Mileage"] == pytest.approx(np.log1p(60000))
    assert "Car_Age" not in df.columns


def test_build_feature_table_keeps_raw_categories_and_target() -> None:
    feature_df = build_feature_table(make_clean_dataset())

    assert list(feature_df.columns) == [*CATEGORICAL_COLUMNS, *MODEL_NUMERIC_COLUMNS, TARGET_COLUMN]
    assert all(column in feature_df.columns for column in DERIVED_NUMERIC_COLUMNS)


def test_encode_categorical_features_creates_numeric_model_table() -> None:
    feature_df = build_feature_table(make_clean_dataset())
    encoded_df = encode_categorical_features(feature_df)

    assert TARGET_COLUMN in encoded_df.columns
    assert "Make" not in encoded_df.columns
    assert "Make_Toyota" in encoded_df.columns
    assert "Region_Riyadh" in encoded_df.columns
    assert encoded_df["Make_Toyota"].tolist() == [1, 0]


def test_split_features_target_separates_x_and_y() -> None:
    encoded_df = encode_categorical_features(build_feature_table(make_clean_dataset()))
    x, y = split_features_target(encoded_df)

    assert TARGET_COLUMN not in x.columns
    assert y.tolist() == [70000, 45000]


def test_build_encoded_features_returns_summary() -> None:
    encoded_df, summary = build_encoded_features(make_clean_dataset())

    assert summary.input_rows == 2
    assert summary.output_rows == 2
    assert summary.raw_feature_columns == len(CATEGORICAL_COLUMNS) + len(MODEL_NUMERIC_COLUMNS)
    assert summary.encoded_feature_columns == len(encoded_df.columns) - 1
    assert summary.target_column == TARGET_COLUMN


def test_save_feature_dataset_writes_csv(tmp_path: Path) -> None:
    encoded_df, _summary = build_encoded_features(make_clean_dataset())
    output_path = save_feature_dataset(encoded_df, tmp_path / "features.csv")

    loaded_df = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(loaded_df) == 2
    assert TARGET_COLUMN in loaded_df.columns
