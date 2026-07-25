from pathlib import Path

import pandas as pd
import pytest

from saudi_car_ai.data.dataset import (
    FEATURE_CANDIDATES,
    TARGET_COLUMN,
    build_dataset_overview,
    create_modeling_frame,
    get_modeling_columns,
    load_dataset,
    validate_required_columns,
)


def make_sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Make": "Toyota",
                "Type": "Camry",
                "Year": 2018,
                "Origin": "Saudi",
                "Color": "Black",
                "Options": "Full",
                "Engine_Size": 2.5,
                "Fuel_Type": "Gas",
                "Gear_Type": "Automatic",
                "Mileage": 120000,
                "Region": "Riyadh",
                "Price": 58000,
                "Negotiable": False,
            },
            {
                "Make": "Nissan",
                "Type": "Patrol",
                "Year": 2016,
                "Origin": "Saudi",
                "Color": "White",
                "Options": "Full",
                "Engine_Size": 4.8,
                "Fuel_Type": "Gas",
                "Gear_Type": "Automatic",
                "Mileage": 5448,
                "Region": "Riyadh",
                "Price": 0,
                "Negotiable": True,
            },
        ]
    )


def test_load_dataset_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "cars.csv"
    make_sample_dataset().to_csv(csv_path, index=False)

    df = load_dataset(csv_path)

    assert len(df) == 2
    assert list(df.columns) == [*FEATURE_CANDIDATES, TARGET_COLUMN, "Negotiable"]


def test_load_dataset_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.csv")


def test_validate_required_columns_rejects_missing_column() -> None:
    df = make_sample_dataset().drop(columns=["Mileage"])

    with pytest.raises(ValueError, match="Mileage"):
        validate_required_columns(df)


def test_build_dataset_overview_identifies_target_features_and_filters() -> None:
    overview = build_dataset_overview(make_sample_dataset())

    assert overview.row_count == 2
    assert overview.column_count == 13
    assert overview.target_column == "Price"
    assert overview.feature_candidates == FEATURE_CANDIDATES
    assert overview.filter_columns == ["Negotiable"]
    assert overview.missing_values["Price"] == 0


def test_get_modeling_columns_returns_features_plus_target() -> None:
    assert get_modeling_columns() == [*FEATURE_CANDIDATES, TARGET_COLUMN]


def test_create_modeling_frame_excludes_filter_columns() -> None:
    modeling_df = create_modeling_frame(make_sample_dataset())

    assert list(modeling_df.columns) == [*FEATURE_CANDIDATES, TARGET_COLUMN]
    assert "Negotiable" not in modeling_df.columns
