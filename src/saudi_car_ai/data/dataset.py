"""Dataset loading and first inspection utilities for SaudiCar AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/UsedCarsSA_Clean_EN.csv")
TARGET_COLUMN = "Price"
FEATURE_CANDIDATES = [
    "Make",
    "Type",
    "Year",
    "Origin",
    "Color",
    "Options",
    "Engine_Size",
    "Fuel_Type",
    "Gear_Type",
    "Mileage",
    "Region",
]
FILTER_COLUMNS = ["Negotiable"]
REQUIRED_COLUMNS = [*FEATURE_CANDIDATES, TARGET_COLUMN, *FILTER_COLUMNS]


@dataclass(frozen=True)
class DatasetOverview:
    """Small summary of the raw dataset before detailed EDA."""

    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    target_column: str
    feature_candidates: list[str]
    filter_columns: list[str]


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the selected Kaggle CSV into a pandas DataFrame."""
    if not path.exists():
        msg = (
            f"Dataset not found at {path}. Download the Kaggle dataset and place "
            "UsedCarsSA_Clean_EN.csv in data/raw/."
        )
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """Make sure the CSV has the columns expected by the project."""
    expected_columns = required_columns or REQUIRED_COLUMNS
    missing_columns = [column for column in expected_columns if column not in df.columns]

    if missing_columns:
        msg = f"Dataset is missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)


def build_dataset_overview(df: pd.DataFrame) -> DatasetOverview:
    """Build a beginner-friendly overview from a loaded dataset."""
    validate_required_columns(df)

    return DatasetOverview(
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        dtypes={column: str(dtype) for column, dtype in df.dtypes.items()},
        missing_values={column: int(count) for column, count in df.isna().sum().items()},
        target_column=TARGET_COLUMN,
        feature_candidates=FEATURE_CANDIDATES.copy(),
        filter_columns=FILTER_COLUMNS.copy(),
    )


def get_modeling_columns() -> list[str]:
    """Return the first modeling column set: features plus target."""
    return [*FEATURE_CANDIDATES, TARGET_COLUMN]


def create_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the first feature candidates and target from the raw dataset."""
    validate_required_columns(df)
    return df[get_modeling_columns()].copy()


def print_overview(overview: DatasetOverview) -> None:
    """Print the Day 3 dataset overview."""
    print("Day 3 Dataset Overview")
    print("======================")
    print(f"Rows: {overview.row_count}")
    print(f"Columns: {overview.column_count}")
    print(f"Target column: {overview.target_column}")
    print(f"Feature candidates: {', '.join(overview.feature_candidates)}")
    print(f"Filter columns: {', '.join(overview.filter_columns)}")
    print()

    print("Column types")
    for column, dtype in overview.dtypes.items():
        print(f"- {column}: {dtype}")
    print()

    print("Missing values")
    for column, count in overview.missing_values.items():
        print(f"- {column}: {count}")


def main() -> None:
    """Run the Day 3 dataset loading practice."""
    df = load_dataset()
    overview = build_dataset_overview(df)
    print_overview(overview)


if __name__ == "__main__":
    main()
