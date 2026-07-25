"""Data cleaning pipeline for the Saudi used car dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from saudi_car_ai.data.dataset import (
    TARGET_COLUMN,
    get_modeling_columns,
    load_dataset,
    validate_required_columns,
)

PROCESSED_DATA_PATH = Path("data/processed/saudi_used_cars_clean.csv")
MIN_PRICE = 5_000
MAX_MILEAGE = 1_000_000
TEXT_COLUMNS = [
    "Make",
    "Type",
    "Origin",
    "Color",
    "Options",
    "Fuel_Type",
    "Gear_Type",
    "Region",
]


@dataclass(frozen=True)
class CleaningSummary:
    """Row counts removed by each cleaning step."""

    original_rows: int
    duplicate_rows_removed: int
    negotiable_rows_removed: int
    invalid_price_rows_removed: int
    extreme_mileage_rows_removed: int
    final_rows: int
    output_columns: list[str]


def normalize_text_columns(
    df: pd.DataFrame,
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Strip surrounding spaces from text columns without changing the source DataFrame."""
    columns_to_clean = text_columns or TEXT_COLUMNS
    cleaned_df = df.copy()

    for column in columns_to_clean:
        cleaned_df[column] = cleaned_df[column].astype("string").str.strip()

    return cleaned_df


def clean_dataset(
    df: pd.DataFrame,
    min_price: int = MIN_PRICE,
    max_mileage: int = MAX_MILEAGE,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Clean the raw dataset into a first model-ready training table."""
    validate_required_columns(df)

    original_rows = len(df)
    cleaned_df = df.copy()

    before_step = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicate_rows_removed = before_step - len(cleaned_df)

    before_step = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df["Negotiable"].eq(False)]
    negotiable_rows_removed = before_step - len(cleaned_df)

    before_step = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df[TARGET_COLUMN].ge(min_price)]
    invalid_price_rows_removed = before_step - len(cleaned_df)

    before_step = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df["Mileage"].le(max_mileage)]
    extreme_mileage_rows_removed = before_step - len(cleaned_df)

    cleaned_df = normalize_text_columns(cleaned_df)
    cleaned_df = cleaned_df[get_modeling_columns()].reset_index(drop=True)

    summary = CleaningSummary(
        original_rows=original_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        negotiable_rows_removed=negotiable_rows_removed,
        invalid_price_rows_removed=invalid_price_rows_removed,
        extreme_mileage_rows_removed=extreme_mileage_rows_removed,
        final_rows=len(cleaned_df),
        output_columns=list(cleaned_df.columns),
    )

    return cleaned_df, summary


def save_clean_dataset(
    cleaned_df: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    """Save the cleaned dataset as a local processed CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return output_path


def print_cleaning_summary(summary: CleaningSummary, output_path: Path) -> None:
    """Print a beginner-friendly cleaning summary."""
    print("Day 5 Data Cleaning Summary")
    print("===========================")
    print(f"Original rows: {summary.original_rows}")
    print(f"Duplicate rows removed: {summary.duplicate_rows_removed}")
    print(f"Negotiable rows removed: {summary.negotiable_rows_removed}")
    print(f"Invalid or very low price rows removed: {summary.invalid_price_rows_removed}")
    print(f"Extreme mileage rows removed: {summary.extreme_mileage_rows_removed}")
    print(f"Final rows: {summary.final_rows}")
    print(f"Output columns: {', '.join(summary.output_columns)}")
    print(f"Saved cleaned CSV: {output_path}")


def main() -> None:
    """Run the Day 5 cleaning pipeline."""
    raw_df = load_dataset()
    cleaned_df, summary = clean_dataset(raw_df)
    output_path = save_clean_dataset(cleaned_df)
    print_cleaning_summary(summary, output_path)


if __name__ == "__main__":
    main()
