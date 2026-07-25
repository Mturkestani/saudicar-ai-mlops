"""Feature engineering for the SaudiCar AI price model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from saudi_car_ai.data.clean import PROCESSED_DATA_PATH
from saudi_car_ai.data.dataset import TARGET_COLUMN, validate_required_columns

FEATURE_DATA_PATH = Path("data/processed/saudi_used_cars_features.csv")
CURRENT_YEAR = 2026
CATEGORICAL_COLUMNS = [
    "Make",
    "Type",
    "Origin",
    "Color",
    "Options",
    "Fuel_Type",
    "Gear_Type",
    "Region",
]
BASE_NUMERIC_COLUMNS = ["Year", "Engine_Size", "Mileage"]
DERIVED_NUMERIC_COLUMNS = ["Car_Age", "Mileage_Per_Year", "Log_Mileage"]
MODEL_NUMERIC_COLUMNS = [*BASE_NUMERIC_COLUMNS, *DERIVED_NUMERIC_COLUMNS]


@dataclass(frozen=True)
class FeatureSummary:
    """Summary of the engineered feature table."""

    input_rows: int
    output_rows: int
    raw_feature_columns: int
    encoded_feature_columns: int
    categorical_columns: list[str]
    numeric_columns: list[str]
    target_column: str


def load_clean_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned Day 5 dataset."""
    if not path.exists():
        msg = (
            f"Clean dataset not found at {path}. Run "
            "`python -m saudi_car_ai.data.clean` first."
        )
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def add_derived_numeric_features(
    df: pd.DataFrame,
    current_year: int = CURRENT_YEAR,
) -> pd.DataFrame:
    """Add numeric features that make car price patterns easier to learn."""
    validate_required_columns(df, [*CATEGORICAL_COLUMNS, *BASE_NUMERIC_COLUMNS, TARGET_COLUMN])
    feature_df = df.copy()

    feature_df["Car_Age"] = (current_year - feature_df["Year"]).clip(lower=0)
    feature_df["Mileage_Per_Year"] = feature_df["Mileage"] / (feature_df["Car_Age"] + 1)
    feature_df["Log_Mileage"] = np.log1p(feature_df["Mileage"])

    return feature_df


def build_feature_table(
    df: pd.DataFrame,
    current_year: int = CURRENT_YEAR,
) -> pd.DataFrame:
    """Create the unencoded feature table with derived numeric columns."""
    feature_df = add_derived_numeric_features(df, current_year=current_year)
    selected_columns = [*CATEGORICAL_COLUMNS, *MODEL_NUMERIC_COLUMNS, TARGET_COLUMN]

    return feature_df[selected_columns].copy()


def encode_categorical_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns for beginner-friendly model training."""
    validate_required_columns(
        feature_df,
        [*CATEGORICAL_COLUMNS, *MODEL_NUMERIC_COLUMNS, TARGET_COLUMN],
    )

    encoded_df = pd.get_dummies(
        feature_df,
        columns=CATEGORICAL_COLUMNS,
        dtype=int,
    )

    feature_columns = sorted(column for column in encoded_df.columns if column != TARGET_COLUMN)
    return encoded_df[[*feature_columns, TARGET_COLUMN]].copy()


def split_features_target(encoded_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an encoded table into model inputs X and target y."""
    if TARGET_COLUMN not in encoded_df.columns:
        msg = f"Encoded dataset is missing target column: {TARGET_COLUMN}"
        raise ValueError(msg)

    x = encoded_df.drop(columns=[TARGET_COLUMN])
    y = encoded_df[TARGET_COLUMN]
    return x, y


def build_encoded_features(
    df: pd.DataFrame,
    current_year: int = CURRENT_YEAR,
) -> tuple[pd.DataFrame, FeatureSummary]:
    """Build model-ready numeric features from the cleaned dataset."""
    feature_df = build_feature_table(df, current_year=current_year)
    encoded_df = encode_categorical_features(feature_df)
    x, _y = split_features_target(encoded_df)

    summary = FeatureSummary(
        input_rows=len(df),
        output_rows=len(encoded_df),
        raw_feature_columns=len(feature_df.columns) - 1,
        encoded_feature_columns=len(x.columns),
        categorical_columns=CATEGORICAL_COLUMNS.copy(),
        numeric_columns=MODEL_NUMERIC_COLUMNS.copy(),
        target_column=TARGET_COLUMN,
    )

    return encoded_df, summary


def save_feature_dataset(
    encoded_df: pd.DataFrame,
    output_path: Path = FEATURE_DATA_PATH,
) -> Path:
    """Save the encoded feature table as a local processed CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(output_path, index=False)
    return output_path


def print_feature_summary(summary: FeatureSummary, output_path: Path) -> None:
    """Print a beginner-friendly feature engineering summary."""
    print("Day 6 Feature Engineering Summary")
    print("=================================")
    print(f"Input rows: {summary.input_rows}")
    print(f"Output rows: {summary.output_rows}")
    print(f"Raw feature columns before encoding: {summary.raw_feature_columns}")
    print(f"Model feature columns after encoding: {summary.encoded_feature_columns}")
    print(f"Categorical columns encoded: {', '.join(summary.categorical_columns)}")
    print(f"Numeric columns prepared: {', '.join(summary.numeric_columns)}")
    print(f"Target column kept separate: {summary.target_column}")
    print(f"Saved feature CSV: {output_path}")


def main() -> None:
    """Run the Day 6 feature engineering pipeline."""
    clean_df = load_clean_dataset()
    encoded_df, summary = build_encoded_features(clean_df)
    output_path = save_feature_dataset(encoded_df)
    print_feature_summary(summary, output_path)


if __name__ == "__main__":
    main()
