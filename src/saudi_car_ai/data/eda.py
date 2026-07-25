"""Exploratory data analysis for the Saudi used car dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from saudi_car_ai.data.dataset import load_dataset

plt.switch_backend("Agg")


EDA_OUTPUT_DIR = Path("docs/assets/eda")
LOW_PRICE_THRESHOLD = 5_000
EXTREME_MILEAGE_THRESHOLD = 1_000_000


@dataclass(frozen=True)
class EdaFindings:
    """Key Day 4 EDA findings for the raw dataset."""

    total_rows: int
    total_columns: int
    duplicate_rows: int
    total_missing_values: int
    priced_rows: int
    negotiable_counts: dict[str, int]
    price_median: float
    price_mean: float
    price_max: float
    mileage_median: float
    mileage_p99: float
    mileage_max: float
    low_price_rows: int
    extreme_mileage_rows: int
    top_makes: dict[str, int]
    top_regions: dict[str, int]
    numeric_correlations: dict[str, float]


def get_priced_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a real listed price.

    Negotiable listings in this dataset use Price = 0, so they should not be used
    for the first price prediction model.
    """
    return df[df["Negotiable"].eq(False) & df["Price"].gt(0)].copy()


def summarize_categorical_column(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
) -> dict[str, int]:
    """Return the most common values for a categorical column."""
    return {
        str(value): int(count) for value, count in df[column].value_counts().head(top_n).items()
    }


def calculate_numeric_correlations(priced_df: pd.DataFrame) -> dict[str, float]:
    """Calculate simple numeric correlations with the target price."""
    correlations = (
        priced_df[["Price", "Year", "Mileage", "Engine_Size"]]
        .corr(numeric_only=True)["Price"]
        .sort_values(ascending=False)
    )

    return {str(column): float(value) for column, value in correlations.items()}


def build_eda_findings(df: pd.DataFrame) -> EdaFindings:
    """Create the structured Day 4 EDA findings."""
    priced_df = get_priced_listings(df)
    price_summary = priced_df["Price"].describe()
    mileage_summary = df["Mileage"].describe(percentiles=[0.99])

    return EdaFindings(
        total_rows=len(df),
        total_columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        total_missing_values=int(df.isna().sum().sum()),
        priced_rows=len(priced_df),
        negotiable_counts={
            str(value): int(count) for value, count in df["Negotiable"].value_counts().items()
        },
        price_median=float(price_summary["50%"]),
        price_mean=float(price_summary["mean"]),
        price_max=float(price_summary["max"]),
        mileage_median=float(mileage_summary["50%"]),
        mileage_p99=float(mileage_summary["99%"]),
        mileage_max=float(mileage_summary["max"]),
        low_price_rows=int(priced_df["Price"].lt(LOW_PRICE_THRESHOLD).sum()),
        extreme_mileage_rows=int(df["Mileage"].gt(EXTREME_MILEAGE_THRESHOLD).sum()),
        top_makes=summarize_categorical_column(df, "Make"),
        top_regions=summarize_categorical_column(df, "Region"),
        numeric_correlations=calculate_numeric_correlations(priced_df),
    )


def print_summary(df: pd.DataFrame, priced_df: pd.DataFrame) -> None:
    """Print the most important dataset health and modeling facts."""
    findings = build_eda_findings(df)

    print("SaudiCar AI EDA Summary")
    print("=" * 24)
    print(f"Full dataset shape: ({findings.total_rows}, {findings.total_columns})")
    print(f"Priced listings shape: {priced_df.shape}")
    print(f"Duplicate rows: {findings.duplicate_rows}")
    print(f"Total missing values: {findings.total_missing_values}")
    print()

    print("Columns")
    print(df.dtypes.to_string())
    print()

    print("Negotiable counts")
    for value, count in findings.negotiable_counts.items():
        print(f"{value}: {count}")
    print()

    print("Priced listings price summary")
    price_summary = priced_df["Price"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    print(price_summary.to_string())
    print()

    print("Mileage summary")
    print(df["Mileage"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(
        f"Extreme mileage rows above {EXTREME_MILEAGE_THRESHOLD:,} km: "
        f"{findings.extreme_mileage_rows}"
    )
    print()

    print("Top makes")
    for value, count in findings.top_makes.items():
        print(f"{value}: {count}")
    print()

    print("Top regions")
    for value, count in findings.top_regions.items():
        print(f"{value}: {count}")
    print()

    print("Numeric correlation with price")
    for value, correlation in findings.numeric_correlations.items():
        print(f"{value}: {correlation:.3f}")


def save_price_distribution(priced_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(priced_df["Price"], bins=50)
    plt.title("Price Distribution for Non-Negotiable Listings")
    plt.xlabel("Price (SAR)")
    plt.ylabel("Listings")
    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution.png", dpi=160)
    plt.close()


def save_top_makes(df: pd.DataFrame, output_dir: Path) -> None:
    top_makes = df["Make"].value_counts().head(12)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_makes.values, y=top_makes.index)
    plt.title("Top Car Makes by Listing Count")
    plt.xlabel("Listings")
    plt.ylabel("Make")
    plt.tight_layout()
    plt.savefig(output_dir / "top_makes.png", dpi=160)
    plt.close()


def save_median_price_by_make(priced_df: pd.DataFrame, output_dir: Path) -> None:
    make_summary = (
        priced_df.groupby("Make")
        .agg(count=("Price", "size"), median_price=("Price", "median"))
        .query("count >= 50")
        .sort_values("median_price", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(x=make_summary["median_price"], y=make_summary.index)
    plt.title("Median Price by Make, Minimum 50 Priced Listings")
    plt.xlabel("Median Price (SAR)")
    plt.ylabel("Make")
    plt.tight_layout()
    plt.savefig(output_dir / "median_price_by_make.png", dpi=160)
    plt.close()


def save_year_price_trend(priced_df: pd.DataFrame, output_dir: Path) -> None:
    year_summary = priced_df.groupby("Year")["Price"].median().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=year_summary, x="Year", y="Price", marker="o")
    plt.title("Median Price by Car Year")
    plt.xlabel("Year")
    plt.ylabel("Median Price (SAR)")
    plt.tight_layout()
    plt.savefig(output_dir / "median_price_by_year.png", dpi=160)
    plt.close()


def save_charts(
    df: pd.DataFrame,
    priced_df: pd.DataFrame,
    output_dir: Path = EDA_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_price_distribution(priced_df, output_dir)
    save_top_makes(df, output_dir)
    save_median_price_by_make(priced_df, output_dir)
    save_year_price_trend(priced_df, output_dir)


def main() -> None:
    df = load_dataset()
    priced_df = get_priced_listings(df)
    print_summary(df, priced_df)
    save_charts(df, priced_df)
    print()
    print(f"Charts saved to: {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
