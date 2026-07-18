"""Exploratory data analysis for the Saudi used car dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.switch_backend("Agg")


RAW_DATA_PATH = Path("data/raw/UsedCarsSA_Clean_EN.csv")
EDA_OUTPUT_DIR = Path("docs/assets/eda")


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the selected clean English Saudi used car dataset."""
    return pd.read_csv(path)


def get_priced_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a real listed price.

    Negotiable listings in this dataset use Price = 0, so they should not be used
    for the first price prediction model.
    """
    return df[(df["Negotiable"] == False) & (df["Price"] > 0)].copy()  # noqa: E712


def print_summary(df: pd.DataFrame, priced_df: pd.DataFrame) -> None:
    """Print the most important dataset health and modeling facts."""
    print("SaudiCar AI EDA Summary")
    print("=" * 24)
    print(f"Full dataset shape: {df.shape}")
    print(f"Priced listings shape: {priced_df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Total missing values: {int(df.isna().sum().sum())}")
    print()

    print("Columns")
    print(df.dtypes.to_string())
    print()

    print("Negotiable counts")
    print(df["Negotiable"].value_counts().to_string())
    print()

    print("Priced listings price summary")
    price_summary = priced_df["Price"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    print(price_summary.to_string())
    print()

    print("Mileage summary")
    print(df["Mileage"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print()

    print("Top makes")
    print(df["Make"].value_counts().head(10).to_string())
    print()

    print("Top regions")
    print(df["Region"].value_counts().head(10).to_string())
    print()

    print("Numeric correlation with price")
    print(
        priced_df[["Price", "Year", "Mileage", "Engine_Size"]]
        .corr(numeric_only=True)["Price"]
        .sort_values(ascending=False)
        .to_string()
    )


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
