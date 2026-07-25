"""Visualize the SaudiCar AI models: the baseline line and the Day 8 comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from saudi_car_ai.models.compare import train_random_forest
from saudi_car_ai.models.train import (
    RANDOM_STATE,
    TEST_SIZE,
    load_feature_dataset,
    split_train_test,
    train_baseline_model,
)

PLOT_OUTPUT_DIR = Path("docs/assets/models")
BASELINE_COLOR = "#d98a2b"
FOREST_COLOR = "#2f9268"
IDEAL_COLOR = "#c0392b"

plt.switch_backend("Agg")


@dataclass(frozen=True)
class PredictionBundle:
    """Held-out actual prices and each model's predictions for the same test cars."""

    y_test: pd.Series
    baseline_pred: np.ndarray
    forest_pred: np.ndarray


def build_model_predictions(
    encoded_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PredictionBundle:
    """Train both models on one split and return their test-set predictions."""
    x_train, x_test, y_train, y_test = split_train_test(
        encoded_df,
        test_size=test_size,
        random_state=random_state,
    )
    baseline = train_baseline_model(x_train, y_train)
    forest = train_random_forest(x_train, y_train, random_state=random_state)

    return PredictionBundle(
        y_test=y_test,
        baseline_pred=baseline.predict(x_test),
        forest_pred=forest.predict(x_test),
    )


def save_regression_line(encoded_df: pd.DataFrame, output_dir: Path) -> Path:
    """Teaching chart: what a straight-line baseline looks like on one feature."""
    ages = encoded_df[["Car_Age"]]
    prices = encoded_df["Price"]

    one_feature_model = LinearRegression().fit(ages, prices)
    age_grid = pd.DataFrame(
        {"Car_Age": np.linspace(ages["Car_Age"].min(), ages["Car_Age"].max(), 100)}
    )
    line = one_feature_model.predict(age_grid)

    plt.figure(figsize=(10, 6))
    plt.scatter(ages["Car_Age"], prices, s=10, alpha=0.15, color=BASELINE_COLOR, label="Cars")
    plt.plot(age_grid, line, color=IDEAL_COLOR, linewidth=2.5, label="Linear Regression line")
    plt.title("The Baseline Idea: One Straight Line Through the Cars")
    plt.xlabel("Car Age (years)")
    plt.ylabel("Price (SAR)")
    plt.ylim(0, prices.quantile(0.99))
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "regression_line_car_age.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_predicted_vs_actual(bundle: PredictionBundle, output_dir: Path) -> Path:
    """Side-by-side: how close each model's predictions land to the true price."""
    upper = float(bundle.y_test.quantile(0.99))
    limits = [0, upper]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    panels = (
        (axes[0], "Baseline: Linear Regression", bundle.baseline_pred, BASELINE_COLOR),
        (axes[1], "Day 8: Random Forest", bundle.forest_pred, FOREST_COLOR),
    )

    for axis, title, predictions, color in panels:
        axis.scatter(bundle.y_test, predictions, s=12, alpha=0.25, color=color)
        axis.plot(
            limits,
            limits,
            color=IDEAL_COLOR,
            linestyle="--",
            linewidth=2,
            label="Perfect prediction",
        )
        axis.set_title(title)
        axis.set_xlabel("Actual Price (SAR)")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.legend(loc="upper left")

    axes[0].set_ylabel("Predicted Price (SAR)")
    fig.suptitle("Predicted vs Actual Price (closer to the dashed line is better)")
    fig.tight_layout()

    output_path = output_dir / "predicted_vs_actual.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_residual_distribution(bundle: PredictionBundle, output_dir: Path) -> Path:
    """Compare how tightly each model's errors cluster around zero."""
    baseline_errors = bundle.y_test.to_numpy() - bundle.baseline_pred
    forest_errors = bundle.y_test.to_numpy() - bundle.forest_pred
    span = float(np.quantile(np.abs(baseline_errors), 0.99))
    bins = np.linspace(-span, span, 60)

    plt.figure(figsize=(10, 6))
    plt.hist(baseline_errors, bins=bins, alpha=0.55, color=BASELINE_COLOR, label="Baseline")
    plt.hist(forest_errors, bins=bins, alpha=0.55, color=FOREST_COLOR, label="Random Forest")
    plt.axvline(0, color=IDEAL_COLOR, linestyle="--", linewidth=2, label="Zero error")
    plt.title("Prediction Errors: Random Forest Clusters Tighter Around Zero")
    plt.xlabel("Error = Actual - Predicted (SAR)")
    plt.ylabel("Number of cars")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "residual_distribution.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_metrics_comparison(bundle: PredictionBundle, output_dir: Path) -> Path:
    """Bar charts comparing MAE, RMSE, and R2 for both models."""
    models = ["Baseline", "Random Forest"]
    colors = [BASELINE_COLOR, FOREST_COLOR]
    mae = [
        mean_absolute_error(bundle.y_test, bundle.baseline_pred),
        mean_absolute_error(bundle.y_test, bundle.forest_pred),
    ]
    rmse = [
        root_mean_squared_error(bundle.y_test, bundle.baseline_pred),
        root_mean_squared_error(bundle.y_test, bundle.forest_pred),
    ]
    r2 = [
        r2_score(bundle.y_test, bundle.baseline_pred),
        r2_score(bundle.y_test, bundle.forest_pred),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    positions = np.arange(len(models))
    width = 0.35
    axes[0].bar(positions - width / 2, mae, width, label="MAE", color=colors, alpha=0.9)
    axes[0].bar(positions + width / 2, rmse, width, label="RMSE", color=colors, alpha=0.5)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Error in SAR (lower is better)")
    axes[0].set_title("Error: MAE (solid) and RMSE (faded)")
    for index, (mae_value, rmse_value) in enumerate(zip(mae, rmse, strict=True)):
        axes[0].text(index - width / 2, mae_value, f"{mae_value:,.0f}", ha="center", va="bottom")
        axes[0].text(index + width / 2, rmse_value, f"{rmse_value:,.0f}", ha="center", va="bottom")

    axes[1].bar(positions, r2, color=colors)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("R2 (higher is better)")
    axes[1].set_title("R2: Share of Price Variation Explained")
    for index, r2_value in enumerate(r2):
        axes[1].text(index, r2_value, f"{r2_value:.3f}", ha="center", va="bottom")

    fig.suptitle("Baseline vs Random Forest: Scoreboard")
    fig.tight_layout()

    output_path = output_dir / "metrics_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_all_charts(
    encoded_df: pd.DataFrame,
    output_dir: Path = PLOT_OUTPUT_DIR,
) -> list[Path]:
    """Generate every model chart and return the saved file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_model_predictions(encoded_df)

    return [
        save_regression_line(encoded_df, output_dir),
        save_predicted_vs_actual(bundle, output_dir),
        save_residual_distribution(bundle, output_dir),
        save_metrics_comparison(bundle, output_dir),
    ]


def main() -> None:
    """Generate all model charts from the Day 6 feature table."""
    encoded_df = load_feature_dataset()
    saved_paths = save_all_charts(encoded_df)

    print("Model charts saved:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
