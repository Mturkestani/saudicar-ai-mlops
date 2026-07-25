"""Day 8 model comparison: baseline Linear Regression vs Random Forest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from saudi_car_ai.models.train import (
    RANDOM_STATE,
    TEST_SIZE,
    ModelMetrics,
    evaluate_model,
    load_feature_dataset,
    split_train_test,
    train_baseline_model,
)

FOREST_MODEL_PATH = Path("models/random_forest.joblib")
COMPARISON_PATH = Path("models/model_comparison.json")
N_ESTIMATORS = 200


@dataclass(frozen=True)
class ModelComparison:
    """Side-by-side scores for the baseline and the Random Forest."""

    baseline: ModelMetrics
    random_forest: ModelMetrics
    best_model: str
    mae_improvement: float
    r2_improvement: float


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> RandomForestRegressor:
    """Fit the Day 8 Random Forest on the training set only."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def compare_models(
    encoded_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestRegressor, ModelComparison]:
    """Train both models on the same split and compare their test scores."""
    x_train, x_test, y_train, y_test = split_train_test(
        encoded_df,
        test_size=test_size,
        random_state=random_state,
    )

    baseline = train_baseline_model(x_train, y_train)
    forest = train_random_forest(x_train, y_train, random_state=random_state)

    baseline_metrics = evaluate_model(baseline, x_train, x_test, y_test)
    forest_metrics = evaluate_model(forest, x_train, x_test, y_test)

    # Lower MAE is better, so the smaller error wins.
    best_model = (
        "random_forest" if forest_metrics.mae < baseline_metrics.mae else "baseline"
    )

    comparison = ModelComparison(
        baseline=baseline_metrics,
        random_forest=forest_metrics,
        best_model=best_model,
        mae_improvement=baseline_metrics.mae - forest_metrics.mae,
        r2_improvement=forest_metrics.r2 - baseline_metrics.r2,
    )

    return forest, comparison


def save_forest_model(
    model: RandomForestRegressor,
    output_path: Path = FOREST_MODEL_PATH,
) -> Path:
    """Save the trained Random Forest artifact with joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_comparison(
    comparison: ModelComparison,
    output_path: Path = COMPARISON_PATH,
) -> Path:
    """Save the two-model comparison as JSON for later review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(comparison), indent=2), encoding="utf-8")
    return output_path


def print_comparison_summary(
    comparison: ModelComparison,
    model_path: Path,
    comparison_path: Path,
) -> None:
    """Print a beginner-friendly comparison table."""
    baseline = comparison.baseline
    forest = comparison.random_forest

    print("Day 8 Model Comparison Summary")
    print("==============================")
    print(f"{'Metric':<12}{'Baseline':>16}{'Random Forest':>18}")
    print(f"{'MAE (SAR)':<12}{baseline.mae:>16,.0f}{forest.mae:>18,.0f}")
    print(f"{'RMSE (SAR)':<12}{baseline.rmse:>16,.0f}{forest.rmse:>18,.0f}")
    print(f"{'R2':<12}{baseline.r2:>16.3f}{forest.r2:>18.3f}")
    print()
    print(f"Best model: {comparison.best_model}")
    print(f"MAE improvement (SAR): {comparison.mae_improvement:,.0f}")
    print(f"R2 improvement: {comparison.r2_improvement:+.3f}")
    print(f"Saved Random Forest artifact: {model_path}")
    print(f"Saved comparison: {comparison_path}")


def main() -> None:
    """Run the Day 8 model comparison pipeline."""
    encoded_df = load_feature_dataset()
    forest, comparison = compare_models(encoded_df)
    model_path = save_forest_model(forest)
    comparison_path = save_comparison(comparison)
    print_comparison_summary(comparison, model_path, comparison_path)


if __name__ == "__main__":
    main()
