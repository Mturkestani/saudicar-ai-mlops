"""Day 9 repeatable training workflow: one command, fixed split, saved outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import RegressorMixin

from saudi_car_ai.features.build_features import FEATURE_DATA_PATH
from saudi_car_ai.models.compare import train_random_forest
from saudi_car_ai.models.train import (
    RANDOM_STATE,
    TEST_SIZE,
    ModelMetrics,
    evaluate_model,
    load_feature_dataset,
    split_train_test,
    train_baseline_model,
)

PRODUCTION_MODEL_PATH = Path("models/production_model.joblib")
TRAINING_RUN_PATH = Path("models/training_run.json")


@dataclass(frozen=True)
class TrainingRun:
    """A reproducible record of one training run and the model it selected."""

    dataset_path: str
    total_rows: int
    train_rows: int
    test_rows: int
    feature_count: int
    test_size: float
    random_state: int
    selected_model: str
    selected_metrics: ModelMetrics
    all_metrics: dict[str, ModelMetrics]
    created_at: str


def run_training(
    encoded_df: pd.DataFrame,
    dataset_path: str = str(FEATURE_DATA_PATH),
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[RegressorMixin, TrainingRun]:
    """Train the candidate models on one fixed split and select the best by MAE."""
    x_train, x_test, y_train, y_test = split_train_test(
        encoded_df,
        test_size=test_size,
        random_state=random_state,
    )

    baseline = train_baseline_model(x_train, y_train)
    forest = train_random_forest(x_train, y_train, random_state=random_state)

    candidates: dict[str, tuple[RegressorMixin, ModelMetrics]] = {
        "baseline": (baseline, evaluate_model(baseline, x_train, x_test, y_test)),
        "random_forest": (forest, evaluate_model(forest, x_train, x_test, y_test)),
    }

    # Lower MAE wins, so the model with the smallest average error is selected.
    selected_model = min(candidates, key=lambda name: candidates[name][1].mae)
    selected_estimator, selected_metrics = candidates[selected_model]

    run = TrainingRun(
        dataset_path=dataset_path,
        total_rows=len(encoded_df),
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_count=x_train.shape[1],
        test_size=test_size,
        random_state=random_state,
        selected_model=selected_model,
        selected_metrics=selected_metrics,
        all_metrics={name: metrics for name, (_estimator, metrics) in candidates.items()},
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    return selected_estimator, run


def save_production_model(
    model: RegressorMixin,
    output_path: Path = PRODUCTION_MODEL_PATH,
) -> Path:
    """Save the selected model as the current production artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_training_run(
    run: TrainingRun,
    output_path: Path = TRAINING_RUN_PATH,
) -> Path:
    """Save the training run manifest as JSON so the run can be reproduced."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
    return output_path


def print_run_summary(run: TrainingRun, model_path: Path, run_path: Path) -> None:
    """Print a beginner-friendly summary of the training run."""
    metrics = run.selected_metrics

    print("Day 9 Training Run Summary")
    print("==========================")
    print(f"Dataset: {run.dataset_path}")
    print(f"Total rows: {run.total_rows}")
    print(f"Train rows: {run.train_rows}")
    print(f"Test rows: {run.test_rows}")
    print(f"Feature columns: {run.feature_count}")
    print(f"Split: test_size={run.test_size}, random_state={run.random_state}")
    print(f"Selected model: {run.selected_model}")
    print(f"  MAE (SAR): {metrics.mae:,.0f}")
    print(f"  RMSE (SAR): {metrics.rmse:,.0f}")
    print(f"  R2: {metrics.r2:.3f}")
    print(f"Created at (UTC): {run.created_at}")
    print(f"Saved production model: {model_path}")
    print(f"Saved training run: {run_path}")


def main() -> None:
    """Run the Day 9 repeatable training workflow."""
    encoded_df = load_feature_dataset()
    model, run = run_training(encoded_df)
    model_path = save_production_model(model)
    run_path = save_training_run(run)
    print_run_summary(run, model_path, run_path)


if __name__ == "__main__":
    main()
