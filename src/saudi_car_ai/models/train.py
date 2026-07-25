"""Day 7 baseline model training for the SaudiCar AI price model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from saudi_car_ai.features.build_features import FEATURE_DATA_PATH, split_features_target

MODEL_PATH = Path("models/baseline_linear_regression.joblib")
METRICS_PATH = Path("models/baseline_metrics.json")
TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelMetrics:
    """Baseline evaluation scores on the held-out test set."""

    train_rows: int
    test_rows: int
    feature_count: int
    mae: float
    rmse: float
    r2: float


def load_feature_dataset(path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    """Load the encoded Day 6 feature table."""
    if not path.exists():
        msg = (
            f"Feature dataset not found at {path}. Run "
            "`python -m saudi_car_ai.features.build_features` first."
        )
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def split_train_test(
    encoded_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the encoded table into reproducible train and test sets."""
    x, y = split_features_target(encoded_df)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_baseline_model(x_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit the Day 7 Linear Regression baseline."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelMetrics:
    """Score the fitted model on the untouched test set."""
    predictions = model.predict(x_test)

    return ModelMetrics(
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_count=x_train.shape[1],
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(root_mean_squared_error(y_test, predictions)),
        r2=float(r2_score(y_test, predictions)),
    )


def train_and_evaluate(
    encoded_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[LinearRegression, ModelMetrics]:
    """Run the full Day 7 baseline training and evaluation workflow."""
    x_train, x_test, y_train, y_test = split_train_test(
        encoded_df,
        test_size=test_size,
        random_state=random_state,
    )
    model = train_baseline_model(x_train, y_train)
    metrics = evaluate_model(model, x_train, x_test, y_test)
    return model, metrics


def save_model(model: LinearRegression, output_path: Path = MODEL_PATH) -> Path:
    """Save the trained model artifact with joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_metrics(metrics: ModelMetrics, output_path: Path = METRICS_PATH) -> Path:
    """Save the evaluation metrics as JSON so runs can be compared later."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return output_path


def print_training_summary(
    metrics: ModelMetrics,
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Print a beginner-friendly training summary."""
    print("Day 7 Baseline Model Summary")
    print("============================")
    print(f"Training rows: {metrics.train_rows}")
    print(f"Test rows: {metrics.test_rows}")
    print(f"Feature columns: {metrics.feature_count}")
    print(f"MAE (average error in SAR): {metrics.mae:,.0f}")
    print(f"RMSE (penalizes big misses): {metrics.rmse:,.0f}")
    print(f"R2 (share of price variance explained): {metrics.r2:.3f}")
    print(f"Saved model artifact: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    """Run the Day 7 baseline training pipeline."""
    encoded_df = load_feature_dataset()
    model, metrics = train_and_evaluate(encoded_df)
    model_path = save_model(model)
    metrics_path = save_metrics(metrics)
    print_training_summary(metrics, model_path, metrics_path)


if __name__ == "__main__":
    main()
