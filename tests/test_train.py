import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from saudi_car_ai.features.build_features import build_encoded_features
from saudi_car_ai.models.train import (
    ModelMetrics,
    evaluate_model,
    load_feature_dataset,
    save_metrics,
    save_model,
    split_train_test,
    train_and_evaluate,
    train_baseline_model,
)


def make_feature_dataset(rows: int = 40) -> pd.DataFrame:
    makes = ["Toyota", "Hyundai", "Ford", "Nissan"]
    regions = ["Riyadh", "Jeddah", "Dammam"]
    records = []
    for index in range(rows):
        year = 2010 + (index % 12)
        mileage = 20000 + index * 3000
        records.append(
            {
                "Make": makes[index % len(makes)],
                "Type": "Sedan",
                "Year": year,
                "Origin": "Saudi",
                "Color": "White",
                "Options": "Full",
                "Engine_Size": 1.6 + (index % 4) * 0.3,
                "Fuel_Type": "Gas",
                "Gear_Type": "Automatic",
                "Mileage": mileage,
                "Region": regions[index % len(regions)],
                # A learnable price so Linear Regression is not degenerate.
                "Price": 90000 + year * 100 - mileage // 10,
            }
        )
    return pd.DataFrame(records)


def make_encoded_dataset(rows: int = 40) -> pd.DataFrame:
    encoded_df, _summary = build_encoded_features(make_feature_dataset(rows))
    return encoded_df


def test_load_feature_dataset_reads_processed_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    make_encoded_dataset().to_csv(csv_path, index=False)

    loaded_df = load_feature_dataset(csv_path)

    assert len(loaded_df) == 40
    assert "Price" in loaded_df.columns


def test_load_feature_dataset_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Feature dataset not found"):
        load_feature_dataset(tmp_path / "missing.csv")


def test_split_train_test_is_reproducible_and_excludes_target() -> None:
    encoded_df = make_encoded_dataset()

    x_train_a, x_test_a, y_train_a, _ = split_train_test(encoded_df, random_state=42)
    x_train_b, _, _, _ = split_train_test(encoded_df, random_state=42)

    assert len(x_train_a) + len(x_test_a) == len(encoded_df)
    assert "Price" not in x_train_a.columns
    assert len(y_train_a) == len(x_train_a)
    assert x_train_a.index.tolist() == x_train_b.index.tolist()


def test_train_baseline_model_returns_fitted_linear_regression() -> None:
    x_train, _, y_train, _ = split_train_test(make_encoded_dataset())

    model = train_baseline_model(x_train, y_train)

    assert isinstance(model, LinearRegression)
    assert model.coef_.shape[0] == x_train.shape[1]


def test_evaluate_model_reports_metric_shape() -> None:
    x_train, x_test, y_train, y_test = split_train_test(make_encoded_dataset())
    model = train_baseline_model(x_train, y_train)

    metrics = evaluate_model(model, x_train, x_test, y_test)

    assert isinstance(metrics, ModelMetrics)
    assert metrics.train_rows == len(x_train)
    assert metrics.test_rows == len(x_test)
    assert metrics.feature_count == x_train.shape[1]
    assert metrics.mae >= 0
    assert metrics.rmse >= metrics.mae


def test_train_and_evaluate_learns_a_reasonable_signal() -> None:
    encoded_df = make_encoded_dataset()

    _model, metrics = train_and_evaluate(encoded_df)

    # The synthetic price is a linear function of the inputs, so a linear
    # baseline should explain most of the variance.
    assert metrics.r2 > 0.5


def test_save_model_and_metrics_write_files(tmp_path: Path) -> None:
    _model, metrics = train_and_evaluate(make_encoded_dataset())

    model_path = save_model(_model, tmp_path / "model.joblib")
    metrics_path = save_metrics(metrics, tmp_path / "metrics.json")

    assert model_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["feature_count"] == metrics.feature_count
    assert saved["mae"] == pytest.approx(metrics.mae)
