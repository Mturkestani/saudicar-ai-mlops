import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from saudi_car_ai.features.build_features import build_encoded_features
from saudi_car_ai.models.compare import (
    ModelComparison,
    compare_models,
    save_comparison,
    save_forest_model,
    train_random_forest,
)
from saudi_car_ai.models.train import split_train_test


def make_feature_dataset(rows: int = 60) -> pd.DataFrame:
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
                "Price": 90000 + year * 100 - mileage // 10,
            }
        )
    return pd.DataFrame(records)


def make_encoded_dataset(rows: int = 60) -> pd.DataFrame:
    encoded_df, _summary = build_encoded_features(make_feature_dataset(rows))
    return encoded_df


def test_train_random_forest_returns_fitted_forest() -> None:
    x_train, _, y_train, _ = split_train_test(make_encoded_dataset())

    model = train_random_forest(x_train, y_train, n_estimators=20)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_features_in_ == x_train.shape[1]


def test_compare_models_returns_both_scores() -> None:
    _forest, comparison = compare_models(make_encoded_dataset())

    assert isinstance(comparison, ModelComparison)
    assert comparison.best_model in {"baseline", "random_forest"}
    assert comparison.baseline.mae >= 0
    assert comparison.random_forest.mae >= 0


def test_comparison_math_is_consistent() -> None:
    _forest, comparison = compare_models(make_encoded_dataset())

    expected_mae_gain = comparison.baseline.mae - comparison.random_forest.mae
    expected_r2_gain = comparison.random_forest.r2 - comparison.baseline.r2

    assert comparison.mae_improvement == pytest.approx(expected_mae_gain)
    assert comparison.r2_improvement == pytest.approx(expected_r2_gain)

    # The winner must actually have the lower MAE.
    if comparison.best_model == "random_forest":
        assert comparison.random_forest.mae < comparison.baseline.mae
    else:
        assert comparison.baseline.mae <= comparison.random_forest.mae


def test_compare_models_is_reproducible() -> None:
    encoded_df = make_encoded_dataset()

    _, comparison_a = compare_models(encoded_df, random_state=42)
    _, comparison_b = compare_models(encoded_df, random_state=42)

    assert comparison_a.random_forest.mae == pytest.approx(comparison_b.random_forest.mae)
    assert comparison_a.random_forest.r2 == pytest.approx(comparison_b.random_forest.r2)


def test_save_forest_model_and_comparison_write_files(tmp_path: Path) -> None:
    forest, comparison = compare_models(make_encoded_dataset())

    model_path = save_forest_model(forest, tmp_path / "forest.joblib")
    comparison_path = save_comparison(comparison, tmp_path / "comparison.json")

    assert model_path.exists()
    saved = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert saved["best_model"] == comparison.best_model
    assert saved["baseline"]["mae"] == pytest.approx(comparison.baseline.mae)
    assert saved["random_forest"]["mae"] == pytest.approx(comparison.random_forest.mae)
