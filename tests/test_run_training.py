import json
from pathlib import Path

import pandas as pd
import pytest

from saudi_car_ai.features.build_features import build_encoded_features
from saudi_car_ai.models.run_training import (
    TrainingRun,
    run_training,
    save_production_model,
    save_training_run,
)


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


def test_run_training_selects_lowest_mae_model() -> None:
    encoded_df = make_encoded_dataset()

    _model, run = run_training(encoded_df)

    assert isinstance(run, TrainingRun)
    assert run.selected_model in run.all_metrics
    lowest_mae = min(metrics.mae for metrics in run.all_metrics.values())
    assert run.selected_metrics.mae == pytest.approx(lowest_mae)
    assert run.train_rows + run.test_rows == run.total_rows


def test_run_training_records_split_settings() -> None:
    encoded_df = make_encoded_dataset()

    _model, run = run_training(encoded_df, test_size=0.25, random_state=7)

    assert run.test_size == 0.25
    assert run.random_state == 7
    assert set(run.all_metrics) == {"baseline", "random_forest"}


def test_run_training_is_reproducible() -> None:
    encoded_df = make_encoded_dataset()

    _model_a, run_a = run_training(encoded_df, random_state=42)
    _model_b, run_b = run_training(encoded_df, random_state=42)

    assert run_a.selected_model == run_b.selected_model
    assert run_a.selected_metrics.mae == pytest.approx(run_b.selected_metrics.mae)
    assert run_a.selected_metrics.r2 == pytest.approx(run_b.selected_metrics.r2)


def test_save_production_model_and_run_write_files(tmp_path: Path) -> None:
    model, run = run_training(make_encoded_dataset())

    model_path = save_production_model(model, tmp_path / "production_model.joblib")
    run_path = save_training_run(run, tmp_path / "training_run.json")

    assert model_path.exists()
    saved = json.loads(run_path.read_text(encoding="utf-8"))
    assert saved["selected_model"] == run.selected_model
    assert "created_at" in saved
    assert set(saved["all_metrics"]) == {"baseline", "random_forest"}
    assert saved["selected_metrics"]["mae"] == pytest.approx(run.selected_metrics.mae)
