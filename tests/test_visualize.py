from pathlib import Path

import numpy as np
import pandas as pd

from saudi_car_ai.features.build_features import build_encoded_features
from saudi_car_ai.models.visualize import (
    PredictionBundle,
    build_model_predictions,
    save_all_charts,
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


def test_build_model_predictions_matches_test_rows() -> None:
    encoded_df = make_encoded_dataset()

    bundle = build_model_predictions(encoded_df)

    assert isinstance(bundle, PredictionBundle)
    assert len(bundle.baseline_pred) == len(bundle.y_test)
    assert len(bundle.forest_pred) == len(bundle.y_test)
    assert np.isfinite(bundle.forest_pred).all()


def test_save_all_charts_writes_four_png_files(tmp_path: Path) -> None:
    saved_paths = save_all_charts(make_encoded_dataset(), tmp_path)

    assert len(saved_paths) == 4
    for path in saved_paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
