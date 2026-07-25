from pathlib import Path

import pytest

from saudi_car_ai.day02_refresh import (
    SAMPLE_CAR,
    build_car_summary,
    calculate_car_age,
    check_required_imports,
    load_summary,
    save_summary,
)


def test_calculate_car_age() -> None:
    assert calculate_car_age(2018, current_year=2026) == 8


def test_calculate_car_age_rejects_future_year() -> None:
    with pytest.raises(ValueError, match="future"):
        calculate_car_age(2030, current_year=2026)


def test_build_car_summary_normalizes_text_and_numbers() -> None:
    raw_car = {
        "make": " toyota ",
        "model": " camry ",
        "year": "2018",
        "mileage": "120000",
        "region": " riyadh ",
        "price": "58000",
    }

    summary = build_car_summary(raw_car, current_year=2026)

    assert summary == {
        "make": "Toyota",
        "model": "Camry",
        "region": "Riyadh",
        "year": 2018,
        "age": 8,
        "mileage": 120000,
        "price": 58000,
    }


def test_save_and_load_summary_round_trip(tmp_path: Path) -> None:
    summary = build_car_summary(SAMPLE_CAR, current_year=2026)
    output_path = save_summary(summary, tmp_path / "car_summary.json")

    assert output_path.exists()
    assert load_summary(output_path) == summary


def test_required_packages_can_be_imported() -> None:
    results = check_required_imports()

    assert all(results.values())
