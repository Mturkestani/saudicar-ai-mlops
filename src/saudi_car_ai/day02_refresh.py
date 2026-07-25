"""Day 2 Python refresh exercises for the SaudiCar AI bootcamp."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

SAMPLE_CAR: dict[str, Any] = {
    "make": "Toyota",
    "model": "Camry",
    "year": 2018,
    "mileage": 120_000,
    "region": "Riyadh",
    "price": 58_000,
}

REQUIRED_IMPORTS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
}

DEFAULT_OUTPUT_PATH = Path("data/processed/day02_sample_car.json")


def calculate_car_age(year: int, current_year: int = 2026) -> int:
    """Return the age of a car in years."""
    if year > current_year:
        msg = "Car year cannot be in the future."
        raise ValueError(msg)

    return current_year - year


def normalize_text(value: str) -> str:
    """Clean simple text values for display or later feature work."""
    return value.strip().title()


def build_car_summary(car: dict[str, Any], current_year: int = 2026) -> dict[str, Any]:
    """Create a clean summary from one raw car listing."""
    return {
        "make": normalize_text(str(car["make"])),
        "model": normalize_text(str(car["model"])),
        "region": normalize_text(str(car["region"])),
        "year": int(car["year"]),
        "age": calculate_car_age(int(car["year"]), current_year=current_year),
        "mileage": int(car["mileage"]),
        "price": int(car["price"]),
    }


def save_summary(summary: dict[str, Any], path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Save a summary as JSON and return the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def load_summary(path: Path) -> dict[str, Any]:
    """Load a saved JSON summary."""
    return json.loads(path.read_text(encoding="utf-8"))


def check_required_imports(required_imports: dict[str, str] = REQUIRED_IMPORTS) -> dict[str, bool]:
    """Check whether the project packages can be imported."""
    results: dict[str, bool] = {}

    for import_name, package_name in required_imports.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            results[package_name] = False
        else:
            results[package_name] = True

    return results


def main() -> None:
    """Run the Day 2 practice exercise."""
    summary = build_car_summary(SAMPLE_CAR)
    output_path = save_summary(summary)
    package_results = check_required_imports()

    print("Day 2 Python refresh")
    print("====================")
    print(f"Sample car: {summary['year']} {summary['make']} {summary['model']}")
    print(f"Age: {summary['age']} years")
    print(f"Saved JSON: {output_path}")
    print()
    print("Package import checks")
    for package_name, is_ready in package_results.items():
        status = "ready" if is_ready else "missing"
        print(f"- {package_name}: {status}")


if __name__ == "__main__":
    main()
