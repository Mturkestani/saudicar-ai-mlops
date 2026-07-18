def test_core_ml_stack_imports() -> None:
    import fastapi
    import joblib
    import numpy
    import pandas
    import pydantic
    import sklearn

    assert pandas is not None
    assert numpy is not None
    assert sklearn is not None
    assert fastapi is not None
    assert pydantic is not None
    assert joblib is not None
