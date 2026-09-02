from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from inference import BatteryPrognosticsEngine


@pytest.fixture(scope="module")
def engine():
    return BatteryPrognosticsEngine(
        MODELS_DIR
    )


@pytest.fixture
def synthetic_history():
    cycles = np.arange(1, 21)

    capacity = np.linspace(
        1.85,
        1.65,
        len(cycles),
    )

    soh = (
        capacity
        / capacity[0]
        * 100.0
    )

    return pd.DataFrame(
        {
            "cycle": cycles,
            "capacity_Ah": capacity,
            "avg_voltage_V": np.linspace(
                3.75,
                3.68,
                len(cycles),
            ),
            "max_temperature_C": np.linspace(
                31.0,
                35.0,
                len(cycles),
            ),
            "SOH_percent": soh,
        }
    )


def test_synthetic_prediction_is_finite(
    engine,
    synthetic_history,
):
    result = engine.predict(
        synthetic_history
    )

    assert np.isfinite(
        result["predicted_soh_percent"]
    )

    assert np.isfinite(
        result["raw_rul_cycles"]
    )

    assert (
        result["reported_rul_cycles"]
        >= 0.0
    )


def test_synthetic_schema(
    engine,
    synthetic_history,
):
    result = engine.predict(
        synthetic_history
    )

    expected_keys = {
        "model_version",
        "cycle",
        "observed_soh_percent",
        "predicted_soh_percent",
        "raw_rul_cycles",
        "reported_rul_cycles",
        "eol_threshold_soh_percent",
        "eol_reached",
    }

    assert set(result.keys()) == expected_keys


def test_synthetic_insufficient_history(
    engine,
    synthetic_history,
):
    short_history = (
        synthetic_history
        .iloc[:5]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="Insufficient battery history",
    ):
        engine.predict(short_history)


def test_synthetic_duplicate_cycle(
    engine,
    synthetic_history,
):
    bad_history = synthetic_history.copy()

    bad_history.loc[
        bad_history.index[-1],
        "cycle",
    ] = bad_history.iloc[-2]["cycle"]

    with pytest.raises(
        ValueError,
        match="Duplicate discharge-cycle",
    ):
        engine.predict(bad_history)


def test_synthetic_missing_column(
    engine,
    synthetic_history,
):
    bad_history = (
        synthetic_history
        .drop(columns=["avg_voltage_V"])
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="Missing required battery columns",
    ):
        engine.predict(bad_history)