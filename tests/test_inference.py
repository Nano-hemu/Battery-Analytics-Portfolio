from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from nasa_loader import load_nasa_battery
from battery_features import add_soh_features
from inference import BatteryPrognosticsEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture(scope="module")
def engine():
    return BatteryPrognosticsEngine(
        MODELS_DIR
    )


@pytest.fixture(scope="module")
def b0005():
    df = load_nasa_battery(
        DATA_DIR / "B0005.mat",
        "B0005",
    )

    return add_soh_features(df)


def test_engine_loads(engine):
    assert engine.model_version == "1.0.0"
    assert engine.eol_threshold == 80.0


def test_pre_eol_prediction(engine, b0005):
    history = b0005.loc[
        b0005["cycle"] <= 90
    ].copy()

    result = engine.predict(history)

    assert result["cycle"] == 90
    assert result["eol_reached"] is False

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


def test_eol_safeguard(engine, b0005):
    history = b0005.loc[
        b0005["cycle"] <= 101
    ].copy()

    result = engine.predict(history)

    assert result["eol_reached"] is True
    assert result["reported_rul_cycles"] == 0.0


def test_insufficient_history(engine, b0005):
    history = b0005.iloc[:5].copy()

    with pytest.raises(
        ValueError,
        match="Insufficient battery history",
    ):
        engine.predict(history)


def test_duplicate_cycle(engine, b0005):
    history = b0005.iloc[:20].copy()

    history.loc[
        history.index[-1],
        "cycle",
    ] = history.iloc[-2]["cycle"]

    with pytest.raises(
        ValueError,
        match="Duplicate discharge-cycle",
    ):
        engine.predict(history)


def test_missing_column(engine, b0005):
    history = (
        b0005.iloc[:20]
        .drop(
            columns=["max_temperature_C"]
        )
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="Missing required battery columns",
    ):
        engine.predict(history)


def test_missing_numeric_value(engine, b0005):
    history = b0005.iloc[:20].copy()

    history.loc[
        history.index[5],
        "SOH_percent",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        engine.predict(history)


def test_non_chronological_history(
    engine,
    b0005,
):
    history = b0005.iloc[:20].copy()

    history = pd.concat(
        [
            history.iloc[:10],
            history.iloc[10:].iloc[::-1],
        ]
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        engine.predict(history)