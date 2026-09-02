from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EOL_THRESHOLD_SOH = 80.0
DEFAULT_FEATURE_WINDOW = 5


PROGNOSTIC_FEATURES = [
    "SOH_lag1",
    "SOH_roll_mean_5",
    "SOH_roll_std_5",
    "SOH_delta_5",
    "temp_roll_mean_5",
    "temperature_delta_5",
    "voltage_roll_mean_5",
]


@dataclass(frozen=True)
class BatteryEOL:
    battery_id: str
    eol_cycle: int
    soh_at_eol: float
    previous_soh: float | None


def validate_battery_history(df: pd.DataFrame) -> None:
    """
    Validate discharge-level battery telemetry before feature engineering.

    Raises
    ------
    ValueError
        If required columns are missing or if the discharge history is
        not suitable for chronological prognostic inference.
    """

    required_columns = {
        "cycle",
        "capacity_Ah",
        "avg_voltage_V",
        "max_temperature_C",
        "SOH_percent",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required battery columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Battery history is empty.")

    if df["cycle"].isna().any():
        raise ValueError("Cycle column contains missing values.")

    if not np.isfinite(df["cycle"]).all():
        raise ValueError("Cycle column contains non-finite values.")

    if df["cycle"].duplicated().any():
        raise ValueError("Duplicate discharge-cycle values detected.")

    cycle_values = df["cycle"].to_numpy()

    if np.any(np.diff(cycle_values) <= 0):
        raise ValueError(
            "Battery history must be ordered by strictly increasing "
            "discharge cycle."
        )

    numeric_columns = [
        "capacity_Ah",
        "avg_voltage_V",
        "max_temperature_C",
        "SOH_percent",
    ]

    for column in numeric_columns:
        if df[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values."
            )

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Column '{column}' contains non-finite values."
            )


def add_causal_prognostic_features(
    df: pd.DataFrame,
    window: int = DEFAULT_FEATURE_WINDOW,
) -> pd.DataFrame:
    """
    Create leakage-safe degradation features.

    Features associated with discharge cycle t use battery information
    available only through cycle t-1.

    Parameters
    ----------
    df
        Discharge-level battery history containing SOH, capacity,
        voltage and temperature measurements.

    window
        Rolling-history window length.

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with causal degradation features.
    """

    if window < 2:
        raise ValueError("Feature window must be at least 2 cycles.")

    validate_battery_history(df)

    result = (
        df.copy()
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    result["SOH_lag1"] = result["SOH_percent"].shift(1)
    result["capacity_lag1"] = result["capacity_Ah"].shift(1)

    past_soh = result["SOH_percent"].shift(1)
    past_capacity = result["capacity_Ah"].shift(1)
    past_temperature = result["max_temperature_C"].shift(1)
    past_voltage = result["avg_voltage_V"].shift(1)

    result["SOH_roll_mean_5"] = (
        past_soh
        .rolling(window=window, min_periods=window)
        .mean()
    )

    result["SOH_roll_std_5"] = (
        past_soh
        .rolling(window=window, min_periods=window)
        .std()
    )

    result["capacity_roll_mean_5"] = (
        past_capacity
        .rolling(window=window, min_periods=window)
        .mean()
    )

    result["temp_roll_mean_5"] = (
        past_temperature
        .rolling(window=window, min_periods=window)
        .mean()
    )

    result["voltage_roll_mean_5"] = (
        past_voltage
        .rolling(window=window, min_periods=window)
        .mean()
    )

    lag_distance = window + 1

    result["SOH_delta_5"] = (
        result["SOH_percent"].shift(1)
        - result["SOH_percent"].shift(lag_distance)
    ) / window

    result["capacity_delta_5"] = (
        result["capacity_Ah"].shift(1)
        - result["capacity_Ah"].shift(lag_distance)
    ) / window

    result["temperature_delta_5"] = (
        result["max_temperature_C"].shift(1)
        - result["max_temperature_C"].shift(lag_distance)
    ) / window

    return result


def identify_eol(
    df: pd.DataFrame,
    battery_id: str,
    threshold: float = EOL_THRESHOLD_SOH,
) -> BatteryEOL:
    """
    Identify the first measured discharge cycle at or below the
    specified SOH threshold.
    """

    validate_battery_history(df)

    crossing = df.loc[df["SOH_percent"] <= threshold]

    if crossing.empty:
        raise ValueError(
            f"{battery_id} does not reach the "
            f"{threshold:.1f}% SOH threshold."
        )

    first_index = crossing.index[0]
    eol_row = df.loc[first_index]

    previous_soh = None

    position = df.index.get_loc(first_index)

    if position > 0:
        previous_soh = float(
            df.iloc[position - 1]["SOH_percent"]
        )

    return BatteryEOL(
        battery_id=battery_id,
        eol_cycle=int(eol_row["cycle"]),
        soh_at_eol=float(eol_row["SOH_percent"]),
        previous_soh=previous_soh,
    )


def build_rul_dataset(
    df: pd.DataFrame,
    battery_id: str,
    threshold: float = EOL_THRESHOLD_SOH,
    features: Iterable[str] = PROGNOSTIC_FEATURES,
) -> tuple[pd.DataFrame, BatteryEOL]:
    """
    Construct a leakage-safe pre-EOL RUL modelling dataset.

    RUL is defined as:

        RUL(t) = observed_EOL_cycle - discharge_cycle(t)

    The EOL threshold is used only for historical target construction.
    """

    feature_columns = list(features)

    featured = add_causal_prognostic_features(df)

    eol = identify_eol(
        featured,
        battery_id=battery_id,
        threshold=threshold,
    )

    pre_eol = featured.loc[
        featured["cycle"] < eol.eol_cycle
    ].copy()

    pre_eol["RUL_cycles"] = (
        eol.eol_cycle - pre_eol["cycle"]
    )

    required = feature_columns + ["RUL_cycles"]

    dataset = (
        pre_eol
        .dropna(subset=required)
        .reset_index(drop=True)
    )

    if dataset.empty:
        raise ValueError(
            f"No usable RUL observations were generated "
            f"for {battery_id}."
        )

    return dataset, eol


def build_soh_dataset(
    df: pd.DataFrame,
    features: Iterable[str] = PROGNOSTIC_FEATURES,
) -> pd.DataFrame:
    """
    Construct the causal SOH modelling dataset.

    Unlike RUL training, SOH state-estimation uses the complete
    available degradation trajectory, including observations after
    the 80% SOH EOL threshold.
    """

    feature_columns = list(features)

    featured = add_causal_prognostic_features(df)

    dataset = (
        featured
        .dropna(
            subset=feature_columns + ["SOH_percent"]
        )
        .reset_index(drop=True)
    )

    if dataset.empty:
        raise ValueError(
            "No usable SOH observations were generated."
        )

    return dataset


def constrain_rul(raw_rul: float | np.ndarray) -> float | np.ndarray:
    """
    Apply the hard physical constraint RUL >= 0.

    Raw model predictions should still be retained separately
    for diagnostics and validation.
    """

    constrained = np.maximum(
        np.asarray(raw_rul, dtype=float),
        0.0,
    )

    if constrained.ndim == 0:
        return float(constrained)

    return constrained