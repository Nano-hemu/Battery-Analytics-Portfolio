import numpy as np
import pandas as pd


def add_soh_features(df):
    """Add capacity retention and SOH-related features."""

    df = df.copy()

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    required_columns = {"cycle", "capacity_Ah"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    initial_capacity = df["capacity_Ah"].iloc[0]

    if initial_capacity <= 0:
        raise ValueError("Initial capacity must be greater than zero.")

    df["SOH_percent"] = (
        df["capacity_Ah"] / initial_capacity
    ) * 100

    df["capacity_retention"] = (
        df["capacity_Ah"] / initial_capacity
    )

    df["capacity_fade_percent"] = (
        1 - df["capacity_retention"]
    ) * 100

    return df


def add_degradation_features(df, window=5):
    """Add rolling and degradation-related features."""

    df = df.copy()

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if window < 2:
        raise ValueError("Rolling window must be at least 2.")

    if "SOH_percent" not in df.columns:
        raise ValueError(
            "SOH_percent not found. Run add_soh_features() first."
        )

    df["SOH_change_percent"] = (
        df["SOH_percent"].diff()
    )

    df["SOH_rolling_mean"] = (
        df["SOH_percent"]
        .rolling(window=window)
        .mean()
    )

    if "avg_temperature_C" in df.columns:
        df["temperature_rolling_mean"] = (
            df["avg_temperature_C"]
            .rolling(window=window)
            .mean()
        )

    return df