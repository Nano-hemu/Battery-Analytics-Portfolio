from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from prognostics import (
    PROGNOSTIC_FEATURES,
    add_causal_prognostic_features,
    constrain_rul,
    validate_battery_history,
)


class BatteryPrognosticsEngine:
    """
    Production-style inference interface for battery SOH and RUL models.

    The engine:
    - loads serialized sklearn pipelines,
    - validates model metadata,
    - validates incoming battery history,
    - constructs the latest leakage-safe causal feature vector,
    - generates SOH and RUL estimates,
    - retains raw RUL for diagnostics,
    - applies a nonnegative reporting constraint.
    """

    def __init__(
        self,
        model_dir: str | Path,
    ) -> None:
        self.model_dir = Path(model_dir)

        self.soh_model_path = (
            self.model_dir / "soh_model.joblib"
        )
        self.rul_model_path = (
            self.model_dir / "rul_model.joblib"
        )
        self.metadata_path = (
            self.model_dir / "model_metadata.json"
        )

        self._validate_artifact_paths()

        self.soh_model = joblib.load(
            self.soh_model_path
        )

        self.rul_model = joblib.load(
            self.rul_model_path
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self._validate_metadata()


    def _validate_artifact_paths(self) -> None:
        required_paths = [
            self.soh_model_path,
            self.rul_model_path,
            self.metadata_path,
        ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing model artifacts: "
                + ", ".join(missing)
            )


    def _validate_metadata(self) -> None:
        feature_order = self.metadata[
            "feature_engineering"
        ]["feature_order"]

        if feature_order != PROGNOSTIC_FEATURES:
            raise ValueError(
                "Model metadata feature schema does not "
                "match the production feature schema."
            )

        threshold = self.metadata[
            "targets"
        ]["rul"]["eol_threshold_soh_percent"]

        if float(threshold) != 80.0:
            raise ValueError(
                "Unsupported EOL threshold in model metadata."
            )


    @property
    def model_version(self) -> str:
        return str(
            self.metadata["model_version"]
        )


    @property
    def eol_threshold(self) -> float:
        return float(
            self.metadata[
                "targets"
            ]["rul"]["eol_threshold_soh_percent"]
        )


    def prepare_latest_features(
        self,
        battery_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate battery history and create the latest causal
        feature vector.

        For prediction at cycle t, all prognostic features use
        information available only through cycle t-1.
        """

        validate_battery_history(
            battery_history
        )

        featured = add_causal_prognostic_features(
            battery_history
        )

        latest = featured.iloc[[-1]].copy()

        missing_features = [
            feature
            for feature in PROGNOSTIC_FEATURES
            if pd.isna(latest.iloc[0][feature])
        ]

        if missing_features:
            raise ValueError(
                "Insufficient battery history to construct "
                "production prognostic features. "
                f"Unavailable features: {missing_features}"
            )

        X_latest = latest[
            PROGNOSTIC_FEATURES
        ].copy()

        feature_values = X_latest.to_numpy(
            dtype=float
        )

        if not np.isfinite(feature_values).all():
            raise ValueError(
                "Latest prognostic feature vector contains "
                "non-finite values."
            )

        return X_latest


    def predict(
        self,
        battery_history: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate current SOH and RUL estimates from discharge-level
        battery history.
        """

        X_latest = self.prepare_latest_features(
            battery_history
        )

        latest_cycle = int(
            battery_history.iloc[-1]["cycle"]
        )

        latest_observed_soh = float(
            battery_history.iloc[-1][
                "SOH_percent"
            ]
        )

        predicted_soh = float(
            self.soh_model.predict(
                X_latest
            )[0]
        )

        raw_rul = float(
            self.rul_model.predict(
                X_latest
            )[0]
        )

        reported_rul = float(
            constrain_rul(raw_rul)
        )

        eol_reached = bool(
            latest_observed_soh
            <= self.eol_threshold
        )

        if eol_reached:
            reported_rul = 0.0

        return {
            "model_version": self.model_version,
            "cycle": latest_cycle,
            "observed_soh_percent": (
                latest_observed_soh
            ),
            "predicted_soh_percent": (
                predicted_soh
            ),
            "raw_rul_cycles": raw_rul,
            "reported_rul_cycles": (
                reported_rul
            ),
            "eol_threshold_soh_percent": (
                self.eol_threshold
            ),
            "eol_reached": eol_reached,
        }