from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
MODEL_DIR = PROJECT_ROOT / "models"

for path in (SRC_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from inference import BatteryPrognosticsEngine
from prognostics import add_causal_prognostic_features

from dashboard_utils import (
    build_capacity_chart,
    build_rul_chart,
    build_soh_chart,
    build_temperature_chart,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "cycle",
    "capacity_Ah",
    "avg_voltage_V",
    "max_temperature_C",
    "SOH_percent",
]

FEATURE_COLUMNS = [
    "SOH_lag1",
    "SOH_roll_mean_5",
    "SOH_roll_std_5",
    "SOH_delta_5",
    "temp_roll_mean_5",
    "temperature_delta_5",
    "voltage_roll_mean_5",
]


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Battery Intelligence Dashboard",
    page_icon="🔋",
    layout="wide",
)

st.title("🔋 Battery Intelligence Dashboard")

st.caption(
    "Production-oriented State-of-Health and Remaining Useful Life "
    "prognostics using causal degradation features."
)


# ---------------------------------------------------------------------
# Production model
# ---------------------------------------------------------------------

@st.cache_resource
def load_engine():
    return BatteryPrognosticsEngine(MODEL_DIR)


engine = load_engine()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("Model Configuration")

st.sidebar.metric(
    "Model Version",
    engine.model_version,
)

st.sidebar.metric(
    "EOL Threshold",
    f"{engine.eol_threshold:.1f}% SOH",
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload battery history CSV",
    type=["csv"],
)

st.sidebar.markdown("### Required CSV Columns")

for column in REQUIRED_COLUMNS:
    st.sidebar.code(column)

st.sidebar.markdown("---")

st.sidebar.caption(
    "The production model expects sequential discharge-cycle history "
    "and requires sufficient observations to calculate causal lagged "
    "and rolling degradation features."
)


# ---------------------------------------------------------------------
# Landing state
# ---------------------------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload a battery-history CSV from the sidebar to run "
        "SOH and RUL prognostics."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Causal Feature Engineering")
        st.write(
            "RUL prediction uses lagged and rolling battery-health "
            "indicators derived only from historical observations."
        )

    with c2:
        st.markdown("### Physical Reporting Constraint")
        st.write(
            "Raw RUL predictions are retained for diagnostics while "
            "reported RUL is constrained to remain nonnegative."
        )

    with c3:
        st.markdown("### End-of-Life Policy")
        st.write(
            f"EOL is defined at or below {engine.eol_threshold:.1f}% SOH. "
            "Once observed SOH reaches this threshold, reported RUL is "
            "forced to zero."
        )

    st.stop()


# ---------------------------------------------------------------------
# Read input data
# ---------------------------------------------------------------------

try:
    battery_df = pd.read_csv(uploaded_file)

except Exception as exc:
    st.error(
        f"Unable to read CSV: {exc}"
    )
    st.stop()


# ---------------------------------------------------------------------
# Required-column validation
# ---------------------------------------------------------------------

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in battery_df.columns
]

if missing_columns:
    st.error(
        "CSV is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ---------------------------------------------------------------------
# Input quality summary
# ---------------------------------------------------------------------

quality_checks = {
    "Rows": len(battery_df),
    "Missing values": int(
        battery_df[
            REQUIRED_COLUMNS
        ].isna().sum().sum()
    ),
    "Duplicate cycles": int(
        battery_df[
            "cycle"
        ].duplicated().sum()
    ),
    "Chronological": bool(
        battery_df[
            "cycle"
        ].is_monotonic_increasing
    ),
}

quality_pass = (
    quality_checks["Rows"] >= 7
    and quality_checks["Missing values"] == 0
    and quality_checks["Duplicate cycles"] == 0
    and quality_checks["Chronological"]
)


# ---------------------------------------------------------------------
# Latest production prediction
# ---------------------------------------------------------------------

try:
    prediction = engine.predict(
        battery_df
    )

except Exception as exc:
    st.error(
        f"Prediction failed: {exc}"
    )
    st.stop()


# ---------------------------------------------------------------------
# Build causal trajectory
# ---------------------------------------------------------------------

featured_df = add_causal_prognostic_features(
    battery_df
)

valid_featured_df = featured_df.dropna(
    subset=FEATURE_COLUMNS
).copy()


predicted_soh_values = []
raw_rul_values = []
reported_rul_values = []


for _, row in valid_featured_df.iterrows():

    current_cycle = int(
        row["cycle"]
    )

    history = battery_df.loc[
        battery_df["cycle"]
        <= current_cycle
    ].copy()

    try:
        point_prediction = engine.predict(
            history
        )

        predicted_soh_values.append(
            point_prediction[
                "predicted_soh_percent"
            ]
        )

        raw_rul_values.append(
            point_prediction[
                "raw_rul_cycles"
            ]
        )

        reported_rul_values.append(
            point_prediction[
                "reported_rul_cycles"
            ]
        )

    except Exception:
        predicted_soh_values.append(
            np.nan
        )

        raw_rul_values.append(
            np.nan
        )

        reported_rul_values.append(
            np.nan
        )


valid_featured_df[
    "predicted_SOH_percent"
] = predicted_soh_values

valid_featured_df[
    "raw_RUL_cycles"
] = raw_rul_values

valid_featured_df[
    "reported_RUL_cycles"
] = reported_rul_values


# ---------------------------------------------------------------------
# Derived dashboard metrics
# ---------------------------------------------------------------------

soh_gap = (
    prediction[
        "observed_soh_percent"
    ]
    -
    prediction[
        "predicted_soh_percent"
    ]
)

soh_margin = (
    prediction[
        "observed_soh_percent"
    ]
    -
    engine.eol_threshold
)

latest_temperature = float(
    battery_df.iloc[-1][
        "max_temperature_C"
    ]
)


# ---------------------------------------------------------------------
# Battery health summary
# ---------------------------------------------------------------------

st.subheader(
    "Battery Health Summary"
)

k1, k2, k3, k4, k5 = st.columns(5)


k1.metric(
    "Latest Cycle",
    f"{prediction['cycle']:.0f}",
)


k2.metric(
    "Observed SOH",
    f"{prediction['observed_soh_percent']:.2f}%",
)


k3.metric(
    "Predicted SOH",
    f"{prediction['predicted_soh_percent']:.2f}%",
    delta=(
        f"{-soh_gap:.2f} pp vs observed"
    ),
)


k4.metric(
    "Reported RUL",
    f"{prediction['reported_rul_cycles']:.1f} cycles",
    help=(
        "Estimated remaining discharge cycles "
        "before the configured EOL threshold."
    ),
)


k5.metric(
    "SOH Margin to EOL",
    f"{soh_margin:.2f} pp",
    help=(
        "Observed SOH minus the configured "
        "end-of-life SOH threshold."
    ),
)


# ---------------------------------------------------------------------
# Health-state message
# ---------------------------------------------------------------------

st.markdown("---")


if prediction["eol_reached"]:

    st.error(
        "Observed SOH has reached or crossed the configured EOL "
        "threshold. Reported RUL is therefore forced to 0 cycles."
    )

else:

    st.success(
        "Observed SOH remains above the configured EOL threshold "
        f"by {soh_margin:.2f} percentage points."
    )


# ---------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------

with st.expander(
    "Input Data Quality Checks",
    expanded=False,
):

    q1, q2, q3, q4 = st.columns(4)


    q1.metric(
        "Rows",
        quality_checks["Rows"],
    )


    q2.metric(
        "Missing Values",
        quality_checks[
            "Missing values"
        ],
    )


    q3.metric(
        "Duplicate Cycles",
        quality_checks[
            "Duplicate cycles"
        ],
    )


    q4.metric(
        "Chronological",
        (
            "PASS"
            if quality_checks[
                "Chronological"
            ]
            else "FAIL"
        ),
    )


    if quality_pass:
        st.success(
            "Input data quality checks passed."
        )

    else:
        st.warning(
            "One or more input quality checks require attention."
        )


# ---------------------------------------------------------------------
# SOH trajectory
# ---------------------------------------------------------------------

st.markdown("---")

st.subheader(
    "State-of-Health Degradation"
)


soh_chart = build_soh_chart(
    valid_featured_df,
    engine.eol_threshold,
)


st.altair_chart(
    soh_chart,
    use_container_width=True,
)


st.caption(
    "Observed SOH is compared against the causal model estimate. "
    "The dashed horizontal line represents the configured "
    f"{engine.eol_threshold:.0f}% SOH end-of-life threshold."
)


# ---------------------------------------------------------------------
# RUL trajectory
# ---------------------------------------------------------------------

st.markdown("---")

st.subheader(
    "Remaining Useful Life Trajectory"
)


rul_chart = build_rul_chart(
    valid_featured_df
)


st.altair_chart(
    rul_chart,
    use_container_width=True,
)


st.caption(
    "Raw RUL is preserved for model diagnostics. "
    "Reported RUL applies the nonnegative constraint "
    "and observed-SOH EOL safeguard."
)


# ---------------------------------------------------------------------
# Engineering diagnostics
# ---------------------------------------------------------------------

st.markdown("---")

st.subheader(
    "Engineering Diagnostics"
)

left, right = st.columns(2)


with left:

    st.markdown(
        "#### Capacity Degradation"
    )

    st.altair_chart(
        build_capacity_chart(
            battery_df
        ),
        use_container_width=True,
    )


with right:

    st.markdown(
        "#### Maximum Cell Temperature"
    )

    st.altair_chart(
        build_temperature_chart(
            battery_df
        ),
        use_container_width=True,
    )


st.caption(
    f"Latest recorded maximum cell temperature: "
    f"{latest_temperature:.2f} °C."
)


# ---------------------------------------------------------------------
# Latest prognostic output
# ---------------------------------------------------------------------

st.markdown("---")

st.subheader(
    "Latest Prognostic Output"
)


diagnostic_df = pd.DataFrame(
    {
        "Metric": [
            "Cycle",
            "Observed SOH",
            "Predicted SOH",
            "Observed − Predicted SOH",
            "SOH margin to EOL",
            "Raw RUL",
            "Reported RUL",
            "EOL threshold",
            "EOL reached",
            "Model version",
        ],
        "Value": [
            f"{prediction['cycle']:.0f}",
            (
                f"{prediction['observed_soh_percent']:.3f}%"
            ),
            (
                f"{prediction['predicted_soh_percent']:.3f}%"
            ),
            (
                f"{soh_gap:+.3f} percentage points"
            ),
            (
                f"{soh_margin:+.3f} percentage points"
            ),
            (
                f"{prediction['raw_rul_cycles']:.3f} cycles"
            ),
            (
                f"{prediction['reported_rul_cycles']:.3f} cycles"
            ),
            (
                f"{prediction['eol_threshold_soh_percent']:.1f}%"
            ),
            prediction[
                "eol_reached"
            ],
            prediction[
                "model_version"
            ],
        ],
    }
)


st.dataframe(
    diagnostic_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Raw battery history
# ---------------------------------------------------------------------

with st.expander(
    "Inspect Battery History",
    expanded=False,
):

    st.dataframe(
        battery_df.tail(30),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Methodology and deployment limitations
# ---------------------------------------------------------------------

with st.expander(
    "Model Methodology & Deployment Limitations",
    expanded=False,
):

    st.markdown(
        """
### State-of-Health Model

- Uses causal lagged degradation information.
- The model estimates present battery health using preceding history.
- SOH disagreement should be interpreted as model-state disagreement,
  not automatically as measurement error.

### Remaining Useful Life Model

- Uses historical SOH behaviour, rolling degradation statistics,
  temperature evolution and voltage history.
- Current cycle number is not used as an RUL predictor.
- Current SOH and current capacity are not direct RUL predictors.
- Raw linear-model output is preserved for diagnostics.
- Reported RUL is constrained to remain nonnegative.

### Model Validation

Model-family selection was performed using leave-one-battery-out
validation across four NASA lithium-ion ageing cells.

The final production artifacts were subsequently refitted using all
four cells after model selection.

Therefore the exact serialized production artifact does not have an
additional unseen fifth-cell validation dataset.

### Uncertainty

A probabilistic confidence interval is intentionally not displayed.

Chronological conformal experiments did not provide reliable coverage
under battery degradation regime shift.

Displaying such intervals here would imply more confidence than the
validation evidence supports.

### Interpretation

Feature coefficients and model explanations represent conditional
statistical associations.

They should not be interpreted as causal physical mechanisms.

### Deployment Scope

This dashboard is an engineering analytics and portfolio application.

It is not a certified Battery Management System safety function and
should not be used as the sole basis for operational or safety-critical
battery decisions.
        """
    )