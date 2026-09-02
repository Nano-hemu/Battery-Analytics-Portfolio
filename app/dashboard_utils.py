import altair as alt
import pandas as pd


def build_soh_chart(
    data: pd.DataFrame,
    eol_threshold: float,
) -> alt.Chart:
    """
    Build the observed-vs-predicted SOH trajectory with
    the configured end-of-life threshold.
    """

    plot_df = data[
        [
            "cycle",
            "SOH_percent",
            "predicted_SOH_percent",
        ]
    ].copy()

    long_df = plot_df.melt(
        id_vars="cycle",
        value_vars=[
            "SOH_percent",
            "predicted_SOH_percent",
        ],
        var_name="Series",
        value_name="SOH",
    )

    long_df["Series"] = long_df["Series"].replace(
        {
            "SOH_percent": "Observed SOH",
            "predicted_SOH_percent": "Predicted SOH",
        }
    )

    y_min = max(
        0,
        min(
            float(long_df["SOH"].min()),
            float(eol_threshold),
        )
        - 5,
    )

    lines = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "cycle:Q",
                title="Discharge Cycle",
            ),
            y=alt.Y(
                "SOH:Q",
                title="State of Health (%)",
                scale=alt.Scale(
                    domain=[y_min, 102]
                ),
            ),
            color=alt.Color(
                "Series:N",
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "cycle:Q",
                    title="Cycle",
                    format=".0f",
                ),
                alt.Tooltip(
                    "Series:N",
                    title="Series",
                ),
                alt.Tooltip(
                    "SOH:Q",
                    title="SOH (%)",
                    format=".2f",
                ),
            ],
        )
    )

    threshold_df = pd.DataFrame(
        {
            "EOL": [eol_threshold],
        }
    )

    threshold_rule = (
        alt.Chart(threshold_df)
        .mark_rule(
            strokeDash=[8, 5],
            size=2,
        )
        .encode(
            y="EOL:Q"
        )
    )

    label_df = pd.DataFrame(
        {
            "cycle": [
                float(
                    plot_df["cycle"].min()
                )
            ],
            "EOL": [
                eol_threshold
            ],
            "label": [
                f"EOL Threshold ({eol_threshold:.0f}% SOH)"
            ],
        }
    )

    threshold_label = (
        alt.Chart(label_df)
        .mark_text(
            align="left",
            baseline="bottom",
            dx=5,
            dy=-5,
        )
        .encode(
            x="cycle:Q",
            y="EOL:Q",
            text="label:N",
        )
    )

    return (
        (
            lines
            + threshold_rule
            + threshold_label
        )
        .properties(height=420)
        .interactive()
    )


def build_rul_chart(
    data: pd.DataFrame,
) -> alt.Chart:
    """
    Build raw and production-reported RUL trajectories.
    """

    plot_df = data[
        [
            "cycle",
            "raw_RUL_cycles",
            "reported_RUL_cycles",
        ]
    ].copy()

    long_df = plot_df.melt(
        id_vars="cycle",
        value_vars=[
            "raw_RUL_cycles",
            "reported_RUL_cycles",
        ],
        var_name="Series",
        value_name="RUL",
    )

    long_df["Series"] = long_df["Series"].replace(
        {
            "raw_RUL_cycles": "Raw RUL",
            "reported_RUL_cycles": "Reported RUL",
        }
    )

    return (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "cycle:Q",
                title="Discharge Cycle",
            ),
            y=alt.Y(
                "RUL:Q",
                title="Remaining Useful Life (cycles)",
            ),
            color=alt.Color(
                "Series:N",
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "cycle:Q",
                    title="Cycle",
                    format=".0f",
                ),
                alt.Tooltip(
                    "Series:N",
                    title="Series",
                ),
                alt.Tooltip(
                    "RUL:Q",
                    title="RUL",
                    format=".2f",
                ),
            ],
        )
        .properties(height=420)
        .interactive()
    )


def build_capacity_chart(
    data: pd.DataFrame,
) -> alt.Chart:
    """
    Build measured capacity degradation trajectory.
    """

    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "cycle:Q",
                title="Discharge Cycle",
            ),
            y=alt.Y(
                "capacity_Ah:Q",
                title="Capacity (Ah)",
                scale=alt.Scale(
                    zero=False
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "cycle:Q",
                    title="Cycle",
                    format=".0f",
                ),
                alt.Tooltip(
                    "capacity_Ah:Q",
                    title="Capacity (Ah)",
                    format=".4f",
                ),
            ],
        )
        .properties(height=330)
        .interactive()
    )


def build_temperature_chart(
    data: pd.DataFrame,
) -> alt.Chart:
    """
    Build maximum cell-temperature trajectory.
    """

    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "cycle:Q",
                title="Discharge Cycle",
            ),
            y=alt.Y(
                "max_temperature_C:Q",
                title="Maximum Temperature (°C)",
                scale=alt.Scale(
                    zero=False
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "cycle:Q",
                    title="Cycle",
                    format=".0f",
                ),
                alt.Tooltip(
                    "max_temperature_C:Q",
                    title="Temperature (°C)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=330)
        .interactive()
    )