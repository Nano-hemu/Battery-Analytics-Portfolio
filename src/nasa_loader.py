from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_nasa_battery(file_path, battery_name):
    """
    Load a NASA battery .mat file and extract discharge-level features.

    Parameters
    ----------
    file_path : str or Path
        Path to the NASA .mat file.

    battery_name : str
        Battery identifier, for example "B0005".

    Returns
    -------
    pandas.DataFrame
        One row per discharge observation.

        Key indexing columns:
        - record_index:
            Original position of the discharge event within the complete
            NASA experimental event sequence, which may also contain
            charge and impedance measurements.

        - discharge_cycle:
            Sequential discharge count: 1, 2, 3, ...

        - cycle:
            Alias of discharge_cycle retained for compatibility with
            existing analysis code.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Battery file not found: {file_path.resolve()}"
        )

    mat_data = loadmat(file_path)

    if battery_name not in mat_data:
        raise KeyError(
            f"Battery '{battery_name}' not found in {file_path.name}"
        )

    battery = mat_data[battery_name][0, 0]
    cycles = battery["cycle"]

    records = []

    # NASA MATLAB files may store the cycle array as (1, N) or (N, 1)
    if cycles.shape[0] == 1:
        cycle_indices = [(0, i) for i in range(cycles.shape[1])]
    else:
        cycle_indices = [(i, 0) for i in range(cycles.shape[0])]

    discharge_cycle = 0

    for record_index, (row, col) in enumerate(cycle_indices, start=1):

        cycle = cycles[row, col]
        cycle_type = str(cycle["type"][0]).strip()

        if cycle_type != "discharge":
            continue

        discharge_cycle += 1

        data = cycle["data"][0, 0]

        voltage = np.asarray(
            data["Voltage_measured"]
        ).flatten()

        current = np.asarray(
            data["Current_measured"]
        ).flatten()

        temperature = np.asarray(
            data["Temperature_measured"]
        ).flatten()

        time = np.asarray(
            data["Time"]
        ).flatten()

        capacity = float(
            np.asarray(data["Capacity"]).flatten()[0]
        )

        records.append({
            "record_index": record_index,
            "discharge_cycle": discharge_cycle,

            # Compatibility alias.
            # Existing notebook cells using "cycle" will now operate
            # on the true sequential discharge count.
            "cycle": discharge_cycle,

            "capacity_Ah": capacity,
            "avg_voltage_V": float(np.mean(voltage)),
            "min_voltage_V": float(np.min(voltage)),
            "max_voltage_V": float(np.max(voltage)),
            "avg_current_A": float(np.mean(current)),
            "avg_temperature_C": float(np.mean(temperature)),
            "max_temperature_C": float(np.max(temperature)),
            "discharge_time_s": float(np.max(time)),
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No discharge records found for battery '{battery_name}'."
        )

    return df