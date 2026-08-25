from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_nasa_battery(file_path, battery_name):
    """
    Load a NASA battery .mat file and extract
    cycle-level discharge information.

    Parameters
    ----------
    file_path : str or Path
        Path to the NASA .mat file.
    battery_name : str
        Battery identifier, e.g. "B0005".

    Returns
    -------
    pandas.DataFrame
        One row per discharge cycle.
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

    # NASA MATLAB files can store cycles as (1, N) or (N, 1)
    if cycles.shape[0] == 1:
        cycle_indices = [(0, i) for i in range(cycles.shape[1])]
    else:
        cycle_indices = [(i, 0) for i in range(cycles.shape[0])]

    for cycle_number, (row, col) in enumerate(cycle_indices, start=1):

        cycle = cycles[row, col]

        cycle_type = str(cycle["type"][0])
        

        print(cycle_number, repr(cycle_type))

        if cycle_type != "discharge":
            continue

        data = cycle["data"][0, 0]

        voltage = data["Voltage_measured"].flatten()
        current = data["Current_measured"].flatten()
        temperature = data["Temperature_measured"].flatten()
        time = data["Time"].flatten()
        capacity = data["Capacity"].flatten()[0]

        records.append({
            "cycle": cycle_number,
            "capacity_Ah": float(capacity),
            "avg_voltage_V": float(np.mean(voltage)),
            "min_voltage_V": float(np.min(voltage)),
            "max_voltage_V": float(np.max(voltage)),
            "avg_current_A": float(np.mean(current)),
            "avg_temperature_C": float(np.mean(temperature)),
            "max_temperature_C": float(np.max(temperature)),
            "discharge_time_s": float(np.max(time))
        })

    return pd.DataFrame(records)