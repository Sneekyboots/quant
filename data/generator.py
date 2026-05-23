"""
data/generator.py
Generates synthetic patient vitals + AQI data for demo.
Replace with real UCI Heart Disease CSV for benchmarking.
"""

import numpy as np
import pandas as pd

# Feature columns match quantum circuit input (4 qubits = 4 features)
FEATURE_COLS = ["bp_deviation", "spo2_deficit", "temperature_delta", "aqi_pm25"]

RESOURCE_NAMES = {
    0: "ICU / Trauma",
    1: "Ventilator Unit",
    2: "General Ward",
}

RESOURCE_WEIGHTS = [2.5, 2.0, 0.7]   # clinical utility per resource tier
RESOURCE_CAPACITY = [2, 2, 4]         # max patients per resource


def generate_patients(n: int = 8, aqi_level: float = 50.0, seed: int = 42) -> pd.DataFrame:
    """
    Simulate n emergency intake patients.
    aqi_level: 0–500 PM2.5 scale. Higher = more respiratory cases.
    """
    rng = np.random.default_rng(seed)

    # AQI influence: high pollution shifts SpO2 deficit and BP upward
    aqi_factor = np.clip(aqi_level / 500.0, 0, 1)

    bp    = rng.uniform(0.0, 1.0, n) + 0.3 * aqi_factor
    spo2  = rng.uniform(0.0, 1.0, n) + 0.4 * aqi_factor
    temp  = rng.uniform(0.0, 1.0, n)
    aqi   = np.full(n, aqi_factor) + rng.normal(0, 0.05, n)

    # Clip all to [0, 1] before angle embedding
    data = np.clip(np.stack([bp, spo2, temp, aqi], axis=1), 0, 1)

    labels = (bp + spo2 > 1.2).astype(int)   # simple proxy ground truth

    df = pd.DataFrame(data, columns=FEATURE_COLS)
    df["label"] = labels
    df["patient_id"] = [f"P{i+1:02d}" for i in range(n)]
    return df
