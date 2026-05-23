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
RESOURCE_CAPACITY = [4, 4, 8]         # max patients per resource (16 total)

# ── Staff model constants ─────────────────────────────────────────────────────

STAFF_ROLES = {
    0: "ICU Physician",
    1: "ICU Nurse",
    2: "Vent Specialist",
    3: "General Nurse",
}

# Which wards (resource_idx) each role is qualified to cover
STAFF_QUALIFICATIONS = {
    0: [0],        # ICU Physician → ICU only
    1: [0],        # ICU Nurse → ICU only
    2: [1],        # Vent Specialist → Ventilator only
    3: [0, 1, 2],  # General Nurse → any ward (float staff)
}

# Max patients per staff member (nurse-patient ratio)
STAFF_CAPACITY = {
    0: 3,   # physician : up to 3 patients
    1: 2,   # ICU nurse : up to 2 patients
    2: 2,   # vent specialist : up to 2 patients
    3: 4,   # general nurse : up to 4 patients
}

# Clinical utility weight per role (higher = more impact per assignment)
STAFF_WEIGHTS = {
    0: 3.0,   # physician has highest impact
    1: 2.0,
    2: 2.5,
    3: 1.0,
}


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


def generate_staff(
    n_icu_physician: int = 2,
    n_icu_nurse: int = 3,
    n_vent_specialist: int = 2,
    n_general_nurse: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic shift roster of hospital staff.

    Each member receives:
      skill_level      ∈ [0, 1]  — years of experience, normalised
      shift_remaining  ∈ [2, 8]  — hours left in the current shift
      fatigue_score    = 1 - shift_remaining / 8  (higher = more fatigued)
    """
    rng = np.random.default_rng(seed)

    role_counts = [n_icu_physician, n_icu_nurse, n_vent_specialist, n_general_nurse]
    roles: list[int] = []
    for role_id, count in enumerate(role_counts):
        roles.extend([role_id] * count)

    n = len(roles)
    skill_level     = np.clip(rng.normal(0.7, 0.15, n), 0.1, 1.0)
    shift_remaining = rng.uniform(2.0, 8.0, n)
    fatigue_score   = 1.0 - shift_remaining / 8.0

    return pd.DataFrame({
        "staff_id":       [f"S{i+1:02d}" for i in range(n)],
        "role_id":        roles,
        "role_name":      [STAFF_ROLES[r] for r in roles],
        "skill_level":    np.round(skill_level, 3),
        "shift_remaining": np.round(shift_remaining, 1),
        "fatigue_score":  np.round(fatigue_score, 3),
    })
