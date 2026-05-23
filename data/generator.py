"""
data/generator.py
Generates synthetic patient vitals + AQI data for demo.
Replace with real UCI Heart Disease CSV for benchmarking.
"""

import numpy as np
import pandas as pd

# Feature columns — 8 features → 8-qubit QSVM kernel
FEATURE_COLS = [
    "bp_deviation",      # blood-pressure deviation from baseline
    "spo2_deficit",      # SpO2 deficit (oxygen saturation drop)
    "temperature_delta", # fever / hypothermia delta
    "aqi_pm25",          # local PM2.5 per-patient reading
    "hr_deviation",      # heart-rate deviation (tachycardia / bradycardia)
    "resp_rate",         # respiratory-rate deviation (tachypnoea)
    "gcs_deficit",       # (15 - GCS) / 12  — 0 = alert, 1 = deep coma
    "lactate",           # serum lactate normalised; >0.5 ≈ >4 mmol/L (shock)
]

RESOURCE_NAMES = {
    0: "ICU / Trauma",
    1: "Ventilator Unit",
    2: "General Ward",
}

RESOURCE_WEIGHTS  = [2.5, 2.0, 0.7]   # clinical utility per resource tier
RESOURCE_CAPACITY = [2, 2, 5]   # realistic hospital capacity: ICU / Vent / General

# ── Staff model constants ─────────────────────────────────────────────────────

STAFF_ROLES = {
    0:  "Emergency Physician",
    1:  "Intensivist (ICU)",
    2:  "Hospitalist",
    3:  "Pulmonologist",
    4:  "Trauma Surgeon",
    5:  "Critical Care Nurse",
    6:  "Step-Down Nurse",
    7:  "ED Nurse",
    8:  "Medical Nurse",
    9:  "Respiratory Therapist",
    10: "Patient Care Tech",
}

# Which wards (resource_idx) each role is qualified to cover
STAFF_QUALIFICATIONS = {
    0:  [0, 2],     # Emergency Physician    → ICU + General
    1:  [0],        # Intensivist            → ICU only
    2:  [0, 2],     # Hospitalist            → ICU support + General
    3:  [0, 1],     # Pulmonologist          → ICU + Vent
    4:  [0],        # Trauma Surgeon         → ICU only
    5:  [0],        # Critical Care Nurse    → ICU only
    6:  [0, 1],     # Step-Down Nurse        → Vent + ICU overflow
    7:  [0, 2],     # ED Nurse               → ICU + General
    8:  [1, 2],     # Medical Nurse          → Vent + General
    9:  [0, 1],     # Respiratory Therapist  → ICU + Vent
    10: [2],        # Patient Care Tech      → General only
}

# Max patients per staff member (evidence-based nurse-patient ratios)
STAFF_CAPACITY = {
    0:  4,   # Emergency Physician   (1:4 ED standard)
    1:  3,   # Intensivist           (1:3 ICU physician)
    2:  6,   # Hospitalist           (1:6 general medicine)
    3:  4,   # Pulmonologist         (1:4 respiratory)
    4:  3,   # Trauma Surgeon        (1:3 trauma critical)
    5:  2,   # Critical Care Nurse   (1:2 ICU mandate)
    6:  3,   # Step-Down Nurse       (1:3 step-down/PCU)
    7:  3,   # ED Nurse              (1:3 emergency)
    8:  5,   # Medical Nurse         (1:5 general ward)
    9:  4,   # Respiratory Therapist (1:4 vent/airway)
    10: 6,   # Patient Care Tech     (1:6 support role)
}

# Clinical utility weight per role (relative impact per patient assignment)
STAFF_WEIGHTS = {
    0:  3.5,  # Emergency Physician   — high-acuity decision maker
    1:  4.0,  # Intensivist           — highest ICU impact
    2:  2.5,  # Hospitalist           — general management
    3:  3.5,  # Pulmonologist         — respiratory expert
    4:  4.0,  # Trauma Surgeon        — life-saving intervention
    5:  2.5,  # Critical Care Nurse   — ICU bedside
    6:  2.0,  # Step-Down Nurse       — intermediate care
    7:  2.5,  # ED Nurse              — emergency triage
    8:  1.5,  # Medical Nurse         — routine monitoring
    9:  3.0,  # Respiratory Therapist — ventilator management
    10: 1.0,  # Patient Care Tech     — ADL support
}

# Skill baseline per role (mean of normal distribution for skill_level)
_SKILL_MEAN = {
    0:  0.85,  # Emergency Physician
    1:  0.90,  # Intensivist
    2:  0.75,  # Hospitalist
    3:  0.85,  # Pulmonologist
    4:  0.90,  # Trauma Surgeon
    5:  0.80,  # Critical Care Nurse
    6:  0.70,  # Step-Down Nurse
    7:  0.75,  # ED Nurse
    8:  0.65,  # Medical Nurse
    9:  0.80,  # Respiratory Therapist
    10: 0.50,  # Patient Care Tech
}


def generate_patients(n: int = 100, aqi_level: float = 50.0, seed: int = 42) -> pd.DataFrame:
    """
    Simulate n emergency intake patients with 8 clinical features.
    aqi_level: 0–500 PM2.5 scale. Higher = more respiratory cases.

    Critical feature rationale
    ──────────────────────────
    hr_deviation  — AQI stress → reflex tachycardia (HR > 100 bpm)
    resp_rate     — AQI → tachypnoea (RR > 20); first sign of respiratory failure
    gcs_deficit   — (15 − GCS) / 12; right-skewed toward 0 (most patients conscious);
                    any value > 0.33 (GCS < 11) flags altered consciousness
    lactate       — serum lactate proxy; exponential dist (most patients normal);
                    > 0.50 (≈ > 4 mmol/L) indicates tissue hypoperfusion / shock
    """
    rng = np.random.default_rng(seed)

    aqi_factor = np.clip(aqi_level / 500.0, 0, 1)

    # Existing haemodynamic / respiratory features
    bp      = rng.uniform(0.0, 1.0, n) + 0.30 * aqi_factor
    spo2    = rng.uniform(0.0, 1.0, n) + 0.40 * aqi_factor
    temp    = rng.uniform(0.0, 1.0, n)
    aqi     = np.full(n, aqi_factor) + rng.normal(0, 0.05, n)

    # Critical clinical features
    hr      = rng.uniform(0.0, 0.8, n) + 0.25 * aqi_factor   # reflex tachycardia
    resp    = rng.uniform(0.0, 0.8, n) + 0.35 * aqi_factor   # tachypnoea
    gcs     = rng.beta(1.5, 5.0, n)                           # mostly alert; right-skewed
    lactate = np.clip(rng.exponential(0.25, n), 0.0, 1.0)     # shock marker; right-skewed

    # Clip all to [0, 1] before angle embedding
    data = np.clip(
        np.stack([bp, spo2, temp, aqi, hr, resp, gcs, lactate], axis=1), 0, 1
    )

    # Richer label: classic haemodynamic distress OR impaired consciousness OR shock
    labels = (
        (bp + spo2 > 1.2)    # respiratory / haemodynamic crisis
        | (gcs > 0.33)        # GCS < 11 — altered consciousness
        | (lactate > 0.50)    # lactate > 4 mmol/L — tissue hypoperfusion
    ).astype(int)

    df = pd.DataFrame(data, columns=FEATURE_COLS)
    df["label"] = labels
    df["patient_id"] = [f"P{i+1:02d}" for i in range(n)]
    return df


def generate_staff(
    n_emergency_physician: int = 3,
    n_intensivist: int = 4,
    n_hospitalist: int = 5,
    n_pulmonologist: int = 3,
    n_trauma_surgeon: int = 2,
    n_cc_nurse: int = 10,
    n_stepdown_nurse: int = 7,
    n_ed_nurse: int = 7,
    n_medical_nurse: int = 8,
    n_resp_therapist: int = 6,
    n_pct: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic shift roster of 60 hospital staff across 11 clinical roles.

    Default roster (60 staff, calibrated to 100-patient triage batch):
      3 Emergency Physicians, 4 Intensivists, 5 Hospitalists, 3 Pulmonologists,
      2 Trauma Surgeons, 10 Critical Care Nurses, 7 Step-Down Nurses, 7 ED Nurses,
      8 Medical Nurses, 6 Respiratory Therapists, 5 Patient Care Techs

    Each member receives:
      skill_level      ∈ [0, 1]  — role-calibrated experience level
      shift_remaining  ∈ [2, 8]  — hours left in current shift
      fatigue_score    = 1 - shift_remaining / 8
    """
    rng = np.random.default_rng(seed)

    role_counts = [
        n_emergency_physician, n_intensivist, n_hospitalist, n_pulmonologist,
        n_trauma_surgeon, n_cc_nurse, n_stepdown_nurse, n_ed_nurse,
        n_medical_nurse, n_resp_therapist, n_pct,
    ]
    roles: list[int] = []
    for role_id, count in enumerate(role_counts):
        roles.extend([role_id] * count)

    n = len(roles)
    # Role-specific skill distributions (more experienced roles have higher baseline)
    skill_level = np.clip(
        np.array([rng.normal(_SKILL_MEAN[r], 0.10) for r in roles]),
        0.1, 1.0,
    )
    shift_remaining = rng.uniform(2.0, 8.0, n)
    fatigue_score   = 1.0 - shift_remaining / 8.0

    return pd.DataFrame({
        "staff_id":        [f"S{i+1:02d}" for i in range(n)],
        "role_id":         roles,
        "role_name":       [STAFF_ROLES[r] for r in roles],
        "skill_level":     np.round(skill_level, 3),
        "shift_remaining": np.round(shift_remaining, 1),
        "fatigue_score":   np.round(fatigue_score, 3),
    })
