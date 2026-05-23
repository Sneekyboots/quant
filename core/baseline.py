"""
core/baseline.py
Classical greedy triage baseline for benchmarking.
Sort patients by raw feature score, assign greedily by capacity.
Used to generate the "Before Quantum" comparison in the dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as _f1
from sklearn.model_selection import train_test_split
from data.generator import RESOURCE_NAMES, RESOURCE_CAPACITY


def classical_greedy(df: pd.DataFrame) -> list[dict]:
    """
    Greedy baseline: rank by (bp_deviation + spo2_deficit), 
    fill ICU first, then Ventilator, then General Ward.
    No optimization — first-come-first-served within each tier.
    """
    df = df.copy()
    df["classical_score"] = df["bp_deviation"] + df["spo2_deficit"]
    df = df.sort_values("classical_score", ascending=False).reset_index(drop=True)

    capacity = list(RESOURCE_CAPACITY)
    allocation = []

    for _, row in df.iterrows():
        assigned = False
        for r in range(len(capacity)):
            if capacity[r] > 0:
                allocation.append({
                    "patient_idx": row.name,
                    "resource_idx": r,
                    "resource_name": RESOURCE_NAMES.get(r, f"Resource {r}"),
                    "urgency": round(row["classical_score"] / 2.0, 4),
                })
                capacity[r] -= 1
                assigned = True
                break
        if not assigned:
            allocation.append({
                "patient_idx": row.name,
                "resource_idx": -1,
                "resource_name": "⚠️ Unallocated",
                "urgency": round(row["classical_score"] / 2.0, 4),
            })

    return allocation


def compute_utilization(allocation: list[dict]) -> dict:
    """Returns utilization % per resource."""
    counts = {}
    for row in allocation:
        r = row["resource_name"]
        counts[r] = counts.get(r, 0) + 1
    total = sum(counts.values())
    return {k: round(v / total * 100, 1) for k, v in counts.items()}


def random_forest_scores(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Train a Random Forest on all (X, y) for urgency probabilities.
    F1 is measured on a stratified 25% held-out split so the score is
    honest (not inflated by train-set self-prediction).
    Citation: Breiman, Machine Learning 45, 5-32 (2001)
    """
    # ── Held-out F1 ──────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_eval.fit(X_tr, y_tr)
    f1 = float(_f1(y_te, rf_eval.predict(X_te), average="macro", zero_division=0))

    # ── Full-data urgency proba for bed allocation ────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    urgency_proba = rf.predict_proba(X)[:, 1]
    return urgency_proba, f1
