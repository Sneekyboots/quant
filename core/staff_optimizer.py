"""
core/staff_optimizer.py
Stage 2 QUBO: Staff → Patient assignment (conditioned on Stage 1 bed allocation).

QUBO Hamiltonian
────────────────
  H = -∑_{n,p} (skill_n · urgency_p · (1 - fatigue_n)) · w_role_n · s_{n,p}
                                                           ↑ maximize skill-acuity match

      + α_s · ∑_p (∑_n s_{n,p} - 1)²               ← each patient gets ≥1 staff

      + β_s · ∑_n (∑_p s_{n,p} - C_n)²              ← staff capacity limits

      + γ   · ∑_{p,n} (1 - qual_{n,ward_p}) · s_{n,p} ← qualification penalty
                                                          (coupling to Stage 1 output)

Variable count:  N_staff × N_patients  (default 11 × 8 = 88 variables)
"""

import numpy as np
import neal
import pandas as pd

from data.generator import STAFF_QUALIFICATIONS, STAFF_CAPACITY, STAFF_WEIGHTS, STAFF_ROLES

BETA_S = 10.0   # capacity penalty weight
GAMMA  = 50.0   # qualification mismatch penalty (must dominate utility to block mismatches)


def _svar(n: int, p: int) -> str:
    return f"s{n}_p{p}"


def build_staff_qubo(
    staff_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    quantum_allocation: list[dict],
    urgency_scores: np.ndarray,
    beta_s: float = BETA_S,
    gamma: float = GAMMA,
) -> tuple[dict, float]:
    """
    Assemble the Stage 2 QUBO dictionary.

    Parameters
    ----------
    staff_df          : DataFrame from generate_staff()
    patient_df        : DataFrame from generate_patients()
    quantum_allocation: Stage 1 result — patient_idx → resource_idx
    urgency_scores    : QSVM urgency scores per patient (length = len(patient_df))

    Returns
    -------
    (Q_dict, alpha_s)  — same shape contract as optimizer.build_qubo()
    """
    N = len(staff_df)
    P = len(patient_df)
    Q: dict = {}

    # Patient → ward mapping from Stage 1
    ward_lookup = {a["patient_idx"]: a["resource_idx"] for a in quantum_allocation}

    # ── Derive safe α_s ──────────────────────────────────────────────────
    max_utility  = float(np.max(urgency_scores)) * max(STAFF_WEIGHTS.values())
    max_cap_diag = beta_s * (2 * max(STAFF_CAPACITY.values()) - 1)
    alpha_s      = (max_utility + max_cap_diag) * 1.5

    # ── Term 1: Maximise skill-acuity match ──────────────────────────────
    # Minimise -utility  →  diagonal reward
    for n_idx in range(N):
        staff_row = staff_df.iloc[n_idx]
        role_id   = int(staff_row["role_id"])
        skill     = float(staff_row["skill_level"])
        fatigue   = float(staff_row["fatigue_score"])
        w_role    = STAFF_WEIGHTS.get(role_id, 1.0)

        for p_idx in range(P):
            urgency = float(urgency_scores[p_idx])
            utility = skill * urgency * (1.0 - fatigue) * w_role
            v = _svar(n_idx, p_idx)
            Q[(v, v)] = Q.get((v, v), 0.0) - utility

    # ── Term 2: Each patient assigned to ≥1 staff member ─────────────────
    # Full expansion of α_s · (∑_n s_{n,p} - 1)²
    for p_idx in range(P):
        for n_idx in range(N):
            v = _svar(n_idx, p_idx)
            Q[(v, v)] = Q.get((v, v), 0.0) - alpha_s        # α·(1-2) = -α

        for n1 in range(N):
            for n2 in range(n1 + 1, N):
                v1, v2 = _svar(n1, p_idx), _svar(n2, p_idx)
                key = (v1, v2) if v1 < v2 else (v2, v1)
                Q[key] = Q.get(key, 0.0) + 2.0 * alpha_s

    # ── Term 3: Staff capacity constraints ───────────────────────────────
    # Full expansion of β_s · (∑_p s_{n,p} - C_n)²
    for n_idx in range(N):
        staff_row = staff_df.iloc[n_idx]
        role_id   = int(staff_row["role_id"])
        C_n       = STAFF_CAPACITY.get(role_id, 2)

        for p1 in range(P):
            v1 = _svar(n_idx, p1)
            Q[(v1, v1)] = Q.get((v1, v1), 0.0) + beta_s * (1 - 2 * C_n)

            for p2 in range(p1 + 1, P):
                v2  = _svar(n_idx, p2)
                key = (v1, v2) if v1 < v2 else (v2, v1)
                Q[key] = Q.get(key, 0.0) + 2.0 * beta_s

    # ── Term 4: Qualification penalty (Stage 1 coupling) ─────────────────
    # γ · (1 - qual_{n, ward_p}) · s_{n,p}
    # This is a diagonal penalty — no cross-terms needed.
    for n_idx in range(N):
        staff_row       = staff_df.iloc[n_idx]
        role_id         = int(staff_row["role_id"])
        qualified_wards = set(STAFF_QUALIFICATIONS.get(role_id, []))

        for p_idx in range(P):
            ward    = ward_lookup.get(p_idx, 2)   # default General Ward
            penalty = gamma * (0 if ward in qualified_wards else 1)
            if penalty > 0:
                v = _svar(n_idx, p_idx)
                Q[(v, v)] = Q.get((v, v), 0.0) + penalty

    return Q, alpha_s


def solve_staff_qubo(Q_or_tuple, num_reads: int = 200) -> dict:
    """
    Simulated Annealing solver for the Stage 2 staff QUBO.
    QPU-portable: same interface as optimizer.solve_qubo().
    """
    Q = Q_or_tuple[0] if isinstance(Q_or_tuple, tuple) else Q_or_tuple
    sampler = neal.SimulatedAnnealingSampler()
    result  = sampler.sample_qubo(Q, num_reads=num_reads)
    return result.first.sample


def parse_staff_allocation(
    sample: dict,
    staff_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    quantum_allocation: list[dict],
    urgency_scores: np.ndarray,
) -> list[dict]:
    """Convert flat binary sample into structured staff-assignment rows."""
    ward_lookup = {a["patient_idx"]: a["resource_name"] for a in quantum_allocation}

    rows = []
    for var, active in sample.items():
        if active != 1 or not var.startswith("s"):
            continue
        parts = var.split("_p")
        n_idx = int(parts[0][1:])
        p_idx = int(parts[1])

        if n_idx >= len(staff_df) or p_idx >= len(patient_df):
            continue

        staff_row = staff_df.iloc[n_idx]
        rows.append({
            "staff_idx":      n_idx,
            "patient_idx":    p_idx,
            "staff_id":       staff_row["staff_id"],
            "role_name":      staff_row["role_name"],
            "role_id":        int(staff_row["role_id"]),
            "skill_level":    round(float(staff_row["skill_level"]), 3),
            "fatigue_score":  round(float(staff_row["fatigue_score"]), 3),
            "patient_id":     patient_df.iloc[p_idx]["patient_id"],
            "ward":           ward_lookup.get(p_idx, "Unallocated"),
            "urgency":        round(float(urgency_scores[p_idx]), 4),
        })

    return sorted(rows, key=lambda x: x["urgency"], reverse=True)


def compute_staff_metrics(
    staff_allocation: list[dict],
    staff_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    quantum_allocation: list[dict],
    urgency_scores: np.ndarray,
) -> dict:
    """
    Compute the 5 pitch metrics.

    Returns
    -------
    {
      utilization_pct     : dict[role_name → float]  (0–100 %)
      skill_acuity_match  : float                     (0–1)
      unassigned_count    : int
      unassigned_rate     : float                     (0–100 %)
      cross_qual_rate     : float                     (0–100 %)
    }
    """
    n_patients = len(patient_df)

    # ── Staff utilisation % per role ─────────────────────────────────────
    role_patient_count: dict[str, int] = {}
    for a in staff_allocation:
        role = a["role_name"]
        role_patient_count[role] = role_patient_count.get(role, 0) + 1

    role_capacity_total: dict[str, int] = {}
    for _, row in staff_df.iterrows():
        role = row["role_name"]
        cap  = STAFF_CAPACITY.get(int(row["role_id"]), 2)
        role_capacity_total[role] = role_capacity_total.get(role, 0) + cap

    utilization_pct = {
        role: round(role_patient_count.get(role, 0) / max(role_capacity_total.get(role, 1), 1) * 100, 1)
        for role in role_capacity_total
    }

    # ── Skill-acuity match score ─────────────────────────────────────────
    match_scores = [a["skill_level"] * a["urgency"] for a in staff_allocation]
    skill_acuity_match = round(float(np.mean(match_scores)) if match_scores else 0.0, 4)

    # ── Unassigned patients ───────────────────────────────────────────────
    covered    = {a["patient_idx"] for a in staff_allocation}
    unassigned = n_patients - len(covered)
    unassigned_rate = round(unassigned / max(n_patients, 1) * 100, 1)

    # ── Cross-qualification rate ──────────────────────────────────────────
    ward_lookup = {a["patient_idx"]: a["resource_idx"] for a in quantum_allocation}
    cross_qual = sum(
        1 for a in staff_allocation
        if ward_lookup.get(a["patient_idx"], 2) not in set(STAFF_QUALIFICATIONS.get(a["role_id"], []))
    )
    cross_qual_rate = round(cross_qual / max(len(staff_allocation), 1) * 100, 1)

    return {
        "utilization_pct":    utilization_pct,
        "skill_acuity_match": skill_acuity_match,
        "unassigned_count":   unassigned,
        "unassigned_rate":    unassigned_rate,
        "cross_qual_rate":    cross_qual_rate,
    }
