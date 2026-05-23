"""
core/optimizer.py
Builds QUBO from urgency scores and solves via Simulated Annealing.

QUBO Hamiltonian:
  H = -∑ P_i · w_r · x_{i,r}           ← maximize clinical utility
    + α · ∑_i (∑_r x_{i,r} - 1)²       ← one resource per patient
    + β · ∑_r (∑_i x_{i,r} - C_r)²     ← capacity constraints

Citation: Glover et al., arXiv:1811.11538
QAOA reference: Farhi et al., arXiv:1411.4028 (2014)
"""

import numpy as np
import neal
from data.generator import RESOURCE_WEIGHTS, RESOURCE_CAPACITY, RESOURCE_NAMES


BETA = 15.0   # capacity penalty — α is computed dynamically, never hardcoded


def _var(p: int, r: int) -> str:
    return f"p{p}_r{r}"


def compute_min_alpha(urgency_scores: np.ndarray,
                      resource_weights: list,
                      resource_capacity: list,
                      beta: float,
                      margin: float = 1.5) -> float:
    """
    Derives the minimum α that guarantees uniqueness constraints dominate.

    Root cause of the bug: the capacity diagonal β(1-2C_r) is a large
    negative number when C_r > 0.5. This rewards double-assignment,
    making the optimizer prefer assigning one patient to two resources.

    For uniqueness to hold, assigning patient i to a single resource r*
    must always be cheaper than assigning them to r* AND any other r':

        d_{i,r*} < d_{i,r*} + d_{i,r'} + 2α
        ⟹  0 < d_{i,r'} + 2α
        ⟹  α  > (P_i·w_r + α - β(1-2C_r)) / 2   [substituting d_{i,r'} expansion]

    Solving for α:
        α > max(P_i) · max(w_r) + β·(2·max(C_r) - 1)

    We add a multiplicative margin (default 1.5x) for numerical safety.
    """
    max_utility    = float(np.max(urgency_scores)) * max(resource_weights)
    max_cap_diag   = beta * (2 * max(resource_capacity) - 1)   # worst-case β(2C-1)
    alpha_min      = (max_utility + max_cap_diag) * margin
    return alpha_min


def build_qubo(urgency_scores: np.ndarray,
               resource_weights: list = RESOURCE_WEIGHTS,
               resource_capacity: list = RESOURCE_CAPACITY,
               beta: float = BETA) -> dict:
    """
    Assembles the QUBO dictionary from live urgency scores.
    α is computed dynamically — never hardcoded — to guarantee
    the uniqueness constraint always dominates capacity incentives.
    Returns (Q_dict, alpha) so the pipeline can log the computed α.
    """
    M = len(urgency_scores)
    R = len(resource_weights)
    Q = {}

    # ── Clamp capacity to n_patients ──────────────────────────────────────
    # β·(1−2·C_r) becomes a huge negative diagonal when C_r is large (e.g.
    # 1500 for General Ward), overwhelming the utility objective and pulling
    # every patient into the largest-capacity ward.  Clamping to min(C_r, M)
    # keeps the capacity term correctly scaled for the current batch size.
    eff_cap = [min(c, M) for c in resource_capacity]

    # ── Derive safe α from actual problem parameters ──────────────────────
    alpha = compute_min_alpha(urgency_scores, resource_weights, eff_cap, beta)

    # ── Term 1: Clinical utility (maximize → minimize negative) ──────────
    for i in range(M):
        for r in range(R):
            v = _var(i, r)
            Q[(v, v)] = Q.get((v, v), 0.0) - urgency_scores[i] * resource_weights[r]

    # ── Term 2: Uniqueness — each patient to exactly 1 resource ──────────
    # Full expansion of α·(∑_r x_{i,r} - 1)²:
    #   diagonal: α·(1-2) = -α  per variable
    #   off-diag: +2α           per (r1, r2) pair
    for i in range(M):
        for r in range(R):
            v = _var(i, r)
            Q[(v, v)] = Q.get((v, v), 0.0) - alpha          # α*(1-2) = -α
        for r1 in range(R):
            for r2 in range(r1 + 1, R):
                v1, v2 = _var(i, r1), _var(i, r2)
                key = (v1, v2) if v1 < v2 else (v2, v1)
                Q[key] = Q.get(key, 0.0) + 2 * alpha

    # ── Term 3: Capacity — soft penalty for exceeding resource limit ──────
    # Use clamped effective capacity (eff_cap) so the diagonal term
    # β(1-2·C) stays proportionate to n_patients rather than ward total.
    for r in range(R):
        C = eff_cap[r]
        for i1 in range(M):
            v1 = _var(i1, r)
            Q[(v1, v1)] = Q.get((v1, v1), 0.0) + beta * (1 - 2 * C)
            for i2 in range(i1 + 1, M):
                v2 = _var(i2, r)
                key = (v1, v2) if v1 < v2 else (v2, v1)
                Q[key] = Q.get(key, 0.0) + 2 * beta

    return Q, alpha   # return alpha so pipeline can log it


def solve_qubo(Q_or_tuple, num_reads: int = 200) -> dict:
    """
    Runs Simulated Annealing on the QUBO.
    Returns the best binary assignment found.
    Directly maps to QAOA Ising Hamiltonian — QPU-portable.
    Accepts either a raw dict or a (dict, alpha) tuple from build_qubo.
    """
    Q = Q_or_tuple[0] if isinstance(Q_or_tuple, tuple) else Q_or_tuple
    sampler = neal.SimulatedAnnealingSampler()
    result = sampler.sample_qubo(Q, num_reads=num_reads)
    return result.first.sample


def parse_allocation(sample: dict, urgency_scores: np.ndarray,
                     resource_capacity: list = RESOURCE_CAPACITY) -> list[dict]:
    """
    Converts flat binary sample back into human-readable allocation rows.

    Two hard constraints enforced here (SA may violate the soft QUBO penalties):
      1. One resource per patient — if SA assigned the same patient to multiple
         wards, keep only the highest-urgency resource match.
      2. Ward capacity — if a ward is over-assigned, keep only the top-urgency
         patients up to RESOURCE_CAPACITY[r].  Excess patients fall through to
         fill_unallocated() for greedy reassignment.
    """
    # Collect all active assignments sorted highest-urgency first
    raw: list[dict] = []
    for var, active in sample.items():
        if active == 1:
            p = int(var.split("_")[0][1:])
            r = int(var.split("_")[1][1:])
            raw.append({
                "patient_idx":  p,
                "resource_idx": r,
                "resource_name": RESOURCE_NAMES.get(r, f"Resource {r}"),
                "urgency":      round(float(urgency_scores[p]), 4),
                "fallback":     False,
            })
    raw.sort(key=lambda x: x["urgency"], reverse=True)

    # ── 1. Deduplicate: one assignment per patient (highest urgency ward wins)
    seen_patients: set[int] = set()
    deduped: list[dict] = []
    for row in raw:
        if row["patient_idx"] not in seen_patients:
            deduped.append(row)
            seen_patients.add(row["patient_idx"])

    # ── 2. Hard capacity cap per ward
    ward_counts: dict[int, int] = {}
    allocation: list[dict] = []
    for row in deduped:
        r   = row["resource_idx"]
        cap = resource_capacity[r] if r < len(resource_capacity) else 0
        if ward_counts.get(r, 0) < cap:
            allocation.append(row)
            ward_counts[r] = ward_counts.get(r, 0) + 1
        # else: over-capacity — fill_unallocated() will reassign this patient

    return allocation


def fill_unallocated(
    allocation: list[dict],
    n_patients: int,
    urgency_scores: np.ndarray,
    resource_capacity: list = RESOURCE_CAPACITY,
) -> tuple[list[dict], list[dict]]:
    """
    Greedy post-processing with triage preemption.

    Phase 1 — Free-bed assignment:
      Unallocated patients (highest urgency first) fill any spare beds in
      their preferred ward order.

    Phase 2 — Preemption:
      If ALL beds are full but the incoming patient has strictly higher
      urgency than any current occupant, they BUMP the lowest-urgency
      occupant.  Bumped patients go onto the waitlist — a higher-acuity
      patient needs the bed.

    Returns: (allocated_list, unallocated_records)
    """
    allocated   = {a["patient_idx"] for a in allocation}
    ward_counts = {r: sum(1 for a in allocation if a["resource_idx"] == r)
                   for r in range(len(resource_capacity))}

    # Clamp ward capacity to batch size
    effective_capacity = {r: min(resource_capacity[r], n_patients)
                          for r in range(len(resource_capacity))}

    unallocated_list = sorted(
        [p for p in range(n_patients) if p not in allocated],
        key=lambda p: urgency_scores[p],
        reverse=True,
    )

    # Waitlist entries: {patient_idx, urgency, displaced, displaced_by_urgency}
    waitlist_entries: list[dict] = []

    for p in unallocated_list:
        u = float(urgency_scores[p])
        if u >= 0.70:
            preference = [0, 1, 2]
        elif u >= 0.40:
            preference = [1, 0, 2]
        else:
            preference = [2, 1, 0]

        # ── Phase 1: fill a free bed ──────────────────────────────────────
        assigned = False
        for r in preference:
            if (r < len(effective_capacity) and
                    ward_counts.get(r, 0) < effective_capacity[r]):
                allocation.append({
                    "patient_idx":   p,
                    "resource_idx":  r,
                    "resource_name": RESOURCE_NAMES.get(r, f"Resource {r}"),
                    "urgency":       round(u, 4),
                    "fallback":      True,
                })
                ward_counts[r] = ward_counts.get(r, 0) + 1
                assigned = True
                break

        # ── Phase 2: preemption — bump the lowest-urgency occupant ───────
        if not assigned:
            # All preferred wards are full; find the lowest-urgency occupant
            # across those wards whose urgency is strictly below ours
            candidates = [
                (i, a) for i, a in enumerate(allocation)
                if a["resource_idx"] in preference
                and float(a["urgency"]) < u
            ]
            if candidates:
                worst_i, worst = min(candidates, key=lambda x: float(x[1]["urgency"]))
                bumped_p   = worst["patient_idx"]
                bumped_r   = worst["resource_idx"]
                bumped_u   = worst["urgency"]

                # Swap: remove bumped patient, place incoming in same ward
                allocation.pop(worst_i)
                allocation.append({
                    "patient_idx":   p,
                    "resource_idx":  bumped_r,
                    "resource_name": RESOURCE_NAMES.get(bumped_r, f"Resource {bumped_r}"),
                    "urgency":       round(u, 4),
                    "fallback":      True,
                    "preempted":     True,
                })
                # ward_counts unchanged (one out, one in, same ward)
                assigned = True

                waitlist_entries.append({
                    "patient_idx":          bumped_p,
                    "urgency":              bumped_u,
                    "displaced":            True,
                    "displaced_by_urgency": round(u, 4),
                })

        if not assigned:
            waitlist_entries.append({
                "patient_idx": p,
                "urgency":     round(u, 4),
                "displaced":   False,
            })

    return sorted(allocation, key=lambda x: x["urgency"], reverse=True), waitlist_entries
