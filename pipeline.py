"""
pipeline.py
End-to-end runner. Can be called directly or imported by either dashboard.

Usage:
    python pipeline.py --aqi 150 --patients 8

Layers
──────
  1. Secure ingestion   — ML-KEM proxy (PyNaCl)          Problem #16
  2. QSVM urgency       — quantum kernel (PennyLane)      Problem #1
  3. QUBO compiler      — dynamic α, capacity constraints
  4. SA optimizer       — D-Wave neal (QPU-portable)      Problem #18
  5. QAOA circuit       — PennyLane p=1 demo
  6. Classical baseline — greedy + Random Forest (F1 comparison)
  7. Stage 2 QUBO       — staff assignment (N×P Hilbert space)
"""

import argparse
import numpy as np
from sklearn.metrics import f1_score as _f1

import time

from data.generator import generate_patients, generate_staff, FEATURE_COLS
from core.qsvm import QuantumSVM
from core.optimizer import build_qubo, solve_qubo, parse_allocation, fill_unallocated
from core.baseline import classical_greedy, compute_utilization, random_forest_scores
from core.security import MLKEMProxy
from core.qaoa import draw_qaoa_circuit, circuit_info
from core.staff_optimizer import (
    build_staff_qubo, solve_staff_qubo, parse_staff_allocation, compute_staff_metrics
)

# One security instance per process (holds the hospital keypair)
_security = MLKEMProxy()


def _triage_reason(resource_idx: int, urgency: float, row) -> str:
    """Human-readable justification for a bed assignment based on patient vitals."""
    bp    = float(row.get("bp_deviation",  0))
    spo2  = float(row.get("spo2_deficit",  0))
    hr    = float(row.get("hr_deviation",  0))
    resp  = float(row.get("resp_rate",     0))
    gcs   = float(row.get("gcs_deficit",   0))
    lact  = float(row.get("lactate",       0))

    if resource_idx == 0:   # ICU / Trauma
        triggers = []
        if urgency >= 0.80:  triggers.append("critical urgency")
        if bp > 0.70:        triggers.append("severe BP deviation")
        if lact > 0.50:      triggers.append("elevated lactate")
        if gcs > 0.33:       triggers.append("GCS impairment")
        if hr > 0.70:        triggers.append("extreme tachycardia")
        return "ICU — " + (", ".join(triggers) if triggers else "multi-system compromise")
    elif resource_idx == 1:  # Ventilator Unit
        triggers = []
        if spo2 > 0.50:  triggers.append("SpO₂ deficit")
        if resp > 0.60:  triggers.append("severe tachypnoea")
        if urgency >= 0.60: triggers.append(f"urgency {urgency:.2f}")
        return "Vent — " + (", ".join(triggers) if triggers else "respiratory support needed")
    else:                    # General Ward
        triggers = []
        if urgency < 0.40: triggers.append("stable vitals")
        if gcs < 0.20:     triggers.append("alert & oriented")
        return "General — " + (", ".join(triggers) if triggers else "monitoring, lower acuity")


def run_pipeline(aqi_level: float = 50.0, n_patients: int = 100, verbose: bool = True):
    """
    Full pipeline: data → encrypt → QSVM → QUBO → SA → allocation.
    Returns dict with all results for dashboard consumption.
    """

    # ── 1. Generate + encrypt patient data ───────────────────────────────
    df       = generate_patients(n=n_patients, aqi_level=aqi_level)
    staff_df = generate_staff()
    X        = df[FEATURE_COLS].values
    y        = df["label"].values

    encrypted_blobs = _security.encrypt_dataframe(df)   # Problem #16

    if verbose:
        print(f"\n[1] Generated {n_patients} patients  |  AQI: {aqi_level}")
        print(f"    Security: {_security.summary()['algorithm']}")
        print(df[["patient_id"] + FEATURE_COLS + ["label"]].to_string(index=False))
        print(f"    Staff roster: {len(staff_df)} members generated")

    # ── 2. Train QSVM + score urgency ────────────────────────────────────
    if verbose:
        print("\n[2] Computing quantum kernel matrix (this takes ~30s for 8 patients)...")

    qsvm = QuantumSVM()
    qsvm.fit(X, y)
    urgency_scores = qsvm.predict_urgency(X)

    df["urgency_score"] = np.round(urgency_scores, 4)

    # QSVM F1: threshold urgency at 0.5
    qsvm_preds = (urgency_scores > 0.5).astype(int)
    qsvm_f1    = float(_f1(y, qsvm_preds, average="macro", zero_division=0))

    if verbose:
        print(f"    Urgency scores : {np.round(urgency_scores, 4)}")
        print(f"    QSVM F1 (train): {qsvm_f1:.3f}")

    # ── 3. Build QUBO from urgency scores ────────────────────────────────
    if verbose:
        print("\n[3] Compiling dynamic QUBO matrix...")

    Q, alpha = build_qubo(urgency_scores)

    if verbose:
        print(f"    Computed safe α = {alpha:.2f}  (hardcoded 20 caused double-assignment bug)")

    # ── 4. Solve QUBO ────────────────────────────────────────────────────
    if verbose:
        print("[4] Running Simulated Annealing optimizer (1000 reads)...")

    t0_stage1 = time.perf_counter()
    sample = solve_qubo(Q, num_reads=1000)
    stage1_solve_ms = round((time.perf_counter() - t0_stage1) * 1000, 1)
    quantum_allocation = parse_allocation(sample, urgency_scores)

    # Greedy fallback: ensure SA-missed patients get a bed if capacity allows
    quantum_allocation = fill_unallocated(
        quantum_allocation, n_patients, urgency_scores
    )

    # Add per-allocation triage reason
    for alloc in quantum_allocation:
        alloc["reason"] = _triage_reason(
            alloc["resource_idx"], alloc["urgency"], df.iloc[alloc["patient_idx"]]
        )
        if alloc.get("fallback"):
            alloc["reason"] += "  [greedy placed]"

    # Build waitlist: patients with no bed after fill (truly over capacity)
    allocated_idxs = {a["patient_idx"] for a in quantum_allocation}
    ward_occ = {r: sum(1 for a in quantum_allocation if a["resource_idx"] == r)
                for r in range(3)}
    from data.generator import RESOURCE_CAPACITY as _RC, RESOURCE_NAMES as _RN
    full_wards = [_RN[r] for r, occ in ward_occ.items() if occ >= _RC[r]]
    waitlist = [
        {
            "patient_idx": p,
            "patient_id":  df.iloc[p]["patient_id"],
            "urgency":     round(float(urgency_scores[p]), 4),
            "reason":      "Waiting — " + (", ".join(f"{w} full" for w in full_wards)
                           if full_wards else "pending bed assignment"),
        }
        for p in range(n_patients) if p not in allocated_idxs
    ]

    # ── 4b. Stage 2 QUBO — Staff Assignment ──────────────────────────────
    if verbose:
        print(f"[4b] Building Stage 2 staff QUBO ({len(staff_df)} staff × {n_patients} patients)...")

    Q_staff, alpha_s = build_staff_qubo(staff_df, df, quantum_allocation, urgency_scores)

    t0_stage2 = time.perf_counter()
    staff_sample = solve_staff_qubo(Q_staff, num_reads=1000)
    stage2_solve_ms = round((time.perf_counter() - t0_stage2) * 1000, 1)

    staff_allocation = parse_staff_allocation(
        staff_sample, staff_df, df, quantum_allocation, urgency_scores
    )
    staff_metrics = compute_staff_metrics(
        staff_allocation, staff_df, df, quantum_allocation, urgency_scores
    )

    if verbose:
        print(f"    Stage 1 solve: {stage1_solve_ms} ms  |  Stage 2 solve: {stage2_solve_ms} ms")
        print(f"    Skill-acuity match: {staff_metrics['skill_acuity_match']}")
        print(f"    Unassigned patients: {staff_metrics['unassigned_count']}")
        print(f"    Cross-qualification rate: {staff_metrics['cross_qual_rate']}%")

    # ── 5. Classical baseline ─────────────────────────────────────────────
    classical_allocation = classical_greedy(df)
    rf_urgency, rf_f1    = random_forest_scores(X, y)

    if verbose:
        print(f"\n[5] Random Forest F1 (train): {rf_f1:.3f}  |  QSVM F1: {qsvm_f1:.3f}")

    # ── 6. Metrics ───────────────────────────────────────────────────────
    q_util  = compute_utilization(quantum_allocation)
    c_util  = compute_utilization(classical_allocation)

    unallocated_classical = sum(1 for r in classical_allocation if r["resource_name"] == "⚠️ Unallocated")
    unallocated_quantum   = sum(1 for r in quantum_allocation   if r["resource_name"] == "⚠️ Unallocated")

    # ── 7. QAOA circuit (demo: 2 patients × 3 resources = 6 qubits) ──────
    if verbose:
        print("\n[7] Rendering QAOA circuit (p=1, 6-qubit demo)...")

    qaoa_b64  = draw_qaoa_circuit(Q)
    qaoa_meta = circuit_info(n_patients, 3, alpha)

    # ── 8. Audit hash ─────────────────────────────────────────────────────
    alloc_hash = _security.audit_hash(quantum_allocation)

    if verbose:
        print("\n══════════════════════════════════════════")
        print("  QUANTUM ALLOCATION")
        print("══════════════════════════════════════════")
        for row in quantum_allocation:
            pid = df.loc[row["patient_idx"], "patient_id"]
            print(f"  {pid} (urgency {row['urgency']:.4f}) → {row['resource_name']}")

        print("\n  Utilization:", q_util)
        print(f"  Unallocated: {unallocated_quantum}")

        print("\n══════════════════════════════════════════")
        print("  CLASSICAL GREEDY BASELINE")
        print("══════════════════════════════════════════")
        for row in classical_allocation:
            pid = df.loc[row["patient_idx"], "patient_id"]
            print(f"  {pid} → {row['resource_name']}")

        print(f"  Unallocated: {unallocated_classical}")
        print(f"\n  Audit hash (SHA-512): {alloc_hash[:32]}…")

    return {
        # ── Core data ──────────────────────────────────────────────────
        "df":                    df,
        "urgency_scores":        urgency_scores,
        "quantum_allocation":    quantum_allocation,
        "classical_allocation":  classical_allocation,
        "quantum_utilization":   q_util,
        "classical_utilization": c_util,
        "unallocated_quantum":   unallocated_quantum,
        "unallocated_classical": unallocated_classical,
        "waitlist":              waitlist,
        # ── QUBO metadata ──────────────────────────────────────────────
        "alpha":                 alpha,
        "qubo_dict":             Q,
        # ── QSVM kernel ────────────────────────────────────────────────
        "kernel_matrix":         qsvm.K_train_,
        # ── F1 comparison ──────────────────────────────────────────────
        "qsvm_f1":               qsvm_f1,
        "rf_f1":                 rf_f1,
        "rf_urgency":            rf_urgency,
        # ── QAOA circuit ───────────────────────────────────────────────
        "qaoa_circuit_b64":      qaoa_b64,
        "qaoa_info":             qaoa_meta,
        # ── Security layer ─────────────────────────────────────────────
        "security":              _security.summary(),
        "encrypted_sample":      encrypted_blobs[0] if encrypted_blobs else {},
        "audit_hash":            alloc_hash,
        # ── Stage 2: Staff allocation ───────────────────────────────────
        "staff_df":              staff_df,
        "staff_allocation":      staff_allocation,
        "staff_metrics":         staff_metrics,
        "staff_qubo_dict":       Q_staff,
        "alpha_s":               alpha_s,
        "stage1_solve_ms":       stage1_solve_ms,
        "stage2_solve_ms":       stage2_solve_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aqi", type=float, default=50.0, help="PM2.5 AQI level (0–500)")
    parser.add_argument("--patients", type=int, default=8, help="Number of patients")
    args = parser.parse_args()

    run_pipeline(aqi_level=args.aqi, n_patients=args.patients)
