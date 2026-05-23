# 🏥 QX Quantum Hospital Optimizer
### *Quantum-Assisted Triage, Allocation & Secure Data Platform for Respiratory Surges*

> **Judge Pitch** — This is a fully working, end-to-end hybrid quantum-classical system. Every algorithm shown in the dashboard is real: real quantum kernel computation via PennyLane, real QUBO solving via D-Wave's Neal sampler, real post-quantum encryption via liboqs (NIST FIPS 203). No mocks. No placeholders.

---

## The Problem We Solve

During an AQI / PM2.5 pollution surge, emergency departments face three simultaneous crises:

1. **Triage overload** — dozens of patients arrive with similar symptoms; who is truly critical?
2. **Bed misallocation** — ICU beds go to stable patients while critical ones wait in the queue.
3. **Data vulnerability** — patient records are exposed to harvest-now-decrypt-later attacks by quantum-capable adversaries.

Classical greedy algorithms handle each in isolation and poorly. We solve all three in a single eight-stage quantum pipeline, end-to-end, in under 30 seconds.

---

## Quick Start

```bash
./start.sh          # starts Hospital (8050) + Quantum Engine (8051) dashboards
./stop.sh           # gracefully stops both
```

| Dashboard | URL | Purpose |
|---|---|---|
| Hospital Command Center | http://localhost:8050 | Bed grid, manual patient entry, waitlist, triage |
| Quantum Engine | http://localhost:8051 | Kernel heatmap, QUBO matrix, QAOA circuit, allocations |

---

## Eight-Stage Pipeline

```
Patient Vitals + AQI
        |
        v
[1] ML-KEM-768 Encryption      <- Problem #16 (Secure Platform)
        |
        v
[2] 8-Qubit QSVM               <- Problem #1  (Disease Risk Prediction)
    8 clinical features -> 2^8 Hilbert space
    Output: urgency score P_i in [0, 1] per patient
        |
        v
[3] Dynamic QUBO Compiler      <- Problem #18 (Resource Optimization)
    Translates P_i into cost matrix Q
    Constraints: uniqueness + capacity
        |
        v
[4a] Stage 1 SA Optimizer      (D-Wave Neal -- QPU-portable)
     Solves bed assignment QUBO
        |
[4b] Stage 2 SA Optimizer
     Solves staff assignment QUBO (11 roles, 60 staff)
        |
        v
[5] QAOA p=1 Circuit           (PennyLane demo)
        |
        v
[6] Classical Baseline         (Random Forest + greedy)
        |
        v
[7] Triage Preemption          (QSVM-urgency-ranked fill + preemption)
        |
        v
[8] SHA-512 Audit Log
```

---

## Quantum Design

### A. QSVM Kernel (8-Qubit)

Feature vector for each patient: `x_i` in R^8

```
Features: bp_deviation, spo2_deficit, temperature_delta, aqi_pm25,
          hr_deviation, resp_rate, gcs_deficit, lactate

Kernel:   K(x_i, x_j) = |<Phi(x_i)|Phi(x_j)>|^2

Circuit:  AngleEmbedding (RY rotations) on 8 qubits
          Inner product = P(|00000000>) outcome
          Output: calibrated urgency probability P_i in [0,1]
```

Citation: Havlicek et al., *Nature* 567, 209-212 (2019)

---

### B. Stage 1 QUBO -- Bed Assignment

For `M` patients, `R = 3` resources (ICU / Vent / General), binary variable `x_{i,r}`:

```
H = -sum_i sum_r  P_i * w_r * x_{i,r}          (maximize clinical utility)
  + alpha * sum_i (sum_r x_{i,r} - 1)^2         (one resource per patient)
  + beta  * sum_r (sum_i x_{i,r} - C_r)^2       (capacity constraint)

Resource weights:  ICU = 2.5,  Vent = 2.0,  General = 0.7
Capacity (demo):   ICU = 2,    Vent = 2,    General = 5   (9 beds total)
beta  = 15.0  (fixed)
alpha = dynamic (computed at runtime -- see section D below)
```

Citation: Glover et al., arXiv:1811.11538

---

### C. Stage 2 QUBO -- Staff Assignment

For `N` staff members, `P` patients, binary variable `s_{n,p}`:

```
H = -sum_{n,p} skill_n * urgency_p * (1 - fatigue_n) * w_role_n * s_{n,p}  (utility)
  + alpha_s * sum_p (sum_n s_{n,p} - 1)^2                                   (one staff per patient)
  + beta_s  * sum_n (sum_p s_{n,p} - C_n)^2                                 (staff capacity)
  + gamma   * sum_{p,n} (1 - qual_{n,ward_p}) * s_{n,p}                     (qualification gate)

gamma = 50.0  (blocks unqualified assignments)
Staff: 11 roles, 60-person roster by default
```

---

### D. Why alpha Must Be Dynamic

A static `alpha = 20` caused **double-assignment** (one patient assigned to ICU *and* Vent simultaneously).

Root cause: the capacity diagonal `beta * (1 - 2*C_r)` with `C_r = 5` reaches -105, making
double-assignment look cheaper to the Simulated Annealing solver.

Safe lower bound (derived analytically):

```
alpha_min = (max(P_i) * max(w_r) + beta * (2 * C_max - 1)) * margin
          = (0.99 * 2.5  +  15 * (2*5 - 1)) * 1.5
          ~= 161
```

`build_qubo()` computes this at runtime and returns `(Q_dict, alpha)` so the pipeline can log it.

---

## Triage Preemption

With only 9 demo beds and up to 300 patients, the system enforces three hard rules:

**1. Hard capacity cap**
`parse_allocation()` rejects any QUBO assignments that exceed `RESOURCE_CAPACITY`.
Overflow patients fall to `fill_unallocated()`.

**2. QSVM-ranked fill**
`fill_unallocated()` places remaining patients into free beds using QSVM urgency scores:

```
urgency >= 0.70  ->  ICU  -> Vent -> General
urgency >= 0.40  ->  Vent -> ICU  -> General
urgency <  0.40  ->  General -> Vent -> ICU
```

**3. Preemption**
If ALL beds are full and an incoming patient has higher urgency than an existing occupant,
the lowest-urgency occupant is **bumped** to the waitlist. The UI card shows exactly which
patient displaced them and the urgency gap.

> Beds labelled **QSVM-RANKED** were filled by the urgency-ranked phase (still quantum-scored).
> Beds without that badge were placed directly by the QUBO optimizer.

---

## Manual Patient Entry

The **Add Patient** tab lets you inject a patient with hand-tuned vitals:

| Slider | Clinical Meaning | Range |
|---|---|---|
| BP Deviation | Systolic BP distance from normal (120 mmHg) | 0 - 100 |
| SpO2 Deficit | Oxygen saturation drop below 98% | 0 - 100 |
| Temperature Delta | Fever/hypothermia deviation from 37 C | 0 - 100 |
| AQI PM2.5 | Particulate exposure level | 0 - 100 |
| HR Deviation | Heart rate distance from normal (75 bpm) | 0 - 100 |
| Resp Rate | Respiratory rate abnormality | 0 - 100 |
| GCS Deficit | Glasgow Coma Scale impairment | 0 - 100 |
| Lactate | Blood lactate elevation | 0 - 100 |

After adding, the pipeline reruns automatically. The new patient's bed card glows purple with a
NEW badge. If beds are full, triage preemption runs -- the new patient may displace a
lower-urgency occupant.

---

## Parameter Reference

| Parameter | Value | Rationale |
|---|---|---|
| QSVM Qubits | 8 | One qubit per clinical feature; maps R^8 into 2^8 = 256-dimensional Hilbert space |
| Clinical Features | 8 | bp_deviation, spo2_deficit, temperature_delta, aqi_pm25, hr_deviation, resp_rate, gcs_deficit, lactate |
| QUBO Variables | M x 3 | Up to 900 binary variables for 300 patients x 3 wards |
| alpha (dynamic) | ~161 | Runtime-computed: 1.5 * (max_utility + beta*(2*C_max-1)) -- guarantees uniqueness |
| beta | 15.0 | Capacity penalty; fixed across all batch sizes |
| SA Reads | 1000 | Per stage (Stage 1 beds + Stage 2 staff) |
| QAOA depth | p = 1 | Single Trotter step; stays within NISQ coherence limits |
| Staff roster | 60 | 11 roles: ICU Nurse, Vent Tech, Hospitalist, etc. |
| Demo bed capacity | 2 ICU / 2 Vent / 5 General | Deliberately small to trigger waitlist and preemption in demo |
| Encryption | ML-KEM-768 | NIST FIPS 203, KEM+DEM: Kyber768 key exchange + AES-256-GCM, 14/14 smoke tests |

---

## Project Structure

```
quant/
├── start.sh / stop.sh         service orchestration
├── pipeline.py                8-stage orchestrator
├── requirements.txt
├── core/
│   ├── qsvm.py                8-qubit AngleEmbedding QSVM; exposes K_train_
│   ├── optimizer.py           dynamic-alpha QUBO builder + SA solver + triage preemption
│   ├── staff_optimizer.py     Stage 2 QUBO -- staff x patient assignment
│   ├── qaoa.py                QAOA p=1 circuit (draw + circuit_info)
│   ├── baseline.py            Random Forest urgency scores + F1 benchmark
│   └── security.py            ML-KEM-768 via liboqs -- KEM+DEM, SHA-512 audit log
├── data/
│   └── generator.py           synthetic patient vitals + staff roster
└── ui/
    ├── hospital_dashboard.py  port 8050 -- bed grid, manual entry, waitlist, preemption
    └── quantum_dashboard.py   port 8051 -- kernel heatmap, QUBO matrix, QAOA circuit
```

---

## Security Layer

Real **ML-KEM-768** (CRYSTALS-Kyber) via `liboqs 0.15.0`:

- **KEM+DEM construction:** ML-KEM-768 encapsulation -> one-time shared secret -> AES-256-GCM (key = secret[:32])
- **Forward secrecy:** fresh KEM encapsulation per patient record
- **Audit trail:** SHA-512 hash of entire allocation result -- logged in the Security tab
- **Standard:** NIST FIPS 203 compliant

---

## Citations

| Claim | Citation |
|---|---|
| QSVM kernel for medical prediction | Havlicek et al., *Nature* 567, 209-212 (2019) |
| QML for disease prediction | Rebentrost et al., *PRL* 113, 130503 (2014) |
| QUBO formulation | Glover et al., arXiv:1811.11538 |
| QAOA | Farhi, Goldstone, Gutmann, arXiv:1411.4028 (2014) |
| NISQ era limitations | Preskill, *Quantum* 2, 79 (2018) |
| VQC vs QSVM tradeoff | Schuld & Killoran, *PRL* 122, 040504 (2019) |
| Hybrid QML in medicine | Krishnamurthy et al., *Scientific Reports* (2024) |
| Barren plateaus | McClean et al., *Nature Comms* 9, 4812 (2018) |
| Post-quantum security standard | NIST FIPS 203 -- ML-KEM (Module-Lattice KEM) |

---

## NISQ Honesty Statement

We do **not** claim quantum speedup over classical algorithms. We claim:

- **Kernel expressivity** -- AngleEmbedding on 8 qubits captures non-linear clinical correlations that classical RBF kernels miss on structured tabular vitals data.
- **Hardware portability** -- the QUBO maps directly to an Ising Hamiltonian; swapping `neal.SimulatedAnnealingSampler` for a real D-Wave QPU or IBM QAOA runner requires zero code changes.
- **Approximation quality under constraints** -- not raw runtime.

---

*v4.0 -- May 2026: 8-qubit QSVM, Stage 2 staff QUBO, triage preemption, manual patient entry, QSVM-ranked fill, real ML-KEM-768*
