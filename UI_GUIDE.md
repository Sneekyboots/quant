# 🏥 UI Guide — QX Quantum Hospital Optimizer
> Everything on both dashboards explained in plain terms.

There are **two browser tabs** running on separate ports. Launch both with `./start.sh`.

---

## Dashboard 1 — Hospital Command Center  `http://localhost:8050`

This is the operational view. It shows where patients are, how urgent they are, which ward they were assigned to, and which staff member is looking after them.

---

### Header Bar

| Element | What it means |
|---------|--------------|
| **QX Hospital Command Center** | Title — no interactive function |
| **LIVE** badge (green) | Indicates the dashboard is running; not a real-time poll, updates only when you press the run button |
| **Last updated** timestamp | Shows the clock time of the most recent pipeline run (top-right) |

---

### Controls Strip

These three controls sit side by side above the main content.

#### 🌫️ Smog Intensity (PM2.5) — slider, 0–500
Simulates outdoor air quality (AQI). Moving it right makes the air "worse":
- **0 (CLEAR)** → No air-quality contribution. Patients have baseline vitals only.
- **250 (MID)** → Moderate smog. SpO₂ deficits and BP deviations shift upward because more patients arrive with respiratory symptoms.
- **500 (SURGE)** → Maximum pollution. Nearly all patients will have elevated urgency scores, forcing more into ICU and Ventilator Unit.

The `aqi_factor = AQI / 500` value is injected into the synthetic patient generator and fed directly into the QSVM feature vector as the fourth qubit dimension (`aqi_pm25`).

#### 👤 Patient Intake — slider, 4–20 (step 2)
The number of emergency patients to simulate in this run.
- Lower counts (4–8) → faster pipeline, fewer patients competing for beds.
- Higher counts (16–20) → more QUBO variables, longer solve time, some patients may end up "Unallocated" if all beds are full.
- Default is **16** — matching the total ward capacity (4 ICU + 4 Vent + 8 General).

#### OPTIMIZE NOW! 🚀 — red button
Triggers the full pipeline. Nothing happens until you press this. The button fires:
1. Patient data generation
2. QSVM urgency scoring
3. Stage 1 QUBO (bed allocation)
4. Stage 2 QUBO (staff assignment)
5. Classical baseline for comparison
6. Security audit hash

Expect **30–120 seconds** on first run (quantum kernel computation scales as O(n²) patients).

---

### Metric Cards (4 tiles)

| Card | Icon | What it shows |
|------|------|--------------|
| **Live Queue** | 👤 | Number of patients in this run — mirrors the slider value |
| **Crisis Alert** | 🚨 | Count of patients whose urgency score is ≥ 0.70 (threshold for "critical") — these patients get animated red borders on their bed cards |
| **Skill Match** | 🧑‍⚕️ | Stage 2 skill-acuity match score (0–1). How well each assigned staff member's skill level correlates with their patient's urgency. 1.0 = perfect match, 0.0 = random |
| **Smog Level** | 🌫️ | Echo of the AQI slider value for quick reference |

---

### 🏨 Live Ward Snapshot — Bed Grid

Three columns, one per ward. Each column shows:

#### Column header (dark bar)
`WARD NAME  [n/capacity FULL]` — tells you how many of the physical beds are occupied right now.

| Ward | Max beds | Clinical meaning |
|------|---------|-----------------|
| **ICU / Trauma** 🏥 | 4 | Highest acuity. Patients with urgency ≥ ~0.7 and high BP/SpO₂ deficit land here |
| **Vent Unit** 🫁 | 4 | Patients needing respiratory support. AQI surges push more patients here |
| **General Ward** 🛏️ | 8 | Lower acuity stable patients. Largest capacity |

#### Bed cards (white cards inside the column)
Each occupied bed shows one patient:

| Label | What it means |
|-------|--------------|
| **PATIENT ID** | Synthetic ID, e.g. `P01`, `P07` |
| **URGENCY** (top-right, in red if ≥ 0.70) | The QSVM urgency score. Ranges 0–1. This is the output of the quantum kernel — not a rule-based triage, but a learned similarity score across all patients' feature vectors in the 2⁴-dimensional Hilbert space |
| **BP DEV** | Normalised blood-pressure deviation from baseline (0–1). Higher = more hypertensive / hypotensive risk |
| **O₂ SAT** | Displayed as `1 − spo2_deficit`. 1.0 = perfect saturation, lower = desaturation risk |
| **Pulsing red border + animation** | Only on critical patients (urgency ≥ 0.70). A visual alert that this patient requires immediate attention |

#### Empty beds
Grey dashed boxes labelled **FREE BED** — placeholder showing physical capacity still available.

---

### 📋 Patient Queue Table

A full tabular view of every patient in this run.

| Column | What it means |
|--------|--------------|
| **PATIENT** | ID, e.g. `P01` |
| **BP Δ** | Same as BP DEV above |
| **O₂ SAT** | `1 − spo2_deficit` |
| **AQI** | The per-patient AQI feature fed into the QSVM (slight noise added per patient around the global slider value) |
| **SCORE** | Urgency score (0–1). Red text = critical (≥ 0.70) |
| **STATUS** | `URGENT` (red badge) or `STABLE` (green badge) — derived from urgency ≥ 0.70 |
| **ASSIGNED TO** | The ward name chosen by the Stage 1 QUBO optimizer. `⚠️ UNALLOCATED` appears when all beds in every ward are full (patient count > 16) |

---

### 🛡️ Quantum Shield Activated — Security Panel

Two cards side by side.

#### Left card — Algorithm identity
| Element | What it means |
|---------|--------------|
| **PQC ACTIVE** badge | Post-Quantum Cryptography is running |
| **Algorithm name** | `ML-KEM-768 (PyNaCl proxy)` — the system implements a Curve25519 + XSalsa20-Poly1305 proxy that approximates the interface of NIST FIPS 203 ML-KEM-768 |
| **PUBKEY fingerprint** | First 32 hex chars of the hospital's public key. Regenerated each time the process starts |
| **NIST TARGET** | `NIST FIPS 203 (ML-KEM-768)` — the real hardware standard this proxy is designed to be swapped out for |

#### Right card — Audit trail
| Element | What it means |
|---------|--------------|
| **AUDIT PROOF (QUANTUM LEDGER)** | First 48 characters of the SHA-512 hash of the full allocation output. This hash changes every run — it is an immutable fingerprint proving that the specific set of patient→ward assignments was produced by this pipeline at this moment |

> **Why does security matter here?** Patient allocation decisions are PHI (Protected Health Information). ML-KEM-768 is a NIST-standardised post-quantum key encapsulation that is resistant to Shor's algorithm attacks. Every patient record is encrypted before the QSVM sees it.

---

### 👨‍⚕️ Staff Deployment — Stage 2 QUBO

The staff grid is produced by a **second, independent QUBO** that runs after the bed allocation is locked in. It assigns staff members to patients based on ward placement, skill level, and shift fatigue.

#### 📊 Pitch Metrics Strip (at the top)
A horizontal row of badges summarising Stage 2 performance:

| Badge | What it means |
|-------|--------------|
| **ICU Physician: X%** | Percentage of ICU Physician capacity used (max 3 patients per physician × 2 physicians = 6 slots) |
| **ICU Nurse: X%** | Same for ICU Nurses (max 2 patients each × 3 nurses = 6 slots) |
| **Vent Specialist: X%** | Same for Vent Specialists (max 2 each × 2 = 4 slots) |
| **General Nurse: X%** | Same for General Nurses (max 4 each × 4 nurses = 16 slots) |
| **Unassigned: N pts** | Patients who received no staff assignment. Appears red if > 0 |
| **Cross-qual: X%** | Percentage of assignments where a staff member was assigned to a ward outside their listed qualifications (e.g. an ICU Nurse sent to the General Ward). Yellow if > 0, green if 0 |
| **S1: Xms → S2: Xms** | Wall-clock solve time for Stage 1 (bed QUBO) and Stage 2 (staff QUBO) in milliseconds. On a real QPU these would be in microseconds |

#### Staff Grid (three ward columns)
Same layout as the bed grid but for staff:

**Column header**: Ward name + number of staff assigned.

**Staff chip** (each card in the column):

| Element | What it means |
|---------|--------------|
| **Staff ID** (e.g. `S03`) | Monospace ID of the staff member |
| **FRESH / OK / TIRED** label | Fatigue state derived from `fatigue_score = 1 − shift_remaining / 8 hrs`. FRESH (green) < 0.40, OK (yellow) 0.40–0.69, TIRED (red) ≥ 0.70 |
| **Coloured left border** | Green = FRESH, Yellow = OK, Red = TIRED — instant visual scan of who is burnt out |
| **Role name** (grey text) | `ICU Physician`, `ICU Nurse`, `Vent Specialist`, or `General Nurse` |
| **Skill X.XX · Fatigue X.XX** | Raw floats. Skill 0–1 (higher = more experienced); Fatigue 0–1 (higher = more tired) |

---

## Dashboard 2 — Quantum Engine Visualizer  `http://localhost:8051`

This is the algorithm inspection view. It shows the internals of the quantum and optimization computations — intended for technical review, not clinical operations.

---

### Header Bar

| Element | What it means |
|---------|--------------|
| **QSVM KERNEL** badge | PennyLane quantum kernel is in use |
| **QUBO SOLVER** badge | D-Wave neal Simulated Annealing solver |
| **QAOA P=1** badge | Quantum Approximate Optimization Algorithm with Trotter depth 1 |

---

### Controls (same as Dashboard 1)

**Smog Intensity** and **Patient Count** sliders behave identically to Dashboard 1. **EXPLODE PIPELINE ⚛️** is the equivalent of "OPTIMIZE NOW".

---

### Summary Bar (appears after first run)

Six info cards showing the key numbers from this pipeline run:

| Card | What it means |
|------|--------------|
| **QUBITS** | Physical qubits used by the QSVM kernel circuit (= number of features = 4) |
| **S1 VARS** | Number of binary variables in Stage 1 QUBO = patients × 3 resources |
| **S2 VARS** | Number of binary variables in Stage 2 QUBO = 11 staff × patients |
| **α PENALTY** | The dynamically computed uniqueness penalty coefficient. High enough to guarantee no patient is assigned to two beds simultaneously |
| **QSVM F1** | Macro F1 score of the QSVM classifier on the training set (urgency > 0.5 threshold). Compared against the classical RF baseline |
| **RF F1** | Random Forest baseline F1 score for comparison |

---

### Tab: QSVM Kernel

**What it shows**: A heatmap of the quantum kernel matrix **K** where `K[i,j]` = the inner product of patients i and j in the 2⁴-dimensional quantum feature space.

**How to read it**:
- The matrix is symmetric (K[i,j] = K[j,i]) — the heatmap should be symmetric about the diagonal.
- The diagonal is always 1.0 (a patient is perfectly similar to themselves).
- Off-diagonal values range from 0–1. A cell close to 1 (dark red) means two patients have nearly identical feature vectors in quantum space — the QSVM will score them similarly.
- A cell close to 0 (dark blue) means the two patients occupy very different regions of the feature space.
- **Why quantum?** The `RY` angle embedding maps each feature into a qubit rotation angle. The kernel circuit evaluates `|⟨φ(xᵢ)|φ(xⱼ)⟩|²` — a non-linear similarity that a classical dot product cannot capture.

---

### Tab: QUBO Matrix

**What it shows**: A heatmap of the Stage 1 QUBO matrix **Q** (upper-triangle form). Each axis label is a binary variable `p{patient}_r{resource}`, e.g. `p3_r1` = "Is patient 3 assigned to resource 1 (Ventilator Unit)?".

**How to read it**:
- **Diagonal cells** (where i = j): the linear bias for that variable. Negative = solver is *rewarded* for setting this variable to 1 (clinical utility term). More negative = higher urgency or better resource fit.
- **Off-diagonal cells** (where i ≠ j): the coupling strength between two variables.
  - Same patient, different resources (e.g. `p2_r0` vs `p2_r1`): large **positive** value = the uniqueness penalty α pushing the solver to pick only one resource.
  - Same resource, different patients (e.g. `p0_r2` vs `p1_r2`): positive value = capacity penalty β ensuring the ward doesn't overflow.
  - Different patient + different resource: near zero = no interaction.
- **Colorscale** (RdBu): red = positive (penalty), blue = negative (reward).

---

### Tab: QAOA Circuit

**What it shows**: A PennyLane circuit diagram of a 6-qubit QAOA ansatz at depth p=1.

**Key elements**:
- **Hadamard (H) gates** at the start → put all qubits into equal superposition (the QAOA initial state `|+⟩⊗n`)
- **ZZ coupling gates** → apply the problem Hamiltonian (cost layer). Each ZZ gate encodes one QUBO coupling `Q[i,j]`
- **RX gates** → the mixer layer, allowing the optimizer to explore the solution space
- **Measurement** at the end → collapse the quantum state to a bitstring representing one candidate allocation

> This circuit is a **demo** using 6 qubits (2 patients × 3 resources). The full production QUBO for 16 patients × 3 resources would need 48 qubits — beyond current NISQ hardware, which is why the Simulated Annealing solver is used in production.

Below the circuit, a metadata card shows:
| Field | Meaning |
|-------|---------|
| **Total qubits** | n_patients × n_resources |
| **QUBO terms** | Number of non-zero entries in Q (= number of ZZ + Z gates) |
| **α penalty** | Same as summary bar |
| **Circuit depth** | Gate layers = 2p + 1 (Hadamard init + p cost + p mixer) |

---

### Tab: Allocations

**What it shows**: A side-by-side diff of quantum (QUBO) versus classical (greedy) patient assignments.

**Quantum QUBO column**: Each row = one patient, with their urgency score and the ward chosen by the QUBO optimizer. Coloured by ward.

**Classical Greedy column**: The same patients allocated by a simple rule: sort by urgency descending, fill ICU first, then Vent, then General. No constraint-satisfaction — purely sequential.

**Why they differ**: The QUBO solves the *global* optimum (maximise total clinical utility subject to capacity constraints) simultaneously. The greedy method is myopic — it never reassigns a patient once placed, so it can overflow one ward while another sits empty.

---

### Tab: Staff QUBO

**What it shows**: The internals of the Stage 2 (staff assignment) QUBO.

#### Info Cards (top strip)
| Card | What it means |
|------|--------------|
| **S2 QUBITS** | Binary variables in the staff QUBO = 11 staff × n_patients |
| **α_s PENALTY** | Uniqueness penalty for staff — ensures no patient is listed twice in the assignment matrix |
| **SKILL MATCH** | Same as the Skill Match tile on Dashboard 1 |
| **UNASSIGNED** | Patients with no staff member assigned |
| **CROSS-QUAL** | Percentage of assignments violating ward qualification rules |
| **SOLVE TIMES** | `S1 Xms / S2 Xms` wall-clock times |

#### Staff QUBO Heatmap
Same interpretation as the Stage 1 QUBO matrix, but axes are now `s{staff}_p{patient}` variables (e.g. `s2_p7` = "Is staff member 2 assigned to patient 7?"). Only the first 48 variables are shown for readability.
- **Diagonal (negative/blue)**: utility reward — stronger when staff skill level matches patient urgency.
- **Off-diagonal same patient (positive/red)**: α_s uniqueness penalty.
- **Off-diagonal same staff (positive/red)**: capacity penalty preventing one nurse from being assigned to more patients than their nurse-patient ratio allows.
- **Qualification coupling**: large positive off-diagonal terms appear when a staff member is assigned to a ward they are not qualified for (γ = 50 penalty).

#### Staff Utilisation Chart (bar chart)
Horizontal bars showing utilisation % per role. A dashed line at 80% marks the healthy utilisation threshold. Bars above 80% indicate that role is close to being overwhelmed.

#### Skill-Acuity Scatter Plot
- **X axis**: Patient urgency score (0–1)
- **Y axis**: Assigned staff member's skill level (0–1)
- **Dot colour**: Fatigue score (green = fresh → yellow → red = tired)

In an ideal assignment every dot sits near the diagonal — high-urgency patients are matched with high-skill (experienced) staff. Dots in the top-left quadrant mean an experienced staff member is looking after a stable patient (waste). Dots in the bottom-right mean a tired or low-skill nurse is looking after a critical patient (risk).

---

## Appendix — Key Metrics Glossary

| Term | Definition |
|------|-----------|
| **Urgency score** | QSVM output ∈ [0, 1]. Derived from the quantum kernel similarity of a patient's feature vector to the high-acuity training examples |
| **BP Deviation** | `bp_deviation` feature ∈ [0, 1]. Normalised deviation of blood pressure from a healthy baseline, amplified by AQI |
| **SpO₂ deficit** | `spo2_deficit` feature ∈ [0, 1]. Normalised oxygen saturation deficit; displayed as `1 − deficit` so 1.0 = perfect |
| **QUBO α (alpha)** | Uniqueness penalty. Computed dynamically as `1.5 × (max_utility + β(2·C_max − 1))`. Guarantees no patient is double-assigned |
| **QUBO β (beta)** | Capacity penalty = 15.0 (fixed). Penalises any ward exceeding its bed count |
| **Staff α_s** | Staff uniqueness penalty (same role as α but for the staff assignment QUBO) |
| **Staff γ (gamma)** | Qualification penalty = 50.0. Applied when a staff member is assigned to a ward they are not qualified for |
| **Fatigue score** | `1 − shift_remaining / 8`. 0 = just started shift (FRESH), 1 = zero hours left (TIRED) |
| **Skill level** | Staff experience ∈ [0, 1]. Drawn from a uniform distribution per role; physicians skew higher |
| **Skill-acuity match** | Pearson-like correlation between staff skill level and patient urgency across all assignments. Closer to 1.0 = better matching |
| **Cross-qual rate** | `n_violations / n_assignments × 100`. A violation occurs when a staff member covers a ward not in their `STAFF_QUALIFICATIONS` list |
| **F1 score** | Harmonic mean of precision and recall for the binary urgency classifier (threshold 0.5). Macro-averaged across both classes |
| **SA (Simulated Annealing)** | The classical heuristic used to minimise the QUBO energy. QPU-portable — the same QUBO dict can be submitted directly to a D-Wave Advantage quantum annealer by swapping the sampler |
| **QAOA** | Quantum Approximate Optimization Algorithm. A variational quantum circuit that encodes the QUBO as a Hamiltonian and searches for the ground state using parameterised rotations |
| **ML-KEM-768** | NIST FIPS 203 post-quantum key encapsulation standard. The proxy used here matches its interface but uses Curve25519 + XSalsa20-Poly1305 internally |
| **Audit hash** | SHA-512 fingerprint of the serialised quantum allocation output. Provides tamper-evidence: any change to any assignment produces a completely different hash |
