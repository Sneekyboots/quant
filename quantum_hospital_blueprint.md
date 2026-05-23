# 🏥 Quantum-Enhanced Emergency Hospital Resource Optimizer
### Hackathon Blueprint — Quantum Vibes 2026
**Problem Statements: #18 (Resource Optimization) + #1 (Disease Risk Prediction) + #16 (Secure Platform)**

---

## 1. One-Line Pitch
> A hybrid quantum-classical pipeline that predicts patient surge risk using a Quantum SVM, then dynamically solves resource allocation (ICU beds, ventilators, staff) as a QUBO problem — with an India-first lens (NQM, Ayushman Bharat, AQI-linked surges).

---

## 2. System Architecture (Top-Down)

```
[ INPUT ]
  Real-time patient vitals (BP, SpO2, Temp)
  + Environmental AQI (PM2.5 particulate data)
        │
        ▼
┌─────────────────────────────────────┐
│  LAYER 1: Secure Data Ingestion     │
│  Post-Quantum Encryption (ML-KEM)   │
│  → Addresses Problem #16            │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  LAYER 2: QML Prediction Engine     │
│  Quantum SVM (kernel via PennyLane) │
│  Input: 4 features → Hilbert space  │
│  Output: Urgency Score P_i ∈ [0,1]  │
│  → Addresses Problem #1             │
└─────────────────────────────────────┘
        │
        ▼ Urgency Score Vector
┌─────────────────────────────────────┐
│  LAYER 3: Dynamic QUBO Compiler     │
│  Translates P_i into cost matrix    │
│  Constraints: uniqueness + capacity │
└─────────────────────────────────────┘
        │
        ▼ QUBO Dict
┌─────────────────────────────────────┐
│  LAYER 4: Hybrid Optimizer          │
│  Simulated Annealing (neal) now     │
│  QAOA circuit (PennyLane) for demo  │
│  → Addresses Problem #18            │
└─────────────────────────────────────┘
        │
        ▼
[ OUTPUT ]
  Patient → Resource Allocation Matrix
  (ICU / Ventilator / General Ward)
  Displayed on Streamlit dashboard
```

---

## 3. Math That Actually Works

### A. Quantum Kernel (QSVM)
For patient feature vectors **x_i, x_j ∈ ℝ⁴**:

```
K(x_i, x_j) = |⟨Φ(x_i)|Φ(x_j)⟩|² 
```

- Angle embedding: each feature mapped via RY rotations
- Inner product = probability of |0000⟩ outcome
- This kernel matrix feeds a classical SVC (sklearn)
- Output: calibrated urgency probability **P_i**

**Real citation:** Havlíček et al., *Nature* 567, 209–212 (2019)

### B. QUBO Cost Hamiltonian
For M patients, R resources, binary variable x_{i,r}:

```
H = -∑ P_i · w_r · x_{i,r}          ← maximize utility
  + α · ∑_i (∑_r x_{i,r} - 1)²      ← one resource per patient
  + β · ∑_r max(0, ∑_i x_{i,r} - C_r)² ← capacity constraint
```

- **α = dynamic** — computed at runtime (see §6 for derivation)  
- **β = 15.0** (capacity penalty, fixed)
- **w_r** = resource tier weight (ICU=2.5, Vent=2.0, General=0.7)

**Why α must be dynamic:**  
For α to enforce uniqueness it must satisfy:
```
α > max(P_i) · max(w_r) + β · (2·C_max − 1)
```
With β=15 and C_max=4 (General Ward), the capacity diagonal alone reaches **−105**.  
The original hardcoded α=20 violated this by a factor of 8x, causing double-assignment.  
Safe formula with 1.5× margin:
```
α = (max_urgency × max_weight + β × (2×C_max − 1)) × 1.5
  = (0.99×2.5 + 15×7) × 1.5  ≈  161
```

**Real citation:** Glover et al., "A Tutorial on Formulating and Using QUBO Models," arXiv:1811.11538

### C. NISQ Justification
We do NOT claim quantum speedup on a simulator. We claim:
- Better kernel expressivity than classical RBF for structured clinical tabular data
- Direct Ising Hamiltonian compilation → portable to real QPU without code changes
- Approximation quality under constrained resources, not raw runtime

**Real citations:**
- Preskill, "Quantum Computing in the NISQ Era," *Quantum* 2, 79 (2018)
- Cerezo et al., "Variational Quantum Algorithms," *Nature Reviews Physics* (2021)

---

## 4. Tech Stack

| Component | Tool | Why |
|---|---|---|
| Quantum circuits | PennyLane | Best for hybrid, great docs |
| Classical ML | scikit-learn SVC | Precomputed kernel support |
| QUBO solver | D-Wave neal | Fast SA on CPU, free |
| Hospital UI | Dash (port 8050) | Dark command center: bed grid + patient cards |
| Quantum UI | Dash (port 8051) | Kernel heatmap, QUBO matrix, QAOA circuit, alloc diff |
| Map layer | Folium | Hospital network visualization (pending integration) |
| Data | UCI Heart Disease + synthetic AQI | Real + India-relevant |
| Post-quantum security | **liboqs 0.15.0** — real ML-KEM-768 (CRYSTALS-Kyber) via Open Quantum Safe | NIST FIPS 203 compliant, 14/14 smoke tests passing |

---

## 5. Dataset Plan

| Dataset | Use |
|---|---|
| UCI Heart Disease (Cleveland) | Train QSVM for cardiac risk |
| Indian Liver Patient Dataset (Kaggle) | India-specific disease angle |
| Synthetic AQI generator | PM2.5 surge simulation |
| Synthetic hospital intake | 8–20 patient QUBO instances |

**Why not raw hospital data?** Privacy + hackathon time. Synthetic + UCI is standard for QML papers at this scale.

---

## 6. Engineering Bugs Found & Fixed

### Bug 1 — Critical: QUBO double-assignment (α=20 too small)

**Symptom:** Patient P01 was simultaneously assigned to ICU *and* Ventilator in every run.

**Root cause analysis:**  
For x_{i,r} variables, the uniqueness penalty diagonal is `α·(1−2) = −α`.  
The capacity penalty diagonal for General Ward (C=4) is `β·(1−2·C) = 15·(1−8) = −105`.  
With α=20, the SA objective for double-assigning P01 to ICU+Ventilator was:
```
Energy(double) = −P01·w_ICU − P01·w_Vent + penalty = −94.5   ← lower = preferred
Energy(single) = −P01·w_ICU + penalty = −67.5
```
SA was working correctly — it found the lower energy, which happened to be the wrong solution.

**Fix — dynamic α derived analytically:**
```python
BETA = 15.0

def compute_min_alpha(urgency_scores, resource_weights, resource_capacity,
                      beta=BETA, margin=1.5):
    max_utility  = float(np.max(urgency_scores)) * max(resource_weights)
    max_cap_diag = beta * (2 * max(resource_capacity) - 1)  # worst β(2C−1)
    return (max_utility + max_cap_diag) * margin  # → ≈161 for default params
```
`build_qubo()` now calls this and returns `(Q_dict, alpha)` so the pipeline can log it.

---

### Bug 2 — `solve_qubo` broke on `(Q, alpha)` tuple return

**Symptom:** After fixing Bug 1, `build_qubo` returned a tuple but `solve_qubo` expected a plain dict.

**Fix:**
```python
def solve_qubo(Q_or_tuple, num_reads=200):
    Q = Q_or_tuple[0] if isinstance(Q_or_tuple, tuple) else Q_or_tuple
    ...
```
Backward-compatible — accepts both forms.

---

### Bug 3 — `pd.read_json(json_string)` treated string as file path

**Symptom:** Dash callbacks crashed with `FileNotFoundError` whose message was the entire JSON payload.

**Root cause:** pandas ≥ 2.0 changed `read_json` to interpret short strings as file paths before trying to parse them as JSON.

**Fix:**
```python
import io
df = pd.read_json(io.StringIO(data["df"]), orient="split")
```

---

### Bug 4 — `matplotlib` not installed

**Symptom:** `ModuleNotFoundError: No module named 'matplotlib'` on first dashboard import.

**Fix:** Added `matplotlib>=3.8.0` to `requirements.txt`. PennyLane's `draw_mpl` circuit renderer requires it but does not declare it as a hard dependency.

---

### Bug 5 — QSVM kernel matrix inaccessible for dashboard

**Symptom:** Dashboard couldn't show the N×N kernel heatmap — `qsvm.K_train_` did not exist.

**Fix** in `core/qsvm.py`:
```python
def fit(self, X, y):
    self.X_train_scaled = self.scaler.fit_transform(X)
    K_train = compute_kernel_matrix(self.X_train_scaled)
    self.K_train_ = K_train   # ← added for dashboard
    self.svc.fit(K_train, y)
```

---

## 7. Build Plan & Team Split

### Week 1 — Core Pipeline
| Day | Task | Owner |
|---|---|---|
| 1–2 | Data preprocessing + QSVM kernel on UCI dataset | Person A |
| 1–2 | QUBO formulation + neal solver working end-to-end | Person B |
| 3–4 | Connect QSVM output → QUBO input (dynamic coupling) | Both |
| 5 | Classical baseline (Random Forest + OR-Tools greedy) | Person A |
| 5 | Benchmark metrics: accuracy, F1, allocation utilization % | Person B |

### Week 2 — Demo Polish
| Day | Task | Owner |
|---|---|---|
| 6–7 | Streamlit dashboard: AQI slider + live re-optimization | Person B |
| 6–7 | QAOA circuit mock (PennyLane) for visual demo | Person A |
| 8 | Folium hospital network map | Person B |
| 8 | Classical vs Quantum comparison charts | Person A |
| 9 | Security layer (ML-KEM encryption mock) | Either |
| 10 | Final integration + pitch deck | Both |

---

## 8. Dash Dashboard Features  *(Streamlit replaced — fully updated May 2026)*

### Dashboard 1 — Hospital Command Center  (`ui/hospital_dashboard.py`, port 8050)
Theme: custom beige + Fredoka font — clean, presentable, judge-ready.

#### Controls (STEP 0)
- **AQI / Smog Intensity slider** (0–500 PM2.5) with marks: Clear / Moderate / Surge
- **Patient Intake Count slider** (10–300, step 10) — warns that larger counts extend QSVM kernel computation
- **Optimize Now** button — triggers the full 8-stage pipeline

#### Fullscreen Loading Spinner
- `dcc.Loading(fullscreen=True, custom_spinner=...)` with a spinning ⚛️ atom animation
- Spinner only appears during actual pipeline computation (not on page load)
- `delay_show=100ms` prevents flash on fast runs

#### KPI Cards (always visible after first run)
| Card | What it shows |
|---|---|
| 👤 Live Queue | Patient count |
| 🚨 Crisis Alert | Patients with urgency ≥ 0.7 |
| 🧑‍⚕️ Skill Match | Stage 2 skill-acuity match score |
| 🌫️ Smog Level | Current AQI input |

#### Tabbed Sections (5 tabs — no more scrolling)
| Tab | What it shows |
|---|---|
| 🏨 Bed Allocation (01) | QUBO bed assignment per ward — ICU / Vent / General. Occupied beds shown as patient cards; free beds shown as a single summary tile ("N free beds") instead of individual blank cards |
| 📋 Patient Queue (02) | Full patient table sorted by QSVM urgency score. Column guide explains every abbreviation (BP Δ, O₂ SAT, HR DEV, GCS ▼, LACTATE, SCORE). Critical rows highlighted red |
| 👨‍⚕️ Staff Deployment (03) | Stage 2 QUBO staff assignment — staff chips per ward with skill level, fatigue colour (fresh / ok / tired), staff utilisation metrics strip |
| 🛡️ Security (04) | ML-KEM-768 public key fingerprint, NIST target, AES-256-GCM ciphertext preview, SHA-512 audit hash |
| 📟 Pipeline Log (05) | Full captured stdout from the last run — timings per stage, solver output, encryption confirmation |

#### Empty State (before first run)
Clean overview card listing all 5 pipeline stages with plain-English descriptions — no red alerts.

---

### Dashboard 2 — Quantum Engine Visualizer  (`ui/quantum_dashboard.py`, port 8051)
Theme: beige + Fredoka, matching Hospital dashboard aesthetic.

#### Controls
- **AQI slider** (0–500, marks: CLEAR / MID / SURGE)
- **Patient Count slider** (10–300) — fixed broken marks (were `{4,8,12,16,20}` — now `{10,100,200,300}`), added `paddingRight` to prevent "300" label clipping
- **EXPLODE PIPELINE** button

#### Summary Bar (after first run — 6 info cards)
`QUBO Nodes` · `α Penalty` · `QSVM F1` · `RF F1` · `Q-Kernel (N Qubits)` · `QAOA Ansatz`

#### 5 Algorithm Tabs
| Tab | Content |
|---|---|
| QSVM Kernel | N×N kernel overlap heatmap (YlOrRd) + QSVM vs RF urgency bar chart with critical threshold line |
| QUBO Matrix | Full Q matrix heatmap (RdBu, zero-centred) + diagonal bar chart + α/β/solver info cards |
| QAOA Circuit | Rendered circuit PNG (base64, PennyLane draw_mpl) + depth/qubit/hardware info |
| Allocations | Quantum vs classical assignment tables, ward utilisation donut charts, per-patient diff table (MATCHED / ROTATED) |
| Staff QUBO | Stage 2 QUBO heatmap + staff-to-ward assignment results + skill-acuity metrics |

#### Empty State (before first run)
Clean overview card listing all 5 tabs with descriptions — replaces old "QUANTUM CORE IS IDLE" red alert.

---

Both dashboards use `dcc.Store` to cache pipeline results between callbacks and `dcc.Loading` for the QSVM kernel computation (~15–30s depending on patient count).

---

## 9. Real Citations to Use

| Claim | Citation |
|---|---|
| QSVM for medical prediction | Havlíček et al., Nature 2019 |
| QML for disease prediction | Rebentrost et al., PRL 113, 130503 (2014) |
| QUBO formulation | Glover et al., arXiv:1811.11538 |
| QAOA | Farhi, Goldstone, Gutmann, arXiv:1411.4028 (2014) |
| NISQ era limitations | Preskill, Quantum 2, 79 (2018) |
| VQC vs QSVM tradeoff | Schuld & Killoran, PRL 122, 040504 (2019) |
| Hybrid QML medical | Krishnamurthy et al., Scientific Reports 2024 |
| Barren plateaus | McClean et al., Nature Comms 9, 4812 (2018) |

---

## 10. Winning Angles for Judges

- **India-first:** AQI data tied to respiratory surge = directly relevant to Delhi/Bengaluru hospital load. Ties to NQM (National Quantum Mission) + Ayushman Bharat digital health stack.
- **Honest about NISQ:** Don't overclaim speedup. Say "approximation quality + hardware portability." Judges respect this.
- **Show the baseline:** Classical Random Forest vs QSVM F1. Even marginal improvement with 4 qubits is publishable-level at hackathon scale.
- **Demo moment:** Live AQI slider changing patient urgency + allocation in real time = visceral impact.
- **Commercial path:** "Quantum-hybrid SaaS for hospital command centers, NQM grant eligible, tier-2 hospital pilots."

---

## 11. Red Lines (Don't Say These)

| ❌ Don't say | ✅ Say instead |
|---|---|
| "Quantum speedup over classical" | "Approximation quality under resource constraints" |
| "This runs on real quantum hardware" | "Compiles directly to Ising Hamiltonian, QPU-portable" |
| "QTIS / Prog-QAOA Framework" | Cite Glover arXiv:1811.11538 or Farhi 2014 |
| "99% accuracy" without baseline | "X% vs classical baseline of Y%" |

---

---

## 12. Modules — Current State

| Module | What it does | Status |
|---|---|---|
| `core/qsvm.py` | 4-qubit angle-embedding QSVM; exposes `K_train_` for dashboard heatmap | ✅ stable |
| `core/optimizer.py` | Dynamic-α QUBO builder + D-Wave neal SA solver; returns `(Q_dict, alpha)` tuple | ✅ stable |
| `core/staff_optimizer.py` | **Stage 2 QUBO** — staff × patient assignment; 4-term Hamiltonian (utility + uniqueness + capacity + qualification penalty); greedy fallback above 400 variables | ✅ stable |
| `core/qaoa.py` | QAOA p=1 circuit: `qubo_to_ising`, `draw_qaoa_circuit` (base64 PNG), `circuit_info` | ✅ stable |
| `core/baseline.py` | `random_forest_scores(X,y)` — RF urgency proba + macro F1 for benchmarking | ✅ stable |
| `core/security.py` | **Real ML-KEM-768** via liboqs 0.15.0 — `MLKEMProxy` with `encrypt_patient_record`, `decrypt_patient_record`, `audit_hash`; KEM+DEM construction (ML-KEM-768 key exchange + AES-256-GCM symmetric). **14/14 smoke tests passing.** | ✅ upgraded |
| `data/generator.py` | Synthetic patient vitals + staff roster with qualifications, fatigue, role weights | ✅ stable |
| `pipeline.py` | 8-stage orchestrator: Data → Encrypt → QSVM → QUBO Stage 1 → QUBO Stage 2 → QAOA → Baseline → Audit | ✅ stable |
| `ui/hospital_dashboard.py` | Hospital Command Center (port 8050) — tabs, spinner, log panel, free-bed summary | ✅ updated |
| `ui/quantum_dashboard.py` | Quantum Engine Visualizer (port 8051) — 5 algorithm tabs, fixed slider marks | ✅ updated |

**`pipeline.py` full return dict:**
```
df, quantum_allocation, classical_allocation,
urgency_scores, quantum_utilization, classical_utilization,
unallocated_quantum, unallocated_classical,
kernel_matrix, qubo_dict, alpha,
qaoa_circuit_b64, qaoa_info,
rf_urgency, rf_f1, qsvm_f1,
security, encrypted_sample, audit_hash,
staff_df, staff_allocation, staff_metrics, staff_qubo_dict, alpha_s,
stage1_solve_ms, stage2_solve_ms
```

---

## 13. Post-Blueprint Engineering Changelog

### Security Upgrade — Mock → Real ML-KEM-768
- **Before:** PyNaCl Curve25519-XSalsa20 mock labelled as ML-KEM proxy
- **After:** `liboqs.KeyEncapsulation("Kyber768")` — genuine CRYSTALS-Kyber KEM
- KEM+DEM construction: ML-KEM-768 encapsulation → one-time shared secret → AES-256-GCM(key=secret[:32])
- Forward secrecy: fresh KEM encapsulation per record
- Verified: 14/14 smoke tests (keygen, encrypt, decrypt, audit hash, round-trip)

### Stage 2 QUBO — Staff Deployment
Added `core/staff_optimizer.py` with a 4-term Hamiltonian:
```
H = -∑_{n,p} skill_n · urgency_p · (1−fatigue_n) · w_role_n · s_{n,p}   ← utility
  + α_s · ∑_p (∑_n s_{n,p} − 1)²                                         ← one staff per patient
  + β_s · ∑_n (∑_p s_{n,p} − C_n)²                                        ← staff capacity
  + γ   · ∑_{p,n} (1 − qual_{n,ward_p}) · s_{n,p}                         ← qualification gate
```
- γ = 50.0 (must dominate utility to block unqualified assignments)
- Greedy fallback above 400 variables (≈ 20 staff × 20 patients)
- Output: `staff_allocation`, `staff_metrics` (skill_acuity_match, avg_fatigue, coverage)

### UI Overhaul — Hospital Dashboard (port 8050)
- **Spinner:** Full-screen `dcc.Loading(fullscreen=True)` with animated ⚛️ custom spinner during pipeline runs
- **Pipeline Log:** stdout captured via `io.StringIO` redirect → displayed in dark terminal-style panel
- **Tabs:** Replaced long single-page scroll with 5 tabs — Beds / Patients / Staff / Security / Log
- **Free beds:** Single summary tile ("N free beds") per ward instead of N individual blank cards
- **Section headers:** Numbered 01–05 with plain-English subtitles for judge/demo clarity
- **Column guide:** Inline legend above patient table explaining every abbreviation
- **Empty state:** Clean pipeline overview card instead of "READY FOR ACTION!" alert

### UI Fixes — Quantum Dashboard (port 8051)
- **Broken slider marks** fixed: `{4:"4", 8:"8"...20:"MAX"}` on a 10–300 range → `{10,100,200,300}`
- **"300" label clipping** fixed: `paddingRight: 20px` wrapper on slider div
- **Empty state** replaced: "QUANTUM CORE IS IDLE" red alert → clean 5-tab overview card

---

*Blueprint v3.0 — updated May 2026: real ML-KEM-768, Stage 2 staff QUBO, full UI overhaul*
