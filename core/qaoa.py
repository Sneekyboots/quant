"""
core/qaoa.py
QAOA (Quantum Approximate Optimization Algorithm) circuit mock.

Maps the QUBO objective to a p=1 QAOA circuit using PennyLane.
Demonstrates QPU portability: the same QUBO that neal solves via SA
compiles directly to an Ising Hamiltonian executable on real quantum hardware.

Demo circuit : 2 patients × 3 resources = 6 qubits  (p = 1 layer)
Full circuit : 8 patients × 3 resources = 24 qubits — QPU target

QUBO → Ising mapping
─────────────────────
  x_i = (1 − z_i) / 2,   z_i ∈ {−1, +1}

  Diagonal   Q_{ii} :  h_i  += Q_{ii} / 2
  Off-diag   Q_{ij} :  J_ij += Q_{ij} / 4
                        h_i  += Q_{ij} / 4
                        h_j  += Q_{ij} / 4

QAOA circuit structure  (p = 1)
────────────────────────────────
  |+⟩^n  →  U_C(γ) [RZ + CNOT-RZ-CNOT blocks]  →  U_M(β) [RX gates]  →  ⟨Z_i⟩

Citations
─────────
  Farhi, Goldstone, Gutmann — arXiv:1411.4028 (2014)
  Ising mapping: Glover et al. — arXiv:1811.11538
"""

import io
import base64
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.optimizer import BETA

# ── Demo circuit dimensions ───────────────────────────────────────────────────

DEMO_PATIENTS  = 2
DEMO_RESOURCES = 3
DEMO_QUBITS    = DEMO_PATIENTS * DEMO_RESOURCES   # 6 qubits
QAOA_P         = 1                                  # layers

_DEMO_VARS = [f"p{p}_r{r}"
              for p in range(DEMO_PATIENTS)
              for r in range(DEMO_RESOURCES)]
_VAR_IDX   = {v: i for i, v in enumerate(_DEMO_VARS)}


# ── QUBO → Ising conversion ───────────────────────────────────────────────────

def qubo_to_ising(Q: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the 6-variable demo QUBO subset to Ising coefficients (h, J).
    Variables outside _DEMO_VARS are silently ignored.

    Returns
    -------
    h : ndarray shape (6,)   — linear Ising coefficients (single-qubit RZ)
    J : ndarray shape (6,6)  — quadratic coefficients (upper triangle, ZZZ gates)
    """
    h = np.zeros(DEMO_QUBITS)
    J = np.zeros((DEMO_QUBITS, DEMO_QUBITS))

    for (v1, v2), coeff in Q.items():
        if v1 not in _VAR_IDX or v2 not in _VAR_IDX:
            continue
        i, j = _VAR_IDX[v1], _VAR_IDX[v2]
        if i == j:                          # diagonal → linear h term
            h[i] += coeff / 2.0
        else:                               # off-diagonal → ZZ coupling + linear corrections
            a, b = min(i, j), max(i, j)
            J[a, b] += coeff / 4.0
            h[a]    += coeff / 4.0
            h[b]    += coeff / 4.0

    return h, J


# ── QAOA QNode ────────────────────────────────────────────────────────────────

def build_qaoa_qnode(h: np.ndarray, J: np.ndarray) -> qml.QNode:
    """
    Build a p=1 QAOA QNode for Ising Hamiltonian (h, J).

      |+⟩^6  →  U_C(γ) [cost layer]  →  U_M(β) [mixer layer]  →  ⟨Z_i⟩

    The cost layer applies:
      - RZ(2γ h_i)       for non-zero linear terms
      - CNOT·RZ(2γ J_ij)·CNOT  for non-zero ZZ couplings

    The mixer layer applies RX(2β) on every qubit (standard QAOA mixer).
    """
    dev = qml.device("default.qubit", wires=DEMO_QUBITS)

    @qml.qnode(dev)
    def circuit(gamma: float, beta: float):
        # ── Initial state |+⟩^n ──────────────────────────────────────────
        for i in range(DEMO_QUBITS):
            qml.Hadamard(wires=i)

        # ── Cost Hamiltonian  U_C(γ) = exp(−iγ H_C) ─────────────────────
        for i in range(DEMO_QUBITS):
            if abs(h[i]) > 1e-9:
                qml.RZ(2.0 * gamma * float(h[i]), wires=i)
        for i in range(DEMO_QUBITS):
            for j in range(i + 1, DEMO_QUBITS):
                if abs(J[i, j]) > 1e-9:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * gamma * float(J[i, j]), wires=j)
                    qml.CNOT(wires=[i, j])

        # ── Mixer  U_M(β) = exp(−iβ H_B) = Π_i RX(2β) ──────────────────
        for i in range(DEMO_QUBITS):
            qml.RX(2.0 * beta, wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(DEMO_QUBITS)]

    return circuit


# ── Circuit visualisation ─────────────────────────────────────────────────────

def draw_qaoa_circuit(Q_full: dict,
                      gamma: float = 0.5,
                      beta:  float = 0.3) -> str:
    """
    Render the 6-qubit p=1 QAOA circuit as a base64 PNG data URI.
    Restricts Q_full to the 6 demo variables before drawing.

    Returns  'data:image/png;base64,...'
    """
    Q_demo = {k: v for k, v in Q_full.items()
              if k[0] in _VAR_IDX and k[1] in _VAR_IDX}
    h, J = qubo_to_ising(Q_demo)
    circuit = build_qaoa_qnode(h, J)

    fig, ax = qml.draw_mpl(circuit, decimals=2, style="pennylane")(gamma, beta)

    # Dark background
    bg = "#0d1117"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#c9d1d9")

    fig.suptitle(
        f"QAOA  p={QAOA_P}  ·  {DEMO_QUBITS} qubits  "
        f"({DEMO_PATIENTS} patients × {DEMO_RESOURCES} resources)  "
        f"·  γ={gamma}  β={beta}  ·  Farhi et al. arXiv:1411.4028",
        color="#8b5cf6", fontsize=9, y=1.02,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# ── Info panel metadata ───────────────────────────────────────────────────────

def circuit_info(n_patients: int, n_resources: int, alpha: float) -> dict:
    """
    Returns metadata about the full-scale (non-demo) QAOA circuit.
    Shown in the quantum engine dashboard info panel.
    """
    n_full = n_patients * n_resources
    return {
        "demo_qubits":    DEMO_QUBITS,
        "full_qubits":    n_full,
        "layers_p":       QAOA_P,
        "gamma":          0.5,
        "beta":           0.3,
        "alpha_computed": round(alpha, 2),
        "beta_penalty":   BETA,
        "cost_gates":     f"≤ {n_full + n_full*(n_full-1)//2} RZ/RZZ gates",
        "mixer_gates":    f"{n_full} × RX gates",
        "ising_ready":    True,
        "hardware_target": "D-Wave (Ising-native)  ·  IonQ  ·  IBM (CNOT decomp)",
        "citation":       "Farhi, Goldstone, Gutmann — arXiv:1411.4028 (2014)",
    }
