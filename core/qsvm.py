"""
core/qsvm.py
Quantum Support Vector Machine for patient urgency scoring.

Quantum kernel: K(x_i, x_j) = |<Φ(x_i)|Φ(x_j)>|²
Angle embedding via RY rotations → overlap measured as P(|0000>).

Citation: Havlíček et al., Nature 567, 209–212 (2019)
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

NUM_QUBITS = 4
dev = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(dev)
def _kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Encodes x1 and x2 into Hilbert space via angle embedding.
    Returns probability distribution over all 2^4 basis states.
    The |0000> probability = kernel value K(x1, x2).
    """
    for i in range(NUM_QUBITS):
        qml.RY(x1[i] * np.pi, wires=i)          # embed x1
    for i in reversed(range(NUM_QUBITS)):
        qml.RY(-x2[i] * np.pi, wires=i)         # invert x2 embedding
    return qml.probs(wires=range(NUM_QUBITS))


def compute_kernel_matrix(X: np.ndarray) -> np.ndarray:
    """
    Builds the full symmetric NxN Gram matrix.
    K[i,j] = P(|0000>) from kernel circuit on (x_i, x_j).
    """
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            val = float(_kernel_circuit(X[i], X[j])[0])
            K[i, j] = val
            K[j, i] = val
    return K


class QuantumSVM:
    """
    Wrapper around sklearn SVC using a precomputed quantum kernel.
    Outputs calibrated urgency probabilities P_i ∈ [0, 1].
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.svc = SVC(kernel="precomputed", probability=True, random_state=42)
        self.X_train_scaled = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_scaled = self.scaler.fit_transform(X)
        K_train = compute_kernel_matrix(self.X_train_scaled)
        self.K_train_ = K_train   # stored for kernel heatmap in dashboard
        self.svc.fit(K_train, y)

    def predict_urgency(self, X: np.ndarray) -> np.ndarray:
        """
        Returns urgency score (P of critical status) for each patient.
        For demo: we use the training set itself as the test set.
        """
        X_scaled = self.scaler.transform(X)
        # Kernel between test and train points
        N_test = X_scaled.shape[0]
        N_train = self.X_train_scaled.shape[0]
        K_test = np.zeros((N_test, N_train))
        for i in range(N_test):
            for j in range(N_train):
                K_test[i, j] = float(_kernel_circuit(X_scaled[i], self.X_train_scaled[j])[0])
        return self.svc.predict_proba(K_test)[:, 1]
