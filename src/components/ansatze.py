import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector, Operator

#######  QUANTUM CIRCUITS GENERATIION OF SEPARABLE AND ENTANGLED STATES TO BE TESTED #######
def generate_GHZ():

    alpha_beta_gamma_bounds = (0, np.pi / 2)
    delta_bounds = (0, np.pi / 4)
    phi_bounds = (0, 2 * np.pi)
    alpha = np.random.uniform(*alpha_beta_gamma_bounds)
    beta = np.random.uniform(*alpha_beta_gamma_bounds)
    gamma = np.random.uniform(*alpha_beta_gamma_bounds)
    delta = np.random.uniform(*delta_bounds)
    phi = np.random.uniform(*phi_bounds)
    # Compute components
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    cos_phi = np.cos(phi)
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    sin_gamma = np.sin(gamma)
    # Component 000
    amp_000 = cos_d

    # |φ_A φ_B φ_C>
    phi_A = np.array([np.cos(alpha), np.sin(alpha)])
    phi_B = np.array([np.cos(beta), np.sin(beta)])
    phi_C = np.array([np.cos(gamma), np.sin(gamma)])

    # Tensor product
    phi_ABC = np.kron(np.kron(phi_A, phi_B), phi_C)
    amp_phi = sin_d * np.exp(1j * phi)

    # Normalization constant, using this normalization there is some error as it doesn't return 1 for the normalized state, thus I will normalize the state using numpy normalization
    K_GHZ = 1 / (1 + cos_d * sin_d * cos_alpha * cos_beta * cos_gamma * cos_phi)
    sqrt_K = np.sqrt(K_GHZ)

    # Full state vector
    state = np.zeros(8, dtype=complex)
    state[0] = amp_000
    state += amp_phi * phi_ABC
    norm = np.linalg.norm(state)
    state /= norm
    # Create Statevector object
    psi = Statevector(state)
    qc = QuantumCircuit(3)
    qc.initialize(psi, [0, 1, 2])
    return qc

def generate_W():
    a, b, c, d = np.random.rand(4)

    # Normalize
    norm = np.sqrt(a**2 + b**2 + c**2 + d**2)
    # We normalize the probabilities so the quantum state is normalized
    a, b, c, d = a/norm, b/norm, c/norm, d/norm

    # Define state vector in computational basis order: |000>, |001>, ..., |111>
    state_vector = [d, a, b, 0, c, 0, 0, 0]

    # Initialize circuit
    qc = QuantumCircuit(3)
    qc.initialize(state_vector, [0, 1, 2])
    return qc

def generate_bipartite(training):
    qc = QuantumCircuit(3)
    qc.unitary(Operator(training[0]), [0], label="U1")
    qc.h(1)
    qc.cx(1,2)
    U_ = np.kron(training[1],training[2])
    qc.unitary(Operator(U_), [1,2], label="U2⊗U3")
    return qc

def generate_sep(training):
    qc = QuantumCircuit(3)
    qc.unitary(Operator(training[0]), [0], label="U1")
    qc.unitary(Operator(training[1]), [1], label="U2")
    qc.unitary(Operator(training[2]), [2], label="U3")
    return qc


#######  QUANTUM CIRCUIT GENERATOR OF FULLY SEPARABLE STATE TO EXPLORE THE HILBERT SPACE #######

def generate_3qubit_control():
    qc = QuantumCircuit(3)
    params = ParameterVector("θ", length=9)
    qc.rx(params[0], 0)
    qc.rx(params[1], 1)
    qc.rx(params[2], 2)
    qc.ry(params[3], 0)
    qc.ry(params[4], 1)
    qc.ry(params[5], 2)
    qc.rz(params[6], 0)
    qc.rz(params[7], 1)
    qc.rz(params[8], 2)
    return qc