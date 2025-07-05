import json
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp,Operator, state_fidelity, Statevector, partial_trace, entropy
import random
from scipy.optimize import minimize

def generate_GHZ_op():
    qc = QuantumCircuit(3)
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
    amp_000 = cos_d
    # |φ_A φ_B φ_C>
    phi_A = np.array([np.cos(alpha), np.sin(alpha)])
    phi_B = np.array([np.cos(beta), np.sin(beta)])
    phi_C = np.array([np.cos(gamma), np.sin(gamma)])
    # Tensor product
    phi_ABC = np.kron(np.kron(phi_A, phi_B), phi_C)
    amp_phi = sin_d * np.exp(1j * phi)
    # Full state vector
    state = np.zeros(8, dtype=complex)
    state[0] = amp_000
    psi_ = state + amp_phi * phi_ABC
    norm = np.linalg.norm(state)
    psi_ /= norm
    state += amp_phi * phi_ABC
    norm = np.linalg.norm(state)
    state /= norm
    # Create Statevector object
    psi = Statevector(psi_)
    qc = QuantumCircuit(3)
    qc.initialize(psi, [0, 1, 2])
    return qc
def generate_separable(unitaries):
    qc = QuantumCircuit(3)
    qc.unitary(Operator(unitaries[0]), [0], label="U1")
    qc.unitary(Operator(unitaries[1]), [1], label="U2")
    qc.unitary(Operator(unitaries[2]), [2], label="U3")
    return qc
def generate_bipartite_entangled(unitaries):
    qc = QuantumCircuit(3)
    qc.unitary(Operator(unitaries[0]), [0], label="U1")
    qc.h(1)
    qc.cx(1,2)
    U_ = np.kron(unitaries[1],unitaries[2])
    qc.unitary(Operator(U_), [1,2], label="U2⊗U3")
    return qc
def generate_W():
    a, b, c = np.random.rand(3)
    # Normalize

    norm = np.sqrt(a**2 + b**2 + c**2 )
    # We normalize the probabilities so the quantum state is normalized
    a, b, c = a/norm, b/norm, c/norm
    # Define state vector in computational basis order: |000>, |001>, ..., |111>
    state_vector = [0, a, b, 0, c, 0, 0, 0]
    # Initialize circuit
    qc = QuantumCircuit(3)
    qc.initialize(state_vector, [0, 1, 2])
    theta = np.random.uniform(0, 2 * np.pi)
    qc.global_phase=theta
    return qc
def read_unitaries(file_path):
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    loaded_matrices = [np.array(data["real"]) + 1j * np.array(data["imag"]) for data in loaded_data]
    return loaded_matrices
def cost_function(parameters,ansatz,sampler,backend,dict,q):
    isa_ansatz = transpile(ansatz, backend=backend)
    isa_ansatz = isa_ansatz.assign_parameters(parameters)
    counts = sampler.run(([isa_ansatz])).result()[0].data.meas.get_counts()
    fid = counts.get("000",0)/sum(counts.values())
    dict["state"+str(q)]["fidelity"].append(-fid)
    dict["state"+str(q)]["angles"].append([float(p) for p in parameters])
    dict["state"+str(q)]["counts"].append(counts)
    return -fid
def main():
    unitaries = read_unitaries(r"C:\Users\aquim\TFM\random_unitaries_2q.json")
    backend = AerSimulator()
    states_to_classify = 10
    sampler = Sampler(mode=backend)
    ansatz_sep = QuantumCircuit(3)
    params = ParameterVector('θ',9)
    ansatz_sep.ry(params[0],0)
    ansatz_sep.rx(params[1],0)
    ansatz_sep.ry(params[2],1)
    ansatz_sep.rx(params[3],1)
    ansatz_sep.rz(params[4],0)
    ansatz_sep.rz(params[5],1)
    ansatz_sep.ry(params[6],2)
    ansatz_sep.rx(params[7],2)
    ansatz_sep.rz(params[8],2)
    dict = {}
    computed_fidelity = []
    for q in range(int(states_to_classify)):
        print("State", q)
        dict["state"+str(q)] = {}
        dict["state"+str(q)]["fidelity"]=[]
        dict["state"+str(q)]["angles"]=[]
        dict["state"+str(q)]["counts"]=[]
        x0 = np.array([0]*9)
        rand1 = np.random.rand()
        if rand1<0.2:
            dict["state"+str(q)]["type"] = "GHZ"
            qc = generate_custom_GHZ()
        elif rand1 >= 0.2 and rand1<0.4:
            dict["state"+str(q)]["type"] = "separable"
            qc = generate_separable(unitaries[q:q+3])
        elif rand1 >= 0.4 and rand1<0.6:
            dict["state"+str(q)]["type"] = "bipartite_entangled"
            qc = generate_bipartite_entangled(unitaries[q:q+3])
        elif rand1 >= 0.6 and rand1<0.8:
            dict["state"+str(q)]["type"] = "W"
            qc = generate_W()
        else:
            dict["state"+str(q)]["type"] = "GHZ_Local_Op"
            print("GHZ Local Op")
            qc = generate_GHZ_op()
        combined_qc = QuantumCircuit(3)
        combined_qc.compose(qc, inplace=True)
        combined_qc.barrier()
        combined_qc.compose(ansatz_sep.inverse(), inplace=True)
        combined_qc.measure_all()
        res = minimize(cost_function, x0, args=(combined_qc,sampler,backend,dict,q), method='COBYLA', tol=1e-3)
        computed_fidelity.append(-float(res["fun"]))
    

if __name__ == "__main__":
    main()
