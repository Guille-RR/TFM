import json
from joblib import Parallel, delayed
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp,Operator, state_fidelity, Statevector, partial_trace
from scipy.optimize import minimize, differential_evolution
import time
import os
def generate_Bell_state(unitaries):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    U_ = np.kron(unitaries[0], unitaries[1])
    qc.unitary(Operator(U_), [0,1], label="U1⊗U2")
    qc.barrier()
    return qc
def generate_separable(unitaries):
    qc = QuantumCircuit(2)
    qc.unitary(Operator(unitaries[0]), [0], label="U1")
    qc.unitary(Operator(unitaries[1]), [1], label="U2")
    qc.barrier()
    return qc
def generate_nonmax_entangles():
    # Non maximally entangled state
    r = np.random.rand(4) # We create a vector of 4 random amplitudes
    r_norm = r/np.linalg.norm(r) # We normalize the vector
    phi = np.random.rand(4) * 2 * np.pi # We create a random phase
    target_state = np.array([
        r_norm[0] * np.exp(1j * phi[0]),
        r_norm[1] * np.exp(1j * phi[1]),
        r_norm[2] * np.exp(1j * phi[2]),
        r_norm[3] * np.exp(1j * phi[3])
    ])
    target_state = Statevector(target_state)
    qc = QuantumCircuit(2)
    qc.initialize(target_state.data, [0, 1]) # We initialize the state
    qc.barrier()
    return qc
def read_unitaries(file_path):
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    loaded_matrices = [np.array(data["real"]) + 1j * np.array(data["imag"]) for data in loaded_data]
    return loaded_matrices
def cost_function(parameters, ansatz, sampler, backend, history, q):
    isa_ansatz = transpile(ansatz, backend=backend)
    isa_ansatz = isa_ansatz.assign_parameters(parameters)
    counts = sampler.run(([isa_ansatz]),shots = 5000).result()[0].data.meas.get_counts()
    counts_000 = counts.get('00', 0)
    fid = counts_000 / sum(counts.values())
    FS_dis = np.sqrt(1 - abs(fid))  # Fidelity score distance
    history["fidelity"].append(abs(fid))
    history["angles"].append([float(p) for p in parameters])
    history["counts"].append(counts)
    history["FS_dis"].append(FS_dis)
    return FS_dis
def paralellize_state_processing(q, unitaries, ansatz_sep, sampler, backend):
    x0 = np.array([0]*6)
    rand1 = np.random.rand()
    if q % 3==0:
        state_type = "separable"
        qc = generate_separable(unitaries[q:q+2])
    elif q % 3==1:
        state_type = "bell_state"
        qc = generate_Bell_state(unitaries[q:q+2])
    elif q % 3==2:
        state_type = "non_max_entangled"
        qc = generate_nonmax_entangles()
    combined_qc = QuantumCircuit(2)
    combined_qc.compose(qc, inplace=True)
    combined_qc.barrier()
    combined_qc.compose(ansatz_sep.inverse(), inplace=True)
    combined_qc.measure_all()

    history = {"fidelity": [], "angles": [], "counts": [], "FS_dis": []}
    res = minimize(cost_function, x0, args=(combined_qc, sampler, backend, history, q), method='COBYLA', tol=1e-5)
    final_FS = float(res["fun"])
    print(f"State {q} processed")
    return q, state_type, final_FS, history
def de_state_processing(q, unitaries, ansatz_sep, sampler, backend):
    x0 = np.array([0]*2)
    rand1 = np.random.rand()
    if rand1 < 1/3:
        state_type = "separable"
        qc = generate_separable(unitaries[q:q+2])
    elif rand1>=1/2 and rand1 < 2/3:
        state_type = "bell_state"
        qc = generate_Bell_state(unitaries[q:q+2])
    else:
        state_type = "non_max_entangled"
        qc = generate_nonmax_entangles()
    combined_qc = QuantumCircuit(2)
    combined_qc.compose(qc, inplace=True)
    combined_qc.barrier()
    combined_qc.compose(ansatz_sep.inverse(), inplace=True)
    combined_qc.measure_all()
    bounds = [(-np.pi, np.pi)] * 6  # Bounds for each parameter
    history = {"fidelity": [], "angles": [], "counts": []}
    res = differential_evolution(cost_function, bounds, args=(combined_qc, sampler, backend, history, q),strategy = 'best1exp',popsize = 3,tol=0.000001 ,maxiter=10000,polish=True,init = 'halton')
    final_fidelity = float(res["fun"])

    return q, state_type, final_fidelity, history
def main():
    print("Starting the SGM test for 2-qubit states...")
    time1 = time.time()
    unitaries = read_unitaries(r"random_unitaries_2q.json")
    backend = AerSimulator()
    sampler = Sampler(mode=backend)
    states_to_classify = 30
    optimizer = 'COBYLA'

    ansatz_max_ent = QuantumCircuit(2)
    params = ParameterVector('θ', 6)
    ansatz_max_ent.h( 0)
    ansatz_max_ent.cx(0, 1)
    # Example local operations that do not decrease the entanglement
    ansatz_max_ent.rx(params[0], 0)
    ansatz_max_ent.ry(params[1], 0)
    ansatz_max_ent.rz(params[2], 0)
    ansatz_max_ent.rx(params[3], 1)
    ansatz_max_ent.ry(params[4], 1)
    ansatz_max_ent.rz(params[5], 1)
    if optimizer == 'COBYLA':
        #results = Parallel(n_jobs=-1)(
        #    delayed(paralellize_state_processing)(q, unitaries, ansatz_max_ent, sampler, backend)
        #    for q in range(states_to_classify)
        #)
        results = [paralellize_state_processing(q, unitaries, ansatz_max_ent, sampler, backend) for q in range(states_to_classify)]
    elif optimizer == 'DE':
        results = Parallel(n_jobs=-1)(delayed(de_state_processing)(q, unitaries, ansatz_max_ent, sampler, backend)
            for q in range(states_to_classify))
    else:
        raise ValueError("Unsupported optimizer. Use 'COBYLA' or 'DE'.")
    dictionary = {}
    computed_FS = []
    for q, state_type, final_FS, history in results:
        state_key = f"state{q}"
        dictionary[state_key] = {
            "state_type": state_type,
            "fidelity": history["fidelity"],
            "angles": history["angles"],
            "counts": history["counts"]
        }
        computed_FS.append(final_FS)

    dictionary['results'] = computed_FS
    dictionary['time'] = time.time() - time1
    if not os.path.exists('2_qubit/'):
        os.makedirs('2_qubit/')
    else:
        print("Directory already exists")
    with open(f'2_qubit/sgm_test_states_2.json', 'w') as f:
        json.dump(dictionary, f, indent=4)

if __name__ == "__main__":
    main()
