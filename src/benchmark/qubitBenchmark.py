import numpy as np
from scipy.optimize import differential_evolution, minimize
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp,Operator, state_fidelity, Statevector, partial_trace, entropy
from src.components.ansatze import generate_GHZ, generate_W, generate_bipartite, generate_sep, generate_3qubit_control
import json
import os
from qiskit import transpile
class Benchmark_3qubit:
    def __init__(self,param_dict):
        self.backend_name = param_dict['backend']
        self.opt_path = param_dict['opt_path']
        self.training = None
        self.experiment_type = param_dict['experiment_type']
        self.test_size = param_dict['test_size']
        self.optimizer = param_dict['optimizer']['name']
        self.maxiter = param_dict['optimizer']['maxiter']
        self.savepath = param_dict['savepath']
        if self.maxiter == None:
            self.maxiter = 100
        self.tol = param_dict['optimizer']['tol']
        if self.tol == None:
            self.tol = 0.01
        self.strategy = param_dict['optimizer']['strategy']
        if self.strategy == None:
            self.strategy = 'best1exp'
        self.popsize = param_dict['optimizer']['popsize']
        if self.popsize == None:
            self.popsize = 1
        self.bound = param_dict['bound']
        if self.bound == None:
            self.bound = 0.1
        self.i=0
        self.cost_evol = {}
        self.cost_evol_minimize = {}

    def read_data(self):
        # Read the data from the JSON file
        with open(self.opt_path, 'r') as f:
            loaded_data = json.load(f)
        loaded_matrices = [ np.array(data["real"]) + 1j * np.array(data["imag"])for data in loaded_data]
        return loaded_matrices
    
    def custom_callback(self,intermediate_result):
        self.cost_evol[f'state_{self.i}']['iter'].append(len(self.cost_evol['cost'])+1)
        self.cost_evol[f'state_{self.i}']['cost'].append(intermediate_result['fun'])
        self.cost_evol[f'state_{self.i}']['params'].append(intermediate_result['x'])
        if 'population' in intermediate_result:
            self.cost_evol[f'state_{self.i}']['populations'].append(intermediate_result['population'])
    
    def backend(self):
        if self.backend_name == 'aer_simulator':
            backend = Aer.get_backend('aer_simulator')
        else:
            backend = Aer.get_backend(self.backend_name)
        return backend

    def _cost_fun(self,params,ansatz1,ansatz2,backend,i):
        ansatz2 = ansatz2.assign_parameters(params)
        isa_ansatz = transpile(ansatz1,backend=backend)
        isa_ansatz2 = transpile(ansatz2,backend=backend)
        sv1 = Statevector(isa_ansatz)
        sv2 = Statevector(isa_ansatz2)
        fid = state_fidelity(sv1,sv2)
        self.cost_evol_minimize[f'state_{i}']['params'].append(list(params))
        self.cost_evol_minimize[f'state_{i}']['iter'].append(len(self.cost_evol_minimize[f'state_{i}']['cost'])+1)
        self.cost_evol_minimize[f'state_{i}']['cost'].append(-float(fid))
        return -fid    

    def run_experiment(self):
        self.training = self.read_data()
        backend = self.backend()
        if self.experiment_type == 'Benchmark':
            print("Running Benchmark for entangled vs separable states")
            real_E = []
            opt_F = []
            for i in range(self.test_size):
                self.i = i 
                l=i
                training = self.training[l:l+3]
                rand_num = np.random.rand()
                if rand_num <= 1/4:
                    name = 'GHZ'
                    real_E.append(1)
                    ansatz1 = generate_GHZ()
                elif rand_num > 1/4 and rand_num <= 2/4:
                    name = 'W'
                    real_E.append(1)
                    ansatz1 = generate_W()
                elif rand_num > 2/4 and rand_num <= 3/4:
                    name = 'bipartite'
                    real_E.append(1)
                    ansatz1 = generate_bipartite(training)
                elif rand_num > 3/4 and rand_num <= 1:
                    name = 'separable'
                    real_E.append(0)
                    ansatz1 = generate_sep(training)
                self.cost_evol[f'state_{i}'] = {'iter': [], 'cost': [], 'params': [], 'populations': []}
                self.cost_evol_minimize[f'state_{i}'] = {'iter': [], 'cost': [], 'params': []}
                ansatz2 = generate_3qubit_control()
                l=l+3
                if self.optimizer == 'differential_evolution':
                    bounds = [(0, 2 * np.pi) for _ in range(9)]
                    result = differential_evolution(self._cost_fun,
                                                        bounds,
                                                        args=(ansatz1, ansatz2,backend,i),
                                                        maxiter=self.maxiter,
                                                        tol=self.tol,
                                                        callback=self.custom_callback,
                                                        strategy=self.strategy,
                                                        popsize=self.popsize)
                    opt_F.append(-result.fun)
                else:
                    result = minimize(self._cost_fun,
                                        x0=np.random.rand(9),
                                        args=(ansatz1, ansatz2,backend,i),
                                        method=self.optimizer,
                                        options={'maxiter': self.maxiter},
                                        tol=self.tol)
                    opt_F.append(-result.fun)
            E = [(1 - p**2) for p in opt_F]
            for i in range(len(E)):
                # I set a bound at 0.1 for classification of separable and entangled
                if E[i] < self.bound:
                    E[i]=0
                elif E[i]>=self.bound:
                    E[i]=1
            q= 0
            for z in range(len(real_E)):
                if E[z]==real_E[z]:
                    q+=1
            result_dict = {'real_E': real_E, 'E': E, 'opt_F': opt_F}
            print(f'The method`s accuaracy is {(q/len(real_E))*100}%')
            print('Benchmark completed')
        elif self.experiment_type == 'Entanglement':
            # Test to only calculate the geometric entanglement measure of the entangled states
            print("Running Benchmark for entangled states")
            opt_F_GHZ = []
            for i in range(self.test_size):
                self.i = i
                name = 'GHZ'
                ansatz1 = generate_GHZ()
                ansatz2 = generate_3qubit_control()
                self.cost_evol[f'state_{i}'] = {'iter': [], 'cost': [], 'params': [], 'populations': [], 'state': name }
                if self.optimizer == 'differential_evolution':
                    bounds = [(0, 2 * np.pi) for _ in range(9)]
                    result = differential_evolution(self._cost_fun,
                                                        bounds,
                                                        args=(ansatz1, ansatz2,backend),
                                                        maxiter=self.maxiter,
                                                        tol=self.tol,
                                                        callback=self.custom_callback,
                                                        strategy=self.strategy,
                                                        popsize=self.popsize)
                    opt_F_GHZ.append(-result.fun)
                else:
                    result = minimize(self._cost_fun,
                                        x0=np.random.rand(9),
                                        args=(ansatz1, ansatz2,backend),
                                        method=self.optimizer,
                                        options={'maxiter': self.maxiter},
                                        tol=self.tol,
                                        callback=self.custom_callback)
                    opt_F_GHZ.append(-result.fun)
            E_GHZ = [float((1 - p**2)) for p in opt_F_GHZ]
            opt_F_W = []
            for i in range(self.test_size):
                self.i = i
                name = 'W'
                ansatz1 = generate_W()
                ansatz2 = generate_3qubit_control()
                self.cost_evol[f'state_{i}'] = {'iter': [], 'cost': [], 'params': [], 'populations': [],'state':name}
                if self.optimizer == 'differential_evolution':
                    bounds = [(0, 2 * np.pi) for _ in range(9)]
                    result = differential_evolution(self._cost_fun,
                                                        bounds,
                                                        args=(ansatz1, ansatz2,backend),
                                                        maxiter=self.maxiter,
                                                        tol=self.tol,
                                                        callback=self.custom_callback,
                                                        strategy=self.strategy,
                                                        popsize=self.popsize)
                    opt_F_W.append(-result.fun)
                else:
                    result = minimize(self._cost_fun,
                                        x0=np.random.rand(9),
                                        args=(ansatz1, ansatz2,backend),
                                        method=self.optimizer,
                                        options={'maxiter': self.maxiter},
                                        tol=self.tol,
                                        callback=self.custom_callback)
                    opt_F_W.append(-result.fun)
            E_W = [float((1 - p**2)) for p in opt_F_W]
            opt_F_bipartite = []
            for i in range(self.test_size):
                self.i = i
                name = 'Bipartite'
                ansatz1 = generate_bipartite()
                ansatz2 = generate_3qubit_control()
                self.cost_evol[f'state_{i}'] = {'iter': [], 'cost': [], 'params': [], 'populations': [],'state':name}
                if self.optimizer == 'differential_evolution':
                    bounds = [(0, 2 * np.pi) for _ in range(9)]
                    result = differential_evolution(self._cost_fun,
                                                        bounds,
                                                        args=(ansatz1, ansatz2,backend),
                                                        maxiter=self.maxiter,
                                                        tol=self.tol,
                                                        callback=self.custom_callback,
                                                        strategy=self.strategy,
                                                        popsize=self.popsize)
                    opt_F_bipartite.append(-result.fun)
                else:
                    result = minimize(self._cost_fun,
                                        x0=np.random.rand(9),
                                        args=(ansatz1, ansatz2,backend),
                                        method=self.optimizer,
                                        options={'maxiter': self.maxiter},
                                        tol=self.tol,
                                        callback=self.custom_callback)
                    opt_F_bipartite.append(-result.fun)

            E_bipartite = [float((1 - p**2)) for p in opt_F_bipartite]
            E_dict = {'GHZ': E_GHZ, 'W': E_W, 'Bipartite': E_bipartite}

            with open(self.savepath +'entanglement_measure.json', 'w') as f:
                json.dump(E_dict, f)
            print('Entanglement measure completed')
    
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        if self.optimizer == 'differential_evolution':
            with open(self.savepath +'cost_evolution.json', 'w') as f:
                json.dump(self.cost_evol, f)
        else:
            with open(self.savepath +'cost_evolution.json', 'w') as f:
                json.dump(self.cost_evol_minimize, f)
        with open(self.savepath +'result.json', 'w') as f:
            json.dump(result_dict, f)
        print('Benchmark completed')