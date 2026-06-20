import pandas as pd
import numpy as np

from .CausalModels import Causal_Models


class TransitionModel:
    def __init__(self, model):
        self.model = model
        self.N = len(model)
        self.name = model.name

    def sample_output(self, model, n_photons: int):
        rows = []
        n_survive = n_photons
        for i in range(len(model)):
            p_survive, _ = self.get_output_probabilities(len(model), i)
            n_next = np.random.binomial(n_survive, float(p_survive))
            n_lost = n_survive - n_next
            rows.append({"N path |0>": n_next, "N path |1>": n_lost})
            n_survive = n_next
        return pd.DataFrame(rows)

    def __len__(self):
        return self.N


class ExactTransitionModel(TransitionModel):
    def __init__(self, model):
        super().__init__(model)
        self.title = 'Exact'
    
    def get_output_probabilities(self, N, j):
        epsilon = 1e-19
        p_0 = (N-j-1)/(N-j+epsilon)
        p_1 = 1/(N-j+epsilon)
        return p_0, p_1


class QuantumTransitionModel(TransitionModel):
    """
    Optics matrix path 1 and 2 are cols 0,2 and 1,3
    """
    def __init__(self, model):
        super().__init__(model)
        self.title = 'Quantum'
        
    def get_output_probabilities(self, N, j):
        v = self.model.U @ self.model.states[j]
        path2, path1 = v[[0, 2]], v[[1, 3]]   # NTU: path2 at 0,2 — path1 at 1,3
        return float(np.linalg.norm(path2) ** 2), float(np.linalg.norm(path1) ** 2)

def print_model_probabilities():
    for model in Causal_Models.values():
        print(f'\n{model.name}')
        rows = []
        for i, state in enumerate(model[:-1]):
            v = model.U @ state
            v13, v24 = v[[0, 2]], v[[1, 3]]
            α, β = np.linalg.norm(v13), np.linalg.norm(v24)
            u13, u24 = v13 / α, v24 / β
            s_next = model[i + 1][[0, 2]]
            ip1 = abs(np.vdot(u13, s_next))
            ip2 = abs(np.vdot(u24, model.s0[[0, 2]]))
            rows.append({
                "State": f"s{i}",
                "α²": round(α ** 2, 2),
                "β²": round(β ** 2, 2),
                "⟨u13|sj+1⟩": round(ip1, 2),
                "⟨u24|s0⟩": round(ip2, 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))
