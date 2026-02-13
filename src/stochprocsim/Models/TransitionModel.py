import sympy as sp
import pandas as pd 
import numpy as np

from .CausalModels import Causal_Models, flipped

α, β = sp.symbols('α β')

class TransitionModel():
    def __init__(self, model):
        self.model = model
        self.N = len(model)
        self.name = model.name

    def sample_output(self, model, NSamples_α, NSamples_β = 0):
        rows = []

        for i in range(len(model)):
            α_val, β_val = self.get_output_probabilities(len(model), i)
            
            # NSamples_β = np.random.poisson(β_val * NSamples_α)
            # NSamples_α = np.random.poisson(α_val * NSamples_α)

            NSurvive = np.random.binomial(NSamples_α, α_val)
            NLost = NSamples_α - NSurvive

            NSamples_α = NSurvive
            NSamples_β = NLost

            rows.append({
                "N path |0>": NSamples_α,
                "N path |1>": NSamples_β
            })
        return pd.DataFrame(rows)

    def __len__(self):
        return self.N

class ClassicalTransitionModel(TransitionModel):
    def __init__(self):
        pass

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
        v = self.model.U@self.model.states[j]
        if flipped:
            path2 = sp.Matrix([[v[2]],[v[3]]])
            path1 = sp.Matrix([[v[0]],[v[1]]])
        else:
            path2 = sp.Matrix([[v[0]],[v[2]]])
            path1 = sp.Matrix([[v[1]],[v[3]]])
        α, β = path2.norm(), path1.norm()
        return α**2, β**2

def print_model_probabilities():
    for model in Causal_Models.values(): 
        print(f'\n{model.name}')
        N = len(model)
        rows = []
        for i, state in enumerate(model[:-1]):
            v = model.U@state
            print(f'V norm: {v.norm()}')
            # v13 = sp.Matrix([[v[0]],[v[1]]])
            # v24 = sp.Matrix([[v[2]],[v[3]]])
            v13 = sp.Matrix([[v[0]],[v[2]]])
            v24 = sp.Matrix([[v[1]],[v[3]]])

            α, β = v13.norm(), v24.norm()
            u13, u24 = v13/α, v24/β
            α_val, β_val = α**2, β**2
            ip1 = sp.Abs((u13.conjugate().T @ sp.Matrix([[model[i+1][0]],[model[i+1][2]]]))[0])
            ip2 = sp.Abs((u24.conjugate().T @ sp.Matrix([[model.s0[0]],[model.s0[2]]]))[0]) 

            rows.append({
                "State": f"s{i}",
                "α²": round(α_val, 2),
                "β²": round(β_val, 2),
                "⟨u13|sj+1⟩": round(float(sp.N(ip1)),2),
                "⟨u24|s0⟩": round(float(sp.N(ip2)),2)
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
