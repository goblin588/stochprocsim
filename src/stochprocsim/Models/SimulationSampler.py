import numpy as np
import math 

from .CausalModels import *
from .TransitionModel import QuantumTransitionModel, ExactTransitionModel, TransitionModel

def transition(v):
    path2 = sp.Matrix([[v[0]],[v[2]]])
    path1 = sp.Matrix([[v[1]],[v[3]]])
    α, β = path2.norm(), path1.norm()
    return path2, α, β

class Simulator():
    def __init__(self, transition_model: TransitionModel, nphotons:int=None):
        self._transition_model = transition_model
        self.name = transition_model.name
        self.N = len(transition_model)
        self._model = transition_model.model
        self._nphotons = nphotons 
        self.outputs = []

    def run(self):
        if self._nphotons is not None:
            self.outputs = []
            s = self._transition_model.sample_output(self._model, self._nphotons)  
            self.outputs.append(s.to_dict(orient="records"))  
        else:
            print("Cannot sample when nphotons and mrepititions have not been set")
    
    def get_transition_probabilities(self):
        """
        Calculate probability of 1 (a) and 0 (b) at each step 
        """
        for j in range(self.N):
            a, b = self._transition_model.get_output_probabilities(self.N, j)
            print(f'a:{a:.2f}, b:{b:.2f}')

    def get_output_distribution(self):
        """
        Calculate % of total samples which will end up in each output bin 
        """
        a_prod = 1
        res = []
        for j in range(self.N):
            a, b = self._transition_model.get_output_probabilities(self.N, j)
            res.append(b * a_prod)
            # print(f"s{j} p1:{(b * a_pro d):0.2f}")
            a_prod *= a
        return res
    
    def get_output_distribution_exp(self, propagate_outputs: bool = False, 
                                    include_loss: bool = False):
        """
        Experimental output distribution with optional error models
        """

        # --- choose probability model ---
        if propagate_outputs:
            probs = self.get_output_distribution_propagaged_output_states()
        else:
            probs = self.get_output_distribution()

        probs = np.array(probs, dtype=float)

        # print(f'probs: {len(probs)}')
        # print(f'loss: {len(self._transition_model.model.loop_loss)}')

        # --- include loss ---
        if include_loss:
            loss = np.asarray(self._transition_model.model.loop_loss, dtype=float)
            probs = probs * (1 - loss)
        return probs.tolist()

    def get_output_distribution_propagaged_output_states(self):
        # Get transition probabilities propagating output states
        model = self._transition_model.model
        U = model.U
        s0 = model.states[0]

        transition_probs = []
        for i in range(len(self._transition_model.model)):
            # print(f's{i}: {s0}')
            v = self._transition_model.model.U@s0
            path2, a, b = transition(v)
            transition_probs.append([a**2,b**2])
            s0 = (sp.Matrix([[path2[0]], [0], [path2[1]], [0]]))/a

        # Calculate distribution
        a_prod = 1
        res = []
        for j in range(self.N):
            a, b = transition_probs[j][0], transition_probs[j][1]
            res.append(b * a_prod)
            a_prod *= a
        # print(res)
        return res

        #[0.575959815440372, 0.678796191555467, 0.417159375550698]

    # POISSON SAMPLING 
    def sample_transition_probabilities(self):
        self.run()

        num_runs = len(self.outputs)
        num_settings = len(self.outputs[0])

        for i in range(num_settings):
            # Get avg num counts 
            n0 = sum(run[i]["N path |0>"] for run in self.outputs) / num_runs
            n1 = sum(run[i]["N path |1>"] for run in self.outputs) / num_runs

            # Poisson uncertainties on mean counts
            s0 = math.sqrt(n0)
            s1 = math.sqrt(n1)

            # print(f"a_avg: {n0:.3f} ± {s0:.3f}, b_avg: {n1:.3f} ± {s1:.3f}")

            denom = n0 + n1
            p0 = n0 / denom
            p1 = n1 / denom

            # error propagation for p = N / (N0 + N1)
            dp0 = math.sqrt(
                (s0 / denom) ** 2 +
                (n0 * s1 / denom**2) ** 2
            )
            dp1 = math.sqrt(
                (s1 / denom) ** 2 +
                (n1 * s0 / denom**2) ** 2
            )

            print(f"a: {p0:.3f} ± {dp0:.3f}, b: {p1:.3f} ± {dp1:.3f}")

    def sample_counts(self, print_outputs=True):
        self.run()

        num_runs = len(self.outputs)
        num_settings = len(self.outputs[0])

        res = []

        for i in range(num_settings):
            # Get avg num counts 
            n0 = sum(run[i]["N path |0>"] for run in self.outputs) / num_runs
            n1 = sum(run[i]["N path |1>"] for run in self.outputs) / num_runs

            # Poisson uncertainties on mean counts
            s0 = math.sqrt(n0)
            s1 = math.sqrt(n1)

            if print_outputs:
                print(f"a: {n0:.3f} ± {s0:.3f}, b: {n1:.3f} ± {s1:.3f}")
            res.append([n0,n1])
        return res
    
    def sample_output_distribution(self):
        """
        
        """
        self.run()
        num_runs = len(self.outputs)
        num_loops = len(self.outputs[0])

        res = []

        for i in range(num_loops):
            n1_avg = sum(run[i]["N path |1>"] for run in self.outputs) / num_runs

            p = n1_avg / self._nphotons
            dp = math.sqrt(n1_avg) / self._nphotons

            res.append([p, dp])

        return res 

    def save(self):
        out_file = f"Data/{self.name}_{self.model.name}_M_{self.mrepetitions}_photons_{self.nphotons}.npz"
        np.savez(out_file, outputs=np.array(self.outputs, dtype=object))
        print(f"Saved outputs to {out_file}")

