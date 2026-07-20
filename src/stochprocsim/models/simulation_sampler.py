import numpy as np
import math

from .causal_models import *
from .transition_model import QuantumTransitionModel, ExactTransitionModel, TransitionModel


def _split_paths(v: np.ndarray):
    """Split a 4-component output vector into (path2, α, β)."""
    path2, path1 = v[[0, 2]], v[[1, 3]]
    return path2, np.linalg.norm(path2), np.linalg.norm(path1)


class Simulator:
    def __init__(self, transition_model: TransitionModel, nphotons: int = None):
        self._transition_model = transition_model
        self.name = transition_model.name
        self.N = len(transition_model)
        self._model = transition_model.model
        self._nphotons = nphotons
        self.outputs = []

    def run(self):
        if self._nphotons is None:
            raise ValueError("nphotons must be set before calling run()")
        self.outputs = []
        s = self._transition_model.sample_output(self._model, self._nphotons)
        self.outputs.append(s.to_dict(orient="records"))

    def get_transition_probabilities(self):
        """Return list of (p_survive, p_emit) at each step."""
        return [
            self._transition_model.get_output_probabilities(self.N, j)
            for j in range(self.N)
        ]


    def get_output_distribution(self, propagate_outputs: bool = False, start_state: int = 0):
        """Output distribution with optional noise propagation and loop-loss scaling.

        start_state picks the memory state the machine is initialised in
        (only supported on the propagated path).
        """
        if propagate_outputs:
            probs = self._get_output_distribution_propagated(start_state)
        else:
            if start_state != 0:
                raise ValueError("start_state requires propagate_outputs=True")
            probs = self._get_output_distribution_stationary()

        probs = np.array(probs, dtype=float)

        return probs.tolist()

    def _get_output_distribution_stationary(self):
        """Probability of each output bin: P(emit at step j, survive all previous)."""
        a_prod = 1
        res = []
        for j in range(self.N):
            a, b = self._transition_model.get_output_probabilities(self.N, j)
            res.append(b * a_prod)
            a_prod *= a
        return res

    def _get_output_distribution_propagated(self, start_state: int = 0):
        """Output distribution propagating the output state after each emission."""
        model = self._transition_model.model
        s = model.states[start_state].copy()

        transition_probs = []
        for _ in range(len(model)):
            v = model.U @ s
            path2, a, b = _split_paths(v)
            transition_probs.append((a ** 2, b ** 2))
            if a < 1e-12:  # everything emitted; no survivor path to propagate
                s = np.zeros(4, dtype=complex)
            else:
                s = np.array([path2[0], 0, path2[1], 0], dtype=complex) / a

        a_prod = 1
        res = []
        for a, b in transition_probs:
            res.append(b * a_prod)
            a_prod *= a
        return res

    def sample_counts(self, print_outputs=True):
        """Return average counts [[n0, n1], ...] per loop step with Poisson errors."""
        self.run()
        num_runs = len(self.outputs)
        res = []
        for i in range(len(self.outputs[0])):
            n0 = sum(run[i]["N path |0>"] for run in self.outputs) / num_runs
            n1 = sum(run[i]["N path |1>"] for run in self.outputs) / num_runs
            if print_outputs:
                print(f"a: {n0:.3f} ± {math.sqrt(n0):.3f}, b: {n1:.3f} ± {math.sqrt(n1):.3f}")
            res.append([n0, n1])
        return res

    def save(self, directory: str = "Data"):
        """Save sampled outputs to a .npz file."""
        import os
        os.makedirs(directory, exist_ok=True)
        out_file = os.path.join(directory, f"{self.name}_photons_{self._nphotons}.npz")
        np.savez(out_file, outputs=np.array(self.outputs, dtype=object))
        print(f"Saved outputs to {out_file}")

