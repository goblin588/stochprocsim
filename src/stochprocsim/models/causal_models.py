import numpy as np
from ..libraries.optics_lib import getUtot
from .unitaries import *
from .waveplate_angles import *

class CausalModel():
    def __init__(self, U, states:list, angles_NTU, angles_GU, name=None, transmission: float = 1.0):
        self.U = U
        self.U_theo = U
        self.U_optics_NTU = getUtot(angles_NTU)
        self.U_optics_GU = getUtot(angles_GU)
        self.states = list(states)
        self._state_map = {f"s{i}": s for i, s in enumerate(states)}
        self.name = name
        self.angles_NTU = angles_NTU
        self.angles_GU = angles_GU
        self.set_transmission(transmission)

    def set_U(self, U):
        self.U = U

    def set_transmission(self, T: float):
        """Per-loop transmission T. Bin k has traversed k loops so survival = T^k."""
        self.transmission = T
        self.loop_transmission = np.array([T ** k for k in range(len(self))])

    def __getitem__(self, idx):
        return self.states[idx]
    
    def __getattr__(self, name):
        # Guard prevents infinite recursion if _state_map hasn't been set yet
        if '_state_map' not in self.__dict__:
            raise AttributeError(name)
        if name in self._state_map:
            return self._state_map[name]
        raise AttributeError(f'No state named {name}')
    
    def __len__(self):
        return len(self.states)-1

def reorder_matrix(U: np.ndarray, row_order: list[int], col_order: list[int]) -> np.ndarray:
    """Return a copy of U with rows and columns permuted by row_order and col_order."""
    return U[np.ix_(row_order, col_order)]


def reorder_states(states: list[np.ndarray], order: list[int]) -> list[np.ndarray]:
    """Return a copy of each state vector with elements permuted by order."""
    return [state[order] for state in states]


CS_3 = CausalModel(U=U_3, states=U_3_states, angles_NTU=U_3_angles_NTU, angles_GU=U_3_angles_GU, name='U_3')
CS_4 = CausalModel(U=U_4, states=U_4_states, angles_NTU=U_4_angles_NTU, angles_GU=U_4_angles_GU, name='U_4')
CS_5 = CausalModel(U=U_5, states=U_5_states, angles_NTU=U_5_angles_NTU, angles_GU=U_5_angles_GU, name='U_5')
CS_6 = CausalModel(U=U_6, states=U_6_states, angles_NTU=U_6_angles_NTU, angles_GU=U_6_angles_GU, name='U_6')


Causal_Models = {
    3: CS_3,
    4: CS_4,
    5: CS_5,
    6: CS_6
}

