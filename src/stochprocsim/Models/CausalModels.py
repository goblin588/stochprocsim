import numpy as np
from ..Libraries.OpticsLib import getUtot, flipped
from .Unitaries import *
from .WaveplateAngles import *

# For reordering theory unitary path mode definitions into experimental ones
# This should always be True unless intentionally trying to check unflipped versions

loop_loss = [0.4, 0.48, 0.5, 0.51, 0.52, 0.53, 0.54]

class CausalModel():
    def __init__(self, U, states:list, angles_NTU, angles_GU, name=None):
        self.U = U
        self.U_theo = U 
        self.U_optics_NTU = getUtot(angles_NTU)
        self.U_optics_GU = getUtot(angles_GU)    
        self.states = list(states)
        self._state_map = {f"s{i}": s for i, s in enumerate(states)}
        self.name = name
        self.angles_NTU = angles_NTU
        self.angles_GU = angles_GU
        self.angles_GU = angles_GU
        loss = np.asarray(loop_loss[:len(self)], dtype=float)
        # cumulative loss
        self.loop_loss = 1 - np.cumprod(1 - loss)

    def set_U(self, U):
        self.U = U

    def __getitem__(self, idx):
        return self.states[idx]
    
    def __getattr__(self, name):
        if name in self._state_map:
            return self._state_map[name]
        raise AttributeError(f'No state named {name}')
    
    def __len__(self):
        return len(self.states)-1

def reorder_matrix(U: sp.Matrix, row_order: list[int], col_order: list[int]) -> sp.Matrix:
    """
    Return a reordered copy of U with rows and columns rearranged according to
    row_order and col_order (0-indexed).

    Example:
        new_U = reorder_matrix(U, [1, 3, 0, 2], [1, 3, 0, 2])
    """
    return U.extract(row_order, col_order)

def reorder_states(states: list[sp.Matrix], order: list[int]) -> list[sp.Matrix]:
    """
    Return a reordered copy of a list of state vectors (sp.Matrix objects).

    """
    reordered_matrices = []
    for state in states:
        reordered = [state[i] for i in order]
        reordered_matrices.append(sp.Matrix(reordered))
    return reordered_matrices


if flipped == True:
    # Models
    CS_3 = CausalModel(U=reorder_matrix(U_3, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_3_states, [1, 3, 0, 2]), angles_NTU=U_3_angles_NTU, 
                        angles_GU=U_3_angles_GU, name='U_3')
    CS_4 = CausalModel(U=reorder_matrix(U_4, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_4_states, [1, 3, 0, 2]), angles_NTU=U_4_angles_NTU, 
                        angles_GU=U_4_angles_GU, name='U_4')
    CS_5 = CausalModel(U=reorder_matrix(U_5, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_5_states, [1, 3, 0, 2]), angles_NTU=U_5_angles_NTU, 
                        angles_GU=U_5_angles_GU, name='U_5')
    CS_6 = CausalModel(U=reorder_matrix(U_6, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_6_states, [1, 3, 0, 2]), angles_NTU=U_6_angles_NTU, 
                        angles_GU=U_6_angles_GU, name='U_6')

else:
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

