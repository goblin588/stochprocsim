import numpy as np
from stochprocsim.stochprocq.models.renewal import RenewalProcess


def generate_quantum_model(output_distribution) -> RenewalProcess:
    """Convert a quantum output distribution to an equivalent RenewalProcess."""
    q_emit = np.asarray(output_distribution, dtype=float)
    q_survive_st = np.zeros_like(q_emit)
    q_survive_st[0] = q_emit[0]
    for i in range(1, len(q_emit)):
        q_survive_st[i] = q_emit[i] / np.prod(1 - q_survive_st[:i])
    return RenewalProcess([1 - q for q in q_survive_st[:-1]])